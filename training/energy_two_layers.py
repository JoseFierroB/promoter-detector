"""2-Layer GELU MLP Energy Fusion ([B, H] + [B, 80] -> Linear -> GELU -> Linear)."""
import argparse
from pathlib import Path
import types
import launch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
parser.add_argument("--output-dir", default="outputs/energy_mlp")
parser.add_argument("--max-steps", type=int, default=-1)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--hidden-dim", type=int, default=384)
parser.add_argument("--raw-energy", action="store_true")
parser.add_argument("--no-energy", action="store_true")
launch.add_args(parser)
args = parser.parse_args()
out_dir = Path(args.output_dir)
launch.setup(args, out_dir)

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from transformers import (AutoConfig, AutoModelForSequenceClassification, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)
from transformers.modeling_outputs import SequenceClassifierOutput
import common

device = launch.device(args)
tokenizer = common.load_tokenizer()
split = common.load_split(3)

STABILITY = {
    "AA": -1.00, "AT": -0.88, "TA": -0.58, "AG": -1.30,
    "GA": -1.30, "TT": -1.00, "AC": -1.45, "CA": -1.45,
    "TG": -1.44, "GT": -1.44, "TC": -1.28, "CT": -1.28,
    "CC": -1.84, "CG": -2.24, "GC": -2.27, "GG": -1.84,
}
MEAN_DG, STD_DG = -1.35, 0.45


def get_energy_profile(seq, length=80, raw=False):
    seq = seq.upper().strip()
    dinucs = [STABILITY[seq[i:i + 2]] for i in range(len(seq) - 1) if seq[i:i + 2] in STABILITY]
    profile = np.zeros(length, dtype=np.float32)
    n = min(len(dinucs), length)
    if n > 0:
        vals = np.array(dinucs[:n], dtype=np.float32)
        profile[:n] = vals if raw else (vals - MEAN_DG) / STD_DG
    return profile


def forward_with_energy(self, input_ids=None, attention_mask=None, token_type_ids=None,
                        position_ids=None, inputs_embeds=None, energy=None, labels=None,
                        output_attentions=None, output_hidden_states=None, return_dict=None):
    assert energy is not None, "energy tensor was dropped by data_collator!"
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    outputs = self.bert(
        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
        position_ids=position_ids, inputs_embeds=inputs_embeds,
        output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    seq_out = outputs[0]
    scores = self.weighting_layer(seq_out).squeeze(-1)
    if attention_mask is not None:
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
    pooled = (seq_out * torch.softmax(scores, dim=-1).unsqueeze(-1)).sum(dim=1)  # [B, H]

    fused = torch.cat([self.dropout(pooled), energy.to(pooled.dtype)], dim=-1)  # [B, H + 80]
    logits = self.classifier(fused)

    loss = nn.CrossEntropyLoss()(logits.view(-1, 2), labels.view(-1)) if labels is not None else None
    if not return_dict:
        return ((loss, logits) + outputs[2:]) if loss is not None else ((logits,) + outputs[2:])
    return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


def attach_energy_head(model, hidden_dim=384):
    h = int(model.config.hidden_size)
    old_clf = model.classifier
    fc1 = nn.Linear(h + 80, hidden_dim)
    with torch.no_grad():
        nn.init.normal_(fc1.weight[:, h:], mean=0.0, std=1e-4)
    fc2 = nn.Linear(hidden_dim, 2)
    with torch.no_grad():
        fc2.bias.copy_(old_clf.bias)
    model.classifier = nn.Sequential(fc1, nn.GELU(), fc2)
    model.forward = types.MethodType(forward_with_energy, model)
    return model


@torch.no_grad()
def score_energy(model, tokenizer, seqs, masked_ids=(2, 3), batch_size=1024, raw=False):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    seqs, out = list(seqs), []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        enc = {k: v.to(dev) for k, v in common.encode(tokenizer, chunk, masked_ids).items()}
        energies = torch.tensor(np.stack([get_energy_profile(s, raw=raw) for s in chunk]), device=dev)
        out.append(torch.softmax(model(**enc, energy=energies)["logits"].float(), -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def tokenize(batch):
    enc = common.encode(tokenizer, batch["segment"])
    res = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": batch["y"]}
    if not args.no_energy:
        res["energy"] = np.stack([get_energy_profile(s, raw=args.raw_energy) for s in batch["segment"]])
    return res


def collate_batch(features):
    batch = {k: torch.tensor([f[k] for f in features]) for k in ("input_ids", "attention_mask", "labels")}
    if "energy" in features[0]:
        batch["energy"] = torch.tensor([f["energy"] for f in features], dtype=torch.float32)
    return batch


data = {s: Dataset.from_pandas(split[s][["segment", "y"]], preserve_index=False)
        .map(tokenize, batched=True, batch_size=1000, remove_columns=["segment", "y"]) for s in ("train", "val")}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"roc_auc": roc_auc_score(labels, softmax(logits, axis=-1)[:, 1])}


for seed in args.seeds:
    seed_dir = out_dir / f"seed_{seed}"
    set_seed(seed)

    cfg = AutoConfig.from_pretrained(common.BASE, revision=common.BASE_REV, trust_remote_code=True)
    if args.dropout is not None:
        cfg.classification_dropout_rate = cfg.hidden_dropout_prob = cfg.attention_probs_dropout_prob = args.dropout
    model = AutoModelForSequenceClassification.from_pretrained(common.BASE, config=cfg, revision=common.BASE_REV, trust_remote_code=True)

    if not args.no_energy:
        model = attach_energy_head(model, hidden_dim=args.hidden_dim)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(seed_dir / "checkpoints"),
            per_device_train_batch_size=128, per_device_eval_batch_size=128,
            learning_rate=1e-4, warmup_ratio=0.1, weight_decay=0.01,
            num_train_epochs=10, max_steps=args.max_steps,
            eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=100,
            save_total_limit=2, load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False,
            logging_steps=50, report_to="none", remove_unused_columns=False, seed=seed,
        ),
        train_dataset=data["train"], eval_dataset=data["val"],
        data_collator=collate_batch, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )
    train_output = trainer.train()
    state = trainer.state
    trainer.save_model(str(seed_dir / "model"))

    if args.no_energy:
        row = common.val_row(seed, model, tokenizer, split["val"])
    else:
        s = score_energy(model, tokenizer, split["val"].segment, raw=args.raw_energy)
        val = split["val"]
        row = {"seed": seed, "val_auc": roc_auc_score(val.y, s),
               "val_auc_hf": roc_auc_score(val.y[val.origin.str.startswith("hf")], s[val.origin.str.startswith("hf")]),
               "val_auc_pneumo": roc_auc_score(val.y[val.origin == "pneumo"], s[val.origin == "pneumo"])}
    common.save_seed(seed_dir, row | {"best_step": state.best_global_step, "last_step": state.global_step,
                                      "gpu": device, "train_min": train_output.metrics["train_runtime"] / 60})

launch.finish(args, out_dir)