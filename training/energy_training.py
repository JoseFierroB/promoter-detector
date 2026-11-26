"""Residual energy fusion: baseline ProkBERT logits + zero-initialized energy branch.

    logits = old_classifier(dropout(pooled)) + energy_head(energy)
    energy_head = Linear(80, hidden) -> GELU -> Linear(hidden, 2)

The last energy layer starts at zero, so at initialization the model is exactly the baseline
classifier. The encoder, weighting-layer pooling, old classifier and energy branch are fine-tuned.

Arms:
    zero    energy vector is zero (capacity control)
    real    positional SantaLucia dG profile
    shuffle energy profiles rolled across the batch (match control)
"""

import argparse
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import launch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
parser.add_argument("--output-dir", default="outputs/energy_residual")
parser.add_argument("--max-steps", type=int, default=-1, help="smoke tests only")
parser.add_argument("--dropout", type=float, default=None,
                    help="override config dropout (default: pretrained config)")
parser.add_argument("--hidden-dim", type=int, default=128)
parser.add_argument("--energy-mode", choices=["zero", "real", "shuffle"], default="real")
parser.add_argument("--raw-energy", action="store_true")
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

STABILITY = {
    "AA": -1.00, "AT": -0.88, "TA": -0.58, "AG": -1.30,
    "GA": -1.30, "TT": -1.00, "AC": -1.45, "CA": -1.45,
    "TG": -1.44, "GT": -1.44, "TC": -1.28, "CT": -1.28,
    "CC": -1.84, "CG": -2.24, "GC": -2.27, "GG": -1.84,
}
MEAN_DG = float(np.mean(list(STABILITY.values())))
STD_DG = float(np.std(list(STABILITY.values())))


def get_energy_profile(seq, length, raw=False):
    seq = seq.upper().strip()
    vals = np.array([STABILITY.get(seq[i:i + 2], MEAN_DG)
                     for i in range(min(len(seq) - 1, length))], dtype=np.float32)
    if not raw:
        vals = (vals - MEAN_DG) / STD_DG
    out = np.zeros(length, dtype=np.float32)
    out[:len(vals)] = vals
    return out


def forward_with_residual(self, input_ids=None, attention_mask=None, token_type_ids=None,
                          position_ids=None, inputs_embeds=None, energy=None, labels=None,
                          output_attentions=None, output_hidden_states=None, return_dict=None):
    assert energy is not None, "energy tensor was dropped by data_collator!"
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    outputs = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    sequence_output = outputs[0]
    scores = self.weighting_layer(sequence_output).squeeze(-1)
    weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
    pooled = (sequence_output * weights).sum(dim=1)

    pooled = self.dropout(pooled)
    logits = self.classifier(pooled)

    if self.energy_mode == "zero":
        e = torch.zeros_like(energy)
    elif self.energy_mode == "shuffle":
        e = torch.roll(energy, shifts=1, dims=0)
    else:
        e = energy
    logits = logits + self.energy_head(e.to(pooled.dtype))

    loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
    return SequenceClassifierOutput(loss=loss, logits=logits)


def attach_residual_head(model, energy_len, hidden_dim, energy_mode):
    model.energy_head = nn.Sequential(
        nn.Linear(energy_len, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 2),
    )
    with torch.no_grad():
        model.energy_head[-1].weight.zero_()
        model.energy_head[-1].bias.zero_()
    model.energy_mode = energy_mode
    model.forward = types.MethodType(forward_with_residual, model)
    return model


def cpu_state(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


device = launch.device(args)
print(f"Training on {device}")

tokenizer = common.load_tokenizer()
split = common.load_split(3)
max_len_nt = int(max(split["train"].segment.str.len().max(), split["val"].segment.str.len().max()))
energy_len = max_len_nt - 1
print(f"max sequence: {max_len_nt} nt -> energy vector: {energy_len}")


def tokenize(batch):
    enc = common.encode(tokenizer, batch["segment"])
    energies = np.stack([get_energy_profile(s, energy_len, raw=args.raw_energy) for s in batch["segment"]])
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "energy": energies, "labels": batch["y"]}


def collate_batch(features):
    batch = {k: torch.tensor([f[k] for f in features]) for k in ("input_ids", "attention_mask", "labels")}
    batch["energy"] = torch.tensor([f["energy"] for f in features], dtype=torch.float32)
    return batch


data = {s: Dataset.from_pandas(split[s][["segment", "y"]], preserve_index=False)
        .map(tokenize, batched=True, batch_size=1000, remove_columns=["segment", "y"])
        for s in ("train", "val")}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"roc_auc": roc_auc_score(labels, softmax(logits, axis=-1)[:, 1])}


@torch.no_grad()
def score_energy(model, tokenizer, seqs, batch_size=256):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    seqs, out = list(seqs), []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        enc = {k: v.to(dev) for k, v in common.encode(tokenizer, chunk).items()}
        energies = torch.tensor(np.stack([get_energy_profile(s, energy_len, raw=args.raw_energy)
                                          for s in chunk]), device=dev)
        out.append(torch.softmax(model(**enc, energy=energies).logits.float(), -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def val_row_energy(seed, model, tokenizer, val):
    s = score_energy(model, tokenizer, val.segment)
    row = {"seed": seed, "val_auc": roc_auc_score(val.y, s)}
    for name, mask in (("hf", val.origin.fillna("").str.startswith("hf")), ("pneumo", val.origin == "pneumo")):
        if mask.any():
            row[f"val_auc_{name}"] = roc_auc_score(val.y[mask], s[mask])
    return row


for seed in args.seeds:
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    if args.dropout is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            common.BASE, revision=common.BASE_REV, trust_remote_code=True)
    else:
        cfg = AutoConfig.from_pretrained(common.BASE, revision=common.BASE_REV, trust_remote_code=True)
        cfg.classification_dropout_rate = args.dropout
        cfg.hidden_dropout_prob = args.dropout
        cfg.attention_probs_dropout_prob = args.dropout
        model = AutoModelForSequenceClassification.from_pretrained(
            common.BASE, config=cfg, revision=common.BASE_REV, trust_remote_code=True)
    model.config.use_cache = False
    model = attach_residual_head(model, energy_len, args.hidden_dim, args.energy_mode)
    set_seed(seed)

    training_args = TrainingArguments(
        output_dir=str(seed_dir / "checkpoints"),
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        num_train_epochs=10,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=data["train"], eval_dataset=data["val"],
                      data_collator=collate_batch, compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=10)])
    train_output = trainer.train()

    state = trainer.state
    if state.best_model_checkpoint is not None:
        assert state.best_model_checkpoint.endswith(f"checkpoint-{state.best_global_step}"), "stale best checkpoint"
    best_step = state.best_global_step if state.best_global_step is not None else state.global_step
    best_metric = state.best_metric if state.best_metric is not None else float("nan")
    print(f"seed {seed}: best step {best_step} of {state.global_step}, val loss {best_metric:.4f}")

    model_dir = seed_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state(model), model_dir / "model.pt")
    torch.save(cpu_state(model.energy_head), model_dir / "head.pt")
    model.config.save_pretrained(model_dir)
    with open(model_dir / "energy_config.json", "w") as f:
        json.dump({"energy_mode": args.energy_mode,
                   "energy_len": energy_len,
                   "hidden_dim": args.hidden_dim,
                   "raw_energy": args.raw_energy,
                   "mean_dg": MEAN_DG,
                   "std_dg": STD_DG,
                   "dropout": args.dropout,
                   "seed": seed,
                   "residual": True,
                   "pooling_masked": False,
                   "dropout_before_concat": True,
                   "encoder": common.BASE,
                   "encoder_revision": common.BASE_REV}, f, indent=2)

    common.save_seed(seed_dir, val_row_energy(seed, model, tokenizer, split["val"])
                     | {"best_step": best_step, "last_step": state.global_step,
                        "gpu": device, "train_min": train_output.metrics["train_runtime"] / 60})

launch.finish(args, out_dir)
