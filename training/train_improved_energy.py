"""train_improved"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import launch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
parser.add_argument("--output-dir", default="outputs/task3_energy")
parser.add_argument("--max-steps", type=int, default=-1, help="smoke tests only")
parser.add_argument("--dropout", type=float, default=None,
                    help="override head+encoder+attention dropout (default: pretrained config)")
parser.add_argument("--energy-mode", choices=["zero", "real", "shuffle"], default="real")
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


def get_sequence_energy(seq, length):
    seq = seq.upper().strip()
    vals = np.array([STABILITY.get(seq[i:i + 2], MEAN_DG)
                     for i in range(min(len(seq) - 1, length))], dtype=np.float32)
    vals = (vals - MEAN_DG) / STD_DG
    out = np.zeros(length, dtype=np.float32)
    out[:len(vals)] = vals
    return out


class ProkBERTEnergy(nn.Module):
    """Authors' encoder + weighting pooling + dropout, then concat energy and Linear(H + 80, 2)."""

    def __init__(self, base_model, energy_len, energy_mode="real"):
        super().__init__()
        self.config = base_model.config
        self.bert = base_model.bert
        self.weighting_layer = base_model.weighting_layer
        self.dropout = base_model.dropout
        self.energy_len = energy_len
        self.energy_mode = energy_mode

        hidden = self.config.hidden_size
        old_classifier = base_model.classifier
        self.classifier = nn.Linear(hidden + energy_len, 2)
        with torch.no_grad():
            self.classifier.weight[:, :hidden] = old_classifier.weight
            self.classifier.weight[:, hidden:] = 0
            self.classifier.bias.copy_(old_classifier.bias)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, energy=None, labels=None, **kwargs):
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        weights = torch.softmax(self.weighting_layer(sequence_output).squeeze(-1), dim=1)
        pooled = (weights.unsqueeze(-1) * sequence_output).sum(dim=1)

        pooled = self.dropout(pooled)

        if self.energy_mode == "zero":
            e = torch.zeros_like(energy)
        elif self.energy_mode == "shuffle":
            e = torch.roll(energy, shifts=1, dims=0)
        else:
            e = energy

        z = torch.cat([pooled, e], dim=-1)
        logits = self.classifier(z)
        loss = self.loss_fct(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


def cpu_state(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


gpu_name = launch.device(args)
print(f"Training on {gpu_name}")

tokenizer = common.load_tokenizer()
split = common.load_split(3)
max_len_nt = int(max(split["train"].segment.str.len().max(), split["val"].segment.str.len().max()))
energy_len = max_len_nt - 1
print(f"max sequence: {max_len_nt} nt -> energy vector: {energy_len}")


def preprocess(batch):
    enc = common.encode(tokenizer, batch["segment"])
    energies = np.stack([get_sequence_energy(s, energy_len) for s in batch["segment"]])
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "energy": energies, "labels": batch["y"]}


data = {s: Dataset.from_pandas(split[s][["segment", "y"]], preserve_index=False)
        .map(preprocess, batched=True, batch_size=1000, remove_columns=["segment", "y"])
        for s in ("train", "val")}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"roc_auc": roc_auc_score(labels, softmax(logits, axis=-1)[:, 1])}


@torch.no_grad()
def score_model(model, seqs, batch_size=256):
    device = next(model.parameters()).device
    model.eval()
    seqs, out = list(seqs), []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        enc = common.encode(tokenizer, chunk)
        inputs = {k: v.to(device) for k, v in enc.items()}
        energies = torch.tensor(np.stack([get_sequence_energy(s, energy_len) for s in chunk]), device=device)
        out.append(torch.softmax(model(**inputs, energy=energies).logits.float(), dim=-1)[:, 1].cpu())
    return torch.cat(out).numpy()


def val_row(seed, model, val):
    s = score_model(model, val.segment)
    row = {"seed": seed, "val_auc": roc_auc_score(val.y, s)}
    for name, mask in (("hf", val.origin.fillna("").str.startswith("hf")), ("pneumo", val.origin == "pneumo")):
        if mask.any():
            row[f"val_auc_{name}"] = roc_auc_score(val.y[mask], s[mask])
    return row


for seed in args.seeds:
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    cfg = AutoConfig.from_pretrained(common.BASE, revision=common.BASE_REV, trust_remote_code=True)
    cfg.use_cache = False
    if args.dropout is not None:
        cfg.classification_dropout_rate = args.dropout
        cfg.hidden_dropout_prob = args.dropout
        cfg.attention_probs_dropout_prob = args.dropout
    base_model = AutoModelForSequenceClassification.from_pretrained(
        common.BASE, config=cfg, revision=common.BASE_REV, trust_remote_code=True)
    model = ProkBERTEnergy(base_model, energy_len, args.energy_mode)
    del base_model
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
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=10)])
    train_output = trainer.train()

    state = trainer.state
    best_step = state.best_global_step if state.best_global_step is not None else state.global_step
    best_metric = state.best_metric if state.best_metric is not None else float("nan")
    print(f"seed {seed}: best step {best_step} of {state.global_step}, val loss {best_metric:.4f}")

    model_dir = seed_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state(model), model_dir / "model.pt")
    torch.save(cpu_state(model.classifier), model_dir / "head.pt")
    model.config.save_pretrained(model_dir)
    with open(model_dir / "energy_config.json", "w") as f:
        json.dump({"energy_mode": args.energy_mode,
                   "energy_len": energy_len,
                   "features": "dinucleotide_dg_along_sequence",
                   "mean_dg": MEAN_DG,
                   "std_dg": STD_DG,
                   "dropout": args.dropout,
                   "seed": seed,
                   "encoder": common.BASE,
                   "encoder_revision": common.BASE_REV}, f, indent=2)

    common.save_seed(seed_dir, val_row(seed, model, split["val"])
                     | {"best_step": best_step, "last_step": state.global_step,
                        "gpu": gpu_name, "train_min": train_output.metrics["train_runtime"] / 60})

launch.finish(args, out_dir)
