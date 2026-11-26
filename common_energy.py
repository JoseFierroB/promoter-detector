"""Load and score custom ProkBERT late-fusion energy models.

`AutoModelForSequenceClassification.from_pretrained` only knows the authors' 384 -> 2 head and does not
pass an energy tensor, so it silently discards `classifier.*`/`classifier.0.*` from energy checkpoints.
This module reconstructs the head from the saved state dict, loads it strictly, and scores with energy.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForSequenceClassification
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


def get_energy_profile(seq, length=80, raw=False, mean_dg=MEAN_DG, std_dg=STD_DG, filter_unknown=False):
    seq = seq.upper().strip()
    if filter_unknown:
        vals = [STABILITY[seq[i:i + 2]] for i in range(len(seq) - 1) if seq[i:i + 2] in STABILITY]
    else:
        vals = [STABILITY.get(seq[i:i + 2], mean_dg) for i in range(min(len(seq) - 1, length))]
    vals = np.asarray(vals[:length], dtype=np.float32)
    if not raw and vals.size:
        vals = (vals - mean_dg) / std_dg
    out = np.zeros(length, dtype=np.float32)
    out[:len(vals)] = vals
    return out


class LinearResidualClassifier(nn.Module):
    """classifier.base_clf(pooled) + classifier.energy_clf(energy), as in energy_res.py."""
    def __init__(self, base_clf, energy_clf):
        super().__init__()
        self.base_clf = base_clf
        self.energy_clf = energy_clf

    def forward(self, pooled, energy):
        return self.base_clf(pooled) + self.energy_clf(energy)


class ConvResidualClassifier(nn.Module):
    """classifier.base_clf(pooled) + classifier.energy_net(energy[:, None, :]), as in energy_conv_res.py."""
    def __init__(self, base_clf, energy_net):
        super().__init__()
        self.base_clf = base_clf
        self.energy_net = energy_net

    def forward(self, pooled, energy):
        return self.base_clf(pooled) + self.energy_net(energy.unsqueeze(1))


class EnergyEvalModel(nn.Module):
    def __init__(self, base_model, classifier, energy_len, pooling_masked=True,
                 dropout_before_concat=True, energy_mode="real", energy_head=None, residual=False,
                 residual_classifier=False):
        super().__init__()
        self.config = base_model.config
        self.bert = base_model.bert
        self.weighting_layer = base_model.weighting_layer
        self.dropout = base_model.dropout
        self.classifier = classifier
        self.energy_len = energy_len
        self.pooling_masked = pooling_masked
        self.dropout_before_concat = dropout_before_concat
        self.energy_mode = energy_mode
        self.energy_head = energy_head
        self.residual = residual
        self.residual_classifier = residual_classifier

    def forward(self, input_ids=None, attention_mask=None, energy=None, labels=None, **kwargs):
        sequence_output = self.bert(input_ids=input_ids,
                                    attention_mask=attention_mask).last_hidden_state
        scores = self.weighting_layer(sequence_output).squeeze(-1)
        if self.pooling_masked and attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (sequence_output * weights).sum(dim=1)

        if self.dropout_before_concat:
            pooled = self.dropout(pooled)

        e = None
        if self.energy_len:
            if energy is None:
                raise ValueError("this model needs an energy tensor")
            if self.energy_mode == "zero":
                e = torch.zeros_like(energy)
            elif self.energy_mode == "shuffle":
                e = torch.roll(energy, 1, 0)
            else:
                e = energy.to(pooled.dtype)

        if self.residual_classifier:
            if e is None:
                raise ValueError("this residual model needs an energy tensor")
            logits = self.classifier(pooled, e)
        elif self.residual:
            logits = self.classifier(pooled)
            if e is not None and self.energy_head is not None:
                logits = logits + self.energy_head(e)
        else:
            if self.energy_len:
                z = torch.cat([pooled, e], dim=-1)
            else:
                z = pooled
            if not self.dropout_before_concat:
                z = self.dropout(z)
            logits = self.classifier(z)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


def _read_state(path):
    p = Path(path)
    if p.is_dir():
        for name in ("model.safetensors", "model.pt", "pytorch_model.bin"):
            if (p / name).exists():
                p = p / name
                break
        else:
            st = sorted(p.glob("*.safetensors"))
            pt = sorted(p.glob("*.pt"))
            bn = sorted(p.glob("*.bin"))
            p = (st or pt or bn)[0] if (st or pt or bn) else p

    if p.suffix == ".safetensors":
        state = load_file(str(p))
    else:
        state = torch.load(p, map_location="cpu", weights_only=True)
    if any(k.startswith("encoder.") for k in state):
        state = {("bert." + k[len("encoder."):] if k.startswith("encoder.") else k): v
                 for k, v in state.items()}
    return state


def load_energy_model(path, pooling_masked=True, dropout_before_concat=True,
                      raw=None, mean_dg=None, std_dg=None, filter_unknown=False,
                      energy_mode="real"):
    cfg_path = Path(path)
    if cfg_path.is_file():
        cfg_path = cfg_path.parent
    energy_cfg = cfg_path / "energy_config.json"
    meta = json.loads(energy_cfg.read_text()) if energy_cfg.exists() else {}
    # A model's config is authoritative for the training-time pooling/dropout choices.
    pooling_masked = bool(meta.get("pooling_masked", pooling_masked))
    dropout_before_concat = bool(meta.get("dropout_before_concat", dropout_before_concat))

    state = _read_state(path)
    cfg = AutoConfig.from_pretrained(common.BASE, revision=common.BASE_REV, trust_remote_code=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        common.BASE, config=cfg, revision=common.BASE_REV, trust_remote_code=True)
    base.config.use_cache = False
    h = base.config.hidden_size

    energy_head = None
    residual = False
    residual_classifier = False

    if "classifier.base_clf.weight" in state:
        in_dim = state["classifier.base_clf.weight"].shape[1]
        out_dim = state["classifier.base_clf.weight"].shape[0]
        base_clf = nn.Linear(in_dim, out_dim)
        if "classifier.energy_clf.weight" in state:
            energy_len = state["classifier.energy_clf.weight"].shape[1]
            has_bias = "classifier.energy_clf.bias" in state
            energy_clf = nn.Linear(energy_len, out_dim, bias=has_bias)
            classifier = LinearResidualClassifier(base_clf, energy_clf)
            residual_classifier = True
        elif any(k.startswith("classifier.energy_net.") for k in state):
            energy_len = int(meta.get("energy_len", 80))
            energy_net = nn.Sequential(
                nn.AvgPool1d(kernel_size=6, stride=1, padding=3),
                nn.Conv1d(1, 16, kernel_size=7, padding=3),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(4),
                nn.Flatten(),
                nn.Linear(64, out_dim),
            )
            classifier = ConvResidualClassifier(base_clf, energy_net)
            residual_classifier = True
        else:
            raise ValueError(f"unknown residual classifier state in {path}")
    elif "energy_head.0.weight" in state:
        residual = True
        energy_len = state["energy_head.0.weight"].shape[1]
        hidden = state["energy_head.0.weight"].shape[0]
        out_dim = state["energy_head.2.weight"].shape[0]
        energy_head = nn.Sequential(nn.Linear(energy_len, hidden), nn.GELU(),
                                    nn.Linear(hidden, out_dim))
        in_dim = state["classifier.weight"].shape[1]
        classifier = nn.Linear(in_dim, state["classifier.weight"].shape[0])
    elif "energy_head.weight" in state:
        residual = True
        energy_len = state["energy_head.weight"].shape[1]
        out_dim = state["energy_head.weight"].shape[0]
        energy_head = nn.Linear(energy_len, out_dim, bias=False)
        in_dim = state["classifier.weight"].shape[1]
        classifier = nn.Linear(in_dim, state["classifier.weight"].shape[0])
    elif "classifier.0.weight" in state:
        in_dim = state["classifier.0.weight"].shape[1]
        hidden = state["classifier.0.weight"].shape[0]
        out_dim = state["classifier.2.weight"].shape[0]
        classifier = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        energy_len = in_dim - h
    elif "classifier.weight" in state:
        in_dim = state["classifier.weight"].shape[1]
        out_dim = state["classifier.weight"].shape[0]
        classifier = nn.Linear(in_dim, out_dim)
        energy_len = in_dim - h
    else:
        raise ValueError(f"no classifier weights in {path}")

    if energy_len < 0:
        raise ValueError(f"energy_len {energy_len} is negative")

    model = EnergyEvalModel(base, classifier, energy_len, pooling_masked,
                            dropout_before_concat, energy_mode,
                            energy_head=energy_head, residual=residual,
                            residual_classifier=residual_classifier)
    model.load_state_dict(state, strict=True)
    model.raw = bool(meta.get("raw_energy", False) if raw is None else raw)
    model.mean_dg = float(meta.get("mean_dg", MEAN_DG) if mean_dg is None else mean_dg)
    model.std_dg = float(meta.get("std_dg", STD_DG) if std_dg is None else std_dg)
    model.filter_unknown = bool(meta.get("filter_unknown", filter_unknown))
    model.energy_name = str(path)
    return model, energy_len, meta

@torch.no_grad()
def score_energy(model, tokenizer, seqs, batch_size=256, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    seqs, out = list(seqs), []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        enc = {k: v.to(device) for k, v in common.encode(tokenizer, chunk).items()}
        energies = torch.tensor(
            np.stack([get_energy_profile(s, model.energy_len, raw=model.raw,
                                         mean_dg=model.mean_dg, std_dg=model.std_dg,
                                         filter_unknown=model.filter_unknown) for s in chunk]),
            device=device)
        out.append(torch.softmax(model(**enc, energy=energies).logits.float(), -1)[:, 1].cpu())
    return torch.cat(out).numpy()
