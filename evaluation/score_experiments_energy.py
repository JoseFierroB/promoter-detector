"""Score/do the inference of every model on the sets its comparisons need: results/scores/<model>/<set>.tsv (cached)."""
import argparse
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--force", action="store_true", help="recompute existing score files")
parser.add_argument("--gpu", type=int, default=None,
                    help="GPU index in CUDA order (0 = fastest); default: the first GPU, or the CPU if there is none")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = (str(args.gpu) if args.gpu is not None
                                      else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification
from transformers import logging as hf_logging
from transformers.modeling_outputs import SequenceClassifierOutput

import common
import common_energy

hf_logging.set_verbosity_error()

RUNS = {
    "task3": (["task3_test", "sigma70"], "train_task3"),
    "energy_fusion": (["task3_test", "sigma70"], "train_task3"),
    "energy_fusion_raw": (["task3_test", "sigma70"], "train_task3"),
    "energy_direct": (["task3_test", "sigma70"], "train_task3"),
    "energy_res": (["task3_test", "sigma70"], "train_task3"),
    "energy_residual_logits": (["task3_test", "sigma70"], "train_task3"),
    "energy_exp_energy_mlp": (["task3_test", "sigma70"], "train_task3"),
    "energy_exp_energy_mlp_raw": (["task3_test", "sigma70"], "train_task3"),
    "energy_exp_energy_residual": (["task3_test", "sigma70"], "train_task3"),
    "energy_exp_no_energy": (["task3_test", "sigma70"], "train_task3"),
    "energy_w1w2_conv": (["task3_test", "sigma70"], "train_task3"),
    "energy_w1w2_direct": (["task3_test", "sigma70"], "train_task3"),
}

PUBLISHED = {
    "published-2025": ["multispecies", "sigma70", "pneumo_test"],
    "published-2024": ["multispecies", "sigma70", "pneumo_test"],
}

ENERGY_POOLING = {
    "no_energy_fusion": False,
    "energy_exp_no_energy": False,
}


# =====================================================================
# Clases y forwards para arquitecturas custom
# =====================================================================
class EnergyW1AttentionPooling(nn.Module):
    def __init__(self, base_weighting_layer: nn.Module, with_gate: bool = False):
        super().__init__()
        self.base_weighting = base_weighting_layer
        out_ch = 8 if with_gate else 16
        self.energy_scorer = nn.Sequential(
            nn.Conv1d(1, out_ch, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(out_ch, 1, kernel_size=3, padding=1),
        )
        if with_gate:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, sequence_output: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        sem_scores = self.base_weighting(sequence_output).squeeze(-1)
        e_scores = self.energy_scorer(energy.unsqueeze(1).to(sequence_output.dtype))
        e_aligned = nn.functional.interpolate(
            e_scores, size=sem_scores.size(1), mode="linear", align_corners=False
        ).squeeze(1)
        scale = self.gate if hasattr(self, "gate") else 1.0
        return sem_scores + scale * e_aligned


class SimpleLinearResidualHead(nn.Module):
    def __init__(self, old_clf: nn.Linear, num_labels: int):
        super().__init__()
        self.base_clf = old_clf
        self.energy_clf = nn.Linear(80, num_labels, bias=False)

    def forward(self, pooled_dropped: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        return self.base_clf(pooled_dropped) + self.energy_clf(energy.to(pooled_dropped.dtype))


def _forward_custom_energy(self, input_ids=None, attention_mask=None, token_type_ids=None,
                           position_ids=None, inputs_embeds=None, energy=None, labels=None,
                           output_attentions=None, output_hidden_states=None, return_dict=None):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    outputs = self.bert(
        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
        position_ids=position_ids, inputs_embeds=inputs_embeds,
        output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    seq_out = outputs[0]
    seq_len = seq_out.size(1)

    # 1. Pooling (W1)
    if isinstance(self.weighting_layer, EnergyW1AttentionPooling):
        scores = self.weighting_layer(seq_out, energy)
    elif getattr(self.weighting_layer, "in_features", 384) == 385:
        e_tok = nn.functional.interpolate(
            energy.unsqueeze(1).to(seq_out.dtype), size=seq_len, mode="linear", align_corners=False
        ).transpose(1, 2)
        scores = self.weighting_layer(torch.cat([seq_out, e_tok], dim=-1)).squeeze(-1)
    else:
        scores = self.weighting_layer(seq_out).squeeze(-1)

    if attention_mask is not None:
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
    pooled = (seq_out * torch.softmax(scores, dim=-1).unsqueeze(-1)).sum(dim=1)
    dropped = self.dropout(pooled)

    # 2. Clasificador (W2)
    if isinstance(self.classifier, SimpleLinearResidualHead):
        logits = self.classifier(dropped, energy)
    elif getattr(self.classifier, "in_features", 384) == 464:
        logits = self.classifier(torch.cat([dropped, energy.to(dropped.dtype)], dim=-1))
    else:
        logits = self.classifier(dropped)

    if not return_dict:
        return (logits,) + outputs[2:]
    return SequenceClassifierOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


def load_model_auto(path_or_id, run_name: str, masked):
    """Autodetecta la arquitectura inspeccionando directamente los tensores de model.safetensors."""
    if not isinstance(path_or_id, Path):
        model = common.load_model(path_or_id)
        return lambda seqs: common.score(model, tokenizer, seqs, masked)

    st_file = path_or_id / "model.safetensors"
    sd = load_file(str(st_file), device="cpu") if st_file.exists() else {}

    has_w1_conv = "weighting_layer.energy_scorer.0.weight" in sd
    has_w1_direct = "weighting_layer.weight" in sd and sd["weighting_layer.weight"].shape[1] == 385
    has_w2_res = "classifier.energy_clf.weight" in sd
    has_w2_mlp = "classifier.0.weight" in sd
    has_w2_direct = "classifier.weight" in sd and sd["classifier.weight"].shape[1] == 464

    is_energy = has_w1_conv or has_w1_direct or has_w2_res or has_w2_mlp or has_w2_direct
    raw_flag = "_raw" in run_name

    # Caso 1: Modelo estándar sin energía [384 -> 2]
    if not is_energy:
        model = common.load_model(path_or_id)
        return lambda seqs: common.score(model, tokenizer, seqs, masked)

    # Caso 2: MLP o concatenación estándar de common_energy (sin modificación en W1)
    if not has_w1_conv and not has_w1_direct and not has_w2_res:
        model, _, _ = common_energy.load_energy_model(
            path_or_id,
            pooling_masked=ENERGY_POOLING.get(run_name, True),
            dropout_before_concat=True,
            raw=raw_flag,
            mean_dg=-1.35,
            std_dg=0.45,
            filter_unknown=True,
        )
        return lambda seqs: common_energy.score_energy(model, tokenizer, seqs)

    # Caso 3: Cualquier variante custom (W1 conv, W1 direct, W2 residual, o W1+W2)
    model = AutoModelForSequenceClassification.from_pretrained(
        common.BASE, revision=common.BASE_REV, trust_remote_code=True
    )
    h = int(model.config.hidden_size)
    num_labels = int(model.config.num_labels)

    if has_w1_conv:
        model.weighting_layer = EnergyW1AttentionPooling(
            model.weighting_layer, with_gate=("weighting_layer.gate" in sd)
        )
    elif has_w1_direct:
        model.weighting_layer = nn.Linear(h + 1, 1)

    if has_w2_res:
        model.classifier = SimpleLinearResidualHead(model.classifier, num_labels)
    elif has_w2_direct:
        model.classifier = nn.Linear(h + 80, num_labels)

    model.load_state_dict(sd, strict=True)
    model.forward = types.MethodType(_forward_custom_energy, model)

    model.energy_len = 80
    model.raw = raw_flag
    model.mean_dg = -1.35
    model.std_dg = 0.45
    model.filter_unknown = True
    return lambda seqs: common_energy.score_energy(model, tokenizer, seqs)


def models():
    for run, (sets, train) in RUNS.items():
        d = common.ROOT / f"outputs/{run}"
        if (d / "best").exists():
            yield run, d / "best", sets + [train]
        for m in sorted(d.glob("seed_*/model")):
            yield f"{run}_{m.parent.name}", m, sets
    for name, sets in PUBLISHED.items():
        yield name, name, sets


tokenizer = None


def main():
    global tokenizer
    if args.gpu is not None and not torch.cuda.is_available():
        sys.exit(f"GPU {args.gpu} not found")
    print(f"Scoring on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    pool = common.load_pool()
    s1, s2, s3 = (common.load_split(t, pool) for t in (1, 2, 3))
    data = {
        "multispecies": pool[pool.origin == "hf_multispecies"],
        "sigma70": common.load_sigma70(),
        "pneumo_test": s2["test"],
        "task3_test": s3["test"],
        "train_task1": s1["train"],
        "train_task2": s2["train"],
        "train_task3": s3["train"],
    }
    tokenizer = common.load_tokenizer()
    for name, path, sets in models():
        out_dir = common.ROOT / f"results/scores/{name}"
        seed = (path / "SEED").read_text() if name in RUNS else None
        stale = seed is not None and (not (out_dir / "SEED").exists() or (out_dir / "SEED").read_text() != seed)
        todo = [s for s in sets if args.force or stale or not (out_dir / f"{s}.tsv").exists()]
        if not todo:
            continue

        masked = common.MASKED_IDS.get(name, (2, 3))
        run_name = name.rsplit("_seed_", 1)[0]
        score_fn = load_model_auto(path, run_name, masked)

        out_dir.mkdir(parents=True, exist_ok=True)
        for s in todo:
            d = data[s].copy()
            d["score"] = score_fn(d.segment)
            cols = [c for c in ("segment_id", "origin", "y", "kind", "sigma", "score") if c in d]
            d[cols].to_csv(out_dir / f"{s}.tsv", sep="\t", index=False)
            print(f"{name:26s} {s:14s} n={len(d)}")
        if seed is not None:
            (out_dir / "SEED").write_text(seed)


if __name__ == "__main__":
    main()