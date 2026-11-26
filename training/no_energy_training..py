"""Baseline control script: frozen ProkBERT encoder + attention pooling + linear head WITHOUT SantaLucia energy profiles."""

# ==================================
# Imports
# ==================================

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

import common



if __name__ == "__main__":
    import launch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    parser.add_argument("--output-dir", default="outputs/baseline_frozen")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=-1, help="smoke tests only")
    parser.add_argument("--dropout", type=float, default=None,
                        help="dropout (default: pretrained config)")
    launch.add_args(parser)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    launch.setup(args, out_dir)  # before torch is imported; several GPUs: one process per GPU, then summary and exit

    from datasets import Dataset
    from scipy.special import softmax
    from sklearn.metrics import roc_auc_score
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

    # ==================================
    # Function/method definitions
    # ==================================
    device = launch.device(args)
    print(f"Training on {device}")

    tokenizer = common.load_tokenizer()
    split = common.load_split(3)  # 3 is the new one


    def tokenize(batch):
        enc = common.encode(tokenizer, batch["segment"])
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": batch["y"]}


    data = {s: Dataset.from_pandas(split[s][["segment", "y"]], preserve_index=False)
            .map(tokenize, batched=True, batch_size=1000, remove_columns=["segment", "y"]) for s in ("train", "val")}


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        p = softmax(logits, axis=-1)[:, 1]  # AUC on the class probability, not on a raw logit
        return {"roc_auc": roc_auc_score(labels, p)}


    # ==================================
    # Training loop for all seeds
    # ==================================
    # The model is the same; --dropout overrides the pretrained config at load time
    if args.dropout is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            common.BASE, revision=common.BASE_REV, trust_remote_code=True)
    else:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(common.BASE, revision=common.BASE_REV, trust_remote_code=True)
        cfg.classification_dropout_rate = args.dropout
        cfg.hidden_dropout_prob = args.dropout
        cfg.attention_probs_dropout_prob = args.dropout
        model = AutoModelForSequenceClassification.from_pretrained(
            common.BASE, config=cfg, revision=common.BASE_REV, trust_remote_code=True)

     training_args = TrainingArguments(
        output_dir=str(seed_dir / "checkpoints"),
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        num_train_epochs=10,
        max_steps=args.max_steps,
            # Evaluate and save on the same steps, so the reloaded "best" checkpoint is really the best one.
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
       load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # the best checkpoint and early stopping follow the validation loss
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
    )

        # Create the trainer
        trainer = Trainer(model=model, args=training_args, train_dataset=data["train"], eval_dataset=data["val"],
                          compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=10)])

        # TRAIN!
        train_output = trainer.train()

        state = trainer.state
        assert state.best_model_checkpoint.endswith(f"checkpoint-{state.best_global_step}"), "stale best checkpoint"
        print(f"seed {seed}: best step {state.best_global_step} of {state.global_step}, val loss {state.best_metric:.4f}")

        trainer.save_model(str(seed_dir / "model"))
        common.save_seed(seed_dir, val_row(seed, model, tokenizer, split["val"])
                         | {"best_step": state.best_global_step, "last_step": state.global_step,
                            "gpu": device, "train_min": train_output.metrics["train_runtime"] / 60})

    # ==================================
    # Summarising after the end
    # ==================================
    launch.finish(args, out_dir)  # seeds.tsv and best/ (training/summarise_seeds.py)