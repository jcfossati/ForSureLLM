"""Distillation KL-divergence sur MiniLM multilingue (3 classes yes/no/unknown).

Usage:
    python scripts/train.py --epochs 4 --batch-size 32 --lr 2e-5

Entrée :
    data/labeled/{en,fr}.jsonl

Sorties :
    data/splits/{train,val,test}.jsonl   (splits reproductibles, stratifiés)
    checkpoints/best/                    (modèle + tokenizer HF)
    checkpoints/metrics.json             (accuracy + ECE sur test)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BACKBONE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CLASSES = ["yes", "no", "unknown"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}


def load_labeled(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def make_splits(rows: list[dict], seed: int) -> tuple[list, list, list]:
    """Split 80/10/10 stratifié par (argmax_class, lang)."""
    strata = [f"{max(r['labels'], key=r['labels'].get)}_{r['lang']}" for r in rows]
    train, temp, _, strata_temp = train_test_split(
        rows, strata, test_size=0.2, random_state=seed, stratify=strata
    )
    val, test, _, _ = train_test_split(
        temp, strata_temp, test_size=0.5, random_state=seed, stratify=strata_temp
    )
    return train, val, test


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_dataset(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    texts = [r["phrase"] for r in rows]
    soft = [[r["labels"][c] for c in CLASSES] for r in rows]
    enc = tokenizer(texts, truncation=True, max_length=max_length)
    enc["soft_labels"] = soft
    return Dataset.from_dict(enc)


class SoftLabelCollator:
    def __init__(self, tokenizer):
        self.pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, features: list[dict]) -> dict:
        soft = torch.tensor([f.pop("soft_labels") for f in features], dtype=torch.float32)
        batch = self.pad(features)
        batch["soft_labels"] = soft
        return batch


class KLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        soft = inputs.pop("soft_labels")
        outputs = model(**inputs)
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        loss = F.kl_div(log_probs, soft, reduction="batchmean")
        return (loss, outputs) if return_outputs else loss


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float32)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (confs > edges[i]) & (confs <= edges[i + 1])
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confs[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def make_compute_metrics():
    def compute_metrics(eval_pred):
        logits, soft = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        true = np.array(soft).argmax(axis=-1)
        preds = probs.argmax(axis=-1)
        return {
            "accuracy": float((preds == true).mean()),
            "ece": expected_calibration_error(probs, true),
        }
    return compute_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-dir", type=Path, default=Path("data/labeled"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--backbone", default=BACKBONE)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    sources = [args.labeled_dir / "en_clean.jsonl", args.labeled_dir / "fr_clean.jsonl"]
    deep = args.labeled_dir / "idioms_deep.jsonl"
    if deep.exists():
        sources.append(deep)
    weak = args.labeled_dir / "seed_weak_categories.jsonl"
    if weak.exists():
        sources.append(weak)
    slang = args.labeled_dir / "seed_slang_abbrev.jsonl"
    if slang.exists():
        sources.append(slang)
    rows = load_labeled(sources)
    print(f"[data] {len(rows)} phrases labeled total")

    train_rows, val_rows, test_rows = make_splits(rows, seed=args.seed)
    save_jsonl(train_rows, args.splits_dir / "train.jsonl")
    save_jsonl(val_rows, args.splits_dir / "val.jsonl")
    save_jsonl(test_rows, args.splits_dir / "test.jsonl")
    print(f"[split] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.backbone,
        num_labels=len(CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = to_dataset(train_rows, tokenizer, args.max_length)
    val_ds = to_dataset(val_rows, tokenizer, args.max_length)
    test_ds = to_dataset(test_rows, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        label_names=["soft_labels"],
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = KLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=SoftLabelCollator(tokenizer),
        compute_metrics=make_compute_metrics(),
    )

    trainer.train()

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(f"[test] {test_metrics}")

    best_dir = args.output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print(f"[done] best model -> {best_dir}")


if __name__ == "__main__":
    main()
