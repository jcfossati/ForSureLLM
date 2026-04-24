"""Temperature scaling post-hoc pour recalibrer le modèle.

Apprend un scalaire T tel que softmax(logits / T) minimise la NLL sur le val set.
T > 1  = softens predictions (réduit sur-confiance)
T < 1  = sharpens predictions (rare en pratique)

Usage:
    python scripts/calibrate.py --src checkpoints/best

Output :
    checkpoints/best/temperature.json   -> {"temperature": <float>}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASSES = ["yes", "no", "unknown"]


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float32)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    e = 0.0
    for i in range(n_bins):
        m = (confs > edges[i]) & (confs <= edges[i + 1])
        if m.sum() > 0:
            e += (m.sum() / n) * abs(correct[m].mean() - confs[m].mean())
    return float(e)


def collect_logits(model, tokenizer, rows, device, max_length=64, batch_size=64):
    all_logits, all_soft = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = [r["phrase"] for r in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            logits = model(**enc).logits.cpu().numpy()
            soft = np.array([[r["labels"][c] for c in CLASSES] for r in batch], dtype=np.float32)
            all_logits.append(logits)
            all_soft.append(soft)
    return np.concatenate(all_logits), np.concatenate(all_soft)


def optimize_temperature(logits: np.ndarray, soft: np.ndarray) -> float:
    """LBFGS sur NLL avec hard labels (argmax des soft labels).

    Standard temperature scaling : on minimise -log p(y_hard | x, T) pour T scalaire.
    Ça aligne l'objectif avec l'ECE, contrairement à la soft-NLL.
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    hard = torch.tensor(soft.argmax(axis=-1), dtype=torch.long)
    log_T = torch.zeros(1, requires_grad=True)

    optim = torch.optim.LBFGS([log_T], lr=0.05, max_iter=200, tolerance_grad=1e-7)

    def closure():
        optim.zero_grad()
        T = torch.exp(log_T)
        loss = F.cross_entropy(logits_t / T, hard)
        loss.backward()
        return loss

    optim.step(closure)
    return float(torch.exp(log_T).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("checkpoints/best"))
    parser.add_argument("--val-path", type=Path, default=Path("data/splits/val.jsonl"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[calibrate] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    model = AutoModelForSequenceClassification.from_pretrained(str(args.src)).to(device)

    rows = [json.loads(l) for l in args.val_path.open(encoding="utf-8") if l.strip()]
    print(f"[calibrate] {len(rows)} phrases val")

    logits, soft = collect_logits(model, tokenizer, rows, device)
    labels = soft.argmax(axis=-1)

    # Before
    probs_before = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    acc_before = (probs_before.argmax(-1) == labels).mean()
    ece_before = ece(probs_before, labels)
    print(f"[before]  T=1.000  acc={acc_before:.3f}  ece={ece_before:.4f}")

    T = optimize_temperature(logits, soft)

    # After
    probs_after = torch.softmax(torch.tensor(logits) / T, dim=-1).numpy()
    acc_after = (probs_after.argmax(-1) == labels).mean()
    ece_after = ece(probs_after, labels)
    print(f"[after]   T={T:.3f}  acc={acc_after:.3f}  ece={ece_after:.4f}")
    print(f"[gain]    ECE: {ece_before:.4f} -> {ece_after:.4f}  ({(1-ece_after/ece_before)*100:+.1f}%)")

    out_path = args.src / "temperature.json"
    out_path.write_text(json.dumps({"temperature": T}, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
