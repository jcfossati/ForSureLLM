"""Comparaison complète : modèle full vs pruned sur l'ensemble du corpus
(test split + adversarial bench), avec sweep de seuils.

Sortie :
1. Diff prediction-par-prediction full vs pruned
2. Tableau accuracy/coverage par seuil pour chaque variante
3. Tableau identification du seuil optimal selon priorité (action confirmation
   vs triage)

Usage:
    python tools/compare_variants.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "forsurellm" / "models"

THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def _load_variant(suffix: str):
    cfg = json.loads((MODELS_DIR / f"config{suffix}.json").read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(str(MODELS_DIR / f"tokenizer{suffix}.json"))
    tok.enable_truncation(max_length=cfg["max_length"])
    sess = ort.InferenceSession(
        str(MODELS_DIR / f"forsurellm-int8{suffix}.onnx"),
        providers=["CPUExecutionProvider"],
    )
    return {
        "name": "full" if suffix == "" else "pruned",
        "tok": tok,
        "sess": sess,
        "input_names": {i.name for i in sess.get_inputs()},
        "classes": cfg["classes"],
        "T": float(cfg.get("temperature", 1.0)),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _classify(v, phrase: str) -> tuple[str, np.ndarray]:
    enc = v["tok"].encode(phrase.lower())
    feeds = {"input_ids": np.array([enc.ids], dtype=np.int64),
             "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
    if "token_type_ids" in v["input_names"]:
        feeds["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
    feeds = {k: x for k, x in feeds.items() if k in v["input_names"]}
    logits = v["sess"].run(None, feeds)[0][0]
    probs = _softmax(logits / v["T"])
    return v["classes"][int(probs.argmax())], probs


def load_corpus():
    rows = []
    for path in [ROOT / "data" / "splits" / "test.jsonl",
                 ROOT / "evals" / "adversarial.jsonl"]:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            phrase = r.get("phrase", "")
            if not phrase:
                continue
            if "expected" in r:
                expected = r["expected"]
            elif "labels" in r:
                expected = max(r["labels"], key=r["labels"].get)
            else:
                continue
            rows.append({"phrase": phrase, "expected": expected,
                         "source": "adversarial" if "expected" in r else "test"})
    return rows


def run_predictions(variant, rows):
    out = []
    for r in rows:
        label, probs = _classify(variant, r["phrase"])
        out.append({"phrase": r["phrase"], "expected": r["expected"],
                    "predicted": label, "max_conf": float(probs.max()),
                    "probs": {c: float(p) for c, p in zip(variant["classes"], probs)},
                    "source": r["source"]})
    return out


def diff_predictions(full_preds, pruned_preds):
    diffs = []
    for f, p in zip(full_preds, pruned_preds):
        if f["predicted"] != p["predicted"]:
            diffs.append({
                "phrase": f["phrase"], "expected": f["expected"], "source": f["source"],
                "full": f"{f['predicted']} ({f['max_conf']:.2f})",
                "pruned": f"{p['predicted']} ({p['max_conf']:.2f})",
            })
    return diffs


def threshold_metrics(preds, threshold):
    """Returns:
      - accuracy : % correct over all inputs (forced unknowns count as wrong if
        expected wasn't unknown, correct if it was)
      - coverage : % of inputs where final ∈ {yes, no}
      - confident_precision : P(correct | final ∈ {yes, no}) — KEY metric for
        action confirmation, where a wrong yes/no is much worse than an
        unnecessary unknown re-prompt
      - n_unknown_forced : how many predictions got pushed to unknown by threshold
    """
    correct = 0
    covered = 0
    covered_correct = 0
    unknown_forced = 0
    for p in preds:
        argmax = p["predicted"]
        max_conf = p["max_conf"]
        if max_conf < threshold and argmax != "unknown":
            final = "unknown"
            unknown_forced += 1
        else:
            final = argmax
        is_correct = (final == p["expected"])
        if final != "unknown":
            covered += 1
            if is_correct:
                covered_correct += 1
        if is_correct:
            correct += 1
    n = len(preds)
    return {
        "threshold": threshold,
        "n": n,
        "accuracy": correct / n,
        "coverage": covered / n,
        "confident_precision": (covered_correct / covered) if covered else 0.0,
        "n_unknown_forced": unknown_forced,
    }


def main():
    print("[load] full variant...")
    full = _load_variant("")
    print("[load] pruned variant...")
    pruned = _load_variant("_fr-en")

    print("[corpus] loading test + adversarial...")
    rows = load_corpus()
    print(f"[corpus] {len(rows)} rows total")

    print("[predict] full...")
    t0 = time.time()
    full_preds = run_predictions(full, rows)
    print(f"  done ({time.time()-t0:.1f}s)")

    print("[predict] pruned...")
    t0 = time.time()
    pruned_preds = run_predictions(pruned, rows)
    print(f"  done ({time.time()-t0:.1f}s)")

    # 1. Diff
    diffs = diff_predictions(full_preds, pruned_preds)
    print(f"\n=== Diff full vs pruned : {len(diffs)} predictions différentes / {len(rows)} ===")
    if diffs:
        by_source = Counter(d["source"] for d in diffs)
        print(f"  par source : {dict(by_source)}")
        print("  echantillon (10 premiers) :")
        for d in diffs[:10]:
            print(f"    [{d['source']:11s}] {d['phrase']!r:30s}  expected={d['expected']:7s}  full={d['full']:18s}  pruned={d['pruned']}")
    else:
        print("  -> les deux variantes prédisent strictement la même chose")

    # 2. Threshold sweep
    print("\n=== Threshold sweep ===")
    for variant_name, preds in [("full", full_preds), ("pruned", pruned_preds)]:
        print(f"\n  {variant_name} :")
        print(f"  {'threshold':>10s}  {'accuracy':>10s}  {'coverage':>10s}  {'conf_prec':>10s}  {'forced':>8s}")
        print(f"  {'(all)':>10s}  {'(global)':>10s}  {'(yes/no)':>10s}  {'(when y/n)':>10s}  {'unkn':>8s}")
        rows_metrics = [threshold_metrics(preds, t) for t in THRESHOLDS]
        for m in rows_metrics:
            print(f"  {m['threshold']:>10.2f}  {m['accuracy']:>9.1%}  {m['coverage']:>9.1%}  {m['confident_precision']:>9.1%}  {m['n_unknown_forced']:>8d}")

    # 3. Save reports
    out = {
        "n_rows": len(rows),
        "n_diffs": len(diffs),
        "diffs": diffs[:50],
        "thresholds": {
            "full": [threshold_metrics(full_preds, t) for t in THRESHOLDS],
            "pruned": [threshold_metrics(pruned_preds, t) for t in THRESHOLDS],
        },
    }
    out_path = ROOT / "evals" / "compare_variants.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
