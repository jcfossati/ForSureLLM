"""Audit du test set : prédit chaque phrase, liste les échecs, demande à
Sonnet d'arbitrer pour distinguer les vraies erreurs modèle des labels
douteux (ground truth pourri).

Sortie : evals/test_audit.json + résumé console.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tools._variant  # noqa: F401 — must come before forsurellm import

import litellm
from dotenv import load_dotenv

from forsurellm import classify

load_dotenv(override=True)
litellm.suppress_debug_info = True


JUDGE_PROMPT = """Tu arbitres entre un label de référence et la prédiction d'un modèle ML pour une classification yes/no/unknown sur des phrases courtes.

Phrase : "{phrase}"
Langue : {lang}
Label de référence : {expected}
Prédiction du modèle : {predicted} (confiance {conf:.2f})

Question : qui a raison ? Réponds en JSON strict :
{{"verdict": "label" | "model" | "ambiguous", "best_label": "yes" | "no" | "unknown", "reason": "1 phrase max"}}

Règles :
- "label" : le label de référence est correct, le modèle se trompe
- "model" : la prédiction du modèle est correcte, le label est mauvais
- "ambiguous" : phrase intrinsèquement ambiguë sans contexte, aucune réponse n'est clairement bonne

Sois strict : ne dis "model" que si tu es sûr que le label de référence est faux."""


def argmax_label(labels: dict[str, float]) -> str:
    return max(labels, key=labels.get)


def judge_one(row: dict, conf: float) -> dict | None:
    try:
        resp = litellm.completion(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                phrase=row["phrase"],
                lang=row["lang"],
                expected=row["expected"],
                predicted=row["predicted"],
                conf=conf,
            )}],
            max_tokens=200,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        return {"verdict": "error", "best_label": row["expected"], "reason": str(e)[:80]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=Path("data/splits/test.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("evals/test_audit.json"))
    parser.add_argument("--judge", action="store_true", help="ask Sonnet to arbitrate failures")
    parser.add_argument("--judge-limit", type=int, default=200)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.split.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[load] {len(rows)} phrases")

    failures = []
    correct = 0
    for r in rows:
        expected = argmax_label(r["labels"])
        predicted, conf = classify(r["phrase"])
        if predicted == expected:
            correct += 1
        else:
            failures.append({
                "phrase": r["phrase"],
                "lang": r["lang"],
                "expected": expected,
                "predicted": predicted,
                "confidence": conf,
                "label_dist": r["labels"],
                "teacher": r.get("teacher", ""),
                "source_idiom": r.get("source_idiom", ""),
            })

    acc = correct / len(rows)
    print(f"[acc] {correct}/{len(rows)} = {acc:.1%}")
    print(f"[fail] {len(failures)} failures")

    by_kind = Counter()
    for f in failures:
        by_kind[(f["expected"], f["predicted"])] += 1
    print("\nMatrice de confusion (expected -> predicted) :")
    for (e, p), n in by_kind.most_common():
        print(f"  {e:8s} -> {p:8s}  {n:3d}")

    failures.sort(key=lambda f: -f["confidence"])

    print(f"\nTop 20 failures (high-conf flips = bugs candidats) :")
    for f in failures[:20]:
        print(f"  [{f['expected']:7s} -> {f['predicted']:7s} {f['confidence']:.2f}] '{f['phrase']}' ({f['lang']})")

    if args.judge:
        print(f"\nArbitrage Sonnet sur top {args.judge_limit} (par confiance décroissante)...")
        t0 = time.time()
        verdicts = Counter()
        for i, f in enumerate(failures[:args.judge_limit], 1):
            verdict = judge_one(f, f["confidence"])
            f["judge"] = verdict
            verdicts[verdict.get("verdict", "error")] += 1
            if i % 20 == 0:
                print(f"  [{i}/{args.judge_limit}] {dict(verdicts)} ({time.time()-t0:.0f}s)")
        print(f"\nVerdicts : {dict(verdicts)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"accuracy": acc, "n": len(rows), "failures": failures}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nRapport -> {args.out}")


if __name__ == "__main__":
    main()
