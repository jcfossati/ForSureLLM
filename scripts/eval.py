"""Évalue le classifier sur l'eval set adversarial.

Usage:
    python scripts/eval.py

Input  : tests/eval_adversarial.jsonl
Output : rapport console + tests/eval_report.json

Chaque phrase a une catégorie ({canonical, hedging, missing_accents, typos,
slang_abbrev, degenerate, repetition, sarcasm, compound, off_topic, interjection}).
Le flag `known_failure: true` marque les cas où on s'attend à un échec Phase 1 —
ils sont comptés séparément dans le rapport.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from yesno import classify

EVAL_PATH = Path(__file__).parent.parent / "tests" / "eval_adversarial.jsonl"
REPORT_PATH = Path(__file__).parent.parent / "tests" / "eval_report.json"


def main() -> None:
    cases = []
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    buckets: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        label, conf = classify(case["phrase"])
        case["predicted"] = label
        case["confidence"] = round(conf, 3)
        case["correct"] = (label == case["expected"])
        buckets[case["category"]].append(case)

    print(f"\n=== Eval adversarial ({len(cases)} cas) ===\n")

    total_correct = total_known_failures = total_unexpected = 0
    category_stats = {}

    for cat in sorted(buckets):
        items = buckets[cat]
        correct = sum(1 for c in items if c["correct"])
        known_fail = sum(1 for c in items if not c["correct"] and c.get("known_failure"))
        unexpected_fail = sum(
            1 for c in items if not c["correct"] and not c.get("known_failure")
        )
        total_correct += correct
        total_known_failures += known_fail
        total_unexpected += unexpected_fail

        acc = correct / len(items)
        marker = "OK" if unexpected_fail == 0 else "!!"
        print(f"  [{marker}] {cat:<18} {correct:>2}/{len(items)} ({acc:.0%})")
        category_stats[cat] = {
            "correct": correct,
            "total": len(items),
            "accuracy": round(acc, 3),
            "known_failures": known_fail,
            "unexpected_failures": unexpected_fail,
        }

        for c in items:
            if not c["correct"]:
                tag = "expected_fail" if c.get("known_failure") else "REGRESSION"
                print(
                    f"      [{tag}] {c['phrase']!r:<45} "
                    f"expected={c['expected']:<7} got={c['predicted']} ({c['confidence']:.2f})"
                )

    total = len(cases)
    overall_acc = total_correct / total
    print(f"\nTotal : {total_correct}/{total} corrects ({overall_acc:.1%})")
    print(f"  dont échecs attendus (sarcasm etc.) : {total_known_failures}")
    print(f"  échecs INATTENDUS (régressions)    : {total_unexpected}")

    report = {
        "total": total,
        "correct": total_correct,
        "accuracy": round(overall_acc, 3),
        "known_failures": total_known_failures,
        "unexpected_failures": total_unexpected,
        "by_category": category_stats,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport sauvegardé -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
