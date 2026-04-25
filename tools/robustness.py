"""Robustesse aux variantes de surface.

Pour chaque phrase de l'eval adversarial, on génère un set de variantes
graphiques (casse, ponctuation, espaces, accents). On vérifie que la
prédiction reste *stable* (même label) quelle que soit la variante.

Métrique : pour chaque transformation, taux de variantes qui flippent le
label par rapport à la phrase canonique. Identifie les transformations
fragiles et les phrases fragiles.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

from forsurellm.classifier import classify


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


TRANSFORMS: dict[str, callable] = {
    "lower": lambda s: s.lower(),
    "upper": lambda s: s.upper(),
    "title": lambda s: s.title(),
    "trail_excl": lambda s: s.rstrip(" .!?") + "!",
    "trail_excl3": lambda s: s.rstrip(" .!?") + "!!!",
    "trail_qmark": lambda s: s.rstrip(" .!?") + "?",
    "trail_dots": lambda s: s.rstrip(" .!?") + "...",
    "strip_punct": lambda s: "".join(c for c in s if c.isalnum() or c.isspace()).strip(),
    "lead_space": lambda s: "  " + s,
    "trail_space": lambda s: s + "   ",
    "double_space": lambda s: s.replace(" ", "  "),
    "no_accent": strip_accents,
    "mixed_case": lambda s: "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(s)),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=Path("evals/adversarial.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("evals/robustness_report.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.eval.read_text(encoding="utf-8").splitlines() if line.strip()]

    by_transform: dict[str, dict] = {name: {"flips": 0, "total": 0, "examples": []} for name in TRANSFORMS}
    fragile_phrases: list[dict] = []

    total_variants = 0
    total_flips = 0

    for row in rows:
        phrase = row["phrase"]
        canonical_label, canonical_conf = classify(phrase)

        phrase_flips = []
        for tname, tfn in TRANSFORMS.items():
            variant = tfn(phrase)
            if variant == phrase or not variant.strip():
                continue
            v_label, v_conf = classify(variant)
            by_transform[tname]["total"] += 1
            total_variants += 1
            if v_label != canonical_label:
                by_transform[tname]["flips"] += 1
                total_flips += 1
                if len(by_transform[tname]["examples"]) < 5:
                    by_transform[tname]["examples"].append({
                        "phrase": phrase,
                        "variant": variant,
                        "canonical": f"{canonical_label} ({canonical_conf:.2f})",
                        "got": f"{v_label} ({v_conf:.2f})",
                    })
                phrase_flips.append({"transform": tname, "variant": variant, "got": v_label})

        if phrase_flips:
            fragile_phrases.append({
                "phrase": phrase,
                "canonical": canonical_label,
                "category": row.get("category", ""),
                "flips": phrase_flips,
            })

    overall_stability = 1 - (total_flips / total_variants) if total_variants else 1.0

    print(f"=== Robustness ({len(rows)} phrases canonical, {total_variants} variantes testees) ===\n")
    print(f"Stabilite globale : {overall_stability:.1%}  ({total_variants - total_flips}/{total_variants})\n")

    print("Par transformation :")
    for tname, stats in sorted(by_transform.items(), key=lambda kv: kv[1]["flips"], reverse=True):
        if stats["total"] == 0:
            continue
        rate = stats["flips"] / stats["total"]
        marker = "[!!]" if rate > 0.05 else "[OK]"
        print(f"  {marker} {tname:18s} {stats['flips']:3d}/{stats['total']:3d} flips ({rate:5.1%})")
        if stats["examples"] and rate > 0.05:
            for ex in stats["examples"][:3]:
                print(f"        '{ex['phrase']}' -> '{ex['variant']}'  {ex['canonical']} -> {ex['got']}")

    print(f"\nPhrases fragiles ({len(fragile_phrases)}/{len(rows)}) :")
    for fp in fragile_phrases[:15]:
        names = ",".join(f["transform"] for f in fp["flips"])
        print(f"  [{fp['category']:15s}] '{fp['phrase']}' (canonical={fp['canonical']}) flippe sur : {names}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "n_phrases": len(rows),
                "n_variants": total_variants,
                "n_flips": total_flips,
                "stability": overall_stability,
                "by_transform": by_transform,
                "fragile_phrases": fragile_phrases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nRapport -> {args.out}")


if __name__ == "__main__":
    main()
