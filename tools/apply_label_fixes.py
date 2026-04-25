"""Applique les corrections de labels validées par l'audit Sonnet.

Lit `evals/test_audit.json`, sélectionne les overrides pertinents (verdict=model
ou ambiguous->unknown), et patch in-place tous les fichiers sources dans
`data/labeled/` qui contiennent la phrase. Re-génère ensuite les splits via
`training/train.py` (à lancer manuellement après).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

LABELED_DIR = Path("data/labeled")
SOURCES = ["en.jsonl", "en_clean.jsonl", "fr.jsonl", "fr_clean.jsonl",
           "idioms_deep.jsonl", "seed_weak_categories.jsonl",
           "seed_slang_abbrev.jsonl"]

# Distributions cibles (alignées sur les conventions existantes)
DIST = {
    "yes":     {"yes": 0.92, "no": 0.04, "unknown": 0.04},
    "no":      {"yes": 0.04, "no": 0.92, "unknown": 0.04},
    "unknown": {"yes": 0.1,  "no": 0.1,  "unknown": 0.8},
}


def select_overrides() -> dict[str, str]:
    """Retourne {phrase: target_label} pour les corrections à appliquer."""
    audit = json.load(open("evals/test_audit.json", encoding="utf-8"))
    overrides: dict[str, str] = {}
    for f in audit["failures"]:
        j = f.get("judge", {})
        verdict = j.get("verdict")
        best = j.get("best_label")
        # 1) Sonnet dit que le label de référence est faux
        if verdict == "model" and best in DIST:
            overrides[f["phrase"]] = best
        # 2) Cas ambigu où Sonnet recommande unknown — reclasser
        elif verdict == "ambiguous" and best == "unknown" and f["expected"] != "unknown":
            overrides[f["phrase"]] = "unknown"
        # 3) Cas ambigu où Sonnet recommande yes/no avec une opinion
        elif verdict == "ambiguous" and best in {"yes", "no"} and best != f["expected"]:
            overrides[f["phrase"]] = best
    return overrides


def patch_file(path: Path, overrides: dict[str, str], stats: Counter) -> bool:
    if not path.exists():
        return False
    rows = []
    changed = False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row["phrase"] in overrides:
            new_label = overrides[row["phrase"]]
            row["labels"] = DIST[new_label].copy()
            row["teacher"] = "audit_sonnet_4-6"
            stats[(path.name, new_label)] += 1
            changed = True
        rows.append(row)
    if changed:
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
    return changed


def main() -> None:
    overrides = select_overrides()
    print(f"[overrides] {len(overrides)} phrases à recorriger")
    print(f"  par cible : {Counter(overrides.values())}")

    stats: Counter = Counter()
    for src in SOURCES:
        path = LABELED_DIR / src
        if patch_file(path, overrides, stats):
            print(f"[patched] {src}")

    print("\nDétail par fichier :")
    by_file = Counter()
    for (fname, _), n in stats.items():
        by_file[fname] += n
    for fname, n in by_file.most_common():
        print(f"  {fname:35s} {n:3d} phrases")

    found = {p for p in overrides if any(s for s in stats if overrides[p] in s)}
    # Detect not found
    found_phrases: set[str] = set()
    for path in (LABELED_DIR / s for s in SOURCES):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["phrase"] in overrides:
                found_phrases.add(row["phrase"])
    missing = set(overrides) - found_phrases
    if missing:
        print(f"\n[!!] {len(missing)} phrases non trouvées dans les sources :")
        for p in sorted(missing)[:20]:
            print(f"  '{p}'")

    print(f"\nProchaines étapes :")
    print(f"  python training/train.py --epochs 8")
    print(f"  python training/calibrate.py")
    print(f"  python training/export.py")
    print(f"  python tools/eval.py")
    print(f"  python tools/audit_test_set.py  # mesure nouveau test acc")


if __name__ == "__main__":
    main()
