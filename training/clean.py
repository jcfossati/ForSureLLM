"""Filtre les unknowns 'pure noise' (faits soutenus/neutres très confiants).

Usage:
    python scripts/clean_dataset.py

Lit :
    data/labeled/{en,fr}.jsonl
Écrit :
    data/labeled/{en,fr}_clean.jsonl

Règle : drop si argmax=='unknown' ET unknown_prob>0.9 ET register∈{neutre,soutenu}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DROP_CONF_THRESHOLD = 0.9
DROP_REGISTERS = {"neutre", "soutenu"}


def should_drop(row: dict) -> bool:
    labels = row["labels"]
    argmax = max(labels, key=labels.get)
    if argmax != "unknown":
        return False
    if labels["unknown"] <= DROP_CONF_THRESHOLD:
        return False
    return row.get("register") in DROP_REGISTERS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-dir", type=Path, default=Path("data/labeled"))
    args = parser.parse_args()

    for lang in ("en", "fr"):
        src = args.labeled_dir / f"{lang}.jsonl"
        dst = args.labeled_dir / f"{lang}_clean.jsonl"
        kept = dropped = 0
        with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                if should_drop(row):
                    dropped += 1
                    continue
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
        print(f"[{lang}] kept={kept} dropped={dropped} -> {dst}")


if __name__ == "__main__":
    main()
