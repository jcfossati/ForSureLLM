"""REPL interactif pour tester le classifier.

Usage:
    python scripts/repl.py

Commandes :
    <phrase>          classifie la phrase
    /t <float>        change le threshold (défaut 0.0)
    /q                quitte
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from forsurellm import classify


BAR_WIDTH = 20
COLORS = {"yes": "\033[92m", "no": "\033[91m", "unknown": "\033[93m"}
RESET = "\033[0m"


def bar(value: float, width: int = BAR_WIDTH) -> str:
    n = int(round(value * width))
    return "█" * n + "░" * (width - n)


def main() -> None:
    threshold = 0.0
    print("REPL classifier yes/no/unknown. Tape /q pour quitter, /t <val> pour changer le seuil.\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in {"/q", "/quit", "/exit"}:
            break
        if line.startswith("/t "):
            try:
                threshold = float(line[3:].strip())
                print(f"  threshold = {threshold}\n")
            except ValueError:
                print("  usage: /t <float>\n")
            continue

        label, conf = classify(line, threshold=threshold)

        from forsurellm.classifier import _load, _softmax
        import numpy as np
        tokenizer, session, classes, input_names = _load()
        enc = tokenizer.encode(line)
        feeds = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
        feeds = {k: v for k, v in feeds.items() if k in input_names}
        logits = session.run(None, feeds)[0][0]
        probs = _softmax(logits)

        color = COLORS.get(label, "")
        print(f"  => {color}{label.upper()}{RESET} (conf={conf:.3f})")
        for cls, p in zip(classes, probs):
            mark = "<-" if cls == label else "  "
            print(f"    {cls:<8} {bar(float(p))} {p:.3f} {mark}")
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
