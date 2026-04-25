"""Compare ForSureLLM head-to-head against baseline classifiers.

Rationale
---------
ForSureLLM annonce 91.4% test / 88.9% adversarial. Pour décider si ça vaut
l'adoption, il faut des points de comparaison. Ce benchmark fait tourner
trois baselines sur **le même jeu adversarial** que `tools/eval.py` :

  1. **Haiku 4.5 zero-shot** - petit LLM d'Anthropic, prompt minimal.
  2. **GPT-4o-mini zero-shot** - équivalent OpenAI côté prix/latence.
  3. **Cosine similarity** sur `paraphrase-multilingual-MiniLM-L12-v2` -
     baseline embeddings sans fine-tune : on embed la phrase, on compare
     à des prototypes par classe, on prend l'argmax.

Sortie : tableau markdown + latences + JSON exhaustif pour reproductibilité.

Usage
-----
    python tools/bench_baselines.py
    python tools/bench_baselines.py --include-test    # ajoute data/splits/test.jsonl

Dépendances
-----------
Installe les packages nécessaires en plus des deps standard :

    pip install litellm sentence-transformers python-dotenv

Clés API (optionnelles - chaque baseline est skippée si la clé manque) :
  - ANTHROPIC_API_KEY  (Haiku)
  - OPENAI_API_KEY     (GPT-4o-mini)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from forsurellm import classify as forsurellm_classify  # noqa: E402

ADV_PATH    = Path(__file__).parent.parent / "evals" / "adversarial.jsonl"
TEST_PATH   = Path(__file__).parent.parent / "data" / "splits" / "test.jsonl"
REPORT_PATH = Path(__file__).parent.parent / "evals" / "bench_baselines.json"

CLASSES = ("yes", "no", "unknown")

ZERO_SHOT_SYSTEM = (
    "You classify a short phrase (French or English) as exactly one of: "
    "yes, no, unknown. Return ONLY the single word label, nothing else. "
    "Use 'unknown' when the phrase is neither a clear affirmation nor a clear "
    "negation (sarcasm, hedging, off-topic, gibberish, ambiguous)."
)

# ─── Adapters (phrase → (label, latency_ms)) ────────────────────────────────

def forsurellm_adapter(phrase: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    label, _ = forsurellm_classify(phrase)
    return label, (time.perf_counter() - t0) * 1000


def _litellm_adapter(model: str) -> Callable[[str], tuple[str, float]]:
    import litellm
    litellm.suppress_debug_info = True

    def adapter(phrase: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": ZERO_SHOT_SYSTEM},
                {"role": "user",   "content": phrase},
            ],
            max_tokens=4,
            temperature=0.0,
            num_retries=2,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        raw = (resp["choices"][0]["message"]["content"] or "").strip().lower()
        # Le LLM peut renvoyer "yes." ou "no " - on normalise.
        tokens = raw.split()[:2]
        label = next((c for c in CLASSES if c in tokens), None)
        if label is None:
            for c in CLASSES:
                if c in raw:
                    label = c
                    break
        return label or "unknown", elapsed

    return adapter


def make_cosine_adapter():
    """Baseline cosine sans fine-tune : embed la phrase et compare à des
    prototypes par classe. Les prototypes sont moyennés pour donner un centroïde
    par classe - argmax cosinus = classe prédite."""
    from sentence_transformers import SentenceTransformer, util
    import torch

    prototypes = {
        "yes": [
            "yes", "yeah", "yep", "of course", "absolutely", "sure", "definitely",
            "oui", "ouais", "bien sûr", "carrément", "tout à fait", "évidemment",
            "fax no printer",  # idiome courant dans l'eval
        ],
        "no": [
            "no", "nope", "not at all", "absolutely not", "never",
            "non", "pas du tout", "jamais", "absolument pas", "surtout pas",
        ],
        "unknown": [
            "I don't know", "maybe", "it depends", "perhaps",
            "je ne sais pas", "peut-être", "ça dépend", "je vois pas",
            "what did you eat yesterday", "blablabla",
        ],
    }
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    centroids = {
        c: torch.nn.functional.normalize(
            model.encode(prots, convert_to_tensor=True).mean(dim=0),
            dim=0,
        )
        for c, prots in prototypes.items()
    }
    C = torch.stack([centroids[c] for c in CLASSES])  # (3, D)

    def adapter(phrase: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        emb = model.encode(phrase, convert_to_tensor=True)
        emb = torch.nn.functional.normalize(emb, dim=0)
        sims = util.cos_sim(emb.unsqueeze(0), C).squeeze(0)  # (3,)
        idx = int(sims.argmax())
        return CLASSES[idx], (time.perf_counter() - t0) * 1000

    return adapter


# ─── Loaders ────────────────────────────────────────────────────────────────

def load_adversarial() -> list[dict]:
    cases = []
    with ADV_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_test() -> list[dict]:
    """Le test set a des labels soft (distribs de probas) - on argmax pour
    avoir un label net, cohérent avec le mode adversarial."""
    cases = []
    with TEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            labels = d.get("labels") or {}
            expected = max(labels, key=labels.get) if labels else None
            if expected is None:
                continue
            cases.append({
                "phrase":   d["phrase"],
                "lang":     d.get("lang", "?"),
                "expected": expected,
                "category": "test_set",
            })
    return cases


# ─── Runner ─────────────────────────────────────────────────────────────────

def run_classifier(
    name: str,
    adapter: Callable[[str], tuple[str, float]],
    cases: list[dict],
) -> dict:
    preds = []
    t0 = time.perf_counter()
    for c in cases:
        try:
            label, ms = adapter(c["phrase"])
        except Exception as e:
            label, ms = f"error:{type(e).__name__}", 0.0
        preds.append({
            "phrase":     c["phrase"],
            "expected":   c["expected"],
            "predicted":  label,
            "correct":    label == c["expected"],
            "category":   c.get("category", "?"),
            "latency_ms": round(ms, 1),
        })
    wall = time.perf_counter() - t0

    correct = sum(p["correct"] for p in preds)
    total   = len(preds)
    latencies = sorted(p["latency_ms"] for p in preds if not str(p["predicted"]).startswith("error"))
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "n": 0})
    for p in preds:
        by_cat[p["category"]]["n"] += 1
        if p["correct"]:
            by_cat[p["category"]]["ok"] += 1

    return {
        "name":             name,
        "accuracy":         round(correct / total, 3) if total else 0.0,
        "correct":          correct,
        "total":            total,
        "latency_p50_ms":   round(p50, 1),
        "latency_p95_ms":   round(p95, 1),
        "wall_time_s":      round(wall, 2),
        "by_category":      {k: {**v, "acc": round(v["ok"] / v["n"], 3)} for k, v in by_cat.items()},
        "predictions":      preds,
    }


def print_markdown_table(results: list[dict], categories: list[str]) -> str:
    lines = []
    header = ["Classifier", "Acc", "p50 (ms)", "p95 (ms)", "Wall (s)"] + categories
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for r in results:
        row = [
            r["name"],
            f"{r['accuracy']:.1%}",
            f"{r['latency_p50_ms']}",
            f"{r['latency_p95_ms']}",
            f"{r['wall_time_s']}",
        ]
        for cat in categories:
            c = r["by_category"].get(cat)
            row.append(f"{c['acc']:.0%} ({c['ok']}/{c['n']})" if c else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true", help="Ajoute le test set (avec argmax sur labels soft).")
    ap.add_argument("--skip-cosine",  action="store_true", help="Skip cosine (évite l'install sentence-transformers).")
    ap.add_argument("--skip-llms",    action="store_true", help="Skip Haiku et GPT-4o-mini.")
    args = ap.parse_args()

    # Charge .env - override=True pour couvrir les environnements qui
    # exportent ANTHROPIC_API_KEY= (vide) par défaut, sinon la clé du .env
    # est masquée.
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

    cases = load_adversarial()
    label = f"adversarial ({len(cases)} cas)"
    if args.include_test:
        test = load_test()
        cases = cases + test
        label = f"adversarial + test set ({len(cases)} cas)"

    print(f"\n=== Benchmark head-to-head - {label} ===\n")

    runs = []

    # 1. ForSureLLM - toujours dispo
    print("» ForSureLLM (ONNX int8)…", end=" ", flush=True)
    runs.append(run_classifier("ForSureLLM", forsurellm_adapter, cases))
    print("OK")

    # 2. Haiku 4.5 (Anthropic)
    if not args.skip_llms and os.environ.get("ANTHROPIC_API_KEY"):
        print("» Haiku 4.5 (anthropic)…", end=" ", flush=True)
        try:
            runs.append(run_classifier(
                "Haiku 4.5 zero-shot",
                _litellm_adapter("anthropic/claude-haiku-4-5-20251001"),
                cases,
            ))
            print("OK")
        except Exception as e:
            print(f"FAIL ({type(e).__name__}: {e})")
    else:
        print("» Haiku skippé (pas d'ANTHROPIC_API_KEY ou --skip-llms)")

    # 3. GPT-4o-mini (OpenAI)
    if not args.skip_llms and os.environ.get("OPENAI_API_KEY"):
        print("» GPT-4o-mini (openai)…", end=" ", flush=True)
        try:
            runs.append(run_classifier(
                "GPT-4o-mini zero-shot",
                _litellm_adapter("gpt-4o-mini"),
                cases,
            ))
            print("OK")
        except Exception as e:
            print(f"FAIL ({type(e).__name__}: {e})")
    else:
        print("» GPT-4o-mini skippé (pas d'OPENAI_API_KEY ou --skip-llms)")

    # 4. Cosine similarity
    if not args.skip_cosine:
        print("» Cosine MiniLM (local)…", end=" ", flush=True)
        try:
            runs.append(run_classifier(
                "Cosine MiniLM-L12",
                make_cosine_adapter(),
                cases,
            ))
            print("OK")
        except Exception as e:
            print(f"FAIL ({type(e).__name__}: {e})")

    # ─── Rapport ──────────────────────────────────────────────────
    all_cats = sorted({p["category"] for r in runs for p in r["predictions"]})
    table = print_markdown_table(runs, all_cats)
    print("\n" + table + "\n")

    # Focus sur les catégories où ForSureLLM est faible (rapport last_report.json)
    weak = ["slang_abbrev", "repetition", "degenerate", "compound"]
    weak_in_runs = [c for c in weak if c in all_cats]
    if weak_in_runs:
        print(f"\n### Focus catégories faibles ({', '.join(weak_in_runs)})\n")
        print(print_markdown_table(runs, weak_in_runs))

    out = {
        "cases_count": len(cases),
        "source": label,
        "runs": [
            # On externalise les predictions pour ne pas spammer la console
            {k: v for k, v in r.items() if k != "predictions"}
            for r in runs
        ],
        "predictions": {r["name"]: r["predictions"] for r in runs},
    }
    REPORT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport complet -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
