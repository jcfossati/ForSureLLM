"""Labeling soft des phrases via Claude Haiku 4.5 (+ re-label Sonnet sur cas incertains).

Usage:
    python scripts/label_dataset.py --concurrency 16

Lit :
    data/raw/{en,fr}.jsonl  ({phrase, lang, intended_class?, register?})

Écrit :
    data/labeled/{en,fr}.jsonl
        {phrase, lang, labels: {yes, no, unknown}, teacher: "haiku"|"sonnet", register?}

Re-label automatique via Sonnet si max(labels) < --reconfirm-threshold (défaut 0.6).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from forsurellm.llm import LLMClient

SYSTEM_PROMPT = """Tu es un classifieur sémantique pour phrases courtes (EN et FR).

Classifie la phrase dans UNE des trois classes :
- "yes" : exprime un accord, une confirmation, un oui (même ironique affirmatif ou implicite).
- "no" : exprime un refus, un désaccord, un non (même implicite).
- "unknown" : phrase neutre / ambiguë / hors-sujet qui n'est PAS une réponse yes/no.

Renvoie une distribution de probabilité calibrée sur {yes, no, unknown} dont la somme = 1.0.

Règles de calibration :
- Utilise des probabilités REALISTES, pas du 1.0 quand c'est ambigu.
- Si la phrase est claire et non-ambiguë : la classe dominante peut atteindre 0.90-0.98.
- Si la phrase est légèrement ambiguë : classe dominante 0.70-0.85.
- Si la phrase est très ambiguë ou ironique : dominante 0.50-0.65, répartis le reste.
- Une phrase hors-sujet (météo, fait, question, small talk) → forte masse sur "unknown".
- Le sarcasme "ah ouais bien sûr" qui veut dire NON → labelise comme "no" avec confidence modérée (0.65-0.80).

Output JSON STRICT uniquement, schéma :
{"yes": <float>, "no": <float>, "unknown": <float>}

Pas de texte avant/après. Pas de markdown. Les trois valeurs doivent sommer à 1.0 (±0.01)."""

USER_TEMPLATE = 'Phrase ({lang}) : "{phrase}"\n\nClassifie.'

LANG_NAMES = {"en": "anglais", "fr": "français"}


def parse_labels(text: str) -> dict[str, float] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        labels = {k: float(data[k]) for k in ("yes", "no", "unknown")}
    except (KeyError, ValueError, TypeError):
        return None
    total = sum(labels.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in labels.items()}


async def classify_one(
    client: LLMClient,
    phrase: str,
    lang: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, float] | None:
    user_msg = USER_TEMPLATE.format(lang=LANG_NAMES[lang], phrase=phrase)
    async with semaphore:
        for attempt in range(3):
            try:
                text = await client.complete(system=SYSTEM_PROMPT, user=user_msg)
                labels = parse_labels(text)
                if labels is not None:
                    return labels
            except Exception as e:
                if attempt == 2:
                    print(f"  [fail] {phrase!r} : {e}", file=sys.stderr)
                    return None
                await asyncio.sleep(2 ** attempt)
        return None


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def label_language(
    primary: LLMClient,
    fallback: LLMClient,
    lang: str,
    raw_path: Path,
    out_path: Path,
    concurrency: int,
    reconfirm_threshold: float,
) -> None:
    raw = load_jsonl(raw_path)
    already = {row["phrase"]: row for row in load_jsonl(out_path)}
    todo = [r for r in raw if r["phrase"] not in already]

    print(f"[{lang}] raw={len(raw)} déjà_labeled={len(already)} à_traiter={len(todo)}")
    if not todo:
        return

    semaphore = asyncio.Semaphore(concurrency)

    async def label_row(row: dict) -> dict | None:
        labels = await classify_one(primary, row["phrase"], lang, semaphore)
        if labels is None:
            return None
        teacher = "primary"
        if max(labels.values()) < reconfirm_threshold:
            refined = await classify_one(fallback, row["phrase"], lang, semaphore)
            if refined is not None:
                labels = refined
                teacher = "fallback"
        out = {"phrase": row["phrase"], "lang": lang, "labels": labels, "teacher": teacher}
        if "register" in row:
            out["register"] = row["register"]
        return out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = [label_row(r) for r in todo]
    done = 0
    sonnet_count = 0
    fail_count = 0
    with out_path.open("a", encoding="utf-8") as f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            done += 1
            if result is None:
                fail_count += 1
            else:
                if result["teacher"] == "sonnet":
                    sonnet_count += 1
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if done % 100 == 0 or done == len(tasks):
                f.flush()
                print(
                    f"[{lang}] {done}/{len(tasks)} | sonnet_refines={sonnet_count} fails={fail_count}"
                )


async def main(
    raw_dir: Path, out_dir: Path, concurrency: int, reconfirm_threshold: float
) -> None:
    primary = LLMClient("labeling", max_retries=8)
    fallback = LLMClient("labeling_fallback", max_retries=8)
    print(f"[model] primary  = {primary.model}")
    print(f"[model] fallback = {fallback.model}")
    for lang in ("en", "fr"):
        await label_language(
            primary=primary,
            fallback=fallback,
            lang=lang,
            raw_path=raw_dir / f"{lang}.jsonl",
            out_path=out_dir / f"{lang}.jsonl",
            concurrency=concurrency,
            reconfirm_threshold=reconfirm_threshold,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/labeled"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--reconfirm-threshold", type=float, default=0.6)
    args = parser.parse_args()
    asyncio.run(main(args.raw_dir, args.out_dir, args.concurrency, args.reconfirm_threshold))
