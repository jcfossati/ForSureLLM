"""Génère ~10k phrases courtes EN/FR via Claude Sonnet 4.6.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_dataset.py --target-per-lang 5000 --concurrency 8

Output:
    data/raw/en.jsonl
    data/raw/fr.jsonl

Chaque ligne : {"phrase": "...", "lang": "en|fr", "intended_class": "yes|no|unknown", "register": "..."}
La classe "intended_class" est seulement une indication de génération — le vrai label
sera produit par label_dataset.py (teacher Haiku).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 50
MAX_TOKENS = 4096

CLASSES = ("yes", "no", "unknown")
REGISTERS = (
    "familier",
    "soutenu",
    "argotique",
    "indirect",
    "sarcastique",
    "interjection",
    "avec_fautes_de_frappe",
    "neutre",
)

SYSTEM_PROMPT = """Tu génères des phrases courtes (1 à 8 mots) pour entraîner un classifieur yes/no/unknown.

Règles absolues :
- Chaque phrase fait entre 1 et 8 mots.
- Pas de ponctuation finale redondante sauf si naturelle (« ! », « ? »).
- Diversité maximale : registre, ton, syntaxe, niveau de langue.
- Pas de doublons ni de quasi-doublons entre les 50 phrases d'une même réponse.
- Output JSON strict : un tableau de 50 strings, rien d'autre.

Classes :
- "yes" : la phrase exprime un accord, une confirmation, un oui (même implicite/ironique affirmatif).
- "no" : la phrase exprime un refus, un désaccord, un non (même implicite).
- "unknown" : phrase neutre/ambiguë/hors-sujet qui N'EST PAS une réponse yes/no claire (météo, small talk, question, hésitation floue, fait divers).
"""

USER_TEMPLATE = """Génère 50 phrases en {lang_name} de classe "{klass}", registre "{register}".

Exigences :
- Exactement 50 phrases uniques.
- Toutes en {lang_name}.
- Toutes cohérentes avec la classe "{klass}" et le registre "{register}".
- 1 à 8 mots par phrase.

Réponds UNIQUEMENT avec un tableau JSON de 50 strings. Pas de markdown, pas de texte avant/après."""

LANG_NAMES = {"en": "anglais", "fr": "français"}


@dataclass
class BatchSpec:
    lang: str
    klass: str
    register: str


def plan_batches(target_per_lang: int) -> list[BatchSpec]:
    """Crée la liste des batchs à générer, équilibrée par (lang, class, register)."""
    per_class = target_per_lang // len(CLASSES)
    batches_per_class = per_class // BATCH_SIZE
    specs: list[BatchSpec] = []
    for lang in ("en", "fr"):
        for klass in CLASSES:
            for i in range(batches_per_class):
                register = REGISTERS[i % len(REGISTERS)]
                specs.append(BatchSpec(lang=lang, klass=klass, register=register))
    random.shuffle(specs)
    return specs


async def generate_batch(
    client: AsyncAnthropic,
    spec: BatchSpec,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    user_msg = USER_TEMPLATE.format(
        lang_name=LANG_NAMES[spec.lang], klass=spec.klass, register=spec.register
    )
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_msg}],
                )
                text = resp.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                phrases = json.loads(text)
                if not isinstance(phrases, list):
                    raise ValueError("not a list")
                return [
                    {
                        "phrase": str(p).strip(),
                        "lang": spec.lang,
                        "intended_class": spec.klass,
                        "register": spec.register,
                    }
                    for p in phrases
                    if isinstance(p, str) and p.strip()
                ]
            except Exception as e:
                if attempt == 2:
                    print(f"  [fail] {spec} : {e}", file=sys.stderr)
                    return []
                await asyncio.sleep(2 ** attempt)
    return []


async def main(target_per_lang: int, concurrency: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {lang: out_dir / f"{lang}.jsonl" for lang in ("en", "fr")}

    seen: dict[str, set[str]] = {"en": set(), "fr": set()}
    for lang, path in paths.items():
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        seen[lang].add(json.loads(line)["phrase"].lower())
                    except Exception:
                        pass
            print(f"[resume] {lang}: {len(seen[lang])} phrases existantes")

    specs = plan_batches(target_per_lang)
    print(f"[plan] {len(specs)} batchs × {BATCH_SIZE} = {len(specs) * BATCH_SIZE} phrases cibles")

    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)
    files = {lang: paths[lang].open("a", encoding="utf-8") for lang in ("en", "fr")}

    written = {"en": len(seen["en"]), "fr": len(seen["fr"])}
    try:
        tasks = [generate_batch(client, s, semaphore) for s in specs]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            rows = await coro
            for row in rows:
                key = row["phrase"].lower()
                if key in seen[row["lang"]]:
                    continue
                seen[row["lang"]].add(key)
                files[row["lang"]].write(json.dumps(row, ensure_ascii=False) + "\n")
                written[row["lang"]] += 1
            if i % 10 == 0 or i == len(tasks):
                files["en"].flush()
                files["fr"].flush()
                print(f"[progress] {i}/{len(tasks)} batchs | en={written['en']} fr={written['fr']}")
    finally:
        for f in files.values():
            f.close()

    print(f"[done] en={written['en']} fr={written['fr']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-per-lang", type=int, default=5000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    asyncio.run(main(args.target_per_lang, args.concurrency, args.out_dir))
