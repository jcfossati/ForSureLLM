"""Augmentation profonde : pour ~80 idiomes clés, génère 20 variantes contextuelles.

Contrairement à augment_idioms.py qui explore en largeur, celui-ci renforce
en profondeur chaque idiome clé pour donner au modèle ~20 exemples par expression.

Labels : pré-assignés (pas de labeling Haiku). On fait confiance au fait que
Sonnet génère des variantes cohérentes avec la classe demandée. Soft labels
légèrement lissés (0.92/0.04/0.04) pour éviter la sur-confiance.

Usage:
    python scripts/augment_idioms_deep.py --concurrency 8

Output :
    data/labeled/idioms_deep.jsonl  (directement labellisé, skip du pipeline Haiku)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from forsurellm.llm import LLMClient

PER_IDIOM = 20

# Lissage des soft labels
CONFIDENT = 0.92
DISTRIB_REST = 0.04


SYSTEM_PROMPT = """Tu génères des variantes contextuelles d'une expression idiomatique courte.

Chaque variante DOIT :
- Conserver le sens originel de l'expression (même classe : yes/no/unknown).
- Inclure l'expression telle quelle OU une variante orthographique proche (fautes, contractions, accents).
- Être courte (2-12 mots).
- Être naturelle, authentiquement dite par un locuteur natif.

Types de variantes à produire :
1. L'idiome seul : "que dalle"
2. Avec intensifieur : "que dalle de chez que dalle", "vraiment que dalle"
3. En contexte court : "j'ai eu que dalle", "pour moi que dalle"
4. Avec ponctuation variée : "que dalle !", "que dalle...", "que dalle ?"
5. Avec fautes/contractions : "ke dalle", "qdalle"
6. Avec emphase : "mais que dalle !", "ah que dalle"
7. Avec majuscules : "QUE DALLE"

Output : tableau JSON de strings, rien d'autre. Zéro duplicat exact.
"""

USER_TEMPLATE = """Idiome : "{idiom}"
Langue : {lang_name}
Classe (sens) : "{klass}"
Note : {note}

Génère exactement {n} variantes UNIQUES respectant les règles.

Réponds UNIQUEMENT avec un tableau JSON de {n} strings."""


@dataclass
class IdiomSeed:
    idiom: str
    lang: str
    klass: str
    note: str = ""


# ~80 idiomes clés à renforcer
IDIOMS: list[IdiomSeed] = [
    # ---------- FR / YES ----------
    IdiomSeed("je veux mon neveu", "fr", "yes", "expression désuète familière = absolument"),
    IdiomSeed("carrément", "fr", "yes", "affirmation forte"),
    IdiomSeed("grave", "fr", "yes", "argot = totalement d'accord"),
    IdiomSeed("grave de chez grave", "fr", "yes", "emphase argotique"),
    IdiomSeed("tu m'étonnes", "fr", "yes", "paradoxalement = d'accord/confirmation"),
    IdiomSeed("et comment", "fr", "yes", "confirmation emphatique"),
    IdiomSeed("banco", "fr", "yes", "deal scellé"),
    IdiomSeed("tope-là", "fr", "yes", "pacte conclu"),
    IdiomSeed("évidemment", "fr", "yes", "réponse positive (sauf sarcasme)"),
    IdiomSeed("ça marche", "fr", "yes", "accord familier"),
    IdiomSeed("ça roule", "fr", "yes", "accord"),
    IdiomSeed("tiguidou", "fr", "yes", "québécois = parfait/d'accord"),
    IdiomSeed("mets-en", "fr", "yes", "québécois = absolument"),
    IdiomSeed("pour vrai", "fr", "yes", "québécois = sérieusement oui"),
    IdiomSeed("ben ouais", "fr", "yes", "familier oui évident"),
    IdiomSeed("à donf", "fr", "yes", "argot = à fond oui"),
    IdiomSeed("sans hésiter", "fr", "yes", "affirmatif fort"),
    IdiomSeed("vendu", "fr", "yes", "accord marchand figuré"),
    IdiomSeed("ça me va", "fr", "yes", "accord"),
    IdiomSeed("au taquet", "fr", "yes", "motivé/partant"),

    # ---------- FR / NO ----------
    IdiomSeed("que dalle", "fr", "no", "argot = rien du tout"),
    IdiomSeed("walou", "fr", "no", "argot maghrébin = rien/non"),
    IdiomSeed("nada", "fr", "no", "argot = rien"),
    IdiomSeed("niet", "fr", "no", "argot russe adopté = non ferme"),
    IdiomSeed("mon œil", "fr", "no", "négation sarcastique = je n'y crois pas"),
    IdiomSeed("et mon cul c'est du poulet", "fr", "no", "négation sarcastique vulgaire"),
    IdiomSeed("tintin", "fr", "no", "argot = rien obtenir"),
    IdiomSeed("peau de balle", "fr", "no", "argot = rien"),
    IdiomSeed("que tchi", "fr", "no", "argot = rien"),
    IdiomSeed("compte là-dessus", "fr", "no", "ironique = surtout pas"),
    IdiomSeed("dans tes rêves", "fr", "no", "refus ironique"),
    IdiomSeed("laisse tomber", "fr", "no", "refus/abandon"),
    IdiomSeed("laisse béton", "fr", "no", "verlan de tomber = laisse tomber"),
    IdiomSeed("hors de question", "fr", "no", "refus formel"),
    IdiomSeed("jamais de la vie", "fr", "no", "refus fort"),
    IdiomSeed("pas question", "fr", "no", "refus"),
    IdiomSeed("c'est mort", "fr", "no", "fini/refusé argot"),
    IdiomSeed("ça va pas la tête", "fr", "no", "refus incrédule"),

    # ---------- FR / UNKNOWN ----------
    IdiomSeed("bof", "fr", "unknown", "hésitation faible"),
    IdiomSeed("mouais", "fr", "unknown", "oui hésitant"),
    IdiomSeed("peut-être", "fr", "unknown", "incertain"),
    IdiomSeed("à voir", "fr", "unknown", "indécis"),
    IdiomSeed("faut voir", "fr", "unknown", "indécis"),
    IdiomSeed("ça dépend", "fr", "unknown", "conditionnel"),
    IdiomSeed("j'hésite", "fr", "unknown", "indécision"),
    IdiomSeed("ni oui ni non", "fr", "unknown", "neutre"),
    IdiomSeed("j'en sais rien", "fr", "unknown", "ignorance"),
    IdiomSeed("chais pas", "fr", "unknown", "ignorance familière"),

    # ---------- FR / SARCASM (yes-form = NO) ----------
    IdiomSeed("oui bien sûr", "fr", "no", "ATTENTION sarcastique = non"),
    IdiomSeed("ah oui vraiment", "fr", "no", "sarcasme ironique"),
    IdiomSeed("mais oui mais oui", "fr", "no", "sarcasme"),
    IdiomSeed("tu parles", "fr", "no", "ironie = non"),
    IdiomSeed("c'est ça oui", "fr", "no", "ironie = non"),
    IdiomSeed("ben voyons", "fr", "no", "ironie incrédule"),

    # ---------- EN / YES ----------
    IdiomSeed("no cap", "en", "yes", "AAVE = pas de mensonge = pour de vrai/d'accord"),
    IdiomSeed("no cap fr", "en", "yes", "AAVE = pour de vrai d'accord"),
    IdiomSeed("fax", "en", "yes", "AAVE = facts = je confirme"),
    IdiomSeed("fax no printer", "en", "yes", "AAVE = facts = pure vérité/accord"),
    IdiomSeed("facts", "en", "yes", "AAVE/casual = je confirme"),
    IdiomSeed("bet", "en", "yes", "AAVE = d'accord"),
    IdiomSeed("say less", "en", "yes", "AAVE = compris/d'accord"),
    IdiomSeed("hell yeah", "en", "yes", "affirmatif fort argot"),
    IdiomSeed("heck yes", "en", "yes", "version douce de hell yeah"),
    IdiomSeed("you bet", "en", "yes", "affirmatif"),
    IdiomSeed("darn tootin'", "en", "yes", "argot US = exactement"),
    IdiomSeed("on god", "en", "yes", "AAVE = je jure/d'accord"),
    IdiomSeed("periodt", "en", "yes", "point final = confirmation"),
    IdiomSeed("word", "en", "yes", "AAVE = je confirme"),
    IdiomSeed("straight up", "en", "yes", "AAVE = sans doute"),
    IdiomSeed("for real", "en", "yes", "sérieusement/confirmation"),
    IdiomSeed("fo sho", "en", "yes", "AAVE de 'for sure'"),
    IdiomSeed("indeed", "en", "yes", "soutenu affirmatif"),
    IdiomSeed("too right", "en", "yes", "british/aussie = tout à fait"),
    IdiomSeed("bloody oath", "en", "yes", "aussie = absolument"),

    # ---------- EN / NO ----------
    IdiomSeed("hell no", "en", "no", "refus fort"),
    IdiomSeed("heck no", "en", "no", "version douce"),
    IdiomSeed("nah", "en", "no", "non informel"),
    IdiomSeed("nah fam", "en", "no", "AAVE refus"),
    IdiomSeed("hard pass", "en", "no", "refus ferme"),
    IdiomSeed("not a chance", "en", "no", "refus"),
    IdiomSeed("fat chance", "en", "no", "ironique = aucune chance"),
    IdiomSeed("in your dreams", "en", "no", "refus ironique"),
    IdiomSeed("when pigs fly", "en", "no", "idiome = jamais"),
    IdiomSeed("over my dead body", "en", "no", "refus fort"),
    IdiomSeed("nuh-uh", "en", "no", "enfantin/informel non"),
    IdiomSeed("you wish", "en", "no", "refus ironique"),
    IdiomSeed("bollocks", "en", "no", "british = n'importe quoi/non"),
    IdiomSeed("ion think so", "en", "no", "AAVE = I don't think so"),

    # ---------- EN / UNKNOWN ----------
    IdiomSeed("meh", "en", "unknown", "mitigé"),
    IdiomSeed("eh", "en", "unknown", "hésitant"),
    IdiomSeed("I dunno", "en", "unknown", "ignorance"),
    IdiomSeed("idk", "en", "unknown", "I don't know"),
    IdiomSeed("not sure", "en", "unknown", "incertain"),
    IdiomSeed("it depends", "en", "unknown", "conditionnel"),
    IdiomSeed("kinda", "en", "unknown", "mi-figue mi-raisin"),
    IdiomSeed("sort of", "en", "unknown", "hésitant"),

    # ---------- EN / SARCASM ----------
    IdiomSeed("yeah right", "en", "no", "ATTENTION sarcastique = non"),
    IdiomSeed("sure, Jan", "en", "no", "meme sarcastique = non"),
    IdiomSeed("oh great", "en", "no", "sarcasme"),
    IdiomSeed("as if", "en", "no", "refus incrédule"),
    IdiomSeed("oh totally", "en", "no", "sarcasme"),
    IdiomSeed("tell me more", "en", "no", "sarcasme/refus d'écouter"),
]


async def generate_batch(
    client: LLMClient,
    seed: IdiomSeed,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    user_msg = USER_TEMPLATE.format(
        idiom=seed.idiom,
        lang_name="français" if seed.lang == "fr" else "anglais",
        klass=seed.klass,
        note=seed.note,
        n=PER_IDIOM,
    )

    labels = {"yes": DISTRIB_REST, "no": DISTRIB_REST, "unknown": DISTRIB_REST}
    labels[seed.klass] = CONFIDENT

    async with semaphore:
        for attempt in range(3):
            try:
                text = (await client.complete(system=SYSTEM_PROMPT, user=user_msg)).strip()
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
                        "lang": seed.lang,
                        "labels": labels,
                        "teacher": "hardcoded",
                        "source_idiom": seed.idiom,
                    }
                    for p in phrases
                    if isinstance(p, str) and p.strip()
                ]
            except Exception as e:
                if attempt == 2:
                    print(f"  [fail] {seed.idiom!r} : {e}", file=sys.stderr)
                    return []
                await asyncio.sleep(2 ** attempt)
    return []


async def main(concurrency: int, out_path: Path) -> None:
    print(f"[plan] {len(IDIOMS)} idiomes × {PER_IDIOM} variantes = {len(IDIOMS)*PER_IDIOM} phrases cibles")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing.add(json.loads(line)["phrase"].lower())
                except Exception:
                    pass
        print(f"[resume] {len(existing)} phrases existantes")

    client = LLMClient("generation")
    print(f"[model] generation = {client.model}")
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [generate_batch(client, s, semaphore) for s in IDIOMS]

    written = len(existing)
    with out_path.open("a", encoding="utf-8") as f:
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            rows = await coro
            for row in rows:
                key = row["phrase"].lower()
                if key in existing:
                    continue
                existing.add(key)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            if i % 10 == 0 or i == len(tasks):
                f.flush()
                print(f"[progress] {i}/{len(tasks)} idiomes | total uniques={written}")

    print(f"[done] {written} phrases uniques -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("data/labeled/idioms_deep.jsonl"))
    args = parser.parse_args()
    asyncio.run(main(args.concurrency, args.out))
