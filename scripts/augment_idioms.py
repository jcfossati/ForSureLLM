"""Augmente le dataset avec des expressions idiomatiques exhaustives EN+FR.

Usage:
    python scripts/augment_idioms.py --concurrency 8

Stratégie :
- 60+ "seeds" thématiques (catégorie × langue × registre × région).
- Sonnet 4.6 génère 30 expressions par seed, uniques, avec la classe cible.
- Chaque expression inclut 3-5 variantes (accents, fautes, contractions, ponct).
- Output : data/raw/idioms.jsonl (à merger ensuite avec data/raw/{en,fr}.jsonl).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from forsurellm.llm import LLMClient

PER_SEED = 30

SYSTEM_PROMPT = """Tu es expert en linguistique des expressions idiomatiques françaises et anglaises.

Tâche : générer EXHAUSTIVEMENT des expressions idiomatiques courtes (1 à 10 mots) qui
signifient une classe donnée (yes / no / unknown) dans un registre donné.

Règles absolues :
- Phrases courtes, naturelles, AUTHENTIQUEMENT utilisées par des locuteurs natifs.
- Pas d'expressions inventées. Uniquement des vrais idiomes ou tournures attestées.
- Inclure variantes orthographiques (sans accents, fautes courantes, contractions, ponctuation).
- Inclure régionalismes quand pertinent (québécois, belge, british, US, AAVE, australien...).
- Inclure formes sarcastiques quand le seed l'indique (yes-forme qui veut dire no).
- Output JSON strict : tableau de N strings (N = quantité demandée), rien d'autre.
- Zéro doublon (pas de quasi-duplicats : "grave" et "grave !" comptent comme 2 entrées OK).
"""

USER_TEMPLATE = """Génère {n} expressions uniques en {lang_name}, classe "{klass}", catégorie "{seed_title}".

Description : {description}
Registre : {register}

Exigences :
- EXACTEMENT {n} entrées, toutes uniques.
- 1 à 10 mots chacune.
- Doit être de vraies expressions idiomatiques connues, PAS des phrases génériques.
- Inclure variantes (accents, fautes, contractions).
{extra}

Réponds UNIQUEMENT avec un tableau JSON de {n} strings. Pas de markdown, pas de préambule."""


# -------------------------------------------------------------------------
# Seeds : chaque seed = 1 batch de 30 expressions.
# ~60 seeds => ~1800 expressions brutes, dédup ensuite.
# -------------------------------------------------------------------------

SEEDS = [
    # ======================== FR / YES ========================
    ("fr", "yes", "familier_oui_argot", "Affirmations familières/argot",
     "\"carrément\", \"grave\", \"à donf\", \"de ouf\", \"à fond\", \"bah ouais\" et toutes leurs variantes.",
     "familier argot"),
    ("fr", "yes", "idiomes_consentement_FR", "Idiomes de consentement",
     "\"je veux, mon neveu\", \"et comment\", \"tu m'étonnes\", \"et alors\", \"évidemment\", \"sans problème\".",
     "familier"),
    ("fr", "yes", "marques_affirmation_FR", "Marqueurs d'affirmation",
     "\"banco\", \"tope-là\", \"deal\", \"ça marche\", \"ça roule\", \"nickel\", \"parfait\", \"ok\", \"au top\".",
     "familier"),
    ("fr", "yes", "soutenu_affirmatif", "Affirmatifs soutenus/formels",
     "\"assurément\", \"indubitablement\", \"certainement\", \"formellement\", \"absolument\", \"sans nul doute\".",
     "soutenu"),
    ("fr", "yes", "quebec_oui", "Affirmatifs québécois",
     "\"pantoute ça marche\", \"ben voyons\", \"mets-en\", \"c'est sûr\", \"certain\", \"pour vrai\".",
     "familier régional québécois"),
    ("fr", "yes", "oui_interjections", "Interjections affirmatives",
     "\"ah oui !\", \"eh oui\", \"bah oui\", \"mais oui\", \"ouais !\", \"yes !\", \"youpi\", \"cool\".",
     "interjection"),
    ("fr", "yes", "avec_fautes_oui", "Oui avec fautes de frappe/contractions",
     "\"ouai\", \"ui\", \"ouiii\", \"ouaip\", \"oué\", \"wi\", \"ouep\", \"yep\".",
     "contraction fautes"),
    ("fr", "yes", "oui_engagement", "Engagements/promesses affirmatives",
     "\"compte sur moi\", \"ça marche\", \"t'inquiète\", \"tkt\", \"sans problème\", \"pas de souci\".",
     "familier"),

    # ======================== FR / NO ========================
    ("fr", "no", "familier_non_argot", "Négations familières/argot",
     "\"que dalle\", \"walou\", \"nada\", \"niet\", \"tintin\", \"peau de balle\", \"que tchi\", \"zéro\".",
     "argot"),
    ("fr", "no", "idiomes_refus_FR", "Idiomes de refus",
     "\"laisse tomber\", \"laisse béton\", \"compte là-dessus\", \"dans tes rêves\", \"va te faire voir\".",
     "familier"),
    ("fr", "no", "non_sarcastique_direct", "Refus sarcastiques directs",
     "\"mon œil\", \"tu parles\", \"et mon cul c'est du poulet\", \"c'est ça oui\", \"cause toujours\".",
     "sarcastique familier"),
    ("fr", "no", "non_formel_soutenu", "Refus soutenus/formels",
     "\"hors de question\", \"je refuse catégoriquement\", \"en aucun cas\", \"nullement\", \"aucunement\".",
     "soutenu"),
    ("fr", "no", "non_emphase", "Négations avec emphase",
     "\"jamais de la vie\", \"pour rien au monde\", \"à aucun prix\", \"pas question\", \"c'est mort\".",
     "familier"),
    ("fr", "no", "non_interjections", "Interjections négatives",
     "\"ah non !\", \"oh non\", \"mais non\", \"bah non\", \"non mais\", \"pfff non\", \"hein non\".",
     "interjection"),
    ("fr", "no", "avec_fautes_non", "Non avec fautes/contractions",
     "\"nn\", \"nan\", \"non non\", \"noooon\", \"nop\", \"nope\", \"naan\", \"naaah\".",
     "contraction fautes"),
    ("fr", "no", "quebec_non", "Négations québécoises",
     "\"voyons donc\", \"pantoute\", \"ben non là\", \"jamais de la vie\", \"c'est ben pour dire\".",
     "familier régional québécois"),
    ("fr", "no", "non_decline_poli", "Refus polis",
     "\"non merci\", \"sans façon\", \"je préfère pas\", \"c'est gentil mais non\", \"très peu pour moi\".",
     "poli"),

    # ======================== FR / SARCASM (yes-forme = no) ========================
    ("fr", "no", "fr_sarcasme_oui_ironique", "Sarcasmes (oui ironique = non)",
     "Expressions qui ressemblent à yes mais signifient NO par sarcasme. \"oui bien sûr\", \"évidemment\", \"tu parles\", \"oh super\", \"formidable\".",
     "sarcastique"),
    ("fr", "no", "fr_ironie_affirmative", "Ironie affirmative piégeuse",
     "\"mais oui mais oui\", \"c'est ça bien sûr\", \"ben voyons\", \"ah oui vraiment\", \"oh génial\", \"super idée\".",
     "sarcastique"),

    # ======================== FR / UNKNOWN ========================
    ("fr", "unknown", "hedges_FR", "Hésitations/hedges français",
     "\"bof\", \"mouais\", \"peut-être\", \"on verra\", \"à voir\", \"ça dépend\", \"je sais pas\", \"j'hésite\".",
     "hesitation"),
    ("fr", "unknown", "unknown_reflexion", "Expressions de réflexion",
     "\"faut voir\", \"faudrait voir\", \"je suis partagé\", \"difficile à dire\", \"ni oui ni non\", \"p't'être\".",
     "hesitation"),
    ("fr", "unknown", "unknown_ignorance", "Expressions d'ignorance",
     "\"j'en sais rien\", \"aucune idée\", \"chais pas\", \"mystère\", \"va savoir\", \"dieu seul sait\".",
     "ignorance"),
    ("fr", "unknown", "unknown_filler", "Hésitations/fillers",
     "\"euh...\", \"ben...\", \"hmm\", \"heu\", \"ouf\", \"bof bof\", \"humm\".",
     "filler"),

    # ======================== EN / YES ========================
    ("en", "yes", "en_slang_affirmative", "Slang affirmatives",
     "\"heck yes\", \"hell yeah\", \"you bet\", \"fo sho\", \"for sure\", \"damn straight\", \"darn tootin'\".",
     "argot"),
    ("en", "yes", "en_idiom_yes", "Idioms of agreement",
     "\"that works\", \"sounds good\", \"works for me\", \"I'm in\", \"count me in\", \"let's do it\".",
     "familier"),
    ("en", "yes", "en_formal_affirm", "Formal affirmatives",
     "\"certainly\", \"indeed\", \"absolutely\", \"undoubtedly\", \"without a doubt\", \"most definitely\".",
     "soutenu"),
    ("en", "yes", "en_british_yes", "British affirmatives",
     "\"right-o\", \"proper\", \"aye\", \"sound\", \"spot on\", \"brilliant\", \"cheers mate\", \"alright\".",
     "familier régional british"),
    ("en", "yes", "en_aae_yes", "AAVE/informal affirmatives",
     "\"facts\", \"no cap\", \"bet\", \"fr fr\", \"periodt\", \"say less\", \"word\", \"straight up\".",
     "argot AAVE"),
    ("en", "yes", "en_austr_yes", "Aussie affirmatives",
     "\"too right\", \"yeah nah yeah\", \"bloody oath\", \"no worries\", \"spot on mate\", \"fair dinkum\".",
     "familier régional australien"),
    ("en", "yes", "en_yes_typos", "Yes with typos/contractions",
     "\"yse\", \"yeaaaa\", \"yuhh\", \"ye\", \"yhh\", \"yass\", \"yasssss\", \"yepp\".",
     "contraction fautes"),
    ("en", "yes", "en_yes_interject", "Yes interjections",
     "\"oh yeah\", \"oh yes\", \"yes!\", \"heck yes\", \"woohoo\", \"yay\", \"uh-huh\", \"mm-hmm\".",
     "interjection"),

    # ======================== EN / NO ========================
    ("en", "no", "en_slang_no", "Slang negatives",
     "\"no way\", \"nah\", \"nope\", \"hell no\", \"heck no\", \"hard pass\", \"not a chance\".",
     "argot"),
    ("en", "no", "en_idiom_no", "Idiom negatives",
     "\"when pigs fly\", \"over my dead body\", \"in your dreams\", \"not in a million years\", \"fat chance\".",
     "familier"),
    ("en", "no", "en_formal_no", "Formal refusals",
     "\"absolutely not\", \"I must decline\", \"I beg to differ\", \"certainly not\", \"by no means\".",
     "soutenu"),
    ("en", "no", "en_british_no", "British negatives",
     "\"nah mate\", \"bollocks\", \"not likely\", \"you're having a laugh\", \"do me a favour\".",
     "familier régional british"),
    ("en", "no", "en_aae_no", "AAVE negatives",
     "\"nah fam\", \"ain't happening\", \"ion think so\", \"cap\", \"I don't believe you\", \"you trippin'\".",
     "argot AAVE"),
    ("en", "no", "en_no_typos", "No with typos/contractions",
     "\"nooo\", \"nuuu\", \"naah\", \"naw\", \"nuh\", \"nuh-uh\", \"nah man\", \"noope\".",
     "contraction fautes"),
    ("en", "no", "en_no_interject", "No interjections",
     "\"oh no\", \"hell no\", \"nope nope nope\", \"aww no\", \"heck no\", \"god no\", \"please no\".",
     "interjection"),
    ("en", "no", "en_polite_decline", "Polite declines",
     "\"no thanks\", \"I'll pass\", \"not for me\", \"I'm good\", \"I'd rather not\", \"maybe not\".",
     "poli"),

    # ======================== EN / SARCASM ========================
    ("en", "no", "en_sarcasm_yes_meaning_no", "Yes-form sarcasms meaning NO",
     "\"yeah right\", \"sure, Jan\", \"oh yeah totally\", \"oh great\", \"oh fantastic\", \"as if\", \"tell me more\".",
     "sarcastique"),
    ("en", "no", "en_sarcasm_compliment_inverse", "Sarcastic faux-positives",
     "\"oh brilliant\", \"oh super\", \"oh wonderful\", \"how original\", \"how revolutionary\", \"how clever\".",
     "sarcastique"),

    # ======================== EN / UNKNOWN ========================
    ("en", "unknown", "en_hedges", "English hedges",
     "\"meh\", \"eh\", \"I guess\", \"sort of\", \"kinda\", \"somewhat\", \"I dunno\", \"I'm not sure\".",
     "hesitation"),
    ("en", "unknown", "en_reflection", "Reflection expressions",
     "\"let me think\", \"I need to think\", \"not sure yet\", \"hard to say\", \"it depends\", \"maybe\".",
     "hesitation"),
    ("en", "unknown", "en_ignorance", "Expressions of ignorance",
     "\"no idea\", \"beats me\", \"who knows\", \"god knows\", \"your guess is as good as mine\", \"idk\".",
     "ignorance"),
    ("en", "unknown", "en_filler", "Fillers",
     "\"uh...\", \"um...\", \"erm\", \"hmm\", \"well...\", \"so...\", \"ah...\", \"you know\".",
     "filler"),

    # ======================== Both / MIXED REGISTER ========================
    ("fr", "yes", "oui_emphase_fort", "Oui avec emphase forte",
     "Oui répétés, avec exclamations multiples, emphase. \"oui oui oui\", \"YEEES\", \"ouiiii !!\", \"100%\".",
     "emphase"),
    ("fr", "no", "non_emphase_fort", "Non avec emphase forte",
     "Non répétés, exclamations multiples. \"non non non\", \"NOOOOON\", \"nooon !!\", \"certainement pas\".",
     "emphase"),
    ("en", "yes", "en_yes_emphasis", "English yes with strong emphasis",
     "\"yes yes yes\", \"YESSSS\", \"absolutely 100%\", \"totally yes\", \"a thousand times yes\".",
     "emphase"),
    ("en", "no", "en_no_emphasis", "English no with strong emphasis",
     "\"NO NO NO\", \"absolutely not\", \"hard no\", \"a big fat no\", \"never ever\".",
     "emphase"),

    # ======================== One-word/compact ========================
    ("fr", "yes", "fr_one_word_yes", "Oui en 1-2 mots compacts",
     "\"ok\", \"yes\", \"gogo\", \"ça passe\", \"chaud\", \"vendu\", \"adjugé\", \"reçu\", \"roger\".",
     "compact"),
    ("fr", "no", "fr_one_word_no", "Non en 1-2 mots compacts",
     "\"nope\", \"jamais\", \"refusé\", \"négatif\", \"passe\", \"stop\", \"halte\", \"basta\".",
     "compact"),
    ("en", "yes", "en_one_word_yes", "English one-word yes",
     "\"okay\", \"k\", \"kk\", \"sure\", \"roger\", \"copy\", \"aye\", \"affirmative\", \"indeed\".",
     "compact"),
    ("en", "no", "en_one_word_no", "English one-word no",
     "\"negative\", \"never\", \"pass\", \"decline\", \"no-no\", \"denied\", \"rejected\".",
     "compact"),

    # ======================== Contextual indirect ========================
    ("fr", "yes", "fr_indirect_yes", "Accord indirect (underhanded)",
     "\"ça me va\", \"j'approuve\", \"validé\", \"bon plan\", \"pourquoi pas\", \"good\", \"ça me tente\".",
     "indirect"),
    ("fr", "no", "fr_indirect_no", "Refus indirect",
     "\"bof non\", \"je sens pas\", \"pas chaud\", \"pas trop\", \"je passe\", \"sans moi\", \"comptez sans moi\".",
     "indirect"),
    ("en", "yes", "en_indirect_yes", "Indirect yes",
     "\"why not\", \"sounds good\", \"I'm game\", \"I'm down\", \"looks good to me\", \"works for me\".",
     "indirect"),
    ("en", "no", "en_indirect_no", "Indirect no",
     "\"I'll pass\", \"not feeling it\", \"count me out\", \"leave me out\", \"I'm out\", \"no can do\".",
     "indirect"),

    # ======================== Approval/validation specific ========================
    ("fr", "yes", "fr_validation_pro", "Validation en contexte pro",
     "\"validé\", \"approuvé\", \"go\", \"feu vert\", \"on lance\", \"on valide\", \"banco\", \"ça part\".",
     "professionnel"),
    ("en", "yes", "en_approval_pro", "Professional approval",
     "\"approved\", \"greenlit\", \"go ahead\", \"proceed\", \"cleared\", \"signed off\", \"confirmed\".",
     "professionnel"),
    ("fr", "no", "fr_rejet_pro", "Rejet en contexte pro",
     "\"refusé\", \"rejeté\", \"bloqué\", \"stop\", \"annulé\", \"abort\", \"on arrête\", \"on annule\".",
     "professionnel"),
    ("en", "no", "en_rejection_pro", "Professional rejection",
     "\"denied\", \"rejected\", \"blocked\", \"abort\", \"cancelled\", \"dropped\", \"called off\".",
     "professionnel"),
]


@dataclass
class Seed:
    lang: str
    klass: str
    seed_id: str
    title: str
    description: str
    register: str


async def generate_batch(
    client: LLMClient,
    seed: Seed,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    extra = ""
    if "sarcasme" in seed.seed_id or "sarcasm" in seed.seed_id:
        extra = "- IMPORTANT : ces expressions ont une FORME affirmative/positive mais signifient NO par sarcasme. C'est la classe no."

    user_msg = USER_TEMPLATE.format(
        n=PER_SEED,
        lang_name="français" if seed.lang == "fr" else "anglais",
        klass=seed.klass,
        seed_title=seed.title,
        description=seed.description,
        register=seed.register,
        extra=extra,
    )

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
                        "intended_class": seed.klass,
                        "register": seed.register,
                        "seed": seed.seed_id,
                    }
                    for p in phrases
                    if isinstance(p, str) and p.strip()
                ]
            except Exception as e:
                if attempt == 2:
                    print(f"  [fail] {seed.seed_id} : {e}", file=sys.stderr)
                    return []
                await asyncio.sleep(2 ** attempt)
    return []


async def main(concurrency: int, out_path: Path) -> None:
    seeds = [Seed(*s) for s in SEEDS]
    print(f"[plan] {len(seeds)} seeds × {PER_SEED} = {len(seeds) * PER_SEED} expressions cibles")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing.add(json.loads(line)["phrase"].lower())
                except Exception:
                    pass
        print(f"[resume] {len(existing)} phrases déjà présentes")

    client = LLMClient("generation")
    print(f"[model] generation = {client.model}")
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [generate_batch(client, s, semaphore) for s in seeds]
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
            if i % 5 == 0 or i == len(tasks):
                f.flush()
                print(f"[progress] {i}/{len(tasks)} seeds | total uniques={written}")

    print(f"[done] {written} expressions uniques -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("data/raw/idioms.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    asyncio.run(main(args.concurrency, args.out))
