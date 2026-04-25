"""One-shot helper : demande à Sonnet d'énumérer les tokens symboliques
yes/no/unknown (emojis, signes, ponctuation). Outil ponctuel pour étendre
la liste manuelle d'emojis/symboles dans `forsurellm/classifier.py`.

NB : pour les patterns numériques (fractions n/d, pourcentages, ±N) le
classifier utilise des règles regex (cf. `_classify_symbolic`), pas une
liste — inutile de demander au LLM d'énumérer toutes les variantes.

Sortie : data/symbolic_raw.json (à curer manuellement, gitignored).
"""
from __future__ import annotations

import json
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv(override=True)
litellm.suppress_debug_info = True

PROMPT = r"""Construis 3 tables exhaustives de tokens symboliques courts (sans lettres) utilisés comme réponses oui/non/incertain dans chats, forums, Slack, GitHub, formulaires.

Règles strictes :
- Tokens SANS LETTRE uniquement (chiffres, ponctuation, emojis, fractions, pourcentages, signes math, scores)
- Chaque entrée = UNE chaîne exacte (pas de regex), normalisée trim+lower
- Variantes plausibles incluses (avec/sans signe, espaces, ponctuation finale)
- Ambigus → unknown, pas yes/no
- Format JSON strict : {"yes": [...], "no": [...], "unknown": [...]}

Couvrir : votes (+1, -1), pourcentages (0%, 25%, 50%, 75%, 100% et variantes), fractions courantes (n/5, n/10, n/20, n/100), notes scolaires FR (20/20, 18/20, 0/20, 10/20), emojis (👍 👎 ✅ ❌ 🆗 🚫 ⛔ 🤷 💯), signes math (=, ≠, ≈), shrug ¯\_(ツ)_/¯, doubles (++, --, ?!?).

Pour les fractions : n/d est yes si n/d ≥ 0.7, no si ≤ 0.3, unknown sinon. Énumère explicitement les variantes courantes.

Réponds UNIQUEMENT avec le JSON, sans bloc markdown."""


def main() -> None:
    resp = litellm.completion(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=8000,
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    Path("data/symbolic_raw_text.txt").write_text(text, encoding="utf-8")
    data = json.loads(text.strip())

    out = Path("data/symbolic_raw.json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"yes ({len(data['yes'])}):", data["yes"])
    print(f"no  ({len(data['no'])}):", data["no"])
    print(f"unk ({len(data['unknown'])}):", data["unknown"])
    print(f"\nsaved -> {out}")
    print(f"tokens : in={resp.usage.prompt_tokens} out={resp.usage.completion_tokens}")


if __name__ == "__main__":
    main()
