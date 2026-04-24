# ForSureLLM

> 🇫🇷 **Français** · [🇬🇧 English](README_en.md)

Classifier `yes` / `no` / `unknown` distillé EN + FR. Inférence locale sans appel API, ONNX int8 quantizé, ~2.5 ms CPU.

Destiné à reconnaître l'intention de consentement dans une réponse courte d'utilisateur (bot, CLI, IVR, automation). L'application hôte maintient l'état de l'action en cours ; le classifier dit seulement si l'utilisateur **adhère**, **refuse**, ou **hésite**.

## Métriques

| | |
|---|---|
| Accuracy test | **91.4 %** |
| ECE (calibration) | **0.007** |
| Latence CPU (mean) | **2.5 ms** |
| Latence CPU (p95) | 4.3 ms |
| Taille modèle | 113 MB |
| Training (GPU) | 2m24 |
| Inference throughput | ~4000 samp/s (GPU), ~1000 samp/s (CPU) |

Eval adversarial sur 63 phrases-pièges (sarcasme, abbréviations, accents manquants, idiomes régionaux) : **88.9 %**.

## Usage

```python
from forsurellm import classify

classify("carrément")                 # ("yes", 0.977)
classify("laisse tomber")             # ("no", 0.980)
classify("il pleut dehors")           # ("unknown", 0.970)
classify("absolutely")                # ("yes", 0.975)
classify("nah fam")                   # ("no", 0.930)
classify("no cap")                    # ("yes", 0.960)
classify("tiguidou")                  # ("yes", 0.979)

# Sarcasme détecté via ponctuation + pattern
classify("oui bien sûr...")           # ("no", 0.904)
classify("yeah right")                # ("no", 0.88)

# Seuil de confiance : fallback unknown si max < threshold
classify("ok", threshold=0.95)        # ("yes" si >0.95, sinon "unknown")
```

## Installation

```bash
pip install -e .
```

Runtime : `onnxruntime`, `tokenizers`, `numpy`. Le tokenizer et la config sont bundlés dans `forsurellm/models/`. Le fichier `forsurellm-int8.onnx` (113 MB) n'est **pas** commité (limite GitHub 100 MB) — il faut soit le récupérer séparément, soit le re-générer via la section "Reproduire l'entraînement" ci-dessous.

Le LLM est disponible sur Huggingface : https://huggingface.co/jcfossati/ForSureLLM

## Interface web de test

```bash
pip install -e ".[web]"    # fastapi + uvicorn
python tools/server.py
```

Puis `http://localhost:8000` — input live, threshold slider, barres de distribution, tokens visualisés, 17 presets cliquables.

## Architecture du pipeline

```
generate_dataset.py      (Sonnet 4.6 : 8700 phrases équilibrées EN+FR × 3 classes × 8 registres)
        │
augment_idioms.py        (+1500 phrases idiomatiques en largeur : Québec, AAVE, british, aussie, sarcasmes)
augment_idioms_deep.py   (+1850 variantes des 102 idiomes clés, soft labels hardcodés Sonnet)
        │
label_dataset.py         (Haiku 4.5 : soft labels {yes, no, unknown} + re-label Sonnet sur cas incertains)
        │
clean_dataset.py         (retrait des unknowns pure noise : faits neutres/soutenus très confiants)
        │
train.py                 (MiniLM-L12 multilingue + head 3 classes, loss KL-div, 8 epochs)
        │
calibrate.py             (temperature scaling post-hoc via LBFGS sur val set)
        │
export.py                (ONNX + int8 dynamique + benchmark CPU + T dans config.json)
        │
forsurellm/classifier.py      (runtime : onnxruntime + tokenizers + soft prob calibrée)
```

## Reproduire l'entraînement

```bash
pip install -e ".[train]"
cp .env.example .env   # remplir la clé du provider choisi

python training/generate.py --target-per-lang 5000
python training/augment_idioms.py
python training/augment_idioms_deep.py
python training/label.py
python training/clean.py
python training/train.py --epochs 8
python training/calibrate.py
python training/export.py
pytest tests/
python tools/eval.py
```

Coût API observé (setup Anthropic par défaut) : ~$15 (Sonnet génération+augmentation + Haiku labeling).
Training : 2m24 sur RTX Blackwell (CUDA 12.8), 5m11 sur CPU moderne.

## Configuration multi-provider LLM

Le pipeline utilise [LiteLLM](https://github.com/BerriAI/litellm) pour supporter n'importe quel provider (Anthropic, OpenAI, Google Gemini, Mistral, Groq, Ollama local, OpenRouter, Azure, Bedrock…). Deux fichiers contrôlent le choix :

**`llm_config.yaml`** — modèles et paramètres (commité, versionné) :

```yaml
generation:                                # phrase génération (créatif)
  model: anthropic/claude-sonnet-4-6
  max_tokens: 4096

labeling:                                  # soft-label JSON (économique)
  model: anthropic/claude-haiku-4-5-20251001
  max_tokens: 128

labeling_fallback:                         # re-label si primary hésite (<0.6)
  model: anthropic/claude-sonnet-4-6
  max_tokens: 128
```

**`.env`** — clés API (gitignored, seulement les providers utilisés) :

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

### Exemples de configs alternatives

**Tout OpenAI** :
```yaml
generation:
  model: gpt-4o
labeling:
  model: gpt-4o-mini
labeling_fallback:
  model: gpt-4o
```

**Gemini** :
```yaml
generation:
  model: gemini/gemini-2.5-pro
labeling:
  model: gemini/gemini-2.5-flash
labeling_fallback:
  model: gemini/gemini-2.5-pro
```

**Local (gratuit, Ollama)** :
```yaml
generation:
  model: ollama/qwen2.5:32b
labeling:
  model: ollama/qwen2.5:7b
labeling_fallback:
  model: ollama/qwen2.5:32b
```

**Hybride** (Sonnet pour la créativité, Haiku labeling via Groq pour la vitesse) :
```yaml
generation:
  model: anthropic/claude-sonnet-4-6
labeling:
  model: groq/llama-3.3-70b-versatile
labeling_fallback:
  model: anthropic/claude-sonnet-4-6
```

Le caching de prompt Anthropic est activé automatiquement quand le provider est Anthropic, sinon ignoré silencieusement.

## Tests & eval

```bash
pytest tests/                      # 37 tests unitaires (API, hard cases EN/FR, threshold, perf)
python tools/eval.py             # eval adversarial sur 63 phrases curées
python tools/repl.py             # REPL interactif avec visualisation
```

## Calibration & seuil

Le modèle est **calibré** par temperature scaling (T=0.689) : la confidence retournée = probabilité réelle que la classe soit correcte (ECE=0.007, soit <1% d'écart moyen).

| Threshold | Régime | Usage |
|---|---|---|
| `0.0` | Laxiste | Toujours classifier, usage conversationnel |
| `0.7` | Équilibré | ~80% des cas, précision 95%+ |
| `0.9` | Strict | Accepte les canoniques, reste = unknown |
| `0.95` | Très strict | Auto-accept des cas clair-cut (>95% fiable) |

Pattern typique en production :

```python
label, conf = classify(user_reply)
if conf > 0.95:
    execute(pending_action)           # auto-accept
elif conf > 0.7:
    confirm(pending_action)           # demander confirmation
else:
    return "could not understand"     # escalader
```

## Ce qui fonctionne bien

- Yes/no canoniques EN+FR : 100% (y compris *"ouep"*, *"nan"*, *"nope"*, *"nah"*)
- Registres variés : familier, soutenu, argot, interjection — tous >95%
- Robustesse aux typos et accents manquants (*"carrement"*, *"peut-etre"*, *"noooon"*)
- Idiomes régionaux : Québec (*"tiguidou"*), AAVE (*"no cap"*, *"fax no printer"*), british (*"bollocks"*), aussie (*"too right"*)
- Hedges : *"bof"*, *"peut-être"*, *"I dunno"* — 100%
- Certains sarcasmes via pattern + ponctuation : *"oui bien sûr..."* → no, *"yeah right"* → no, *"oh great"* → no
- Détection pragmatique des "..." comme marqueur de sarcasme/résignation en FR

## Limites

- **Sarcasmes hors patterns fréquents** — un sarcasme très subtil sans marqueur ponctuel peut être loupé.
- **Abbreviations chat** (*"lol"*, *"np"*, *"grv"*) — sous-représentées, résultats instables.
- **OOD** (entrées totalement hors distribution, ex: texte aléatoire *"tape une p"*) — softmax force une classe. Utiliser un threshold élevé pour fallback.
- **Langues autres que EN/FR** — le tokenizer couvre mais le modèle n'a pas été entraîné dessus (hors scope).

## Roadmap

- [ ] **Vocab pruning** — retirer les ~245k tokens inutilisés (CJK, arabe, thaï, etc.) pour passer de 113 MB à ~25 MB. Tradeoff : perte de robustesse aux langues hors-FR/EN.
- [ ] **Coverage abbréviations chat** — ajout ciblé *"lol, np, grv, mdr, omg, wtf..."* pour améliorer la catégorie `slang_abbrev`.

## Structure

```
ForSureLLM/
├── forsurellm/                    # package runtime distribuable
│   ├── __init__.py
│   ├── classifier.py              # inference (onnxruntime + tokenizers)
│   └── models/
│       ├── forsurellm-int8.onnx   # (gitignored, 113 MB)
│       ├── tokenizer.json
│       └── config.json            # classes, max_length, temperature
│
├── training/                      # pipeline de distillation
│   ├── llm_client.py              # wrapper LiteLLM multi-provider
│   ├── generate.py                # génération large EN+FR
│   ├── augment_idioms.py          # idiomes en largeur (60 seeds)
│   ├── augment_idioms_deep.py     # idiomes en profondeur (102 × 20)
│   ├── label.py                   # soft labels
│   ├── clean.py                   # drop unknowns pure-noise
│   ├── train.py                   # distillation KL-div
│   ├── calibrate.py               # temperature scaling
│   └── export.py                  # ONNX int8 + benchmark
│
├── tools/                         # utilitaires standalone
│   ├── repl.py                    # REPL interactif terminal
│   ├── eval.py                    # eval adversarial curée
│   └── server.py                  # FastAPI + interface web
│
├── web/
│   └── index.html                 # interface de test
│
├── data/                          # datasets (contenu gitignored)
│   ├── raw/                       # phrases générées
│   ├── labeled/                   # phrases labellisées
│   └── splits/                    # train/val/test stratifiés
│
├── evals/
│   ├── adversarial.jsonl          # 63 phrases-pièges curées
│   └── last_report.json           # dernier rapport d'éval
│
├── tests/
│   └── test_classifier.py         # 37 tests unitaires
│
├── docs/
│   └── brief.md                   # brief projet
│
├── llm_config.yaml                # config modèles LLM (éditable)
├── .env.example                   # template clés API
└── checkpoints/                   # artefacts training (gitignored)
```
