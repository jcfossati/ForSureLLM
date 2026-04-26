# ForSureLLM

> 🇫🇷 **Français** · [🇬🇧 English](README_en.md)

Classifier `yes` / `no` / `unknown` distillé EN + FR. Inférence locale sans appel API, ONNX int8 quantizé, ~2.5 ms CPU.

🎯 **Demo en ligne** : [huggingface.co/spaces/jcfossati/ForSureLLM](https://huggingface.co/spaces/jcfossati/ForSureLLM)

Destiné à reconnaître l'intention de consentement dans une réponse courte d'utilisateur (bot, CLI, IVR, automation). L'application hôte maintient l'état de l'action en cours ; le classifier dit seulement si l'utilisateur **adhère**, **refuse**, ou **hésite**.

## Sommaire

- [Métriques](#métriques)
- [Usage](#usage)
  - [Confirmation d'action (bot, IVR, automation)](#confirmation-daction-bot-ivr-automation)
  - [Variante prunée FR+EN (24 MB)](#variante-prunée-fren-24-mb)
- [Installation](#installation)
- [Interface web de test](#interface-web-de-test)
- [API de production (FastAPI + Docker)](#api-de-production-fastapi--docker)
- [Architecture du pipeline](#architecture-du-pipeline)
- [Reproduire l'entraînement](#reproduire-lentraînement)
- [Configuration multi-provider LLM](#configuration-multi-provider-llm)
- [Tests & eval](#tests--eval)
- [Benchmark vs baselines](#benchmark-vs-baselines)
- [Calibration & seuil](#calibration--seuil)
- [Ce qui fonctionne bien](#ce-qui-fonctionne-bien)
- [Limites](#limites)
- [Roadmap](#roadmap)
- [Structure](#structure)

## Métriques

| | |
|---|---|
| Accuracy test | **91.7 %** |
| ECE (calibration) | **0.012** |
| Latence CPU (mean) | **2.6 ms** |
| Latence CPU (p95) | 3.9 ms |
| Taille modèle | 113 MB (full multilingual) · **24 MB (pruné FR+EN)** |
| Training (GPU) | 2m05 |
| Inference throughput | ~4000 samp/s (GPU), ~1000 samp/s (CPU) |

Eval adversarial sur 124 phrases-pièges (sarcasme, abbréviations, accents manquants, idiomes régionaux, code-switching, slang Gen-Z, conditionnel, emoji, négation/affirmation implicite, politesse, résignation, symbolique) : **95.2 %** (118/124).

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

### Confirmation d'action (bot, IVR, automation)

Pour les contextes où la classification déclenche une action côté serveur (annuler une commande, confirmer un paiement, lancer un déploiement), passe `threshold=0.85` ou plus :

```python
classify("yeah right", threshold=0.9)   # ("unknown", …) → re-prompt l'utilisateur
classify("ouais", threshold=0.9)        # ("yes", 0.97) → action déclenchée
```

Pourquoi : certains idiomes culturels (`yeah right`, `tu m'étonnes`, `oh toootally`) sont par défaut sarcastiques en usage général mais peuvent être sincères chez un utilisateur qui ne ponctue pas. Avec un seuil élevé, ces cas borderline tombent en `unknown` → ton app re-prompte au lieu de déclencher la mauvaise action. C'est strictement plus sûr en confirmation d'action ; le coût (un re-prompt occasionnel) est faible comparé à exécuter l'inverse de l'intention de l'utilisateur.

### Variante prunée FR+EN (24 MB)

Le modèle de base couvre 50+ langues via un vocab SentencePiece de 250 002 tokens (≈ 88 MB d'embeddings sur les 113 MB totaux). La variante prunée jette les tokens inutilisés par le corpus FR+EN, gardant 5 424 tokens utiles + filet de sécurité Latin-1.

```bash
# Génération (une fois) :
python tools/prune_vocab.py
python training/export.py --src checkpoints/best_fr-en --work-dir checkpoints/onnx_fr-en --out-suffix _fr-en

# Sélection à l'inférence via flag CLI (server, eval, robustness, audit, bench) :
python tools/server.py --variant _fr-en
python tools/eval.py --variant _fr-en
python tools/robustness.py --variant _fr-en

# Sans le flag : modèle par défaut (113 MB).
# La variable d'environnement FORSURELLM_VARIANT est aussi supportée si besoin.
```

```python
# Côté Python (cas où on n'utilise pas les CLI) :
import os; os.environ["FORSURELLM_VARIANT"] = "_fr-en"
from forsurellm import classify
classify("carrément")   # même résultat, depuis un modèle 5× plus léger
```

| | Original | Pruné FR+EN |
|---|---|---|
| ONNX int8 (disque) | 113 MB | **24 MB** (-79 %) |
| Tokenizer (disque) | 17 MB | **0.5 MB** (-97 %) |
| **Total déploiement** | **130 MB** | **24.5 MB** (-81 %) |
| **RAM résidente du process** | **+418 MB** | **+85 MB** (-80 %) |
| Adversarial 124 cas | 95.2 % | **95.2 %** (identique) |
| Robustness 1227 variantes | 95.8 % | **95.8 %** (identique) |
| Tests unitaires | 68/68 | **68/68** |
| Latence p50 CPU | 2.0 ms | **1.96 ms** (inchangée) |

**À propos de la RAM** : la mémoire résidente baisse encore plus que la taille fichier. Causes : ONNX Runtime ajoute du métadata graph et des buffers d'exécution préalloués, et les hashmaps Rust du tokenizer sont proportionnels à la taille du vocab. Mesure faite avec `psutil.Process().memory_info().rss` après chargement et un warmup `classify()`.

**À propos de la latence** : inchangée (l'inférence est dominée par les 12 couches d'attention de l'encoder, identiques dans les deux variantes ; le pruning n'affecte que le lookup d'embedding qui est O(1)).

**Tradeoff** : tokens hors-FR/EN deviennent `<unk>` (espagnol/allemand/cyrillique/etc. non supportés). Acceptable pour un produit FR+EN ciblé. Cas d'usage où le pruning fait une vraie différence : edge / mobile / IoT (Raspberry Pi 4 GB ne sature plus), multi-tenant (4× plus d'instances par machine), serverless cold start (333 MB de moins à allouer).

## Installation

```bash
pip install -e .
```

Runtime : `onnxruntime`, `tokenizers`, `numpy`. Le tokenizer et la config sont bundlés dans `forsurellm/models/`. Le fichier `forsurellm-int8.onnx` (113 MB) n'est **pas** commité (limite GitHub 100 MB) — il faut soit le récupérer séparément, soit le re-générer via la section "Reproduire l'entraînement" ci-dessous.

Le LLM est disponible sur Huggingface : https://huggingface.co/jcfossati/ForSureLLM

## Interface web de test

Une interface FastAPI + HTML embarquée pour tester le classifier en local sans dépendre de la demo HF Space (utile pour itérer sur le modèle, debugger un cas précis, ou tester la variante prunée).

```bash
pip install -e ".[web]"                          # fastapi + uvicorn (en plus de l'install de base)

python tools/server.py                           # modèle par défaut (113 MB), http://localhost:8000
python tools/server.py --variant _fr-en          # variante prunée (24 MB)
python tools/server.py --port 9000 --host 0.0.0.0  # custom host/port
```

**Endpoints exposés** :

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Page HTML de test (`web/index.html`) |
| `GET` | `/info` | `{variant, onnx_file, size_mb, vocab_size, temperature}` — quel modèle est chargé |
| `POST` | `/classify` | Body `{phrase, threshold}` → `{label, confidence, probabilities, tokens}` |

**Fonctionnalités UI** :
- Input live avec debounce 150 ms (classification à chaque frappe)
- Slider de threshold pour tester le fallback `unknown` (cf. section action confirmation)
- Badge dans le header indiquant la variante chargée (default vs `_fr-en`) avec taille + vocab
- Barres de distribution colorées par classe (yes vert, no rouge, unknown jaune)
- Liste des tokens produits par le tokenizer (utile pour comprendre la segmentation)
- 17 presets cliquables couvrant les cas typiques (sarcasme, idiomes, slang, ponctuation)

**Exemple d'appel API direct** :

```bash
curl -X POST http://localhost:8000/classify \
     -H 'Content-Type: application/json' \
     -d '{"phrase": "ouais grave", "threshold": 0.0}'
# {"label":"yes","confidence":0.98,"probabilities":{"yes":0.98,"no":0.01,"unknown":0.01},"tokens":["<s>","▁ou","ais","▁grave","</s>"]}
```

## API de production (FastAPI + Docker)

Une API HTTP plus minimaliste et orientée prod (pas de page HTML, pas de tokens dans la réponse, batch endpoint, healthchecks, logs JSON struct) — distincte du serveur de test ci-dessus. Pensée pour décharger les coûts d'un autre service qui paye un LLM hosté pour des classifications oui/non/incertain.

```bash
# Build (variante prunée par défaut, image ~291 MB) :
docker build -t forsurellm-api -f api/Dockerfile .

# Variante full multilingual (image ~380 MB) :
docker build --build-arg MODEL_VARIANT=full -t forsurellm-api:full -f api/Dockerfile .

# Run :
docker run -d -p 8000:8000 --name forsurellm forsurellm-api

# Ou via compose (lit MODEL_VARIANT depuis l'env, défault pruned) :
cd api && docker compose up -d
```

**Endpoints** :

| Méthode | Route | Body | Réponse |
|---|---|---|---|
| `POST` | `/classify` | `{phrase, threshold}` | `{label, confidence, probabilities}` |
| `POST` | `/classify/batch` | `{phrases: [...], threshold}` (max 100) | `{results: [...], duration_ms}` |
| `GET` | `/info` | — | `{variant, onnx_file, size_mb, vocab_size, temperature, api_version}` |
| `GET` | `/health` | — | `{status: "ok"}` (liveness) |
| `GET` | `/ready` | — | `{status: "ready"}` ou 503 (readiness, modèle chargé) |

**Caractéristiques** :
- Le modèle est chargé **une fois au démarrage** (lifespan handler), pas à la première requête
- **Batch endpoint** : 100 phrases dans la même requête → ~5 ms par phrase au lieu de ~5 ms réseau + 2 ms compute par appel séparé. Gros gain en pipeline.
- **Logs JSON structurés** sur stdout (compatibles Loki, CloudWatch, Datadog, etc.)
- **Healthcheck Docker** intégré, healthcheck Compose configuré
- Image basée sur `python:3.12-slim`, multi-stage build, l'image finale ne contient pas de compilateur ni la stdlib de build
- L'utilisateur Linux dans le container est non-root (`app:app`)
- ONNX du modèle baked dans l'image au moment du build (téléchargé depuis HuggingFace Model repo)

**Calcul d'optimisation de coût type** : si ton autre service appelle Haiku 4.5 zero-shot pour classifier ~100 000 phrases/mois (typiquement chatbot ou form), ça coûte ~$3-10/mois (selon prompt size) + 600 ms de latence p50. Une instance Docker ForSureLLM tourne en continu pour ~$0.50-1/mois (1 vCPU, 512 MB RAM sur Fly/Render/Hetzner) avec 2 ms p50 et 0 coût d'API. ROI positif dès ~10 000 classifications/mois.

```bash
# Test rapide du batch endpoint :
curl -X POST http://localhost:8000/classify/batch \
     -H 'Content-Type: application/json' \
     -d '{"phrases": ["ouais grave", "tu rêves", "peut-être", "no cap"], "threshold": 0}'
```

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
pytest tests/                      # 47 tests unitaires (API, hard cases EN/FR, threshold, normalisation, perf)
python tools/eval.py             # eval adversarial sur 63 phrases curées
python tools/robustness.py       # 602 variantes graphiques (casse, ponctuation, espaces) → stabilité 98.7 %
python tools/repl.py             # REPL interactif avec visualisation
python tools/bench_baselines.py  # comparaison head-to-head (Haiku, cosine, GPT-4o-mini)
```

## Benchmark vs baselines

Pour donner un point de comparaison concret, ForSureLLM est mis face à
deux baselines représentatives sur les **124 phrases adversarial** :

- **Haiku 4.5 zero-shot** (Anthropic, prompt minimal) — un LLM premium qui
  classe sans entraînement spécifique
- **Cosine MiniLM-L12** (`paraphrase-multilingual-MiniLM-L12-v2`) — embeddings
  multilingues sans fine-tune, classification par similarité aux centroïdes
  de classe

| Classifier | Accuracy | p50 | p95 | Wall time |
|---|---|---|---|---|
| **ForSureLLM** (ONNX int8 local) | **95.2 %** | **1.8 ms** | 4.9 ms | 1.0 s |
| Haiku 4.5 zero-shot | 75.0 % | 602 ms | 1536 ms | 85 s |
| Cosine MiniLM-L12 | 67.7 % | 8 ms | 10 ms | 1.0 s |

**Lecture** : sur un bench élargi à 22 catégories (anciennes + nouvelles :
code-switching, slang Gen-Z, conditionnel, emoji+texte, négation/affirmation
implicite, politesse, résignation, symbolique), ForSureLLM bat Haiku 4.5
zero-shot de **+20.2 points absolus** et la baseline cosine de **+27.5
points**, en tournant **~330× plus vite** sur CPU (1.8 ms vs 602 ms par
classification) et sans coût API.

Catégories où l'écart vs Haiku est massif (>30 pts) :
`modern_slang` (100 % vs 43 %), `negated_verb` (83 % vs 17 %), `sarcasm`
(100 % vs 40 %), `slang_abbrev` (100 % vs 50 %), `symbolic` (100 % vs 40 %),
`resignation` (83 % vs 33 %).

Régressions résiduelles ForSureLLM (6/124, à attaquer dans des PRs ciblées) :
- `conditional` (4/6) : `only if X happens first`, `yes but only halfway`
- `code_switching` (6/7) : `yes mais non` lu comme yes
- `emoji_text` (4/5) : `sure 😅` (l'emoji incertitude n'est pas lu)
- `negated_verb` (5/6) : `ce n'est pas un non` (double négation FR)
- `resignation` (5/6) : `if I must` (yes à contrecœur)

Pour reproduire :

```bash
pip install -e ".[bench]"            # litellm, sentence-transformers, python-dotenv
echo "ANTHROPIC_API_KEY=..." >> .env # pour activer Haiku (sinon skipped)
python tools/bench_baselines.py
```

Le rapport détaillé (toutes les prédictions phrase par phrase) est sauvegardé
dans `evals/bench_baselines.json`.

## Calibration & seuil

Le modèle est **calibré** par temperature scaling (T=0.676) : la confidence retournée = probabilité réelle que la classe soit correcte (ECE=0.012, soit ~1% d'écart moyen).

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
├── llm_config.yaml                # config modèles LLM (éditable)
├── .env.example                   # template clés API
└── checkpoints/                   # artefacts training (gitignored)
```
