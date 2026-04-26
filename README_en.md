# ForSureLLM

> [🇫🇷 Français](README.md) · 🇬🇧 **English**

Distilled `yes` / `no` / `unknown` classifier, EN + FR. Local inference with no API call, int8-quantized ONNX, ~2.5 ms CPU.

🎯 **Live demo**: [huggingface.co/spaces/jcfossati/ForSureLLM](https://huggingface.co/spaces/jcfossati/ForSureLLM)

Designed to recognize consent intent in a user's short reply (bot, CLI, IVR, automation). The host application tracks the state of the pending action; the classifier only says whether the user **agrees**, **refuses**, or **hesitates**.

## Table of contents

- [Metrics](#metrics)
- [Usage](#usage)
  - [Action confirmation (bot, IVR, automation)](#action-confirmation-bot-ivr-automation)
  - [FR+EN pruned variant (24 MB)](#fren-pruned-variant-24-mb)
- [Installation](#installation)
- [Web test interface](#web-test-interface)
- [Production API (FastAPI + Docker)](#production-api-fastapi--docker)
- [Pipeline architecture](#pipeline-architecture)
- [Reproducing training](#reproducing-training)
- [Multi-provider LLM configuration](#multi-provider-llm-configuration)
- [Tests & eval](#tests--eval)
- [Benchmark vs baselines](#benchmark-vs-baselines)
- [Calibration & threshold](#calibration--threshold)
- [What works well](#what-works-well)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Structure](#structure)

## Metrics

| | |
|---|---|
| Test accuracy | **91.7 %** |
| ECE (calibration) | **0.012** |
| CPU latency (mean) | **2.6 ms** |
| CPU latency (p95) | 3.9 ms |
| Model size | 113 MB (full multilingual) · **24 MB (pruned FR+EN)** |
| Training (GPU) | 2m05 |
| Inference throughput | ~4000 samp/s (GPU), ~1000 samp/s (CPU) |

Adversarial eval on 124 trap phrases (sarcasm, abbreviations, missing accents, regional idioms, code-switching, Gen-Z slang, conditional, emoji, implied negation/affirmation, politeness, resignation, symbolic): **95.2 %** (118/124).

## Usage

```python
from forsurellm import classify

classify("absolutely")                # ("yes", 0.975)
classify("no way")                    # ("no", 0.980)
classify("it's raining outside")      # ("unknown", 0.970)
classify("carrément")                 # ("yes", 0.977)
classify("nah fam")                   # ("no", 0.930)
classify("no cap")                    # ("yes", 0.960)
classify("tiguidou")                  # ("yes", 0.979)

# Sarcasm detected via punctuation + pattern
classify("oui bien sûr...")           # ("no", 0.904)
classify("yeah right")                # ("no", 0.88)

# Confidence threshold: fallback to unknown if max < threshold
classify("ok", threshold=0.95)        # ("yes" if >0.95, else "unknown")
```

### Action confirmation (bot, IVR, automation)

For contexts where the classification triggers a server-side action (cancel an order, confirm a payment, kick off a deployment), pass `threshold=0.85` or higher:

```python
classify("yeah right", threshold=0.9)   # ("unknown", …) → re-prompt the user
classify("ouais", threshold=0.9)        # ("yes", 0.97) → action triggered
```

Why: some cultural idioms (`yeah right`, `oh totally`, `tu m'étonnes`) default to sarcasm in general usage but may be sincere from users who don't punctuate. With a higher threshold, these borderline cases fall into `unknown` → your app re-prompts instead of taking the wrong action. Strictly safer for action confirmation; the cost (an occasional re-prompt) is small compared to executing the opposite of the user's intent.

### FR+EN pruned variant (24 MB)

The base model covers 50+ languages via a 250,002-token SentencePiece vocab (≈ 88 MB of embeddings out of the 113 MB total). The pruned variant drops tokens unused by the FR+EN corpus, keeping 5,424 useful tokens + a Latin-1 safety net.

```bash
# Generate (one-shot):
python tools/prune_vocab.py
python training/export.py --src checkpoints/best_fr-en --work-dir checkpoints/onnx_fr-en --out-suffix _fr-en

# Select at inference via CLI flag (server, eval, robustness, audit, bench):
python tools/server.py --variant _fr-en
python tools/eval.py --variant _fr-en
python tools/robustness.py --variant _fr-en

# Without the flag: default model (113 MB).
# The FORSURELLM_VARIANT environment variable is also supported.
```

```python
# From Python (when not using the CLI tools):
import os; os.environ["FORSURELLM_VARIANT"] = "_fr-en"
from forsurellm import classify
classify("carrément")   # same result, from a 5x lighter model
```

| | Original | Pruned FR+EN |
|---|---|---|
| ONNX int8 (disk) | 113 MB | **24 MB** (-79 %) |
| Tokenizer (disk) | 17 MB | **0.5 MB** (-97 %) |
| **Total deployment** | **130 MB** | **24.5 MB** (-81 %) |
| **Process resident memory** | **+418 MB** | **+85 MB** (-80 %) |
| Adversarial 124 cases | 95.2 % | **95.2 %** (identical) |
| Robustness 1227 variants | 95.8 % | **95.8 %** (identical) |
| Unit tests | 68/68 | **68/68** |
| Latency p50 CPU | 2.0 ms | **1.96 ms** (unchanged) |

**About RAM**: process memory drops more than the file size. Reasons: ONNX Runtime adds graph metadata and pre-allocated execution buffers, and the tokenizer's Rust hashmaps scale with vocab size. Measured with `psutil.Process().memory_info().rss` after load + a warmup `classify()`.

**About latency**: unchanged. Inference is dominated by the 12 attention layers in the encoder, identical in both variants; pruning only affects the embedding lookup which is O(1).

**Tradeoff**: tokens outside FR+EN become `<unk>` (Spanish/German/Cyrillic/etc. not supported). Acceptable for a focused FR+EN product. Use cases where pruning makes a real difference: edge / mobile / IoT (a Raspberry Pi 4 GB no longer saturates), multi-tenant servers (4× more instances per machine), serverless cold start (333 MB less to allocate).

## Installation

```bash
pip install -e .
```

Runtime deps: `onnxruntime`, `tokenizers`, `numpy`. The tokenizer and config are bundled in `forsurellm/models/`. The `forsurellm-int8.onnx` file (113 MB) is **not** committed (GitHub's 100 MB limit) — you must either fetch it separately or rebuild it via the "Reproducing training" section below.

LLM is downloadable on Huggingface : https://huggingface.co/jcfossati/ForSureLLM

## Web test interface

A FastAPI + embedded HTML interface to test the classifier locally without depending on the HF Space demo (useful for iterating on the model, debugging a specific case, or testing the pruned variant).

```bash
pip install -e ".[web]"                          # fastapi + uvicorn (on top of the base install)

python tools/server.py                           # default model (113 MB), http://localhost:8000
python tools/server.py --variant _fr-en          # pruned variant (24 MB)
python tools/server.py --port 9000 --host 0.0.0.0  # custom host/port
```

**Exposed endpoints**:

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | HTML test page (`web/index.html`) |
| `GET` | `/info` | `{variant, onnx_file, size_mb, vocab_size, temperature}` — which model is loaded |
| `POST` | `/classify` | Body `{phrase, threshold}` → `{label, confidence, probabilities, tokens}` |

**UI features**:
- Live input with 150 ms debounce (classifies on every keystroke)
- Threshold slider to test the `unknown` fallback (see action confirmation section)
- Header badge showing the loaded variant (default vs `_fr-en`) with size + vocab
- Color-coded distribution bars (yes green, no red, unknown yellow)
- Token list produced by the tokenizer (useful to understand segmentation)
- 17 clickable presets covering typical cases (sarcasm, idioms, slang, punctuation)

**Direct API call example**:

```bash
curl -X POST http://localhost:8000/classify \
     -H 'Content-Type: application/json' \
     -d '{"phrase": "ouais grave", "threshold": 0.0}'
# {"label":"yes","confidence":0.98,"probabilities":{"yes":0.98,"no":0.01,"unknown":0.01},"tokens":["<s>","▁ou","ais","▁grave","</s>"]}
```

## Production API (FastAPI + Docker)

A more minimal, prod-oriented HTTP API (no HTML page, no tokens in response, batch endpoint, healthchecks, structured JSON logs) — distinct from the test server above. Designed to offload costs from another service that pays a hosted LLM for yes/no/unknown classifications.

```bash
# Build (pruned variant by default, ~291 MB image):
docker build -t forsurellm-api -f api/Dockerfile .

# Full multilingual variant (~380 MB image):
docker build --build-arg MODEL_VARIANT=full -t forsurellm-api:full -f api/Dockerfile .

# Run:
docker run -d -p 8000:8000 --name forsurellm forsurellm-api

# Or via compose (reads MODEL_VARIANT from env, defaults to pruned):
cd api && docker compose up -d
```

**Endpoints**:

| Method | Route | Body | Response |
|---|---|---|---|
| `POST` | `/classify` | `{phrase, threshold}` | `{label, confidence, probabilities}` |
| `POST` | `/classify/batch` | `{phrases: [...], threshold}` (max 100) | `{results: [...], duration_ms}` |
| `GET` | `/info` | — | `{variant, onnx_file, size_mb, vocab_size, temperature, api_version}` |
| `GET` | `/health` | — | `{status: "ok"}` (liveness) |
| `GET` | `/ready` | — | `{status: "ready"}` or 503 (readiness, model loaded) |

**Features**:
- Model loaded **once at startup** (lifespan handler), not on first request
- **Batch endpoint**: 100 phrases per request → ~5 ms per phrase instead of ~5 ms network + 2 ms compute per separate call. Big pipeline win.
- **Structured JSON logs** on stdout (compatible with Loki, CloudWatch, Datadog, etc.)
- Built-in **Docker healthcheck**, configured Compose healthcheck
- Image based on `python:3.12-slim`, multi-stage build, final image has no compiler and no build stdlib
- Runs as non-root (`app:app`) inside the container
- Model ONNX baked into the image at build time (downloaded from HuggingFace Model repo)

**Typical cost-optimization calculation**: if your other service calls Haiku 4.5 zero-shot to classify ~100,000 phrases/month (typical chatbot or form), that's ~$3-10/month (depending on prompt size) + 600 ms p50 latency. One ForSureLLM Docker instance running continuously costs ~$0.50-1/month (1 vCPU, 512 MB RAM on Fly/Render/Hetzner) with 2 ms p50 and 0 API cost. Positive ROI from ~10,000 classifications/month.

```bash
# Quick batch endpoint test:
curl -X POST http://localhost:8000/classify/batch \
     -H 'Content-Type: application/json' \
     -d '{"phrases": ["ouais grave", "tu rêves", "peut-être", "no cap"], "threshold": 0}'
```

## Pipeline architecture

```
generate_dataset.py      (Sonnet 4.6: 8700 balanced EN+FR phrases × 3 classes × 8 registers)
        │
augment_idioms.py        (+1500 idiomatic phrases in breadth: Québec, AAVE, British, Aussie, sarcasm)
augment_idioms_deep.py   (+1850 variants of 102 key idioms, hardcoded Sonnet soft labels)
        │
label_dataset.py         (Haiku 4.5: soft labels {yes, no, unknown} + Sonnet re-label on uncertain cases)
        │
clean_dataset.py         (drop pure-noise unknowns: highly-confident neutral/formal facts)
        │
train.py                 (multilingual MiniLM-L12 + 3-class head, KL-div loss, 8 epochs)
        │
calibrate.py             (post-hoc temperature scaling via LBFGS on val set)
        │
export.py                (ONNX + dynamic int8 + CPU benchmark + T in config.json)
        │
forsurellm/classifier.py      (runtime: onnxruntime + tokenizers + calibrated soft probs)
```

## Reproducing training

```bash
pip install -e ".[train]"
cp .env.example .env   # fill in the API key of your chosen provider

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

Observed API cost (default Anthropic setup): ~$15 (Sonnet generation+augmentation + Haiku labeling).
Training: 2m24 on RTX Blackwell (CUDA 12.8), 5m11 on a modern CPU.

## Multi-provider LLM configuration

The pipeline uses [LiteLLM](https://github.com/BerriAI/litellm) to support any provider (Anthropic, OpenAI, Google Gemini, Mistral, Groq, Ollama local, OpenRouter, Azure, Bedrock…). Two files control the choice:

**`llm_config.yaml`** — models and parameters (committed, versioned):

```yaml
generation:                                # creative phrase generation
  model: anthropic/claude-sonnet-4-6
  max_tokens: 4096

labeling:                                  # JSON soft-label (cost-efficient)
  model: anthropic/claude-haiku-4-5-20251001
  max_tokens: 128

labeling_fallback:                         # re-label if primary hesitates (<0.6)
  model: anthropic/claude-sonnet-4-6
  max_tokens: 128
```

**`.env`** — API keys (gitignored, only providers in use):

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

### Alternative config examples

**All OpenAI**:
```yaml
generation:
  model: gpt-4o
labeling:
  model: gpt-4o-mini
labeling_fallback:
  model: gpt-4o
```

**Gemini**:
```yaml
generation:
  model: gemini/gemini-2.5-pro
labeling:
  model: gemini/gemini-2.5-flash
labeling_fallback:
  model: gemini/gemini-2.5-pro
```

**Local (free, Ollama)**:
```yaml
generation:
  model: ollama/qwen2.5:32b
labeling:
  model: ollama/qwen2.5:7b
labeling_fallback:
  model: ollama/qwen2.5:32b
```

**Hybrid** (Sonnet for creativity, Llama on Groq for fast labeling):
```yaml
generation:
  model: anthropic/claude-sonnet-4-6
labeling:
  model: groq/llama-3.3-70b-versatile
labeling_fallback:
  model: anthropic/claude-sonnet-4-6
```

Anthropic prompt caching is automatically enabled when the provider is Anthropic, silently ignored otherwise.

## Tests & eval

```bash
pytest tests/                      # 68 unit tests (API, EN/FR hard cases, threshold, normalization, symbolic, perf)
python tools/eval.py             # adversarial eval on 124 curated phrases
python tools/robustness.py       # 1227 surface variants (case, punctuation, whitespace) → 95.8 % stability
python tools/repl.py             # interactive REPL with visualization
python tools/bench_baselines.py  # head-to-head comparison (Haiku, cosine, GPT-4o-mini)
```

## Benchmark vs baselines

To give a concrete reference point, ForSureLLM is benchmarked against
two representative baselines on the **124 adversarial phrases**:

- **Haiku 4.5 zero-shot** (Anthropic, minimal prompt) — a premium LLM
  classifying without specific training
- **Cosine MiniLM-L12** (`paraphrase-multilingual-MiniLM-L12-v2`) —
  multilingual embeddings without fine-tuning, classification by
  similarity to class centroids

| Classifier | Accuracy | p50 | p95 | Wall time |
|---|---|---|---|---|
| **ForSureLLM** (local ONNX int8) | **95.2 %** | **1.8 ms** | 4.9 ms | 1.0 s |
| Haiku 4.5 zero-shot | 75.0 % | 602 ms | 1536 ms | 85 s |
| Cosine MiniLM-L12 | 67.7 % | 8 ms | 10 ms | 1.0 s |

**Reading**: on a bench expanded to 22 categories (older + new:
code-switching, Gen-Z slang, conditional, emoji+text, implied
negation/affirmation, politeness, resignation, symbolic), ForSureLLM
beats Haiku 4.5 zero-shot by **+20.2 absolute points** and the cosine
baseline by **+27.5 points**, running **~330× faster** on CPU (1.8 ms
vs 602 ms per classification) and with no API cost.

Categories where the gap vs Haiku is massive (>30 pts):
`modern_slang` (100 % vs 43 %), `negated_verb` (83 % vs 17 %), `sarcasm`
(100 % vs 40 %), `slang_abbrev` (100 % vs 50 %), `symbolic` (100 % vs 40 %),
`resignation` (83 % vs 33 %).

ForSureLLM residual regressions (6/124, to attack in dedicated PRs):
- `conditional` (4/6): `only if X happens first`, `yes but only halfway`
- `code_switching` (6/7): `yes mais non` read as yes
- `emoji_text` (4/5): `sure 😅` (uncertainty emoji not parsed)
- `negated_verb` (5/6): `ce n'est pas un non` (FR double negation)
- `resignation` (5/6): `if I must` (reluctant yes)

To reproduce:

```bash
pip install -e ".[bench]"            # litellm, sentence-transformers, python-dotenv
echo "ANTHROPIC_API_KEY=..." >> .env # to enable Haiku (else skipped)
python tools/bench_baselines.py
```

Detailed report (per-phrase predictions) is saved in `evals/bench_baselines.json`.

## Calibration & threshold

The model is **calibrated** via temperature scaling (T=0.680): the returned confidence = actual probability that the class is correct (ECE=0.012, meaning ~1% average gap).

| Threshold | Mode | Use case |
|---|---|---|
| `0.0` | Lenient | Always classify, conversational use |
| `0.7` | Balanced | ~80% of cases, 95%+ precision |
| `0.9` | Strict | Accepts canonical cases, rest = unknown |
| `0.95` | Very strict | Auto-accept only clear-cut cases (>95% reliable) |

Typical production pattern:

```python
label, conf = classify(user_reply)
if conf > 0.95:
    execute(pending_action)           # auto-accept
elif conf > 0.7:
    confirm(pending_action)           # ask for confirmation
else:
    return "could not understand"     # escalate
```

## What works well

- Canonical yes/no EN+FR: 100% (including *"ouep"*, *"nan"*, *"nope"*, *"nah"*)
- Varied registers: casual, formal, slang, interjections — all >95%
- Robust to typos and missing accents (*"carrement"*, *"peut-etre"*, *"noooon"*)
- Regional idioms: Québécois (*"tiguidou"*), AAVE (*"no cap"*, *"fax no printer"*), British (*"bollocks"*), Aussie (*"too right"*)
- Hedges: *"bof"*, *"peut-être"*, *"I dunno"* — 100%
- Some sarcasm via pattern + punctuation: *"oui bien sûr..."* → no, *"yeah right"* → no, *"oh great"* → no
- Pragmatic detection of "..." as a sarcasm/resignation marker in French

## Limitations

- **Sarcasm outside frequent patterns** — a very subtle sarcasm with no punctuation marker may be missed.
- **Chat abbreviations** (*"lol"*, *"np"*, *"grv"*) — under-represented, unstable results.
- **OOD** (fully out-of-distribution inputs, e.g. random text *"tape une p"*) — softmax forces a class. Use a high threshold for fallback.
- **Languages other than EN/FR** — the tokenizer covers them but the model wasn't trained on them (out of scope).

## Roadmap

- [ ] **Vocab pruning** — drop the ~245k unused tokens (CJK, Arabic, Thai, etc.) to bring the model from 113 MB down to ~25 MB. Tradeoff: loss of robustness on non-FR/EN languages.
- [ ] **Chat abbreviation coverage** — targeted additions of *"lol, np, grv, mdr, omg, wtf..."* to improve the `slang_abbrev` category.

## Structure

```
ForSureLLM/
├── forsurellm/                    # distributable runtime package
│   ├── __init__.py
│   ├── classifier.py              # inference (onnxruntime + tokenizers)
│   └── models/
│       ├── forsurellm-int8.onnx   # (gitignored, 113 MB)
│       ├── tokenizer.json
│       └── config.json            # classes, max_length, temperature
│
├── training/                      # distillation pipeline
│   ├── llm_client.py              # multi-provider LiteLLM wrapper
│   ├── generate.py                # broad EN+FR generation
│   ├── augment_idioms.py          # breadth idioms (60 seeds)
│   ├── augment_idioms_deep.py     # depth idioms (102 × 20)
│   ├── label.py                   # soft labels
│   ├── clean.py                   # drop pure-noise unknowns
│   ├── train.py                   # KL-div distillation
│   ├── calibrate.py               # temperature scaling
│   └── export.py                  # ONNX int8 + benchmark
│
├── tools/                         # standalone utilities
│   ├── repl.py                    # interactive terminal REPL
│   ├── eval.py                    # curated adversarial eval
│   └── server.py                  # FastAPI + web interface
│
├── web/
│   └── index.html                 # test interface
│
├── data/                          # datasets (content gitignored)
│   ├── raw/                       # generated phrases
│   ├── labeled/                   # labeled phrases
│   └── splits/                    # stratified train/val/test
│
├── evals/
│   ├── adversarial.jsonl          # 124 curated trap phrases
│   ├── bench_baselines.json       # head-to-head vs Haiku 4.5 + cosine
│   ├── robustness_report.json     # surface-variant stability report
│   └── last_report.json           # latest eval report
│
├── tests/
│   └── test_classifier.py         # 68 unit tests
│
├── llm_config.yaml                # LLM model config (editable)
├── .env.example                   # API keys template
└── checkpoints/                   # training artifacts (gitignored)
```
