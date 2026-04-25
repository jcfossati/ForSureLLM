# ForSureLLM

> [🇫🇷 Français](README.md) · 🇬🇧 **English**

Distilled `yes` / `no` / `unknown` classifier, EN + FR. Local inference with no API call, int8-quantized ONNX, ~2.5 ms CPU.

🎯 **Live demo**: [huggingface.co/spaces/jcfossati/ForSureLLM](https://huggingface.co/spaces/jcfossati/ForSureLLM)

Designed to recognize consent intent in a user's short reply (bot, CLI, IVR, automation). The host application tracks the state of the pending action; the classifier only says whether the user **agrees**, **refuses**, or **hesitates**.

## Metrics

| | |
|---|---|
| Test accuracy | **91.4 %** |
| ECE (calibration) | **0.007** |
| CPU latency (mean) | **2.5 ms** |
| CPU latency (p95) | 4.3 ms |
| Model size | 113 MB |
| Training (GPU) | 2m24 |
| Inference throughput | ~4000 samp/s (GPU), ~1000 samp/s (CPU) |

Adversarial eval on 63 trap phrases (sarcasm, abbreviations, missing accents, regional idioms): **88.9 %**.

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

## Installation

```bash
pip install -e .
```

Runtime deps: `onnxruntime`, `tokenizers`, `numpy`. The tokenizer and config are bundled in `forsurellm/models/`. The `forsurellm-int8.onnx` file (113 MB) is **not** committed (GitHub's 100 MB limit) — you must either fetch it separately or rebuild it via the "Reproducing training" section below.

LLM is downloadable on Huggingface : https://huggingface.co/jcfossati/ForSureLLM

## Web test interface

```bash
pip install -e ".[web]"    # fastapi + uvicorn
python tools/server.py
```

Then open `http://localhost:8000` — live input, threshold slider, distribution bars, token visualization, 17 clickable presets.

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
pytest tests/                      # 37 unit tests (API, EN/FR hard cases, threshold, perf)
python tools/eval.py             # adversarial eval on 63 curated phrases
python tools/repl.py             # interactive REPL with visualization
```

## Calibration & threshold

The model is **calibrated** via temperature scaling (T=0.689): the returned confidence = actual probability that the class is correct (ECE=0.007, meaning <1% average gap).

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
│   ├── adversarial.jsonl          # 63 curated trap phrases
│   └── last_report.json           # latest eval report
│
├── tests/
│   └── test_classifier.py         # 37 unit tests
│
├── llm_config.yaml                # LLM model config (editable)
├── .env.example                   # API keys template
└── checkpoints/                   # training artifacts (gitignored)
```
