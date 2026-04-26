---
license: apache-2.0
language:
  - fr
  - en
library_name: onnx
pipeline_tag: text-classification
base_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
tags:
  - yes-no
  - intent-classification
  - distillation
  - onnx
  - edge
  - quantization
  - sentence-transformers
metrics:
  - accuracy
model-index:
  - name: ForSureLLM
    results:
      - task:
          type: text-classification
          name: Yes/No/Unknown classification
        metrics:
          - type: accuracy
            value: 0.952
            name: Adversarial accuracy (124 cases)
          - type: accuracy
            value: 0.917
            name: Test accuracy (1178 cases)
---

# ForSureLLM

Ultra-rapid `yes` / `no` / `unknown` classifier for short user replies, distilled from Claude Sonnet 4.6 into a multilingual MiniLM-L12. **2 ms on CPU**, **24-113 MB**, no API call needed.

- 🎯 **Try it live**: [HuggingFace Space demo](https://huggingface.co/spaces/jcfossati/ForSureLLM)
- 📦 **Source code**: [github.com/jcfossati/ForSureLLM](https://github.com/jcfossati/ForSureLLM)

## What it does

Given a short French or English reply (1-30 words typically), returns whether the user is **agreeing**, **refusing**, or **hesitating** about a pending action. Designed as a consent-intent oracle for chatbots, IVR systems, CLI confirmations, and automation flows.

```python
from forsurellm import classify

classify("carrément")            # ("yes", 0.97)
classify("laisse tomber")        # ("no", 0.98)
classify("je sais pas trop")     # ("unknown", 0.96)
classify("oui mais non")         # ("unknown", 0.92)
classify("yeah right")           # ("no", 0.87)   # sarcasm detected
classify("+1")                   # ("yes", 1.00)  # symbolic preprocessor
classify("👍")                    # ("yes", 1.00)
```

## Numbers

| Metric | Value |
|---|---|
| Adversarial accuracy (124 trap phrases, 22 categories) | **95.2 %** |
| Surface-variant robustness (1227 variants) | 95.8 % |
| Test set accuracy (1178 phrases) | 91.7 % |
| Calibration ECE | 0.012 |
| **CPU latency p50** | **1.8 ms** |
| ONNX int8 size | 113 MB (multilingual) · **24 MB (FR+EN pruned variant)** |

**Head-to-head on the 124-case adversarial bench**:

| Classifier | Accuracy | p50 latency | API cost |
|---|---|---|---|
| **ForSureLLM** | **95.2 %** | **1.8 ms** | 0 |
| Haiku 4.5 zero-shot | 75.0 % | 602 ms | $$ |
| Cosine MiniLM-L12 (no fine-tune) | 67.7 % | 8 ms | 0 |

ForSureLLM beats Haiku 4.5 zero-shot by **+20.2 pts** while running **~330× faster**.

## Strengths

Categories where ForSureLLM crushes a generalist LLM (Haiku 4.5):

- `modern_slang` (Gen-Z): `no cap`, `bet`, `say less`, `deadass` — 100 % vs 43 %
- `negated_verb`: `I wouldn't say no`, `ce n'est pas un non` — 83 % vs 17 %
- `sarcasm`: `oui bien sûr...`, `yeah right` — 100 % vs 40 %
- `symbolic`: `+1`, `100%`, `👍`, `10/10` — 100 % vs 40 % (deterministic preprocessor)
- `slang_abbrev`: `np`, `tkt`, `kk`, `nope` — 100 % vs 50 %

## Files in this repo

- `forsurellm-int8.onnx` — full multilingual model, 113 MB (50+ languages supported via shared subwords, FR+EN tuned)
- (Optional) `forsurellm-int8_fr-en.onnx` — vocab-pruned FR+EN variant, 24 MB. Same predictions as the full model on FR+EN inputs, 5× lighter on disk and in RAM (+85 MB process memory vs +418 MB), latency unchanged. Tokens outside FR+EN become `<unk>`.

## How to use (without the package)

```python
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import numpy as np

onnx_path = hf_hub_download("jcfossati/ForSureLLM", "forsurellm-int8.onnx")
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
# tokenizer.json must be downloaded from the GitHub repo (space/tokenizer.json)
# or installed via the forsurellm package once published.
```

For the full preprocessing pipeline (case normalisation, symbolic shortcuts, sarcasm-aware threshold), use the `forsurellm` Python package — see the [GitHub repo](https://github.com/jcfossati/ForSureLLM) for installation.

## Training procedure

- **Backbone**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (12 layers, 384 hidden)
- **Teacher**: Claude Sonnet 4.6 (generation) + Claude Haiku 4.5 (labeling, with Sonnet fallback when confidence < 0.6)
- **Loss**: KL-divergence on soft labels (3 classes)
- **Dataset**: ~5,800 hand-curated + LLM-generated EN+FR phrases, balanced across 22 adversarial categories
- **Training**: 8 epochs, batch 32, lr 2e-5, warmup 10%, weight decay 0.01 (~2 min on RTX Blackwell)
- **Calibration**: temperature scaling (T = 0.680, fitted by LBFGS on val set NLL)
- **Export**: ONNX dynamic quantization (avx512-vnni, int8)

## Limitations

- **EN + FR only**. The full model (113 MB) keeps the multilingual vocab and may produce reasonable cross-lingual outputs on related Latin-script languages (Spanish/Italian/German), but is not trained for them. The pruned variant (24 MB) drops non-FR/EN tokens entirely.
- **Short replies**. Optimized for 1-30 word answers. Long passages will be truncated at 64 tokens.
- **Sarcasm detection has cultural priors**. `yeah right` defaults to "no" because it's overwhelmingly sarcastic in modern English usage — a sincere user without punctuation might get the wrong call. Use `threshold=0.85` for action-confirmation contexts to fall back to `unknown` on borderline cases.

## License

Apache 2.0 — same as the base MiniLM model.
