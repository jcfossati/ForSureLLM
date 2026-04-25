---
title: ForSureLLM
emoji: 🏆
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Ultra-fast yes/no/unknown classifier (FR+EN), 2ms CPU
models:
  - jcfossati/ForSureLLM
---

# ForSureLLM — interactive demo

This Space hosts the live demo of [ForSureLLM](https://github.com/jcfossati/ForSureLLM),
a 113 MB MiniLM-L12 multilingual model distilled from Claude Sonnet for
classifying short French/English phrases as `yes` / `no` / `unknown`.

The ONNX checkpoint is loaded from the
[jcfossati/ForSureLLM](https://huggingface.co/jcfossati/ForSureLLM) Model
repo at startup. Tokenizer and config are bundled in the Space.

## Numbers

| Metric | Value |
|---|---|
| Adversarial accuracy (124 cases) | **95.2 %** |
| vs Haiku 4.5 zero-shot | **+20.2 pts** |
| vs Cosine MiniLM-L12 | **+27.5 pts** |
| Latency p50 (CPU) | 1.8 ms |
| Model size | 113 MB |

## Source

App and tokenizer/config files are mirrored from
[`space/`](https://github.com/jcfossati/ForSureLLM/tree/main/space)
in the GitHub repo. Update via `python tools/deploy_space.py` after each
model retrain.
