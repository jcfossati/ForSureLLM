# Brief projet : ForSureLLM

YES / NO / unknown classifier EN+FR distillé — lib Python autonome, inférence locale.

## Objectif
Classifier une phrase courte (réponse utilisateur) en `yes` / `no` / `unknown`, sans appel API, latence <5 ms CPU, modèle autonome. Rôle : oracle de consentement — l'application hôte maintient l'état de l'action en cours, le classifier dit seulement si l'utilisateur adhère/refuse/hésite.

## Stack
- **Teacher génération** : défaut Claude Sonnet 4.6 (configurable via `llm_config.yaml`)
- **Teacher labeling** : défaut Claude Haiku 4.5 (configurable, avec fallback sur modèle plus puissant si confidence <0.6)
- **Provider router** : [LiteLLM](https://github.com/BerriAI/litellm) — supporte Anthropic, OpenAI, Gemini, Mistral, Groq, Ollama, OpenRouter, Azure, Bedrock
- **Backbone student** : `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Training** : HuggingFace `transformers` + `datasets`
- **Export** : ONNX + quantization int8 via `optimum`
- **Calibration** : temperature scaling post-hoc (LBFGS sur val set)

## Pipeline

### 1. Génération dataset (`training/generate.py`)
Sonnet génère ~8700 phrases courtes EN+FR, équilibrées sur 3 classes × 8 registres (familier, soutenu, argot, indirect, sarcastique, interjection, fautes, neutre).

### 2. Augmentation idiomatique (`training/augment_idioms.py` + `augment_idioms_deep.py`)
~3300 phrases idiomatiques supplémentaires :
- `augment_idioms.py` : largeur (60 seeds thématiques × 30 variantes) — couvre régionalismes (Québec, AAVE, british, aussie) et sarcasmes.
- `augment_idioms_deep.py` : profondeur (102 idiomes clés × 20 variantes contextuelles, soft labels hardcodés 0.92/0.04/0.04).

### 3. Labeling soft (`training/label.py`)
Claude Haiku 4.5 attribue `{yes, no, unknown}` calibré. Re-label Sonnet sur confiance max <0.6.

### 4. Nettoyage (`training/clean.py`)
Retrait des unknowns "pure noise" (faits soutenus/neutres très confiants, aucune valeur comme réponse).

### 5. Training (`training/train.py`)
Fine-tuning MiniLM-L12 multilingue avec head 3 classes, loss KL-divergence sur soft labels, 8 epochs, batch 32, lr 2e-5, warmup 10%, weight decay 0.01.

### 6. Calibration (`training/calibrate.py`)
Temperature scaling post-hoc : LBFGS sur NLL (hard labels) pour trouver T optimal. Sauvegarde T dans `temperature.json`.

### 7. Export (`training/export.py`)
ONNX via `optimum.exporters.onnx` → quantization int8 dynamique (avx512-vnni) → benchmark CPU. Temperature injectée dans `config.json`.

### 8. API d'usage (`forsurellm/`)
```python
from forsurellm import classify
classify("carrément")           # ("yes", 0.98)
classify("laisse tomber")       # ("no", 0.98)
classify("il pleut")            # ("unknown", 0.97)
classify("ok", threshold=0.9)   # fallback unknown si max < 0.9
```

## Métriques finales
- **Accuracy test** : 91.4 %
- **ECE** : 0.007 (9× plus calibré que non-calibré)
- **Latence CPU** : 2.5 ms (p95 : 4.3 ms)
- **Taille** : 113 MB ONNX int8
- **Training GPU** : 2m24 sur RTX Blackwell
- **Budget API total** : ~$15 sur $25 dispo

## Structure projet
```
ForSureLLM/
├── data/
│   ├── raw/              # {en,fr}.jsonl + idioms.jsonl (Sonnet gen)
│   ├── labeled/          # {en,fr}.jsonl + idioms_deep.jsonl (Haiku label)
│   └── splits/           # train/val/test.jsonl stratifiés
├── scripts/              # pipeline reproductible
├── forsurellm/                # package runtime (classifier.py + ONNX int8)
├── web/                  # interface HTML de test (fastapi)
├── tests/                # tests unitaires + eval adversarial
└── checkpoints/          # HF + ONNX fp32/int8 (gitignored)
```

## Amélioration laissée en TODO
- **Vocab pruning** : retirer les ~245k tokens non-utilisés par EN+FR pour passer de 113 MB à ~25 MB. Tradeoff : perte de robustesse aux langues non-FR/EN.
