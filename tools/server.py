"""Serveur web pour tester le classifier en live.

Usage:
    python tools/server.py                      # modèle par défaut (113 MB multilingual)
    python tools/server.py --variant _fr-en     # variante prunée (24 MB FR+EN)
    # puis ouvre http://localhost:8000

Expose :
    GET  /              -> interface HTML (web/index.html)
    POST /classify      -> {phrase, threshold} -> {label, confidence, probabilities}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# `--variant` is consumed by tools._variant before forsurellm is imported.
import tools._variant  # noqa: F401

_parser = argparse.ArgumentParser()
_parser.add_argument("--host", default="127.0.0.1")
_parser.add_argument("--port", type=int, default=8000)
_args = _parser.parse_args()

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from forsurellm import classify
from forsurellm.classifier import _load, _softmax

ROOT = Path(__file__).parent.parent
WEB_DIR = ROOT / "web"

app = FastAPI(title="ForSureLLM classifier")


class ClassifyRequest(BaseModel):
    phrase: str = Field(..., max_length=2000)
    threshold: float = Field(0.0, ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    tokens: list[str]


@app.post("/classify", response_model=ClassifyResponse)
def classify_endpoint(req: ClassifyRequest) -> ClassifyResponse:
    tokenizer, session, classes, input_names, temperature = _load()
    enc = tokenizer.encode(req.phrase)
    feeds = {
        "input_ids": np.array([enc.ids], dtype=np.int64),
        "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
    }
    if "token_type_ids" in input_names:
        feeds["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
    feeds = {k: v for k, v in feeds.items() if k in input_names}

    logits = session.run(None, feeds)[0][0]
    probs = _softmax(logits / temperature)
    probs_dict = {c: float(p) for c, p in zip(classes, probs)}

    label, conf = classify(req.phrase, threshold=req.threshold)
    return ClassifyResponse(
        label=label,
        confidence=conf,
        probabilities=probs_dict,
        tokens=enc.tokens,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if __name__ == "__main__":
    variant_label = tools._variant._VARIANT or "(default 113 MB)"
    print(f"Loading model variant={variant_label!r}...")
    _load()
    print(f"Ready. Ouvre http://{_args.host}:{_args.port}")
    uvicorn.run(app, host=_args.host, port=_args.port, log_level="warning")
