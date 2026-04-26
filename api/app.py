"""Production FastAPI service for ForSureLLM.

Distinct from `tools/server.py` (which is a local dev/test UI). This app:
  - exposes `/classify` (single) and `/classify/batch` (up to 100 phrases)
  - has `/health` (liveness) and `/ready` (readiness with model loaded)
  - logs structured JSON on stdout
  - selects model variant via env var `FORSURELLM_VARIANT` (default = full)
  - is designed to be deployed via Docker behind a reverse proxy

Run locally:
    uvicorn api.app:app --host 0.0.0.0 --port 8000

Run in Docker (pruned variant by default):
    docker build -t forsurellm-api .
    docker run -p 8000:8000 forsurellm-api
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conlist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forsurellm import classify
from forsurellm.classifier import _load, _softmax

# --- Logging (structured JSON) ----------------------------------------------
logger = logging.getLogger("forsurellm.api")
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key in ("variant", "phrase_len", "label", "confidence", "duration_ms", "batch_size"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


_handler.setFormatter(JsonFormatter())
logger.addHandler(_handler)
logger.propagate = False


# --- Lifespan : pre-load model on boot --------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    _load()  # warm the lru_cache
    classify("warmup")
    logger.info(
        "model loaded",
        extra={"variant": os.environ.get("FORSURELLM_VARIANT", "default"),
               "duration_ms": round((time.perf_counter() - t0) * 1000, 1)},
    )
    yield


app = FastAPI(
    title="ForSureLLM API",
    version="0.1.0",
    description="Yes/No/Unknown classifier — local ONNX, ~2 ms CPU.",
    lifespan=lifespan,
)


# --- Schemas ----------------------------------------------------------------
class ClassifyRequest(BaseModel):
    phrase: str = Field(..., max_length=2000)
    threshold: float = Field(0.0, ge=0.0, le=1.0)


class BatchRequest(BaseModel):
    phrases: list[str] = Field(..., min_length=1, max_length=100)
    threshold: float = Field(0.0, ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class BatchItem(BaseModel):
    phrase: str
    label: str
    confidence: float
    probabilities: dict[str, float]


class BatchResponse(BaseModel):
    results: list[BatchItem]
    duration_ms: float


class InfoResponse(BaseModel):
    variant: str
    onnx_file: str
    size_mb: float | None
    vocab_size: int
    temperature: float
    api_version: str


# --- Helpers ----------------------------------------------------------------
def _classify_with_probs(phrase: str, threshold: float) -> tuple[str, float, dict[str, float]]:
    tokenizer, session, classes, input_names, temperature = _load()
    label, conf = classify(phrase, threshold=threshold)

    enc = tokenizer.encode(phrase)
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
    return label, conf, probs_dict


# --- Endpoints --------------------------------------------------------------
@app.post("/classify", response_model=ClassifyResponse)
def classify_endpoint(req: ClassifyRequest) -> ClassifyResponse:
    t0 = time.perf_counter()
    try:
        label, conf, probs = _classify_with_probs(req.phrase, req.threshold)
    except Exception as e:
        logger.exception("classify failed")
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("classify",
                extra={"phrase_len": len(req.phrase), "label": label,
                       "confidence": round(conf, 3), "duration_ms": elapsed})
    return ClassifyResponse(label=label, confidence=conf, probabilities=probs)


@app.post("/classify/batch", response_model=BatchResponse)
def classify_batch_endpoint(req: BatchRequest) -> BatchResponse:
    t0 = time.perf_counter()
    results = []
    for phrase in req.phrases:
        label, conf, probs = _classify_with_probs(phrase, req.threshold)
        results.append(BatchItem(phrase=phrase, label=label, confidence=conf, probabilities=probs))
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("classify_batch",
                extra={"batch_size": len(req.phrases), "duration_ms": elapsed})
    return BatchResponse(results=results, duration_ms=elapsed)


@app.get("/info", response_model=InfoResponse)
def info_endpoint() -> InfoResponse:
    tokenizer, _, _, _, temperature = _load()
    variant = os.environ.get("FORSURELLM_VARIANT", "")
    models_dir = Path(__file__).resolve().parent.parent / "forsurellm" / "models"
    onnx_name = f"forsurellm-int8{variant}.onnx"
    onnx_path = models_dir / onnx_name
    if not onnx_path.exists():
        onnx_name = "forsurellm-int8.onnx"
        onnx_path = models_dir / onnx_name
    size_mb = round(onnx_path.stat().st_size / (1024 * 1024), 1) if onnx_path.exists() else None
    return InfoResponse(
        variant=variant or "default",
        onnx_file=onnx_name,
        size_mb=size_mb,
        vocab_size=tokenizer.get_vocab_size(),
        temperature=round(temperature, 3),
        api_version=app.version,
    )


@app.get("/health")
def health_endpoint() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/ready")
def ready_endpoint() -> JSONResponse:
    try:
        _load()
        return JSONResponse({"status": "ready"})
    except Exception as e:
        return JSONResponse({"status": "loading", "error": str(e)}, status_code=503)
