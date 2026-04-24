"""Runtime inference : charge le modèle ONNX int8 et expose classify()."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

_MODEL_DIR = Path(__file__).parent / "models"


@lru_cache(maxsize=1)
def _load():
    with (_MODEL_DIR / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    tokenizer = Tokenizer.from_file(str(_MODEL_DIR / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=cfg["max_length"])
    session = ort.InferenceSession(
        str(_MODEL_DIR / "forsurellm-int8.onnx"),
        providers=["CPUExecutionProvider"],
    )
    input_names = {i.name for i in session.get_inputs()}
    temperature = float(cfg.get("temperature", 1.0))
    return tokenizer, session, cfg["classes"], input_names, temperature


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def classify(phrase: str, threshold: float = 0.0) -> tuple[str, float]:
    """Classifie une phrase courte en yes/no/unknown.

    Args:
        phrase: texte à classer (EN ou FR).
        threshold: si la confiance max est inférieure, retourne ("unknown", conf).

    Returns:
        (label, confidence) avec label ∈ {"yes", "no", "unknown"} et confidence ∈ [0, 1].
    """
    tokenizer, session, classes, input_names, temperature = _load()
    enc = tokenizer.encode(phrase)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.array([enc.attention_mask], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in input_names:
        feeds["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
    feeds = {k: v for k, v in feeds.items() if k in input_names}

    logits = session.run(None, feeds)[0][0]
    probs = _softmax(logits / temperature)
    idx = int(probs.argmax())
    conf = float(probs[idx])
    label = classes[idx]
    if conf < threshold and label != "unknown":
        unknown_idx = classes.index("unknown")
        return "unknown", float(probs[unknown_idx])
    return label, conf
