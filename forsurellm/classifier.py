"""Runtime inference : charge le modèle ONNX int8 et expose classify()."""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

_MODEL_DIR = Path(__file__).parent / "models"

# Pré-processeur : une phrase sans aucune lettre (ex: "!", "?", "...", "123",
# espaces seuls, vide) ne peut pas porter d'intention yes/no - on court-circuite
# le modèle et on retourne `unknown` avec confiance maximale. Évite des
# régressions du modèle sur des cas dégénérés (cf. eval adversarial catégorie
# `degenerate`).
_HAS_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Tokens symboliques sans lettre fréquemment utilisés comme réponses oui/non
# dans les chats, forums, Slack, GitHub, formulaires. La liste couvre les
# emojis usuels et les symboles non-numériques ; les patterns numériques
# (fractions, pourcentages, +/-N) sont gérés par règles regex plus bas.
_SYMBOLIC_YES: set[str] = {
    "👍", "👍🏻", "👍🏼", "👍🏽", "👍🏾", "👍🏿", "👍👍", "👍!",
    "✅", "✅!", "🆗", "💯", "💯💯", "💯!",
    "++", "+", "✓", "✔", "✔️", "☑", "☑️",
    "🙆", "🙆‍♂️", "🙆‍♀️",
}
_SYMBOLIC_NO: set[str] = {
    "👎", "👎🏻", "👎🏼", "👎🏽", "👎🏾", "👎🏿", "👎👎", "👎!",
    "❌", "❌!", "🚫", "⛔", "🛑",
    "--", "✗", "✘", "✖", "✖️",
    "🙅", "🙅‍♂️", "🙅‍♀️",
    "≠",
}
_SYMBOLIC_UNKNOWN: set[str] = {
    "?", "??", "???", "?!", "?!?", "?!?!", "!?", "‽",
    "🤷", "🤷‍♂️", "🤷‍♀️", "🤷🏻", "🤷🏼", "🤷🏽", "🤷🏾", "🤷🏿",
    r"¯\_(ツ)_/¯", r"¯\(ツ)/¯", r"\_(ツ)_/",
    "🤔", "😐", "😶",
}

# n/d, n%, +/-n : ratio >= 0.7 = yes, <= 0.3 = no, sinon unknown.
_FRACTION_RE = re.compile(r"^([+-]?\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)$")
_PERCENT_RE = re.compile(r"^([+-]?\d+(?:[.,]\d+)?)\s*%$")
_SIGNED_INT_RE = re.compile(r"^([+-])\s*(\d+)$")


def _classify_symbolic(s: str) -> tuple[str, float] | None:
    """Reconnaît les tokens symboliques (sans lettre) qui portent une intention
    yes/no. Retourne None si aucun pattern ne matche, laissant la phrase
    poursuivre vers le modèle (ou le court-circuit `unknown`).
    """
    s = s.strip()
    if not s:
        return None
    if s in _SYMBOLIC_YES:
        return "yes", 1.0
    if s in _SYMBOLIC_NO:
        return "no", 1.0
    if s in _SYMBOLIC_UNKNOWN:
        return "unknown", 1.0

    m = _FRACTION_RE.match(s)
    if m:
        try:
            n = float(m.group(1).replace(",", "."))
            d = float(m.group(2).replace(",", "."))
        except ValueError:
            return None
        if d == 0:
            return "unknown", 1.0
        ratio = n / d
        if ratio >= 0.7:
            return "yes", 1.0
        if ratio <= 0.3:
            return "no", 1.0
        return "unknown", 1.0

    m = _PERCENT_RE.match(s)
    if m:
        try:
            v = float(m.group(1).replace(",", ".")) / 100.0
        except ValueError:
            return None
        if v >= 0.7:
            return "yes", 1.0
        if v <= 0.3:
            return "no", 1.0
        return "unknown", 1.0

    m = _SIGNED_INT_RE.match(s)
    if m:
        sign, mag = m.group(1), int(m.group(2))
        if mag == 0:
            return "unknown", 1.0
        return ("yes", 1.0) if sign == "+" else ("no", 1.0)

    return None


def _normalize(phrase: str) -> str:
    """Normalise les variantes de surface qui ne portent pas de sens : casse,
    forme unicode, espaces multiples. Réduit la fragilité aux entrées type
    `Np`, `PEUT-ETRE`, `oUi`, `ben    oui` (cf. tools/robustness.py).
    """
    s = unicodedata.normalize("NFC", phrase)
    s = _WS_RE.sub(" ", s).strip()
    return s.lower()


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
    # Pré-processeurs déterministes : symboles connus d'abord, puis fallback
    # `unknown` pour toute entrée sans aucune lettre (cf. _HAS_LETTER_RE).
    sym = _classify_symbolic(phrase or "")
    if sym is not None:
        return sym
    if not _HAS_LETTER_RE.search(phrase or ""):
        return "unknown", 1.0

    tokenizer, session, classes, input_names, temperature = _load()
    enc = tokenizer.encode(_normalize(phrase))
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
