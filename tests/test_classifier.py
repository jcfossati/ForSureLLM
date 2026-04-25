"""Tests du classifier Phase 1 (yes/no/unknown)."""
from __future__ import annotations

import time

import pytest

from forsurellm import classify


_MODEL_AVAILABLE = True
try:
    classify("ok")
except Exception:
    _MODEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _MODEL_AVAILABLE, reason="modèle ONNX non exporté — lance scripts/export.py"
)

CLASSES = {"yes", "no", "unknown"}


# --- API contract ---------------------------------------------------------

def test_returns_tuple_label_conf():
    result = classify("ok")
    assert isinstance(result, tuple) and len(result) == 2
    label, conf = result
    assert label in CLASSES
    assert isinstance(conf, float) and 0.0 <= conf <= 1.0


def test_deterministic():
    """Même phrase → même sortie (sanity check inference)."""
    r1 = classify("carrément")
    r2 = classify("carrément")
    assert r1 == r2


def test_empty_string_does_not_crash():
    label, conf = classify("")
    assert label in CLASSES
    assert 0.0 <= conf <= 1.0


def test_long_string_truncated():
    """Phrases très longues : pas de crash, troncature interne à 64 tokens."""
    long = "oui " * 200
    label, conf = classify(long)
    assert label in CLASSES


# --- Hard cases EN --------------------------------------------------------

@pytest.mark.parametrize("phrase", ["yes", "yeah", "yep", "absolutely", "of course", "for sure"])
def test_english_yes(phrase):
    label, conf = classify(phrase)
    assert label == "yes", f"{phrase!r} -> {label} ({conf:.2f})"
    assert conf > 0.6


@pytest.mark.parametrize("phrase", ["no", "nope", "no way", "not a chance", "never", "absolutely not"])
def test_english_no(phrase):
    label, conf = classify(phrase)
    assert label == "no", f"{phrase!r} -> {label} ({conf:.2f})"
    assert conf > 0.6


@pytest.mark.parametrize("phrase", [
    "what time is it",
    "it's raining outside",
    "the meeting starts at three",
])
def test_english_unknown(phrase):
    label, conf = classify(phrase)
    assert label == "unknown", f"{phrase!r} -> {label} ({conf:.2f})"


# --- Hard cases FR --------------------------------------------------------

@pytest.mark.parametrize("phrase", ["oui", "ouais", "ouep", "carrément", "bien sûr", "exactement"])
def test_french_yes(phrase):
    label, conf = classify(phrase)
    assert label == "yes", f"{phrase!r} -> {label} ({conf:.2f})"
    assert conf > 0.6


@pytest.mark.parametrize("phrase", ["non", "nan", "laisse tomber", "non merci", "jamais de la vie"])
def test_french_no(phrase):
    label, conf = classify(phrase)
    assert label == "no", f"{phrase!r} -> {label} ({conf:.2f})"
    assert conf > 0.6


@pytest.mark.parametrize("phrase", [
    "il pleut dehors",
    "quelle heure est-il",
    "la réunion commence à trois heures",
])
def test_french_unknown(phrase):
    label, conf = classify(phrase)
    assert label == "unknown", f"{phrase!r} -> {label} ({conf:.2f})"


# --- Threshold fallback ---------------------------------------------------

def test_threshold_zero_is_noop():
    label, _ = classify("carrément", threshold=0.0)
    assert label == "yes"


def test_threshold_above_max_falls_back_to_unknown():
    """Si threshold > confiance max et label ≠ unknown, retourne unknown."""
    label, _ = classify("ok", threshold=0.999)
    assert label == "unknown"


def test_threshold_does_not_override_true_unknown():
    """Un vrai unknown reste unknown même avec threshold élevé."""
    label, _ = classify("il pleut dehors", threshold=0.999)
    assert label == "unknown"


# --- Robustness aux variantes de surface ----------------------------------

@pytest.mark.parametrize(
    "phrase,variant",
    [
        ("oui", "OUI"),
        ("oui", "oUi"),
        ("oui", "  oui  "),
        ("oui", "oui "),  # nbsp
        ("np", "Np"),
        ("np", "NP"),
        ("carrement", "CARREMENT"),
        ("peut-etre", "PEUT-ETRE"),
        ("grv", "Grv"),
        ("ben oui", "ben    oui"),
    ],
)
def test_normalization_stable(phrase, variant):
    """Casse, espaces multiples et NFC ne doivent pas changer la prédiction."""
    label_canon, _ = classify(phrase)
    label_var, _ = classify(variant)
    assert label_canon == label_var, f"flip: '{phrase}'={label_canon}  '{variant}'={label_var}"


# --- Symbolic preprocessor ------------------------------------------------

@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("+1", "yes"),
        ("-1", "no"),
        ("100%", "yes"),
        ("0%", "no"),
        ("75%", "yes"),
        ("50%", "unknown"),
        ("25%", "no"),
        ("10/10", "yes"),
        ("20/20", "yes"),
        ("0/10", "no"),
        ("5/10", "unknown"),
        ("18/20", "yes"),
        ("👍", "yes"),
        ("👎", "no"),
        ("💯", "yes"),
        ("🚫", "no"),
        ("🤷", "unknown"),
        ("?", "unknown"),
        ("??", "unknown"),
        ("✅", "yes"),
        ("❌", "no"),
    ],
)
def test_symbolic_tokens(phrase, expected):
    label, conf = classify(phrase)
    assert label == expected, f"{phrase!r}: got {label} (conf={conf})"
    assert conf == 1.0, f"symbolic preprocessor must be deterministic"


# --- Performance ----------------------------------------------------------

def test_inference_under_10ms():
    classify("warmup")  # load model cache
    t0 = time.perf_counter()
    for _ in range(50):
        classify("carrément")
    avg_ms = (time.perf_counter() - t0) * 1000 / 50
    assert avg_ms < 10.0, f"inference trop lente : {avg_ms:.2f} ms/call"
