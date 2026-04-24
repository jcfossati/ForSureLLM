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


# --- Performance ----------------------------------------------------------

def test_inference_under_10ms():
    classify("warmup")  # load model cache
    t0 = time.perf_counter()
    for _ in range(50):
        classify("carrément")
    avg_ms = (time.perf_counter() - t0) * 1000 / 50
    assert avg_ms < 10.0, f"inference trop lente : {avg_ms:.2f} ms/call"
