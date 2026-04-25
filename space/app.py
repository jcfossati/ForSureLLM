"""Gradio demo for ForSureLLM hosted on HuggingFace Spaces.

Loads the ONNX model from the Model repo (jcfossati/ForSureLLM) at startup,
keeps a small inference function in memory, and exposes a simple yes/no/unknown
classifier UI with click-to-try examples.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL_REPO = "jcfossati/ForSureLLM"
ONNX_FILE = "forsurellm-int8.onnx"

# --- Load artefacts ---------------------------------------------------------
ROOT = Path(__file__).parent
TOKENIZER = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
with (ROOT / "config.json").open(encoding="utf-8") as f:
    CFG = json.load(f)
TOKENIZER.enable_truncation(max_length=CFG["max_length"])
CLASSES = CFG["classes"]
TEMPERATURE = float(CFG.get("temperature", 1.0))

print(f"[boot] downloading {ONNX_FILE} from {MODEL_REPO}...")
ONNX_PATH = hf_hub_download(MODEL_REPO, ONNX_FILE)
SESSION = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
INPUT_NAMES = {i.name for i in SESSION.get_inputs()}
print(f"[boot] ready (model {Path(ONNX_PATH).stat().st_size / 1024 / 1024:.0f} MB)")

# --- Preprocessing (mirror forsurellm/classifier.py) ------------------------
_HAS_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_FRACTION_RE = re.compile(r"^([+-]?\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)$")
_PERCENT_RE = re.compile(r"^([+-]?\d+(?:[.,]\d+)?)\s*%$")
_SIGNED_INT_RE = re.compile(r"^([+-])\s*(\d+)$")
_SYMBOLIC_YES = {"👍", "👍👍", "✅", "🆗", "💯", "💯💯", "++", "+", "✓", "✔", "✔️", "☑", "☑️"}
_SYMBOLIC_NO = {"👎", "👎👎", "❌", "🚫", "⛔", "🛑", "--", "✗", "✘", "✖", "✖️", "≠"}
_SYMBOLIC_UNK = {"?", "??", "???", "?!", "?!?", "🤷", "🤔", "😐", "😶", r"¯\_(ツ)_/¯"}


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = _WS_RE.sub(" ", s).strip()
    return s.lower()


def _classify_symbolic(s: str) -> tuple[str, float] | None:
    s = s.strip()
    if not s:
        return None
    if s in _SYMBOLIC_YES:
        return "yes", 1.0
    if s in _SYMBOLIC_NO:
        return "no", 1.0
    if s in _SYMBOLIC_UNK:
        return "unknown", 1.0
    m = _FRACTION_RE.match(s)
    if m:
        try:
            n, d = float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", "."))
        except ValueError:
            return None
        if d == 0:
            return "unknown", 1.0
        r = n / d
        return ("yes", 1.0) if r >= 0.7 else ("no", 1.0) if r <= 0.3 else ("unknown", 1.0)
    m = _PERCENT_RE.match(s)
    if m:
        try:
            v = float(m.group(1).replace(",", ".")) / 100.0
        except ValueError:
            return None
        return ("yes", 1.0) if v >= 0.7 else ("no", 1.0) if v <= 0.3 else ("unknown", 1.0)
    m = _SIGNED_INT_RE.match(s)
    if m:
        sign, mag = m.group(1), int(m.group(2))
        if mag == 0:
            return "unknown", 1.0
        return ("yes", 1.0) if sign == "+" else ("no", 1.0)
    return None


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def classify(phrase: str) -> tuple[str, np.ndarray]:
    sym = _classify_symbolic(phrase or "")
    if sym is not None:
        label, conf = sym
        probs = np.zeros(3)
        probs[CLASSES.index(label)] = conf
        for i, c in enumerate(CLASSES):
            if c != label:
                probs[i] = (1 - conf) / 2
        return label, probs
    if not _HAS_LETTER_RE.search(phrase or ""):
        return "unknown", np.array([0.0, 0.0, 1.0])
    enc = TOKENIZER.encode(_normalize(phrase))
    feeds = {"input_ids": np.array([enc.ids], dtype=np.int64),
             "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
    if "token_type_ids" in INPUT_NAMES:
        feeds["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
    feeds = {k: v for k, v in feeds.items() if k in INPUT_NAMES}
    logits = SESSION.run(None, feeds)[0][0]
    probs = _softmax(logits / TEMPERATURE)
    label = CLASSES[int(probs.argmax())]
    return label, probs


# --- UI helpers -------------------------------------------------------------
LABEL_EMOJI = {"yes": "✅ YES", "no": "❌ NO", "unknown": "❓ UNKNOWN"}
LABEL_COLOR = {"yes": "#22c55e", "no": "#ef4444", "unknown": "#a3a3a3"}

EXAMPLES = [
    ["carrément"],
    ["tu rêves"],
    ["np"],
    ["oh toootally"],
    ["bah oui"],
    ["+1"],
    ["is the pope catholic"],
    ["je passe"],
    ["yes mais non"],
    ["no cap"],
    ["mouais bof"],
    ["100%"],
    ["if I must"],
    ["nan nan jamais"],
]


def predict(phrase: str) -> tuple[str, dict, str]:
    if not phrase or not phrase.strip():
        return "—", {}, ""
    t0 = time.perf_counter()
    label, probs = classify(phrase)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    badge = f"<div style='font-size:48px;font-weight:700;color:{LABEL_COLOR[label]};text-align:center'>{LABEL_EMOJI[label]}</div>"
    dist = {c: float(p) for c, p in zip(CLASSES, probs)}
    timing = f"<div style='text-align:center;color:#888;font-size:12px;margin-top:8px'>inférence : {elapsed_ms:.1f} ms</div>"
    return badge, dist, timing


# --- Layout -----------------------------------------------------------------
DESCRIPTION = """
# ForSureLLM

Classifier yes/no/unknown ultra-rapide pour réponses courtes (FR + EN). Distillé de Claude Sonnet vers MiniLM-L12 multilingue.

- **95.2 %** sur 124 phrases adversarial (vs Haiku 4.5 zero-shot **75 %**, vs Cosine MiniLM **68 %**)
- **~2 ms** sur CPU, **113 MB** quantifié int8, **+20 pts** vs Haiku
- Préprocesseurs déterministes pour symboles (`+1`, `100%`, `10/10`, `👍`...)

[GitHub](https://github.com/jcfossati/ForSureLLM) · [Model](https://huggingface.co/jcfossati/ForSureLLM)
"""

with gr.Blocks(title="ForSureLLM", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=2):
            inp = gr.Textbox(
                label="Phrase à classer",
                placeholder="Tape une phrase courte en français ou anglais",
                lines=2,
                autofocus=True,
            )
            btn = gr.Button("Classer", variant="primary")
        with gr.Column(scale=3):
            badge = gr.HTML(label="Résultat")
            timing = gr.HTML()
            dist = gr.Label(label="Distribution de probabilités", num_top_classes=3)

    gr.Examples(examples=EXAMPLES, inputs=[inp], label="Exemples (clic pour tester)")

    inp.submit(predict, inputs=[inp], outputs=[badge, dist, timing])
    btn.click(predict, inputs=[inp], outputs=[badge, dist, timing])

if __name__ == "__main__":
    demo.launch()
