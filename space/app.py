"""Gradio demo for ForSureLLM hosted on HuggingFace Spaces.

Loads BOTH variants (full multilingual + pruned FR+EN) at startup and exposes
a dropdown so visitors can compare predictions / sizes / speed in real time.

ONNX files are downloaded from the jcfossati/ForSureLLM Model repo at boot.
Tokenizer + config files are bundled in this Space (small).
"""
from __future__ import annotations

import html
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL_REPO = "jcfossati/ForSureLLM"

# --- Variant registry ------------------------------------------------------
ROOT = Path(__file__).parent


@dataclass
class Variant:
    label: str            # display name in the dropdown
    onnx_remote: str      # filename in the Model repo to download
    tokenizer_local: str  # bundled tokenizer file in space/
    config_local: str     # bundled config file in space/
    tokenizer: Tokenizer = None
    session: ort.InferenceSession = None
    classes: list = None
    input_names: set = None
    temperature: float = 1.0
    size_mb: float = 0.0
    vocab_size: int = 0


VARIANTS: dict[str, Variant] = {
    "Full multilingual (113 MB)": Variant(
        label="Full multilingual (113 MB)",
        onnx_remote="forsurellm-int8.onnx",
        tokenizer_local="tokenizer.json",
        config_local="config.json",
    ),
    "Pruned FR+EN (24 MB)": Variant(
        label="Pruned FR+EN (24 MB)",
        onnx_remote="forsurellm-int8_fr-en.onnx",
        tokenizer_local="tokenizer_fr-en.json",
        config_local="config_fr-en.json",
    ),
}
DEFAULT_VARIANT = "Pruned FR+EN (24 MB)"


def _load_variant(v: Variant) -> None:
    print(f"[boot] loading {v.label}: downloading {v.onnx_remote}...")
    onnx_path = hf_hub_download(MODEL_REPO, v.onnx_remote)
    v.size_mb = round(Path(onnx_path).stat().st_size / (1024 * 1024), 1)
    v.tokenizer = Tokenizer.from_file(str(ROOT / v.tokenizer_local))
    with (ROOT / v.config_local).open(encoding="utf-8") as f:
        cfg = json.load(f)
    v.tokenizer.enable_truncation(max_length=cfg["max_length"])
    v.classes = cfg["classes"]
    v.temperature = float(cfg.get("temperature", 1.0))
    v.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    v.input_names = {i.name for i in v.session.get_inputs()}
    v.vocab_size = v.tokenizer.get_vocab_size()
    print(f"[boot] -> {v.label} ready ({v.size_mb} MB ONNX, {v.vocab_size} tokens)")


for v in VARIANTS.values():
    _load_variant(v)


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


def classify_full(phrase: str, variant_label: str, threshold: float = 0.0) -> dict:
    """Returns {label, confidence, probs, tokens, source, variant}."""
    v = VARIANTS[variant_label]
    sym = _classify_symbolic(phrase or "")
    if sym is not None:
        label, conf = sym
        probs = {c: 0.0 for c in v.classes}
        probs[label] = conf
        return {"label": label, "confidence": conf, "probs": probs,
                "tokens": [phrase.strip()], "source": "symbolic", "variant": variant_label}
    if not _HAS_LETTER_RE.search(phrase or ""):
        return {"label": "unknown", "confidence": 1.0,
                "probs": {"yes": 0.0, "no": 0.0, "unknown": 1.0},
                "tokens": [], "source": "no-letter shortcut", "variant": variant_label}

    enc = v.tokenizer.encode(_normalize(phrase))
    feeds = {"input_ids": np.array([enc.ids], dtype=np.int64),
             "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
    if "token_type_ids" in v.input_names:
        feeds["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
    feeds = {k: x for k, x in feeds.items() if k in v.input_names}
    logits = v.session.run(None, feeds)[0][0]
    probs_arr = _softmax(logits / v.temperature)
    probs = {c: float(p) for c, p in zip(v.classes, probs_arr)}
    idx = int(probs_arr.argmax())
    label = v.classes[idx]
    conf = probs[label]
    if conf < threshold and label != "unknown":
        label = "unknown"
        conf = probs["unknown"]
    return {"label": label, "confidence": conf, "probs": probs,
            "tokens": [t for t in enc.tokens if t not in {"<s>", "</s>", "[CLS]", "[SEP]", "[PAD]"}],
            "source": "model", "variant": variant_label}


# --- Rendering --------------------------------------------------------------
EMPTY_HTML = '<div class="fsl-result fsl-result-empty">Tape une phrase pour la classifier.</div>'

RESULT_TEMPLATE = """
<div class="fsl-result">
  <div class="fsl-header">
    <div class="fsl-label fsl-{label}">{label_upper}</div>
    <div class="fsl-conf">conf {confidence:.3f}</div>
  </div>
  <div class="fsl-bars">{bars}</div>
  <div class="fsl-tokens">
    <div class="fsl-tokens-label">{token_label}</div>
    <div class="fsl-tokens-list">{tokens_html}</div>
  </div>
  <div class="fsl-meta">{source} · {variant} · {elapsed_ms:.1f} ms</div>
</div>
"""

BAR_TEMPLATE = """
<div class="fsl-bar-row">
  <div class="fsl-bar-name">{name}</div>
  <div class="fsl-bar-track"><div class="fsl-bar-fill fsl-{name}" style="width:{pct:.1f}%"></div></div>
  <div class="fsl-bar-val">{value:.3f}</div>
</div>
"""


def render_result(phrase: str, variant_label: str, threshold: float) -> str:
    if not phrase or not phrase.strip():
        return EMPTY_HTML
    t0 = time.perf_counter()
    res = classify_full(phrase, variant_label, threshold)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    bars = "".join(
        BAR_TEMPLATE.format(name=cls, pct=res["probs"][cls] * 100, value=res["probs"][cls])
        for cls in ["yes", "no", "unknown"]
    )
    tokens = res["tokens"]
    if tokens and res["source"] == "model":
        tokens_html = "".join(f'<span class="fsl-tok">{html.escape(t)}</span>' for t in tokens)
        token_label = f"tokens ({len(tokens)})"
    else:
        tokens_html = '<span class="fsl-tok-empty">—</span>'
        token_label = "tokens"
    return RESULT_TEMPLATE.format(
        label=res["label"],
        label_upper=res["label"].upper(),
        confidence=res["confidence"],
        bars=bars,
        tokens_html=tokens_html,
        token_label=token_label,
        source=res["source"],
        variant=res["variant"].split(" (")[0],  # short label
        elapsed_ms=elapsed_ms,
    )


PRESETS = [
    "carrément", "absolutely", "laisse tomber", "no way",
    "peut-être", "maybe", "bof", "je sais pas trop",
    "il pleut dehors", "what time is it",
    "oui mais non", "yeah right", "tu rêves",
    "pourquoi pas", "ouais", "nope", "tkt", "np",
    "no cap", "is the pope catholic", "+1", "100%", "👍",
    "oh toootally", "if I must", "bah oui",
]

CSS = """
:root {
  --fsl-bg: #0e0f13;
  --fsl-surface: #171922;
  --fsl-surface-2: #0b0c10;
  --fsl-border: #262937;
  --fsl-text: #e5e7ee;
  --fsl-muted: #8b90a5;
  --fsl-yes: #22c55e;
  --fsl-no: #ef4444;
  --fsl-unknown: #eab308;
  --fsl-accent: #6366f1;
}

.gradio-container { background: var(--fsl-bg) !important; max-width: 760px !important; margin: 0 auto !important; }
.gradio-container, .gradio-container * { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif !important; }

#fsl-header { padding: 8px 0 24px; border-bottom: 1px solid var(--fsl-border); margin-bottom: 24px; }
#fsl-header h1 { color: var(--fsl-text); font-size: 24px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
#fsl-header .sub { color: var(--fsl-muted); font-size: 13px; margin: 0; }
#fsl-header .links { color: var(--fsl-muted); font-size: 12px; margin-top: 8px; }
#fsl-header .links a { color: var(--fsl-accent); text-decoration: none; margin-right: 12px; }
#fsl-header .links a:hover { text-decoration: underline; }

.fsl-input-card { background: var(--fsl-surface); border: 1px solid var(--fsl-border) !important; border-radius: 12px; padding: 4px; }
.fsl-input-card textarea, .fsl-input-card input {
  background: var(--fsl-surface-2) !important; color: var(--fsl-text) !important;
  border: 1px solid var(--fsl-border) !important; border-radius: 8px !important;
  font-size: 15px !important;
}
.fsl-input-card textarea:focus, .fsl-input-card input:focus { border-color: var(--fsl-accent) !important; box-shadow: none !important; }
.fsl-input-card label span { color: var(--fsl-muted) !important; font-size: 13px !important; }

.fsl-threshold .wrap { background: transparent !important; }
.fsl-threshold input[type="range"] { accent-color: var(--fsl-accent) !important; }

.fsl-variant select, .fsl-variant .wrap-inner { background: var(--fsl-surface-2) !important; color: var(--fsl-text) !important; border-color: var(--fsl-border) !important; }
.fsl-variant label span { color: var(--fsl-muted) !important; font-size: 13px !important; }

.fsl-result {
  background: var(--fsl-surface); border: 1px solid var(--fsl-border); border-radius: 12px;
  padding: 20px; min-height: 120px; color: var(--fsl-text);
}
.fsl-result-empty { color: var(--fsl-muted); font-size: 13px; display: flex; align-items: center; justify-content: center; min-height: 160px; }

.fsl-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 18px; }
.fsl-label { font-size: 32px; font-weight: 700; letter-spacing: -0.02em; }
.fsl-label.fsl-yes { color: var(--fsl-yes); }
.fsl-label.fsl-no { color: var(--fsl-no); }
.fsl-label.fsl-unknown { color: var(--fsl-unknown); }
.fsl-conf { color: var(--fsl-muted); font-size: 14px; font-variant-numeric: tabular-nums; }

.fsl-bars { display: flex; flex-direction: column; gap: 10px; }
.fsl-bar-row { display: grid; grid-template-columns: 72px 1fr 56px; gap: 12px; align-items: center; }
.fsl-bar-name { font-size: 13px; color: var(--fsl-muted); text-transform: uppercase; letter-spacing: 0.04em; }
.fsl-bar-track { height: 10px; background: var(--fsl-surface-2); border-radius: 4px; overflow: hidden; }
.fsl-bar-fill { height: 100%; border-radius: 4px; transition: width 0.2s ease; }
.fsl-bar-fill.fsl-yes { background: var(--fsl-yes); }
.fsl-bar-fill.fsl-no { background: var(--fsl-no); }
.fsl-bar-fill.fsl-unknown { background: var(--fsl-unknown); }
.fsl-bar-val { font-size: 13px; font-variant-numeric: tabular-nums; color: var(--fsl-text); text-align: right; }

.fsl-tokens { margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--fsl-border); }
.fsl-tokens-label { font-size: 11px; color: var(--fsl-muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }
.fsl-tokens-list { display: flex; flex-wrap: wrap; gap: 4px; }
.fsl-tok { background: var(--fsl-surface-2); border: 1px solid var(--fsl-border); border-radius: 4px; padding: 2px 6px;
           font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 11px; color: var(--fsl-muted); }
.fsl-tok-empty { color: var(--fsl-muted); font-size: 11px; }

.fsl-meta { margin-top: 12px; color: var(--fsl-muted); font-size: 11px; text-align: right; font-variant-numeric: tabular-nums; }

#fsl-presets-label { color: var(--fsl-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; margin: 8px 0; }
.fsl-presets-row { display: flex !important; flex-wrap: wrap !important; gap: 6px !important; align-items: flex-start !important; }
.fsl-presets-row > * { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
.fsl-presets-row button {
  background: var(--fsl-surface) !important; color: var(--fsl-muted) !important;
  border: 1px solid var(--fsl-border) !important; padding: 6px 10px !important; border-radius: 6px !important;
  font-size: 12px !important; font-weight: normal !important;
  min-width: 0 !important; width: auto !important; height: auto !important;
  white-space: nowrap !important; line-height: 1.2 !important;
  flex: 0 0 auto !important;
}
.fsl-presets-row button:hover { border-color: var(--fsl-accent) !important; color: var(--fsl-text) !important; }

footer, .footer, .built-with-gradio { display: none !important; }
"""

with gr.Blocks(title="ForSureLLM", css=CSS, theme=gr.themes.Base()) as demo:
    gr.HTML("""
    <div id="fsl-header">
      <h1>ForSureLLM</h1>
      <div class="sub">yes / no / unknown classifier · EN + FR · MiniLM-L12 distilled · 113 MB ONNX int8 · ~2 ms CPU</div>
      <div class="links">
        <a href="https://github.com/jcfossati/ForSureLLM" target="_blank">GitHub</a>
        <a href="https://huggingface.co/jcfossati/ForSureLLM" target="_blank">Model</a>
        <span style="color:var(--fsl-muted)">95.2% adversarial · +20.2 pts vs Haiku 4.5</span>
      </div>
    </div>
    """)

    with gr.Column(elem_classes="fsl-input-card"):
        inp = gr.Textbox(
            label="phrase",
            placeholder="tape une phrase… (ex: carrément, laisse tomber, il pleut dehors)",
            lines=1,
            autofocus=True,
            show_label=False,
        )
        with gr.Row():
            variant_dd = gr.Dropdown(
                choices=list(VARIANTS.keys()),
                value=DEFAULT_VARIANT,
                label="model variant",
                elem_classes="fsl-variant",
                scale=2,
            )
            thr = gr.Slider(0, 1, value=0, step=0.01,
                            label="threshold (force unknown si conf < seuil)",
                            elem_classes="fsl-threshold", scale=3)

    gr.HTML('<div id="fsl-presets-label">exemples</div>')
    with gr.Row(elem_classes="fsl-presets-row"):
        preset_btns = [gr.Button(p, size="sm") for p in PRESETS]

    out = gr.HTML(value=EMPTY_HTML)

    inp.change(render_result, inputs=[inp, variant_dd, thr], outputs=[out], show_progress="hidden")
    thr.change(render_result, inputs=[inp, variant_dd, thr], outputs=[out], show_progress="hidden")
    variant_dd.change(render_result, inputs=[inp, variant_dd, thr], outputs=[out], show_progress="hidden")
    for btn, p in zip(preset_btns, PRESETS):
        btn.click(lambda v=p: v, outputs=[inp])


if __name__ == "__main__":
    demo.launch()
