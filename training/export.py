"""Export ONNX + quantization int8 dynamique + benchmark CPU.

Usage:
    python scripts/export.py --src checkpoints/best

Entrée : checkpoint HuggingFace (AutoModelForSequenceClassification).
Sortie :
    forsurellm/models/forsurellm-int8.onnx
    forsurellm/models/tokenizer.json
    forsurellm/models/config.json   (id2label, max_length)
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

CLASSES = ["yes", "no", "unknown"]


def export_and_quantize(src: Path, work_dir: Path) -> Path:
    fp32_dir = work_dir / "fp32"
    int8_dir = work_dir / "int8"
    for d in (fp32_dir, int8_dir):
        if d.exists():
            shutil.rmtree(d)

    print(f"[export] {src} -> {fp32_dir}")
    model = ORTModelForSequenceClassification.from_pretrained(str(src), export=True)
    model.save_pretrained(str(fp32_dir))
    # Prefer the source dir's tokenizer (handles pruned variants); fallback to base.
    src_tok = src / "tokenizer.json"
    if src_tok.exists():
        shutil.copy2(src_tok, fp32_dir / "tokenizer.json")
        for aux in ("special_tokens_map.json", "tokenizer_config.json"):
            if (src / aux).exists():
                shutil.copy2(src / aux, fp32_dir / aux)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        tokenizer.save_pretrained(str(fp32_dir))

    print(f"[quantize int8 dynamic] -> {int8_dir}")
    quantizer = ORTQuantizer.from_pretrained(str(fp32_dir))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(int8_dir), quantization_config=qconfig)
    return int8_dir


def pick_onnx(dir_: Path) -> Path:
    for name in ("model_quantized.onnx", "model.onnx"):
        p = dir_ / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Aucun .onnx dans {dir_}")


def benchmark(onnx_path: Path, tokenizer_path_or_dir: Path, n_runs: int = 200) -> dict:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if tokenizer_path_or_dir.is_file():
        tok = Tokenizer.from_file(str(tokenizer_path_or_dir))
        enc = tok.encode("carrément d'accord")
        feeds = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
        }
        feeds = {k: v for k, v in feeds.items() if k in {i.name for i in sess.get_inputs()}}
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path_or_dir))
        sample = "carrément d'accord"
        enc = tokenizer(sample, return_tensors="np", truncation=True, max_length=64)
        feeds = {k: v for k, v in enc.items() if k in {i.name for i in sess.get_inputs()}}

    for _ in range(20):
        sess.run(None, feeds)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    return {
        "size_mb": round(size_mb, 2),
        "mean_ms": round(float(np.mean(times)), 3),
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "p99_ms": round(float(np.percentile(times, 99)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("checkpoints/best"))
    parser.add_argument("--work-dir", type=Path, default=Path("checkpoints/onnx"))
    parser.add_argument("--out-dir", type=Path, default=Path("forsurellm/models"))
    parser.add_argument("--out-suffix", type=str, default="",
                        help="suffix appended to filenames, e.g. '_fr-en'")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    int8_dir = export_and_quantize(args.src, args.work_dir)
    onnx_path = pick_onnx(int8_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.out_suffix
    final_onnx = args.out_dir / f"forsurellm-int8{suffix}.onnx"
    shutil.copy(onnx_path, final_onnx)

    # Copy tokenizer files (use source dir if it has tokenizer.json — pruned case)
    src_tok = args.src / "tokenizer.json"
    if src_tok.exists():
        shutil.copy2(src_tok, args.out_dir / f"tokenizer{suffix}.json")
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(int8_dir))
        tokenizer.save_pretrained(str(args.out_dir))

    temperature = 1.0
    temp_file = args.src / "temperature.json"
    if temp_file.exists():
        temperature = json.loads(temp_file.read_text())["temperature"]
        print(f"[calibration] T={temperature:.3f} appliqué")

    config = {"classes": CLASSES, "max_length": args.max_length, "temperature": temperature,
              "tokenizer_file": f"tokenizer{suffix}.json"}
    with (args.out_dir / f"config{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n[benchmark]")
    stats = benchmark(final_onnx, args.out_dir / f"tokenizer{suffix}.json" if src_tok.exists() else args.out_dir)
    print(json.dumps(stats, indent=2))

    with (args.out_dir / f"benchmark{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[done] modèle prêt -> {final_onnx}")


if __name__ == "__main__":
    main()
