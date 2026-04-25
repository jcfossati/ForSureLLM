"""Deploy le Space Gradio + (optionnellement) le modèle ONNX vers HF.

Usage :
    python tools/deploy_space.py                  # push space/ vers le Space
    python tools/deploy_space.py --with-model     # + push ONNX vers Model repo
    python tools/deploy_space.py --dry-run        # liste sans push

Pré-requis : HF_TOKEN dans .env avec write access sur les deux repos.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(override=True)

SPACE_REPO = "jcfossati/ForSureLLM"
MODEL_REPO = "jcfossati/ForSureLLM"
SPACE_DIR = Path(__file__).resolve().parent.parent / "space"
ONNX_PATH = Path(__file__).resolve().parent.parent / "forsurellm" / "models" / "forsurellm-int8.onnx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-model", action="store_true",
                        help="upload ONNX to the Model repo as well")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN missing from .env")
    api = HfApi(token=token)

    files = sorted(p for p in SPACE_DIR.iterdir() if p.is_file())
    print(f"[space] uploading {len(files)} files to {SPACE_REPO} (Space)")
    for p in files:
        print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")

    if not args.dry_run:
        api.upload_folder(
            folder_path=str(SPACE_DIR),
            repo_id=SPACE_REPO,
            repo_type="space",
            commit_message="sync from GitHub repo (space/)",
        )
        print(f"[space] -> https://huggingface.co/spaces/{SPACE_REPO}")

    if args.with_model:
        if not ONNX_PATH.exists():
            raise SystemExit(f"ONNX missing: {ONNX_PATH}")
        size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
        print(f"\n[model] uploading {ONNX_PATH.name} ({size_mb:.1f} MB) to {MODEL_REPO}")
        if not args.dry_run:
            api.upload_file(
                path_or_fileobj=str(ONNX_PATH),
                path_in_repo=ONNX_PATH.name,
                repo_id=MODEL_REPO,
                repo_type="model",
                commit_message="update ONNX checkpoint",
            )
            print(f"[model] -> https://huggingface.co/{MODEL_REPO}")


if __name__ == "__main__":
    main()
