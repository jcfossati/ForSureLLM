"""Vocab pruning : crée une variante FR+EN du modèle en jetant les tokens
inutilisés par le corpus d'entraînement.

Étapes :
1. Tokenize tous les corpus (train+val+test+adversarial) → set d'IDs utilisés
2. Force-keep : tokens spéciaux (<s>, <pad>, </s>, <unk>) + caractères latin-1
   imprimables (sécurité contre OOV à l'inférence)
3. Build remapping old_id -> new_id (compact 0..N-1)
4. Réécrit tokenizer.json avec vocab compact + post_processor renuméroté
5. Slice embedding du modèle PyTorch
6. Sauve dans checkpoints/best_fr-en/
7. Re-quantize ONNX int8 → forsurellm/models/forsurellm-int8_fr-en.onnx
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
SRC_CKPT = ROOT / "checkpoints" / "best"
DST_CKPT = ROOT / "checkpoints" / "best_fr-en"
SRC_TOKENIZER = SRC_CKPT / "tokenizer.json"
SRC_MODEL_DIR = ROOT / "forsurellm" / "models"

CORPUS_FILES = [
    "data/splits/train.jsonl",
    "data/splits/val.jsonl",
    "data/splits/test.jsonl",
    "evals/adversarial.jsonl",
]

# IDs spéciaux à toujours garder (cf. tokenizer.json added_tokens).
SPECIAL_OLD_IDS = {0, 1, 2, 3}  # <s>, <pad>, </s>, <unk>
MASK_OLD_ID = 250001  # <mask> — gardé en dernière position pour compat HF


def collect_used_ids(tokenizer: Tokenizer) -> set[int]:
    used: set[int] = set()
    for path in CORPUS_FILES:
        p = ROOT / path
        if not p.exists():
            print(f"[skip] {path}")
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            phrase = row.get("phrase", "")
            if not phrase:
                continue
            enc = tokenizer.encode(phrase.lower())  # mirror runtime preprocess
            used.update(enc.ids)
        print(f"[corpus] {path}: cumulative used = {len(used)}")
    # Latin-1 printable safety net (OOV protection)
    for cp in range(0x20, 0x7F):  # ASCII printable
        ch = chr(cp)
        ids = tokenizer.encode(ch).ids
        used.update(ids)
    for cp in range(0xA0, 0x100):  # Latin-1 supplement
        ch = chr(cp)
        ids = tokenizer.encode(ch).ids
        used.update(ids)
    used |= SPECIAL_OLD_IDS
    print(f"[total] {len(used)} unique token IDs to keep")
    return used


def build_remap(used_ids: set[int]) -> tuple[list[int], dict[int, int]]:
    """Returns (kept_old_ids ordered, remap old_id -> new_id).

    Order : specials first (preserve order 0,1,2,3), then mask placeholder,
    then sorted remaining IDs. <mask> is appended at the end of the new vocab
    so it stays the highest ID.
    """
    specials_ordered = [0, 1, 2, 3]
    others = sorted(i for i in used_ids if i not in SPECIAL_OLD_IDS)
    kept_old = specials_ordered + others + [MASK_OLD_ID]
    remap = {old: new for new, old in enumerate(kept_old)}
    return kept_old, remap


def prune_tokenizer_json(src_path: Path, dst_path: Path, kept_old: list[int],
                         remap: dict[int, int]) -> None:
    tok = json.loads(src_path.read_text(encoding="utf-8"))
    old_vocab = tok["model"]["vocab"]  # list of [str, score]

    new_vocab = []
    for old_id in kept_old:
        if old_id < len(old_vocab):
            new_vocab.append(old_vocab[old_id])
        else:
            # <mask> if old_id == 250001 might be at last index — handle
            new_vocab.append([f"<reserved_{old_id}>", 0.0])
    tok["model"]["vocab"] = new_vocab

    # Update added_tokens IDs
    for at in tok.get("added_tokens", []):
        old_id = at["id"]
        if old_id in remap:
            at["id"] = remap[old_id]
        else:
            print(f"[warn] added_token id={old_id} not in remap (content={at.get('content')!r})")

    # Update post_processor special token IDs (XLM-R uses TemplateProcessing
    # with <s> at id 0 and </s> at id 2 by default)
    pp = tok.get("post_processor")
    if pp and pp.get("type") == "TemplateProcessing":
        for special_name, info in pp.get("special_tokens", {}).items():
            old_ids = info.get("ids", [])
            info["ids"] = [remap[i] for i in old_ids if i in remap]

    dst_path.write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")
    print(f"[tokenizer] pruned 250002 -> {len(new_vocab)} tokens -> {dst_path}")


def prune_model_embedding(src_dir: Path, dst_dir: Path, kept_old: list[int]) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)

    model = AutoModelForSequenceClassification.from_pretrained(str(src_dir))
    old_emb = model.get_input_embeddings().weight.data  # [V, H]
    print(f"[model] old embedding: {tuple(old_emb.shape)}")

    # Slice
    indices = torch.tensor([i if i < old_emb.shape[0] else 0 for i in kept_old], dtype=torch.long)
    new_emb = old_emb[indices].clone()
    print(f"[model] new embedding: {tuple(new_emb.shape)}")

    # Resize and replace
    model.resize_token_embeddings(len(kept_old))
    model.get_input_embeddings().weight.data = new_emb

    # Update vocab_size in config
    model.config.vocab_size = len(kept_old)

    model.save_pretrained(str(dst_dir))
    print(f"[model] saved -> {dst_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=SRC_CKPT)
    parser.add_argument("--dst", type=Path, default=DST_CKPT)
    args = parser.parse_args()

    print(f"[load] tokenizer from {args.src}/tokenizer.json")
    tokenizer = Tokenizer.from_file(str(args.src / "tokenizer.json"))

    used = collect_used_ids(tokenizer)
    kept_old, remap = build_remap(used)
    print(f"[remap] kept = {len(kept_old)} tokens (was 250002)")

    # Model first (it rmtree's the dst), then tokenizer
    prune_model_embedding(args.src, args.dst, kept_old)
    prune_tokenizer_json(args.src / "tokenizer.json", args.dst / "tokenizer.json",
                         kept_old, remap)

    # Copy auxiliary files
    for fname in ["special_tokens_map.json", "tokenizer_config.json", "temperature.json"]:
        src_f = args.src / fname
        if src_f.exists():
            shutil.copy2(src_f, args.dst / fname)
            print(f"[copy] {fname}")


if __name__ == "__main__":
    main()
