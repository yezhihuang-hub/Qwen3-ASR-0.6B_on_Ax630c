#!/usr/bin/env python3
"""Prepare reusable local assets for the AX630C Qwen3-ASR workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RV_MODELS = Path(
    "/home/ye/Projects/qwen3_asr/asr_rv1126b/deploy/models"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_asset(src: Path, dst: Path, force: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source": str(src),
        "destination": str(dst),
        "exists": src.is_file(),
    }
    if not src.is_file():
        record["status"] = "missing"
        return record

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        record["status"] = "kept_existing"
    else:
        shutil.copy2(src, dst)
        record["status"] = "copied"

    record["bytes"] = dst.stat().st_size
    record["sha256"] = sha256(dst)
    return record


def write_bf16_embedding(src_npy: Path, dst_bin: Path, force: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source": str(src_npy),
        "destination": str(dst_bin),
        "exists": src_npy.is_file(),
        "format": "torch.bfloat16 raw little-endian uint16 payload",
    }
    if not src_npy.is_file():
        record["status"] = "missing"
        return record
    if dst_bin.exists() and not force:
        record["status"] = "kept_existing"
        record["bytes"] = dst_bin.stat().st_size
        record["sha256"] = sha256(dst_bin)
        return record

    import torch

    arr = np.load(src_npy)
    tensor = torch.from_numpy(arr.astype(np.float32, copy=False)).to(torch.bfloat16)
    raw = tensor.view(torch.uint16).cpu().numpy()
    dst_bin.parent.mkdir(parents=True, exist_ok=True)
    raw.tofile(dst_bin)

    record["status"] = "created_from_npy"
    record["source_dtype"] = str(arr.dtype)
    record["shape"] = list(arr.shape)
    record["bytes"] = dst_bin.stat().st_size
    record["sha256"] = sha256(dst_bin)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rv-models", type=Path, default=DEFAULT_RV_MODELS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--make-bf16-bin", action="store_true")
    args = parser.parse_args()

    rv = args.rv_models
    assets = [
        (rv / "mel_filters.npy", ROOT / "models" / "mel_filters.npy"),
        (
            rv / "embed_tokens.npy",
            ROOT / "models" / "embeddings" / "embed_tokens.fp16.npy",
        ),
        (
            rv / "tokenizer" / "tokenizer.json",
            ROOT / "models" / "tokenizer" / "tokenizer.json",
        ),
        (rv / "vad" / "silero_vad.onnx", ROOT / "models" / "vad" / "silero_vad.onnx"),
        (rv / "data_quant.json", ROOT / "models" / "data_quant.rv1126b.json"),
    ]

    manifest: dict[str, Any] = {
        "workspace": str(ROOT),
        "rv_models": str(rv),
        "assets": [copy_asset(src, dst, args.force) for src, dst in assets],
    }

    if args.make_bf16_bin:
        manifest["bf16_embedding"] = write_bf16_embedding(
            ROOT / "models" / "embeddings" / "embed_tokens.fp16.npy",
            ROOT
            / "models"
            / "embeddings"
            / "model.embed_tokens.weight.bfloat16.bin",
            args.force,
        )

    out = ROOT / "reports" / "local_assets_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
