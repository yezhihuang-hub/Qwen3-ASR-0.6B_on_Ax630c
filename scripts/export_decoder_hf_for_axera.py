#!/usr/bin/env python3
"""Export Qwen3-ASR thinker text decoder as a standard Qwen3 HF directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "chat_template.json",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("models/decoder_hf_axera"))
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    src_weights = args.asr_model_dir / "model.safetensors"
    if not src_weights.is_file():
        raise FileNotFoundError(src_weights)

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    state = {}
    with safe_open(src_weights, framework="pt") as f:
        for key in f.keys():
            if key.startswith("thinker.model."):
                out_key = key.removeprefix("thinker.")
                state[out_key] = f.get_tensor(key).to(dtype).contiguous()
            elif key == "thinker.lm_head.weight":
                state["lm_head.weight"] = f.get_tensor(key).to(dtype).contiguous()

    if "model.embed_tokens.weight" not in state:
        raise RuntimeError("model.embed_tokens.weight was not exported")

    save_file(state, args.out_dir / "model.safetensors")

    full_cfg = json.loads((args.asr_model_dir / "config.json").read_text())
    text_cfg = full_cfg["thinker_config"]["text_config"]
    text_cfg["model_type"] = "qwen3"
    text_cfg["architectures"] = ["Qwen3ForCausalLM"]
    text_cfg["tie_word_embeddings"] = True
    text_cfg["vocab_size"] = full_cfg["thinker_config"].get("vocab_size", text_cfg["vocab_size"])
    text_cfg["torch_dtype"] = args.dtype
    text_cfg["dtype"] = args.dtype
    text_cfg["rope_scaling"] = {
        "rope_type": "default",
        "type": "default",
    }
    for noisy_key in ["audio_config", "vision_config", "audio_token_id"]:
        text_cfg.pop(noisy_key, None)

    (args.out_dir / "config.json").write_text(
        json.dumps(text_cfg, indent=2, ensure_ascii=False) + "\n"
    )

    copied = []
    for name in TOKENIZER_FILES:
        src = args.asr_model_dir / name
        if src.is_file():
            shutil.copy2(src, args.out_dir / name)
            copied.append(name)

    print(f"saved tensors: {len(state)}")
    print(f"saved params: {sum(t.numel() for t in state.values())}")
    print(f"output: {args.out_dir}")
    print(f"copied tokenizer files: {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
