#!/usr/bin/env python3
"""Export fixed-length Qwen3-ASR audio encoder ONNX candidates.

The merged export follows the working RV1126B baseline. The split exports are
prepared for AXERA-style conversion and must be validated before SDK compiling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen_asr import Qwen3ASRModel


def aftercnn_len(n: int) -> int:
    for _ in range(3):
        n = (n - 1) // 2 + 1
    return n


def make_cu_seqlens(chunks: int, tokens_per_chunk: int, policy: str) -> torch.Tensor:
    total = chunks * tokens_per_chunk
    if policy == "rv_chunk":
        return torch.arange(0, total + 1, tokens_per_chunk, dtype=torch.int32)
    if policy == "full":
        return torch.tensor([0, total], dtype=torch.int32)
    raise ValueError(f"Unsupported cu_seqlens policy: {policy}")


class FixedConvFrontend(torch.nn.Module):
    def __init__(self, audio_tower: torch.nn.Module, seconds: int, add_position: bool):
        super().__init__()
        self.audio_tower = audio_tower
        self.add_position = add_position
        self.frames = seconds * 100
        self.chunk_frames = int(audio_tower.config.n_window) * 2
        if self.frames % self.chunk_frames != 0:
            raise ValueError(
                f"frames={self.frames} must be divisible by chunk_frames={self.chunk_frames}"
            )
        self.chunks = self.frames // self.chunk_frames

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # input_features: [1, 128, frames], matching the RV1126B baseline.
        x = input_features.squeeze(0).transpose(0, 1)
        x = x.reshape(self.chunks, self.chunk_frames, 128)
        x = x.transpose(1, 2).unsqueeze(1)

        x = F.gelu(self.audio_tower.conv2d1(x))
        x = F.gelu(self.audio_tower.conv2d2(x))
        x = F.gelu(self.audio_tower.conv2d3(x))

        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x = self.audio_tower.conv_out(x)

        if self.add_position:
            pos = self.audio_tower.positional_embedding.positional_embedding[: x.shape[1], :]
            x = x + pos.unsqueeze(0).to(x.dtype)
        return x.reshape(1, self.chunks * x.shape[1], x.shape[2])


class FixedEncoderBackend(torch.nn.Module):
    def __init__(
        self,
        audio_tower: torch.nn.Module,
        seconds: int,
        cu_policy: str,
        add_position: bool,
    ):
        super().__init__()
        self.audio_tower = audio_tower
        self.add_position = add_position
        frames = seconds * 100
        chunk_frames = int(audio_tower.config.n_window) * 2
        if frames % chunk_frames != 0:
            raise ValueError(
                f"frames={frames} must be divisible by chunk_frames={chunk_frames}"
            )
        chunks = frames // chunk_frames
        tokens_per_chunk = aftercnn_len(chunk_frames)
        self.chunks = chunks
        self.tokens_per_chunk = tokens_per_chunk
        self.register_buffer(
            "cu_seqlens", make_cu_seqlens(chunks, tokens_per_chunk, cu_policy)
        )

    def forward(self, conv_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = conv_embeds.squeeze(0)
        if self.add_position:
            hidden_states = hidden_states.reshape(
                self.chunks, self.tokens_per_chunk, hidden_states.shape[-1]
            )
            pos = self.audio_tower.positional_embedding.positional_embedding[: hidden_states.shape[1], :]
            hidden_states = hidden_states + pos.unsqueeze(0).to(hidden_states.dtype)
            hidden_states = hidden_states.reshape(
                self.chunks * self.tokens_per_chunk, hidden_states.shape[-1]
            )

        for layer in self.audio_tower.layers:
            hidden_states = layer(hidden_states, self.cu_seqlens)[0]

        hidden_states = self.audio_tower.ln_post(hidden_states)
        hidden_states = self.audio_tower.proj1(hidden_states)
        hidden_states = self.audio_tower.act(hidden_states)
        hidden_states = self.audio_tower.proj2(hidden_states)
        return hidden_states.unsqueeze(0)


class FixedMergedEncoder(torch.nn.Module):
    def __init__(
        self,
        audio_tower: torch.nn.Module,
        seconds: int,
        cu_policy: str,
        position_placement: str,
    ):
        super().__init__()
        self.frontend = FixedConvFrontend(
            audio_tower, seconds, add_position=position_placement == "frontend"
        )
        self.backend = FixedEncoderBackend(
            audio_tower,
            seconds,
            cu_policy,
            add_position=position_placement == "backend",
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.backend(self.frontend(input_features))


def export_model(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], path: Path, input_names, output_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        y = model(*inputs)
        print(f"{path.name}: test output shape={tuple(y.shape)} dtype={y.dtype}")

    torch.onnx.export(
        model,
        inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=13,
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        dynamo=False,
        verbose=False,
    )
    print(f"Exported {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("onnx"))
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument(
        "--cu-seqlens-policy",
        choices=["rv_chunk", "full"],
        default="rv_chunk",
        help="rv_chunk matches the previous RV1126B fixed export; full is an official-style short-window candidate.",
    )
    parser.add_argument(
        "--position-placement",
        choices=["frontend", "backend"],
        default="frontend",
        help="Put audio positional embedding in conv_frontend or encoder_backend.",
    )
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    model = Qwen3ASRModel.from_pretrained(
        str(args.model_dir),
        dtype=dtype,
        device_map="cpu",
        max_new_tokens=8,
    )
    audio_tower = model.model.thinker.audio_tower.eval()
    audio_tower.config._attn_implementation = "eager"
    for layer in audio_tower.layers:
        layer.self_attn.config._attn_implementation = "eager"

    frames = args.seconds * 100
    chunk_frames = int(audio_tower.config.n_window) * 2
    tokens_per_chunk = aftercnn_len(chunk_frames)
    chunks = frames // chunk_frames
    tokens = chunks * tokens_per_chunk
    print(
        "export config:",
        {
            "seconds": args.seconds,
            "frames": frames,
            "chunk_frames": chunk_frames,
            "chunks": chunks,
            "tokens_per_chunk": tokens_per_chunk,
            "tokens": tokens,
            "cu_seqlens_policy": args.cu_seqlens_policy,
            "position_placement": args.position_placement,
            "dtype": str(dtype),
        },
    )

    dummy = torch.randn(1, 128, frames, dtype=dtype)

    frontend = FixedConvFrontend(
        audio_tower, args.seconds, add_position=args.position_placement == "frontend"
    ).eval()
    conv_dummy = frontend(dummy)
    backend = FixedEncoderBackend(
        audio_tower,
        args.seconds,
        args.cu_seqlens_policy,
        add_position=args.position_placement == "backend",
    ).eval()
    merged = FixedMergedEncoder(
        audio_tower, args.seconds, args.cu_seqlens_policy, args.position_placement
    ).eval()

    tag = f"{args.seconds}s.{args.cu_seqlens_policy}.pos_{args.position_placement}.{args.dtype}"
    export_model(
        frontend,
        (dummy,),
        args.out_dir / f"qwen3_asr_conv_frontend.{tag}.onnx",
        ["input_features"],
        ["conv_embeds"],
    )
    export_model(
        backend,
        (conv_dummy,),
        args.out_dir / f"qwen3_asr_encoder_backend.{tag}.onnx",
        ["conv_embeds"],
        ["audio_embeds"],
    )
    export_model(
        merged,
        (dummy,),
        args.out_dir / f"qwen3_asr_encoder_merged.{tag}.onnx",
        ["input_features"],
        ["audio_embeds"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
