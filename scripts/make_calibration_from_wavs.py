#!/usr/bin/env python3
"""Create Pulsar2-style Numpy calibration tar for Qwen3-ASR encoder input."""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]


def load_audio(path: Path, sr: int) -> np.ndarray:
    audio, in_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio[:, 0]
    if in_sr != sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=in_sr, target_sr=sr)
    return audio.astype(np.float32, copy=False)


def mel_features(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    import librosa

    stft = librosa.stft(audio, n_fft=400, hop_length=160, window="hann", center=True)
    magnitudes = np.abs(stft) ** 2
    mel_spec = np.dot(mel_filters.T, magnitudes)
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    n_frames = audio.shape[-1] // 160
    return log_spec[:, :n_frames].astype(np.float32)


def normalize_audio(audio: np.ndarray, samples: int) -> np.ndarray:
    if audio.shape[0] >= samples:
        return audio[:samples]
    out = np.zeros(samples, dtype=np.float32)
    out[: audio.shape[0]] = audio
    return out


def write_tar(arrays: list[np.ndarray], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        names = []
        for i, arr in enumerate(arrays):
            name = f"input_features_{i:04d}.npy"
            np.save(tmp / name, arr)
            names.append(name)

        with tarfile.open(output, "w") as tar:
            for name in names:
                tar.add(tmp / name, arcname=name)


def smoke_arrays(count: int, frames: int) -> list[np.ndarray]:
    rng = np.random.default_rng(20260601)
    arrays = []
    for _ in range(count):
        noise = rng.normal(loc=0.0, scale=0.03, size=(1, 128, frames))
        arrays.append(noise.astype(np.float32))
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--mel-filters", type=Path, default=ROOT / "models" / "mel_filters.npy")
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fallback-smoke",
        action="store_true",
        help="Create deterministic random calibration input when no wavs are found.",
    )
    args = parser.parse_args()

    frames = args.seconds * 100
    arrays: list[np.ndarray] = []

    if args.audio_dir:
        wavs = sorted(
            p for p in args.audio_dir.rglob("*") if p.suffix.lower() in {".wav", ".flac"}
        )[: args.limit]
        if wavs:
            mel_filters = np.load(args.mel_filters)
            samples = args.seconds * args.sample_rate
            for wav in wavs:
                audio = normalize_audio(load_audio(wav, args.sample_rate), samples)
                feats = mel_features(audio, mel_filters)
                if feats.shape[1] < frames:
                    pad = np.zeros((128, frames), dtype=np.float32)
                    pad[:, : feats.shape[1]] = feats
                    feats = pad
                arrays.append(feats[:, :frames][None, :, :].astype(np.float32))

    if not arrays and args.fallback_smoke:
        arrays = smoke_arrays(args.limit, frames)

    if not arrays:
        raise SystemExit("No calibration wavs found. Pass --fallback-smoke for smoke data.")

    write_tar(arrays, args.output)
    print(f"Wrote {args.output} with {len(arrays)} arrays shaped {arrays[0].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
