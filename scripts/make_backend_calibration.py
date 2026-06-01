#!/usr/bin/env python3
"""Generate encoder-backend calibration tensors from conv_frontend ONNX."""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort


def load_input_tar(path: Path) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    with tarfile.open(path, "r") as tar, tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tar.extractall(tmp)
        for item in sorted(tmp.rglob("*.npy")):
            arrays.append(np.load(item).astype(np.float32))
    if not arrays:
        raise RuntimeError(f"No .npy files found in {path}")
    return arrays


def write_tar(arrays: list[np.ndarray], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w") as tar, tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i, arr in enumerate(arrays):
            name = f"conv_embeds_{i:04d}.npy"
            np.save(tmp / name, arr.astype(np.float32, copy=False))
            tar.add(tmp / name, arcname=name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-onnx", type=Path, required=True)
    parser.add_argument("--input-tar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sess = ort.InferenceSession(str(args.frontend_onnx), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    outputs = []
    for arr in load_input_tar(args.input_tar):
        outputs.append(sess.run([output_name], {input_name: arr})[0])

    write_tar(outputs, args.output)
    print(f"Wrote {args.output} with {len(outputs)} arrays shaped {outputs[0].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
