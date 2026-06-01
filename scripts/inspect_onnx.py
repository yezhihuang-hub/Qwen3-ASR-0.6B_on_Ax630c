#!/usr/bin/env python3
"""Print ONNX inputs, outputs, opset, and node type counts."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import onnx


def value_shape(value) -> list[str]:
    shape = []
    tensor_type = value.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(str(dim.dim_value))
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    return shape


def inspect(path: Path) -> None:
    model = onnx.load(path)
    print(f"\n== {path} ==")
    print("ir_version:", model.ir_version)
    print("opsets:", {o.domain or "ai.onnx": o.version for o in model.opset_import})

    print("inputs:")
    for item in model.graph.input:
        print(f"  {item.name}: {value_shape(item)}")

    print("outputs:")
    for item in model.graph.output:
        print(f"  {item.name}: {value_shape(item)}")

    counts = Counter(node.op_type for node in model.graph.node)
    print("node_types:")
    for name, count in counts.most_common():
        print(f"  {name}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.onnx:
        inspect(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
