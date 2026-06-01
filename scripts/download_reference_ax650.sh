#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/third_party/reference_ax650"
REPO="AXERA-TECH/Qwen3-ASR-0.6B-AX650-C64-P448-CTX2047"

mkdir -p "${OUT}"

root_files=(
  "README.md"
  "config.json"
  "post_config.json"
  "qwen3_tokenizer.txt"
  "silero_vad.onnx"
)

huggingface-cli download "${REPO}" \
  "${root_files[@]}" \
  --local-dir "${OUT}" \
  --local-dir-use-symlinks False

includes=(
  "tokenizer/*"
)

if [[ "${1:-}" == "--with-axmodels" ]]; then
  includes+=(
    "*.axmodel"
    "*.bin"
  )
fi

args=()
for item in "${includes[@]}"; do
  args+=(--include "${item}")
done

huggingface-cli download "${REPO}" \
  --local-dir "${OUT}" \
  --local-dir-use-symlinks False \
  "${args[@]}"

find "${OUT}" -maxdepth 2 -type f -printf "%P\t%k KB\n" | sort
