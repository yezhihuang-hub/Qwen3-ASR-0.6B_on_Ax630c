#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DECODER_DIR="${ROOT}/axmodel/decoder_llm"
DEPLOY_DIR="${ROOT}/deploy_ax630c"

mkdir -p "${DEPLOY_DIR}" "${DEPLOY_DIR}/models"

for i in $(seq 0 27); do
  src="${DECODER_DIR}/qwen3_p64_l${i}_together.axmodel"
  dst="${DEPLOY_DIR}/qwen3_asr_p64_l${i}_together.axmodel"
  if [[ ! -f "${src}" ]]; then
    echo "missing decoder layer: ${src}" >&2
    exit 1
  fi
  cp -f "${src}" "${dst}"
done

if [[ ! -f "${DECODER_DIR}/qwen3_post.axmodel" ]]; then
  echo "missing decoder post model: ${DECODER_DIR}/qwen3_post.axmodel" >&2
  exit 1
fi
cp -f "${DECODER_DIR}/qwen3_post.axmodel" "${DEPLOY_DIR}/qwen3_asr_post.axmodel"

if [[ -f "${DEPLOY_DIR}/models/conv_frontend_5s.axmodel" ]]; then
  cp -f "${DEPLOY_DIR}/models/conv_frontend_5s.axmodel" "${DEPLOY_DIR}/conv_frontend.axmodel"
fi

if [[ -f "${DEPLOY_DIR}/models/encoder_backend_5s.axmodel" ]]; then
  cp -f "${DEPLOY_DIR}/models/encoder_backend_5s.axmodel" "${DEPLOY_DIR}/encoder.axmodel"
fi

find "${DEPLOY_DIR}" -maxdepth 1 -type f -name '*.axmodel' -printf '%f\t%k KB\n' | sort -V
