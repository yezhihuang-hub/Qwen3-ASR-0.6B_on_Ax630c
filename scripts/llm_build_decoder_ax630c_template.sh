#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_DIR="${1:-${ROOT}/models/decoder_hf_axera}"
OUT_DIR="${ROOT}/axmodel/decoder_llm"
CHIP="${AXERA_LLM_CHIP:-AX620E}"
NPU_MODE="${AXERA_LLM_NPU_MODE:-NPU2}"
PARALLEL="${AXERA_LLM_PARALLEL:-4}"
PULSAR2="${ROOT}/third_party/pulsar2/6.0/extracted/ax_pulsar2_6.0_lite_package/bin/pulsar2"

if [[ ! -x "${PULSAR2}" ]]; then
  echo "pulsar2 was not found at ${PULSAR2}. Install/extract Pulsar2 first." >&2
  exit 127
fi

mkdir -p "${OUT_DIR}" "${ROOT}/logs"

# The AX650 reference uses prefill_len=64 naming; keep kv/prefill lengths
# aligned with the board runtime contract.
"${PULSAR2}" llm_build \
  --input_path "${HF_DIR}" \
  --output_path "${OUT_DIR}" \
  --chip "${CHIP}" \
  --npu_mode "${NPU_MODE}" \
  --hidden_state_type bf16 \
  --kv_cache_len 2047 \
  --prefill_len 64 \
  --parallel "${PARALLEL}" \
  -w s8 \
  2>&1 | tee "${ROOT}/logs/decoder_llm_build.log"
