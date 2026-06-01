#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_HARDWARE="${AXERA_TARGET_HARDWARE:-AX620E}"
NPU_MODE="${AXERA_NPU_MODE:-NPU2}"
OUT_DIR="${ROOT}/axmodel"
LOCAL_PULSAR2="${ROOT}/third_party/pulsar2/6.0/extracted/ax_pulsar2_6.0_lite_package/bin/pulsar2"

if [[ -x "${LOCAL_PULSAR2}" ]]; then
  PULSAR2="${LOCAL_PULSAR2}"
elif command -v pulsar2 >/dev/null 2>&1; then
  PULSAR2="$(command -v pulsar2)"
else
  cat >&2 <<EOF
pulsar2 was not found.

Install/source the AXERA Pulsar2 SDK first, then rerun:
  cd ${ROOT}
  AXERA_TARGET_HARDWARE=${TARGET_HARDWARE} AXERA_NPU_MODE=${NPU_MODE} bash scripts/build_ax630c_models.sh
EOF
  exit 127
fi

mkdir -p "${OUT_DIR}" "${ROOT}/logs"
failures=0

build_one() {
  local input="$1"
  local config="$2"
  local name="$3"

  if [[ ! -f "${input}" ]]; then
    echo "skip ${name}: missing ${input}" >&2
    return 0
  fi

  set +e
  "${PULSAR2}" build \
    --input "${input}" \
    --config "${config}" \
    --output_dir "${OUT_DIR}/${name}" \
    --target_hardware "${TARGET_HARDWARE}" \
    --npu_mode "${NPU_MODE}" \
    2>&1 | tee "${ROOT}/logs/${name}.pulsar2.log"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "FAILED ${name}, see ${ROOT}/logs/${name}.pulsar2.log" >&2
    failures=$((failures + 1))
  fi
}

build_one \
  "${ROOT}/onnx/qwen3_asr_conv_frontend.5s.rv_chunk.pos_backend.float32.onnx" \
  "${ROOT}/configs/ax630c/conv_frontend_config.json" \
  "conv_frontend_5s_pos_backend"

build_one \
  "${ROOT}/onnx/qwen3_asr_encoder_backend.5s.rv_chunk.pos_backend.float32.onnx" \
  "${ROOT}/configs/ax630c/encoder_backend_config.json" \
  "encoder_backend_5s_pos_backend"

if [[ "${BUILD_MERGED:-0}" == "1" ]]; then
  build_one \
    "${ROOT}/onnx/qwen3_asr_encoder_merged.5s.rv_chunk.pos_backend.float32.onnx" \
    "${ROOT}/configs/ax630c/merged_encoder_config.json" \
    "encoder_merged_5s_pos_backend"
fi

cat <<EOF

Pulsar2 build wrapper finished.
Target hardware: ${TARGET_HARDWARE}
NPU mode:        ${NPU_MODE}
Output dir:      ${OUT_DIR}
EOF

if [[ "${failures}" -ne 0 ]]; then
  echo "Pulsar2 build finished with ${failures} failure(s)." >&2
  exit 1
fi
