# Qwen3-ASR-0.6B AX630C Port Workspace

This workspace prepares the Qwen3-ASR-0.6B migration from the RV1126B project
to the AX630C direction. It now includes AX620E/NPU2 Pulsar2 build outputs for
the split audio encoder and Qwen3 decoder-layer models.

## Current Status

- Target board: AX630C / AXERA Pi 2.
- Local SDK status: Pulsar2 6.0 lite package installed under
  `third_party/pulsar2/6.0/extracted/ax_pulsar2_6.0_lite_package`.
- Existing reusable assets found in the RV1126B project:
  - `mel_filters.npy`
  - `embed_tokens.npy`
  - `tokenizer/tokenizer.json`
  - `vad/silero_vad.onnx`
- Full `Qwen/Qwen3-ASR-0.6B` Hugging Face checkpoint was not found locally when
  this workspace was created. It is now available at:
  `/home/ye/Projects/ASX/qwen3-ASR-0.6b`.
- AXERA reference model:
  `AXERA-TECH/Qwen3-ASR-0.6B-AX650-C64-P448-CTX2047`
- Built AX630C-direction deploy package:
  `deploy_ax630c/`
- Large deploy binaries are published through the GitHub Release asset described
  in `docs/DEPLOY_ARTIFACTS.md`, not committed into Git history.

## Layout

```text
qwen3_asr_ax630c_port/
├── configs/ax630c/          # Pulsar2 config templates
├── docs/                    # Local AX630C docs and migration notes
├── models/                  # Reused local assets and downloaded HF model
├── onnx/                    # Exported ONNX candidates
├── axmodel/                 # SDK build output target
├── calibration/             # Numpy calibration tar files
├── scripts/                 # Reproducible prep/export/build helpers
├── reports/                 # Manifests and validation reports
└── third_party/             # Reference metadata or downloaded examples
```

## Recommended Flow

Use the existing `qwen_asr` conda environment unless a later SDK package requires
a dedicated environment.

```bash
cd /home/ye/Projects/ASX/qwen3_asr_ax630c_port

# 1) Copy reusable assets from the RV1126B delivery.
conda run -n qwen_asr python scripts/prepare_assets.py

# Optional: also create a bfloat16 embedding bin for AXERA-style runtimes.
conda run -n qwen_asr python scripts/prepare_assets.py --make-bf16-bin

# 2) Download the small metadata files from the AX650 reference repo.
bash scripts/download_reference_ax650.sh

# 3) Download the full upstream HF checkpoint if it is not already present.
huggingface-cli download Qwen/Qwen3-ASR-0.6B \
  --local-dir models/hf/Qwen3-ASR-0.6B

# 4) Export ONNX candidates from the HF checkpoint.
conda run -n qwen_asr python scripts/export_qwen3_asr_encoder_onnx.py \
  --model-dir /home/ye/Projects/ASX/qwen3-ASR-0.6b \
  --seconds 5 \
  --out-dir onnx \
  --position-placement backend

# 5) Build calibration data from 16 kHz wavs or deterministic smoke data.
conda run -n qwen_asr python scripts/make_calibration_from_wavs.py \
  --seconds 5 \
  --output calibration/input_features_5s.tar \
  --fallback-smoke

# 6) Build AX620E/NPU2 split encoder axmodels.
bash scripts/build_ax630c_models.sh

# 7) Export the Qwen3 decoder as a standard HF text model for llm_build.
conda run -n qwen_asr python scripts/export_decoder_hf_for_axera.py \
  --source-dir /home/ye/Projects/ASX/qwen3-ASR-0.6b \
  --out-dir models/decoder_hf_axera

# 8) Build decoder layers and post model.
AXERA_LLM_PARALLEL=4 bash scripts/llm_build_decoder_ax630c_template.sh

# 9) Assemble AXERA-style deploy file names.
bash scripts/assemble_deploy_ax630c.sh
```

## Build Boundary

The split audio encoder and decoder now compile to AX620E/NPU2 `.axmodel`
files. This workspace prepares:

- ONNX export scripts for the Qwen3-ASR audio encoder path.
- Calibration dataset builders.
- Pulsar2 JSON config templates for AX630C/AX620E-style NPU2 builds.
- A Qwen3 decoder HF export at `models/decoder_hf_axera`.
- An AXERA-style deploy package at `deploy_ax630c`.

Current deploy contents include `conv_frontend.axmodel`, `encoder.axmodel`,
`qwen3_asr_p64_l0_together.axmodel` through
`qwen3_asr_p64_l27_together.axmodel`, `qwen3_asr_post.axmodel`, tokenizer,
embedding weights, `config.json`, and `post_config.json`.

The repository tracks source scripts, configs, docs, reports, tokenizer, and
small calibration/reference assets. Large generated `.axmodel`, `.onnx`, and
checkpoint files are intentionally excluded from Git and should be downloaded
from the release asset or regenerated locally.

See `docs/MIGRATION_NOTES.md` for the decisions and open items. See
`docs/RUNTIME_CHAIN_AX620E.md` for how the many `.axmodel` files map to the
AX620E/AX630C board-side runtime compile path.
