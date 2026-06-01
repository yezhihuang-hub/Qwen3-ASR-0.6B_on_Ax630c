# Handoff: Next SDK Compile Step

Last updated: 2026-06-01

This file is the entry point for the next session. Read this first, then
continue from the SDK/runtime compile step. Do not restart the model conversion
work unless the calibration or model split needs to change.

## Repository And Local Paths

GitHub repository:

```text
https://github.com/yezhihuang-hub/Qwen3-ASR-0.6B_on_Ax630c
```

Local working directory:

```text
/home/ye/Projects/ASX/qwen3_asr_ax630c_port
```

Original full Qwen3-ASR checkpoint:

```text
/home/ye/Projects/ASX/qwen3-ASR-0.6b
```

AX650 reference files downloaded locally:

```text
/home/ye/Projects/ASX/ax650
/home/ye/Projects/ASX/qwen3_asr_ax630c_port/third_party/reference_ax650
```

Local deploy archive prepared for release/manual upload:

```text
/home/ye/Projects/ASX/qwen3_asr_ax630c_deploy_20260601.tar.zst
/home/ye/Projects/ASX/qwen3_asr_ax630c_deploy_20260601.tar.zst.sha256
```

Release URL:

```text
https://github.com/yezhihuang-hub/Qwen3-ASR-0.6B_on_Ax630c/releases/tag/v0.1.0-ax630c-deploy-20260601
```

Status of release assets on 2026-06-01:

- The small `.sha256` asset is uploaded.
- The 944 MB `.tar.zst` deploy archive was not uploaded successfully from this
  machine. Upload it manually to the release page if needed.

## What Has Already Been Done

Model-side work is done to the current "before board SDK/runtime" boundary:

- Read the local AXERA algorithm deployment PDF and AXERA reference layout.
- Installed Pulsar2 6.0 lite locally:

```text
third_party/pulsar2/6.0/extracted/ax_pulsar2_6.0_lite_package/bin/pulsar2
```

- Exported Qwen3-ASR encoder ONNX candidates from:

```text
/home/ye/Projects/ASX/qwen3-ASR-0.6b
```

- Split the audio encoder path into:

```text
conv_frontend: input_features -> conv_embeds
encoder_backend: conv_embeds -> audio_embeds
```

- Fixed a Pulsar2 compile problem by moving the audio positional embedding into
  the backend graph. The frontend-position merged graph hit an invalid Gemm bias
  fusion.
- Generated deterministic smoke calibration data:

```text
calibration/input_features_5s.tar
calibration/conv_embeds_5s.tar
```

- Built AX620E/NPU2 split encoder axmodels:

```text
axmodel/conv_frontend_5s_pos_backend/compiled.axmodel
axmodel/encoder_backend_5s_pos_backend/compiled.axmodel
```

- Exported a standard Qwen3 decoder HF directory for Pulsar2 `llm_build`:

```text
models/decoder_hf_axera
```

- Built Qwen3 decoder models with:

```bash
AXERA_LLM_PARALLEL=4 bash scripts/llm_build_decoder_ax630c_template.sh
```

Output:

```text
axmodel/decoder_llm/qwen3_p64_l0_together.axmodel
...
axmodel/decoder_llm/qwen3_p64_l27_together.axmodel
axmodel/decoder_llm/qwen3_post.axmodel
```

Pulsar2 printed `build llm model done!`. It did not exit on its own after the
success message, so the remaining process was stopped only after all expected
files were present.

- Assembled AXERA-style deploy filenames with:

```bash
bash scripts/assemble_deploy_ax630c.sh
```

## Current Deploy Package

Primary deploy directory:

```text
deploy_ax630c/
```

Expected top-level files:

```text
config.json
post_config.json
qwen3_tokenizer.txt
model.embed_tokens.weight.bfloat16.bin
conv_frontend.axmodel
encoder.axmodel
qwen3_asr_p64_l0_together.axmodel
...
qwen3_asr_p64_l27_together.axmodel
qwen3_asr_post.axmodel
```

There are 31 top-level `.axmodel` files:

- 1 audio frontend: `conv_frontend.axmodel`
- 1 audio encoder: `encoder.axmodel`
- 28 Qwen3 decoder layers:
  `qwen3_asr_p64_l0_together.axmodel` through
  `qwen3_asr_p64_l27_together.axmodel`
- 1 post/logits model: `qwen3_asr_post.axmodel`

Why there are many axmodels:

```text
audio features
  -> conv_frontend.axmodel
  -> encoder.axmodel
  -> qwen3_asr_p64_l0_together.axmodel
  -> ...
  -> qwen3_asr_p64_l27_together.axmodel
  -> qwen3_asr_post.axmodel
  -> token id
  -> tokenizer decode
```

The runtime loads decoder layers using this template from `config.json`:

```text
qwen3_asr_p64_l%d_together.axmodel
```

Important: `deploy_ax630c/models/` is only a convenience copy with build-stage
names. Use the top-level files for the AXERA-style runtime package.

## Important Scripts

Rebuild assets and model artifacts:

```text
scripts/prepare_assets.py
scripts/export_qwen3_asr_encoder_onnx.py
scripts/make_calibration_from_wavs.py
scripts/make_backend_calibration.py
scripts/build_ax630c_models.sh
scripts/export_decoder_hf_for_axera.py
scripts/llm_build_decoder_ax630c_template.sh
scripts/assemble_deploy_ax630c.sh
```

Most useful reproduce commands:

```bash
cd /home/ye/Projects/ASX/qwen3_asr_ax630c_port

conda run -n qwen_asr python scripts/export_qwen3_asr_encoder_onnx.py \
  --model-dir /home/ye/Projects/ASX/qwen3-ASR-0.6b \
  --seconds 5 \
  --out-dir onnx \
  --position-placement backend

conda run -n qwen_asr python scripts/make_calibration_from_wavs.py \
  --seconds 5 \
  --output calibration/input_features_5s.tar \
  --fallback-smoke

bash scripts/build_ax630c_models.sh

conda run -n qwen_asr python scripts/export_decoder_hf_for_axera.py \
  --source-dir /home/ye/Projects/ASX/qwen3-ASR-0.6b \
  --out-dir models/decoder_hf_axera

AXERA_LLM_PARALLEL=4 bash scripts/llm_build_decoder_ax630c_template.sh
bash scripts/assemble_deploy_ax630c.sh
```

## Current Quality Caveats

The encoder axmodels were built with deterministic smoke calibration, not a real
speech calibration set. This proves compiler flow and graph support, but it is
not final accuracy calibration.

Observed compiler/runtime comparison quality from the smoke build:

```text
conv_frontend final output cosine: about 0.99969
encoder_backend final output cosine: about 0.88583
```

Before serious board-side accuracy testing:

1. Prepare representative 16 kHz speech wavs.
2. Rebuild calibration tar files.
3. Rerun `scripts/build_ax630c_models.sh`.
4. If encoder quality is still low, inspect mixed precision or higher precision
   handling around attention matmul/softmax paths.

## What Is Not In Git

Large generated or downloaded artifacts are intentionally ignored:

```text
axmodel/
onnx/
deploy_ax630c/*.axmodel
deploy_ax630c/*.bin
deploy_ax630c/models/
models/hf/
models/decoder_hf_axera/
models/embeddings/
third_party/pulsar2/
```

The Git repository contains source scripts, docs, configs, reports, tokenizer,
small calibration/reference files, and release instructions. Large deploy
binaries should be uploaded as release assets or regenerated locally.

## How This Relates To AXERA `compile_620e.md`

AXERA `compile_620e.md` is the board-side C/C++ compile path. It does not
convert ONNX/HF models to `.axmodel`; that model conversion is already done here.

The AX630C path from that document is:

```bash
git clone https://github.com/AXERA-TECH/ax-samples.git
cd ax-samples
mkdir build && cd build

export ax_bsp=/path/to/AX620E_SDK_XXX/package/msp/out/arm64_glibc

cmake \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
  -DBSP_MSP_DIR=${ax_bsp}/ \
  -DAXERA_TARGET_CHIP=ax630c \
  ..

make -j6
make install
```

Expected output:

```text
ax-samples/build/install/ax630c/
```

The plain detection/classification samples are not enough for Qwen3-ASR. The
next runtime must load and call the ASR split chain:

```text
conv_frontend -> encoder -> 28 decoder layers -> post -> tokenizer
```

So the next task is not "compile another axmodel"; it is "compile or port the
board runtime that knows how to run this chain".

## Next Session Start Here

When the AX620E/AX630C SDK package is available:

1. Locate the SDK root and confirm this exists:

```text
${SDK_ROOT}/package/msp.tgz
```

2. Extract `msp.tgz` and confirm:

```text
${SDK_ROOT}/package/msp/out/arm64_glibc
```

3. Confirm the cross compiler is in `PATH`:

```bash
aarch64-none-linux-gnu-gcc --version
aarch64-none-linux-gnu-g++ --version
```

4. Clone/build AXERA `ax-samples` exactly once as a baseline:

```bash
git clone https://github.com/AXERA-TECH/ax-samples.git
cd ax-samples
mkdir build && cd build
export ax_bsp=${SDK_ROOT}/package/msp/out/arm64_glibc
cmake \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
  -DBSP_MSP_DIR=${ax_bsp}/ \
  -DAXERA_TARGET_CHIP=ax630c \
  ..
make -j6
make install
```

5. Find or port a Qwen3/Qwen3-ASR AXERA runtime wrapper. It must:

- load `deploy_ax630c/config.json`;
- mmap/load `model.embed_tokens.weight.bfloat16.bin`;
- load `conv_frontend.axmodel`;
- load `encoder.axmodel`;
- load 28 files matching `qwen3_asr_p64_l%d_together.axmodel`;
- load `qwen3_asr_post.axmodel`;
- run audio feature extraction or consume log-mel features;
- manage KV cache and decoder iteration;
- decode output tokens with `qwen3_tokenizer.txt`.

6. Copy to the board:

```text
compiled runtime executable
required AXERA shared libraries
deploy_ax630c/
test wav files
```

7. Validate in this order:

- can the runtime open every axmodel and config file;
- does frontend/encoder inference run without shape errors;
- does one decoder step run;
- does full token generation terminate;
- compare transcript quality on known short wavs;
- then measure latency and memory.

## Useful Docs In This Repo

Read these before changing anything:

```text
README.md
docs/RUNTIME_CHAIN_AX620E.md
docs/MIGRATION_NOTES.md
reports/build_summary.md
docs/DEPLOY_ARTIFACTS.md
```
