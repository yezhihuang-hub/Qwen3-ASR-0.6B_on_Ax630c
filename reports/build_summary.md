# Qwen3-ASR AX630C Build Summary

Date: 2026-06-01

## Completed

- Installed local Pulsar2 6.0 lite package:
  `third_party/pulsar2/6.0/extracted/ax_pulsar2_6.0_lite_package/bin/pulsar2`
- Exported 5-second fixed Qwen3-ASR encoder ONNX graphs from:
  `/home/ye/Projects/ASX/qwen3-ASR-0.6b`
- Moved audio positional embedding into the backend graph to avoid Pulsar2's
  invalid Gemm bias fusion on the merged/frontend-position graph.
- Generated smoke calibration datasets:
  - `calibration/input_features_5s.tar`
  - `calibration/conv_embeds_5s.tar`
- Built AX620E/NPU2 `.axmodel` files:
  - `axmodel/conv_frontend_5s_pos_backend/compiled.axmodel`
  - `axmodel/encoder_backend_5s_pos_backend/compiled.axmodel`
- Exported standard Qwen3 decoder HF directory for AXERA `llm_build`:
  - `models/decoder_hf_axera`
- Built Qwen3 decoder AX620E/NPU2 models with Pulsar2 `llm_build`:
  - `axmodel/decoder_llm/qwen3_p64_l0_together.axmodel` through
    `axmodel/decoder_llm/qwen3_p64_l27_together.axmodel`
  - `axmodel/decoder_llm/qwen3_post.axmodel`
- Assembled AXERA-style deploy filenames under `deploy_ax630c/`.

## Produced Deploy Skeleton

```text
deploy_ax630c/
├── config.json
├── conv_frontend.axmodel
├── encoder.axmodel
├── post_config.json
├── qwen3_tokenizer.txt
├── model.embed_tokens.weight.bfloat16.bin
├── qwen3_asr_p64_l0_together.axmodel ... qwen3_asr_p64_l27_together.axmodel
├── qwen3_asr_post.axmodel
└── models/
    ├── conv_frontend_5s.axmodel
    └── encoder_backend_5s.axmodel
```

The decoder build used `AXERA_LLM_PARALLEL=4` on the local 8-core host. Pulsar2
printed `build llm model done!`, but the process did not exit by itself after
the success message, so it was stopped after all expected axmodels were present.

## Important Quality Note

The current calibration data is deterministic smoke data, not representative
speech. It is useful for proving graph support and compiler flow only.

Observed precision:

- `conv_frontend`: final output cosine around `0.99969`.
- `encoder_backend`: final output cosine around `0.88583`, with several
  attention/MLP tensors significantly lower.

Before board-side accuracy testing, replace the smoke calibration with real
16 kHz speech wavs and rerun build. The backend may also need mixed precision
or higher-precision treatment around attention matmul/softmax paths.

## Useful Commands

Rebuild split encoder:

```bash
cd /home/ye/Projects/ASX/qwen3_asr_ax630c_port
bash scripts/build_ax630c_models.sh
```

Run optional merged experiment:

```bash
BUILD_MERGED=1 bash scripts/build_ax630c_models.sh
```

Prepare decoder layer models:

```bash
cd /home/ye/Projects/ASX/qwen3_asr_ax630c_port
bash scripts/llm_build_decoder_ax630c_template.sh
bash scripts/assemble_deploy_ax630c.sh
```
