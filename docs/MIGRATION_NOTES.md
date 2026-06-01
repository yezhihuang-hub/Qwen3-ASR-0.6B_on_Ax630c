# Migration Notes

## Sources Read

- Local PDF: `docs/爱芯派2算法部署教程.pdf`
- Local RV1126B project:
  `/home/ye/Projects/qwen3_asr/asr_rv1126b`
- Local runtime project:
  `/home/ye/Projects/qwen3_asr/qwen3asr_rk_vad_git`
- AXERA reference repository:
  `https://huggingface.co/AXERA-TECH/Qwen3-ASR-0.6B-AX650-C64-P448-CTX2047`
- Pulsar2 public docs:
  `https://pulsar2-docs.readthedocs.io/`

## What the AXERA Algorithm PDF Establishes

The local algorithm PDF is written around YOLOv8, but the deployment flow is the
same class of work:

1. Export a stable ONNX graph.
2. Cut or split the graph when the runtime/demo expects intermediate outputs.
3. Prepare a Pulsar2 JSON config with calibration data and I/O processors.
4. Run `pulsar2 build --input ... --config ... --output_dir ...`.
5. Copy `.axmodel` and runtime demo files to the board.

For Qwen3-ASR this maps to audio feature input tensors, encoder outputs, LLM
decoder layer models, tokenizer, and embedding weights instead of image tensors
and YOLO post-processing tensors.

## AXERA Reference Shape

The AX650 reference repo uses a split deployment rather than a single monolithic
model:

- `conv_frontend.axmodel`
- `encoder.axmodel`
- `qwen3_asr_p64_l0_together.axmodel` through decoder layer models
- `qwen3_asr_post.axmodel`
- `model.embed_tokens.weight.bfloat16.bin`
- `qwen3_tokenizer.txt`
- `config.json`
- `post_config.json`

The downloaded `config.json` confirms:

```text
template_filename_axmodel = qwen3_asr_p64_l%d_together.axmodel
axmodel_num               = 28
tokens_embed_num          = 151936
tokens_embed_size         = 1024
use_mmap_load_embed       = true
use_mmap_load_layer       = true
```

For AX630C, the same high-level split is useful, but the compiler target and
performance envelope differ from AX650.

## AX630C Build Target Assumption

Public AXERA examples for AX630C-style boards show Pulsar2 using:

```bash
--target_hardware AX620E --npu_mode NPU2
```

This workspace therefore defaults to `AX620E` + `NPU2` in build templates. If the
installed SDK uses a more specific AX630C target name, override it:

```bash
AXERA_TARGET_HARDWARE=AX630C AXERA_NPU_MODE=NPU2 bash scripts/build_ax630c_models.sh
```

## Local Model Asset Status

The local RV1126B delivery contains reusable runtime assets:

- Reusable now:
  - `mel_filters.npy`
  - `embed_tokens.npy`
  - `tokenizer/tokenizer.json`
  - `vad/silero_vad.onnx`

The full `Qwen/Qwen3-ASR-0.6B` checkpoint is now available locally at:

```text
/home/ye/Projects/ASX/qwen3-ASR-0.6b
```

## Encoder Export Strategy

The RV1126B project has a working fixed-length merged encoder export:

```text
input_features: [1, 128, seconds * 100]
audio_embeds:   [1, audio_tokens, 1024]
```

This workspace keeps that path because it is the known baseline. It also exports
candidate split ONNX graphs:

```text
conv_frontend.onnx:
  input_features -> conv_embeds

encoder.onnx:
  conv_embeds -> audio_embeds
```

The split export is the path used for AX620E/NPU2 compilation in this workspace.
The frontend-position merged graph hit an invalid Pulsar2 Gemm bias fusion, so
audio positional embedding is applied in the backend graph instead.

## Decoder Status

The AX650 reference uses AXERA LLM layer models and a post model. Pulsar2 6.0 is
installed locally, and `models/decoder_hf_axera` has been exported as a standard
Qwen3 HF directory from the ASR checkpoint.

Decoder compilation for the AX630C direction has completed with:

```bash
AXERA_LLM_PARALLEL=4 bash scripts/llm_build_decoder_ax630c_template.sh
```

Produced files:

```text
axmodel/decoder_llm/qwen3_p64_l0_together.axmodel
...
axmodel/decoder_llm/qwen3_p64_l27_together.axmodel
axmodel/decoder_llm/qwen3_post.axmodel
```

The deploy directory renames these to match the AXERA reference runtime config:

```text
deploy_ax630c/qwen3_asr_p64_l0_together.axmodel
...
deploy_ax630c/qwen3_asr_p64_l27_together.axmodel
deploy_ax630c/qwen3_asr_post.axmodel
```

The local Pulsar2 process printed success after building post, but did not exit
cleanly on its own. It was stopped only after all 29 expected decoder axmodels
were present.

## Next SDK Step

Next board/runtime work:

1. Replace smoke calibration with representative 16 kHz speech wavs.
2. Rerun `scripts/build_ax630c_models.sh` and inspect the compiler reports for
   DDR pressure and quantization warnings.
3. Reassemble deploy files with `scripts/assemble_deploy_ax630c.sh`.
4. Integrate or port the AXERA board runtime for AX630C, using `deploy_ax630c/`
   as the model package.
5. Run board-side accuracy/performance tests, especially because the current
   encoder backend was calibrated with smoke data and showed reduced cosine.
