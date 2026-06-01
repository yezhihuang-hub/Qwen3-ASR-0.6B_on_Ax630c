# AX620E/AX630C Runtime Chain

This project has two separate build/use stages:

1. Model compilation on x86:
   HF/ONNX/Pulsar2 produces `.axmodel` files.
2. Board runtime compilation:
   AXERA BSP + cross compiler builds the C/C++ executable that loads and runs
   those `.axmodel` files on AX630C.

The AXERA `compile_620e.md` document belongs to stage 2. It does not compile
ONNX or HF weights into `.axmodel`; it compiles the board-side sample/runtime
program with the AX620E BSP and AX630C target.

## Final Deploy Package

Use the release package:

```text
qwen3_asr_ax630c_deploy_20260601.tar.zst
```

After extraction, the board-side runtime should receive this directory:

```text
deploy_ax630c/
├── config.json
├── post_config.json
├── qwen3_tokenizer.txt
├── model.embed_tokens.weight.bfloat16.bin
├── conv_frontend.axmodel
├── encoder.axmodel
├── qwen3_asr_p64_l0_together.axmodel
├── ...
├── qwen3_asr_p64_l27_together.axmodel
└── qwen3_asr_post.axmodel
```

The local `deploy_ax630c/models/` directory is only a convenience copy with
build-stage names. The release package uses the top-level AXERA-style names.

## Why There Are Many Axmodels

Qwen3-ASR is not deployed as one monolithic model here. It is split to match the
AXERA reference Qwen3-ASR runtime style:

```text
audio features
  -> conv_frontend.axmodel
  -> encoder.axmodel
  -> Qwen3 decoder layer 0
  -> Qwen3 decoder layer 1
  -> ...
  -> Qwen3 decoder layer 27
  -> qwen3_asr_post.axmodel
  -> token id
  -> tokenizer decode
```

File roles:

- `conv_frontend.axmodel`: first audio frontend graph. Input is fixed 5-second
  log-mel features.
- `encoder.axmodel`: audio encoder backend. It turns audio frontend embeddings
  into 1024-wide audio embeddings for the language model side.
- `model.embed_tokens.weight.bfloat16.bin`: mmap-loaded token embedding table.
- `qwen3_asr_p64_l0_together.axmodel` through
  `qwen3_asr_p64_l27_together.axmodel`: the 28 Qwen3 decoder blocks. The runtime
  loads them by the filename template in `config.json`.
- `qwen3_asr_post.axmodel`: final post/logits graph used after the decoder
  blocks to produce next-token scores/ids.
- `qwen3_tokenizer.txt`: tokenizer data used to decode generated token ids.
- `config.json`: tells the runtime the tokenizer path, decoder filename
  template, layer count, embedding file, and mmap loading policy.
- `post_config.json`: post model/runtime configuration copied from the AXERA
  reference style.

## How This Maps To `compile_620e.md`

The AXERA build path for AX630C is:

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

That produces board-side binaries under:

```text
ax-samples/build/install/ax630c/
```

For this ASR project, a plain `ax_yolov5s`-style sample is not enough. We need a
runtime program that understands this Qwen3-ASR split graph and calls the models
in the order shown above. The closest starting point is the AXERA Qwen3-ASR
reference runtime pattern, not an image classification/detection sample.

## What Is Still Missing

This repository already contains the model-side artifacts and scripts. To run on
AX630C, the remaining work is:

1. Obtain the AX620E BSP package and aarch64 cross compiler.
2. Build AXERA sample/runtime code with `-DAXERA_TARGET_CHIP=ax630c`.
3. Port or write the Qwen3-ASR runtime wrapper that loads `deploy_ax630c/`.
4. Copy the executable, required AXERA shared libraries, and `deploy_ax630c/` to
   the board filesystem.
5. Replace smoke calibration with representative 16 kHz speech calibration and
   rebuild encoder models before final accuracy validation.
