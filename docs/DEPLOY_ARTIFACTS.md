# Deploy Artifacts

Large AX630C-direction deployment binaries are not committed to Git history.
They are published as a GitHub Release asset.

Release:

```text
https://github.com/yezhihuang-hub/Qwen3-ASR-0.6B_on_Ax630c/releases/tag/v0.1.0-ax630c-deploy-20260601
```

Expected package:

```text
qwen3_asr_ax630c_deploy_20260601.tar.zst
```

Expected extracted layout:

```text
deploy_ax630c/
├── config.json
├── conv_frontend.axmodel
├── encoder.axmodel
├── post_config.json
├── qwen3_tokenizer.txt
├── model.embed_tokens.weight.bfloat16.bin
├── qwen3_asr_p64_l0_together.axmodel
├── ...
├── qwen3_asr_p64_l27_together.axmodel
└── qwen3_asr_post.axmodel
```

Use the checksum manifest in `reports/deploy_ax630c_manifest.sha256` after
download:

```bash
sha256sum -c reports/deploy_ax630c_manifest.sha256
```

The checked-in `deploy_ax630c/config.json`, `post_config.json`, and tokenizer
are included for reference, while the large `.axmodel` and embedding files are
kept in the release package.
