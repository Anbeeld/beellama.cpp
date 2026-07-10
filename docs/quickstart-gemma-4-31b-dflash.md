# Quickstart: Gemma 4 31B with upstream DFlash

v0.4.0 runs DFlash through llama.cpp's upstream `draft-dflash` implementation.
This replaces BeeLlama's retired GPU-ring, tree-verification, and reduced-verifier
paths.

## Build

```powershell
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 16
```

The default CUDA FlashAttention build has 103 vector cache pairs and supports all
KVarN K/V combinations through native descriptor routing. Use
`-DGGML_CUDA_FA_ALL_QUANTS=ON` only for the full 169 standard-pair and 36 KVarN
fast-decode matrices. `GGML_CUDA_FA_HALF_QUANTS` no longer exists.

## Models

Provide a Gemma 4 target GGUF and a matching upstream-format DFlash draft GGUF.
If a draft uses the historical `dflash-draft` schema, rewrite it before launch:

```powershell
python scripts/convert-dflash-draft-to-upstream.py `
  "D:\models\gemma4-dflash-legacy.gguf" `
  "D:\models\gemma4-dflash-upstream.gguf" `
  --verify
```

The optional multimodal projector remains a normal upstream `--mmproj` input;
do not use retired fork-only speculative flags with it.

## Launch

```powershell
build\bin\llama-server.exe `
  -m "D:\models\gemma-4-31B-it-Q4_K_S.gguf" `
  --mmproj "D:\models\mmproj-BF16.gguf" `
  --spec-type draft-dflash `
  --spec-draft-model "D:\models\gemma-4-31B-it-DFlash.gguf" `
  --spec-draft-n-max 8 `
  --spec-draft-n-min 0 `
  --spec-draft-p-min 0.0 `
  --spec-dm-controller profit `
  --cache-type-k q5_0 --cache-type-v q4_1 `
  --flash-attn on `
  --ctx-size 32768 -b 1024 -ub 512 `
  -ngl all --port 8082 --jinja
```

For a tighter KV budget, use a matched KVarN configuration:

```text
--cache-type-k kvarn4 --cache-type-v kvarn4
```

KVarN is a target-context CUDA/Vulkan feature and must be validated on the exact
model, context length, and ubatch you intend to serve. It is not a draft-cache
replacement.

## Adaptive draft depth

`--spec-dm-controller profit` adapts the upstream DFlash per-call draft limit.
It uses accepted output tokens divided by measured cycle time, including periodic
no-spec baseline samples. To pin the horizon, select `--spec-dm-controller off`
and set `--spec-draft-n-max` directly.

`--spec-draft-p-min` remains the per-position draft confidence threshold; it is
not an adaptive-controller setting.

## Reasoning-loop protection

Enable a guard for thinking models with:

```text
--reasoning-loop-guard force-close
```

The server reports guard telemetry in completion responses. `force-close` may
intervene once or more times according to `--reasoning-loop-interventions`; `stop`
ends the completion on a detected loop.

An opted-in streaming request (`"reasoning_control": true`) can also be moved to
its final answer by POSTing its completion id and `"action": "reasoning_end"` to
`/v1/chat/completions/control`.

## Migration notes

- Use `draft-dflash`. The old `dflash` spelling is a warning-producing alias.
- Removed options include `--spec-dflash-cross-ctx`, `--spec-branch-budget`, and
  all `GGML_DFLASH_*` ring/capture settings.
- TurboQuant/TCQ cache formats are removed. Choose q-cache types or KVarN for a
  new deployment; historical command-line aliases only exist to ease migration.
