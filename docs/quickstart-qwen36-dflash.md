# Quickstart: Qwen 3.6 with upstream DFlash

This guide runs a Qwen 3.6 target together with an upstream-format DFlash draft
model. v0.4.0 uses upstream speculative decoding: the canonical mode is
`draft-dflash`, not the retired fork DFlash stack.

## Build

```powershell
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 16
```

The default CUDA FlashAttention build covers 103 standard cache pairs. Add
`-DGGML_CUDA_FA_ALL_QUANTS=ON` only when an experiment needs the full 169-pair
standard matrix and all 36 KVarN fast-decode pairs. There is no HALF build tier.

## Models

You need a Qwen 3.6 target GGUF and a DFlash draft GGUF trained for that target.
The draft shares the target's embeddings and output projection at runtime. An old
Bee/buun `dflash-draft` file must first be converted:

```powershell
python scripts/convert-dflash-draft-to-upstream.py `
  "D:\models\qwen-dflash-legacy.gguf" `
  "D:\models\qwen-dflash-upstream.gguf" `
  --verify
```

## Launch

Replace the paths and fit settings for your hardware.

```powershell
build\bin\llama-server.exe `
  -m "D:\models\Qwen3.6-27B-Q4_K_M.gguf" `
  --spec-type draft-dflash `
  --spec-draft-model "D:\models\Qwen3.6-27B-DFlash.gguf" `
  --spec-draft-n-min 0 `
  --spec-draft-p-min 0.0 `
  --spec-dm-controller profit `
  --cache-type-k q5_0 --cache-type-v q4_1 `
  --flash-attn on `
  --ctx-size 32768 -b 1024 -ub 512 `
  -ngl all --port 8082 --jinja
```

With `--spec-draft-n-max` omitted, BeeLlama uses the drafter's trained block
depth (`dflash.block_size - 1`, normally 15). Pass the flag explicitly to
override it; the profit controller remains enabled by default and adapts within
that maximum.

The `-ub 512` setting above is suitable for serving. For KVarN quality or KLD
measurements, rerun with `-ub 256`; the known `-ub 512` KLD drift is caused by
ubatch cadence, not a cache defect.

For lower KV memory, use a matched KVarN pair rather than mixing one KVarN side
with a standard cache:

```text
--cache-type-k kvarn4 --cache-type-v kvarn4
```

KVarN is target-context-only and requires an offloaded, supported attention
backend. Test quality at the exact context length and ubatch size you plan to
serve.

## Adaptive draft depth

The profit controller measures accepted tokens per wall-clock cycle and varies
the DFlash horizon. It is enabled with `--spec-dm-controller profit`; choose
`off` for a fixed `--spec-draft-n-max`. `--spec-draft-p-min` is independent: it
stops an individual draft early when confidence is too low.

Useful tuning controls are:

- `--spec-dm-profit-min`
- `--spec-dm-profit-raise-margin`
- `--spec-dm-profit-lower-margin`
- `--spec-dm-profit-min-samples`
- `--spec-dm-profit-baseline-interval`

Use the defaults first and compare a fixed-depth run with the same prompt,
sampling settings, model files, and GPU before changing them.

## Reasoning models

For models that produce long hidden reasoning, the server can watch for repeated
reasoning-token patterns:

```text
--reasoning-loop-guard force-close
```

`force-close` asks the reasoning-budget sampler to finish the hidden section;
`stop` ends generation on a trigger. Request JSON can override the guard and its
window, coverage, period, and intervention settings.

For interactive control instead of automatic detection, set
`"reasoning_control": true` on a streaming chat request and POST its completion
id with `"action": "reasoning_end"` to `/v1/chat/completions/control`.

## Migration notes

- `--spec-type dflash` remains a warned alias for `draft-dflash`.
- `--spec-dflash-cross-ctx`, `--spec-branch-budget`, and the GPU-ring
  environment variables were removed with the old fork verifier.
- TurboQuant/TCQ cache types were removed. Their command-line spellings redirect
  with a warning; use standard q cache types or KVarN explicitly in new setups.
