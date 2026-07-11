# Quickstart: Gemma 4 31B with DFlash

This guide starts a Gemma 4 31B target with an upstream-format DFlash drafter on
CUDA. The multimodal projector is optional; remove `--mmproj` for a text-only
deployment.

## 1. Build

```powershell
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 16
```

The default build contains 103 standard FlashAttention cache pairs and 15
balanced KVarN fast-decode pairs. Use `-DGGML_CUDA_FA_ALL_QUANTS=ON` only when
the workload needs all 169 standard pairs and all 36 ordered KVarN pairs.

## 2. Prepare the models

Use a drafter trained for the exact Gemma target. Convert an earlier Bee/buun
`dflash-draft` file to upstream metadata and tensor names before launch:

```powershell
python scripts/convert-dflash-draft-to-upstream.py `
  "D:\models\gemma4-dflash-legacy.gguf" `
  "D:\models\gemma4-dflash-upstream.gguf" `
  --verify
```

## 3. Start the server

```powershell
build\bin\llama-server.exe `
  -m "D:\models\gemma-4-31B-it-Q4_K_S.gguf" `
  --mmproj "D:\models\mmproj-BF16.gguf" `
  --spec-type draft-dflash `
  --spec-draft-model "D:\models\gemma-4-31B-it-DFlash.gguf" `
  --spec-draft-n-min 0 `
  --spec-draft-p-min 0.0 `
  --spec-dm-controller profit `
  --cache-type-k q5_0 --cache-type-v q4_1 `
  --flash-attn on `
  --ctx-size 32768 -b 1024 -ub 512 `
  -ngl all --port 8082 --jinja
```

`--spec-draft-n-max` is intentionally omitted. BeeLlama reads
`dflash.block_size` and uses one less than the block size, normally 15. The
profit controller is already the default; it adapts within that maximum.

## 4. Verify startup and generation

At INFO verbosity, startup should report the omitted-depth resolution and the
same `n_max` in the DFlash implementation. Send a deterministic short request,
confirm that text is generated, and inspect the timing summary for generated and
accepted draft tokens.

To verify an override, restart with `--spec-draft-n-max 4`; startup must report
`n_max=4` and must not print the omitted-depth message. To keep a static depth,
also pass `--spec-dm-controller off`.

## 5. Choose and measure the target cache

The command uses conventional q caches. For lower target-cache memory on a
supported Gemma CUDA layout, try:

```text
--cache-type-k kvarn4 --cache-type-v kvarn4
```

KVarN is target-context-only and CUDA-native in v0.4.0; keep the draft cache on
a standard type. For serving, `-ub 512` is a valid throughput choice. For KLD or
cache-quality comparisons, rerun both legs with `-ub 256` and keep the corpus,
context, batch, model files, and commit identical.

## 6. Optional reasoning controls

`--reasoning-loop-guard force-close` is the default and watches hidden reasoning
for periodic repetition. A streaming chat request can instead opt into realtime
control with `"reasoning_control": true`, then send its completion id and
`"action": "reasoning_end"` to `/v1/chat/completions/control`.

See the [argument reference](beellama-args.md) for exact defaults and
[Migration from earlier versions](beellama-args.md#migration-from-earlier-versions)
for removed fork DFlash and TurboQuant settings.
