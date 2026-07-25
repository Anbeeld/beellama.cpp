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

The default build contains 50 standard FlashAttention cache pairs and 15
balanced KVarN fast-decode pairs. Use `-DGGML_CUDA_FA_ALL_QUANTS=ON` only when
the workload needs all 169 standard pairs and all 36 ordered KVarN pairs.

For ROCm/HIP on Strix Point, use:

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build-hip -DGGML_HIP=ON -DGGML_CUDA_KVARN=ON \
  -DGGML_CUDA_FA=ON -DGPU_TARGETS=gfx1150 -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip --parallel 16
```

Use `GPU_TARGETS=gfx942` for CDNA3. CDNA fast routing is experimental until
validated on matching hardware.

## 2. Prepare the models

Use a drafter trained for the exact Gemma target with upstream `dflash`
architecture metadata and tensor names. Other DFlash GGUF schemas are
unsupported.

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
supported Gemma layout, try:

```text
--cache-type-k kvarn4 --cache-type-v kvarn4
```

KVarN is target-context-only; keep the draft cache on a standard type. CUDA,
ROCm/HIP, Vulkan, and CPU consume compressed records directly in native
attention paths. Vulkan needs shader Int64 and buffer-device-address support.
For KLD or cache-quality comparisons, use the intended serving backend and
ubatch and keep the corpus, context, logical batch, physical ubatch, model
files, and commit identical between both legs.

Exercise Gemma's SWA cache explicitly in HIP smoke tests:

```bash
GGML_KVARN_DEBUG_ROUTES=1 build-hip/bin/llama-bench \
  -m /models/gemma-4-31B-it-Q5_K_S.gguf -ngl 999 -p 0 -n 4 \
  -d 0,512,2048 -b 2048 -ub 512 -r 1 \
  -ctk kvarn4 -ctv kvarn4 \
  --cache-type-k-swa kvarn4 --cache-type-v-swa kvarn4 \
  -fa on -mmp 0 --no-host 1
```

## 6. Optional reasoning controls

`--reasoning-loop-guard force-close` is the default and watches hidden reasoning
for periodic repetition. A streaming chat request can instead opt into realtime
control with `"reasoning_control": true`, then send its completion id and
`"action": "reasoning_end"` to `/v1/chat/completions/control`.

See the [argument reference](beellama-args.md) for exact defaults and
[Migration from earlier versions](beellama-args.md#migration-from-earlier-versions)
for removed fork DFlash and TurboQuant settings.
