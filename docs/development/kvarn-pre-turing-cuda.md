# Pre-Turing CUDA KVarN attention

Issue [#112](https://github.com/Anbeeld/beellama.cpp/issues/112) reported a
large CUDA compute buffer when KVarN5/KVarN4 was used on a GTX 1070 at 8K
context. The reported KVarN run allocated about 3.6 GiB of compute memory,
compared with about 345 MiB for the standard q5_0/q4_1 control. Fit then moved
roughly 3.2 GiB of model weights to host memory.

## Root cause

The CUDA backend previously treated Turing MMA support as a requirement for
all native KVarN attention. Pascal therefore advertised no native route even
though `fattn-kvarn-portable.cuh` can read rotated compressed records directly.
The intrinsic 128-token exact suffix then made the graph build classic
attention for the body and tail. For the reported full-attention shape, one
F32 score tensor is:

```text
8192 KV rows * 1024 query rows * 16 query heads * 4 bytes = 512 MiB
```

Multiple live score buffers explain the multi-gigabyte compute allocation.
The persistent KVarN records and exact suffix do not.

## Capability contract

CUDA now reports a versioned KVarN capability record through
`ggml_backend_kvarn_capabilities`. It separates:

- store and materialization operations;
- portable direct compressed-body attention;
- integrated F16 and BF16 tails;
- specialized generic MMA, split decode, and vector decode;
- original-V-domain support;
- portable and specialized rotated-query limits;
- supported head dimensions, physical warp size, and minimum shared memory.

The old summary procedures remain available for compatibility, but cache and
graph planning consume the detailed record when a backend exports it. Unknown
ABI versions and undersized records fail closed.

A portable CUDA route requires compiled KVarN instances, a physical 32-thread
warp, a 128-thread block, and sufficient shared memory for the low-shared
KVarN path. It supports D128, D256, and D512 with rotated K and V records. A
portable-only device reports no original-V route and reports its portable
query coverage separately from the 16-query specialized decode threshold.

The integrated portable kernel handles the compressed body and attached F16
or BF16 exact tail in one online-softmax operation. Raw tail request zero still
retains the intrinsic 128-token suffix. Explicit requests keep their existing
128-token rounding, window bounds, SWA behavior, segmented current input,
bodyless mode, and multi-stream ordering.

## Architecture policy

| Architecture | Toolkit | Route | Status |
|---|---|---|---|
| Turing and newer, SM 7.5+ | CUDA 12.4 or 13.1 | Specialized routes with portable fallback | Current release tier |
| Volta, SM 7.0/7.2 | CUDA 12.4 | Portable direct | Build-targeted; real-device validation required |
| Pascal, SM 6.0/6.1/6.2 | CUDA 12.4 | Portable direct | Build-targeted; real-device validation required |
| Maxwell, SM 5.0/5.2/5.3 | CUDA 12.4 | Portable direct | Experimental pending real SM 5.2 validation |
| Kepler | Outside the CUDA 12.4/13.1 lane | None | Unsupported |

## Deterministic validation

Current CUDA hardware can simulate the portable-only capability without
running Turing-or-newer KVarN routes:

```powershell
$env:GGML_KVARN_TEST_PORTABLE_NATIVE_ONLY = "1"
$env:GGML_KVARN_TEST_FORCE_PORTABLE_CAPABILITY = "1"
$env:GGML_KVARN_TEST_BACKEND = "CUDA0"
build-win-cuda-13.1-sm_86\bin\test-kvarn.exe
```

The portable-only suite compares D128, D256, and D512 output with the
materialized CPU reference. It covers mixed K/V widths, F16/BF16 exact tails,
SWA, masks, sinks, compact bodyless input, and D256 prompt batches at 64, 256,
and 1,024 queries. Route telemetry must report portable-native and direct
body-plus-tail entries with zero materialization fallback.

The pure policy test supplies SM 5.x/6.x/7.0-like device facts and verifies
that removing matrix MMA removes only specialized routes. It also rejects
insufficient shared memory, fewer than 128 threads per block, a non-32 CUDA
warp, and builds without KVarN instances.

## Real-device acceptance

Pre-Turing qualification requires the CUDA 12.4 package and matching hardware.
Run `test-kvarn`, the relevant `test-backend-ops` cases, an issue-sized
`llama-bench --kv-memory` reproduction, and a real-model prompt/decode smoke.
With `GGML_KVARN_DEBUG_ROUTES=1`, the issue reproduction must report
`portable-native`, a rotated domain, the configured exact-tail type and rows,
and no materialization fallback. The memory report must show no full-context
F16 K/V mirror and no F32 KQ allocation.

Record the GPU, compute capability, driver, toolkit, commit, model files,
complete commands, K/V cache types, context, batch, ubatch, requested and
effective tail, route counters, peak memory, and raw output. Do not describe a
pre-Turing tier as runtime-qualified from compilation or simulation alone.
