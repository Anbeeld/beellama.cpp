# HIP/ROCm KVarN validation record — 2026-07-25

## Environment

The implementation host has an NVIDIA RTX 3090 and CUDA 13.1/13.2. It has no
AMD GPU or local ROCm toolchain. Consequently this record contains no claimed
ROCm throughput, WMMA/MFMA utilization, graph, or parity measurement.

The existing release workflow is configured to compile one fat ROCm package for
`gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1150;gfx1200;gfx1201`.
Local CMake configuration checks cover KVarN disabled and all-pairs source
selection; those configurations are not AMD compile evidence.

## Local CUDA regression evidence

These results validate the shared-source CUDA path and real-model integration;
they are not ROCm performance evidence.

- Source: implementation commit containing this record, parent
  `3be484a6974c7f16f673106ffffbb458c92f5a48`.
- GPU: NVIDIA GeForce RTX 3090, compute capability 8.6, 24,576 MiB.
- Driver: 596.21.
- CUDA compiler: 13.1.80.
- Build: Release, `GGML_CUDA=ON`, `GGML_CUDA_KVARN=ON`,
  `GGML_CUDA_FA_ALL_QUANTS=ON`, `CMAKE_CUDA_ARCHITECTURES=86`, parallel 16.
- Qwen model:
  `D:\models\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q5_K_S.gguf`.
- Gemma model:
  `E:\models\gemma-4-31b-it-GGUF\gemma-4-31B-it-Q5_K_S.gguf`.

The bounded decode sweep used one load per cache:

```powershell
llama-bench.exe -m <QWEN_MODEL> -ngl 999 -p 0 -n 4 `
  -d 0,512,2048,8192,16384 -b 2048 -ub 512 -r 1 `
  -ctk <CACHE> -ctv <CACHE> -fa on -mmp 0 --no-host 1 -o jsonl
```

| Depth | KVarN5 tok/s | q8_0 tok/s |
|---:|---:|---:|
| 0 | 26.50 | 26.56 |
| 512 | 26.69 | 26.63 |
| 2,048 | 25.78 | 27.65 |
| 8,192 | 24.45 | 28.14 |
| 16,384 | 24.10 | 26.81 |

At 16K, KVarN5 was 89.89% of the matched q8_0 control. The prefill command
used `-p 512,2048,8192,16384 -n 0` with the same remaining arguments:

| Prompt | KVarN5 tok/s | q8_0 tok/s |
|---:|---:|---:|
| 512 | 1,009.78 | 1,045.42 |
| 2,048 | 1,022.69 | 1,049.51 |
| 8,192 | 995.58 | 1,016.74 |
| 16,384 | 945.03 | 973.65 |

Explicit `--kv-tail-tokens 128` decode controls at 16K measured 24.19 tok/s
for KVarN5, 24.33 tok/s for mixed KVarN5/V4, and 25.59 tok/s for q8_0.

Hidden-window `llama-cli` smokes generated four tokens without a rollback,
assertion, or CUDA error:

| Model/cache | Prompt tok/s | Generation tok/s |
|---|---:|---:|
| Qwen 3.6 / KVarN5 | 45.2 | 32.9 |
| Gemma 4 SWA / KVarN4 | 43.8 | 25.4 |

The serving-cadence quality spot used the repository WikiText-2 raw corpus,
the existing model-matched BF16 context-512 baselines, `-c 512 -b 512
-ub 512 --chunks 4 --no-warmup`, and FlashAttention:

| Model/cache | Mean KLD | Median KLD | Same top |
|---|---:|---:|---:|
| Qwen 3.6 / KVarN5 | 0.004680 | 0.001016 | 98.137% |
| Gemma 4 SWA / KVarN4 | 0.297123 | 0.039245 | 81.373% |

At Qwen depth 16K, `llama-bench --kv-memory --no-warmup` reported an
effective 128-token exact tail, 373.438 MiB resident KV storage, 8.062 MiB
exact-tail storage, 16 MiB staging, 46.080 MiB transient use, and 419.518 MiB
KV peak. Descriptor scratch was 768 bytes, split partial output was 6.094 MiB,
and no tail layer used CPU or device fallback.

CUDA validation also covered all 36 K/V pairs, D=128/256/512, GQA and
multi-query shapes, global/SWA/wrap cases, F16/BF16 exact tails, both direct
and compact entry paths, forced portable attention, the shared-WHT oracle,
all six store routes, and 4,267 selected `FLASH_ATTN_EXT` backend-op cases.
Raw local logs are under `tmp/hip-kvarn-evidence/`.

The final Release validation used:

```powershell
ctest --test-dir build-win-cuda-13.1-sm_86 -C Release --output-on-failure
```

All 70 tests passed in 446.74 seconds. The exhaustive `test-backend-ops`
case passed in 279.04 seconds and `test-kvarn` passed in 30.11 seconds.
The suite includes the host-only HIP route-policy matrix, generated-source and
build-selection invariants, compact-tail rollback regression, KVarN graph
on/off coverage, and the CUDA runtime matrix. No workflow file was changed.

## Required AMD hardware run

Record the commit, ROCm runtime, exact GPU and `gfx` target, physical wave size,
compute-unit and LDS limits, graph setting, model checksum/path, resolved tail
tokens, route-counter delta, and complete command output.

Qwen 3.6 smoke:

```bash
GGML_KVARN_DEBUG_ROUTES=1 build-hip/bin/llama-bench \
  -m /models/Qwen3.6-27B-Q5_K_S.gguf -ngl 999 -p 0 -n 4 \
  -d 0,512,2048,8192,16384 -b 2048 -ub 512 -r 1 \
  -ctk kvarn5 -ctv kvarn5 -fa on -mmp 0 --no-host 1 -o jsonl
```

Repeat with `-ctk q8_0 -ctv q8_0`, then run `n_q=1`, `8`, and `16` parity,
forced-portable parity, graph on/off, and the full `test-kvarn` suite. Gemma
validation must also select KVarN for the SWA cache and include a wrapped-SWA
case. Performance acceptance remains the thresholds in the end-to-end plan;
successful compilation alone is not performance evidence.
