# HIP/ROCm KVarN architecture

## Route contract

HIP consumes KVarN records directly. The device capability record inventories
compiled route families, while the dispatcher decides whether an individual
operation can use one of those families. Family presence is not operation
support.

HIP uses the safe-first rotated K/V graph domain for every query count. Its
`original_v_domain` capability is false and its rotated query limit comes from
the unbounded portable route. This prevents a partial AMD matrix route from
moving D256 or D512 graphs into the CUDA-only original-V window contract.

| Route | Intended workload | Required capability |
|---|---|---|
| Split decode | `n_q <= 16`, supported D/GQA/bit pair | wave32 or wave64 AMD MMA |
| Generic MMA | eligible prompt, verification fallback, non-fast pair | AMD WMMA or MFMA plus an operation-valid tile |
| Portable native | older AMD or forced parity | compiled KVarN portable kernel |
| Materialized fallback | unsupported placement/shape only | backend standard attention |

RDNA launches physical wave32 kernels. CDNA launches physical wave64 kernels;
the split kernel never treats a logical CUDA warp constant as AMD lane
ownership. MUSA is explicitly portable-only.

Before an AMD generic launch, the dispatcher applies the same template limits
as the F16 MMA wrapper:

| Family | Valid generic head dimensions | Other launch requirements |
|---|---|---|
| RDNA WMMA, wave32 | D128 | `ncols1 * ncols2 >= 16` and `ncols2 != 1` |
| CDNA MFMA, wave64 | D128 or D256 | `ncols1 * ncols2 >= 16` |

An invalid generic shape increments `generic_shape_rejected` and continues to
portable native attention. It never reaches the wrapper's `NO_DEVICE_CODE`
branch and never materializes a context-sized F16 cache. CUDA keeps its prior
vector, split, prompt, and generic route ordering.

The default build includes 15 balanced fast bit pairs. All 36 ordered pairs are
available with `GGML_CUDA_FA_ALL_QUANTS=ON`. The pair matrix controls optimized
decode instantiations; the portable route remains the correctness fallback.

## Exact tails and graphs

Direct KVarN attention and compact segmented-tail attention share the same
operation-level body predicate. The compact path tags its body dispatch
separately, then merges the body and F16/BF16 exact contribution through the
existing FP32 maximum and denominator metadata. No record, state, or tail ABI
changes.

Routing performs no context-proportional allocation. Descriptor and split
storage continue through the existing CUDA/HIP operation allocator, and route
selection does not synchronize the device. Profiling is opt-in and disabled
during graph capture.

## Diagnostics

- `GGML_KVARN_DEBUG_ROUTES=1` logs backend, architecture code, physical wave,
  D, K/V bits, GQA, query/KV sizes, route, entry path, and fallback reason.
- `GGML_KVARN_TEST_FORCE_PORTABLE_FATTN=1` forces the portable native path for
  reference parity.
- `GGML_KVARN_TEST_FORCE_SHARED_WHT=1` forces the original shared-memory WHT
  oracle.
- `GGML_KVARN_PROFILE=1` enables HIP/CUDA event profiling. Set
  `GGML_KVARN_PROFILE_DUMP_EVERY` to a positive operation interval to print
  bounded summaries.

Versioned attention and store route-stat proc APIs include caller size and ABI
version fields so older test binaries cannot be overwritten by a larger
structure. Attention ABI v2 distinguishes AMD generic/split routes, portable
and materialization fallback, split reduction, direct/compact entry paths, and
operation-specific generic rejection. Store counters distinguish workspace,
monolithic, direct, and high/low-LDS routes.

## AMD runtime qualification

Release builds compile the configured RDNA and CDNA architecture list. That is
compile coverage only. Runtime qualification requires one physical wave32 RDNA
host and one physical wave64 CDNA host.

On each host, run the evidence collector against a configured HIP build:

```bash
python3 scripts/hip/validate-kvarn-runtime.py \
  --build-dir build-hip \
  --evidence-dir artifacts/kvarn-rdna-wave32 \
  --device-class rdna-wave32 \
  --all-tests
```

Use `cdna-wave64` and a separate evidence directory on the CDNA host. The
collector waits for each command, redirects output to log artifacts,
and writes `evidence.json` with the commit, compiler, runtime inventory, CMake
matrix, expected physical wave, commands, durations, and results. The AMD
route-boundary subset refuses to pass unless the backend's reported physical
wave matches the requested device class. A run without `--all-tests` records
the omitted repository tests and qualifies only the KVarN gate.

## Validation status

The existing release jobs provide multi-architecture ROCm compile and Windows
HIP package coverage. Neither job is an AMD runtime claim. Publish the two
`evidence.json` artifacts before describing HIP KVarN as runtime-qualified or
re-enabling the bounded original-V window path. Until then, HIP stays on the
portable-safe rotated domain and specialized routes are optional acceleration.

## Local CUDA control evidence

The implementation commit containing this record was tested on 2026-08-02
from parent `9d59db2e3`, using an RTX 3090, CUDA 13.1, and
`Qwen3.6-27B-Q5_K_S.gguf` (18,947,311,616 bytes). The controlled memory run
used one non-speculative sequence, 8,192 prompt tokens, `-b 512`, `-ub 256`,
`-ngl 99`, FlashAttention, no host offload, and one repetition. Both cache
variants resolved an 8,192-token capacity.

| Category, bytes | q4_0/q4_0 | KVarN4/KVarN4 |
|---|---:|---:|
| K/V payload | 150,994,944 | 146,800,640 |
| Exact tail | 0 | 8,454,144 |
| Staging | 0 | 25,165,824 |
| Total resident KV | 150,994,944 | 180,420,608 |
| Peak KVarN transient | 0 | 21,332,480 |
| Non-KV context buffers | 156,893,184 | 156,893,184 |
| Compute buffers | 284,216,480 | 284,062,016 |
| CUDA runtime/VMM residual | 51,237,728 | 105,852,608 |
| Measured peak delta above model | 643,342,336 | 727,228,416 |

For each row, resident context buffers plus compute buffers plus the explicit
CUDA runtime/VMM residual reconcile exactly to the measured peak delta. KVarN4
used the compact-tail body route 512 times, used no materialization fallback,
and had no context-sized F16 body. Its 80 MiB larger peak consists of bounded
exact/staging state and a larger CUDA runtime/VMM reserve, not a second cache.
Prompt throughput was 1,112.7 tokens/s for q4_0 and 954.6 tokens/s for KVarN4.

The same target also completed a 717-word prompt and 128-token decode with
q8_0 (994.9 prompt and 36.8 decode tokens/s) and KVarN6 (865.5 and 34.4
tokens/s). A matching Qwen DFlash Q4_K_M drafter completed the same KVarN6 run
at 636.6 prompt and 73.8 decode tokens/s. A separate MTP smoke kept the target
cache on KVarN4, the CPU drafter cache on q8_0, and completed without moving
KVarN into the auxiliary context. These are CUDA controls, not AMD runtime
qualification or cross-hardware performance claims.
