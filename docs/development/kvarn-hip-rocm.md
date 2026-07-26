# HIP/ROCm KVarN architecture

## Route contract

HIP consumes KVarN records directly whenever the selected device and shape
support a native route. Route selection uses backend, physical wave size,
compiled KVarN instances, AMD WMMA/MFMA availability, workload shape,
occupancy, and shared-memory limits.

| Route | Intended workload | Required capability |
|---|---|---|
| Split decode | `n_q <= 16`, supported D/GQA/bit pair | wave32 or wave64 AMD MMA |
| Generic MMA | prompt, verification fallback, non-fast pair | AMD WMMA or MFMA |
| Portable native | older AMD or forced parity | compiled KVarN portable kernel |
| Materialized fallback | unsupported placement/shape only | backend standard attention |

RDNA launches physical wave32 kernels. CDNA launches physical wave64 kernels;
the split kernel never treats a logical CUDA warp constant as AMD lane
ownership. MUSA is explicitly portable-only.

The default build includes 15 balanced fast bit pairs. All 36 ordered pairs are
available with `GGML_CUDA_FA_ALL_QUANTS=ON`; generic MMA handles valid pairs
outside the default split-decode matrix.

## Exact tails and graphs

Direct KVarN attention and compact segmented-tail attention call the same
selector. The compact path tags its body dispatch separately, then merges the
body and F16/BF16 exact contribution through the existing FP32 maximum and
denominator metadata. No record, state, or tail ABI changes.

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
structure. Attention counters distinguish AMD generic/split routes, portable
and materialization fallback, split reduction, and direct/compact entry paths.
Store counters distinguish workspace, monolithic, direct, and high/low-LDS
routes.

## Validation status

CUDA parity, forced-portable parity, route policy, generated instances, KVarN
on/off configuration, model smoke tests, and CUDA regressions are validated
locally. The existing release CI is configured to compile a fat ROCm build for
RDNA, CDNA, and older AMD targets with the default pair matrix. AMD compilation
still requires that CI run, and runtime correctness and performance require
matching hardware; CDNA fast routing remains experimental until those results
are published.
