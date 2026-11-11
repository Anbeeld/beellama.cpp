# Hardware-aware KVarN dispatch

The dispatch policies use device resources and workload geometry without treating
one GPU's measured crossover as a universal eligibility rule. They do not change
cache formats, precision-tail policy, cache ownership, or supported architectures.

## CUDA split decode (#183)

`ggml/src/ggml-cuda/fattn-mma-kvarn-decode.cuh` considers every supported
split-64/split-128 specialization without the former D128 4K and wider-head 32K
context gates. Each candidate must have positive occupancy, queried for the actual
compiled kernel on the active device. Its grid includes KV heads, GQA blocks,
streams and query tiles.

The ranking in `fattn-kvarn-geometry-policy.h` counts warp-level K and V tile-loop
passes plus the fixed K/V preparation phases, multiplied by resident waves. This
replaces the locally calibrated million/4000/100/10 score. It is a structural
heuristic, not a prediction of kernel latency: tile passes need not have equal
cost on different devices or cache precisions.

Before amortizing CTA setup, the selector prefers a candidate with at least two
resident waves of independent work when one is available, following the existing
unsplit-grid occupancy guard. This avoids exchanging latency-hiding parallelism
for fewer CTAs. Near-equivalent work estimates then use upstream's 95-percent
efficiency tolerance to prefer fewer partial outputs and less combine traffic.
This avoids a rounded last wave reversing the preference for otherwise similar deep-context kernels.
Larger work differences take precedence. There are no device-count-scaled context
thresholds and no online timing or persistent calibration state.

Generation and speculative verification still choose the partition using
single-query geometry. Verification may widen its query tile only after fixing
the split, warp count and GQA grouping. Changing batch query count therefore does
not change this path's KV partition/summation order.

## Compiler resource policy

The qualified sm_86 four-block register target, capped by its 1536-thread limit,
is scoped to code compiled for sm_86. Other compiled architectures use the
upstream MMA configuration's occupancy target for the head width rather than
inheriting that thread budget. Runtime selection still queries occupancy of the
actual KVarN kernel; the compiler target is not a promise of resident occupancy.
A binary compiled only for sm_86 retains that compiled policy when JIT-compatible
on another GPU, so native/multi-architecture builds remain preferable for tuning.

## SWA vector decode

The existing D256, GQA2, single-query, rotated-domain SWA qualification remains.
The selector independently considers the compiled 8/16/32-token vector kernels,
queries their occupancy, and compares wave work and partial-output counts.
The per-device vector occupancy cache uses relaxed atomics so concurrent backend
contexts can populate it without racing.
`GGML_KVARN_VEC_TOKENS=8|16|32` retains its explicit override behavior; unset or
invalid values select automatically. No additional head widths or bit pairs are
qualified by this change. The Qwen benchmark model below is not a SWA-vector
workload; vector numerical coverage comes from `test-kvarn` and separate synthetic
attention measurements.

## HIP graphs and matrix eligibility

HIP prefill now follows upstream's operation-compatibility and stable-property
warmup/replay path rather than bypassing graphs based on the first node's query
dimension. Property changes still reset warmup, and incompatible operations still
execute directly.

WMMA/MFMA family availability remains separate from operation eligibility. The
existing head-width, column-tile, RDNA single-GQA-column and RDNA4 precision guards
are retained, as are upstream launch/resource checks. There is no evidence here
to replace these with a gfx1100-only performance preference or to expand kernel
qualification. HIP runtime performance is not measured on the NVIDIA test host.

## Vulkan

- KVarN attention's runtime and scratch planner use the same topology convention
  as generic upstream attention: Intel non-Xe2 core counts are doubled, and
  unknown counts fall back to 16. The existing two-workgroups/core target remains.
- Store selection accounts for the monolithic shader's actual grid, not just
  token count. That grid has one CTA per logical head; slices and stream tokens
  are serial work inside each CTA. With more than one record group of tokens
  and fewer than two existing workgroups/core, the workspace path can expose
  record, stream and slice parallelism. Do not count that potential extra work
  as parallelism already present in the serial path. Existing shape, stream,
  SWA and buffer-size constraints remain.
  Worst-case reservation includes all newly admitted short shapes.
- Workspace validation rejects repeated chunks targeting the same stream, since
  their parallel record/stage writes would race. Such inputs use the existing
  serial shader, preserving write order. The multistream byte-parity regression
  covers this case without loosening its assertions.
- NVIDIA retains the qualified full-head WHT preference. Other vendors prefer
  the sliced implementation when the full-head version adds subgroups without
  adding independent row work. Invocation limits and subgroup support remain
  eligibility constraints. Pipeline selection reuses one Bee pipeline-table
  lookup instead of taking the same global mutex/map lookup twice for a full
  head. The shape preference is conservative, not a demonstrated performance
  win on untested devices.

## Upstream references

The architectural reference is upstream commit
`b5cb8f6f470a3a4d6f329bbb050e9a34fde6638d`:

- `ggml/src/ggml-cuda/fattn-common.cuh`, `launch_fattn`: concrete occupancy,
  resident waves, workload-dependent KV parallelism and 95-percent stopping rule.
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`: matched architecture-specific compiler
  and host launch configurations.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: graph compatibility and stable-property warmup.
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`: generic attention topology normalization,
  independent-work-aware multipass selection, and subgroup-aware FWHT policies.

These are blueprints for policy structure, not transferable performance numbers.

## Local validation

Baseline: `d8b5685b0f70a6d3dad3d3b266775b927490cfef`. The tested candidate
implementation is recorded in `6f1ec3817` (CUDA/HIP) and `e768dbe8e` (Vulkan).
The measurements were taken from that implementation before committing it.

Hardware and builds:

- Ryzen 9 9950X; model runs use eight CPU threads and one RTX 3090 (82 SMs,
  24 GiB). Two RTX 3090s are installed; final matrix and control benchmark jobs
  run sequentially. CUDA and Vulkan device indices are API enumeration indices,
  not an assumed common physical-GPU ordering.
- NVIDIA driver 616.56; CUDA 13.3, Release, sm_86, default FA quant matrix,
  `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=OFF`, shared libraries.
- Vulkan SDK 1.4.350.0, Release, `GGML_NATIVE=OFF`, static libraries and
  `GGML_BACKEND_DL=OFF`. Static benchmark/test executables were copied to
  isolated directories to avoid loading stale DLLs in the existing build folder.
- GPU clocks were not locked. Small differences below one percent should not be
  read as a demonstrated gain; the measurements do not establish exact equality.

Model: `C:\models\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-IQ4_XS.gguf`
(`qwen35 27B IQ4_XS`), SHA-256:
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
The prompt and decode tokens are llama-bench's synthetic `std::rand()` token
stream (with BOS where applicable), not a conversational prompt. No output
sampler, temperature, or speculative drafter is used.

### CUDA model matrix

Run each command with both the preserved baseline benchmark and changed build;
`$Bits` is 4 or 5 and `$Bench` selects the executable:

```powershell
$Model = 'C:\models\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-IQ4_XS.gguf'
& $Bench -m $Model -dev CUDA0 -ngl 99 -fa on `
  -ctk "kvarn$Bits" -ctv "kvarn$Bits" -p 512 -n 64 `
  -d 0,4096,16384,32768,65536 -b 2048 -ub 512 -t 8 -r 5 -o json
```

Baseline executable: `tmp/issue183/baseline-cuda/llama-bench.exe`.
Candidate: `build-win-cuda-sm_86-default/bin/llama-bench.exe`.
Mean decode throughput in tokens/s:

| K/V bits | Depth | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: | ---: |
| 4 | 0 | 40.056 | 40.033 | -0.06% |
| 4 | 4096 | 39.067 | 39.191 | +0.32% |
| 4 | 16384 | 37.003 | 37.055 | +0.14% |
| 4 | 32768 | 35.061 | 35.115 | +0.15% |
| 4 | 65536 | 31.841 | 31.864 | +0.07% |
| 5 | 0 | 39.278 | 39.251 | -0.07% |
| 5 | 4096 | 38.933 | 38.827 | -0.27% |
| 5 | 16384 | 36.753 | 36.661 | -0.25% |
| 5 | 32768 | 34.595 | 34.471 | -0.36% |
| 5 | 65536 | 31.203 | 31.240 | +0.12% |

Across the complete CUDA prefill/decode matrix, changes range from -0.36% to
+0.32%. The selected partitions match baseline at these measured depths, while
all candidates remain available on other resource/workload combinations.
This supports practically unchanged CUDA throughput on this workload, not a
cross-device performance guarantee.

### Vulkan model matrix

The same model and baseline/candidate procedure uses:

```powershell
& $Bench -m $Model -dev Vulkan1 -ngl 99 -fa on `
  -ctk kvarn4 -ctv kvarn4 -p 512 -n 64 -d 0,4096 `
  -b 2048 -ub 256,512 -t 8 -r 5 -o json
```

Baseline executable: `tmp/issue183/baseline-vulkan/llama-bench.exe`.
Candidate: `tmp/issue183/adaptive-vulkan-bin/llama-bench.exe`.
Vulkan1 is the first RTX 3090; Vulkan0 is the host's integrated AMD GPU.
Mean throughput in tokens/s:

| Ubatch | Test | Depth | Baseline | Candidate | Change |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | pp512 | 0 | 385.756 | 422.487 | +9.52% |
| 256 | tg64 | 0 | 32.918 | 32.498 | -1.28% |
| 256 | pp512 | 4096 | 101.248 | 104.855 | +3.56% |
| 256 | tg64 | 4096 | 18.161 | 18.189 | +0.16% |
| 512 | pp512 | 0 | 412.168 | 412.137 | -0.01% |
| 512 | tg64 | 0 | 32.745 | 32.586 | -0.48% |
| 512 | pp512 | 4096 | 105.713 | 105.680 | -0.03% |
| 512 | tg64 | 4096 | 18.253 | 18.295 | +0.23% |

The new 256-token workspace route improves prefill in these measurements.
Shallow decode varied between runs. An earlier balanced tg128 comparison was
about 0.6% below baseline, within the 1.6% spread between its baseline runs.
After removing a repeated global pipeline-table lookup from WHT dispatch, a
balanced baseline/candidate/candidate/baseline tg128 comparison measured
32.504 / 32.500 / 32.386 / 32.339 tokens/s: pooled means 32.422 versus 32.443
(+0.07%). This supports effectively unchanged decode throughput, not a speedup
or proof of exact equality. An isolation run removing only the duplicate-stream
safety check did not improve decode, so that safety check is retained.

The ubatch-256 default-priority balanced follow-up contained a noisy candidate
run (31.532 tokens/s, standard deviation 1.081, with two approximately 30-token/s
samples versus the other samples near 32). Its raw negative result is retained.
A separate, identically configured baseline/candidate/candidate/baseline control
using `-p 0 -n 128 -d 0 -ub 256 -r 11 --prio 1` measured
32.035 / 32.121 / 32.250 / 32.163 tokens/s: pooled means 32.099 versus 32.185
(+0.27%). No GPU clocks or power limits were changed. Concurrent NVML telemetry
recorded active Vulkan-device SM clocks from 1245 to 1605 MHz, power-cap events
on 342 of 349 high-utilization samples, and no thermal-slowdown events. This
establishes a variable power-limited test environment, not a proven cause for
any particular earlier slow sample. Neither the negative short runs nor these
small positive controls justify claiming a decode speedup.

### Correctness and limits

- Both CUDA and Vulkan benchmark/test builds succeeded.
- Nine selected CTests passed, including full CUDA `test-kvarn`, both new host
  policy tests, CUDA route/quant policy tests, and HIP/port source checks.
  After the final vector-cache atomic change, the suite and synthetic SWA
  fixture passed again. Final provided-model 4K spot checks measured
  1378.101 pp / 39.406 tg tokens/s for k4/v4 and 1372.760 pp / 39.224 tg for
  k5/v5; both use the matrix command with only `-d 4096`.
- Full Vulkan `test-kvarn` passed on RTX 3090, including CPU/GPU attention parity
  and multistream store byte parity. CUDA-specific route-attestation checks are
  intentionally skipped by that Vulkan executable.
- The expanded short-store route initially failed the existing duplicate-stream
  byte-parity case. The validator fix passed the unchanged assertion and the
  full suite. Additional failure diagnostics identify head width, stream case
  and store mode without weakening coverage.
- A synthetic D256/GQA2 SWA attention fixture compared automatic vector splits
  against forced 16-token splits at 512/4096 KV tokens with k4/v4 and k5/v5.
  Numerical parity passed. Timing includes attention graph execution without
  re-running store quantization; it is not a Qwen model throughput claim.
- No HIP runtime, other GPU architecture, DFlash/MTP acceptance-rate benchmark,
  or cross-device performance claim is established here. Existing qualification
  and precision guards remain; user feedback is still needed for those devices.

Raw JSON samples, stderr, build/test logs, the model fingerprint and exploratory
comparisons are retained under `tmp/issue183/`. Final CUDA matrix files are
`reference-cuda-k{4,5}.json` / `accepted-cuda-k{4,5}.json`; final Vulkan matrix
files are `reference-vulkan-ship.json` / `ship-vulkan-ship.json`. Earlier
rankings and overlapping exploratory runs are not the final CUDA performance
comparison. The checked implementation uses the two-wave population preference
and 95-percent work band described above.
