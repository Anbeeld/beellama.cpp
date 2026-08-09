# KVarN precision-tail fit remediation completion record

- Date: 2026-08-09
- Source baseline: `83d4173d8`
- Implementation repository: `C:\Users\anbee\projects\beellama.cpp`
- Upstream reference: `tmp/upstream-llama.cpp`
- Result: complete on the available Windows Vulkan and CUDA hardware

## Outcome

Fit probes and final contexts now consume one immutable, model-independent tail
request. Each context binds that request to its own model group manifest, so
numeric, `auto`, positional, named, and intrinsic KVarN requests resolve
identically during fitting and final allocation. Invalid model-bound requests
fail context creation instead of silently becoming zero.

Vulkan standard quantized bodies with F16/BF16 tails use a bounded online
softmax operation for head dimensions 128, 256, and 512. KVarN body-plus-tail
operations no longer use the old mixed-tail preference as a graph-wide veto.
Both routes keep one global FP32 softmax across body, history, and current K/V;
neither builds the prior full score tensor or full-context F16 shadow.

Vulkan's KVarN store and split-K private workspaces are reported through the
backend registry and included in context compute memory. The reserve query uses
the same store planner and byte formula as runtime, but pessimistically covers
a full ubatch when a synthetic reserve graph has no real slot-contiguity hint.
For the reporter-shaped Qwen run, reserve reports 1.00 MiB store plus 0.50 MiB
split-K scratch. Runtime measured 976,896 bytes for the store route, so the
estimate covers the observed high-water.

When the immutable request is attached, upstream fit runs from pristine inputs,
then an exact no-allocation probe validates the returned candidate against the
original free-memory snapshot and margins. A positive shortfall is fed back as
a guarded margin before rerunning the existing upstream algorithm; repeated
non-fitting candidate identities terminate. With KVarN disabled and tail zero,
the request is not constructed and `common_fit_params` makes exactly its
original single call.

## Machine-readable matrix

The 52-row result table is
[`kvarn-tail-fit-remediation-matrix-20260809.csv`](kvarn-tail-fit-remediation-matrix-20260809.csv).
It covers Qwen3.6 and Gemma 4, contexts 50,000 and 100,000, ubatches 16 and
512, ordinary q4_0, F16/BF16 and automatic standard tails, KVarN4 intrinsic
and longer tails, mixed KVarN widths, and Gemma named and positional full/SWA
policies. Every row completed successfully.

Selected Vulkan device-memory estimates at ubatch 512 are:

| Model/configuration | Context | Device context MiB | Device compute MiB |
|---|---:|---:|---:|
| Qwen q4_0, tail 0 | 50,000 | 1,031 | 505 |
| Qwen q4_0, tail 128 F16 | 50,000 | 1,039 | 505 |
| Qwen q4_0, tail 1,024 BF16 | 50,000 | 1,095 | 505 |
| Qwen q4_0, `auto` | 50,000 | 1,095 | 505 |
| Qwen KVarN4, intrinsic 128 | 50,000 | 1,039 | 566 |
| Qwen KVarN4, explicit 1,024 | 50,000 | 1,095 | 567 |
| Qwen KVarN3/KVarN5, intrinsic 128 | 50,000 | 1,039 | 566 |
| Qwen KVarN4, intrinsic 128 | 100,000 | 1,892 | 506 |
| Gemma q4_0, tail 0 | 50,000 | 1,440 | 533 |
| Gemma q4_0, named full=1,024/SWA=128 | 50,000 | 1,620 | 533 |
| Gemma KVarN4, intrinsic 128 | 50,000 | 1,740 | 538 |
| Gemma KVarN4, named full=1,024/SWA=128 | 50,000 | 1,810 | 538 |

Compared with the audited baseline at context 50,000 and ubatch 512, q4_0
tail zero remains 505 MiB compute; KVarN4 falls from 6,357 to 566 MiB;
standard tail 128 falls from 7,956 to 505 MiB; and standard tail 1,024 falls
from 8,979 to 505 MiB. At context 100,000, KVarN4 is 506 MiB rather than
11,281 MiB and standard tail 128 is 505 MiB rather than 15,074 MiB. Doubling
context therefore no longer recreates the generic-score compute slope.

## Builds and tests

Hardware and software:

- Windows 11 Pro 10.0.26200
- NVIDIA GeForce RTX 3090, 24,576 MiB, driver 596.21
- Vulkan device `Vulkan0`, subgroup 32, FP16 and BF16 enabled
- CUDA 13.1, compute capability 8.6

Builds:

- `scripts/build-win-vulkan.ps1 -BuildName build-win-vulkan-remediation-tests -AllTests -Parallel 16`: passed; every static test target built.
- `scripts/build-win-vulkan.ps1 -BuildName build-win-vulkan-remediation-runtime -SkipStage -Parallel 16`: passed; final shared runtime, server, CLI, tools, and tests built.
- `scripts/build-win-cuda-13.1-sm_86-default.ps1 -BuildName build-win-cuda-13.1-sm_86-default-remediation -AllTests -Parallel 16`: passed; every CUDA default-matrix static test target built.

Test results:

- CUDA full CTest: 94/94 passed, including the complete backend-operation test.
- Final CUDA `test-kvarn`: passed after the reserve-workspace regression was added.
- Vulkan CTest excluding the unrelated baseline backend aggregate: 93/93 passed.
- Vulkan `FLASH_ATTN_EXT` backend matrix: 4,776/4,776 supported cases passed out of 5,139 selected cases.
- Focused request, fit-retry, standard-tail graph, KVarN CPU/Vulkan parity, state save/load, and eval-callback tests passed.
- Python source/integration guards passed, but runtime graph and numerical tests are the primary route evidence.

The unfiltered Vulkan CTest run exposes an existing `MUL_MAT_ID` test case with
an intentionally four-byte-offset RHS view. It aborts at the pre-existing
`ggml_vk_tensor_subbuffer` alignment assertion after the first eight
`MUL_MAT_ID` cases. The same assertion and call path are present at baseline
and in the local upstream reference, and this change does not touch that path.
It was retained rather than broadening this remediation into ordinary Vulkan
matrix multiplication. CUDA's full backend matrix passes. Evidence is captured
in `artifacts/kvarn-tail-fit-remediation/backend-mul-mat-id.log` and the full
Vulkan CTest logs.

## Real-model fit and runtime validation

Models:

- `D:\models\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q5_K_S.gguf`
- `D:\models\gemma-4-31B-it-GGUF\gemma-4-31B-it-Q4_0.gguf`

The final Qwen KVarN4 intrinsic-tail fit used context 50,000, batch/ubatch 512,
FlashAttention, all layers requested on Vulkan, and a 1,024 MiB fit margin. It
returned `-c 50000 -ngl 999`, projected 17,235 MiB model + 1,039 MiB context +
566 MiB compute, and logged `exact Bee post-fit validation passed`. The q4_0
tail-zero control returned the same placement after one upstream fit call and
emitted zero Bee exact-validation or retry markers.

Final server runs used one slot, context 50,000, batch/ubatch 512, fit enabled,
and hidden redirected processes. Each processed a minimal prompt, a prompt near
one full ubatch, and two forced decode tokens.

| Case | Route | Prompt tokens | Prompt ms | Decode tokens | GPU MiB before/context/after full prompt |
|---|---|---:|---:|---:|---:|
| Qwen KVarN4 + F16 tail 1,024 | `portable-native`, no fallback | 481 | 1,283.045 | 2 | 695 / 19,228 / 19,256 |
| Qwen q4_0 + BF16 tail 1,024 | `bounded-online-softmax`, workspace 0 | 481 | 1,105.192 | 2 | 695 / 19,228 / 19,257 |
| Gemma q4_0 named full=1,024/SWA=128 F16 | bounded online softmax at D=512 and D=256 | 482 | 17,729.153 | 2 | 695 / 19,088 / 19,116 |

The final Qwen KVarN runtime used 976,896 bytes of store workspace and increased
device use by only 28 MiB between initialized context and the completed full
prompt. No multi-gigabyte VRAM-to-host migration or context-sized
materialization occurred. The final reserve estimate was 1.50 MiB combined
private scratch, greater than the observed store allocation plus its separately
reserved 0.50 MiB split-K buffer.

Complete estimator, route, server, response, and GPU snapshot logs are under
`artifacts/kvarn-tail-fit-remediation/` and are intentionally excluded from Git.

## Diff audit and preserved behavior

- `common/common.cpp` constructs and attaches the immutable request only for
  KVarN or a nonzero tail.
- `common/fit.cpp` enters the new exact-validation loop only when
  `cparams.kv_tail_request != nullptr`; the null branch calls the original
  `common_params_fit_impl` once with the original arguments.
- Retry state may adjust only margins and always restores pristine model,
  context, split, override, and request inputs before upstream fit. It never
  chooses context, layer placement, splits, or buffer overrides.
- The larger graph-node allowance is conditional on an effective tail.
- Vulkan's new operation is selected only when tail operands are present and
  its complete shape/type/layout predicate succeeds. Ordinary FlashAttention
  and non-tail backend behavior retain their existing paths.
- KVarN graph changes remove only the fork-owned preference veto and retain the
  backend's final-operation support check.
- No fit implementation was copied, renamed, or added; `common_fit_params_impl`
  remains the sole placement/search algorithm.
- `tmp/upstream-llama.cpp` was not modified.
- The worktree was clean at task start, so there were no unrelated user changes
  to overwrite or retain.

## Hardware gaps

The requested Linux RADV RX 9070-class system, 16 GiB AMD GPU, GTT counters,
and second physical GPU were not available on this Windows host. Consequently,
the RADV-specific before/context/short/full VRAM+GTT capture and explicit
single-device versus second-GPU repetition remain external hardware gates.
Vulkan route and numerical coverage was completed on the RTX 3090; HIP static
validation tests passed, but no claim is made for an unavailable AMD runtime.
