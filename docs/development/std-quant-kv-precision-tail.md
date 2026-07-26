# Standard quantized KV precision-tail decision record

This record freezes the implementation scope that began at commit
`db7e2945f90a2e400df1ce5d635507c8063aea1d`. The product contract is in
`docs/beellama-features.md`; this page records the implementation decisions and
the evidence needed to change them.

## Cache and backend inventory

The applicable standard body registry is `q8_0`, `q6_0`, `q6_1`, `q5_0`,
`q5_1`, `q4_0`, `q4_1`, `iq4_nl`, `q3_0`, `q3_1`, `q2_0`, and `q2_1`.
Cache-facing `q2_0` is `GGML_TYPE_Q2_0S`. Ordered K/V pairs, one-sided
quantized caches, full attention, SWA, hybrid recurrent/attention wrappers,
layer reuse, DSV4 raw SWA attention, and context-local shadows over a shared
body are in scope. Positive K-only MLA and DSA overlays are rejected during
context creation: this release does not define a precision-tail composition for
those selectors. DSV4 compressed block caches, recurrent state, DSA/DSV4
lightning-indexer state, and other non-attention auxiliary memories are not.

The backend completion set frozen from the starting tree is CPU, CUDA,
HIP/ROCm, Metal, Vulkan, SYCL, and CANN. CPU uses the generic ggml route. CUDA
uses native indexed-pool attention when the ordinary body pair has a supported
FlashAttention route and otherwise uses the backend-neutral `GET_ROWS`, matrix,
softmax, and `OUT_PROD` graph. HIP and the remaining backends may schedule that
same generic graph through their supported operations. A backend must fail
context creation rather than ignore a positive tail if neither route can be
placed.

Local hardware evidence is available for CPU and an RTX 3090 CUDA 13.1 build.
Backend status is evidence-qualified:

| Backend | Standard overlay status | Evidence |
|---|---|---|
| CPU | Generic graph | Hardware verified by registry-derived operation and numeric oracles |
| CUDA | Native attached tail and generic graph | Hardware verified on RTX 3090 / CUDA 13.1 |
| Vulkan | Generic graph | Source supported; hardware not run in this remediation |
| HIP/ROCm, Metal, SYCL, OpenCL | Generic route when every required operation is accepted | Source supported; hardware not run |
| CANN | Rejected at context creation | Fused shadow `SET_ROWS` is rejected; Windows static classification verified, no Ascend hardware claim |

Source support is not a hardware pass. Any backend that rejects a required
operation fails context construction instead of silently dropping the tail.

`GGML_CUDA_KVARN=ON` compiles KVarN CUDA kernels by default, with 15 balanced
fast-decode pairs in the standard build and all 36 ordered pairs when
`GGML_CUDA_FA_ALL_QUANTS=ON`. The standard-tail iteration build sets
`GGML_CUDA_KVARN=OFF`, which compiles no KVarN CUDA kernels or template pairs.

## Attention decision

The implementation uses body/history/current partials with one global softmax.
Each query uploads up to `N` committed exact-history indices plus its causal
current-ubatch rows. The generic route gathers the shared history payload,
masks the corresponding quantized body rows, concatenates logits, runs one
FP32 softmax, and combines the V products. It never creates a separate
`[heads, sources, queries]` K/V payload.

CUDA avoids graph-level query-by-query K/V materialization. Ordinary
FlashAttention writes its final `(row max, denominator)` metadata, and a fused
segmented tail kernel reads the persistent compact K/V pool and graph-local
current K/V plus the per-query descriptor. It accumulates both exact segments
into the same FP32
normalization state. Tail work and scratch therefore scale with `N`, not the
64K body, query count times pool payload, or unused capacity. IQ4_NL uses the
MMA body route and a view-aware non-contiguous IQ4-to-F16 converter; it does not
fall back to whole-cache graph materialization.

Standard quantized tails default to BF16. An explicit `--kv-tail-type f16`
request remains supported and takes precedence over that default.

The rejected prototype exposed every reserved shadow slot to softmax. Although
masked slots were mathematically zero, changing `n_seq_max` changed reduction
width and could change sampled output. It also imposed work proportional to
other sequences' capacity. The query-compact descriptor removes both problems.

Relative-position or model-specific attention bias follows the same source
selection as K/V. Mask preparation emits a query-specific flattened body-bias
row index for every exact entry; the generic graph gathers those rows and
concatenates them with the tail logits before the single softmax. This is
required for biased cached attention (for example T5) and also keeps DSV4's
explicit zero-bias generic route shape-correct.

Two prototypes were rejected. Graph-level K/V gathers repeated the compact pool
for every query and inflated both transient VRAM and 64K runtime. A second
independently normalized FlashAttention result was mathematically unsuitable
until every body kernel exposed final normalization metadata. The shipped
route instead exports metadata from vector, tile, WMMA, MMA, stream-K fixup,
and KVarN-compatible body kernels and performs one verified FP32 merge. Mixed,
tail-only, BF16-distinguishing, 512-query, long-body stream-K, and IQ4_NL
oracles cover this boundary.

## Ownership and lifecycle

The quantized body is written through its existing safe path. Exact payload
slots use `(stream, cell, generation)` identities, per-sequence recent
membership, and reference counts. Current-ubatch K/V remains graph-local until
attention has consumed the old history, then an ordered node commits it to the
ring. A successful graph commits prepared host metadata; a failure after a
possible compact write invalidates every affected stream. A
cross-stream sequence copy remaps identities and copies only live exact rows.
Cache clearing, removal, keep, add, divide, state restore, cell recycling, and
RoPE position shift update both representations.

Body membership and public position queries change immediately after a
sequence copy. Exact payload copies may remain deferred until a consumer needs
them; coverage and state-size queries account for pending work without
synchronizing, while state-data save materializes it with at most one batched
barrier.

Compact persistent capacity is `(N + R) * n_seq_max`, where `N` is the resolved
tail and `R` is the promised suffix-rollback horizon. The physical ubatch `U`
sizes only the graph-local current segment and transient attention workspace.
Payload tensors remain address-stable for CUDA graphs, but neither `U` nor
256-row logical padding is multiplied across layers. Each logical sequence owns
a distinct exact history even when the ordinary body is unified.

Partial SWA retains its compressed `W + U` body and adds `N + R` exact rows.
Full-window SWA instead owns a bodyless `W + R` exact ring. Rollback through
`R` is exact; a larger suffix removal is rejected before mutation so the caller
can restore a checkpoint. `R` is never inferred from `U`.

Cell-local removal updates only that sequence's membership, preserving shared
exact rows for other sequences. Range copies/removals and partial position
mutations that can make an evicted row recent report historical-operation
degradation. N subsequently committed original activations clear that reason.
Partial sequence copies union source and destination degradation reasons; they
never erase an existing destination reason merely because the copied source is
complete.
The starting standard cache has no active defragmentation operation; a future
defragmenter must remap `(stream, cell, generation)` atomically and extend the
lifecycle tests before it can be enabled.

## State format

Length zero retains the preceding unframed body format. A positive tail or an
explicit body-only export uses a local precision-tail manifest version 3. The
manifest validates structural group identity, body/tail byte lengths, resolved
length, compact representation, rollback horizon, layer layout, payload counts,
and F16/BF16 type before logical metadata is installed. Tail rows can
be transferred by the host or `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` tensor
protocol; F16 and BF16 are not converted during restore. Metadata-only owners,
including KVarN-backed components, still restore identities even when they own
no standard tail tensors.

The reader accepts the preceding unframed body-only payload. Manifest version 2
remains valid for legacy non-compact representations; compact contexts require
version 3 metadata. Version 1 has no complete provenance, so it restores with
conservative degraded coverage and cannot upgrade itself to exact. Body-only
state likewise reports `LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE` until original
activations refill the tail. Tail-bearing state rejects a disabled, differently
sized, differently typed, or structurally different context before mutation.

Coverage is available per sequence and group and as a context aggregate. The
server exposes requested/exact tokens, complete/partial/none group counts, and
degraded-sequence counts through its metrics result and Prometheus endpoint.
The RAM prompt cache retains the framed exact suffix. Standard unified and
non-unified caches preserve a continuous suffix across requests and message
boundaries; `cache_prompt=false`, slot eviction, or an incompatible
target/draft plan forces reevaluation. A unified idle slot is cleared only
after its complete target and draft state was saved. Failed or unsupported
save/restore is a cache miss, never permission to discard or overwrite live
state.

## Local artifact and benchmark manifest

- Qwen hybrid target: `<QWEN_TARGET_GGUF>`
- Gemma SWA/global target: `<GEMMA_TARGET_GGUF>`
- Build: `tmp/build-local-3090-cuda13.1-default.ps1 -Parallel 16`
- GPU: RTX 3090, CUDA 13.1, architecture 86
- Required comparison controls: identical prompt or corpus, `-b`, `-ub`,
  context, sampler, cache pair, model file, and commit

No pure full-attention target or complete retrieval/agentic validation corpus
exists under the configured model artifact root. No file was downloaded.
Consequently, `auto` remains conservative and has no nonzero entry; every
unknown key resolves to zero. The measurements below establish an explicit
Qwen operating point, but do not silently promote it to `auto` without the
remaining retrieval and cross-architecture gates.

## Qwen3.6 27B Q5_K_S evidence (RTX 3090)

All KLD runs used the same WikiText-2 corpus, BF16 reference file, 65,536
context, `-b 2048`, `-ub 512`, seed 1, unified KV, CUDA FlashAttention, and
symmetric body type. For q4_0, the geometric BF16-tail sweep was:

| Tail | Mean KLD | Same top | Elapsed |
|---:|---:|---:|---:|
| 0 | 0.004673 | 97.199% | 320.386 s |
| 1 | 0.004056 | 97.402% | 320.967 s |
| 64 | 0.003609 | 97.611% | 336.651 s |
| 128 | 0.003676 | 97.676% | 345.471 s |
| 256 | 0.003447 | 97.681% | 353.797 s |
| 512 | 0.003060 | 97.745% | 377.751 s |
| 1024 | 0.003093 | 97.773% | 416.524 s |
| 2048 | 0.003030 | 97.875% | 505.158 s |

Tail 64 is the measured knee for this key: it reduces mean KLD by 22.8% and
improves same-top by 0.412 percentage points while avoiding the rapidly rising
cost of larger tails. A fresh five-repetition paired `llama-bench` run measured
q4_0 body-only versus BF16 tail 64 at 2048 prompt tokens and 128 generated
tokens: prompt 1275.76 versus 1229.46 tok/s (-3.63%), generation 37.00 versus
36.02 tok/s (-2.64%). The persistent tail64 allocation was 36.5 MiB; the
reported 505.75 MiB CUDA compute buffer is the total graph workspace, not
tail-only overhead.

With BF16 tail 64 fixed, the symmetric standard-type matrix was:

| Body K/V | Mean KLD | Same top |
|---|---:|---:|
| q8_0 | 0.002667 | 98.020% |
| q6_0 | 0.002593 | 97.973% |
| q6_1 | 0.002744 | 98.014% |
| q5_0 | 0.003024 | 97.880% |
| q5_1 | 0.002662 | 97.906% |
| q4_0 | 0.003609 | 97.611% |
| q4_1 | 0.003724 | 97.711% |
| iq4_nl | 0.003385 | 97.681% |
| q3_0 | 0.006690 | 96.841% |
| q3_1 | 0.005714 | 96.897% |
| q2_0 | 0.025669 | 93.603% |
| q2_1 | 0.024560 | 93.630% |

Every completed matrix leg took 336.7-338.6 seconds. In particular IQ4_NL
completed in 336.843 seconds after the native strided-conversion fix; its prior
whole-cache fallback took about 1082 seconds per pass and was rejected.

## Gates

Tail zero must retain graph topology, allocation, output, and median speed,
with at most 1% noise. The 1024-2048 region must stay within 10% median decode
and 15% median prefill regression on a published representative matrix. Exact
state, body-only state, host sequence state, continuous batching, ubatch
partitioning, F16/BF16, all ordered body pairs, one-sided caches, and KVarN
removal independence are correctness gates. Missing local hardware must be
listed with exact build and run commands; it cannot be described as passing.
