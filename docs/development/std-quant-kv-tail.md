# Standard quantized KV exact-tail decision record

This record freezes the implementation scope that began at commit
`db7e2945f90a2e400df1ce5d635507c8063aea1d`. The product contract is in
`docs/beellama-features.md`; this page records the implementation decisions and
the evidence needed to change them.

## Cache and backend inventory

The applicable standard body registry is `q8_0`, `q6_0`, `q6_1`, `q5_0`,
`q5_1`, `q4_0`, `q4_1`, `iq4_nl`, `q3_0`, `q3_1`, `q2_0`, and `q2_1`.
Cache-facing `q2_0` is `GGML_TYPE_Q2_0S`. Ordered K/V pairs, one-sided
quantized caches, full attention, SWA, hybrid recurrent/attention wrappers,
layer reuse, K-only MLA, DSV4 raw SWA attention, and context-local shadows over
a shared body are in scope. DSV4 compressed block caches, recurrent state,
DSA/DSV4 lightning-indexer state, and other non-attention auxiliary memories
are not.

The backend completion set frozen from the starting tree is CPU, CUDA,
HIP/ROCm, Metal, Vulkan, SYCL, and CANN. CPU uses the generic ggml route. CUDA
uses native indexed-pool attention when the ordinary body pair has a supported
FlashAttention route and otherwise uses the backend-neutral `GET_ROWS`, matrix,
softmax, and `OUT_PROD` graph. HIP and the remaining backends may schedule that
same generic graph through their supported operations. A backend must fail
context creation rather than ignore a positive tail if neither route can be
placed.

Local hardware evidence is available for CPU and an RTX 3090 CUDA 13.1 build.
Metal, Vulkan, SYCL, CANN, and ROCm require their normal backend build packages
and hardware runs; absence of those devices is a verification gap, not a claim
that they passed.

Standard and KVarN CUDA FlashAttention instance policies are independently
selectable. `GGML_CUDA_FA_ALL_QUANTS=ON` continues to enable every ordered
standard pair. The standard-tail iteration build sets `GGML_CUDA_KVARN_FA=OFF`,
which compiles no KVarN FlashAttention translation units or template pairs.

## Attention decision

The implementation uses body/tail partials with one global softmax. Each query
uploads up to `N` physical shadow indices. The generic route gathers those
rows, masks the corresponding quantized body rows, concatenates logits, runs
one FP32 softmax, and then combines body and tail V products.

CUDA avoids graph-level query-by-query K/V materialization. Ordinary
FlashAttention writes its final `(row max, denominator)` metadata, and a fused
indexed tail kernel reads the persistent compact K/V pool plus the per-query
slot map. It accumulates the exact BF16 or F16 contribution into the same FP32
normalization state. Tail work and scratch therefore scale with `N`, not the
64K body, query count times pool payload, or unused capacity. IQ4_NL uses the
MMA body route and a view-aware non-contiguous IQ4-to-F16 converter; it does not
fall back to whole-cache graph materialization.

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

The quantized body is written eagerly for every token. Exact payload slots use
`(stream, cell, generation)` identities, per-sequence recent membership, and
reference counts. Current-ubatch entries remain available as rollback reserve;
the next preparation trims committed history to the configured length. A
cross-stream sequence copy remaps identities and copies only live exact rows.
Cache clearing, removal, keep, add, divide, state restore, cell recycling, and
RoPE position shift update both representations.

Persistent capacity is `N * n_seq_max + n_ubatch` slots. The first term is
active per-sequence membership and the second is stable in-flight reserve.
Payload tensors are allocated once for CUDA graph address stability. Only a
quantized side receives a shadow.

The reserve boundary is the physical ubatch, not the logical batch. Prompt
ubatches execute sequentially, so a completed ubatch is trimmed to N before
the next graph; speculative verification and its rollback snapshots remain in
one scheduler ubatch. DSV4 sizes this bound for its maximum coupled-sequence
raw-row fanout. Retaining `n_batch` rows was rejected because it adds a large,
N-independent persistent allocation without extending the supported rollback
contract.

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
explicit body-only export uses a framed standard-memory container with magic,
version, structural group ID, body and tail byte lengths, resolved length, and
tail type. The tail section stores body-relative identities, generations,
sequence membership, insertion order, layer layout, and referenced K/V rows.
F16 and BF16 are not converted during restore.

The reader accepts the preceding unframed body-only payload. Coverage then
reports none or partial with `LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE` until
original activations refill the tail. Tail-bearing state rejects a disabled,
differently sized, differently typed, or structurally different context.

Coverage is available per sequence and group and as a context aggregate. The
server exposes requested/exact tokens, complete/partial/none group counts, and
degraded-sequence counts through its metrics result and Prometheus endpoint.
The RAM prompt cache retains the framed exact suffix; trimming a reused prompt
uses ordinary sequence removal and therefore either retains a complete direct
suffix or exposes historical degradation rather than reconstructing rows.

## Local artifact and benchmark manifest

- Qwen hybrid target: `D:/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q5_K_S.gguf`
- Gemma SWA/global target: `D:/models/gemma-4-12b-it-GGUF/gemma-4-12b-it-UD-Q8_K_XL.gguf`
- Build: `tmp/build-local-3090-cuda13.1-default.ps1 -Parallel 16`
- GPU: RTX 3090, CUDA 13.1, architecture 86
- Required comparison controls: identical prompt or corpus, `-b`, `-ub`,
  context, sampler, cache pair, model file, and commit

No pure full-attention target or complete retrieval/agentic validation corpus
exists under the authorized `D:/models` artifact root. No file was downloaded.
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
