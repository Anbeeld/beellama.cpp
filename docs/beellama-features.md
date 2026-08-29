# BeeLlama v0.4.4 features

BeeLlama v0.4.4 keeps a small fork surface on top of upstream llama.cpp. Use
this page to choose a feature; use the [argument reference](beellama-args.md)
for exact names, environment variables, defaults, and validation ranges.

## KVarN target KV cache

### What it is

KVarN is Huawei's calibration-free, variance-normalized KV-cache quantizer,
adapted here for llama.cpp. It applies a per-head Hadamard rotation after RoPE,
normalizes both axes of each 128-token tile, and stores structured 2-, 3-, 4-,
5-, 6-, or 8-bit records with scale metadata. K and V widths are independent,
and supported Qwen 3.6 and Gemma 4 SWA layers can use a separate KVarN pair.
Non-SWA layers keep the first 128 attention-sink tokens exact. Bee also keeps at
least the newest 128 tokens exact, unlike the reference implementation's
partially filled suffix. The physical ubatch controls only temporary workspace;
it never enlarges that logical exact suffix.

### When to use it

Use KVarN when persistent KV-cache memory is the limiting resource and the model
has a supported attention layout. The general benchmark ladder starts with
`kvarn5 / kvarn4` and a 1024-token tail as its balanced default, with
`kvarn6 / kvarn5` as the more conservative body and `kvarn4 / kvarn3` as the
smallest recommended tier. These presets come from Qwen 3.6 27B measurements,
not a universal model guarantee. Measure the quality, prefill speed, generation
speed, and memory use of the exact model, backend, and context you plan to
serve.

See [KVarN KV Cache: Implementation and Benchmarks](https://anbeeld.com/articles/kvarn-kv-cache-implementation-and-benchmarks) for the algorithm, Bee's exact-tail deviation, and matched-format comparisons. The current combined recommendation ladder is in the [README](../README.md#general-kv-cache-ladder).

### Key arguments

- [`--cache-type-k`](beellama-args.md#kvarn-cache-types-and-swa-overrides)
- [`--cache-type-v`](beellama-args.md#kvarn-cache-types-and-swa-overrides)
- [`--cache-type-k-swa`](beellama-args.md#kvarn-cache-types-and-swa-overrides)
- [`--cache-type-v-swa`](beellama-args.md#kvarn-cache-types-and-swa-overrides)
- [`--kv-tail-tokens`](beellama-args.md#kv-cache-precision-tail-for-quantized-caches)
- [`--kv-tail-type`](beellama-args.md#kv-cache-precision-tail-for-quantized-caches)

With KVarN, omitted `--kv-tail-tokens` and numeric `0` both retain the
intrinsic 128-token exact suffix. A positive request enlarges that suffix,
rounding upward to complete 128-token KVarN groups and capping at the group's
full or SWA visibility window. `auto`, positional lists, named roles, and
structural group IDs use the same group resolution as standard-cache tails.
F16 is the paper-faithful KVarN default. Standard quantized tails default to
BF16. Either cache family accepts an explicit F16 or BF16 override.
A request covering the whole group uses one native F16/BF16 cache instead of
allocating compressed records plus a redundant exact overlay. For SWA this is
a compact `W + R` ring; it does not retain the physical `W + U` execution
reserve as persistent exact payload.

KVarN durable prompt-cache reuse remains descriptor-group aligned: current G128
presets publish and restore checkpoints only at complete 128-token boundaries.
The bounded live exact frontier still supports the existing speculative
micro-rollback path, but sealed compressed history is never split into an
arbitrary-position record. Standard KV is not subject to this group boundary;
its precision-tail state can restore a validated logical position while a hot
partial checkpoint references the still-resident body instead of rereading it.

### Measurement and validation

Run KLD or perplexity with the same corpus, context, batch size, and cache pair
as the intended workload. Keep both `-b` and `-ub` identical between baseline
and candidate runs. Record the model file, command, prompt or corpus, sampling
settings, GPU, and commit with every result.

The CUDA specialized split, SWA-vector, and tiled descriptor-native MMA routes
publish the same optional FP32 `(maximum, denominator)` metadata as upstream
FlashAttention. Single-token generation uses split/vector decode, while short
multi-token verification uses tiled MMA to reuse decoded K/V tiles across query
rows. Q9-Q16 batches with GQA above four use a fused 128-column tile when the
concrete kernel fits the device's opt-in shared-memory budget and has nonzero
measured occupancy; all other devices and shapes retain the regular tile
matrix. An attached precision tail therefore does not force KVarN away from its
shape-selected native route. `llama-bench` reports requested and effective
tail sizes plus split, vector, generic, and prefill route counts so this remains
observable.

The July 2026 RTX 3090 / CUDA 13.1 recovery run used the Qwen 3.6 27B Q5_K_S
and Gemma 4 31B Q5_K_S models, `-b 2048 -ub 512`, 128 decode tokens, and five
repetitions in the canonical 56-row matrix. Ratios below use each run's matched
BF16 median at the same depth:

| Model | Context | KVarN4 request 0 / effective 128 | KVarN4 request 1024 | 1024 versus default |
|---|---:|---:|---:|---:|
| Qwen 3.6 27B | 16K | 0.983x | 0.971x | 0.988x |
| Qwen 3.6 27B | 32K | 1.060x | 1.048x | 0.989x |
| Qwen 3.6 27B | 64K | 1.176x | 1.168x | 0.993x |
| Gemma 4 31B | 16K | 0.839x | 1.116x | 1.330x |

No accepted KVarN row used generic MMA fallback. Qwen used split decode;
Gemma request-zero exercised both D512 split and D256 SWA-vector decode. A
requested 1024-token tail covers Gemma's SWA group with native-exact storage.
These ratios predate compact current-source storage and are retained as baseline
evidence, not as a performance claim for the current implementation.

`llama-bench --kv-memory` enables synchronized CUDA checkpoints and cache-owned
component accounting. It is intentionally opt-in and excluded from speed runs.
The corresponding 18-row measurement found:

| Model / context | Request | Q4_0 KV-related peak | KVarN4 KV-related peak | KVarN4 difference |
|---|---:|---:|---:|---:|
| Qwen 16K | 0 | 292.50 MiB | 395.21 MiB | +35.1% |
| Qwen 16K | 1024 | 503.70 MiB | 447.71 MiB | -11.1% |
| Qwen 64K | 0 | 1156.50 MiB | 1253.35 MiB | +8.4% |
| Qwen 64K | 1024 | 1559.70 MiB | 1305.85 MiB | -16.3% |
| Gemma 16K | 0 | 703.12 MiB | 1683.66 MiB | +139.4% |
| Gemma 16K | 1024 | 1936.89 MiB | 1823.04 MiB | -5.9% |

Request-zero is an explicit architectural tradeoff rather than a hidden
regression: KVarN must retain its intrinsic F16 suffix and compression staging,
while Q4_0 request-zero has neither an exact overlay nor tail-merge scratch.
Removing those KVarN allocations merely to beat Q4_0 would violate the quality
and precision-tail contract. At matched 1024/2048 requests KVarN retains a peak
advantage because its tail-attention transient high-water is smaller. From
Qwen 16K to 64K, exact and staging residency stay constant, compressed K and V
each grow by 420 MiB, and transient high-water grows by only 18.14 MiB; no
full-context exact mirror or unexpected context-sized metadata was found.

Compact native-exact storage is reported separately from compact exact-overlay
bytes, including exact-history, rollback-reserve, and transient estimates.
CUDA allocation remainders are also reported with their sign: positive values
are non-KV scheduler/graph/backend reservations, while negative values denote
allocator reuse/overlap or driver-baseline release rather than negative cache
ownership. Repeated grouped contexts showed only bounded first-use CUDA
reservation and zero cumulative per-context growth.

### Backend support and limitations

KVarN is target-context-only. CUDA selects specialized descriptor-native
FlashAttention on Turing and newer GPUs, then falls back to a portable
direct-record route when those matrix instructions are unavailable or the
complete body-plus-tail request does not fit a specialized route. The portable
CUDA route consumes rotated compressed records and attached F16 or BF16 tails
directly for D128, D256, and D512 heads. Its correctness limit is not the
specialized decode threshold of 16 queries, so prompt-sized query batches stay
native instead of creating a full F32 KQ tensor.

ROCm/HIP selects between record-tiled split decode, eligible descriptor-native
WMMA/MFMA, and the same portable direct-record kernel. Unsupported AMD matrix
shapes remain on portable native attention instead of materializing the cache.
CPU has a backend-native direct-record attention path. Vulkan directly consumes
KVarN records and exact tails for supported D128, D256, and D512 shapes. Its
standard-cache segmented route likewise consumes a quantized body, F16/BF16
history, and current K/V with one online FP32 softmax. Explicit materialization
remains a fallback for unsupported placements or shapes. Matrix-capable HIP and
CUDA retain descriptor-native large-batch prefill. Other portable backends
retain their advertised query limits. Vulkan requires shader Int64 and
buffer-device-address support. CPU placement is valid with KV offload disabled.

| NVIDIA architecture | Toolkit and package | Native KVarN route | Qualification |
|---|---|---|---|
| Turing and newer, SM 7.5+ | CUDA 12.4 or 13.3 | Specialized MMA/split/vector routes with portable fallback | Current release tier; CUDA 13.3 requires preview qualification, while its CUDA 13.1 predecessor was locally exercised on SM 8.6 |
| Volta, SM 7.0/7.2 | CUDA 12.4 | Portable direct body-plus-tail attention | Explicit build target; real-device validation required |
| Pascal, SM 6.0/6.1/6.2 | CUDA 12.4 | Portable direct body-plus-tail attention | Priority compatibility target for issue #112; real-device validation required |
| Maxwell, SM 5.0/5.2/5.3 | CUDA 12.4 | Portable direct body-plus-tail attention | Experimental until an SM 5.2 device passes runtime gates |
| Kepler | Not in the CUDA 12.4/13.3 release lane | None | Unsupported |

Use the CUDA 12.4 release package for Maxwell, Pascal, or Volta. CUDA 13.3
does not contain device code for those architectures. Build coverage proves
that a translation unit accepts a target; it does not prove runtime
correctness, memory behavior, or performance on that GPU.

| HIP architecture | Physical wave | Native KVarN route |
|---|---:|---|
| RDNA3, RDNA3.5, RDNA4 | 32 | WMMA generic/prefill and occupancy-selected split decode |
| CDNA1-CDNA4 | 64 | MFMA generic/prefill and physical-wave split decode |
| Older GCN, RDNA1, RDNA2 | device default | Portable direct-record attention |

CDNA fast routing is compiled and selected by capability but remains
experimental until hardware parity and performance results are published.
MUSA explicitly remains on the portable route.

Set `GGML_KVARN_DEBUG_ROUTES=1` to log the selected CUDA/HIP route, compute
capability, rotated/original domain, K/V bit widths, query and KV counts,
attached exact-tail rows and type for integrated entries, entry path, and
fallback reason. Startup tail-policy logs and `llama-bench --kv-memory` remain
the sources of requested and effective tail coverage for compact two-pass
entries.

Vulkan direct decode groups up to four GQA query heads per reconstructed K/V
head and uses split-K when the query/head grid alone cannot occupy the device.
Its split partials are reduced through the standard Vulkan FlashAttention
reducer, including attention sinks and exact-tail contributions.

For large dense prompt blocks, Vulkan validates per-stream record ownership on
the device and performs WHT, quantization, and record commit in parallel. The
route starts at 384 tokens per stream and reuses the cache conversion workspace
rather than adding a new peak allocation. Invalid or non-dense layouts use the
monolithic store fallback. Devices that permit 256- or 512-thread workgroups
use a full-head WHT shader; other devices keep the portable 128-thread path.
Both paths use physical subgroup operations and support subgroup widths 32 and
64. Vulkan materialization prepares the live history/current descriptor once
per stream before expanding output rows.

Set `GGML_KVARN_DEBUG_ROUTES=1` for bounded route diagnostics and
`GGML_VK_PERF_LOGGER=1` for synchronized per-operator Vulkan timestamps.
The route telemetry distinguishes native attention, explicit materialization,
parallel and monolithic stores, and rejected shapes.

`llama-bench --kv-memory` resolves route and transient-memory telemetry from the
configured benchmark device rather than the first loaded backend. Its JSONL
output includes the complete versioned route counters and synchronized device
allocation checkpoints. The historical `cuda_used_*` field names remain for
harness compatibility, but contain the selected device's values on Vulkan.
Vulkan reports backend-private KVarN workspace separately from graph-planned
transient bytes and clamps `VK_EXT_memory_budget` availability to the physical
heap, including the valid high-pressure case where reported usage exceeds the
current budget.

Each selected layer must be owned by a backend that implements KVarN store and
attention, or an explicitly supported materialization fallback. Unsupported or
tensor-split placements fail closed; draft and auxiliary contexts use standard
cache types.

## Standard low-bit KV caches

### What they are

Bee retains the standard cache types `q6_1`, `q6_0`, `q3_1`, `q3_0`, `q2_1`,
and `q2_0` across CPU and CUDA `SET_ROWS`/`GET_ROWS`, including CUDA
FlashAttention vector coverage. Cache-facing `q2_0` is internally
`GGML_TYPE_Q2_0S`, distinct from upstream's serialized Q2_0 weight format.

### When to use them

Use these types for draft caches, for target models that cannot use KVarN, or
when a conventional quantized KV layout is easier to compare across backends.

### Key arguments

- [`--cache-type-k`](beellama-args.md#kvarn-cache-types-and-swa-overrides)
- [`--cache-type-v`](beellama-args.md#kvarn-cache-types-and-swa-overrides)
- Upstream `--spec-draft-type-k`
- Upstream `--spec-draft-type-v`
- [`GGML_CUDA_FA_ALL_QUANTS`](beellama-args.md#cuda-flashattention-build-policy)

### Measurement and validation

Compare cache formats with identical model, context, corpus, `-b`, and `-ub`
values. A build with the default CUDA policy contains 50 standard vector pairs;
the ALL option contains 169.

### Known limitations

The user-facing `q2_0` cache name must not be treated as upstream's Q2_0 weight
format. A requested CUDA FlashAttention pair must be compiled by the selected
build tier.

## KV cache precision tails for quantized caches

### What it is

The KV cache precision tail (KVCPT), set with `--kv-tail-tokens`, makes the newest
attention-visible entries exact in F16 or BF16 for standard quantized and KVarN
target caches. A partial request overlays
a compact `N + R` exact-history ring while retaining the complete selected
quantized cache; current-ubatch K/V is a separate graph-local exact source.
A standard quantized tail defaults to BF16, while a KVarN tail defaults to F16;
`--kv-tail-type` can explicitly select either representation for either family.
A request covering an SWA group's full visibility window uses a bodyless
compact-native `W + R` F16/BF16 ring when the segmented route is supported.
Non-SWA full-context groups keep their established representation. Source selection is
per query, so a 512-token prefill does not make the same 512 rows exact for
every query. Body and tail logits share one FP32 softmax; the runtime does not
normalize two attention results independently.

Route selection also records whether each model layer supplies an explicit
self-attention bias. That capability is derived from loaded layer tensors, not
an architecture-name allowlist. A biased layer never selects a native route
that cannot consume its bias, and graph construction rejects any mismatch
between the recorded route and the actual bias tensor.

The exact-history pool is owned by the standard cache and identifies rows by
stream, physical cell, and generation. Sequence copies share exact rows within
one stream and copy them into context-local slots across streams. Position
shifts rotate exact K rows together with the quantized body. CUDA can read the
compact pool and current K/V directly through per-query descriptors and merge
against ordinary FlashAttention normalization metadata. Vulkan has a bounded
direct operation for standard quantized bodies plus F16/BF16 history and
current segments at head dimensions 128, 256, and 512. It evaluates the body,
history, and current keys with one online FP32 softmax and reports zero private
workspace for that route. The generic graph
composes persistent history and current K/V on the owning device, then gathers
the bounded per-query source union needed by the ordinary attention operators.
That fallback can duplicate graph-local source rows across a physical ubatch,
but never changes persistent capacity or materializes the full context.
Compact-native SWA allocates no compressed body.

Allocation, graph construction, graph identity, backend validation, and memory
telemetry consume one per-layer execution descriptor. It records the realized
body and exact types, body/bodyless representation, graph-local current
segment, padded body execution extent, explicit-bias requirement, owner device,
and selected route. A native route is checked against
`ggml_backend_dev_supports_op()` on the final fused operation, after all
descriptors and current sources are attached. Unsupported native shapes fail
closed before scheduling rather than appearing later as CPU/CUDA split
boundaries.

Exact overlays support both layer and tensor placement. Layer mode keeps each
body and K/V shadow on the device that owns the layer. Tensor mode projects the
body, standard shadow, KVarN records and staging, and exact history through one
typed cache-component split contract. Standard rows and exact tails split on
complete KV-head widths; KVarN records and staging split on their sliced-head
axis, so one head/group record never spans devices. A malformed split fails
during cache construction with the layer and component named in the error.

The local CUDA acceptance rows use two logical shards on one RTX 3090, including
`1,1` standard-tail and `3,1` KVarN-tail placement on Qwen3.6-27B at a 16K
context. Executable CPU/meta tests cover zero-head shards. This verifies local
projection and scheduling but not physical peer copies, NCCL, heterogeneous GPU
rollback, or per-device memory pressure; those remain external two-GPU gates.

KVarN differs from standard caches in three intentional ways. Its exact suffix
has a non-disableable 128-token floor, positive requests round upward to 128
tokens, and completed compressed records are written eagerly even while their
tokens coexist in the exact suffix. The KVarN body exports its FP32 row maximum
and denominator to the same tail merge used by ordinary FlashAttention. Sink,
body, and suffix masks therefore contribute each key exactly once. F16/BF16
canonical K/V rows are stored after RoPE for K and in the original V domain;
the compressed body retains KVarN's rotated-domain records.

Vulkan KVarN route reservation also queries the backend's own store and
split-K workspace planners. The resulting per-backend high-water is included in
the context compute breakdown before fit evaluates a candidate; runtime and
estimation therefore use the same sizing functions. With KVarN or a precision
tail active, fit performs an exact no-allocation validation of the upstream
candidate against the original device-memory snapshot. A measured shortfall is
fed back only as a guarded margin and upstream fit is restarted from pristine
inputs; repeated non-fitting candidates fail deterministically. Tail-disabled
ordinary caches take the unchanged single-call upstream fit path.

Standard unified and non-unified prompt caches preserve one continuous suffix
across requests and message boundaries. Unified KVarN uses the same shared
capacity model: completed 128-token record groups can be borrowed by any slot
instead of partitioning the context into fixed per-slot stripes. Only incomplete
record groups reserve cyclic F16 staging rows.

KVarN state is sequence-selective: live checkpoints retain only the selected
sequence's mutable frontier, exact-tail rows, and logical metadata, while
self-contained RAM state also owns that sequence's sealed record groups. Restore
validates the complete state before it publishes remapped destination records.
Other live sequences in a unified cache are neither gathered nor overwritten.

Prompt-cache planning distinguishes the lexical LCP, the boundary that all
target/draft/speculative components can restore, and the boundary actually
committed. Live slots and RAM entries are ranked by that safe restorable prefix,
not lexical similarity alone. KVarN durable checkpoints are eligible only on
complete descriptor boundaries; standard and recurrent caches retain upstream
batching and impose no Bee-specific 64-token cadence. `cache_prompt=false`,
slot eviction, corrupt or incompatible state, and the absence of one common
target/draft/speculative plan remain reason-coded misses.

RAM entries are immutable and repeatably restorable. Admission serializes a
self-contained target/draft/speculative candidate before publishing it and
clears an idle unified slot only after admission succeeds. Restore prepares and
validates every component without mutation, then commits the complete prepared
transaction; failed preparation leaves the destination unchanged without a
full-state preimage. Serialized checkpoint byte buffers use copy-on-write
sharing, but native KV blocks are not reference-counted or shared. Hot partial
state still emits logical-prefix manifests, so its work is not strictly
frontier-only. Accounted bytes include serialized entry payloads and count
shared immutable checkpoint buffers once; container capacity and allocator
overhead are intentionally excluded.

For hybrid iSWA with multiple slots, eligible non-SWA layers keep KVarN while
SWA layers use the explicit, warned, bit-width-matched standard-cache fallback.
This is a supported hybrid placement rather than record reinterpretation;
single-slot layouts can retain KVarN in both groups when otherwise eligible.

Partial SWA storage retains its upstream-aligned compressed `W + U` body, but
its persistent exact history is `(N + R) * S` rows. Full-window SWA omits that
body and stores `(W + R) * S` exact rows. `U` controls only graph-local current
K/V and transient workspace; backend byte alignment does not add logical rows.
Unified ordinary body storage does not merge these logical exact histories.
Positive K-only MLA and DSA overlays are rejected during context creation.

### When to use it

Use the precision tail when a quantized cache saves needed context memory but
recent-token quantization changes quality. In the Qwen 3.6 27B measurements, a
1024-token tail captured most of the gain for low-bit bodies and was the balanced
starting point for both KVarN and standard caches. Prefer body precision when
old and recent tokens matter equally. A 2048-token tail is most persuasive when
the newest two thousand tokens are genuinely the privileged working set; larger
tails read more F16/BF16 data and are not performance-neutral.

Tail length is a quality, memory, and throughput choice. It is not a
prompt-cache correctness switch: state restore carries the exact rows and
coverage metadata required by the configured tail, including a zero-length
standard tail and KVarN's intrinsic minimum suffix.

Gemma 4 needs a separate policy. Its 1024-token sliding window makes a 1024 tail
exact across most layers, causing a sharp memory and throughput transition.
Standard q8 without a tail is the safer general default when throughput and
older-context coverage matter. See [KV Cache Precision Tail: Implementation and Benchmarks](https://anbeeld.com/articles/kv-cache-precision-tail-implementation-and-benchmarks) and the [combined benchmark review](https://anbeeld.com/articles/kv-cache-quantization-benchmarks-kvarn-precision-tail) for the quality, memory, and throughput tradeoffs.

`auto` requests 1024 exact tokens for every applicable canonical target-cache
group and caps each request by that group's effective context or attention
window. It is deliberately architecture-agnostic and is not a claim that 1024
is the quality or performance optimum for a particular model.

### Key arguments and APIs

- [`--kv-tail-tokens`](beellama-args.md#kv-cache-precision-tail-for-quantized-caches)
- [`--kv-tail-type`](beellama-args.md#kv-cache-precision-tail-for-quantized-caches)
- `llama_kv_tail_config_*` for model-bound group discovery and overrides
- `llama_kv_tail_request_*` for immutable model-independent fit/final requests
- `llama_kv_tail_get_coverage` for per-sequence, per-group coverage
- `llama_kv_tail_get_coverage_aggregate` for context/server aggregation
- `LLAMA_STATE_SEQ_FLAGS_BODY_ONLY` for an intentional lower-precision state export

Implementation decisions and non-local hardware verification packages are in
[`development/std-quant-kv-precision-tail.md`](development/std-quant-kv-precision-tail.md) and
[`development/std-quant-kv-precision-tail-backend-verification.md`](development/std-quant-kv-precision-tail-backend-verification.md).
KVarN-specific workspace, attention, and state decisions are in
[`development/kvarn-precision-tail.md`](development/kvarn-precision-tail.md).

### State and compatibility

For standard caches, the default length is zero and preserves the ordinary
topology. KVarN always resolves at least its intrinsic 128-token suffix.
Sequence state writes validated KV-tail manifest version 5 and can
transfer tail tensors through host buffers or the on-device tensor protocol.
Version 5 records exact source cells and generations, local tail slots,
insertion order, and the per-sequence write cursor so selective unified-cache
restore cannot alias an unrelated sequence or reorder its exact frontier.
Overlay states reject a different structural group, resolved length,
representation, rollback horizon, KVarN preset, or F16/BF16 type before
mutation. Manifest version 2 remains readable for legacy non-compact layouts;
version 1 remains readable with conservative degraded provenance and cannot
upgrade incomplete historical evidence to exact. Native-exact state is already
present in the ordinary body and has no duplicate shadow section. Standard
body-only compatibility state and explicit body-only state begin with
observable degraded coverage and refill from original activations on later
writes. Sequence copies publish body membership and positions immediately;
deferred exact rows materialize in one batch when state data or another direct
consumer needs them. KVarN state version 15 stores sequence-selective logical
record groups, compact exact payloads, selected stage rows, and destination
record remapping independently of transient workspace. Partial live checkpoint
state omits sealed records already retained by the context; self-contained RAM
state owns them. Compatible version 12 and 13 state remains readable; version 11
is rejected because it serialized the old workspace-dependent layout. Dequantized
body rows are never labeled exact. Server metrics report requested and exact
tokens, coverage group states, and degraded sequences.

Completion timings expose `cache_lcp_n`, `cache_planned_n`,
`cache_reprocessed_n`, `cache_source`, and `cache_reason`; API cached-token usage
is the committed count. Prometheus metrics expose RAM admission/restore totals,
resident bytes, busy-slot overlap, and precision-tail coverage/degradation.

BeeLlama v0.3.x sessions and its v11 KVarN state are intentionally incompatible
with the v0.4.0 cache type IDs and logical-record format. Restore fails closed;
there is no compatibility reinterpretation or migration shim.

Restore is transactional for both host and on-device readers. Tensor writes and
KVarN metadata remain private until the complete frame, dimensions, layer set,
and payload lengths validate; any error cancels staged work and leaves the live
destination usable. Deferred standard-tail copy reports allocation/transfer
preparation and compute failures distinctly. A failed destination may be
evicted as cleanup, but state save, the next decode, sequence reuse, and server
handoff cannot observe it as a completed copy.

### Backend routes

CUDA, CPU, and Vulkan KVarN routes are hardware verified. The portable
CUDA/HIP implementation is also exercised on CUDA with the optimized paths
forcibly disabled. AMD WMMA/MFMA routing, physical-wave decode, and wave-aware
WHT are source-policy verified pending the configured ROCm CI build and AMD
hardware reports; no ROCm performance claim is made from CUDA results. Metal,
SYCL, and generic OpenCL contain the required generic operator
families but are not hardware-verified by this release. OpenCL's
Adreno-transformed weight layouts are rejected by these row operators; flat
standard-cache storage uses the generic kernels. CANN overlay contexts are
rejected at context creation because fused shadow `SET_ROWS` is not supported;
successful source classification is not an Ascend hardware claim. Startup
diagnostics name the selected native or generic route; generic long-context
attention can be substantially slower.

For compact current sources, CUDA and HIP share the segmented history/current
implementation. CPU has the dense one-softmax reference route. Vulkan consumes
the quantized body, compact history, and current buffers through buffer device
addresses, validates every source before native dispatch, and evaluates them
with one online FP32 softmax. Supported mixed and bodyless exact routes remain
native; unsupported shapes or source layouts fail closed or use the explicit
device materialization fallback instead of ignoring current rows or relying on
an accidental scheduler fallback.

### July 2026 compact-tail correction

The completion run used Gemma 4 31B Q5_K_S at context 16384, `-b 2048
-ub 512`, the persisted BF16 KLD baseline, and one repetition per requested
row. The GPU was concurrently occupied, so throughput was deliberately
excluded. Quality below is compared with the historical gain-curve article:

| Cache / tail | Result | Median KLD | Mean KLD | P99 KLD | P99.9 KLD | Maximum KLD | Same top |
|---|---|---:|---:|---:|---:|---:|---:|
| q5_0 / 0 | historical | 0.061747 | 0.626347 | 9.084101 | 18.731647 | 36.826859 | 76.802% |
| q5_0 / 0 | corrected | 0.061747 | 0.626347 | 9.084101 | 18.731647 | 36.826859 | 76.802% |
| q5_0 / 1024 | historical | 0.038569 | 0.469998 | 7.542499 | 16.978937 | 39.990627 | 79.989% |
| q5_0 / 1024 | corrected | 0.038596 | 0.462000 | 7.410694 | 16.760483 | 40.052086 | 80.185% |
| kvarn5 / 0 | historical | 0.041262 | 0.483087 | 7.594825 | 16.600363 | 33.928799 | 79.609% |
| kvarn5 / 0 | corrected | 0.041221 | 0.483046 | 7.608277 | 16.575455 | 33.928799 | 79.592% |
| kvarn5 / 1024 | historical | 0.036261 | 0.444201 | 7.325125 | 16.381313 | 38.840488 | 80.622% |
| kvarn5 / 1024 | corrected | 0.035761 | 0.445017 | 7.212839 | 16.502676 | 36.113117 | 80.590% |

The same corrected binary was measured in fresh processes with
`llama-bench --kv-memory --no-warmup -d 16384 -n 1`. These are cache-owned
bytes, not total-VRAM differences:

| Cache / tail | Persistent KV | Exact history + reserve | Staging | Persistent padding | Reusable transient high water | KV-related peak |
|---|---:|---:|---:|---:|---:|---:|
| q5_0 / 0 | 859.38 MiB | 0 | 0 | 0 | 0 | 859.38 MiB |
| q5_0 / 1024 | 1327.73 MiB | 880.86 MiB | 0 | 0 | 249.75 MiB | 1577.49 MiB |
| kvarn5 / 0 | 1170.70 MiB | 110.86 MiB | 220.00 MiB | 0 | 119.32 MiB | 1290.02 MiB |
| kvarn5 / 1024 | 1337.58 MiB | 880.86 MiB | 20.00 MiB | 0 | 126.06 MiB | 1463.64 MiB |

Gemma has 50 SWA and 10 global attention layers. At tail 1024, telemetry
records 50 native bodyless routes and 10 native mixed routes, with zero device
fallback and zero CPU layers. The q5_0 exact payload replaces 412.50 MiB of
SWA quantized body with 800 MiB of native-exact SWA rows and adds 80 MiB of
global exact overlay plus 0.86 MiB of rollback reserve. The resulting
468.36 MiB persistent delta is therefore fully reconciled. The previous
workspace-dependent persistent padding is zero; physical ubatch contributes
only to graph-local and reusable transient memory.

### Measurement and validation

Compare against the same ordinary cache pair with identical context, batch,
ubatch, prompt, and sampling settings. Record the selected representation,
persistent VRAM, and both prompt and generation speed as functions of tail
length. Treat the uniform capped-1024 `auto` setting as a starting policy and
measure the exact workload before deploying it.

## Upstream DFlash with profit adaptation

### What it is

Bee uses upstream `draft-dflash` for drafting and adds a server-side profit
controller for DFlash1. If the draft maximum is omitted, Bee reads
`dflash.block_size` and uses one less than the block size, normally 15; the
controller remains default-on and can select shallower DFlash1 depths at
runtime. DFlash2 uses its fixed trained block limit and selector confidence.

### When to use it

Use DFlash when you have an upstream-format drafter trained for the exact target
model. For DFlash1, let the metadata-derived maximum and profit controller
establish a baseline before pinning a smaller depth. For DFlash2, tune the
static maximum and selector confidence against the target workload.

### Key arguments

- [`--spec-type draft-dflash`](beellama-args.md#dflash-and-adaptive-draft-depth)
- [`--spec-draft-model`](beellama-args.md#dflash-and-adaptive-draft-depth)
- [`--spec-draft-n-max`](beellama-args.md#dflash-and-adaptive-draft-depth)
- [`--spec-dm-controller`](beellama-args.md#dflash-and-adaptive-draft-depth)
- [`--spec-dm-profit-baseline-interval`](beellama-args.md#dflash-and-adaptive-draft-depth)

### Measurement and validation

Compare adaptive and fixed-depth runs with the same prompt, target and draft
files, cache types, sampling settings, and GPU. Report generated and accepted
draft tokens as well as wall-clock throughput; output bytes are not a stable
cross-build oracle for speculative decoding.

### Known limitations

The drafter must expose upstream `dflash` architecture metadata and tensor
names. Other DFlash GGUF schemas are unsupported. The profit controls apply only
to DFlash1; DFlash2, upstream simple, EAGLE3, MTP, and n-gram modes keep their
own draft-depth behavior.

## Reasoning loop guard and realtime control

### What they are

The loop guard detects periodic repetition in hidden reasoning. Its default
`force-close` mode asks the reasoning sampler to end the hidden section; `stop`
ends generation. Separately, an opted-in streaming chat completion can receive
a `reasoning_end` action through `/v1/chat/completions/control`.

### When to use them

Use the guard for models that can become trapped in long repeated reasoning.
Use realtime control when an external client, rather than a repetition score,
should decide when the model moves to its final answer.

### Key arguments

- [`--reasoning-loop-guard`](beellama-args.md#reasoning-loop-guard)
- [`--reasoning-loop-min-tokens`](beellama-args.md#reasoning-loop-guard)
- [`--reasoning-loop-window`](beellama-args.md#reasoning-loop-guard)
- [`--reasoning-loop-interventions`](beellama-args.md#reasoning-loop-guard)
- [Request control fields](beellama-args.md#realtime-reasoning-control)

### Measurement and validation

Test against both repeated and normal reasoning traces. A useful guard test
records the detected period, repeated coverage, intervention count, final stop
reason, and whether a final answer was produced. For realtime control, keep the
SSE stream open, send its completion id to the control endpoint, and verify the
request transitions out of hidden reasoning.

### Known limitations

Force-close and realtime control require a chat template with a usable reasoning
end sequence. Realtime control is opt-in per request, accepts only
`reasoning_end`, and acts only on a live completion id.

## INI presets

### What they are

Presets store ordinary option names without leading dashes. Router mode can load
multiple named model sections, apply a shared `[*]` section, and optionally
autoload selected models.

### When to use them

Use presets to keep model paths, cache policy, DFlash configuration, and loop
guard settings together, especially when one router serves several models.

### Key arguments

- [`--models-preset`](beellama-args.md#presets)
- [Preset key `load-on-startup`](beellama-args.md#presets)
- [Preset key `stop-timeout`](beellama-args.md#presets)
- Upstream `--models-max`

### Measurement and validation

Start with `--models-preset FILE`, inspect the router's model list, and verify
that CLI overrides win over preset values. Treat remote presets as executable
configuration and use only repositories you trust.

### Known limitations

Preset-only keys are not command-line arguments. Earlier TurboQuant, fork
DFlash, tree, and HALF_QUANTS settings are invalid in v0.4.0 presets. See
[INI presets](preset.md) for syntax and examples.

## KLD tooling

### What it is

`llama-perplexity` can save compressed base-model log probabilities and compare
a second run with them, reporting KL divergence and probability differences.

### When to use it

Use KLD to measure the quality effect of a KV-cache format while holding the
model, corpus, context, and batching constant.

### Key arguments

- [`--save-all-logits`](beellama-args.md#kld-measurement)
- [`--kl-divergence`](beellama-args.md#kld-measurement)
- [`--kl-divergence-base`](beellama-args.md#kld-measurement)
- Upstream `-c`, `-b`, and `-ub`

### Measurement and validation

Generate the base file first, then run the candidate with `--kl-divergence` and
the same evaluation tokens, logical batch, and physical ubatch. Treat a nonzero
process exit as a failed measurement rather than a score.

### Known limitations

The base file is tied to its vocabulary, evaluation tokens, and context size.
It is a measurement artifact, not a portable model format.

## Removed systems

TurboQuant/TCQ, DDTree, CopySpec, the fork DFlash ring/capture/tape and reduced
verifier, the fringe controller, and their private arguments are not maintained
in v0.4.0. See [Migration from earlier versions](beellama-args.md#migration-from-earlier-versions)
for redirects and replacements.
