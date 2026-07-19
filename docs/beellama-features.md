# BeeLlama v0.4.0 features

BeeLlama v0.4.0 keeps a small fork surface on top of upstream llama.cpp. Use
this page to choose a feature; use the [argument reference](beellama-args.md)
for exact names, environment variables, defaults, and validation ranges.

## KVarN target KV cache

### What it is

KVarN compresses a target model's K and V cache into structured 2-, 3-, 4-,
5-, 6-, or 8-bit records. K and V widths are independent, and supported Qwen
3.6 and Gemma 4 SWA layers can use a separate KVarN pair. Every KVarN group
keeps the paper-defined exact 128-token sink and newest 128-token suffix around
its compressed body. The physical ubatch controls only temporary workspace; it
never enlarges that logical exact suffix.

### When to use it

Use KVarN when CUDA KV-cache memory is the limiting resource and the model has a
supported attention layout. Start with `kvarn4` on both sides, then measure the
quality and speed of the exact model and context you plan to serve.

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
allocating compressed records plus a redundant exact overlay.

### Measurement and validation

Run KLD or perplexity with the same corpus, context, batch size, and cache pair
as the intended workload. Keep both `-b` and `-ub` identical between baseline
and candidate runs. Record the model file, command, prompt or corpus, sampling
settings, GPU, and commit with every result.

The CUDA specialized split and SWA-vector decode routes publish the same
optional FP32 `(maximum, denominator)` metadata as upstream FlashAttention.
An attached precision tail therefore does not force KVarN through the generic MMA
fallback. `llama-bench` reports requested and effective tail sizes plus split,
vector, generic, and prefill route counts so this remains observable.

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
requested 1024-token tail promotes Gemma's fully covered SWA group to native
exact storage, explaining why that row can be faster than its request-zero
mixed compressed/exact route.

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

Native-exact promotion is reported separately from compact exact-overlay bytes.
CUDA allocation remainders are also reported with their sign: positive values
are non-KV scheduler/graph/backend reservations, while negative values denote
allocator reuse/overlap or driver-baseline release rather than negative cache
ownership. Repeated grouped contexts showed only bounded first-use CUDA
reservation and zero cumulative per-context growth.

### Known limitations

KVarN is target-context-only and v0.4.0 enables native placement only on a CUDA
device that passes the kernel capability checks. CPU execution is not native.
Vulkan contains the store operation but deliberately reports native KVarN as
unsupported until Vulkan FlashAttention consumes KVarN views. Unsupported or
partially offloaded placements fail closed; draft and auxiliary contexts must
use standard cache types.

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
a compact exact shadow while retaining the complete selected quantized cache.
A standard quantized tail defaults to BF16, while a KVarN tail defaults to F16;
`--kv-tail-type` can explicitly select either representation for either family.
A request covering a group's full
visibility window may instead promote its owned body to native F16/BF16 when
that representation is supported and its memory increment is no greater than
the overlay. Shared standard bodies are never promoted. Source selection is
per query, so a 512-token prefill does not make the same 512 rows exact for
every query. Body and tail logits share one FP32 softmax; the runtime does not
normalize two attention results independently.

Route selection also records whether each model layer supplies an explicit
self-attention bias. That capability is derived from loaded layer tensors, not
an architecture-name allowlist. A biased layer never selects a native route
that cannot consume its bias, and graph construction rejects any mismatch
between the recorded route and the actual bias tensor.

The overlay shadow pool is owned by the standard cache and identifies rows by
stream, physical cell, and generation. Sequence copies share exact rows within
one stream and copy them into context-local slots across streams. Position
shifts rotate exact K rows together with the quantized body. CUDA can read the
compact pool directly through per-query slot indices and merge against ordinary
FlashAttention normalization metadata; the generic graph gathers only the
configured compact tail width. Neither overlay route materializes the full
cache. Native-exact groups use the ordinary body graph and allocate no shadow.

Exact overlays support the normal `--split-mode layer` ownership model: each
body and its K/V shadows are allocated on that layer's owning device, and graph
writes, reads, and state payload rows retain the same owner. Tensor/meta split
shards an individual body tensor, so a mirrored compact shadow would be
incorrect; partial standard and KVarN overlays therefore fail during context
creation after the ordinary body placement is known and before any tail arena
or shadow allocation. With `--split-mode tensor`, use `--kv-tail-tokens 0` or a
full-window standard native-exact representation. Native-exact has no shadow
and follows the ordinary KV tensor split when that split descriptor is valid.

KVarN differs from standard caches in three intentional ways. Its exact suffix
has a non-disableable 128-token floor, positive requests round upward to 128
tokens, and completed compressed records are written eagerly even while their
tokens coexist in the exact suffix. The KVarN body exports its FP32 row maximum
and denominator to the same tail merge used by ordinary FlashAttention. Sink,
body, and suffix masks therefore contribute each key exactly once. F16/BF16
canonical K/V rows are stored after RoPE for K and in the original V domain;
the compressed body retains KVarN's rotated-domain records.

Standard unified and non-unified prompt caches preserve one continuous suffix
across requests and message boundaries. KVarN trims divergence in its live
exact suffix exactly. Older KVarN divergence retains all complete groups before
the overlapping 128-token group only on a non-unified or otherwise exclusive
unified stream; at most 127 positions before the requested trim are
reevaluated. Unified contention rejects partial rollback, fully reevaluates the
requesting slot, and leaves the other slot unchanged. Every hybrid child must
accept the boundary, so a recurrent component without retained rollback states
can still require a safe full reevaluation. `cache_prompt=false`, slot eviction,
or the absence of one common target/draft plan can also force a miss.

Unified KVarN per-sequence RAM save and restore require an exclusive structured
stream. Contended save creates no cache entry and does not clear the slot;
contended restore is a cache miss. Non-unified KVarN is the configuration where
multi-slot RAM caching and group-aligned historical reuse are unconditional.
For hybrid iSWA with multiple non-unified slots, eligible non-SWA layers keep
KVarN while SWA layers use a warned standard-cache fallback; a fail-closed
preset rejects the context instead. Unified or single-slot layouts can retain
KVarN in both groups when otherwise eligible.

SWA storage remains upstream-aligned at `W + U` physical rows (window plus
ubatch reserve); this feature does not compact the SWA ring. Sparse packing of
generic body rows is enabled only when the complete physical stream fits in the
per-sequence arena, so cached-graph eligibility cannot change with occupancy,
restore, or coverage degradation. Outside that bound the ordinary body route is
used.

Standard exact storage allocates
`round_up(N + n_ubatch, 256) * n_seq_max + sink_slots` rows. The rounded term is
one active-plus-in-flight arena per logical sequence; `sink_slots` is a separate
multi-sequence reserve. Unified ordinary body storage does not merge these
logical exact arenas. Positive K-only MLA and DSA overlays are rejected during
context creation in this release.

### When to use it

Use the precision tail when a quantized cache saves needed context memory but recent-token
quantization changes quality. For standard q2 through q8 caches, start with 64,
128, or 256 tokens. For KVarN, the starting point is its intrinsic 128-token
suffix; try 512 or 1024 only after measuring the exact model and workload.
Larger values read more F16/BF16 data and are not performance-neutral.

`auto` requests 1024 exact tokens for every applicable canonical target-cache
group and caps each request by that group's effective context or attention
window. It is deliberately architecture-agnostic and is not a claim that 1024
is the quality or performance optimum for a particular model.

### Key arguments and APIs

- [`--kv-tail-tokens`](beellama-args.md#kv-cache-precision-tail-for-quantized-caches)
- [`--kv-tail-type`](beellama-args.md#kv-cache-precision-tail-for-quantized-caches)
- `llama_kv_tail_config_*` for model-bound group discovery and overrides
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
Sequence-state framing version 2 writes a validated KV-tail manifest and can
transfer tail tensors through host buffers or the on-device tensor protocol.
Overlay states reject a different structural group, resolved length,
representation, KVarN preset, or F16/BF16 type before mutation. Manifest
version 1 remains readable with conservative degraded provenance; it cannot
upgrade incomplete historical evidence to exact. Native-exact state is already
present in the ordinary body and has no duplicate shadow section. Standard
body-only compatibility state and explicit body-only state begin with
observable degraded coverage and refill from original activations on later
writes. Sequence copies publish body membership and positions immediately;
deferred exact rows materialize in one batch when state data or another direct
consumer needs them. KVarN state version 12 stores logical compressed records
plus exact payloads and remaps physical workspace on restore; version 11 is
rejected because it serialized the old workspace-dependent layout. Dequantized
body rows are never labeled exact. Server metrics report requested and exact
tokens, coverage group states, and degraded sequences.

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

CUDA's native and generic routes and CPU's generic route are hardware verified.
Vulkan, Metal, HIP, SYCL, and generic OpenCL contain the required generic
operator families and are source supported, but are not hardware-verified by
this release. OpenCL's Adreno-transformed weight layouts are rejected by these
row operators; flat standard-cache storage uses the generic kernels. CANN
overlay contexts are rejected at context creation because fused shadow
`SET_ROWS` is not supported; successful source classification is not an Ascend
hardware claim. Startup diagnostics name the selected native or generic route;
generic long-context attention can be substantially slower.

### Measurement and validation

Compare against the same ordinary cache pair with identical context, batch,
ubatch, prompt, and sampling settings. Record the selected representation,
persistent VRAM, and both prompt and generation speed as functions of tail
length. Treat the uniform capped-1024 `auto` setting as a starting policy and
measure the exact workload before deploying it.

## Upstream DFlash with profit adaptation

### What it is

Bee uses upstream `draft-dflash` for drafting and adds a server-side profit
controller. If the draft maximum is omitted, Bee reads `dflash.block_size` and
uses one less than the block size, normally 15; the controller remains
default-on and can select shallower depths at runtime.

### When to use it

Use DFlash when you have an upstream-format drafter trained for the exact target
model. Let the metadata-derived maximum and profit controller establish a
baseline before pinning a smaller depth.

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
to DFlash; upstream simple, EAGLE3, MTP, and n-gram modes keep their own defaults.

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
