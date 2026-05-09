# BeeLlama Features And Public Repo Diff

BeeLlama.cpp is a llama.cpp fork for people who want the current llama.cpp runtime plus aggressive long-context, speculative-decoding, and KV-cache experiments in one tree. Its center of gravity is DFlash: a draft-model architecture that reads recent target hidden states, a server path that can verify flat or tree-shaped drafts, and runtime controls that let the server raise, lower, or disable draft depth while it is running.

The short feature list is below. For full command-line tuning, including upstream llama.cpp args, DFlash args, TurboQuant/TCQ cache choices, context checkpoints, prompt-cache RAM, chat/reasoning controls, and launch examples, read [beellama-args.md](beellama-args.md).

## Public Comparison Baselines

Checked on 2026-05-06 against GitHub `master` refs:

| Repository | Public ref checked | Commit |
| --- | --- | --- |
| `ggml-org/llama.cpp` | `master` | `a0101225bc60908572ed8b4dfd2ee1a61c69b689` |
| `TheTom/llama-cpp-turboquant` | `master` | `57f6b936574f89bd7b46bbd5c317b5bf0b744321` |
| `spiritbuun/buun-llama-cpp` | `master` | `6575873e9c4872709d374d854b583cfaa270caff` |

The BeeLlama tree used for this page is `d8af2fce3`.

## Strong Feature List

BeeLlama's practical advantage is combination. Public llama.cpp has the main runtime, public TheTom has classic TurboQuant work, and public buun has DFlash/TCQ work. BeeLlama brings those lines together and adds server-facing controls that are not present in the checked public refs.

- Modern llama.cpp base: GGUF model loading, `llama-server`, OpenAI-compatible HTTP endpoints, chat templates, reasoning output handling, sampling controls, multimodal projector loading, prompt caching, context checkpoints, unified KV, continuous batching, GPU offload, Flash Attention selection, and the upstream speculative backends that still apply.
- DFlash draft architecture: `dflash-draft` model support, target hidden-state capture, a 4096-token per-layer CPU ring, a configurable DFlash cross-attention window, and a DFlash drafter context shared across server slots.
- DFlash server execution: explicit `--spec-type dflash`, automatic DFlash detection for DFlash draft GGUFs, flat verification with `--spec-branch-budget 0`, tree verification with positive `--spec-branch-budget`, slot capping with `--spec-dflash-max-slots`, and request-time speculative overrides.
- DDTree controls: Bee uses branch-only budgeting through `--spec-branch-budget`. Public buun's `--tree-budget` is still accepted for compatibility, but Bee converts it to branch nodes beyond the main draft path after parsing.
- Adaptive Draft-Max: DFlash draft depth is not just a fixed `--spec-draft-n-max`. Bee adds server-side `profit` and `fringe` controllers with `--spec-dm-*` args, probe cycles, off dwell, EWMA stats, and continuation-state preservation.
- Sampled DFlash verification: `--spec-draft-temp` supports greedy drafting, explicit positive drafter temperature, and `auto` mirroring of target sampling temperature. The rejection-sampling path is conditional on draft temperature, target temperature, and draft log-probability availability.
- Model-free speculation: CopySpec, suffix-tree speculation, recycle speculation, and ngram variants remain available as separate `--spec-type` modes for repeated-context and no-draft-model workloads.
- Multimodal guardrails: with `--mmproj`, Bee keeps flat DFlash available, forces tree DFlash off, disables context shift/cache reuse where unsupported, and blocks non-DFlash speculative modes.
- Reasoning loop guard: Bee adds server controls that detect repeated reasoning loops and either close the reasoning span or stop generation, with CLI defaults and request JSON overrides.
- TurboQuant and TCQ KV cache: Bee exposes `turbo2`, `turbo3`, `turbo4`, `turbo2_tcq`, and `turbo3_tcq` through normal `--cache-type-k` and `--cache-type-v` args. K and V can use different cache types.
- Combined cache lineage: Bee uses the buun enum layout for Turbo/TCQ types, keeps TCQ types, and uses 128-value `turbo2` and `turbo3` block sizes like TheTom's public TurboQuant fork.
- Long-context serving controls: the fork keeps the important upstream server knobs people actually tune, including context size, batch and microbatch size, context checkpoints, checkpoint cadence, unified KV, idle-slot caching, prompt-cache RAM, Flash Attention, cache precision, and mmap/mlock/direct I/O behavior.

## Feature Matrix

| Area | llama.cpp public master | TheTom public TurboQuant fork | buun public master | BeeLlama current tree |
| --- | --- | --- | --- | --- |
| Base runtime | Current llama.cpp runtime, tools, server, backends | llama.cpp-derived runtime | llama.cpp-derived runtime | llama.cpp-derived runtime with Bee server/cache/spec changes |
| OpenAI-style server | Yes | Inherited | Inherited | Yes, plus Bee request-level speculative and loop-guard controls |
| Standard sampling | Yes | Inherited | Inherited | Yes |
| Chat templates and reasoning output | Yes | Inherited | Inherited | Yes, plus reasoning loop guard |
| Multimodal projector path | Yes | Inherited | Inherited | Yes, with flat-DFlash compatibility rule |
| Upstream speculative modes | Draft model, EAGLE3, ngram variants | Inherited where present | Inherited plus fork modes | Inherited plus DFlash, CopySpec, suffix, recycle |
| DFlash draft architecture | No checked match | No checked match | Yes | Yes |
| DFlash server path | No | No | Yes | Yes |
| DFlash hidden-state CPU ring | No | No | Yes, fixed DFlash window in state | Yes, 4096-token per-layer ring |
| DFlash cross window | No | No | Fixed `LLAMA_DFLASH_PER_SLOT_CTX = 512` at checked ref | `--spec-dflash-cross-ctx`, default `512` |
| DFlash slot cap | No | No | `--dflash-max-slots` | `--spec-dflash-max-slots` |
| Flat DFlash | No | No | Yes | Yes |
| Tree DFlash/DDTree | No | No | Yes, total-node `--tree-budget` | Yes, branch-only `--spec-branch-budget` plus legacy conversion |
| Draft top-k for DFlash tree | No | No | `--draft-topk` | `--spec-draft-top-k`, legacy `--draft-topk` accepted |
| Adaptive DFlash depth | No | No | Speculative-code adaptive tracking | Server-side `profit` and `fringe` controllers with `--spec-dm-*` CLI |
| Sampled DFlash CLI | No | No | No checked CLI surface for draft temperature | `--spec-draft-temp 0`, positive value, or `auto` |
| Multi-slot DFlash | No | No | Shared DFlash slot machinery | Shared drafter context, slot cap, and flat batched drafting |
| Multimodal with speculation | Upstream rules | Upstream rules | Speculative decoding disabled under multimodal | Flat DFlash allowed; tree and non-DFlash spec disabled |
| CopySpec | No checked match | No checked match | Yes | Yes as explicit `--spec-type copyspec` |
| Suffix/recycle speculation | No checked match | No checked match | Yes | Yes |
| Reasoning loop guard | No checked match | No checked match | No checked match | Yes |
| Classic TurboQuant cache | No checked match | `turbo2`, `turbo3`, `turbo4` | `turbo2`, `turbo3`, `turbo4` | `turbo2`, `turbo3`, `turbo4` |
| TCQ cache | No checked match | No checked match | `turbo2_tcq`, `turbo3_tcq` | `turbo2_tcq`, `turbo3_tcq` |
| TurboQuant block size | No TurboQuant type | `QK_TURBO2=128`, `QK_TURBO3=128` | `QK_TURBO2=32`, `QK_TURBO3=32` | `QK_TURBO2=128`, `QK_TURBO3=128` |
| Turbo/TCQ enum layout | No TurboQuant type | `TURBO2=42`, `TURBO3=43`, `TURBO4=44` | `TURBO3=42`, `TURBO4=43`, `TURBO2=44`, TCQ at `45/46` | Same enum order as buun, with TCQ at `45/46` |

## DFlash

DFlash is the main fork feature. A DFlash draft model is loaded as the draft side of speculative decoding, but unlike a plain draft model it receives recent hidden states from the target model. Bee wires this through the server eval callback, target-side capture layers, DFlash cross-data APIs, and the `LLM_ARCH_DFLASH_DRAFT` architecture path.

The target-side hidden-state store is a CPU ring of 4096 tokens per captured layer. The drafter does not necessarily see all 4096 tokens on every draft. `--spec-dflash-cross-ctx` controls the recent hidden-state window exposed to DFlash; the code default is `512`, and the GPU cross ring is allocated for that requested window.

Minimal flat DFlash:

```sh
llama-server -m target.gguf \
  --spec-type dflash \
  --spec-draft-model draft-dflash.gguf \
  --spec-draft-n-max 16 \
  --spec-branch-budget 0
```

DFlash supports both full GPU offload (`-ngl all`) and partial offload (`-ngl < total layers`). The tape replay path detects when recurrent state buffers are in host memory and falls back to CPU replay instead of crashing on a broken DeviceToDevice copy.

The accept path (rollback, state restore, and DeltaNet replay after token acceptance or rejection) uses batched asynchronous GPU-to-GPU copies and a direct CUDA GDN state-replay kernel. This avoids per-layer synchronization and per-cycle ggml graph construction for the recurrent state update, keeping accept-side overhead small relative to target-model verification time.

Public llama.cpp and public TheTom do not have the DFlash draft architecture or DFlash server path at the checked refs. Public buun does have DFlash. Bee's DFlash surface is more complete for server tuning because it adds canonical `--spec-*` names, configurable cross context, branch-only DDTree budgeting, sampled-drafter CLI control, adaptive draft controllers, request overrides, and multimodal flat-DFlash guardrails.

### DFlash Drafting Performance

The drafting pipeline has been progressively optimized to keep overhead small relative to target-model verification time.

**Verifier graph reuse.** The target model's speculative verification graph — including attention, recurrent state, and logit projection — is built once and reused across verify passes. Before this change, each speculative verify cycle constructed a fresh ggml graph, which added measurable overhead at every accepted token. The graph now persists as long as its topology holds, with cache invalidation only on structural changes (context resize, batch size shift, or model-layer reconfiguration).

**Reduced-logits consumer decoupling.** The verifier's logit output path is split into a main verify stream and a separate reduced-verify consumer. The reduced consumer gates independently: it only activates when the speculative path requests a reduced logit set (e.g., after token rejection or draft truncation). This avoids copying the full vocabulary logit tensor on every verify pass, cutting per-cycle logit transfer volume.

**Batched async recurrent state rollback.** After token acceptance or rejection, the server must restore recurrent state buffers (DeltaNet hidden state, SSM conv state) for each draft layer. Previously each layer's copy triggered a device synchronization. The rollback path now issues all GPU-to-GPU copies as a single batched asynchronous block using a no-sync copy primitive (`seq_cp_recurrent_no_sync`), then synchronizes once at the end. For a 5-layer drafter this eliminates four synchronization points per accept cycle.

**Direct GPU GDN state replay.** The DeltaNet recurrent state update after acceptance runs as a dedicated CUDA kernel instead of going through ggml graph construction and backend dispatch. The kernel loads the stored accept-tape recurrent state, replays the GDN gating computation directly, and writes the updated hidden state to the target model's context buffers. This avoids per-cycle graph building, tensor allocation, and backend op-scheduling for the accept path.

**Drafter KV projection caching.** The drafter model's per-layer key/value projections into the KV cache are computed once and cached at the current cross-attention window position. On subsequent draft calls within the same context span, the cached projections are reused instead of recomputed, avoiding redundant matmul and rotary embedding work.

**No-sync K/V cache append.** The target model's K/V cache append in the speculative verify path skips per-layer CUDA synchronization. Append correctness is guaranteed by the surrounding speculative flow (which synchronizes at token boundaries), so the individual per-layer syncs were redundant.

**Grammar-aware reduced verify.** When a JSON schema or GBNF grammar is active, DFlash can skip verification of draft tokens that the grammar would already reject. The grammar engine evaluates draft proposals against the active rule set and flags non-conformant tokens before the verifier runs. Those tokens are excluded from the verify batch, shrinking the effective batch size and reducing target model work per draft cycle. A companion edge case handles lazy-allocated grammars that are declared but not yet active, avoiding false blocking of reduced verify.

**Verifier padding.** The reduced-verify path can produce non-consecutive position spans in the recurrent state cells (e.g., when grammars or branch rejection create gaps in the accepted token sequence). The verifier pads those gaps so that the recurrent state computation sees valid cell positions for every token in the batch. Post-verification rollback corrects `cell.pos` to match the actual token sequence, so the padding is transparent to subsequent operations.

**Smooth cross-window bucket growth.** The DFlash cross-attention context window allocates GPU ring buffer buckets in gradual steps rather than doubling them. When the target context length passes a bucket threshold, the ring grows to the next bucket size incrementally, avoiding large transient allocation spikes during prefill or rapid context expansion.

**Accept-path sub-phase profiling.** Individual sub-phases of the speculative accept path (state copy, replay, KV update, position fixup) are timed separately and surfaced through the DFlash profile log. This lets benchmark runs isolate which sub-phase dominates accept-side latency without manual instrumentation.

## DDTree

Bee supports flat DFlash and tree-shaped DFlash verification:

```sh
--spec-branch-budget 0   # flat DFlash
--spec-branch-budget 4   # up to four branch nodes beyond the main draft path
--spec-draft-top-k 4     # candidates per draft position for tree mode
--spec-draft-p-split 0.1 # probability threshold for branch creation
```

The important distinction is that `--spec-draft-n-max` controls main-path draft length, while `--spec-branch-budget` controls extra branch nodes. In public buun, `--tree-budget` is a total DDTree node budget. Bee still accepts `--tree-budget TOTAL`, then converts it to `max(0, TOTAL - draft_max)` unless `--spec-branch-budget` was also supplied. If both are supplied, Bee uses `--spec-branch-budget` and warns that the legacy value was ignored.

Flat mode forces `--spec-draft-top-k` back to `1`, because there are no branch alternatives to verify when the branch budget is zero.

## Adaptive Draft-Max

Bee's DFlash depth can adapt per server slot. `--spec-draft-n-max` is the base ceiling. The active depth can be lower, can probe after being turned off, and can return upward when telemetry supports it.

The default controller is `profit`, with a second `fringe` controller available through `--spec-dm-controller fringe`. The profit controller starts at full base depth with a configurable warmup burst, uses cross-depth timing estimation to score all depths from one measured depth, tracks acceptance and timing via EWMA, and preserves adaptive state for continuation-like prompts when the prompt similarity and kept-context fraction are high enough.

This is not the same as public buun's checked DFlash adaptive tracking. Bee adds the server controller layer and the `--spec-dm-*` command-line surface:

```sh
--spec-dm-controller profit
--spec-dm-profit-min 0.05
--spec-dm-profit-raise-margin 0.05
--spec-dm-profit-lower-margin 0.08
--spec-dm-profit-ewma-alpha 0.15
--spec-dm-profit-min-samples 3
--spec-dm-profit-warmup 0
```

Use `--no-spec-dm-adaptive` when you need a fixed-depth benchmark. Otherwise, adaptive mode is the safer default for live serving because it can back away from weak drafts without changing the process command line.

## Sampled DFlash

DFlash defaults to greedy drafting:

```sh
--spec-draft-temp 0
```

Positive values enable sampled DFlash drafting, and `auto` copies the target sampling temperature into the drafter after request/server sampling settings are known:

```sh
--spec-draft-temp 0.6
--spec-draft-temp auto
```

The sampled-verification path is deliberately conditional. It runs only when drafter temperature is above zero, target temperature is above zero, and DFlash returned draft log-probabilities. Without those conditions, Bee stays on the normal greedy speculative path.

## Model-Free Speculation

Bee also has no-draft-model speculative modes:

| Mode | What it uses | Best fit |
| --- | --- | --- |
| `ngram-cache` | Static/dynamic lookup cache files | Workloads with reusable lookup cache data |
| `ngram-simple` | In-context ngram matching | Simple repeated-context baseline |
| `ngram-map-k` | Map-backed ngram to mgram proposals | Repeated text where longer continuations recur |
| `ngram-map-k4v` | Alternative map-backed ngram variant | Same tuning family as `ngram-map-k` |
| `ngram-mod` | Shared ngram-mod container | Long repeated context with `n_min/n_max/match` controls |
| `copyspec` | Rolling-hash suffix matches | Prompt-local repetition without a draft model |
| `recycle` | Token successor adjacency | Repeated successor patterns |
| `suffix` | Suffix-tree proposals | Repeated suffix continuations |

CopySpec is explicit in Bee. Public buun's DFlash composition inserts CopySpec before DFlash; Bee's current DFlash config adds DFlash alone, so use `--spec-type copyspec` when you want CopySpec as the selected backend.

## Multimodal

Bee keeps DFlash usable with `--mmproj` when it can do so without pretending unsupported state operations are safe. When a multimodal projector is loaded, the server:

- disables context shift
- disables cache reuse
- allows DFlash only when the speculative type is DFlash
- forces `--spec-branch-budget 0` because tree-based DFlash is not supported with multimodal
- disables non-DFlash speculative decoding

Public buun disables speculative decoding entirely under multimodal at the checked ref. Bee narrows that rule to keep flat DFlash available for text generation.

## Reasoning Loop Guard

Bee adds a server guard for repeated reasoning loops. It watches generated reasoning text after a minimum token count, checks a sliding window at a configured interval, and can either force-close the reasoning section or stop generation. The default mode is `force-close`; the server falls back to stop behavior when force-close is not available for a task.

The public llama.cpp, TheTom, and buun refs checked for this document do not contain this Bee reasoning-loop guard path.

Core controls:

```sh
--reasoning-loop-guard force-close
--reasoning-loop-min-tokens 1024
--reasoning-loop-window 2048
--reasoning-loop-max-period 512
--reasoning-loop-min-coverage 768
--reasoning-loop-check-interval 32
--reasoning-loop-interventions 1
```

The same behavior can be overridden per request with JSON fields such as `reasoning_loop_guard`, `reasoning_loop_window`, and `reasoning_loop_interventions`.

## TurboQuant And TCQ KV Cache

Bee exposes fork cache types through the normal KV cache args:

```sh
--cache-type-k turbo4 --cache-type-v turbo3_tcq
--cache-type-k q8_0   --cache-type-v turbo3_tcq
--cache-type-k turbo3 --cache-type-v turbo3
```

Current storage math from `ggml/src/ggml-common.h` and `ggml/src/ggml.c`:

| Cache type | Block layout | Storage bpv | FP16 storage compression |
| --- | --- | ---: | ---: |
| `turbo2` | 34 bytes / 128 values | `2.125` | `7.53x` |
| `turbo3` | 50 bytes / 128 values | `3.125` | `5.12x` |
| `turbo4` | 66 bytes / 128 values | `4.125` | `3.88x` |
| `turbo2_tcq` | 36 bytes / 128 values | `2.25` | `7.11x` |
| `turbo3_tcq` | 52 bytes / 128 values | `3.25` | `4.92x` |

There is no separate `turbo3_128` command-line type in Bee. `turbo3` itself uses `QK_TURBO3 = 128`.

Do not move GGUF cache metadata or serialized tensors between TurboQuant forks by enum number alone. The public enum assignments differ between TheTom and Bee/buun.

## Long-Context Serving Surface

Bee does not hide the important llama.cpp server knobs. The fork keeps the controls people tune when running long contexts:

- `--ctx-size` for model context allocation
- `-b` and `-ub` for logical and physical prompt batch size
- `--kv-unified` for a unified KV buffer across slots
- `--ctx-checkpoints` and `--checkpoint-every-n-tokens` for prompt context checkpoints
- `--cache-ram` and `--cache-idle-slots` for server prompt-cache RAM behavior
- `--cache-type-k` and `--cache-type-v` for independent K/V cache precision
- `--flash-attn on|off|auto` for Flash Attention selection
- `--mmap`, `--no-mmap`, `--mlock`, `--direct-io`, and `--no-host` for model and buffer placement behavior
- `-ngl all`, `--device`, `--split-mode`, `--main-gpu`, and `--tensor-split` for GPU placement

These are inherited or llama.cpp-derived server controls, not all Bee inventions. They matter because Bee's DFlash and Turbo/TCQ features are most useful when the base server is configured correctly.

## Where Bee Extends Each Public Repo

Compared with public llama.cpp, Bee adds DFlash, DDTree, CopySpec/recycle/suffix fork modes, TurboQuant/TCQ cache types, sampled DFlash controls, DFlash/mmproj compatibility handling, and the reasoning loop guard.

Compared with TheTom's public TurboQuant fork, Bee adds DFlash and DFlash server integration, TCQ cache types, buun-compatible Turbo enum order, branch-aware DDTree, adaptive DFlash controls, sampled DFlash CLI, multimodal DFlash guardrails, and the reasoning loop guard while keeping 128-value `turbo2` and `turbo3` block sizes.

Compared with public buun, Bee adds canonical `--spec-*` DFlash names, `--spec-dflash-cross-ctx`, branch-only `--spec-branch-budget` semantics, `--spec-dm-*` server adaptive controllers, `--spec-draft-temp`, reasoning loop guard controls, DFlash request overrides, and the narrower multimodal rule that keeps flat DFlash available.

## Limits And Non-Claims

This page claims feature presence, defaults, CLI names, enum values, and storage math when those are visible in the current code. It does not claim universal speedups, universal acceptance rates, quality improvements, perplexity changes, or hardware-independent throughput. Those belong in benchmark artifacts with exact models, prompts, commands, hardware, commit IDs, and dirty-worktree status.

DFlash requires a compatible DFlash draft GGUF. TurboQuant and TCQ cache types require runtime/backend support for the operation you are asking for. Multimodal DFlash is flat-only in the current server path. Branch/tree settings, cache compression, batch size, and context length all interact with hardware memory pressure, so treat the command-line reference as a tuning map, not a benchmark substitute.
