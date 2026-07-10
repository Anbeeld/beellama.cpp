# BeeLlama v0.4.0 argument reference

This page documents the Bee-specific surface layered on upstream llama.cpp.
Run `llama-server --help` or `llama-cli --help` for the complete upstream option
list and the exact defaults in a release build.

## KVarN target KV cache

Use the standard cache arguments with a KVarN pseudo type:

```text
--cache-type-k kvarn4 --cache-type-v kvarn4
```

Supported KVarN bit levels are `kvarn2`, `kvarn3`, `kvarn4`, `kvarn5`,
`kvarn6`, and `kvarn8`. KVarN is target-context-only. When one side is KVarN
and the other is not, BeeLlama promotes the other side to a matching KVarN bit
level with a warning. Use both `--cache-type-k-swa` and `--cache-type-v-swa` to
override the KVarN type for SWA layers; supplying only one is an error.

KVarN needs a supported target attention shape and a native offloaded backend. It
is not accepted for a draft cache (`--spec-draft-type-k` / `--spec-draft-type-v`).

## Upstream DFlash and adaptive depth

```text
--spec-type draft-dflash
--spec-draft-model DRAFT.gguf
--spec-draft-n-max N
--spec-draft-n-min N
--spec-draft-p-min P
```

`--spec-type dflash` is a compatibility alias that warns and maps to
`draft-dflash`.

The retained Bee controller is profit-only:

```text
--spec-dm-controller profit|off
--spec-dm-profit-min F
--spec-dm-profit-raise-margin F
--spec-dm-profit-lower-margin F
--spec-dm-profit-ewma-alpha F
--spec-dm-profit-min-samples N
--spec-dm-profit-warmup N
--spec-dm-profit-baseline-interval N
```

`profit` changes the upstream per-call draft limit from measured acceptance and
cycle time. `off` leaves the fixed `--spec-draft-n-max` limit in control.
`--spec-draft-p-min` remains a separate per-position confidence stop.

## Reasoning-loop guard

```text
--reasoning-loop-guard off|force-close|stop
--reasoning-loop-min-tokens N
--reasoning-loop-window N
--reasoning-loop-max-period N
--reasoning-loop-min-coverage N
--reasoning-loop-check-interval N
--reasoning-loop-interventions N
```

The guard observes generated hidden-reasoning tokens. `force-close` asks the
reasoning-budget sampler to end the hidden section, up to the intervention limit;
`stop` ends generation immediately. The server accepts the same settings in
request JSON with underscores, for example `"reasoning_loop_guard": "stop"`.

## Realtime reasoning control

Set `"reasoning_control": true` on a chat-completions request to opt that
in-flight completion into external control. While consuming its SSE stream, send
the returned completion id to:

```text
POST /v1/chat/completions/control
{"id":"chatcmpl-...","action":"reasoning_end"}
```

`reasoning_end` asks the same reasoning-budget sampler to emit its configured end
sequence and continue with the final answer. A finished or unknown id is a safe
no-op. Router mode also requires the request's `model` field.

## Removed or redirected values

| Historical surface | v0.4.0 behavior |
|---|---|
| `turbo2`, `turbo3`, `turbo4` and `_tcq` cache spellings | Warn and redirect to matching KVarN target cache types or standard low-bit draft cache types. |
| TurboQuant/TCQ GGUF cache or TQ weight types | Unsupported; re-quantize from source. |
| `--spec-type dflash` | Warned alias for `draft-dflash`. |
| `--spec-dflash-cross-ctx` | Removed; no upstream equivalent. |
| `--spec-dflash-max-slots` | Removed. |
| `--spec-branch-budget`, `--spec-draft-top-k` | Removed with tree verification. |
| `--spec-draft-temp` | Removed; use normal sampling controls. |
| `GGML_DFLASH_*` environment variables | Removed with the old ring/capture implementation. |
| `GGML_CUDA_FA_HALF_QUANTS` | Removed. Use the default matrix or `GGML_CUDA_FA_ALL_QUANTS=ON`. |

## CUDA FlashAttention build policy

No flag builds 103 standard vector K/V pairs. The predicate is
`rank(K) <= rank(V) || K == f16 || V == f16` over the 13 retained cache types.
`-DGGML_CUDA_FA_ALL_QUANTS=ON` builds all 169 standard pairs. KVarN has 15
balanced fast-decode pairs by default and 36 with ALL; other valid KVarN pairs
use descriptor-native MMA fallback.

## Examples

```text
# Upstream DFlash with fixed depth
--spec-type draft-dflash --spec-draft-model draft.gguf --spec-draft-n-max 8 --spec-dm-controller off

# KVarN target cache
--cache-type-k kvarn4 --cache-type-v kvarn4

# Guard a reasoning model
--reasoning-loop-guard force-close --reasoning-loop-interventions 1
```
