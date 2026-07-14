# BeeLlama v0.4.0 argument reference

This page covers Bee-owned arguments and the upstream arguments whose behavior
BeeLlama extends. Run `llama-server --help` or `llama-cli --help` for the full
upstream surface. See [BeeLlama features](beellama-features.md) for use cases,
limits, and measurement guidance.

## KVarN cache types and SWA overrides

KVarN values are `kvarn2`, `kvarn3`, `kvarn4`, `kvarn5`, `kvarn6`, and
`kvarn8`. K and V may use different bit widths.

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `-ctk TYPE`, `--cache-type-k TYPE` | `LLAMA_ARG_CACHE_TYPE_K` | `f16` | Selects the target K cache. Bee adds the six KVarN values and standard `q6_0`, `q6_1`, `q3_0`, `q3_1`, `q2_0`, and `q2_1`. If only K or V is KVarN, the other side is promoted to the same KVarN width with a warning. |
| `-ctv TYPE`, `--cache-type-v TYPE` | `LLAMA_ARG_CACHE_TYPE_V` | `f16` | Selects the target V cache with the same values and one-sided promotion rule as `--cache-type-k`. |
| `--cache-type-k-swa TYPE` | `LLAMA_ARG_CACHE_TYPE_K_SWA` | Same as `--cache-type-k` | Overrides KVarN K precision for SWA layers. Accepts only the six `kvarnN` values, requires target KVarN, and must be paired with the V override. |
| `--cache-type-v-swa TYPE` | `LLAMA_ARG_CACHE_TYPE_V_SWA` | Same as `--cache-type-v` | Overrides KVarN V precision for SWA layers. Accepts only the six `kvarnN` values, requires target KVarN, and must be paired with the K override. |

## High-precision tail for standard caches

The standard-cache tail keeps the complete ordinary quantized cache and a compact
F16 or BF16 shadow for each query's newest visible entries. It applies only to
quantized sides of target-model standard attention caches. Draft and auxiliary
contexts remain at the disabled default, and KVarN components warn and ignore it.

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `--kv-tail-tokens SPEC` | `LLAMA_ARG_KV_TAIL_TOKENS` | `0` | `0` keeps the ordinary cache path. A number applies to every cache group. `N0,N1` follows canonical group order, while `full=N,swa=N` accepts unique role aliases or structural IDs such as `full@l0`. Invalid, duplicate, incomplete, or wrong-length specifications disable every group. `auto` resolves to zero unless the exact model structure, ordered body pair, tail type, and backend have a published recommendation. |
| `--kv-tail-type TYPE` | `LLAMA_ARG_KV_TAIL_TYPE` | `f16` | Selects `f16` or `bf16` exact-shadow storage for the context. Other types are rejected. |

Explicit values are capped by the group's effective attention window and context
capacity. Startup logs show the structural group ID, participating layer count,
requested length, resolved length, type, physical slots, payload bytes, and
in-flight reserve. Only a quantized K side receives a K shadow, and only a
quantized V side receives a V shadow.

Tail-enabled state uses a framed standard-memory section. Exact restore requires
the same structural group, resolved length, and tail type. The extended full and
sequence state APIs accept `LLAMA_STATE_SEQ_FLAGS_BODY_ONLY` to deliberately omit
exact shadows; loading that state into a tail-enabled context is valid, but the
coverage API reports `LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE` until new writes
refill the recent window.

## DFlash and adaptive draft depth

The first five rows are upstream speculative controls with Bee-specific DFlash
behavior. The `--spec-dm-*` rows are Bee server additions.

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `--spec-type draft-dflash` | `LLAMA_ARG_SPEC_TYPE` | `none` | Enables upstream DFlash. The earlier `dflash` value warns and redirects to `draft-dflash`. |
| `--spec-draft-model FNAME`, `-md FNAME` | `LLAMA_ARG_SPEC_DRAFT_MODEL` | Unused | Loads the draft GGUF. Historical Bee `dflash-draft` files must first be rewritten with `scripts/convert-dflash-draft-to-upstream.py`. |
| `--spec-draft-n-max N` | `LLAMA_ARG_SPEC_DRAFT_N_MAX` | Upstream: `3`; omitted DFlash: `dflash.block_size - 1` | Sets the maximum draft depth. An explicit CLI or env value always wins; upstream clamps values above the drafter's trained limit. A block-16 drafter therefore defaults to 15 only when this setting is omitted. |
| `--spec-draft-n-min N` | `LLAMA_ARG_SPEC_DRAFT_N_MIN` | `0` | Sets the minimum number of draft tokens used by upstream speculation. |
| `--spec-draft-p-min P`, `--draft-p-min P` | `LLAMA_ARG_SPEC_DRAFT_P_MIN` | `0.0` | Stops an individual greedy draft when its probability falls below `P`; this is independent of the profit controller. |
| `--spec-dm-controller MODE` | `LLAMA_ARG_SPEC_DM_CONTROLLER` | `profit` | `profit` adapts DFlash depth from measured cycle profit; `off` keeps the resolved or explicit maximum static. Other speculative modes are unchanged. |
| `--spec-dm-profit-min F` | `LLAMA_ARG_SPEC_DM_PROFIT_MIN` | `0.05` | Sets the minimum margin over the no-spec baseline before clearing disable dwell. Range: `0.0` to `0.50`. |
| `--spec-dm-profit-raise-margin F` | `LLAMA_ARG_SPEC_DM_PROFIT_RAISE_MARGIN` | `0.05` | Sets the relative profit margin required to raise draft depth. Range: `0.0` to `1.0`. |
| `--spec-dm-profit-lower-margin F` | `LLAMA_ARG_SPEC_DM_PROFIT_LOWER_MARGIN` | `0.05` | Sets the relative profit margin required to lower draft depth. Range: `0.0` to `1.0`. |
| `--spec-dm-profit-ewma-alpha F` | `LLAMA_ARG_SPEC_DM_PROFIT_EWMA_ALPHA` | `0.15` | Sets the EWMA weight for profit statistics. Range: `0.01` to `1.0`. |
| `--spec-dm-profit-min-samples N` | `LLAMA_ARG_SPEC_DM_PROFIT_MIN_SAMPLES` | `3` | Sets the samples required before a depth's profit statistics are ready. Range: `1` to `64`. |
| `--spec-dm-profit-warmup N` | `LLAMA_ARG_SPEC_DM_PROFIT_WARMUP` | `0` | Sets measured samples for each initial positive-depth probe. `0` uses `--spec-dm-profit-min-samples`; range: `0` to `64`. |
| `--spec-dm-profit-baseline-interval N` | `LLAMA_ARG_SPEC_DM_PROFIT_BASELINE_INTERVAL` | `1024` | Sets active controller cycles between no-spec baseline probes. `0` disables periodic probes; range: `0` to `4096`. |

## Reasoning loop guard

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `--reasoning-loop-guard MODE` | `LLAMA_ARG_REASONING_LOOP_GUARD` | `force-close` | `off` disables checks, `force-close` asks the reasoning sampler to end hidden reasoning, and `stop` ends generation when a loop triggers. |
| `--reasoning-loop-min-tokens N` | `LLAMA_ARG_REASONING_LOOP_MIN_TOKENS` | `512` | Delays hidden-reasoning checks until `N` reasoning tokens have been seen. Must be non-negative and at least the minimum coverage. |
| `--reasoning-loop-window N` | `LLAMA_ARG_REASONING_LOOP_WINDOW` | `1024` | Sets the token-tail window inspected for repetition. Must be positive and at least the minimum coverage. |
| `--reasoning-loop-max-period N` | `LLAMA_ARG_REASONING_LOOP_MAX_PERIOD` | `128` | Sets the longest periodic loop checked. Must be positive and no more than one third of the window. |
| `--reasoning-loop-min-coverage N` | `LLAMA_ARG_REASONING_LOOP_MIN_COVERAGE` | `256` | Sets the repeated-token coverage required to trigger. Must be positive. |
| `--reasoning-loop-check-interval N` | `LLAMA_ARG_REASONING_LOOP_CHECK_INTERVAL` | `64` | Runs a check after each `N` accepted reasoning tokens. Must be positive. |
| `--reasoning-loop-interventions N` | `LLAMA_ARG_REASONING_LOOP_INTERVENTIONS` | `2` | Sets the maximum successful force-close interventions before a later trigger stops generation. Must be non-negative. |

## Realtime reasoning control

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| Chat request JSON `"reasoning_control": true` | — | `false` | Arms a live `/v1/chat/completions` request for external reasoning control. The chat template must expose a reasoning end sequence. |
| `POST /v1/chat/completions/control` with `{"id":"chatcmpl-...","action":"reasoning_end"}` | — | Disabled per request | Forces the armed completion's reasoning sampler toward its final-answer phase. Unknown or completed ids return a non-success result; `reasoning_end` is the only accepted action. |

## Presets

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `--models-preset PATH` | `LLAMA_ARG_MODELS_PRESET` | Disabled | Loads an INI file containing model presets for router-server mode. Command-line values override values loaded from a preset. |
| Preset key `load-on-startup` | Preset-only | False when absent | A truthy value autoloads that model when router mode starts; the number of startup models may not exceed `--models-max`. |
| Preset key `stop-timeout` | Preset-only | `10` seconds | Force-kills a child model process after this many seconds of graceful shutdown. Invalid values fall back to 10. |

See [INI presets](preset.md) for syntax, inheritance, remote presets, and a Bee
configuration example.

## KLD measurement

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `--save-all-logits FNAME`, `--kl-divergence-base FNAME` | — | Unused | Without `--kl-divergence`, writes the base run's compressed log probabilities to `FNAME`. |
| `--kl-divergence` | — | Off | Compares the current run with the file supplied by `--kl-divergence-base` and returns a nonzero exit code on read or evaluation failure. |

Use the same corpus, context, logical batch, and physical ubatch for both KLD legs.

## CUDA FlashAttention build policy

| Argument | Env var | Default | Behavior |
|---|---|---|---|
| `-DGGML_CUDA_FA_ALL_QUANTS=ON` | — | Off | Expands the CUDA vector matrix from 103 to all 169 standard cache pairs and KVarN fast-decode instances from 15 balanced pairs to all 36 ordered bit pairs. Valid KVarN pairs outside the fast matrix use descriptor-native MMA. |

## Migration from earlier versions

| Earlier spelling or surface | v0.4.0 behavior | Replacement |
|---|---|---|
| Target cache `turbo2`, `turbo3`, `turbo4`, or `_tcq` variants | Warns and redirects by width to `kvarn2`, `kvarn3`, or `kvarn4`. | Use the `kvarnN` name directly. |
| Draft cache `turbo2`, `turbo3`, `turbo4`, or `_tcq` variants | Warns and redirects by width to `q2_0`, `q3_0`, or `q4_0`. | Use the standard q-cache name directly. |
| TurboQuant/TCQ GGUF cache formats and TQ3/TQ4 weight formats | Unsupported; legacy TQ file-type ids fail with a re-quantization error. | Re-quantize from source into a retained format. |
| `--spec-type dflash` | Warns and redirects. | `--spec-type draft-dflash` |
| `copyspec`, `suffix`, or `recycle` speculative types | Rejected with a migration error. | Use `draft-dflash` or an upstream n-gram mode. |
| `--draft`, `--draft-n`, `--draft-max` | Rejected as removed. | `--spec-draft-n-max` or `--spec-ngram-mod-n-max` |
| `--draft-min`, `--draft-n-min` | Rejected as removed. | `--spec-draft-n-min` or `--spec-ngram-mod-n-min` |
| `--spec-dflash-default`, `--dflash-max-slots`, `--tree-budget`, `--draft-topk`, `--draft-model`, `--spec-replace`, `--spec-draft-replace` | Removed with the fork DFlash verifier and tree paths. | Use upstream `--spec-*` controls where an equivalent exists. |
| `--spec-dflash-cross-ctx`, `--spec-branch-budget`, `--spec-draft-temp`, `GGML_DFLASH_*` | Removed with the fork ring, capture, and verifier implementation. | No direct replacement. |
| `GGML_CUDA_FA_HALF_QUANTS` | Removed. | Use the default matrix or `GGML_CUDA_FA_ALL_QUANTS=ON`. |
