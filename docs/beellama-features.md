# BeeLlama v0.4.0 features

BeeLlama v0.4.0 keeps a small fork surface on top of upstream llama.cpp. Use
this page to choose a feature; use the [argument reference](beellama-args.md)
for exact names, environment variables, defaults, and validation ranges.

## KVarN target KV cache

### What it is

KVarN compresses a target model's K and V cache into structured 2-, 3-, 4-,
5-, 6-, or 8-bit records. K and V widths are independent, and supported Qwen
3.6 and Gemma 4 SWA layers can use a separate KVarN pair.

### When to use it

Use KVarN when CUDA KV-cache memory is the limiting resource and the model has a
supported attention layout. Start with `kvarn4` on both sides, then measure the
quality and speed of the exact model and context you plan to serve.

### Key arguments

- [`--cache-type-k`](beellama-args.md#kvarn-cache-types-and-swa-overrides)
- [`--cache-type-v`](beellama-args.md#kvarn-cache-types-and-swa-overrides)
- [`--cache-type-k-swa`](beellama-args.md#kvarn-cache-types-and-swa-overrides)
- [`--cache-type-v-swa`](beellama-args.md#kvarn-cache-types-and-swa-overrides)

### Measurement and validation

Run KLD or perplexity with the same corpus, context, batch size, and cache pair
as the intended workload. Keep both `-b` and `-ub` identical between baseline
and candidate runs. Record the model file, command, prompt or corpus, sampling
settings, GPU, and commit with every result.

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
values. A build with the default CUDA policy contains 103 standard vector pairs;
the ALL option contains 169.

### Known limitations

The user-facing `q2_0` cache name must not be treated as upstream's Q2_0 weight
format. A requested CUDA FlashAttention pair must be compiled by the selected
build tier.

## Optional exact tail for standard quantized caches

### What it is

`--kv-tail-tokens` overlays the newest attention-visible entries with F16 or
BF16 K/V while retaining the complete selected standard quantized cache. Source
selection is per query, so a 512-token prefill does not make the same 512 rows
exact for every query. Body and tail logits share one FP32 softmax; the runtime
does not normalize two attention results independently.

The shadow pool is owned by the standard cache and identifies rows by stream,
physical cell, and generation. Sequence copies share exact rows within one
stream and copy them into context-local slots across streams. Position shifts
rotate exact K rows together with the quantized body. CUDA reads the compact
pool directly through per-query slot indices and merges against ordinary
FlashAttention normalization metadata; generic backends gather only the
configured compact tail width. Neither route materializes the full cache.

### When to use it

Use the tail when q2 through q8 standard caches save needed context memory but
recent-token quantization changes quality. Start with 64, 128, or 256 tokens and
measure the exact model and workload. Larger values read more F16/BF16 data and
are not performance-neutral.

For the documented RTX 3090 Qwen3.6-27B Q5_K_S, symmetric q4_0, BF16-reference
workload, tail 64 is the measured knee: mean KLD improved 22.8%, while paired
prompt and generation benchmarks regressed 3.63% and 2.64%. This is an explicit
workload recommendation, not a broad `auto` match.

### Key arguments and APIs

- [`--kv-tail-tokens`](beellama-args.md#high-precision-tail-for-standard-caches)
- [`--kv-tail-type`](beellama-args.md#high-precision-tail-for-standard-caches)
- `llama_kv_tail_config_*` for model-bound group discovery and overrides
- `llama_kv_tail_get_coverage` for per-sequence, per-group coverage
- `llama_kv_tail_get_coverage_aggregate` for context/server aggregation
- `LLAMA_STATE_SEQ_FLAGS_BODY_ONLY` for an intentional lower-precision state export

Implementation decisions and non-local hardware verification packages are in
[`development/std-quant-kv-tail.md`](development/std-quant-kv-tail.md) and
[`development/std-quant-kv-tail-backend-verification.md`](development/std-quant-kv-tail-backend-verification.md).

### State and compatibility

The default length is zero. That path allocates no shadows, builds no tail
inputs, and retains ordinary FlashAttention dispatch. Exact tail states reject
a different resolved length or F16/BF16 type. A preceding unframed body-only
state and an explicit framed body-only state remain loadable; both begin with
observable degraded coverage and refill from original activations on later
writes. Dequantized body rows are never labeled exact.

### Measurement and validation

Compare against the same ordinary cache pair with identical context, batch,
ubatch, prompt, and sampling settings. Record persistent VRAM and both prompt
and generation speed as functions of tail length. `auto` is conservative: no
unknown model/body/backend combination is enabled, and this release contains no
recommendation without a recorded quality and performance curve.

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

The drafter must expose upstream `dflash` metadata and tensor names. Convert a
historical Bee/buun `dflash-draft` file with:

```text
python scripts/convert-dflash-draft-to-upstream.py legacy.gguf upstream.gguf --verify
```

The conversion rewrites metadata and tensor names without requantizing. The
profit controls apply only to DFlash; upstream simple, EAGLE3, MTP, and n-gram
modes keep their own defaults.

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
