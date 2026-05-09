# BeeLlama Args Reference

This is the practical BeeLlama tuning reference. It covers the arguments that change model loading, GPU placement, context behavior, prompt caching, KV-cache precision, DFlash/speculative decoding, multimodal behavior, sampling, reasoning, server endpoints, and logs. It is not a byte-for-byte replacement for `--help`; it is the single page to read before tuning a real BeeLlama server.

For the feature overview and public-repo comparison, read [beellama-features.md](beellama-features.md).

## Start With This Mental Model

BeeLlama tuning is easiest if you choose settings in this order:

1. Load the target model and optional multimodal projector.
2. Decide GPU/offload placement for the target model and draft model.
3. Set context length, batch sizes, KV precision, and prompt-cache behavior.
4. Enable DFlash or another speculative backend.
5. Tune sampling, chat/reasoning, server endpoints, and logs.

The DFlash controls have several similar names. These are the common confusion points:

| Setting | It controls | It does not control |
| --- | --- | --- |
| `--ctx-size` | Target model context length | DFlash cross-attention window |
| `--spec-draft-ctx-size` | Draft model context allocation | Target context length |
| `--spec-dflash-cross-ctx` | Recent target hidden-state tokens visible to DFlash | Server KV context length |
| `--spec-draft-n-max` | Main-path draft-token ceiling | Extra tree branches |
| `--spec-draft-n-min` | Minimum draft size required before using a draft | Minimum response length |
| `--spec-branch-budget` | Extra DDTree branch nodes beyond the main path | Main-path draft depth |
| `--spec-draft-top-k` | Candidates per position for tree mode | Sampling `--top-k` |
| `--spec-draft-temp` | Drafter sampling temperature | Target sampler temperature unless `auto` is used |
| `--spec-dm-*` | Runtime controller that can lower/raise active draft depth | Static parser default for `--spec-draft-n-max` |

## Launch Shape

This example mirrors the important shape of a typical DFlash launch script while using placeholder paths:

```sh
llama-server \
  -m "path/to/target.gguf" \
  --mmproj "path/to/mmproj.gguf" \
  --no-mmproj-offload \
  --spec-draft-model "path/to/drafter.gguf" \
  --spec-type dflash \
  --spec-dflash-cross-ctx 1024 \
  --port 8082 \
  -np 1 \
  --kv-unified \
  -ngl all \
  --spec-draft-ngl all \
  -b 2048 -ub 256 \
  --ctx-size 122800 \
  --cache-type-k turbo4 --cache-type-v turbo3_tcq \
  --flash-attn on \
  --cache-ram 0 \
  --jinja \
  --no-mmap --mlock \
  --no-host --metrics \
  --log-timestamps --log-prefix --log-colors off \
  --reasoning on \
  --chat-template-kwargs '{"preserve_thinking":true}' \
  --temp 0.6 --top-k 20 --min-p 0.0
```

What this shape means:

| Block | Purpose |
| --- | --- |
| `-m`, `--mmproj`, `--no-mmproj-offload` | Load the target model and projector, while keeping the projector off GPU. |
| `--spec-type dflash`, `--spec-draft-model`, `--spec-draft-n-max`, `--spec-dflash-cross-ctx` | Enable flat DFlash drafting and choose draft depth/window. |
| `-np 1`, `--kv-unified`, `--ctx-size` | Run one server slot with explicit unified KV and a large target context. |
| `-ngl all`, `--spec-draft-ngl all` | Fully offload target and draft models when devices can hold them. |
| `-b`, `-ub` | Override DFlash-safe parser caps for prompt prefill batching. |
| `--cache-type-k`, `--cache-type-v` | Use asymmetric KV precision. |
| `--cache-ram 0` | Disable the server prompt-cache RAM subsystem. |
| `--no-mmap`, `--mlock`, `--no-host` | Prefer locked model memory and direct backend buffer behavior over filesystem cache behavior. |
| `--metrics`, `--log-*` | Expose metrics and make logs easier to capture. |
| `--reasoning on`, `--chat-template-kwargs`, `--temp`, `--top-k`, `--min-p` | Control thinking/template behavior and target sampling. |

## Model Loading

| Arg | Code default | When to use |
| --- | --- | --- |
| `-m`, `--model FNAME` | Required for normal local runs | Target model GGUF path. |
| `-mu`, `--model-url URL` | Unset | Download/load a target model from URL. |
| `-hf`, `-hfr`, `--hf-repo <user>/<model>[:quant]` | Unset | Resolve target model from Hugging Face. |
| `-hff`, `--hf-file FILE` | Unset | Override the file chosen from `--hf-repo`. |
| `--models-dir PATH` | Unset | Server router/model directory mode. |
| `--models-preset PATH` | Unset | Server model preset file. |
| `--models-max N` | `4` | Maximum loaded models in router mode. |
| `--models-autoload`, `--no-models-autoload` | Enabled | Whether router models autoload. |

Draft model loading uses parallel draft-specific names:

| Arg | Code default | When to use |
| --- | --- | --- |
| `--spec-draft-model`, `-md`, `--model-draft FNAME` | Unset | Draft model GGUF path. Required for DFlash unless loaded through draft HF args. |
| `--spec-draft-hf`, `-hfd`, `-hfrd`, `--hf-repo-draft <user>/<model>[:quant]` | Unset | Resolve draft model from Hugging Face. |
| `--spec-draft-replace`, `--spec-replace TARGET DRAFT` | Unset | Map target tokens/strings to draft equivalents when the draft model needs replacements. |

Bee auto-detects DFlash when a loaded draft model reports a DFlash block size and `--spec-type` was not already `dflash`.

## GPU, CPU, And Offload Placement

| Arg | Code default | When to use |
| --- | --- | --- |
| `-ngl`, `--gpu-layers`, `--n-gpu-layers N|auto|all` | `auto` internally (`-1`) | Target model layer offload. Use `all` for full offload when VRAM allows. |
| `--spec-draft-ngl`, `-ngld`, `--gpu-layers-draft`, `--n-gpu-layers-draft N|auto|all` | `auto` internally (`-1`) | Draft model layer offload. For DFlash, start with `all` if VRAM allows. |
| `-dev`, `--device dev1,dev2,...` | All usable devices | Restrict target model devices. |
| `--spec-draft-device`, `-devd`, `--device-draft dev1,dev2,...` | Inherited/available devices | Restrict draft model devices. |
| `-sm`, `--split-mode none|layer|row|tensor` | `layer` | Multi-GPU split strategy. |
| `-mg`, `--main-gpu INDEX` | `0` | Main GPU for scratch/small tensors. |
| `-ts`, `--tensor-split N0,N1,...` | Unset | Manual tensor split across devices. |
| `-fit`, `--fit on|off` | Enabled | Let the runtime fit unset model/context parameters to available memory. |
| `-fitp`, `--fit-print on|off` | Disabled | Print fit estimates. |
| `-fitc`, `--fit-ctx N` | `4096` | Minimum context size that `--fit` may choose. |
| `-fitt`, `--fit-target MiB0,MiB1,...` | `1024` MiB margin per device | Free-memory margin per device for `--fit`. |
| `-nkvo`, `--no-kv-offload` | Off | Keep KV cache off GPU. Usually hurts GPU-serving speed, but can be needed under VRAM pressure. |
| `--no-host` | Off | Bypass host buffer so extra backend buffers can be used. Used in the launch script. |
| `--no-fused-gdn` | Off | Disable fused Gated Delta Net kernels. Use only to isolate backend/kernel issues. |

CPU/thread controls:

| Arg | Code default | When to use |
| --- | --- | --- |
| `-t`, `--threads N` | `-1` auto | CPU threads for generation. The launch script uses `-t 8`. |
| `-tb`, `--threads-batch N` | Same as `--threads` | CPU threads for prompt/batch processing. |
| `--spec-draft-threads`, `-td`, `--threads-draft N` | Same as target threads | Draft model CPU generation threads. |
| `--spec-draft-threads-batch`, `-tbd`, `--threads-batch-draft N` | Same as draft threads | Draft model prompt/batch threads. |
| `--cpu-mask`, `--cpu-range`, `--cpu-strict` | No strict placement | Pin target CPU work. |
| Draft CPU affinity variants | Inherit target CPU settings | Use `--spec-draft-cpu-mask`, `--spec-draft-cpu-range`, `--spec-draft-cpu-strict`, and batch variants when the drafter needs separate CPU placement. |
| `--prio`, `--prio-batch`, draft priority variants | Normal | Change scheduler priority. Avoid realtime unless you know the host can tolerate it. |
| `--poll`, `--poll-batch`, draft poll variants | `50` for base poll | Busy-wait level. Higher can reduce latency at CPU cost. |

Advanced draft placement:

| Arg | Use |
| --- | --- |
| `--spec-draft-override-tensor`, `-otd`, `--override-tensor-draft` | Override draft tensor buffer placement by tensor-name pattern. |
| `--spec-draft-cpu-moe`, `-cmoed`, `--cpu-moe-draft` | Keep draft MoE tensors on CPU. |
| `--spec-draft-n-cpu-moe`, `--spec-draft-ncmoe`, `-ncmoed`, `--n-cpu-moe-draft N` | Keep only some draft MoE layers on CPU. |

## Context, Batch, And Prompt State

| Arg | Code default | When to use |
| --- | --- | --- |
| `-c`, `--ctx-size N` | `0` means model-trained context | Target model context allocation. Large values need KV memory. |
| `-n`, `--predict`, `--n-predict N` | `-1` no limit | Max new tokens for CLI/server defaults where applicable. |
| `-b`, `--batch-size N` | `2048` | Logical prompt batch size. Larger improves prefill until memory/backend limits. |
| `-ub`, `--ubatch-size N` | `512` | Physical microbatch size. Lower when VRAM spikes or DFlash graph memory grows. |
| `--keep N` | `0` | Tokens to keep from the initial prompt when context is managed. |
| `-np`, `--parallel N` | `1` | Server slots. In server help, `-1` means auto. More slots need more KV and DFlash state. |
| `-cb`, `--cont-batching`; `-nocb`, `--no-cont-batching` | Enabled | Continuous batching for server decode. |
| `--context-shift`, `--no-context-shift` | Disabled | Infinite generation via KV shifting when supported. Disabled under multimodal. |
| `--swa-full` | Disabled | Full-size SWA cache for models where it is supported. |
| `-ctxcp`, `--ctx-checkpoints`, `--swa-checkpoints N` | `32` | Max prompt context checkpoints per slot. Important for long prompt reuse and contexts where speculative decoding needs checkpoints. |
| `-cpent`, `--checkpoint-every-n-tokens N` | `8192` | Create checkpoints during prefill every N tokens. `-1` disables cadence. |
| `-cram`, `--cache-ram N` | `8192` MiB | Server prompt-cache RAM limit. `-1` no limit, `0` disables the prompt cache subsystem. |
| `--cache-prompt`, `--no-cache-prompt` | Enabled | Whether request prompts can use prompt caching. |
| `--cache-reuse N` | `0` disabled | Minimum chunk size to reuse from cache through KV shifting. Disabled under multimodal and unsupported contexts. |
| `-kvu`, `--kv-unified`; `-no-kvu`, `--no-kv-unified` | Raw default false; help notes enabled if slots are auto | Single unified KV buffer shared across sequences. Required by idle-slot cache. |
| `--cache-idle-slots`, `--no-cache-idle-slots` | Enabled in params, but requires unified KV and cache RAM | Save and clear idle slots when a new task starts. Disabled by the server if requirements are not met. |

DFlash-specific parser safety:

| Situation | Effective behavior |
| --- | --- |
| `--spec-type dflash` and no `--spec-draft-ctx-size` | Bee sets draft context to `256`. |
| `--spec-type dflash` and no explicit `-b` | Bee caps inherited batch default to `256`. |
| `--spec-type dflash` and no explicit `-ub` | Bee caps inherited microbatch default to `64`. |
| You pass explicit `-b` or `-ub` | Bee keeps your values. |

The parser caps are OOM safety defaults, not performance guidance. For large prompt prefill, explicitly set `-b` and `-ub` after measuring memory.

## KV Cache Precision

K and V cache precision are independent:

```sh
--cache-type-k turbo4 --cache-type-v turbo3_tcq
--cache-type-k q8_0   --cache-type-v turbo3_tcq
--cache-type-k f16    --cache-type-v f16
```

Accepted KV cache type names in the current parser include:

| Family | Values |
| --- | --- |
| Floating | `f32`, `f16`, `bf16` |
| Upstream quantized | `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1` |
| Bee/Turbo lineage | `turbo2`, `turbo3`, `turbo4`, `turbo2_tcq`, `turbo3_tcq` |

Bee fork cache storage:

| Type | Storage | Best first use |
| --- | ---: | --- |
| `turbo4` | `4.125` bpv | Strong first compressed K-cache candidate. |
| `turbo3` | `3.125` bpv | More compression than `turbo4`; uses `QK_TURBO3 = 128` in Bee. |
| `turbo2` | `2.125` bpv | Stronger compression, higher quality risk. |
| `turbo3_tcq` | `3.25` bpv | TCQ path, commonly useful for V in Bee experiments. |
| `turbo2_tcq` | `2.25` bpv | Most compressed TCQ path; verify carefully. |

For long-context DFlash serving, the launch script's asymmetric choice is:

```sh
--cache-type-k turbo4 --cache-type-v turbo3_tcq
```

Do not assume enum compatibility with TheTom's public TurboQuant fork. Bee uses the buun enum order for Turbo/TCQ types while keeping a 128-value `turbo3` block.

## Model Memory Files And OS Cache

| Arg | Default | When to use |
| --- | --- | --- |
| `--mmap`, `--no-mmap` | mmap enabled | `--no-mmap` avoids relying on filesystem cache behavior; the launch script uses it. |
| `--mlock` | Disabled | Ask the OS to keep model memory resident. Useful when paging would ruin latency. |
| `-dio`, `--direct-io` | Disabled | Read model data without normal buffered I/O. |

`--no-mmap --mlock` is a deliberate "keep the model resident" style. It can fail or be limited by OS permissions and available RAM.

## Flash Attention And Backend Behavior

| Arg | Default | When to use |
| --- | --- | --- |
| `-fa`, `--flash-attn on|off|auto` | `auto` | Force Flash Attention on for cache types/backends where you expect support; use `off` to isolate kernel issues. |
| `--no-fused-gdn` | Off | Disable fused GDN kernels for debugging or backend compatibility. |

For Turbo/TCQ cache types, verify the backend path you use. A cache type being accepted by the parser does not prove every backend/kernel combination is fast or supported.

## DFlash And Speculative Decoding

Select the speculative backend:

| Arg | Code default | Values |
| --- | --- | --- |
| `--spec-type MODE` | `none` | `none`, `ngram-cache`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `suffix`, `copyspec`, `recycle`, `dflash` |

DFlash core args:

| Arg | Code default | Use |
| --- | ---: | --- |
| `--spec-draft-model`, `-md`, `--model-draft` | Unset | Draft GGUF path. DFlash needs a DFlash-compatible draft model. |
| `--spec-draft-ctx-size`, `-cd`, `--ctx-size-draft` | Raw `0`; DFlash effective `256` if omitted | Draft context allocation. Not the same as target `--ctx-size`. |
| `--spec-draft-ngl`, `-ngld` | `auto` | Draft model GPU layers. |
| `--spec-draft-type-k`, `-ctkd`, `--cache-type-k-draft` | `f16` | Draft model K cache precision. |
| `--spec-draft-type-v`, `-ctvd`, `--cache-type-v-draft` | `f16` | Draft model V cache precision. |
| `--spec-draft-n-max` | `16` | Base max main-path draft tokens. |
| `--spec-draft-n-min` | `0` | Minimum draft length required before speculative verification is used. |
| `--spec-branch-budget` | `0` | DDTree branch nodes beyond the main draft path. `0` means flat DFlash. |
| `--tree-budget TOTAL` | Unset legacy input | Public-buun-style total-node budget. Bee converts to branch budget unless canonical branch budget is present. |
| `--spec-draft-top-k`, `--draft-topk` | `1` | Candidates per draft position for tree mode. Forced to `1` when branch budget is `0`. |
| `--spec-draft-p-split`, `--draft-p-split` | `0.10` | Probability threshold for creating tree branches. |
| `--spec-draft-p-min`, `--draft-p-min` | `0.0` | Minimum draft probability gate. |
| `--spec-draft-temp T` | `0.0` | DFlash drafter temperature. `0` greedy, positive uses sampled/Gumbel path, `auto` mirrors target temp. |
| `--spec-dflash-cross-ctx N` | `512` | Recent target hidden-state tokens visible to the DFlash drafter. |
| `--spec-dflash-max-slots N` | `1` | Max server slots with DFlash state; higher slots fall back to non-speculative decoding. |

Flat DFlash:

```sh
llama-server -m target.gguf \
  --spec-type dflash \
  --spec-draft-model draft-dflash.gguf \
  --spec-draft-n-max 16 \
  --spec-branch-budget 0 \
  --spec-dflash-cross-ctx 512 \
  -ngl all \
  --spec-draft-ngl all
```

Tree DFlash:

```sh
llama-server -m target.gguf \
  --spec-type dflash \
  --spec-draft-model draft-dflash.gguf \
  --spec-draft-n-max 16 \
  --spec-branch-budget 4 \
  --spec-draft-top-k 4 \
  --spec-draft-p-split 0.10
```

Sampled DFlash:

```sh
llama-server -m target.gguf \
  --spec-type dflash \
  --spec-draft-model draft-dflash.gguf \
  --spec-draft-temp auto \
  --temp 0.6
```

Sampled DFlash activates the rejection-sampling path when both drafter and target temperature exceed zero. Draft log-probabilities must be available for rejection sampling to produce correct output. Without those conditions, Bee uses the normal greedy speculative path.

## Adaptive Draft-Max

Adaptive Draft-Max is enabled by default for DFlash. It can reduce the active draft depth below `--spec-draft-n-max`, turn speculation off after consecutive below-threshold cycles, and probe again periodically.

| Arg | Code default | Use |
| --- | ---: | --- |
| `--spec-dm-adaptive`, `--no-spec-dm-adaptive` | Enabled | Enable or disable adaptive depth. Disable for fixed-depth benchmarks. |
| `--spec-dm-controller MODE` | `profit` | `profit` or `fringe`. |
| `--spec-dm-probe-interval N` | `16` | Cycles to wait before trying a speculative cycle when speculation is off. |
| `--spec-dm-probe-fraction F` | `0.25` | Fraction of base_n_max to use as the probe depth when speculation is off. |
| `--spec-dm-explore-interval N` | `12` | Draft at a higher depth every N cycles to collect timing data beyond the current n_max. |
| `--spec-dm-off-dwell N` | `8` | Consecutive cycles below the profit/fringe threshold before speculation is disabled. |
| `--spec-dm-fringe-min F` | `0.30` | Fringe rate below which n_max drops toward 0 (after off-dwell). |
| `--spec-dm-fringe-max F` | `0.50` | Fringe rate at or above which full base_n_max is used. |
| `--spec-dm-min-reach N` | `3` | Minimum samples at a new draft position before fringe can promote to it. |
| `--spec-dm-profit-min F` | `0.05` | Minimum relative throughput improvement over no-spec baseline before dwell clears. |
| `--spec-dm-profit-raise-margin F` | `0.05` | Relative margin a higher depth must exceed to replace the current depth. |
| `--spec-dm-profit-lower-margin F` | `0.05` | Relative margin a lower depth must exceed to replace the current depth. |
| `--spec-dm-profit-ewma-alpha F` | `0.15` | Smoothing factor for acceptance and timing running averages. |
| `--spec-dm-profit-min-samples N` | `3` | Minimum observations per position/depth before scoring that depth as ready. |
| `--spec-dm-profit-warmup N` | `0` | Initial cycles at full depth before profit decisions run (0 = use --spec-dm-profit-min-samples). |

Use `profit` for normal serving. Use `fringe` when you want behavior tied more directly to observed draft acceptance near the active tail. Use `--no-spec-dm-adaptive` only when comparing fixed `--spec-draft-n-max` values or reproducing a narrow benchmark.

## Other Speculative Backends

| Mode | Key args | Code defaults | When to use |
| --- | --- | --- | --- |
| `ngram-cache` | `--lookup-cache-static`, `--lookup-cache-dynamic` | Paths unset | You have reusable lookup cache files. |
| `ngram-mod` | `--spec-ngram-mod-n-min`, `--spec-ngram-mod-n-max`, `--spec-ngram-mod-n-match` | `48`, `64`, `24` | Repeated-context workloads with controllable draft length. |
| `ngram-simple` | `--spec-ngram-simple-size-n`, `--spec-ngram-simple-size-m`, `--spec-ngram-simple-min-hits` | `12`, `48`, `1` | Simple no-draft repeated-context baseline. |
| `ngram-map-k` | `--spec-ngram-map-k-size-n`, `--spec-ngram-map-k-size-m`, `--spec-ngram-map-k-min-hits` | `12`, `48`, `1` | Map-backed ngram proposal tuning. |
| `ngram-map-k4v` | `--spec-ngram-map-k4v-size-n`, `--spec-ngram-map-k4v-size-m`, `--spec-ngram-map-k4v-min-hits` | `12`, `48`, `1` | Alternative map-backed ngram proposal tuning. |
| `copyspec` | No public CLI gamma knob in current parser | Gamma `6` in params | Copy repeated prompt/context spans without a model drafter. |
| `recycle` | No public CLI k knob in current parser | `k=8` successors | Repeated token-successor patterns. |
| `suffix` | No public CLI suffix knobs in current parser | depth `64`, factor `2.0`, offset `0.0`, min prob `0.1` | Repeated suffix continuations. |

Presets:

| Arg | Behavior |
| --- | --- |
| `--spec-default` | Sets `ngram-mod` with match `24`, min `48`, max `64`. |
| `--spec-dflash-default` | Sets DFlash mode with `p_min=0`, `n_max=16`, `n_min=0`. Prefer explicit `--spec-type dflash --spec-draft-model ... --spec-draft-n-max ...` for reproducible commands, because normal speculative normalization copies draft `n_max/n_min` after parsing. |

Removed generic ngram args:

| Removed arg | Use instead |
| --- | --- |
| `--spec-ngram-size-n` | The backend-specific `--spec-ngram-*-size-n` or `--spec-ngram-mod-n-match`. |
| `--spec-ngram-size-m` | The backend-specific `--spec-ngram-*-size-m`. |
| `--spec-ngram-min-hits` | The backend-specific `--spec-ngram-*-min-hits`. |

## Legacy DFlash Spellings

Accepted aliases:

| Alias | Current target |
| --- | --- |
| `--draft`, `--draft-n`, `--draft-max` | `--spec-draft-n-max` |
| `--draft-min`, `--draft-n-min` | `--spec-draft-n-min` |
| `--draft-topk` | `--spec-draft-top-k` |
| `--draft-p-split` | `--spec-draft-p-split` |
| `--draft-p-min` | `--spec-draft-p-min` |
| `--tree-budget TOTAL` | Legacy total-node DDTree input converted to branch-only semantics |

Names from older buun-era experiments that are not accepted by the current Bee parser include:

```text
--draft-temp
--dflash-cross-ctx
--dflash-max-slots
--dm-adaptive
--dm-ar-up
--dm-ar-down
--dm-ar-window
--dm-probe-interval
--dm-probe-fraction
--spec-dm-ar-up
--spec-dm-ar-down
--spec-dm-ar-window
```

Use the canonical `--spec-*` names in new commands.

## Multimodal

| Arg | Default | Use |
| --- | --- | --- |
| `-mm`, `--mmproj FILE` | Unset | Multimodal projector path. |
| `-mmu`, `--mmproj-url URL` | Unset | Projector URL. |
| `--mmproj-auto`, `--no-mmproj`, `--no-mmproj-auto` | Auto enabled | Use projector file if available, especially with HF model loading. |
| `--mmproj-offload`, `--no-mmproj-offload` | Offload enabled | Put projector on GPU or keep it off GPU. The launch script uses `--no-mmproj-offload`. |
| `--image`, `--audio FILE` | Unset | CLI multimodal input files. |

Runtime compatibility rules when `--mmproj` is loaded:

- context shift is disabled
- cache reuse is disabled
- flat DFlash can remain active
- `--spec-branch-budget` is forced to `0` under DFlash
- non-DFlash speculative types are set to none

This means DFlash plus multimodal is supported only as flat DFlash in the current server path.

## Chat, Reasoning, And Sampling

Chat/template args:

| Arg | Default | Use |
| --- | --- | --- |
| `--jinja`, `--no-jinja` | Enabled | Jinja chat template engine. |
| `--chat-template TEMPLATE` | Model metadata | Inline/custom template name or content depending on parser path. |
| `--chat-template-file FILE` | Unset | Load a custom chat template from file. |
| `--chat-template-kwargs JSON` | Unset | Template kwargs. The launch script uses `{"preserve_thinking":true}`. |
| `--reasoning on|off|auto`, `-rea` | `auto` | Enable thinking if supported by template, disable it, or force it on. |
| `--reasoning-format none|deepseek|deepseek-legacy` | Code default `deepseek`; parser help text says `auto` | How thought tags are parsed and returned. |
| `--reasoning-budget N` | `-1` unrestricted | `0` ends thinking immediately, positive values cap thinking tokens. |
| `--reasoning-budget-message MESSAGE` | None | Message injected when the reasoning budget is exhausted. |

Sampling args people commonly tune:

| Arg | Code default | Use |
| --- | ---: | --- |
| `--temp`, `--temperature` | `0.80` | Target sampler temperature. `0` means greedy. |
| `--top-k N` | `40` | Keep top K target tokens before later samplers. This is not DFlash `--spec-draft-top-k`. |
| `--top-p N` | `0.95` | Nucleus sampling; `1.0` disables. |
| `--min-p N` | `0.05` | Minimum probability relative to best token; `0.0` disables. |
| `--top-nsigma`, `--top-n-sigma N` | `-1.0` | Sigma-based filter; negative disables. |
| `--xtc-probability N` | `0.0` | XTC probability; `0.0` disables. |
| `--xtc-threshold N` | `0.10` | XTC threshold. |
| `--typical`, `--typical-p N` | `1.0` | Locally typical sampling; `1.0` disables. |
| `--repeat-last-n N` | `64` | Repetition penalty window; `-1` means context size. |
| `--repeat-penalty N` | `1.0` | Repetition penalty; `1.0` disables. |
| `--presence-penalty N` | `0.0` | Presence penalty. |
| `--frequency-penalty N` | `0.0` | Frequency penalty. |
| `--dry-multiplier N` | `0.0` | DRY repetition penalty multiplier; `0.0` disables. |
| `--dry-base N` | `1.75` | DRY base. |
| `--dry-allowed-length N` | `2` | DRY allows this repetition length before penalty. |
| `--dry-penalty-last-n N` | `-1` | DRY scan window; `-1` means context size. |
| `--adaptive-target N` | `-1.0` | Adaptive-p target probability; negative disables. |
| `--adaptive-decay N` | `0.90` | Adaptive-p decay. |
| `--dynatemp-range N` | `0.0` | Dynamic temperature range; `0.0` disables. |
| `--dynatemp-exp N` | `1.0` | Dynamic temperature exponent. |
| `--mirostat N` | `0` | `0` disabled, `1` Mirostat, `2` Mirostat 2.0. |
| `--mirostat-lr N` | `0.10` | Mirostat learning rate. |
| `--mirostat-ent N` | `5.00` | Mirostat target entropy. |

If you use `--spec-draft-temp auto`, changing target `--temp` also changes DFlash drafter temperature. If `--spec-draft-temp` is a number, target `--temp` and drafter temperature are independent.

## Reasoning Loop Guard

Bee-specific guard args:

| CLI arg | JSON field | Code default | Use |
| --- | --- | ---: | --- |
| `--reasoning-loop-guard MODE` | `reasoning_loop_guard` | `force-close` | `off`, `force-close`, or `stop`. |
| `--reasoning-loop-min-tokens N` | `reasoning_loop_min_tokens` | `1024` | Do not check before this many hidden reasoning tokens. |
| `--reasoning-loop-window N` | `reasoning_loop_window` | `2048` | Tail window scanned for repeated loops. |
| `--reasoning-loop-max-period N` | `reasoning_loop_max_period` | `512` | Largest periodic loop length checked. |
| `--reasoning-loop-min-coverage N` | `reasoning_loop_min_coverage` | `768` | Minimum repeated coverage needed to trigger. |
| `--reasoning-loop-check-interval N` | `reasoning_loop_check_interval` | `32` | Accepted-token interval between checks. |
| `--reasoning-loop-interventions N` | `reasoning_loop_interventions` | `1` | Force-close interventions before stop behavior. |

Validation rules in the server require positive windows/periods/coverage/intervals, `window >= min_coverage`, `max_period <= window / 3`, and `min_tokens >= min_coverage`.

## Server, Endpoints, And Network

| Arg | Code default | Use |
| --- | --- | --- |
| `--host HOST` | `127.0.0.1` | Server bind host. |
| `--port PORT` | `8080` | Server port. |
| `-to`, `--timeout N` | `600` | Read timeout. |
| `--threads-http N` | `-1` | HTTP worker threads. |
| `--api-key KEY` | Unset | Require API key. Do not put secrets in committed scripts. |
| `--api-key-file FNAME` | Unset | Read API key from file. Prefer this over command-line secrets. |
| `--ssl-key-file FNAME` | Unset | TLS key file. |
| `--ssl-cert-file FNAME` | Unset | TLS cert file. |
| `--metrics` | Disabled | Enable Prometheus-compatible metrics endpoint. |
| `--props` | Disabled | Enable changing global properties via `POST /props`. |
| `--slots`, `--no-slots` | Enabled | Expose or hide slot monitoring endpoint. |
| `--slot-save-path PATH` | Unset | Enable slot save/restore actions. |
| `--media-path PATH` | Unset | Path for served/uploaded media handling. |
| `--webui`, `--no-webui` | Enabled | Enable built-in Web UI. |
| `--webui-config JSON`, `--webui-config-file PATH` | Unset | Web UI settings. |
| `--webui-mcp-proxy`, `--no-webui-mcp-proxy` | Disabled | Experimental MCP CORS proxy; do not enable for untrusted environments. |

## Logs

| Arg | Use |
| --- | --- |
| `--log-timestamps` | Include timestamps in logs. |
| `--log-prefix` | Include log prefixes. |
| `--log-colors on|off|auto` | Control ANSI color output. Use `off` for clean log files. |

The launch script uses all three so captured logs are stable:

```sh
--log-timestamps --log-prefix --log-colors off
```

## Request JSON Overrides

Server requests can override several defaults without restarting the process.

Speculative fields:

| JSON field | Meaning |
| --- | --- |
| `speculative.n_min` | Per-request minimum draft length. |
| `speculative.n_max` | Per-request max main-path draft length. |
| `speculative.branch_budget` | Per-request branch budget. |
| `speculative.tree_budget` | Legacy total-node budget, converted to branch budget if `speculative.branch_budget` is absent. |
| `speculative.p_min` | Per-request draft probability gate. |

Prompt/cache fields:

| JSON field | Meaning |
| --- | --- |
| `cache_prompt` | Whether the request can use prompt cache. |
| `n_cache_reuse` | Minimum chunk size for KV-shift cache reuse. |

Reasoning loop fields are listed in the loop-guard section. Sampling fields such as `temperature`, `top_k`, `top_p`, `min_p`, `mirostat`, `adaptive_target`, and `adaptive_decay` are also read from request JSON in the server task parser.

## Recipes

### Flat DFlash, Long Context, Compressed KV

```sh
llama-server -m target.gguf \
  --spec-type dflash \
  --spec-draft-model draft-dflash.gguf \
  --spec-draft-n-max 16 \
  --spec-dflash-cross-ctx 1024 \
  --spec-branch-budget 0 \
  --ctx-size 51200 \
  -ngl all \
  --spec-draft-ngl all \
  -b 2048 \
  -ub 256 \
  --kv-unified \
  --cache-type-k turbo4 \
  --cache-type-v turbo3_tcq \
  --flash-attn on
```

### Tree DFlash Experiment

```sh
llama-server -m target.gguf \
  --spec-type dflash \
  --spec-draft-model draft-dflash.gguf \
  --spec-draft-n-max 16 \
  --spec-branch-budget 4 \
  --spec-draft-top-k 4 \
  --spec-draft-p-split 0.10 \
  --spec-dm-controller profit
```

Measure tree mode against flat DFlash and no-spec baselines. Tree nodes consume batch/microbatch capacity and extra verification work.

### Multimodal Flat DFlash

```sh
llama-server -m target.gguf \
  --mmproj mmproj.gguf \
  --spec-type dflash \
  --spec-draft-model draft-dflash.gguf \
  --spec-branch-budget 0
```

Do not expect tree DFlash, context shift, or cache reuse to survive multimodal initialization in the current server path.

### No-Draft Repeated-Context Baseline

```sh
llama-server -m model.gguf \
  --spec-type ngram-mod \
  --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 48 \
  --spec-ngram-mod-n-max 64
```

### Reasoning Guard For Thinking Models

```sh
llama-server -m model.gguf \
  --reasoning on \
  --reasoning-loop-guard force-close \
  --reasoning-loop-min-tokens 1024 \
  --reasoning-loop-window 2048 \
  --reasoning-loop-interventions 1
```

## Verification Discipline

When changing DFlash, cache precision, context length, or batch size, compare against a no-spec baseline and keep all other inputs fixed:

- target model file
- draft model file, if any
- exact command line
- commit ID and dirty-worktree status
- prompt or prompt hash
- context length, cache types, batch sizes, and slot count
- target sampling settings
- prompt TPS, generation TPS, wall time, draft count, accepted count, and peak memory

Do not carry performance numbers between machines, model files, backends, or dirty worktrees without rerunning the measurement.
