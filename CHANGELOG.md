# Changelog

## v0.4.0

- Merged upstream llama.cpp through merge `827bfda66`, whose upstream parent is
  `32e789fdf`, and re-based BeeLlama's maintained extensions on its current
  model, memory, speculative, and server architecture.
- Replaced the fork DFlash stack with upstream `draft-dflash`. `dflash` remains a
  warned compatibility alias. Draft GGUFs must use upstream's `dflash`
  architecture, metadata, and tensor names.
- Corrected the upstream DFlash draft-context merge so it owns a normal KV cache.
  Without this, draft graph reservation dereferenced a null memory context and
  crashed both normal startup and the default memory-fit probe.
- Removed TurboQuant/TCQ cache types, TQ weight formats, DDTree, CopySpec, the
  old DFlash GPU ring/capture/tape paths, the fringe adaptive controller, and all
  fork-only DFlash environment variables and arguments.
- Redirected the removed `turbo2`/`turbo3`/`turbo4` and TCQ cache names to the
  same-width KVarN presets with warnings. GGUF files marked as the former TQ3/TQ4
  weight formats now fail early with a re-quantization error instead of being
  interpreted as re-slotted v0.4.0 types.
- Kept KVarN and its low-bit standard cache support. CUDA FlashAttention now
  explicitly dispatches KVarN views to descriptor-native kernels, including clean
  MMA fallback for valid pairs outside the fast decode matrix.
- Restored CUDA `SET_ROWS` and `GET_ROWS` support for the standard `q6_1`,
  `q6_0`, `q3_1`, `q3_0`, `q2_1`, and `q2_0` KV-cache types after the upstream
  merge dropped their backend dispatch and support declarations.
- Replaced the three-tier FlashAttention quant build policy with a 103-pair
  default standard vector matrix and a 169-pair `GGML_CUDA_FA_ALL_QUANTS` matrix.
  KVarN keeps 15 balanced fast decode pairs by default and 36 with ALL. The
  default-on `GGML_CUDA_KVARN` option is its only CUDA compilation gate; disabling
  it omits all dedicated KVarN kernels and templates. The obsolete
  `GGML_CUDA_FA_HALF_QUANTS` option is gone.
- Added migration errors for the removed `copyspec`, `suffix`, and `recycle`
  speculative type names, pointing users to `draft-dflash` or upstream n-gram
  modes instead of returning a generic unknown-type message.
- Re-ported the profit-only adaptive DFlash controller and server reasoning-loop
  guard onto upstream token/sampler and checkpoint behavior.
- Made an omitted DFlash `--spec-draft-n-max` use the drafter's trained block
  depth (`dflash.block_size - 1`, normally 15) instead of upstream's default 3.
  Explicit values still win, and the profit controller remains default-on.
- Added opted-in realtime reasoning control: a streaming chat completion armed
  with `reasoning_control` can be moved to its final answer through
  `/v1/chat/completions/control`. The endpoint reports success only when an
  active reasoning sampler actually accepts the transition; inactive,
  completed, and unknown completions report failure.
- Hardened router management. `GET /models` is read-only and returns sanitized
  model identity, capability, source, and status data. Refresh is now
  `POST /models/reload`, protected by the normal API-key middleware when keys
  are configured. Hugging Face tokens are removed from child arguments,
  presets, logs, and public responses and reach child processes only through
  the `HF_TOKEN` environment variable.
- Made standard and KVarN state restoration transactional: tensor writes and
  metadata are staged until the complete frame validates, then committed once.
  Deferred exact-tail copies now distinguish no work, success, allocation or
  transfer preparation failure, and compute failure; failed copies are rolled
  back and cannot be observed as successful by save, decode, or server callers.
- Made exact-tail attention routes carry the model-owned per-layer relative-bias
  contract. Biased models such as T5 cannot select a bias-incompatible native
  FlashAttention route, and graph construction verifies the recorded contract.
- Kept KVarN prefill eager at completed 128-token record boundaries while the
  live exact suffix remains available. This is a store-scheduling rule, not a
  persistent-layout or state-format change.
- Renamed Bee's internal Q2 cache enum to `GGML_TYPE_Q2_0S` while keeping `q2_0`
  at the cache CLI boundary, avoiding a format collision with upstream's Q2_0
  weight type. Saved sessions from older Bee releases are not compatible with
  the re-slotted cache type IDs.
- Rebased Bee's release workflow, package verification, CUDA 12.4/13.1 assets,
  and container metadata on the new tree. Container descriptions now advertise
  upstream DFlash and KVarN instead of removed TurboQuant/TCQ support. Release
  packages and publication now depend on portable CPU, Windows CUDA default,
  CUDA ALL_QUANTS, and CUDA-without-KVarN behavioral gates at one exact source
  SHA; a manual dry run builds evidence but cannot publish.
- RTX 3090 release validation measured Qwen3.6-27B KVarN4 KLD `0.002400`,
  Gemma-4-31B KVarN4 KLD `0.402575` versus q5_0 `0.415296`, and upstream
  DFlash at `73.81` median decode tokens/s with `40.1%` draft acceptance.
  Upstream DFlash was `12.61%` slower than the v0.3.2 legacy DFlash stack in the
  matched three-prompt server gate, so reduced verification/backend sampling
  remain the first follow-up upstream contribution track.

## v0.3.2

- Merged a newer upstream llama.cpp master after the current `main` baseline. Notable inherited changes include Granite 4 Vision, Gemma 4 MTP including E2B/E4B assistants, Gemma 4 unified conversion and audio-projector fixes, multimodal video input with ffmpeg in the released image, Qwen-VL frame merge support, Mistral-Medium-3.5 conversion, the unified LFM2/LFM2.5 tool parser and reasoning round-trip fixes, speculative vocab-compatibility checks, the placeholder-bitmap token counting and `*/input_tokens` API, optional server prompt logging, KV-cache cell-sharing/copy-avoidance fixes, `GGML_OP_COL2IM_1D`, ggml 0.14.0, CUDA 13.3 release images, HIP gfx1152/gfx1153 support, and backend/UI improvements across WebGPU, Vulkan, SYCL, Metal, CPU, and the Web UI.
- Added experimental KVarN KV-cache compression for target contexts through `--cache-type-k` / `--cache-type-v` pseudo types `kvarn2`, `kvarn3`, `kvarn4`, `kvarn5`, `kvarn6`, and `kvarn8`. K and V can be selected independently across all 2/3/4/5/6/8-bit pairs, with one-sided KVarN inputs normalized to a concrete pair and draft/auxiliary contexts kept on normal cache types.
- Wired KVarN into Qwen3.6 and Gemma 4 target memory paths, including 128/256/512-dimensional K/V heads, unified and non-unified KV storage, supported single-stream iSWA/SWA rings, explicit `--cache-type-k-swa` / `--cache-type-v-swa` overrides for SWA-layer precision, prompt-cache state save/restore, and bit-width-matched fallback cache types for layers that cannot use KVarN records. Unsupported placements fail closed unless normal-KV fallback is explicitly enabled.
- Added native KVarN runtime support: CPU/CUDA store ops, native CUDA FlashAttention view consumption without graph-level F16 materialization, bounded windowed prefill for Qwen/Gemma/SWA paths, compact group-range prompt-cache state, ROCm/HIP low-shared-memory store support, guarded Vulkan store support, and `llama-bench` KVarN cache names.
- Added new KV/cache types: `turbo4_tcq` plus standard quantized `q2_0`, `q2_1`, `q3_0`, `q3_1`, and `q6_1`, with CPU/CUDA quantize/dequantize, MMQ/vec-dot, FlashAttention vec coverage, and use as KVarN non-KVarN-layer fallback targets where appropriate.
- Reworked CUDA/HIP/MUSA FlashAttention cache-type coverage. The no-flag vec build now focuses on the recommended q-cache and KVarN fallback pairs, `GGML_CUDA_FA_HALF_QUANTS` adds the K>=V half matrix for broader Turbo/TCQ experiments, `GGML_CUDA_FA_ALL_QUANTS` keeps the full matrix, and runtime diagnostics report or optionally ignore uncompiled pairs instead of failing opaquely. Turbo/TCQ route planning now validates effective K/V types, fixes the issue #41 CUDA paths, and keeps Gemma-sized D256/D512 mixed-Turbo routes away from unsupported vec paths.
- Improved DFlash serving relative to v0.3.1: tensor-split Meta target placement keeps auto-detected drafters on compatible placement, reduced verification is gated by backend capability, DFlash mixed with other speculative types is detected order-insensitively for rollback planning, and MTP draft policy is isolated from DFlash-only adaptive logic.
- Reduced DFlash/recurrent and prompt-cache memory pressure. Flat DFlash on recurrent targets now uses recurrent-only rollback state instead of dead attention/KV backup streams; prompt-cache saves are prepared and byte-counted before commit; cached prompt entries copy only the newest fitting context checkpoints; and `--cache-ram` remains scoped to prompt-cache storage rather than active context-checkpoint policy.
- Improved the DFlash `profit` adaptive draft-max controller so cold starts hold the maximum useful depth while lower depths are characterized through gated probes, then demote only when measurements show a lower depth is actually faster.
- Hardened shipped builds and release artifacts: ROCm/HIP shuffle compatibility, HIP/MUSA KVarN build fixes, Windows CPU OpenMP runtime packaging, Intel SYCL Docker images updated from compute runtime 25.40 / IGC v2.20.5 to compute runtime 26.18 / IGC v2.34.4, and release downloads/notes filtered to final packages and runtime artifacts.

## v0.3.1

- Merged latest upstream llama.cpp master. This pulls in Gemma 4 12B and Gemma 4 unified multimodal support fixes, including non-causal vision, unified audio/vision projector handling, and FPE fixes; Qwen3.5 post-norm hidden-state behavior for MTP; CUDA KV-cache quantization preallocation and PDL race fixes; WebGPU FlashAttention refactoring with standardized quantization support; CPU backend improvements for RVV/SVE; lower-latency Metal command-buffer status polling; Mermaid diagram rendering and preview support in `tools/ui`; updated BoringSSL, SYCL documentation, save/load-state tests, Docker docs, and small CI/release maintenance.
- Repaired CUDA fused TurboQuant FlashAttention for same-type `turbo2`, `turbo3`, and `turbo4` K/V caches. The fused MMA path now loads each supported format correctly, while mixed TurboQuant and TCQ pairs stay on the established non-fused paths; TurboQuant/TCQ partial KV offload now fails early instead of falling back to an incompatible CPU cache and reaching a scheduler crash. Added `GGML_TURBO_FA_DEBUG=1` path diagnostics and regression coverage for the supported dispatch matrix.
- Updated release packaging and documentation. HIP/ROCm builds now include all quantized FlashAttention combinations, and the prebuilt binary and Docker image lists reflect the current release outputs.

## v0.3.0

- Updated to a much newer llama.cpp base. The upstream refresh brings native MTP speculative decoding, parallel drafting and backend sampling work, the unified `llama` app, newer server/API behavior, the `tools/ui` Web UI restructure, model and converter additions, multimodal improvements, and backend gains across CUDA/HIP, Metal, Vulkan, SYCL, OpenCL, WebGPU, Hexagon, ZenDNN, and SpacemiT.
- Added usable MTP serving on the Bee tree. `draft-mtp`/`mtp` now has its own speculative path instead of being mixed with DFlash state, uses draft KV cache types, cleans up draft resources on sleep, works with target pre-norm hidden-state capture, and handles text requests when an `mmproj` is loaded.
- Cleaned up DFlash command-line behavior around canonical `--spec-*` arguments. DFlash can still be selected with `--spec-type dflash` or auto-detected from compatible draft GGUF metadata, but stale aliases such as `--spec-dflash-default`, `--draft*`, `--draft-topk`, `--draft-model`, `--tree-budget`, `--dflash-max-slots`, `--spec-draft-replace`, and `--spec-replace` were removed.
- Made DFlash defaults match the new upstream speculative surface without changing the practical DFlash defaults. Raw `--spec-draft-n-max` stays at upstream `3`, DFlash raises the effective omitted draft max to `16` only after explicit or auto-detected DFlash, omitted `--spec-draft-ctx-size` still becomes `256`, and DFlash no longer lowers target `-b 2048` / `-ub 512` unless the user asks for smaller batches.
- Made DFlash-only arguments safe around non-DFlash modes. DFlash-only controls warn and no-op for MTP or other speculative types, while still surviving long enough for server-side DFlash draft-model auto-detection when no explicit `--spec-type dflash` was passed.
- Expanded multi-slot DFlash serving. DFlash slots now default to server parallelism, `--spec-dflash-max-slots` caps them below `-np` when needed, uneven `-np` target batches are split safely, mixed speculative/non-speculative target batches are avoided, and flat multi-slot DFlash can use shared drafter batching with `GGML_DFLASH_SHARED_DRAFT_BATCH=0` as the fallback switch.
- Reworked adaptive DFlash draft depth for live serving. The default `profit` controller now seeds and periodically remeasures a no-spec baseline, probes shallow/mid/full positive depths, backs off failed wake probes, avoids premature demotion, resets per-request state correctly, preserves compatible continuation state, and gates timing logs behind DFlash profiling.
- Added default-on device-aware DFlash GPU capture/tape/replay for split CUDA/ROCm target placement. Hidden capture, prefill capture, recurrent tape, conv replay, direct GDN replay, rollback copies, and synchronization now follow each layer's backend device, with CPU/eval-callback fallback and `GGML_DFLASH_MULTI_GPU_TAPE=0` / `GGML_DFLASH_ALLOW_MULTI_GPU_TAPE=0` kill switches.
- Hardened DFlash device placement. Explicit single draft-device placement pins the target output tensor before target load, auto-detected DFlash drafters stay single-device by default unless draft devices are explicit, iGPU backends are accepted for GPU paths, tensor-split/meta placement is guarded, CUDA helper calls preserve the caller device, and peer D2D copies are used where available.
- Hardened DFlash hidden-ring and prompt-cache state. The server guards GPU hidden-ring spans, discards stale DFlash ring checkpoints, skips DFlash checkpoints for uncached prompts, restores shared prefill capture state, and avoids requesting raw prompt logits from DFlash paths that should not read them.
- Reduced DFlash accept and verification overhead while failing closed on bad reduced-logit state. The accept path defers single-slot rollback sync and drafter KV maintenance, keeps the fused GDN 4D state fast path, adds CPU f16 `out_prod` fallback, and disables DFlash drafting after repeated invalid reduced-logits drafts instead of looping or corrupting state.
- Improved DFlash tool-call and reasoning behavior. DFlash can keep drafting before a lazy tool-call grammar actually constrains output, stale drafter KV state is cleared before long tool-call continuations, stable partial tool-call headers stream earlier, streaming reasoning deltas stay isolated, and streamed title generation suppresses leading thinking syntax.
- Kept flat DFlash available with multimodal serving while making unsupported combinations explicit. With `--mmproj`, Bee keeps flat DFlash usable, forces DFlash tree branch budget to `0`, disables non-DFlash speculative modes, disables unsupported context-shift/cache-reuse paths, and fixes MTP text-only requests when an `mmproj` is present.
- Updated Qwen 3.5/3.6 speculative paths. Qwen gets per-layer KV heads, final-layer output gathering, stable MTP draft context behavior, and Qwen DDTree conv/GDN paths for tree verification.
- Removed legacy DDTree total-node semantics from the public CLI. Tree DFlash now uses the branch-only `--spec-branch-budget` model consistently, with `--spec-draft-top-k` controlling candidates per draft position and flat DFlash forcing top-k back to `1`.
- Added new cache and quantization surface beyond v0.2.0. `q6_0` is available as a KV/cache type, and Tom's `TQ3_1S` / `TQ4_1S` model weight formats are exposed through `llama-quantize` with non-conflicting serialized GGML type IDs; existing `turbo2`, `turbo3`, `turbo4`, `turbo2_tcq`, and `turbo3_tcq` cache types remain available on the newer base.
- Hardened backend behavior used by Bee features. HIP TCQ attention stays on the native vector path, ROCm can probe fused GDN support, D512 Flash Attention selection is available again, unsupported CPU BF16 scale and CPU Flash Attention types are rejected instead of misrunning, CUDA fatal-warning cases were fixed, and the build now requires C++17 for the common base.
- Improved server and API behavior inherited from upstream and Bee integration. The server reports prompt token counts in `/slots`, supports SSE ping intervals and HTTP ETags, exposes real-time reasoning interruption, handles router/model metadata fixes, adds built-in tools such as datetime, and keeps malformed tool-looking text out of final responses.
- Updated packaging and release workflows. Docker image links and labels were aligned for BeeLlama, SYCL package/release images were added and documented, release builds were split and cached more reliably, stale package builds are cancelled, and obsolete CUDA architecture options are rejected at configure time.
- Updated user documentation for the new release state, including DFlash args/defaults, removed aliases, adaptive Draft-Max, multi-GPU DFlash behavior, Turbo/TQ cache and weight formats, Docker images, and the upstream multi-GPU guide.
- Expanded regression coverage around DFlash default normalization, removed arg aliases, DFlash-only no-op behavior, DFlash auto-detection, multi-GPU policy helpers, per-layer capture/tape allocation, device-aware replay, CUDA device restoration, adaptive Draft-Max, reduced-logit failure handling, and tool-call/speculative boundaries.

## v0.2.0

- Added compatibility with upstream DFlash PR drafter GGUFs that use `general.architecture = dflash`, including upstream metadata keys and tensor names.
- Tightened DFlash draft model discovery and converter behavior. Bee now prefers exact sibling DFlash draft directories, supports nested `dflash_config` metadata, scopes Gemma4 tokenizer handling correctly, and logs clearer DFlash metadata warnings and summaries during conversion.
- Hardened recurrent memory, prompt-cache restore, and unified-KV scheduling. Recurrent resize now repairs its metadata after shrink/expand, the server shrinks recurrent state before prompt-cache save/load when it is safe, backup-sequence cleanup is tracked correctly, and non-parent tasks defer unified-KV admission so large pending prompts do not over-commit shared cells.
- Added richer DFlash diagnostics, profiling, and validation. `GGML_DFLASH_PROFILE` now exposes categorized summary/replay/copy/prefill/verify/trace logging, routine decode timing is hidden behind debug logging instead of always printing, the profit controller now logs when it disables speculative depth, drafter/target contract and input validation are stricter, and Bee also exposes targeted debug envs such as `GGML_DFLASH_DEBUG`, `GGML_DFLASH_INPUT_DEBUG`, `GGML_DFLASH_CUDA_DEBUG`, `GGML_DFLASH_FORCE_CPU_CROSS`, `GGML_DFLASH_VERBOSE_CONTRACT`, and `GGML_DFLASH_CRASH_TRACE`.
- Improved DFlash CUDA ordering and split-buffer correctness. Hidden capture, recurrent replay, backup copies, K/V projection-cache updates, and DFlash stream waits now use explicit ordering helpers and safer backend ownership checks instead of broader synchronization or wrong-buffer access.
- Added DFlash drafter K/V projection caching for the cross-attention window. Bee now keeps ring-backed drafter K/V state for recent target hidden-state windows, supports chronological D2D append/interleave on CUDA, excludes the unsafe parts from graph capture when needed, and falls back more safely on placements that cannot use the fast GPU path.
- Reworked DFlash prefill capture and flush handling. Prefill capture now uses per-slot and per-view plans, GPU staging buffers, source-aware CPU/GPU ring validity, suffix-span tracking across internal ubatches, graph-reuse keys for source/destination offsets, callback suppression for irrelevant ubatches, and fail-closed behavior for partial or mismatched captures.
- Hardened target hidden-state capture across Qwen3.5, Qwen3.5-MoE, Gemma4-ISWA, hidden-only contexts, GPU tape, and multi-slot GPU cross data. Capture layer assignment, token-count derivation, callback routing, and GPU multi-slot cross collection now have explicit correctness checks.
- Reduced greedy DFlash verification overhead and made verifier control stricter. Eligible verify batches can use reduced top-k logits without raw-logit readback, Bee keeps seed-row alignment correct, the flat verify horizon is capped, server-side depth control is authoritative, and the reduced path falls back when grammar, sampler, or reasoning state requires full logits.
- Hardened DFlash reasoning, draft, and suffix handling. Reasoning-end forcing now goes through the normal full-logits path when needed, invalid reduced-logits drafts are rejected instead of crashing or looping, empty drafts fall back safely, accepted-prefix full-KV commits respect the drafter window, explicit `--spec-draft-ctx-size` overrides are tracked correctly, Bee keeps the DFlash auto-`-cd 256` default path when no draft ctx is passed, and the drafter stays aligned with the live accepted suffix.
- Improved Gemma 4 support substantially. Bee added Gemma4-ISWA DFlash target plumbing and profiling callbacks, ported the cleaner upstream Gemma4 graph and loader path back onto Bee hooks, restored Bee precision behavior where needed, synced SWA max-position authority and 512-dim FlashAttention selection with upstream, and fixed Gemma multimodal image decode and dynamic resize bounds.
- Extended CUDA kernel coverage and backend hardening. Bee now keeps 512-wide quantized FlashAttention instances for standard and TurboQuant/TCQ KV combinations, syncs upstream Hadamard rotation plumbing, propagates CUDA driver links correctly, and hardens op-table / Gated DeltaNet integration alongside long-context GPU ring stability fixes.
- Reduced peak memory in the perplexity tool and fixed streaming perplexity / KLD cache handling. Streaming perplexity now writes bounded chunks, checks stream errors, avoids retaining unbounded logits for long-context KL runs, and keeps the logits-cache format versioning compatible with the legacy magic.
- Completed the malformed tool-call guard path for non-stream responses. Final OpenAI-compatible responses now quarantine malformed raw tool-looking text the same way streamed tool-parsing responses already did.

## v0.1.2

- Fixed the adaptive `profit` controller's no-spec baseline path. Profit mode now seeds baseline samples before positive-depth warmup, can shut DFlash fully off when the measured baseline wins, and no longer makes speculative decisions from draft-only telemetry.
- Fixed profit-controller reset handling across context-bucket and configuration changes so cleared baseline telemetry cannot leave the controller in a stale active or off state.
- Added low-frequency profit-controller baseline reprobes with `--spec-dm-profit-baseline-interval` / `LLAMA_ARG_SPEC_DM_PROFIT_BASELINE_INTERVAL` so runs can refresh target-only timing as context grows. The default interval is 1024 active speculative cycles; reprobes resume the previous active draft depth and avoid off-probe counter starvation.
- Hardened active-reasoning EOS handling. When an end-of-generation token appears while reasoning output is still active, the sampler now forces the reasoning-end sequence through the normal full-logits path; reduced DFlash verification rejects that case instead of accepting an unsafe reduced candidate set.
- Hardened DFlash on split CUDA / multi-GPU placement. GPU cross-ring setup, hidden capture, CUDA graph capture, K/V projection cache updates, recurrent replay, conv replay, and async tensor get/set paths now check buffer/backend ownership and fall back to safer CPU or owning-buffer paths instead of reading or writing recurrent state through the wrong CUDA backend.
- Added clearer diagnostics and regression coverage for multi-GPU DFlash fallback decisions, CUDA graph buffer visibility, wrong-device async tensor access, active-reasoning reduced-sampling rejection, adaptive DM defaults, and profit-controller baseline behavior.
- Fixed ROCm 7 build: added `cudaPointerAttributes` / `cudaMemoryType` shim aliases to `hip.h`, extended `CUDART_VERSION >= 10000` guards with `|| defined(GGML_USE_HIP)` so the `.type` field path is taken on HIP, and removed the `WIN32` guard around TurboQuant flash-attention instance compilation so Linux ROCm builds include the turbo KV-cache kernels (acerspyro#11).
- Known limitation: the current multi-GPU DFlash path is a correctness fallback, not a performant split-GPU implementation. On split target placement it can be slower than non-speculative decoding because recurrent replay and hidden capture avoid unsafe single-backend GPU fast paths. A performant implementation still needs per-device replay graphs or a scheduler that follows ggml's split-buffer ownership model.

## v0.1.1

- Improved agentic tool-call reliability with lazy grammars. DFlash now remains enabled before a lazy grammar trigger, but stops speculating once grammar-constrained output or reasoning-budget forcing requires normal token-by-token sampling.
- Fixed DFlash accept bookkeeping at grammar and tool-call boundaries. The server now distinguishes accepted draft tokens from bonus-token-shaped results, updates DFlash hidden-state rows with the root plus accepted draft tokens, and uses the same keep count for rollback.
- Added a DFlash suppression guard for raw tool-call markers. When a tool marker appears while lazy grammar is enabled, the server suppresses DFlash for the rest of that response without steering sampler state; fenced code and embedded marker-like strings are excluded from the guard.
- Made partial OpenAI-compatible tool-call streaming safer. The server can stream a stable tool name/id early so clients can show a pending tool call, while withholding partial arguments until the parser sees a complete call.
- Quarantined malformed raw tool-call text in tool-parsing streams. Unfinished or malformed tool-looking text no longer leaks into visible assistant content or hidden reasoning deltas before the parser can classify it.
- Accepted direct tag-style function starts for Qwen-style tool calls. Lazy grammar triggers now include structural function markers such as `<function=`, and the tag parser can parse valid direct function calls without the outer `<tool_call>` wrapper.
- Added regression coverage for Kimi and Qwen tool-call streaming, malformed raw marker quarantine, fenced-code false positives, direct Qwen function calls, lazy grammar triggers, and DFlash speculative boundary plumbing.
- Fixed small build issues found after 0.1.0: the DFlash callback setup now uses an explicit callback type for GCC 15, and tests/server code include the required standard headers for `INT_MAX` and `FLT_MAX`.

## v0.1.0

- DFlash speculative decoding: `--spec-type dflash` drives a DFlash draft GGUF alongside the target model. The target captures hidden states into a per-layer 4096-slot ring buffer, the drafter cross-attends to the most recent `--spec-dflash-cross-ctx` hidden-state tokens and proposes drafts for target verification.
- TurboQuant / TCQ KV-cache compression: Five cache types (`turbo2`, `turbo3`, `turbo4`, `turbo2_tcq`, `turbo3_tcq`) spanning from 4x to 7.5x compression, with higher-bit options being practically lossless in many cases. Set independently with `--cache-type-k` and `--cache-type-v`.
- Adaptive draft-max control: The server adjusts the active draft horizon at runtime instead of using a fixed `--spec-draft-n-max`. The default `profit` controller compares speculative throughput against a no-spec baseline; the `fringe` alternative maps acceptance-rate bands to draft depth. Use `--no-spec-dm-adaptive` for a static horizon.
- Full multimodal support: When `--mmproj` is active, the server keeps flat DFlash available for text generation. The model can be fully offloaded to CPU with no problems to reduce VRAM pressure.
- Reasoning-loop protection: The server detects repeated hidden reasoning output and intervenes. Default mode is `force-close` with `--reasoning-loop-window` and `--reasoning-loop-max-period` tuning available.
- Sampled DFlash verification: `--spec-draft-temp` enables rejection-sampling drafter behavior. Activates when both draft and target temperature exceed zero. Draft log probabilities must be available for rejection sampling to produce correct output.
- DDTree branch verification: optional `--spec-branch-budget` adds branch nodes beyond the main draft path with GPU `parent_ids`, tree masks, and recurrent tree kernels. Disabled automatically when the target model spans more than one GPU. This one is very much work in progress!
- Request-level speculative overrides: Draft-max and branch budget can be overridden per-request through JSON fields without restarting the server.
- CopySpec model-free speculation: `--spec-type copyspec` provides rolling-hash suffix matching over previous tokens without a draft model. Results must be benchmarked per workload.
