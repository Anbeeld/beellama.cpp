# Changelog

## v0.1.2

- Fixed the adaptive `profit` controller's no-spec baseline path. Profit mode now seeds baseline samples before positive-depth warmup, can shut DFlash fully off when the measured baseline wins, and no longer makes speculative decisions from draft-only telemetry.
- Fixed a profit-controller bucket-transition deadlock where telemetry reset could clear the no-spec baseline while preserving a positive active draft depth, causing all later cycles to run as unrecorded single-token baseline and leaving DFlash permanently disabled.
- Added periodic profit-controller baseline reprobes with `--spec-dm-profit-baseline-interval` / `LLAMA_ARG_SPEC_DM_PROFIT_BASELINE_INTERVAL` so long-context runs can refresh target-only timing as context grows. The default interval is 512 active speculative cycles, and periodic reprobes start only in longer context buckets; bucket transitions still seed a fresh baseline. Off-state probes now restart with the configured probe depth instead of jumping straight to full draft depth.
- Stabilized profit depth selection on the production ladder (`0`-`8`, `10`, `12`, `14`, `16`, and the configured max) while preserving the previous active depth across baseline reprobes and avoiding off-probe counter starvation from repeated baseline cycles.
- Hardened active-reasoning EOS handling. When an end-of-generation token appears while reasoning output is still active, the sampler now forces the reasoning-end sequence through the normal full-logits path; reduced DFlash verification rejects that case instead of accepting an unsafe reduced candidate set.
- Hardened DFlash on split CUDA / multi-GPU placement. GPU cross-ring setup, hidden capture, CUDA graph capture, K/V projection cache updates, recurrent replay, conv replay, and async tensor get/set paths now check buffer/backend ownership and fall back to safer CPU or owning-buffer paths instead of reading or writing recurrent state through the wrong CUDA backend.
- Added clearer diagnostics and regression coverage for multi-GPU DFlash fallback decisions, CUDA graph buffer visibility, wrong-device async tensor access, active-reasoning reduced-sampling rejection, adaptive DM defaults, and profit-controller baseline behavior.
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
