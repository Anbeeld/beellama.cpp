# Changelog

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