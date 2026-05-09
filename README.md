# Anbeeld's BeeLlama.cpp

![BeeLlama.cpp logo](beellama.jpg)

BeeLlama.cpp (or just Bee) is Anbeeld's performance-focused llama.cpp fork for squeezing more speed and context out of local GGUF inference. It keeps the familiar llama.cpp tools, server flow, and model compatibility, then adds DFlash speculative decoding, adaptive draft control, TurboQuant/TCQ KV-cache compression, reasoning-loop protection, full multimodal support, and experimental speculation modes.

> Not quite a pegasus, but close enough.

Here's a [plug-and-play Qwen 3.6 27B setup](docs/quickstart-qwen36-dflash.md) with a config to run it in Q5 + 200k of practically lossless KV cache + vision on a single RTX 3090 or 4090.

## Fork Features

- **DFlash speculative decoding**: `--spec-type dflash` drives a DFlash draft GGUF alongside the target model. The target captures hidden states into a per-layer 4096-slot ring buffer, the drafter cross-attends to the most recent `--spec-dflash-cross-ctx` hidden-state tokens and proposes drafts for target verification.
- **TurboQuant / TCQ KV-cache compression**: Five cache types (`turbo2`, `turbo3`, `turbo4`, `turbo2_tcq`, `turbo3_tcq`) spanning from 4x to 7.5x compression, with higher-bit options being practically lossless in many cases. Set independently with `--cache-type-k` and `--cache-type-v`.
- **Adaptive draft-max control**: The server adjusts the active draft horizon at runtime instead of using a fixed `--spec-draft-n-max`. The default `profit` controller compares speculative throughput against a no-spec baseline; the `fringe` alternative maps acceptance-rate bands to draft depth. Use `--no-spec-dm-adaptive` for a static horizon.
- **Full multimodal support**: When `--mmproj` is active, the server keeps flat DFlash available for text generation. The model can be fully offloaded to CPU with no problems to reduce VRAM pressure.
- **Reasoning-loop protection**: The server detects repeated hidden reasoning output and intervenes. Default mode is `force-close` with `--reasoning-loop-window` and `--reasoning-loop-max-period` tuning available.
- **Sampled DFlash verification**: `--spec-draft-temp` enables rejection-sampling drafter behavior. Activates when both draft and target temperature exceed zero. Draft log probabilities must be available for rejection sampling to produce correct output.
- **DDTree branch verification**: optional `--spec-branch-budget` adds branch nodes beyond the main draft path with GPU `parent_ids`, tree masks, and recurrent tree kernels. Disabled automatically when the target model spans more than one GPU. This one is very much work in progress!
- **Request-level speculative overrides**: Draft-max and branch budget can be overridden per-request through JSON fields without restarting the server.
- **CopySpec model-free speculation**: `--spec-type copyspec` provides rolling-hash suffix matching over previous tokens without a draft model. Results must be benchmarked per workload.

For the full feature and public-repo comparison, read [docs/beellama-features.md](docs/beellama-features.md). For the complete argument reference, read [docs/beellama-args.md](docs/beellama-args.md).

TurboQuant (WHT-based scalar quantization) originates from [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). TCQ (Trellis-Coded Quantization) and basic DFlash implementation originate from [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) (paper: [Closing the Gap: Trellis-Coded Quantization for KV Cache at 2-3 Bits](https://huggingface.co/datasets/spiritbuun/turboquant-tcq-kv-cache)).

## DFlash Speedup

Here's your typical Python best-case ceiling benchmark with [Qwen 3.6 27B](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) using [Q4_K_M drafter](https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF) on a single RTX 3090 24GB. Like any other speculative prediction, DFlash is strongest on structured, repetitive generation: code, tests, boilerplate, JSON-like formats, and other low-entropy continuations.

| Workload | Model | Output | Baseline | Bee DFlash | Peak speedup | Acceptance | Draft coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short Python linked-list bench | Q4_K_M | ~1.2K tok | 39.2 tok/s | **130.1 tok/s** | **3.32x** | 49.1% | 84.5% |
| Short Python linked-list bench | Q5_K_S | ~1.2K tok | 36.5 tok/s | **135.8 tok/s** | **3.72x** | 47.8% | 85.8% |
| Longer Python cache bench | Q4_K_M | ~3.6K tok | 37.5 tok/s | **91.5 tok/s** | **2.44x** | 40.5% | 78.8% |
| Longer Python cache bench | Q5_K_S | ~3.6K tok | 35.9 tok/s | **83.7 tok/s** | **2.33x** | 36.7% | 76.2% |

*Acceptance = accepted draft tokens / proposed draft tokens. Draft coverage = accepted draft tokens / final generated tokens.*

This is not a claim about all workloads. DFlash can go much faster on highly predictable code generation than on normal chat. Open-ended prose is much less predictable, so gains are smaller.

On the bright side, adaptive draft-max will track how much DFlash is helping on the current task and will adjust its intensity accordingly, or even turn it off if you would dip below the baseline otherwise.

## TurboQuant / TCQ cache

| Type | bpv | Compression | Quality vs f16/q8_0 | Practical verdict |
| --- | ---: | ---: | --- | --- |
| turbo4 | 4.125 | 3.88x | Best scalar TurboQuant quality tier. Available tests show minimal measurable degradation vs f16/q8_0. | Best safe scalar compression target, especially for V cache. |
| turbo3_tcq | 3.25 | 4.92x | Strongest 3-bit quality. The TCQ docs report 10–44% KL reduction over scalar 2–3 bit quantization and lower PPL than FP16 in one Qwen3.5-27B result: 5.802 vs 5.805. | Best high-compression quality-aware option. |
| turbo3 | 3.125 | 5.12x | Strong compression with measurable quality cost. Available tests put it below turbo4 but still usable on tolerant models/configs. | Aggressive scalar compression. Validate per model, especially if used on K. |
| turbo2_tcq | 2.25 | 7.11x | Best 2-bit option. Per the TCQ docs it significantly improves 2-bit quantization and closes much of the gap with 3-bit scalar methods. | Extreme compression with better quality story than scalar 2-bit. |
| turbo2 | 2.125 | 7.53x | Extreme scalar compression. Highest quality risk among scalar TurboQuant types. | Emergency context/VRAM mode. Prefer as a last resort V-only. |

TurboQuant is not *truly* lossless at any point, but on the higher end it might very well be *practically* lossless for most tasks. Especially when one's *practicality* is heavily influenced by VRAM constraints and how to get the most juice out of it.

## Installation

### Quickstart: DFlash on a Single GPU

For a step-by-step walkthrough with Qwen 3.6 on a 24 GB NVIDIA card (RTX 3090/4090, etc.), see [docs/quickstart-qwen36-dflash.md](docs/quickstart-qwen36-dflash.md). It covers model download, prebuilt binaries, and a tuned launch command.

### Prebuilt (Windows)

Download the release archive for your CUDA version (12.4 or 13.1) from the [releases page](https://github.com/Anbeeld/beellama.cpp/releases). Extract it. The server binary is `llama-server.exe`. Don't forget to download a separate archive with CUDA libraries and place it in the same folder!

Building from source with `-DGGML_NATIVE=ON` *may* result in a *tiny* bit better performance, so it might still be a good idea to do that if/when you decide to use this fork long-term.

### CUDA Build

```bash
# Linux (GCC + CUDA)
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Windows (MSVC + CUDA)
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON ^
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON ^
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

# macOS (Metal)
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`GGML_CUDA_FA_ALL_QUANTS=ON` is required for TurboQuant and TCQ cache types. Add `-DCMAKE_CUDA_ARCHITECTURES=86` for RTX 3090, or `-DCMAKE_CUDA_ARCHITECTURES=89` for RTX 4090, if cross-compiling or building in CI without a GPU.

Key binaries:

- `build/bin/llama-server`
- `build/bin/llama-cli`
- `build/bin/llama-bench`
- `build/bin/llama-perplexity`

### Other Backends

BeeLlama inherits llama.cpp backend support, including Metal, HIP, Vulkan, SYCL, BLAS, CANN, MUSA, OpenVINO, OpenCL, and RPC. Use the upstream-style build docs in [docs/build.md](docs/build.md) and backend-specific pages under [docs/backend](docs/backend).

## Common Commands

### Local CLI

```sh
llama-cli -m model.gguf
llama-cli -m model.gguf -cnv --chat-template chatml
llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p "Request: schedule a call at 8pm; Command:"
```

### OpenAI-Compatible Server

```sh
llama-server -m model.gguf --port 8080
llama-server -m model.gguf -c 16384 -np 4
llama-server -m model.gguf -md draft.gguf
```

### DFlash And TurboQuant Together

```sh
llama-server -m target.gguf --spec-type dflash \
  --spec-draft-model drafter.gguf \
  --spec-draft-ngl all \
  --flash-attn on --cache-type-k turbo4 --cache-type-v turbo3_tcq
```

## Documentation

- [BeeLlama features and public repo diff](docs/beellama-features.md)
- [BeeLlama args reference](docs/beellama-args.md)
- [Build docs](docs/build.md)
- [Server docs](tools/server/README.md)
- [Docker docs](docs/docker.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)

## Contributing

Keep PRs small and scoped. Run the narrowest relevant tests or benchmarks before opening a PR, and include the exact commands. For fork-specific speculative decoding, DFlash, TurboQuant, or reasoning-loop changes, update the corresponding docs when behavior or args change.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for inherited llama.cpp contribution conventions and this fork's AI usage policy.

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - single-header HTTP server used by `llama-server` - MIT
- [stb-image](https://github.com/nothings/stb) - single-header image decoder used by multimodal code - public domain
- [nlohmann/json](https://github.com/nlohmann/json) - single-header JSON library - MIT
- [miniaudio.h](https://github.com/mackron/miniaudio) - single-header audio decoder - public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - process launching helper - public domain
- [Snowflake ArcticInference](https://github.com/snowflakedb/ArcticInference) - suffix tree and int32 map used in speculative decoding (`common/suffix-tree.*`, `common/int32-map.h`) - Apache-2.0
- [Intel OpenVINO](https://github.com/openvinotoolkit/openvino) - frontend header used in OpenVINO backend (`ggml/src/ggml-openvino/openvino/frontend.h`) - Apache-2.0
- Intel SYCL/oneAPI - SYCL backend (`ggml/src/ggml-sycl/`) - Apache-2.0 WITH LLVM-exception

See the `licenses/` directory for full license texts.
