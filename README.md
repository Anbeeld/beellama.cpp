# Anbeeld's BeeLlama.cpp

![BeeLlama.cpp logo](beellama.jpg)

BeeLlama.cpp (or just Bee) is a performance-focused llama.cpp fork for squeezing more speed and context out of local GGUF inference. It adds variance-normalized KV-cache quantization (KVarN), KV cache precision tail for recent tokens, low-bit cache types, adaptive draft control for speculative decoding, reasoning-loop protection, and more.

> Not quite a pegasus, but close enough.

[![Support my work!](https://anbeeld.com/images/support.jpg)](https://anbeeld.com/support)

## Fork Features

- **Variance-normalized KV-cache quantization (KVarN)**: provides higher precision at similar memory costs. Independent K and V bit widths at `kvarn2`, `kvarn3`, `kvarn4`, `kvarn5`, `kvarn6`, and `kvarn8`, set with `--cache-type-k` and `--cache-type-v`.
- **KV cache precision tail**: keep most of the KV cache quantized while storing recent tokens in F16/BF16, enabled with `--kv-tail-tokens`. A single global softmax merges the quantized body and the precision tail under FlashAttention, without materializing the whole cache.
- **Standard low-bit KV cache types**: `q2_0`, `q2_1`, `q3_0`, `q3_1`, `q6_0`, and `q6_1`, usable for either target or draft caches alongside the upstream `q4`/`q5`/`q8` types.
- **Adaptive draft-max for DFlash**: adjusts the active DFlash draft horizon at runtime instead of using a fixed `--spec-draft-n-max`, comparing speculative throughput against a no-spec baseline.
- **Reasoning-loop protection**: the server detects repeated hidden reasoning output and intervenes.

For the full feature and public-repo comparison, read [docs/beellama-features.md](docs/beellama-features.md). For the complete argument reference, read [docs/beellama-args.md](docs/beellama-args.md).

## KV Cache Quantization

K and V cache types are set independently with `--cache-type-k` and `--cache-type-v`. The research is covered in articles: [KVarN KV Cache: Implementation and Benchmarks](https://anbeeld.com/articles/kvarn-kv-cache-implementation-and-benchmarks), [KV Cache Precision Tail: Implementation and Benchmarks](https://anbeeld.com/articles/kv-cache-precision-tail-implementation-and-benchmarks), and [KV Cache Quantization Benchmarks: KVarN, Precision Tail](https://anbeeld.com/articles/kv-cache-quantization-benchmarks-kvarn-precision-tail).

### KV Cache Recommenation Ladder

The measurements come from Qwen 3.6 27B Q5_K_S at 64K context on Wikitext-2 raw with `-b 2048 -ub 512` on an RTX 3090. Median KLD is the primary quality metric; lower is better.

| K / V | Tail | Size | Size vs bf16 | Median KLD | 99.9% KLD | What it is for |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bf16 / bf16` | 0 | 4096 MiB | 100.0% | 0 | 0.000050 | Reference |
| `q8_0 / q8_0` | 1024 | 2272 MiB | 55.5% | 0.000897 | 0.087699 | Standard fidelity with a precision tail |
| `kvarn8 / kvarn8` | 1024 | 2256 MiB | 55.1% | 0.000871 | 0.087639 | Best measured quality below BF16 |
| `q8_0 / q8_0` | 0 | 2176 MiB | 53.1% | 0.000909 | 0.093029 | Standard fidelity |
| `q8_0 / q6_0` | 1024 | 2016 MiB | 49.2% | 0.000894 | 0.091098 | q8_0 quality within noise, 256 MiB less |
| `kvarn6 / kvarn6` | 1024 | 1744 MiB | 42.6% | 0.000879 | 0.084629 | High-end value pick |
| `kvarn6 / kvarn5` | 1024 | 1616 MiB | 39.5% | 0.000886 | 0.092778 | Much cheaper, almost as good |
| `kvarn5 / kvarn5` | 1024 | 1488 MiB | 36.3% | 0.000897 | 0.087666 | Highest value in the mid-range |
| `q5_0 / q4_1` | 1024 | 1440 MiB | 35.2% | 0.000966 | 0.089128 | Standard option when VRAM-constrained |
| `kvarn5 / kvarn4` | 1024 | 1360 MiB | 33.2% | 0.000936 | 0.089469 | Balanced default |
| `q4_0 / q4_0` | 1024 | 1248 MiB | 30.5% | 0.001057 | 0.104486 | Compact standard cache |
| `kvarn4 / kvarn4` | 1024 | 1232 MiB | 30.1% | 0.000994 | 0.090391 | Cleaner than q4_0 for less memory |
| `kvarn4 / kvarn3` | 1024 | 1104 MiB | 27.0% | 0.001112 | 0.113968 | Smallest recommended tier |
| `kvarn3 / kvarn3` | 1024 | 976 MiB | 23.8% | 0.001316 | 0.139558 | When the context must fit |
| `kvarn3 / kvarn2` | 1024 | 848 MiB | 20.7% | 0.002424 | 0.238780 | Emergency compression |
| `kvarn2 / kvarn2` | 1024 | 720 MiB | 17.6% | 0.003811 | 0.450496 | Last resort |

A 1024-token tail is a useful starting point for low-bit Qwen caches. Prefer more body precision when old and recent tokens matter equally; consider 2048 only when the newest two thousand tokens are genuinely the privileged working set. Standard caches generally have slightly faster prefill.

Gemma 4 needs a separate decision. Its 1024-token sliding window makes a 1024 tail exact across most layers, sharply changing memory and throughput. Standard `q8_0 / q8_0` without a tail is the safer Gemma default when throughput and older-context coverage matter, and the benchmark results recommend avoiding quantized KV cache when Gemma quality is non-negotiable.

### Standard-Only KV Cache Ladder

Use this generic fallback ladder when KVarN or precision tails are unavailable. It is based on the same Qwen 3.6 27B benchmark, with every standard cache row using tail 0.

| K / V | Size | Size vs bf16 | Median KLD | 99.9% KLD | What it is for |
| --- | ---: | ---: | ---: | ---: | --- |
| `bf16 / bf16` | 4096 MiB | 100.0% | 0 | 0.000050 | Reference |
| `q8_0 / q8_0` | 2176 MiB | 53.1% | 0.000909 | 0.093029 | Compression with minimal losses |
| `q8_0 / q6_0` | 1920 MiB | 46.9% | 0.000937 | 0.093575 | 256 MiB below q8_0 |
| `q6_0 / q6_0` | 1664 MiB | 40.6% | 0.000960 | 0.091134 | High-end value pick |
| `q6_0 / q5_0` | 1536 MiB | 37.5% | 0.001054 | 0.094670 | Balanced default |
| `q5_0 / q5_0` | 1408 MiB | 34.4% | 0.001154 | 0.097070 | Last tier before the cliff |
| `q5_0 / q4_1` | 1344 MiB | 32.8% | 0.001433 | 0.122096 | Default when VRAM-constrained |
| `q5_0 / q4_0` | 1280 MiB | 31.3% | 0.001516 | 0.121068 | 64 MiB cheaper, worse median |
| `q4_0 / q4_0` | 1152 MiB | 28.1% | 0.001846 | 0.154408 | Smallest recommended tier |
| `q4_0 / q3_0` | 1024 MiB | 25.0% | 0.003313 | 0.218912 | When the context must fit |
| `q3_0 / q3_0` | 896 MiB | 21.9% | 0.004696 | 0.304186 | Emergency compression |
| `q2_0 / q2_0` | 640 MiB | 15.6% | 0.019374 | 1.198902 | Last resort |

### KV Cache Type Reference

<details>
<summary><strong>All KV cache types available in BeeLlama</strong></summary>

| Type | Origin | bpv | Size vs bf16 |
| --- | --- | ---: | ---: |
| `q8_0` | upstream | 8.5 | 53.1% |
| `kvarn8` | Huawei / fork | 8.375 | 52.3% |
| `q6_1` | fork | 7.0 | 43.8% |
| `q6_0` | fork | 6.5 | 40.6% |
| `kvarn6` | Huawei / fork | 6.375 | 39.8% |
| `q5_1` | upstream | 6.0 | 37.5% |
| `q5_0` | upstream | 5.5 | 34.4% |
| `kvarn5` | Huawei / fork | 5.375 | 33.6% |
| `q4_1` | upstream | 5.0 | 31.3% |
| `q4_0` | upstream | 4.5 | 28.1% |
| `iq4_nl` | upstream | 4.5 | 28.1% |
| `kvarn4` | Huawei / fork | 4.375 | 27.3% |
| `q3_1` | fork | 4.0 | 25.0% |
| `q3_0` | fork | 3.5 | 21.9% |
| `kvarn3` | Huawei / fork | 3.375 | 21.1% |
| `q2_1` | fork | 3.0 | 18.8% |
| `q2_0` | fork | 2.5 | 15.6% |
| `kvarn2` | Huawei / fork | 2.375 | 14.8% |

Standard ratios come directly from each block format. KVarN ratios describe the compressed record body, including scale and zero-point metadata; the permanent exact sink, exact suffix, staging, and alignment add overhead.

</details>

## Installation

### Prebuilt

Current release binaries are on the [releases page](https://github.com/Anbeeld/beellama.cpp/releases).

| Platform | Backend | Asset suffix |
| --- | --- | --- |
| macOS arm64 | Metal | `bin-macos-arm64.tar.gz` |
| Ubuntu x64 | CPU | `bin-ubuntu-x64.tar.gz` |
| Ubuntu arm64 | CPU | `bin-ubuntu-arm64.tar.gz` |
| Ubuntu x64 | CUDA 12.4 | `bin-ubuntu-cuda-12.4-x64.tar.gz` |
| Ubuntu x64 | CUDA 13.1 | `bin-ubuntu-cuda-13.1-x64.tar.gz` |
| Ubuntu x64 | Vulkan | `bin-ubuntu-vulkan-x64.tar.gz` |
| Ubuntu x64 | ROCm 7.2 | `bin-ubuntu-rocm-7.2-x64.tar.gz` |
| Ubuntu x64 | SYCL | `bin-ubuntu-sycl-x64.tar.gz` |
| Windows x64 | CPU | `bin-win-cpu-x64.zip` |
| Windows x64 | Vulkan | `bin-win-vulkan-x64.zip` |
| Windows x64 | SYCL | `bin-win-sycl-x64.zip` |
| Windows x64 | CUDA 12.4 | `bin-win-cuda-12.4-x64.zip` |
| Windows x64 | CUDA 13.1 | `bin-win-cuda-13.1-x64.zip` |
| Windows x64 | HIP/Radeon | `bin-win-hip-radeon-x64.zip` |

Windows CUDA archives contain a `ggml-cuda.dll` backend; download the matching `beellama-<version>-cudart-win-cuda-*-x64.zip` runtime archive and extract it into the same folder. Windows SYCL and HIP archives ship as standalone packages with all required runtime DLLs bundled.

Docker images are published to `ghcr.io/anbeeld/beellama.cpp`:

| Image | Acceleration | Platforms |
| --- | --- | --- |
| `server`, `server-cpu` | CPU | linux/amd64, linux/arm64 |
| `server-cuda`, `server-cuda12` | CUDA 12.4 | linux/amd64 |
| `server-cuda13` | CUDA 13.1 | linux/amd64 |
| `server-rocm` | ROCm | linux/amd64 |
| `server-vulkan` | Vulkan | linux/amd64 |
| `server-sycl` | SYCL | linux/amd64 |

Building from source with `-DGGML_NATIVE=ON` *may* result in a *tiny* bit better performance, so it might still be a good idea to do that if/when you decide to use this fork long-term.

### CUDA Build

```bash
# Linux (GCC + CUDA)
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Windows (MSVC + CUDA)
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON ^
  -DGGML_CUDA_FA=ON ^
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

# macOS (Metal)
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The default CUDA FlashAttention build covers 50 standard cache pairs and 15 KVarN fast-decode pairs, including the homogeneous F16 and BF16 pairs needed by precision tails. Add `-DGGML_CUDA_FA_ALL_QUANTS=ON` to compile all 169 standard and 36 KVarN pairs, or `-DGGML_CUDA_KVARN=OFF` to build without KVarN kernels.

### Other Backends

Bee inherits llama.cpp backend support, including Metal, HIP, Vulkan, SYCL, BLAS, CANN, MUSA, OpenVINO, OpenCL, and RPC. Use the upstream-style build docs in [docs/build.md](docs/build.md) and backend-specific pages under [docs/backend](docs/backend).

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

### DFlash Speculative Decoding

```sh
llama-server -m target.gguf --spec-type draft-dflash \
  --spec-draft-model drafter.gguf \
  --spec-draft-ngl all \
  --spec-dm-controller profit \
  --flash-attn on --cache-type-k q5_0 --cache-type-v q4_1
```

Keep the draft context on a standard cache type; KVarN is target-cache only.

### KVarN Target Cache

```sh
# Balanced general starting point from the benchmark ladder
llama-server -m model.gguf --flash-attn on \
  --cache-type-k kvarn5 --cache-type-v kvarn4 \
  --kv-tail-tokens 1024
```

### Router Mode With Presets

```sh
llama-server --models-dir /path/to/models
llama-server --models-preset presets.ini
```

## Documentation

- [BeeLlama features and public repo diff](docs/beellama-features.md)
- [BeeLlama args reference](docs/beellama-args.md)
- [Build docs](docs/build.md)
- [Server docs](tools/server/README.md)
- [Docker docs](docs/docker.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)

## Contributing

Keep PRs small and scoped. Run the narrowest relevant tests or benchmarks before opening a PR, and include the exact commands. For fork-specific changes, update the corresponding docs when behavior or args change.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for inherited llama.cpp contribution conventions and this fork's AI usage policy.

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - HTTP client/server library used by `llama-server` - MIT
- [stb-image](https://github.com/nothings/stb) - single-header image decoder used by multimodal code - public domain
- [nlohmann/json](https://github.com/nlohmann/json) - single-header JSON library - MIT
- [miniaudio.h](https://github.com/mackron/miniaudio) - single-header audio decoder - public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - process launching helper - public domain
- [Intel OpenVINO](https://github.com/openvinotoolkit/openvino) - frontend header used in OpenVINO backend (`ggml/src/ggml-openvino/openvino/frontend.h`) - Apache-2.0
- Intel SYCL/oneAPI - SYCL backend (`ggml/src/ggml-sycl/`) - Apache-2.0 WITH LLVM-exception

See the `licenses/` directory for full license texts.

[![Support my work!](https://anbeeld.com/images/support.jpg)](https://anbeeld.com/support)
