# AGENTS.md

This file gives code assistants local context for this repository.

## What This Is

BeeLlama.cpp is Anbeeld's fork of llama.cpp. Fork-specific work is concentrated around:

- **DFlash**: cross-attention speculative decoding with DFlash draft GGUFs, target hidden-state capture, CPU/GPU ring buffers, and server verification paths.
- **Adaptive draft-max**: server controllers that adjust the active DFlash draft horizon. The default controller is `profit`; `fringe` is also available.
- **DDTree**: tree speculative verification with GPU `parent_ids` and recurrent tree kernels.
- **CopySpec**: model-free speculation through rolling-hash suffix matching.
- **TurboQuant / TCQ KV cache types**: `turbo2`, `turbo3`, `turbo4`, `turbo2_tcq`, `turbo3_tcq`, and `turbo4_tcq`, plus newly added low-bit standard quantized KV types `q2_0`/`q2_1`/`q3_0`/`q3_1`/`q6_1`.
- **KVarN KV-cache compression** (experimental): structured KV compression via `--cache-type-k`/`--cache-type-v` pseudo names `kvarn2`…`kvarn8` (all nine 2/3/4/5/6/8 K/V bit combinations); target-context only, wired for Qwen3.6 and Gemma 4, with non-KVarN layers falling back to bit-width-matched standard cache types.
- **Reasoning loop guard**: server-side detection and intervention for repeated hidden reasoning output.

Treat the local codebase as the source of truth for implementation behavior.

## Build

Prebuilt Windows binaries (CUDA 12.4/13.1) are on the releases page. Otherwise build from source:

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

Without `GGML_CUDA_FA_HALF_QUANTS` or `GGML_CUDA_FA_ALL_QUANTS`, the CUDA FlashAttention build compiles Bee's recommended default for standard q cache types or KVarN fallback: 62 vec pairs. The default excludes TurboQuant/TCQ and mixed f16/bf16 pairs. For standard q and KVarN fallback types, a pair is compiled only when K is the same or higher precision than V and V is no more than two tier groups below K. Tier groups are fp (`f16`, `bf16`), high q (`q8_0`, `q6_1`, `q6_0`), q5, q4, q3, and q2; within a bit width, `_1` ranks above `_0`. This default is much faster to compile than the 217-pair HALF or 361-pair ALL modes and leaves TurboQuant/TCQ FA pairs out because TurboQuant/TCQ has not shown a benchmark advantage over standard q or KVarN cache types in current fork benchmarks. The pair policy uses same-or-higher precision K because K loses precision faster under quantization and lower-precision K than V is inefficient, but avoids K far above V because large K/V tier gaps are unbalanced. Use `GGML_CUDA_FA_HALF_QUANTS=ON` for TurboQuant/TCQ or high-delta K>=V experiments, or `GGML_CUDA_FA_ALL_QUANTS=ON` for the full 361-pair matrix. HALF treats q4_1/turbo4/turbo4_tcq, q3_1/turbo3/turbo3_tcq, and q2_1/turbo2/turbo2_tcq as pseudo-equal ranks, so those pairs compile both ways while qX_0-to-Turbo pairs remain excluded as K<V. These two flags are mutually exclusive. Add `-DCMAKE_CUDA_ARCHITECTURES=86` for RTX 3090, or `-DCMAKE_CUDA_ARCHITECTURES=89` for RTX 4090, if cross-compiling or building in CI without a GPU.

Key binaries: `build/bin/llama-server`, `build/bin/llama-cli`, `build/bin/llama-bench`, `build/bin/llama-perplexity`.

## Architecture

### Main Directories

- `ggml/` - tensor library, CUDA/Metal/CPU backends, quantization.
- `src/` - llama.cpp core: model loading, context, graph building, sampling, memory.
- `src/models/` - per-architecture graph builders.
- `common/` - shared utilities, argument parsing, speculative decoding orchestration.
- `tools/server/` - HTTP server, slots, chat completions API, speculative server flow.
- `include/llama.h` - public C API.

### Fork-Specific Files

- `src/models/dflash_draft.cpp` - DFlash draft model graph.
- `src/models/qwen35.cpp` - Qwen3.5/Qwen3.6 target graph paths.
- `common/speculative.cpp` - DFlash state, ring buffer, draft APIs, CopySpec, tree construction.
- `common/speculative.h` - speculative APIs and tree data structures.
- `common/suffix-tree.cpp` / `.h` - suffix tree for CopySpec model-free speculation.
- `common/int32-map.h` - hash map for suffix tree (ported from Snowflake ArcticInference, Apache-2.0).
- `tools/server/server-context.cpp` - server speculative scheduling, DFlash verification, rollback, mmproj rules.
- `tools/server/server-adaptive-dm.h` - profit and fringe adaptive draft-max controllers.
- `tools/server/server-loop-guard.cpp` / `.h` - reasoning loop guard.
- `ggml/src/ggml-turbo-quant.c` - CPU reference quantize/dequantize for turbo2/3/4 and TCQ types.
- `ggml/src/ggml-cuda/turbo-quant-cuda.cuh` - CUDA set-rows/dequantize kernels for all TurboQuant types.
- `src/llama-kvarn.cpp` / `.h` - KVarN type descriptors, tile layout, bit-width presets, and runtime validation.
- `src/llama-kv-cache-kvarn.cpp` / `.h` - target-context KVarN KV cache and group-range state serialization.
- `ggml/src/ggml-cuda/kvarn.cu` / `.cuh` - CUDA KVarN store ops and native KVarN support helpers.
- `ggml/src/ggml-vulkan/vulkan-shaders/kvarn_store.comp` - Vulkan KVarN store shader.
- `ggml/src/ggml-cuda/cross-ring-interleave.cu` - GPU cross-ring management and interleave kernel.
- `ggml/src/ggml-cuda/gated_delta_net.cu` - DeltaNet CUDA kernels.
- `ggml/src/ggml-cuda/ssm-conv.cu` - SSM convolution CUDA kernels.

### Key Docs

- `docs/beellama-features.md` — feature matrix and public repo comparison.
- `docs/beellama-args.md` — complete BeeLlama argument reference.
- `docs/quickstart-qwen36-dflash.md` — step-by-step Qwen 3.6 + DFlash single-GPU guide.
- `docs/preset.md` — INI preset file documentation.

### Key Patterns

- **proc_address**: custom CUDA helpers are resolved through `ggml_backend_cuda_reg_get_proc_address`.
- **Eval callback**: target hidden states are captured through `llama_set_eval_callback`.
- **Ring buffer**: DFlash keeps a CPU ring and optional GPU mirror. `GGML_DFLASH_GPU_RING=0` disables the GPU ring.
- **Tree verify**: `parent_ids_gpu` is used for tree kernels and is disabled automatically on multi-GPU target placement.
- **mmproj**: multimodal use keeps flat DFlash available, forces `--spec-branch-budget 0` under DFlash, and sets non-DFlash speculative types to none. DFlash builds on the initial implementation by [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp).

## Test / Benchmark

```bash
# Perplexity
build/bin/llama-perplexity -m model.gguf -f test.txt -c 4096

# Decode speed
build/bin/llama-bench -m model.gguf -p 0 -n 64 -t 1

# Server with DFlash + recommended q cache
build/bin/llama-server -m target.gguf --spec-type dflash \
  --spec-draft-model drafter.gguf \
  --spec-draft-n-max 8 \
  --spec-branch-budget 0 \
  --spec-dflash-cross-ctx 512 \
  --flash-attn on --cache-type-k q5_0 --cache-type-v q4_1 \
  --port 8080
```

Benchmark results are only meaningful with the exact model files, command line, prompt, sampling settings, hardware, and commit ID.

## Multi-GPU Notes

- Tree verify is disabled when `model.n_devices() > 1`.
- The GPU ring path can be disabled with `GGML_DFLASH_GPU_RING=0` for isolation.
- CUDA paths use `cudaStreamPerThread` heavily.

## Git Conventions

- Keep fork-specific changes small and scoped.
- Do not treat old benchmark notes as current evidence without re-running them.
- Do not commit unless the user explicitly asks.
