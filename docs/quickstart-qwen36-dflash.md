# Quickstart: Qwen 3.6 with DFlash on a Single GPU

Run Qwen3.6-27B with DFlash speculative decoding in [BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp) on one NVIDIA GPU. This guide covers model download, binary setup, and a launch command tuned for a 24 GB VRAM card (RTX 3090, RTX 4090, A5000, etc.).

DFlash is a speculative decoding mode where a small draft model reads recent hidden states from the target model and predicts multiple tokens ahead. The target model then verifies those predictions in a single forward pass. When the draft model is good, this produces multiple accepted tokens per target evaluation.

## Prerequisites

**Hardware.** An NVIDIA GPU with at least 24 GB VRAM. AMD GPUs on ROCm and Apple Silicon on macOS also work with limitations (see [Platform notes](#platform-notes)).

**Software.** One of:

- Windows: a prebuilt binary (CUDA 12.4 or 13.1), or build from source with CUDA Toolkit and CMake.
- Linux: build from source with CUDA Toolkit and CMake.
- macOS: build from source with Xcode command-line tools and CMake. Metal acceleration is available; DFlash runs on the CPU ring path only.

## Get the binary

### Prebuilt (Windows)

Download the release archive for your CUDA version (12.4 or 13.1) from the [releases page](https://github.com/Anbeeld/beellama.cpp/releases). Extract it. The server binary is `llama-server.exe`. Don't forget to download a separate archive with CUDA libraries and place it in the same folder!

Building from source with `-DGGML_NATIVE=ON` *may* result in a *tiny* bit better performance, so it might still be a good idea to do that if/when you decide to use this fork long-term.

### Build from source

**Windows (MSVC + CUDA).** Run in PowerShell or Command Prompt (CMake finds MSVC automatically — no Developer Command Prompt needed):

```powershell
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON ^
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON ^
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

**Linux (GCC + CUDA).**

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`GGML_CUDA_FA_ALL_QUANTS=ON` is required for TurboQuant and TCQ cache types. Add `-DCMAKE_CUDA_ARCHITECTURES=86` for RTX 3090, or `-DCMAKE_CUDA_ARCHITECTURES=89` for RTX 4090, if cross-compiling or building in CI without a GPU.

**macOS (Metal).**

```bash
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The build produces `llama-server` (or `llama-server.exe` on Windows) inside `build/bin/` or `build/bin/Release/`.

## Get the models

You need three files: a target model, a DFlash draft model, and a multimodal projector (mmproj). The mmproj is optional; skip it and remove the `--mmproj` / `--no-mmproj-offload` flags if you do not need vision.

Two practical combos have emerged from benchmarking. Pick one based on what you care about most.

### Model configurations

| Combo | Target | Drafter | K cache | V cache | Best for |
|-------|--------|---------|---------|---------|----------|
| **Precision** | Q5_K_S | Q4_K_M or Q5_K_M | turbo4 | turbo3_tcq | Coding and precision-sensitive tasks where quality matters more than speed |
| **Speed / VRAM** | Q4_K_M or IQ4_XS | Q4_K_M or IQ4_XS | turbo3_tcq | turbo3_tcq (or turbo2_tcq for extreme VRAM saving) | Throughput, VRAM-constrained, or when prompt generation speed matters more than precision |

The precision combo gives the best model quality. The drafter is Q4_K_M or Q5_K_M: at higher draft depths stronger draft quality might help acceptance rates. The speed combo uses a lighter Q4_K_M target which generates tokens faster and free VRAM for higher `--ctx-size` or `-ub`, at the cost of some precision.

For extreme VRAM savings look for IQ4_XS model, specifically the one linked below in the download sections, as it's probably the smallest Q4 model out there with its size rivaling higher end of Q3 models. IQ4_XS drafter might be worth trying and benchmarking as well.

As for heavier drafters like Q8, from my testing it was always a net negative for tok/s, probably due to combination of the model being much larger, and quantization not mattering as much for drafting as all the model needs to do is guess 8-16 tokens at a time. But if someone will provide benchmarks that prove otherwise or point out it's due to an incorrect implementation in the code, I'll be happy to change the stance.

### Where to download

**Target model** — from [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF): Q5_K_S and Q4_K_M.

For the IQ4_XS target used in extreme VRAM situations: [cHunter789/Qwen3.6-27B-i1-IQ4_XS-GGUF](https://huggingface.co/cHunter789/Qwen3.6-27B-i1-IQ4_XS-GGUF).

**DFlash draft model:**
- [spiritbuun/Qwen3.6-27B-DFlash-GGUF](https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF) — Q8 and Q4_K_M
- [Ardenzard/Qwen3.6-27B-DFlash-GGUF](https://huggingface.co/Ardenzard/Qwen3.6-27B-DFlash-GGUF) — Q5_K_M, IQ4_XS, and other quants
- [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) — original full-precision weights

The draft model shares the target's token embedding and LM head at runtime, so the GGUF file only contains the DFlash-specific weights.

### Multimodal projector (optional)

Download from [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) (look for `mmproj-BF16.gguf`). The launch command below uses `--no-mmproj-offload`, which runs the mmproj on the CPU instead of the GPU. On a dedicated GPU this frees VRAM for the model and cache at the cost of some mmproj latency. On systems with unified memory (macOS/Metal) the flag does not save VRAM — the GPU and CPU share the same pool — so drop it and let the mmproj use the GPU via Metal.

## Launch

Replace the model paths with your own.

**If your machine has unified memory (Mac, Ryzen APU, etc.), do not use `--no-mmproj-offload`** — there is no separate VRAM pool to save.

### Precision combo (coding, quality-sensitive)

Q5_K_S target, Q4_K_M drafter, turbo4 K cache, turbo3_tcq V cache. Vision is enabled but is offloaded to CPU as thus consumes no VRAM, which seems to work with no issues whatsoever thanks to fixes implemented in the fork.

Note that this config has low-ish 120k context (`--ctx-size 122800`) and `-ub 256` (important for prefill speed) by default. These values are considered safe and should work even with Windows reserving some VRAM and a couple of Electron-based VRAM hogs being open at the same time.

I can personally confirm that on my RTX 3090 24GB after fresh Windows 11 restart and with minimum background apps, I was able to launch the precision combo with **200k context** and chat with the model about some stuff through **WebUI in Chromium**, with 2 monitors connected to dGPU and HWiNFO 64 open on the second monitor for checking stats.

This of course consumed 99.7% of my VRAM, but after killing the server the system was still using **700MB VRAM**. Which basically means that if you eliminate VRAM pressure from other sources, you should be able to comfortably use the `Qwen 3/6 27B Q5 + Q4/Q5 DFlash + K TQ 4 + V TQ 3_tcq` combo with 150-200k context on a single 3090 or 4090, which to me sounds like a great deal.

So if you are not on Windows, or your monitors are connected to iGPU freeing up gGPU VRAM, or your monitors are turned off and all apps are closed, etc. etc. you can freely experiment with higher values up to like 200k content and 512 or 1024 `-ub`.

Also there's `--spec-dflash-cross-ctx 1024` which is how much context the drafter sees at the time. Higher values eat up more VRAM, but in my testing they didn't increase tok/s as this slowed down the drafter itself quite a bit. Default value is `512`, but for longer context `1024` seemed better from testing, still might be something worth tinkering with.

**Windows (PowerShell):**

```powershell
llama-server.exe `
  -m "path\to\Qwen3.6-27B-Q5_K_S.gguf" `
  --mmproj "path\to\mmproj-BF16.gguf" `
  --no-mmproj-offload `
  --spec-draft-model "path\to\Qwen3.6-27B-DFlash-Q4_K_M.gguf" `
  --spec-type dflash `
  --spec-dflash-cross-ctx 1024 `
  --port 8082 `
  -np 1 `
  --kv-unified `
  -ngl all `
  --spec-draft-ngl all `
  -b 2048 -ub 256 `
  --ctx-size 122800 `
  --cache-type-k turbo4 --cache-type-v turbo3_tcq `
  --flash-attn on `
  --cache-ram 0 `
  --jinja `
  --no-mmap --mlock `
  --no-host --metrics `
  --log-timestamps --log-prefix --log-colors off `
  --reasoning on `
  --chat-template-kwargs '{\"preserve_thinking\":true}' `
  --temp 0.6 --top-k 20 --min-p 0.0
```

**Linux / macOS:**

```bash
llama-server \
  -m "path/to/Qwen3.6-27B-Q5_K_S.gguf" \
  --mmproj "path/to/mmproj-BF16.gguf" \
  --no-mmproj-offload \
  --spec-draft-model "path/to/Qwen3.6-27B-DFlash-Q4_K_M.gguf" \
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

`--reasoning on` gives the drafter richer context via thinking tokens, which improves prediction quality. `--chat-template-kwargs '{"preserve_thinking":true}'` keeps those tokens across turns — recommended when reasoning is on. Turning reasoning off entirely is valid if your task does not benefit from it.

### Speed / VRAM combo (throughput, lighter)

Q4_K_M or IQ4_XS target, maybe IQ4_XS drafter (debatable, default is probably still Q4_K_M), turbo3_tcq K and V cache. Frees a lot of VRAM for higher `--ctx-size` or `-ub`, or just allows to fit in if have less than 24GB VRAM. Prompt generation should be somewhat faster, at the cost of precision.

Change these flags from the precision command above, with drafter change being optional:

```
-m "path\to\Qwen3.6-27B-Q4_K_M.gguf"
--spec-draft-model "path\to\Qwen3.6-27B-DFlash-IQ4_XS.gguf"
--cache-type-k turbo3_tcq
```

For even more VRAM headroom (and even less precision), use `turbo2_tcq` for the V cache: `--cache-type-v turbo2_tcq`.

### Optionally: use `--spec-draft-hf` instead of a local file

Replace `--spec-draft-model "path/to/draft.gguf"` with:

```
--spec-draft-hf spiritbuun/Qwen3.6-27B-DFlash-GGUF:Q4_K_M
```

This downloads the draft model from HuggingFace on first run and caches it locally.

## What the flags do

For full command-line tuning, including upstream llama.cpp args, DFlash args, TurboQuant/TCQ cache choices, context checkpoints, prompt-cache RAM, and chat/reasoning controls, read [beellama-args.md](beellama-args.md).

### DFlash flags

| Flag | Value | What it controls |
|------|-------|-----------------|
| `--spec-type` | `dflash` | Enables DFlash speculative decoding |
| `--spec-draft-model` | path or HF repo | DFlash draft model to load |
| `--spec-draft-ngl` | `all` | Offload all draft layers to GPU |
| `--spec-dflash-cross-ctx` | `1024` | How many tokens of target hidden state the drafter sees. Higher gives more context to cross-attention, lower saves VRAM |

### Context and cache flags

| Flag | Value | What it controls |
|------|-------|-----------------|
| `--ctx-size` | `122800` | Total KV context allocation. Lower to save VRAM |
| `-b` | `2048` | Logical batch size for prompt evaluation |
| `-ub` | `256` | Physical microbatch size |
| `--kv-unified` | — | Single KV buffer shared across server slots |
| `-ngl` | `all` | Offload all target model layers to GPU |
| `--cache-type-k` | `turbo4` | TurboQuant K cache (4.125 bits/value, 3.88× FP16 compression). Slightly slower than `turbo3_tcq` but more precise |
| `--cache-type-v` | `turbo3_tcq` | TurboQuant+TCQ V cache (3.25 bits/value, 4.92× FP16 compression) |
| `--flash-attn` | `on` | Use Flash Attention kernels (auto-enabled when TurboQuant cache types are selected) |
| `--cache-ram` | `0` | Disable prompt cache in system RAM (default is 8192 MiB) |
| `--jinja` | — | Enable Jinja template engine for chat formatting |

### Model and sampling flags

| Flag | Value | What it controls |
|------|-------|-----------------|
| `--mmproj` | path | Multimodal projector for vision input |
| `--no-mmproj-offload` | — | Run mmproj on CPU, freeing GPU VRAM at a latency cost (skip on macOS — unified memory has no separate VRAM pool) |
| `--jinja` | — | Use Jinja template engine for chat formatting |
| `--reasoning` | `on` | Enable reasoning output handling — thinking tokens give the drafter richer context for better predictions. Turn off if the task does not benefit from reasoning |
| `--chat-template-kwargs` | `{"preserve_thinking":true}` | Preserve thinking tokens across turns for better output quality and stronger drafter predictions. Recommended when `--reasoning on` |
| `--temp` | `0.6` | Sampling temperature |
| `--top-k` | `20` | Top-K sampling |
| `--min-p` | `0.0` | Min-P sampling (0 = disabled) |

### Server and infrastructure flags

| Flag | Value | What it controls |
|------|-------|-----------------|
| `-np` | `1` | Parallel slots (DFlash works with one slot by default) |
| `--port` | `8082` | HTTP listen port |
| `--no-host` | — | Bypass host buffer, allowing extra buffers to be used |
| `--metrics` | — | Expose Prometheus metrics at `/metrics` |
| `--no-mmap` | — | Load model into memory instead of memory-mapping |
| `--mlock` | — | Lock model pages in RAM to prevent swapping |
| `--log-timestamps` | — | Timestamp each log line |
| `--log-prefix` | — | Prefix log lines with the log level |
| `--log-colors` | `off` | Disable colored log output |

## Platform notes

### NVIDIA CUDA (Windows, Linux)

Full DFlash acceleration: GPU cross-attention ring buffer, device-to-device hidden-state capture and replay, GPU tape path for Qwen3.5/Qwen3.5-MoE architectures. All TurboQuant and TCQ cache types are available.

### Apple Metal (macOS)

DFlash runs through the CPU ring buffer path — functional but slower than CUDA — because there is no GPU cross-attention ring on Metal. Only `turbo3` and `turbo4` are available on Metal; `turbo2` and the TCQ types (`turbo2_tcq`, `turbo3_tcq`) are CUDA-only. On macOS, use `turbo4` for K cache and `turbo4` or `turbo3` for V cache instead of `turbo3_tcq`.

### AMD ROCm

DFlash falls back to the CPU ring buffer path. The ROCm build compiles from the same CUDA source files (via HIP), so TurboQuant and TCQ cache types may work, but compilation success under HIPCC is not guaranteed. If TCQ types fail, use `turbo4` for K cache and `turbo4` or `turbo3` for V cache instead of `turbo3_tcq`. If all TurboQuant types fail, fall back to `q8_0` or `f16` instead. Build with `-DGGML_HIP=ON` instead of `-DGGML_CUDA=ON`.

### Vulkan

Not recommended for DFlash. Falls back to CPU ring with no TurboQuant cache types.

### Multi-GPU

Tree verification (`--spec-branch-budget` > 0) is automatically disabled when the target model spans more than one GPU, but as of now it's very slow and is not included in any recommended configs anyways. Flat DFlash (`--spec-branch-budget 0`) still works across multiple GPUs. Set `GGML_DFLASH_GPU_RING=0` to disable the GPU ring buffer for isolation or debugging on multi-GPU setups.

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_DFLASH_GPU_RING` | enabled | Set to `0` to disable the GPU cross-attention ring buffer and force CPU-only ring |
| `GGML_DFLASH_MAX_CTX` | `4096` | Cap cross-attention context length in tokens. Set to `0` for unlimited |
| `GGML_DFLASH_PROFILE` | `0` | Set to `1` to enable DFlash timing diagnostics in logs |
| `GGML_DFLASH_KV_CACHE_MODE` | `both` | DFlash K/V cache mode: `off`/`none`/`disabled` disables, `k`/`k-only` keeps K only, `v`/`v-only` keeps V only, unset or any other value keeps both |

## Adjusting for your hardware

The default command targets 24 GB VRAM with Q5_K_S. If you are running out of memory, adjust in this order:

1. **Reduce `--ctx-size`.** Each unit of context costs VRAM for both the target model's KV cache and the DFlash cross-attention buffer. Dropping from 122800 to 65536 or 32768 frees significant memory.
2. **Switch cache types.** Replace `turbo4` / `turbo3_tcq` with more aggressive compression. On CUDA, `turbo2` for K and `turbo2_tcq` for V squeeze further. On Metal, use `turbo3` for both K and V. On any backend, standard types (`q8_0`, `q4_0`) are available as a fallback.
3. **Drop the target quantization.** Move from Q5_K_S to Q4_K_M, or as a last resort to IQ4_XS.
4. **Reduce `--spec-dflash-cross-ctx`.** Lowering from 1024 to 512 saves VRAM at the cost of less context for the drafter's cross-attention.
5. **Lower context checkpoints.** Each checkpoint stores a full KV state copy. The default caps at 32 checkpoints per slot (`--ctx-checkpoints 32`), taken every 8192 tokens during prefill (`--checkpoint-every-n-tokens 8192`). At long contexts this adds up. Drop to 16 or 24 to free RAM:

   ```
   --ctx-checkpoints 16
   ```

6. **Remove `--mlock`.** If system RAM is abundant and swapping is not a concern, `--mlock` can be removed.

Start with the precision combo and drop down if VRAM is tight. The drafter quantization matters too: Q4_K_M as default, Q5_K_M for potential gains from precision, IQ4_XS for a tiny bit more VRAM.

### When you have VRAM to spare

If you switch to a lighter target quantization (Q4_K_M or IQ4_XS) or boast a 5090 with its 32GB, the spare VRAM buys you more than just breathing room. Spend it on:

- **Higher `--ctx-size`** — push past 120K and further without hitting the ceiling.
- **Higher `-ub`** — raise the microbatch from 256 to 512 or 1024. This speeds up prompt prefill noticeably because more tokens are batched per GPU kernel launch. The ideal `-ub` is typically the largest power-of-two that still fits.
- **Heavier KV cache types** — if you previously dropped cache compression to save VRAM, you can go back to `turbo4` / `turbo3_tcq` for better precision at the same context length.

## Adaptive draft depth

By default, the server adjusts draft depth using the `profit` controller, which raises and lowers the active draft depth (up to the ceiling set by `--spec-draft-n-max`) based on real-time acceptance rates. You do not need to change anything to benefit from this.

Adaptive draft is highly configurable, so if you are interested in tinkering with it, check out [beellama-args.md](beellama-args.md). For benchmarking with a fixed draft depth, use `--no-spec-dm-adaptive`.

## Troubleshooting

**Out of VRAM.** Reduce `--ctx-size` first, then cache types, then target quantization. See [Adjusting for your hardware](#adjusting-for-your-hardware).

**`spec-type dflash is set but draft model is not a dflash-draft model`.** The file passed to `--spec-draft-model` is not a DFlash draft GGUF. Make sure you are linking the DFlash draft model, not a regular Qwen model.

**`model.n_devices() > 1: disabling parent_ids_gpu`.** Tree verification is disabled because the target model spans multiple GPUs. Flat DFlash still works. This is expected.

**Port already in use.** Change `--port` or stop the existing server.

**Slow DFlash on macOS.** The CPU ring path is slower than the CUDA GPU ring. This is a platform limitation, not a configuration issue. Reducing `--spec-dflash-cross-ctx` to 512 lowers CPU ring overhead.

**TCQ cache types fail on non-CUDA backends.** `turbo2_tcq` and `turbo3_tcq` are CUDA-only. On Metal, use `turbo3` or `turbo4` for V cache instead (note that `turbo2` is also unavailable on Metal). On ROCm, try the TCQ types first; if they fail, try non-TCQ TurboQuant or use `q8_0` or `f16`.

**DFlash seems disabled.** Check the server log for `dflash:` or `speculative` lines. If DFlash is active, you will see draft acceptance rates and timing. If you see no DFlash output, verify that `--spec-type dflash` is set and the draft model loaded successfully. A DFlash draft GGUF auto-detects as `dflash` even without `--spec-type`, but setting it explicitly avoids ambiguity.