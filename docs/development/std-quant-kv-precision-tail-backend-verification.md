# Standard KV precision-tail backend verification issue drafts

These issue bodies are prepared for hardware owners. They are not validation
claims. Replace only the model path and attach the complete logs; do not change
the cache pair, tail types, or checks.

Every package must run the registered ordered-pair operation matrix and both
storage formats:

```text
ctest --test-dir build --output-on-failure -R "test-(backend-ops|kv-cache-tail|arg-parser|save-load-state|std-kv-tail-static)"
llama-cli -m <MODEL.gguf> -p "tail backend verification" -n 8 -c 512 -b 128 -ub 64 \
  --cache-type-k q3_1 --cache-type-v iq4_nl --kv-tail-tokens 64 --kv-tail-type f16
llama-cli -m <MODEL.gguf> -p "tail backend verification" -n 8 -c 512 -b 128 -ub 64 \
  --cache-type-k q4_0 --cache-type-v f16 --kv-tail-tokens 64 --kv-tail-type bf16
```

For each CLI run, repeat with `-ub 32` and compare greedy output. Report the
device, driver/runtime, model SHA-256, full command, commit, output, startup
memory accounting, and whether any graph node fell back to CPU.

## ROCm/HIP issue draft

Title: `Verify standard quantized KV precision tail on ROCm/HIP`

```bash
cmake -S . -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the common matrix above with `-ngl 99 --flash-attn on`. HIP shares the CUDA
quantized `OUT_PROD` source route, but only a real ROCm run can validate device
dispatch and graph scheduling.

## Metal issue draft

Title: `Verify standard quantized KV precision tail on Metal`

```bash
cmake -S . -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the common matrix with `-ngl 99`. Record CPU fallback nodes explicitly; the
backend-neutral graph is the supported correctness route when Metal lacks a
native quantized value partial.

## Vulkan issue draft

Title: `Verify standard quantized KV precision tail on Vulkan`

```bash
cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the common matrix with `-ngl 99` on one Vulkan device and include validation
layer output when available.

## SYCL issue draft

Title: `Verify standard quantized KV precision tail on SYCL`

```bash
cmake -S . -B build -DGGML_SYCL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the common matrix with `-ngl 99` and record the selected SYCL device and
runtime implementation.

## CANN issue draft

Title: `Verify standard quantized KV precision tail on CANN`

```bash
cmake -S . -B build -DGGML_CANN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the common matrix with `-ngl 99` and attach CANN runtime/device versions.
The generic graph may use explicit CPU splits for unsupported quantized partial
operations; silent omission of the tail is a failure.
