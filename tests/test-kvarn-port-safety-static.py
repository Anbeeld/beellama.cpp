#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CACHE = (ROOT / "src/llama-kv-cache-kvarn.cpp").read_text(encoding="utf-8")
DECODE = (ROOT / "ggml/src/ggml-cuda/fattn-mma-kvarn-decode.cuh").read_text(encoding="utf-8")
DECODE_VEC = (ROOT / "ggml/src/ggml-cuda/fattn-kvarn-vec.cuh").read_text(encoding="utf-8")
DECODE_COMBINE = (ROOT / "ggml/src/ggml-cuda/fattn-mma-kvarn-decode-combine.cuh").read_text(encoding="utf-8")
WINDOW_CASE = (ROOT / "ggml/src/ggml-cuda/fattn-mma-kvarn-case.cuh").read_text(encoding="utf-8")
WINDOW_COMMON = (ROOT / "ggml/src/ggml-cuda/fattn-mma-kvarn-window-common.cuh").read_text(encoding="utf-8")
GENERATOR = (ROOT / "ggml/src/ggml-cuda/template-instances/generate_cu_files.py").read_text(encoding="utf-8")
DECODE_COMBINE_INSTANCE = (ROOT / "ggml/src/ggml-cuda/template-instances/fattn-mma-kvarn-decode-combine-instance.cu").read_text(encoding="utf-8")
WINDOW_COMMON_INSTANCE = (ROOT / "ggml/src/ggml-cuda/template-instances/fattn-mma-kvarn-window-common-instance.cu").read_text(encoding="utf-8")
GGML_CMAKE = (ROOT / "ggml/CMakeLists.txt").read_text(encoding="utf-8")
BACKEND_CMAKE = {
    name: (ROOT / path).read_text(encoding="utf-8")
    for name, path in {
        "CUDA": "ggml/src/ggml-cuda/CMakeLists.txt",
        "HIP": "ggml/src/ggml-hip/CMakeLists.txt",
        "MUSA": "ggml/src/ggml-musa/CMakeLists.txt",
    }.items()
}


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for pos in range(brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1:pos]
    raise AssertionError(f"unterminated function: {signature}")


next_body = function_body(CACHE, "bool llama_kv_cache_kvarn_context::next()")
apply_body = function_body(CACHE, "bool llama_kv_cache_kvarn_context::apply()")
assert "compact_read_plan_cache.clear();" in next_body, \
    "KVarN compact read plan must be invalidated when advancing to another ubatch"
assert "compact_read_plan_cache.clear();" in apply_body, \
    "KVarN compact read plan must be invalidated before applying changed slot metadata"

physical = re.search(r"constexpr int Q_ROWS\s*=\s*MAX_GQA\s*<\s*8\s*\?\s*8\s*:\s*MAX_GQA\s*;", DECODE)
assert physical, "KVarN MMA decode must pad physical query storage to the eight-row ldmatrix tile"
assert re.search(r"q_sh\s*\[Q_TILE\]\s*\[Q_ROWS\]\s*\[Q_STRIDE2\]", DECODE), \
    "KVarN MMA decode query shared storage does not use the padded physical row count"
assert "i < Q_ROWS * D" in DECODE, \
    "KVarN MMA decode does not initialize every physical query row read by ldmatrix"

padded = re.search(r"constexpr int P_ROWS\s*=\s*MAX_GQA\s*<\s*8\s*\?\s*8\s*:\s*MAX_GQA\s*;", DECODE)
assert padded, "KVarN MMA decode must pad physical probability storage to the eight-row ldmatrix tile"
assert re.search(r"p_sh\s*\[Q_TILE\]\s*\[P_ROWS\]\s*\[P_STRIDE2\]", DECODE), \
    "KVarN MMA decode probability shared storage does not use the padded physical row count"
assert re.search(
    r"for \(int i = tid; i < \(P_ROWS - MAX_GQA\) \* SPLIT_TOKENS;\s*i \+= NWARPS \* PHYSICAL_WAVE_SIZE\).*?"
    r"p_h\[.*?\] = __float2half\(0\.0f\);",
    DECODE,
    re.DOTALL,
), "KVarN MMA decode does not initialize padded probability rows for every query tile row"

assert "static __global__ void ggml_cuda_fattn_kvarn_decode_combine_kernel" not in DECODE, \
    "bit-pair decode instances still define the bit-independent combine kernel"
assert "static __global__ void ggml_cuda_fattn_kvarn_decode_combine_kernel" not in DECODE_VEC, \
    "vec bit-pair instances still define the common combine kernel"
assert "ggml_cuda_fattn_kvarn_decode_combine_kernel" in DECODE_COMBINE and \
       "ggml_cuda_fattn_kvarn_decode_combine_get_kernel" in DECODE_COMBINE, \
    "KVarN decode combine machinery lacks a canonical implementation"
assert "ggml_cuda_fattn_kvarn_decode_combine_launch" not in DECODE_COMBINE and \
       "ggml_cuda_fattn_kvarn_decode_combine_launch" not in DECODE and \
       "ggml_cuda_fattn_kvarn_decode_combine_launch" not in DECODE_VEC, \
    "decode still pays a cross-TU host-wrapper call on every launch"
assert "static __global__ void ggml_cuda_fattn_kvarn_window_dequant_kernel" not in WINDOW_CASE and \
       "static __global__ void ggml_cuda_fattn_kvarn_window_finalize_kernel" not in WINDOW_CASE, \
    "column-geometry instances still define D-only window kernels"
assert "ggml_cuda_fattn_kvarn_window_dequant_kernel" in WINDOW_COMMON and \
       "ggml_cuda_fattn_kvarn_window_finalize_kernel" in WINDOW_COMMON, \
    "D-only window kernels lack canonical implementations"
assert "SOURCE_FATTN_MMA_KVARN_DECODE_COMBINE" in GENERATOR and \
       "SOURCE_FATTN_MMA_KVARN_WINDOW_COMMON" in GENERATOR, \
    "KVarN common kernel instance files are not generator-owned"
assert 'list(FILTER _sources EXCLUDE REGEX "fattn-mma-kvarn")' in GGML_CMAKE and \
       'if (NOT SRC MATCHES "fattn-mma-kvarn-decode-instance-")' in GGML_CMAKE, \
    "KVarN source selection no longer excludes all KVarN-off instances or preserves common TUs"
for backend, cmake in BACKEND_CMAKE.items():
    assert 'file(GLOB   SRCS "' in cmake and 'template-instances/fattn-mma*.cu' in cmake and \
           "ggml_cuda_select_kvarn_fast_decode_sources" in cmake, \
        f"{backend} backend does not route generated MMA sources through KVarN selection"
assert re.findall(r"DECL_FATTN_KVARN_DECODE_COMBINE\((128|256|512)\);", DECODE_COMBINE_INSTANCE) == ["128", "256", "512"], \
    "decode combine common TU must instantiate exactly D128/D256/D512"
assert re.findall(r"DECL_FATTN_KVARN_WINDOW_COMMON\((128|256|512)\);", WINDOW_COMMON_INSTANCE) == ["128", "256", "512"], \
    "window common TU must instantiate exactly D128/D256/D512"
assert "ggml_cuda_fattn_kvarn_decode_combine_get_kernel<D>()" in DECODE and \
       "ggml_cuda_fattn_kvarn_decode_combine_get_kernel<D>()" in DECODE_VEC, \
    "MMA or vec decode does not cache the canonical combine kernel"
assert "<<<blocks_combine, GGML_CUDA_FATTN_KVARN_DECODE_THREADS, nbytes_shared_combine, args.stream>>>" in DECODE and \
       "<<<blocks_combine, GGML_CUDA_FATTN_KVARN_DECODE_THREADS," in DECODE_VEC, \
    "MMA or vec decode changed the existing direct-launch block, shared-memory, or stream arguments"
assert "ggml_cuda_fattn_kvarn_window_dequant_get_kernel<DKQ>()" in WINDOW_CASE and \
       "ggml_cuda_fattn_kvarn_window_finalize_get_kernel<DV>()" in WINDOW_CASE, \
    "window path still pays a cross-TU host-wrapper call on every chunk"
assert "GGML_CUDA_FATTN_KVARN_DECODE_RESOURCE_PAD" in DECODE and \
       "__CUDA_ARCH__ == 860" in DECODE and \
       "__CUDACC_VER_MAJOR__ == 13" in DECODE and \
       "__CUDACC_VER_MINOR__ == 1" in DECODE and \
       "!defined(__HIPCC__)" in DECODE and \
       "!defined(GGML_USE_MUSA)" in DECODE, \
    "decode resource compensation is not restricted to the verified NVCC 13.1/sm_86 build"
assert "__shared__ float denom_sh[Q_TILE][MAX_GQA + GGML_CUDA_FATTN_KVARN_DECODE_RESOURCE_PAD]" in DECODE, \
    "decode dedup no longer preserves the verified baseline shared-memory footprint"
for duplicate in (
    "GGML_CUDA_FATTN_KVARN_WINDOW_CHUNK",
    "ggml_cuda_fattn_kvarn_window_enabled",
    "ggml_cuda_fattn_kvarn_window_chunk",
    "ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel",
    "ggml_cuda_fattn_kernel_attr_ptr_t",
):
    assert duplicate not in WINDOW_COMMON, f"window common TU duplicates unrelated case helper: {duplicate}"
assert "const dim3 dequant_block((uint32_t) (2 * plan.slices * warp_size_host), 1, 1)" in WINDOW_CASE and \
       "const dim3 merge_block(DV, 1, 1)" in WINDOW_CASE, \
    "window kernel getter refactor changed dequant or finalize block geometry"

DISPATCH = (ROOT / "ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu").read_text(encoding="utf-8")
assert "GGML_CUDA_FATTN_KVARN_DESCS_THREADS = 1024" in DISPATCH, \
    "KVarN descriptor scan does not use the profiled 1024-thread reduction"
live_index = function_body(DISPATCH, "static __device__ __forceinline__ int ggml_cuda_fattn_kvarn_live_index_for_thread")
assert "uint32_t(payload)" in live_index, \
    "KVarN descriptor scan lost low-32-bit physical-cell decoding"
assert "single_stream" in live_index, \
    "KVarN descriptor scan lacks the single-stream division-free path"

print("KVarN port safety source invariants: OK")
