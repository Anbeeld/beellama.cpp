#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CACHE = (ROOT / "src/llama-kv-cache-kvarn.cpp").read_text(encoding="utf-8")
DECODE = (ROOT / "ggml/src/ggml-cuda/fattn-mma-kvarn-decode.cuh").read_text(encoding="utf-8")


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

DISPATCH = (ROOT / "ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu").read_text(encoding="utf-8")
assert "GGML_CUDA_FATTN_KVARN_DESCS_THREADS = 1024" in DISPATCH, \
    "KVarN descriptor scan does not use the profiled 1024-thread reduction"
live_index = function_body(DISPATCH, "static __device__ __forceinline__ int ggml_cuda_fattn_kvarn_live_index_for_thread")
assert "uint32_t(payload)" in live_index, \
    "KVarN descriptor scan lost low-32-bit physical-cell decoding"
assert "single_stream" in live_index, \
    "KVarN descriptor scan lacks the single-stream division-free path"

print("KVarN port safety source invariants: OK")
