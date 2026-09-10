#!/usr/bin/env python3
from pathlib import Path
import re

src = Path("ggml/src/ggml-cuda/kvarn.cu").read_text(encoding="utf8")

quantize = re.search(
    r"static __device__ void kvarn_d64_quantize_stage\((.*?)\n}\n\nstatic __global__ void kvarn_d64_store_kernel",
    src,
    re.S,
)
assert quantize, "missing D64 quantizer"
body = quantize.group(1)
assert "kvarn_d64_update_best(" in body, "D64 quantization must use the block-parallel reduction"
assert "if (threadIdx.x < rows)" in body, "D64 row quantization must be distributed across CUDA threads"
assert not re.search(r"if \(threadIdx\.x == 0\) \{\s*for \(int c = 0; c < cols", body), \
    "D64 record quantization must not be serialized onto one CUDA thread"

launch = re.search(r"kvarn_d64_store_kernel<<<n_heads,\s*(\d+)", src)
assert launch, "missing D64 store launch"
assert int(launch.group(1)) >= 128, "D64 store needs one CUDA thread for every rectangular record axis element"

for symbol in (
    "kvarn_d64_store_workspace_stage_kernel",
    "kvarn_d64_store_workspace_flush_kernel",
    "kvarn_d64_store_workspace_commit_kernel",
):
    assert symbol in src, f"missing D64 batched-store kernel: {symbol}"
assert "d64_use_workspace" in src, "D64 prompt stores must select the batched workspace route"
assert re.search(r"kvarn_store_workspace_validate_kernel<<<1, 256.*?swa,\s*eager_records,\s*false,\s*workspace_valid", src, re.S), \
    "D64 workspace validation must reject eager non-contiguous group jumps"
assert "kvarn_d64_materialize_group_kernel" in src, \
    "D64 linear materialization must batch each 128-token record into one CUDA block"
assert "KVAR_N_D64_MAX_RECORD_BYTES" in src and "record_sh" in src, \
    "D64 grouped materialization must cooperatively stage each compact record"
assert "kvarn_d64_materialize_record_value" in src, \
    "D64 grouped materialization must compile bit-width-specific unpacking"
assert re.search(r"kvarn_materialize_live_kernel<<<n_stream,\s*256", src), \
    "materialization live-position discovery must parallelize long index scans"
assert "t -= KVAR_N_DIM" in src, \
    "validated D64 stage commits must inspect only same-position record candidates"

dispatch = Path("ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu").read_text(encoding="utf8")
assert "64 * 128 + 8 * 128 + 18" in dispatch, \
    "D64 capability admission must cover the production sealer's full shared workspace"

portable = Path("ggml/src/ggml-cuda/fattn-kvarn-portable.cuh").read_text(encoding="utf8")
assert "GGML_CUDA_FATTN_KVARN_PORTABLE_SPLIT_TOKENS" in portable, \
    "D64 direct attention must split long contexts across CUDA blocks"
assert "ggml_cuda_fattn_kvarn_portable_combine_kernel" in portable, \
    "split D64 attention must merge partial online-softmax results"
assert "ggml_cuda_fattn_kvarn_d64_gqa_kernel" in portable, \
    "portable D64 decode must share each record dequantization across GQA heads"
assert re.search(r"n_splits\s*=.*?n_kv.*?PORTABLE_SPLIT_TOKENS", portable, re.S), \
    "portable launch must derive split count from context length"

decode = Path("ggml/src/ggml-cuda/fattn-mma-kvarn-decode.cuh").read_text(encoding="utf8")
assert "D == 64 || D == 128" in decode, "optimized KVarN decode must admit D64"
assert "k_payload_bytes = RECORD_DIM * GGML_CUDA_FATTN_KVARN_DIM" in decode, \
    "optimized decode must preserve one-record payload offsets for D128+ slices"
assert "v_row_bytes = RECORD_DIM * V_BITS / 8" in decode, \
    "optimized decode must preserve one-record row strides for D128+ slices"
assert "STAGE_PAYLOAD = D == 64 ||" in decode and "payload_sh[STAGE_PAYLOAD" in decode, \
    "optimized D64 decode must stage rectangular payloads cooperatively"
generator = Path("ggml/src/ggml-cuda/template-instances/generate_cu_files.py").read_text(encoding="utf8")
assert "KVARN_DECODE_HEAD_SIZES = [64, 128, 256, 512]" in generator, \
    "generated optimized decode coverage must include D64"

planner = Path("src/llama-kvarn.cpp").read_text(encoding="utf8")
assert "LLAMA_KVARN_D64_NATIVE_MAX_KV" not in planner, \
    "D64 decode must not fall back to whole-context materialization"
graph = Path("src/llama-graph.cpp").read_text(encoding="utf8")
assert graph.count("(uint32_t) q_cur->ne[2],\n        (int) q_cur->ne[0]") == 2, \
    "full and iSWA planners must receive query-token count, not query-head count"

meta = Path("ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf8")
assert "head_width == 64 || head_width == 128" in meta, \
    "tensor-split metadata must admit the published D64 WHT geometry"

print("KVarN D64 parallel store and split-attention source invariants: OK")
