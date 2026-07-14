#!/usr/bin/env python3

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tail_files = [
        ROOT / "src/llama-kv-cache-tail.h",
        ROOT / "src/llama-kv-cache-tail.cpp",
        ROOT / "tests/test-kv-cache-tail.cpp",
    ]
    for path in tail_files:
        source = path.read_text(encoding="utf-8")
        if re.search(r"kvarn", source, re.IGNORECASE):
            raise AssertionError(f"standard-tail dependency firewall failed: {path} names KVarN")

    arg = (ROOT / "common/arg.cpp").read_text(encoding="utf-8")
    registry = arg.split("const std::vector<ggml_type> kv_cache_types = {", 1)[1].split("};", 1)[0]
    registered = set(re.findall(r"GGML_TYPE_[A-Z0-9_]+", registry))
    non_quantized = {"GGML_TYPE_F32", "GGML_TYPE_F16", "GGML_TYPE_BF16"}
    quantized = registered - non_quantized

    cuda_out_prod = (ROOT / "ggml/src/ggml-cuda/out-prod.cu").read_text(encoding="utf-8")
    cpu_out_prod = (ROOT / "ggml/src/ggml-cpu/ops.cpp").read_text(encoding="utf-8")
    backend_tests = (ROOT / "tests/test-backend-ops.cpp").read_text(encoding="utf-8")
    missing = {
        cache_type: [
            name
            for name, source in (
                ("CUDA OUT_PROD", cuda_out_prod),
                ("CPU OUT_PROD", cpu_out_prod),
                ("backend operation matrix", backend_tests),
            )
            if cache_type not in source
        ]
        for cache_type in sorted(quantized)
    }
    missing = {cache_type: routes for cache_type, routes in missing.items() if routes}
    if missing:
        details = "; ".join(f"{cache_type}: {', '.join(routes)}" for cache_type, routes in missing.items())
        raise AssertionError(f"registered standard cache type lacks a tail route: {details}")

    cmake = (ROOT / "src/CMakeLists.txt").read_text(encoding="utf-8")
    if "llama-kv-cache-tail.cpp" not in cmake:
        raise AssertionError("standard-tail source is not independently listed in src/CMakeLists.txt")

    cache_source = (ROOT / "src/llama-kv-cache.cpp").read_text(encoding="utf-8")
    constructor = cache_source.split("llama_kv_cache::llama_kv_cache(", 1)[1].split("GGML_ASSERT(kv_size % n_pad == 0);", 1)[0]
    if "const bool shadow_k = tail &&" not in cache_source:
        raise AssertionError("K tail storage must remain null when the tail feature is disabled")
    if "const bool shadow_v = tail &&" not in cache_source:
        raise AssertionError("V tail storage must remain null when the tail feature is disabled")
    if constructor.find("if (other)") > constructor.find("if (tail_tokens > 0)"):
        raise AssertionError("shared-cache size must resolve before exact-tail generation storage is allocated")

    ggml_cmake = (ROOT / "ggml/CMakeLists.txt").read_text(encoding="utf-8")
    cuda_cmake = (ROOT / "ggml/src/ggml-cuda/CMakeLists.txt").read_text(encoding="utf-8")
    kvarn_dispatch = (ROOT / "ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu").read_text(encoding="utf-8")
    matrix_option = "GGML_CUDA_KVARN_FAST_DECODE_ALL_PAIRS"
    if matrix_option not in ggml_cmake or matrix_option not in cuda_cmake or matrix_option not in kvarn_dispatch:
        raise AssertionError("standard and KVarN CUDA instance matrices are not independently selectable")

    default_build = (ROOT / "tmp/build-local-3090-cuda13.1-default.ps1").read_text(encoding="utf-8")
    kvarn_fa_option = "GGML_CUDA_KVARN_FA"
    if (f"option({kvarn_fa_option} " not in ggml_cmake or
            f"if ({kvarn_fa_option})" not in cuda_cmake or
            f"defined({kvarn_fa_option})" not in kvarn_dispatch):
        raise AssertionError("CUDA build cannot omit every KVarN FlashAttention template instance")
    if f"-D{kvarn_fa_option}=OFF" not in default_build:
        raise AssertionError("default standard-quant iteration build must compile zero KVarN FA pairs")

    ggml_header = (ROOT / "ggml/include/ggml.h").read_text(encoding="utf-8")
    cuda_fattn = (ROOT / "ggml/src/ggml-cuda/fattn.cu").read_text(encoding="utf-8")
    ggml_core = (ROOT / "ggml/src/ggml.c").read_text(encoding="utf-8")
    graph = (ROOT / "src/llama-graph.cpp").read_text(encoding="utf-8")
    for required in ("ggml_flash_attn_ext_add_kv_tail", "ggml_kv_tail_attention_merge"):
        if required not in ggml_header:
            raise AssertionError(f"ggml tail-attention contract lacks {required}")
        if required not in ggml_core:
            raise AssertionError(f"ggml core does not implement {required}")
    if "ggml_kv_tail_attention_merge" not in graph:
        raise AssertionError("model graph does not use native tail attention")
    if "ggml_cuda_flash_attn_ext_tail" not in cuda_fattn:
        raise AssertionError("CUDA lacks the native tail-attention dispatch")

    dsv4_cache = (ROOT / "src/llama-kv-cache-dsv4.cpp").read_text(encoding="utf-8")
    dsv4_graph = (ROOT / "src/models/deepseek4.cpp").read_text(encoding="utf-8")
    for required in ("tail_tokens", "cpy_k_tail", "set_input_kq_mask_tail"):
        if required not in dsv4_cache:
            raise AssertionError(f"DSV4 raw standard-cache route lacks {required}")
    for required in ("build_raw_tail", "get_kq_mask_tail"):
        if required not in dsv4_graph:
            raise AssertionError(f"DSV4 attention route lacks {required}")
    build_raw_tail = dsv4_graph.split(
        "ggml_tensor * llama_model_deepseek4::graph::build_raw_tail(", 1
    )[1].split("\n}\n", 1)[0]
    if "ggml_get_rows_as" not in build_raw_tail:
        raise AssertionError("DSV4 raw tail gather must preserve the stored exact type")

    tail_support = cuda_fattn.split(
        "bool ggml_cuda_flash_attn_ext_tail_supported(", 1
    )[1].split("\n}\n", 1)[0]
    if "GGML_USE_HIP" not in tail_support:
        raise AssertionError("unverified HIP tail acceleration must fail closed")
    cuda_tail = (ROOT / "ggml/src/ggml-cuda/fattn-tail.cuh").read_text(encoding="utf-8")
    if "k_flash_attn_ext_tail_merge" in cuda_tail:
        raise AssertionError("serialized indexed tail merge kernel is still present")

    graph_header = (ROOT / "src/llama-graph.h").read_text(encoding="utf-8")
    if "self_tail_bias_read_idxs" not in graph_header or "build_attn_bias_tail" not in graph:
        raise AssertionError("query-specific tails must gather matching body attention bias rows")

    q_tail_layout = re.compile(
        r"q_tail_batched\s*=\s*k_tail\s*&&\s*!tail_read_idxs\s*\?\s*ggml_reshape_4d\([^;]+;.{0,500}?"
        r"q_tail_batched\s*=\s*ggml_permute\(ctx0,\s*q_tail_batched,\s*0,\s*2,\s*1,\s*3\);",
        re.DOTALL,
    )
    if not q_tail_layout.search(graph):
        raise AssertionError("query-specific tail Q must keep query and GQA head axes distinct")

    v_tail_layout = re.compile(
        r"weights_tail\s*=\s*ggml_cont\(ctx0,\s*ggml_permute\(ctx0,\s*weights_tail,\s*0,\s*2,\s*1,\s*3\)\);"
        r"\s*weights_tail\s*=\s*ggml_reshape_4d\(ctx0,\s*weights_tail,\s*"
        r"weights_tail->ne\[0\],\s*1,\s*weights_tail->ne\[1\],\s*"
        r"weights_tail->ne\[2\]\*weights_tail->ne\[3\]\);"
        r".{0,500}?tail_out\s*=\s*ggml_reshape_4d\(ctx0,\s*tail_out,\s*"
        r"tail_out->ne\[0\],\s*body_out->ne\[2\],\s*body_out->ne\[1\],\s*body_out->ne\[3\]\);\s*"
        r"tail_out\s*=\s*ggml_permute\(ctx0,\s*tail_out,\s*0,\s*2,\s*1,\s*3\);",
        re.DOTALL,
    )
    if not v_tail_layout.search(graph):
        raise AssertionError("tail V reduction must transpose query and GQA head axes on both sides of matmul")

    if "k_tail_written ? k_tail_written" not in graph or "v_tail_written ? v_tail_written" not in graph:
        raise AssertionError("same-graph tail reads must depend on the exact-shadow SET_ROWS results")

    hybrid_setter = graph.split("void llm_graph_input_mem_hybrid::set_input", 1)[1].split(
        "bool llm_graph_input_mem_hybrid::can_reuse", 1)[0]
    if "inp_attn->set_input(ubatch)" not in hybrid_setter:
        raise AssertionError("hybrid attention wrapper must delegate every input, including exact tails")


if __name__ == "__main__":
    main()
