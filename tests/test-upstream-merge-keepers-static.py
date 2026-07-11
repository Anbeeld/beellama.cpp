#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(source: str, needle: str, message: str) -> None:
    if needle not in source:
        raise AssertionError(message)


def main() -> None:
    dream = (ROOT / "src/models/dream.cpp").read_text(encoding="utf-8")
    for layers, model_type in ((26, "3B"), (28, "7B"), (34, "8B"), (40, "14B")):
        require(
            dream,
            f"case {layers}: type = LLM_TYPE_{model_type};",
            f"Dream {model_type} ({layers} layers) model-size mapping was lost during an upstream merge",
        )

    qwen3next = (ROOT / "src/models/qwen3next.cpp").read_text(encoding="utf-8")
    require(
        qwen3next,
        "const int64_t n_head_kv_il = hparams.n_head_kv(il);",
        "Qwen3Next must use the per-layer KV-head count for RYS variants",
    )
    if qwen3next.count("n_embd_head, n_head_kv_il, n_tokens") != 2:
        raise AssertionError("Qwen3Next K and V reshapes must both use the per-layer KV-head count")

    allocator = (ROOT / "ggml/src/ggml-alloc.c").read_text(encoding="utf-8")
    require(
        allocator,
        "ggml_alloc_is_zero_alloc_proxy",
        "KVarN's bufferless proxy allocation handling was lost during an upstream merge",
    )

    vulkan = (ROOT / "ggml/src/ggml-vulkan/ggml-vulkan.cpp").read_text(encoding="utf-8")
    for needle in (
        "pipeline_kvarn_store",
        "static void ggml_vk_kvarn_store",
        "case GGML_OP_KVARN_STORE:",
        'strcmp(name, "ggml_backend_kvarn_native_ops")',
    ):
        require(vulkan, needle, "the Vulkan KVarN store integration was lost during an upstream merge")

    cuda = (ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")
    for needle in (
        '#include "ggml-cuda/kvarn-wht.cuh"',
        "case GGML_OP_KVARN_WHT:",
        "case GGML_OP_KVARN_STORE:",
        'strcmp(name, "ggml_backend_kvarn_native_ops")',
    ):
        require(cuda, needle, "the CUDA KVarN operation integration was lost during an upstream merge")

    set_rows = (ROOT / "ggml/src/ggml-cuda/set-rows.cu").read_text(encoding="utf-8")
    set_rows_support = cuda.rsplit("case GGML_OP_SET_ROWS:", 1)[1].split("case GGML_OP_SET:", 1)[0]
    for cache_type in ("Q6_1", "Q6_0", "Q3_1", "Q3_0", "Q2_1", "Q2_0S"):
        require(
            set_rows,
            f"dst->type == GGML_TYPE_{cache_type}",
            f"CUDA SET_ROWS dispatch for {cache_type} was lost during an upstream merge",
        )
        require(
            set_rows_support,
            f"GGML_TYPE_{cache_type}",
            f"CUDA SET_ROWS support declaration for {cache_type} was lost during an upstream merge",
        )
    kvarn_wht = ROOT / "ggml/src/ggml-cuda/kvarn-wht.cu"
    if not kvarn_wht.is_file():
        raise AssertionError("the KVarN CUDA WHT kernel was removed with the unrelated TurboQuant WHT file")
    fattn = (ROOT / "ggml/src/ggml-cuda/fattn.cu").read_text(encoding="utf-8")
    require(
        fattn,
        "ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, false, false)",
        "descriptor-native KVarN lost the upstream MMA fixup workspace allocation",
    )
    kvarn_case = (ROOT / "ggml/src/ggml-cuda/fattn-mma-kvarn-case.cuh").read_text(encoding="utf-8")
    require(
        kvarn_case,
        "GGML_ASSERT(v_original_domain);",
        "KVarN MMA selection must reject the impossible original-K/rotated-V domain",
    )
    if "GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE, GGML_CUDA_FATTN_KVARN_TYPE>" in kvarn_case:
        raise AssertionError("KVarN MMA selection still instantiates the impossible original-K/rotated-V domain")

    kvarn_cache = (ROOT / "src/llama-kv-cache-kvarn.h").read_text(encoding="utf-8")
    require(
        kvarn_cache,
        "uint64_t(n_batch) + n_ubatch",
        "non-SWA KVarN staging must cover the logical scheduler batch plus one physical ubatch",
    )

    ggml_cmake = (ROOT / "ggml/CMakeLists.txt").read_text(encoding="utf-8")
    cuda_cmake = (ROOT / "ggml/src/ggml-cuda/CMakeLists.txt").read_text(encoding="utf-8")
    require(
        ggml_cmake,
        "CUDA 12.4-12.7 support size via compress-all",
        "the documented CUDA size-compression compatibility range regressed",
    )
    require(
        cuda_cmake,
        'CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.4" AND GGML_CUDA_COMPRESSION_MODE STREQUAL "size"',
        "CUDA 12.4-12.7 size builds must select the fatbinary compression fallback",
    )
    require(
        cuda_cmake,
        "list(APPEND CUDA_FLAGS -Xfatbin=-compress-all)",
        "CUDA 12.4-12.7 size builds lost the fatbinary compression flag",
    )
    graph = (ROOT / "src/llama-graph.cpp").read_text(encoding="utf-8")
    for needle in (
        "llm_flash_attn_ext_set_kvarn_domain",
        "kvarn_ctx->get_k_native",
        "ggml_kvarn_wht_aux",
        "build_input_kvarn_mat_idxs",
        "set_input_kvarn_mat_idxs",
        "set_mat_idxs(inp->self_kvarn_mat_idxs_swa)",
    ):
        require(graph, needle, "the KVarN graph/domain or SWA-index integration was lost during an upstream merge")

    converter = (ROOT / "convert_hf_to_gguf.py").read_text(encoding="utf-8")
    require(
        converter,
        "has_multimodal_config",
        "the text/mmproj multimodal conversion guard was lost during an upstream merge",
    )

    dsa = (ROOT / "src/llama-kv-cache-dsa.cpp").read_text(encoding="utf-8")
    require(
        dsa,
        "if (!can_seq_rm(seq_id, p0, p1))",
        "the atomic DSA cache removal preflight was lost during an upstream merge",
    )

    model = (ROOT / "src/llama-model.cpp").read_text(encoding="utf-8")
    create_memory = model.split("llama_memory_i * llama_model::create_memory", 1)[1]
    null_memory_arches = create_memory.split("case LLM_ARCH_DEEPSEEK32:", 1)[0]
    if "case LLM_ARCH_DFLASH:" in null_memory_arches:
        raise AssertionError("DFlash requires its own KV cache; routing it to null memory crashes graph reservation")

    server_context = (ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")
    load_model = server_context.split("bool load_model(common_params & params)", 1)[1].split("bool init()", 1)[0]
    resolve_pos = load_model.find("common_speculative_resolve_dflash_draft_n_max")
    output_size_pos = load_model.find("params_base.n_outputs_max = server_n_outputs_max(params_base)")
    if resolve_pos < 0 or output_size_pos < 0 or resolve_pos > output_size_pos:
        raise AssertionError("the omitted DFlash draft maximum must resolve before server output-buffer sizing")

    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    require(release, "name: Build / Release", "the Bee release workflow was replaced by upstream's generic workflow")
    require(release, "beellama-${{", "Bee release assets must retain fork-specific names")
    require(release, 'cuda: ["12.4", "13.1"]', "Bee's Windows release matrix must retain CUDA 13.1")
    if "TurboQuant" in release or "TCQ cache" in release:
        raise AssertionError("release metadata still advertises removed TurboQuant/TCQ support")

    for dockerfile in (
        "cpu.Dockerfile",
        "cuda.Dockerfile",
        "intel.Dockerfile",
        "rocm.Dockerfile",
        "vulkan.Dockerfile",
        "runtime-server.Dockerfile",
        "runtime-intel-server.Dockerfile",
    ):
        container = (ROOT / ".devops" / dockerfile).read_text(encoding="utf-8")
        require(container, 'org.opencontainers.image.title="BeeLlama.cpp"', f"{dockerfile} lost Bee image branding")
        require(container, "upstream DFlash and KVarN", f"{dockerfile} advertises a stale feature set")

    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    require(agents, "103 standard vector pairs", "AGENTS.md does not describe the v0.4.0 CUDA policy")
    require(agents, "draft-dflash", "AGENTS.md does not describe upstream DFlash")
    if "GPU ring" in agents or "profit and fringe" in agents:
        raise AssertionError("AGENTS.md still describes removed speculative systems")

    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    require(
        security,
        "https://github.com/Anbeeld/beellama.cpp/security/advisories/new",
        "fork security reports must not be directed to the upstream repository",
    )


if __name__ == "__main__":
    main()
