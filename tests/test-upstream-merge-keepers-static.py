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

    backend = (ROOT / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
    set_2d = backend.split("void ggml_backend_tensor_set_2d_async", 1)[1].split(
        "void ggml_backend_tensor_get_2d_async", 1
    )[0]
    require(set_2d, "iface.set_tensor_2d_async == NULL", "2D async writes must test the write callback")
    require(set_2d, "tensor write out of bounds", "2D async writes must report write bounds")

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

    sycl_set_rows = (ROOT / "ggml/src/ggml-sycl/set_rows.cpp").read_text(encoding="utf-8")
    require(
        sycl_set_rows,
        "convert<sycl::ext::oneapi::bfloat16, sycl::half>",
        "SYCL SET_ROWS must specialize the unsupported direct bfloat16-to-half conversion",
    )

    metal_device = (ROOT / "ggml/src/ggml-metal/ggml-metal-device.m").read_text(encoding="utf-8")
    rsets_init = metal_device.split(
        "ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev)", 1
    )[1].split("\n}", 1)[0]
    require(
        rsets_init,
        "GGML_UNUSED(dev);",
        "Metal residency-set initialization must tolerate SDKs without residency-set support",
    )

    kvarn_cache = (ROOT / "src/llama-kv-cache-kvarn.h").read_text(encoding="utf-8")
    require(
        kvarn_cache,
        "return 1;",
        "non-SWA KVarN staging must retain exactly one incomplete reference group",
    )
    if "kv_size >= 65536" in kvarn_cache:
        raise AssertionError("non-SWA KVarN staging must not use a context-capacity precision heuristic")

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
        "kvarn_ctx->uses_native_attention",
        "kvarn_ctx->native_attention_uses_original_v",
        "mctx_cur->get_k(ctx0, il)",
        "mctx_cur->get_v(ctx0, il)",
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

    generic_kv = (ROOT / "src/llama-kv-cache.cpp").read_text(encoding="utf-8")
    generic_kv_h = (ROOT / "src/llama-kv-cache.h").read_text(encoding="utf-8")
    if generic_kv.count("dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);") < 4:
        raise AssertionError("standard KV-tail routes must retain a concrete CPU owner for CPU buffer types")
    for retired_generic_msa in ("msa_strict_slots", "get_k_idx", "cpy_k_idx", "n_embd_k_idx"):
        if retired_generic_msa in generic_kv or retired_generic_msa in generic_kv_h:
            raise AssertionError("MiniMax MSA index ownership must not return to the generic KV cache")

    msa_h = (ROOT / "src/llama-kv-cache-msa.h").read_text(encoding="utf-8")
    msa = (ROOT / "src/llama-kv-cache-msa.cpp").read_text(encoding="utf-8")
    for needle in ("tail_tokens", "tail_type", "tail_tokens_requested", "tail_rollback_tokens"):
        require(msa_h, needle, "MiniMax MSA must expose exact-tail configuration for its base cache")
    require(msa, "if (!can_seq_rm(seq_id, p0, p1))", "MSA base/index removal must preflight atomically")
    require(msa, "GGML_TYPE_F32, GGML_TYPE_F32", "the MSA index cache must remain ordinary F32 storage")
    base_ctor = msa.split("kv_base = std::make_unique<llama_kv_cache>", 1)[1].split("kv_idx =", 1)[0]
    for needle in ("n_ubatch", "tail_tokens", "tail_type", "tail_tokens_requested", "tail_rollback_tokens"):
        require(base_ctor, needle, "MSA exact-tail configuration must be applied to kv_base")
    idx_ctor = msa.split("kv_idx = std::make_unique<llama_kv_cache>", 1)[1].split("void llama_kv_cache_msa::clear", 1)[0]
    if "tail_tokens" in idx_ctor or "tail_type" in idx_ctor:
        raise AssertionError("MSA index storage must not receive a precision-tail configuration")

    model = (ROOT / "src/llama-model.cpp").read_text(encoding="utf-8")
    create_memory = model.split("llama_memory_i * llama_model::create_memory", 1)[1]
    null_memory_arches = create_memory.split("case LLM_ARCH_DEEPSEEK32:", 1)[0]
    if "case LLM_ARCH_DFLASH:" in null_memory_arches:
        raise AssertionError("DFlash requires its own KV cache; routing it to null memory crashes graph reservation")

    dflash = (ROOT / "src/models/dflash.cpp").read_text(encoding="utf-8")
    if dflash.count("Kcur = llama_mul_mat_hadamard(ctx0, Kcur") != 2:
        raise AssertionError("DFlash must rotate injected K for both base and ISWA quantized caches")
    if dflash.count("Vcur = llama_mul_mat_hadamard(ctx0, Vcur") != 2:
        raise AssertionError("DFlash must rotate injected V for both base and ISWA quantized caches")

    server_context = (ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")
    load_model = server_context.split("bool load_model(common_params & params)", 1)[1].split("bool init()", 1)[0]
    resolve_pos = load_model.find("common_speculative_resolve_dflash_draft_n_max")
    output_size_pos = load_model.find("const auto output_limits = server_output_limits(params_base)")
    if resolve_pos < 0 or output_size_pos < 0 or resolve_pos > output_size_pos:
        raise AssertionError("the omitted DFlash draft maximum must resolve before server output-buffer sizing")
    require(load_model, "params_base.n_outputs_max = output_limits.total;", "server output-buffer total was not applied")
    require(load_model, "params_base.n_outputs_max_per_seq = output_limits.per_seq;", "per-sequence output limit was not applied")

    common_h = (ROOT / "common/common.h").read_text(encoding="utf-8")
    common_cpp = (ROOT / "common/common.cpp").read_text(encoding="utf-8")
    require(common_h, "seq_rm_suffix", "common_memory must own transactional target/draft suffix removal")
    require(common_cpp, "common_memory::seq_rm_suffix", "common_memory suffix removal implementation is missing")
    require(server_context, "slot.mem.seq_rm_suffix(", "the server must use common_memory for suffix transactions")
    suffix_path = server_context.split("cached n_tokens =", 1)[1].split("If using an alora", 1)[0]
    if "llama_memory_seq_rm_plan(" in suffix_path or "server_plan_and_remove_suffix(" in suffix_path:
        raise AssertionError("the server still duplicates target/draft suffix planning outside common_memory")

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
    require(agents, "50 standard vector pairs", "AGENTS.md does not describe the v0.4.0 CUDA policy")
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
