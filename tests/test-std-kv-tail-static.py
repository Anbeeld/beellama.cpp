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
    if "common_params_kv_tail_type_normalize" in arg:
        raise AssertionError("argument parsing must preserve the automatic KV-tail type sentinel")
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
            )
            if cache_type not in source
        ]
        for cache_type in sorted(quantized)
    }
    missing = {cache_type: routes for cache_type, routes in missing.items() if routes}
    if missing:
        details = "; ".join(f"{cache_type}: {', '.join(routes)}" for cache_type, routes in missing.items())
        raise AssertionError(f"registered standard cache type lacks a tail route: {details}")
    if "common_kv_cache_types()" not in backend_tests:
        raise AssertionError("backend operation matrix is not derived from the standard cache type registry")
    for operation in ("test_get_rows", "test_set_rows_with_shadow", "test_out_prod"):
        if operation not in backend_tests:
            raise AssertionError(f"backend operation matrix lacks {operation}")
    if "set_rows_with_shadow_fused_operand_classification" not in backend_tests:
        raise AssertionError("backend operation matrix lacks the fused SET_ROWS operand-classification case")
    if "selected_cases=" not in backend_tests or "selected_cases == 0" not in backend_tests:
        raise AssertionError("backend operation filtering must report and reject zero selected cases")

    cann = (ROOT / "ggml/src/ggml-cann/ggml-cann.cpp").read_text(encoding="utf-8")
    cann_supports_op = cann.split("static bool ggml_backend_cann_supports_op", 1)[1].split(
        "static void ggml_backend_cann_event_record", 1
    )[0]
    cann_set_rows = cann_supports_op.split("case GGML_OP_SET_ROWS:", 1)[1].split("case GGML_OP_CPY:", 1)[0]
    if not re.search(r"op->src\[3\].*op->src\[4\].*return false", cann_set_rows, re.DOTALL):
        raise AssertionError("CANN must reject fused SET_ROWS shadow operands before ordinary shape checks")

    memory_header = (ROOT / "src/llama-memory.h").read_text(encoding="utf-8")
    if "virtual bool seq_rm_plan(" not in memory_header or "llama_memory_seq_rm_plan_all(" not in memory_header:
        raise AssertionError("memory interface lacks side-effect-free removal planning")
    for query in ("state_seq_can_save", "state_seq_can_restore"):
        if f"virtual bool {query}(llama_seq_id seq_id) const" not in memory_header:
            raise AssertionError(f"memory interface lacks direction-specific {query} preflight")
    if "state_seq_restore_requires_exclusive_kv_stream" in memory_header:
        raise AssertionError("legacy directionless sequence-restore exclusivity predicate is still present")

    context_source = (ROOT / "src/llama-context.cpp").read_text(encoding="utf-8")
    context_state = context_source.split("size_t llama_context::state_seq_get_size", 1)[1].split(
        "bool llama_context::state_load_file", 1
    )[0]
    if context_state.find("state_seq_can_save") > context_state.find("llama_io_write_dummy"):
        raise AssertionError("sequence-state size must preflight save safety before constructing its writer")
    if context_state.find("state_seq_can_restore") > context_state.find("llama_io_read_host"):
        raise AssertionError("sequence-state restore must preflight safety before constructing its reader")

    composite_queries = {
        "src/llama-kv-cache-iswa.cpp": ("llama_kv_cache_iswa", "kv_base", "kv_swa"),
        "src/llama-memory-hybrid.cpp": ("llama_memory_hybrid", "mem_attn", "mem_recr"),
        "src/llama-memory-hybrid-iswa.cpp": ("llama_memory_hybrid_iswa", "mem_attn", "mem_recr"),
    }
    for path, (class_name, first_child, second_child) in composite_queries.items():
        source = (ROOT / path).read_text(encoding="utf-8")
        planner_signature = f"bool {class_name}::seq_rm_plan("
        if planner_signature not in source or "llama_memory_seq_rm_plan_all(" not in source.split(planner_signature, 1)[1].split("}", 1)[0]:
            raise AssertionError(f"{class_name} does not forward sequence-removal planning")
        for query in ("state_seq_can_save", "state_seq_can_restore"):
            signature = f"bool {class_name}::{query}(llama_seq_id seq_id) const"
            if signature not in source:
                raise AssertionError(f"{class_name} does not override {query}")
            body = source.split(signature, 1)[1].split("}", 1)[0]
            if f"{first_child}->{query}(seq_id)" not in body or f"{second_child}->{query}(seq_id)" not in body:
                raise AssertionError(f"{class_name} does not conjunct every child for {query}")

    kvarn_tests = (ROOT / "tests/test-kvarn.cpp").read_text(encoding="utf-8")
    for regression in (
        "kvarn_composite_exclusivity_forwards",
        "kvarn_composite_removal_plan_forwards",
        "kvarn_unified_save_requires_exclusive_stream",
        "kvarn_unified_restore_requires_exclusive_stream",
        "kvarn_historical_suffix_plans_group_boundary",
        "kvarn_historical_suffix_rejects_contended_unified_stream",
        "iswa_nonunified_multislot_kvarn_policy",
    ):
        if regression not in kvarn_tests:
            raise AssertionError(f"test-kvarn lacks the {regression} regression")

    kvarn_cache = (ROOT / "src/llama-kv-cache-kvarn.cpp").read_text(encoding="utf-8")
    if "bool llama_kv_cache_kvarn::seq_rm_plan(" not in kvarn_cache:
        raise AssertionError("KVarN cache lacks the historical suffix planner override")
    exclusivity = kvarn_cache.split(
        "bool llama_kv_cache_kvarn::stream_is_exclusive_for", 1
    )[1].split("bool llama_kv_cache_kvarn::state_seq_can_save", 1)[0]
    if "llama_kvarn_stream_is_exclusive_for" not in exclusivity:
        raise AssertionError("KVarN state safety does not use the unit-tested stream ownership policy")

    llama_api = (ROOT / "include/llama.h").read_text(encoding="utf-8")
    for query in ("llama_memory_can_seq_rm", "llama_memory_seq_rm_plan"):
        if f"LLAMA_API bool {query}(" not in llama_api:
            raise AssertionError(f"public memory API lacks {query}")

    server_context = (ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")
    suffix_block = server_context.split(
        "// truncate any tokens that are beyond n_past for this slot", 1
    )[1].split("// If using an alora", 1)[0]
    if "server_plan_and_remove_suffix(" not in suffix_block:
        raise AssertionError("server prompt suffix trimming bypasses the atomic removal transaction")
    if "common_context_seq_rm" in suffix_block:
        raise AssertionError("server prompt suffix trimming still uses the aborting removal wrapper")

    server_tests = (ROOT / "tests/test-server-prompt-checkpoint.cpp").read_text(encoding="utf-8")
    for regression in (
        "server_unsupported_removal_falls_back_to_full_reprocess",
        "server_post_preflight_mutation_failure_clears_both_contexts",
        "prompt_cache_load_target_success_draft_failure_is_atomic",
    ):
        if regression not in server_tests:
            raise AssertionError(f"server checkpoint tests lack {regression}")

    state_cache_source = (ROOT / "src/llama-kv-cache.cpp").read_text(encoding="utf-8")
    state_tail_reader = state_cache_source.split("void llama_kv_cache::state_read_tail(", 1)[1].split(
        "bool llama_kv_cache::state_read_meta", 1
    )[0]
    if "LLAMA_STATE_SEQ_FLAGS_ON_DEVICE" not in state_tail_reader or "io.read_tensor(" not in state_tail_reader:
        raise AssertionError("standard exact-tail device restore does not use the device tensor protocol")

    state_v2_installer = state_cache_source.split(
        "void llama_kv_cache::state_v2_read_payload_and_install(", 1
    )[1].split("void llama_kv_cache::state_write(", 1)[0]
    if "if (manifest.body_only)" not in state_v2_installer:
        raise AssertionError("v2 state restore does not distinguish an explicit body-only frame")
    if "if (manifest.tail_layers.empty())" in state_v2_installer:
        raise AssertionError("v2 state restore mistakes metadata-only tail ownership for a body-only frame")

    recurrent_header = (ROOT / "src/llama-memory-recurrent.h").read_text(encoding="utf-8")
    recurrent_source = (ROOT / "src/llama-memory-recurrent.cpp").read_text(encoding="utf-8")
    if "bool can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const override" not in recurrent_header:
        raise AssertionError("recurrent memory does not expose its actual rollback capability")
    recurrent_can_remove = recurrent_source.split(
        "bool llama_memory_recurrent::can_seq_rm(", 1
    )[1].split("bool llama_memory_recurrent::seq_rm(", 1)[0]
    if "rollback <= llama_pos(n_rs_seq)" not in recurrent_can_remove:
        raise AssertionError("recurrent removal preflight does not enforce the retained rollback-state window")
    recurrent_remove = recurrent_source.split(
        "bool llama_memory_recurrent::seq_rm(", 1
    )[1].split("bool llama_memory_recurrent::seq_cp(", 1)[0]
    if "if (!can_seq_rm(seq_id, p0, p1))" not in recurrent_remove:
        raise AssertionError("recurrent removal can mutate after an unsupported rollback preflight")

    cmake = (ROOT / "src/CMakeLists.txt").read_text(encoding="utf-8")
    if "llama-kv-cache-tail.cpp" not in cmake:
        raise AssertionError("standard-tail source is not independently listed in src/CMakeLists.txt")

    cache_source = (ROOT / "src/llama-kv-cache.cpp").read_text(encoding="utf-8")
    constructor = cache_source.split("llama_kv_cache::llama_kv_cache(", 1)[1].split(
        "void llama_kv_cache::clear(bool data)", 1
    )[0]
    if "tail_plan.shadow_k &&" not in cache_source:
        raise AssertionError("K shadow allocation must follow the explicit overlay storage plan")
    if "tail_plan.shadow_v &&" not in cache_source:
        raise AssertionError("V shadow allocation must follow the explicit overlay storage plan")
    if constructor.find("if (other)") > constructor.find("if (tail_tokens > 0)"):
        raise AssertionError("shared-cache size must resolve before exact-tail generation storage is allocated")
    if "tail_plan = llama_kv_tail_storage_plan_for" not in constructor:
        raise AssertionError("raw standard cache must select one explicit persistent-tail representation")
    if "tail_plan.kind == LLAMA_KV_TAIL_STORAGE_OVERLAY" not in constructor:
        raise AssertionError("exact-shadow allocation must be guarded by the overlay storage plan")
    if "tail_plan.kind == LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT" not in cache_source:
        raise AssertionError("raw standard cache lacks the native-exact storage route")
    if "model.split_mode()" in constructor:
        raise AssertionError("standard overlay placement still uses CLI split mode instead of realized body storage")
    standard_order = [
        constructor.find("ggml_backend_alloc_ctx_tensors_from_buft"),
        constructor.find("realized %s body uses a tensor/meta split buffer"),
        constructor.find("finalize_tail_overlay_metadata()"),
        constructor.find('ggml_format_name(layer.k_tail, "cache_k_tail_l%d"'),
    ]
    if min(standard_order) < 0 or standard_order != sorted(standard_order):
        raise AssertionError("standard cache must allocate body, reject meta placement, then allocate tail metadata/tensors")

    cache_header = (ROOT / "src/llama-kv-cache.h").read_text(encoding="utf-8")
    get_tail_tokens = cache_header.split("uint32_t get_tail_tokens() const", 1)[1].split("}", 1)[0]
    if "tail_plan.kind == LLAMA_KV_TAIL_STORAGE_OVERLAY" not in get_tail_tokens:
        raise AssertionError("tail graph topology must follow the explicit overlay storage plan")

    context_source = (ROOT / "src/llama-context.cpp").read_text(encoding="utf-8")
    if "llama_kv_tail_resolve_groups" not in context_source or "config.automatic ? automatic_standard : true" not in context_source:
        raise AssertionError("automatic and explicit group resolution are not separated at context construction")

    iswa_source = (ROOT / "src/llama-kv-cache-iswa.cpp").read_text(encoding="utf-8")
    if "llama_kv_tail_storage_plan_for" in iswa_source or "LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT" in iswa_source:
        raise AssertionError("iSWA wrapper must not override raw-cache representation planning")

    ggml_cmake = (ROOT / "ggml/CMakeLists.txt").read_text(encoding="utf-8")
    cuda_cmake = (ROOT / "ggml/src/ggml-cuda/CMakeLists.txt").read_text(encoding="utf-8")
    hip_cmake = (ROOT / "ggml/src/ggml-hip/CMakeLists.txt").read_text(encoding="utf-8")
    musa_cmake = (ROOT / "ggml/src/ggml-musa/CMakeLists.txt").read_text(encoding="utf-8")
    cuda_backend = (ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")
    kvarn_dispatch = (ROOT / "ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu").read_text(encoding="utf-8")
    kvarn_option = "GGML_CUDA_KVARN"
    removed_options = ("GGML_CUDA_KVARN_FA", "GGML_CUDA_KVARN_FAST_DECODE_ALL_PAIRS")
    kvarn_option_line = next((line for line in ggml_cmake.splitlines()
                              if line.startswith(f"option({kvarn_option} ")), "")
    if (not kvarn_option_line.endswith(" ON)") or
            f"if ({kvarn_option})" not in cuda_cmake or
            f"defined({kvarn_option})" not in kvarn_dispatch):
        raise AssertionError("CUDA builds must expose one default-on KVarN compilation gate")
    if any(f"option({option}" in ggml_cmake or option in cuda_cmake + kvarn_dispatch
            for option in removed_options):
        raise AssertionError("obsolete KVarN CUDA compilation options must be removed")
    if any(f"unset({option} CACHE)" not in ggml_cmake for option in removed_options):
        raise AssertionError("obsolete KVarN CUDA cache entries must be cleared during reconfiguration")
    if ("if (GGML_CUDA_FA_ALL_QUANTS)" not in ggml_cmake or
            "#if defined(GGML_CUDA_FA_ALL_QUANTS)" not in kvarn_dispatch):
        raise AssertionError("ALL_QUANTS alone must select the full KVarN fast-decode matrix")
    source_filter = 'EXCLUDE REGEX "kvarn(-wht)?[.]cu$"'
    if any(source_filter not in backend_cmake for backend_cmake in (cuda_cmake, hip_cmake, musa_cmake)):
        raise AssertionError("disabling KVarN must omit its dedicated store and WHT CUDA sources")
    if 'list(FILTER _sources EXCLUDE REGEX "fattn-mma-kvarn")' not in ggml_cmake:
        raise AssertionError("disabling KVarN must omit all KVarN FlashAttention template instances")
    if ("#if !defined(GGML_CUDA_KVARN) || defined(GGML_USE_MUSA)" not in cuda_backend or
            "#if defined(GGML_CUDA_KVARN)\n        case GGML_OP_KVARN_WHT:" not in cuda_backend):
        raise AssertionError("disabled KVarN kernels must not be advertised or dispatched by CUDA")

    default_build = (ROOT / "tmp/build-local-3090-cuda13.1-default.ps1").read_text(encoding="utf-8")
    if default_build.count(f"-D{kvarn_option}=ON") != 1:
        raise AssertionError("default CUDA build must enable the single KVarN compilation toggle")
    if default_build.count("-DGGML_CUDA_FA_ALL_QUANTS=OFF") != 1:
        raise AssertionError("default CUDA build must explicitly select only the default FA pair matrices")
    if any(option in default_build for option in removed_options):
        raise AssertionError("default CUDA build must not use obsolete KVarN compilation toggles")
    default_pairs_block = ggml_cmake.split("set(GGML_CUDA_KVARN_DEFAULT_PAIRS", 1)[1].split(")", 1)[0]
    default_pairs = default_pairs_block.split()
    if len(default_pairs) != 15:
        raise AssertionError("default CUDA build must compile all and only the 15 default KVarN pairs")

    ggml_header = (ROOT / "ggml/include/ggml.h").read_text(encoding="utf-8")
    cuda_fattn = (ROOT / "ggml/src/ggml-cuda/fattn.cu").read_text(encoding="utf-8")
    ggml_core = (ROOT / "ggml/src/ggml.c").read_text(encoding="utf-8")
    graph = (ROOT / "src/llama-graph.cpp").read_text(encoding="utf-8")
    if "ggml_backend_kv_tail_attention_supported" in graph or "backend_supports_native_kv_tail" in graph:
        raise AssertionError("decode graph construction must consume the stored route without backend probing")
    if graph.count("get_tail_route(il)") < 2:
        raise AssertionError("standard and iSWA graph builders do not consume the stored tail route")
    if "resolve_native_exact_routes" not in cache_source or "probe_standard_native_exact_route" not in cache_source:
        raise AssertionError("native-exact storage is allocated without a complete preflight route")
    if "metadata->set_tail_routes" not in kvarn_cache:
        raise AssertionError("KVarN does not store its finalized route in the shared tail plan")
    kvarn_order = [
        kvarn_cache.find("ggml_backend_alloc_ctx_tensors_from_buft"),
        kvarn_cache.find("realized body uses a tensor/meta split buffer"),
        kvarn_cache.find("metadata->finalize_tail_overlay_metadata()"),
        kvarn_cache.find('ggml_format_name(layer.k_tail, "cache_kvarn_k_tail_l%d"'),
    ]
    if min(kvarn_order) < 0 or kvarn_order != sorted(kvarn_order):
        raise AssertionError("KVarN must allocate body, reject meta placement, then allocate tail metadata/tensors")
    for required in ("ggml_flash_attn_ext_add_kv_tail", "ggml_kv_tail_attention_merge"):
        if required not in ggml_header:
            raise AssertionError(f"ggml tail-attention contract lacks {required}")
        if required not in ggml_core:
            raise AssertionError(f"ggml core does not implement {required}")
    if "ggml_kv_tail_attention_merge" not in graph:
        raise AssertionError("model graph does not use native tail attention")
    if "ggml_cuda_flash_attn_ext_tail" not in cuda_fattn:
        raise AssertionError("CUDA lacks the native tail-attention dispatch")

    tail_build_calls = re.findall(r"build_attn_inp_tail\((?:(?!\);).)*\);", graph, re.DOTALL)[1:]
    if not tail_build_calls or any(not re.search(r",\s*true\s*\);$", call) for call in tail_build_calls):
        raise AssertionError("every standard-cache wrapper must select sparse-body packing by the shared capacity invariant")

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
