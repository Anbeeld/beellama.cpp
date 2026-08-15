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
    for removed_policy in (
        "recurrent_prompt_slot",
        "checkpoint_min_step_effective",
        "prompt_add_limit",
        "numerically unstable recurrent checkpoint boundary",
    ):
        if removed_policy in server_context:
            raise AssertionError(f"server retains over-scoped prompt policy: {removed_policy}")
    suffix_block = server_context.split(
        "// truncate any tokens that are beyond n_past for this slot", 1
    )[1].split("// If using an alora", 1)[0]
    if "slot.mem.seq_rm_suffix(" not in suffix_block:
        raise AssertionError("server prompt suffix trimming bypasses the common_memory transaction")
    if "llama_memory_seq_rm" in suffix_block or "common_context_seq_rm" in suffix_block:
        raise AssertionError("server prompt suffix trimming still duplicates raw memory removal")

    common_source = (ROOT / "common/common.cpp").read_text(encoding="utf-8")
    fit_callsite = common_source.split(
        "if (params.fit_params) {", 1
    )[1].split("llama_model * model = llama_model_load_from_file", 1)[0]
    if "common_fit_params(" not in fit_callsite:
        raise AssertionError("common init no longer invokes upstream parameter fitting")
    if "fit_status" in fit_callsite or "failed to fit parameters with exact Bee validation" in fit_callsite:
        raise AssertionError("common init hard-fails an advisory upstream fit conflict")

    server_tests = (ROOT / "tests/test-server-prompt-checkpoint.cpp").read_text(encoding="utf-8")
    for regression in (
        "server_unsupported_removal_falls_back_to_full_reprocess",
        "server_post_preflight_mutation_failure_clears_both_contexts",
        "prompt_cache_load_target_success_draft_failure_is_atomic",
    ):
        if regression not in server_tests:
            raise AssertionError(f"server checkpoint tests lack {regression}")

    context_source = (ROOT / "src/llama-context.cpp").read_text(encoding="utf-8")
    if "96u*model.hparams.n_layer_all" not in context_source:
        raise AssertionError("precision-tail graphs lack the post-upstream per-layer node allowance")

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

    io_header = (ROOT / "src/llama-io.h").read_text(encoding="utf-8")
    for operation in ("commit", "cancel", "stage_tensor_set", "stage_tensor_clear", "on_commit"):
        if f"virtual void {operation}(" not in io_header:
            raise AssertionError(f"state reader transaction lacks explicit {operation}")

    reader_source = context_source.split("class llama_io_read_host", 1)[0] + "class llama_io_read_host" + context_source.split(
        "class llama_io_read_host", 1
    )[1].split("size_t llama_context::state_get_size", 1)[0]
    for reader in ("llama_io_read_host", "llama_io_read_file", "llama_io_read_device"):
        body = reader_source.split(f"class {reader}", 1)[1].split("\nclass ", 1)[0]
        destructor = body.split(f"~{reader}()", 1)[1].split("}", 1)[0]
        if "cancel()" not in destructor:
            raise AssertionError(f"{reader} destructor must cancel uncommitted restore writes")

    kvarn_state_reader = kvarn_cache.split(
        "void llama_kv_cache_kvarn::state_read(", 1
    )[1].split("llama_kv_cache * llama_kv_cache_kvarn::get_metadata_cache", 1)[0]
    if "metadata_prepared" not in kvarn_state_reader or "io.on_commit" not in kvarn_state_reader:
        raise AssertionError("KVarN restore does not prepare metadata and publish it only at transaction commit")
    if "ggml_backend_tensor_set" in kvarn_state_reader or "ggml_backend_tensor_memset" in kvarn_state_reader:
        raise AssertionError("KVarN restore directly mutates destination tensors during preparation")

    cache_header_state = (ROOT / "src/llama-kv-cache.h").read_text(encoding="utf-8")
    if "llama_memory_status update(" not in cache_header_state:
        raise AssertionError("deferred KV update still overloads a boolean success result")
    transaction_header_path = ROOT / "src/llama-kv-cache-update.h"
    if not transaction_header_path.exists():
        raise AssertionError("deferred KV update lacks a testable transaction boundary")
    transaction_header = transaction_header_path.read_text(encoding="utf-8")
    for marker in ("llama_kv_tail_copy_transaction", "allocate", "transfer", "compute", "rollback"):
        if marker not in transaction_header:
            raise AssertionError(f"deferred KV update transaction lacks {marker}")
    context_header = (ROOT / "src/llama-context.h").read_text(encoding="utf-8")
    if "llama_memory_status memory_update(bool optimize)" not in context_header:
        raise AssertionError("context memory update cannot distinguish no-work from failure")
    cache_apply = state_cache_source.split("bool llama_kv_cache_context::apply()", 1)[1].split(
        "llama_memory_status llama_kv_cache_context::get_status", 1
    )[0]
    if "status = kv->update" not in cache_apply or "llama_memory_status_is_fail(status)" not in cache_apply:
        raise AssertionError("KV update failure is not propagated through memory-context apply")
    decode_update = context_source.split("// handle any pending shifts/copies", 1)[1].split(
        "llama_memory_context_ptr mctx", 1
    )[0]
    if "llama_memory_status_is_fail" not in decode_update:
        raise AssertionError("decode ignores a failed deferred memory update")
    fault_tests_path = ROOT / "tests/test-kv-tail-copy-transaction.cpp"
    if not fault_tests_path.exists():
        raise AssertionError("deferred KV update lacks fault-injection tests")
    fault_tests = fault_tests_path.read_text(encoding="utf-8")
    for regression in ("tail_copy_allocation_failure", "tail_copy_transfer_failure", "tail_copy_compute_failure"):
        if regression not in fault_tests:
            raise AssertionError(f"state tests lack {regression}")

    model_header = (ROOT / "src/llama-model.h").read_text(encoding="utf-8")
    if "self_attention_uses_explicit_bias(uint32_t il) const" not in model_header:
        raise AssertionError("model does not expose its per-layer self-attention bias contract")
    probe_spec = state_cache_source.split("struct kv_tail_backend_probe_spec", 1)[1].split("};", 1)[0]
    if "bool explicit_bias" not in probe_spec:
        raise AssertionError("KV-tail backend probe does not carry the model-owned bias contract")
    route_resolution = state_cache_source.split("const auto resolve_overlay_routes", 1)[1].split(
        "const auto route_failure", 1
    )[0]
    if "spec.explicit_bias" not in route_resolution or re.search(
            r"hparams\.causal_attn,\s*n_swa\s*>\s*0,\s*false", route_resolution):
        raise AssertionError("standard KV-tail routes still hardcode explicit_bias=false")
    graph_bias_source = (ROOT / "src/llama-graph.cpp").read_text(encoding="utf-8")
    for marker in ("get_tail_explicit_bias(il)", "kq_b != nullptr"):
        if marker not in graph_bias_source:
            raise AssertionError("attention graph does not assert the recorded bias contract")

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
    if "!storage_request.already_exact" not in constructor:
        raise AssertionError("already-exact bodies must not acquire an attached-tail execution route")
    if "model.split_mode()" in constructor:
        raise AssertionError("standard overlay placement still uses CLI split mode instead of realized body storage")
    standard_order = [
        constructor.find("ggml_backend_alloc_ctx_tensors_from_buft"),
        constructor.find("validate_meta_body(layer, owner_k"),
        constructor.find("finalize_tail_overlay_metadata()"),
        constructor.find('ggml_format_name(layer.k_tail, "cache_k_tail_l%d"'),
    ]
    if min(standard_order) < 0 or standard_order != sorted(standard_order):
        raise AssertionError("standard cache must validate its realized meta split before allocating tail metadata/tensors")
    if "realized %s body uses a tensor/meta split buffer" in constructor:
        raise AssertionError("standard precision tails still reject a valid upstream tensor/meta owner")

    model_source = (ROOT / "src/llama-model.cpp").read_text(encoding="utf-8")
    placement_header = (ROOT / "src/llama-kv-cache-placement.h").read_text(encoding="utf-8")
    meta_source = (ROOT / "ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")
    for role in (
        "STANDARD_K_TAIL", "STANDARD_V_TAIL", "KVARN_K_RECORDS",
        "KVARN_V_RECORDS", "KVARN_K_STAGE", "KVARN_V_STAGE",
        "KVARN_K_TAIL", "KVARN_V_TAIL",
    ):
        if f"LLAMA_KV_CACHE_COMPONENT_{role}" not in placement_header or f"LLAMA_KV_CACHE_COMPONENT_{role}" not in model_source:
            raise AssertionError(f"typed meta placement is missing cache component role {role}")
    if "llama_kv_cache_component_from_name" not in model_source:
        raise AssertionError("upstream split callback does not consume the typed cache component adapter")
    if "handle_kvarn_cache" not in meta_source:
        raise AssertionError("meta dispatch does not preserve KVarN's sharded payload state")
    if "cgraph_ij->uid = 0" not in meta_source:
        raise AssertionError("projected meta graphs must declare that they have no stable identity")
    cuda_graph_source = (ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")
    cuda_graph_compatibility = cuda_graph_source.split(
        "static bool ggml_cuda_graph_check_compability", 1
    )[1].split("static const void * ggml_cuda_graph_get_key", 1)[0]
    if "cgraph->uid == 0" not in cuda_graph_compatibility:
        raise AssertionError("CUDA graph replay must reject projected graphs without a stable identity")

    cache_header = (ROOT / "src/llama-kv-cache.h").read_text(encoding="utf-8")
    get_tail_tokens = cache_header.split("uint32_t get_tail_tokens() const", 1)[1].split("}", 1)[0]
    if "has_tail_overlay()" not in get_tail_tokens:
        raise AssertionError("tail graph topology must follow every exact-tail storage representation")

    context_source = (ROOT / "src/llama-context.cpp").read_text(encoding="utf-8")
    if "llama_kv_tail_resolve_groups" not in context_source or "config.automatic ? automatic_standard : true" not in context_source:
        raise AssertionError("automatic and explicit group resolution are not separated at context construction")

    iswa_source = (ROOT / "src/llama-kv-cache-iswa.cpp").read_text(encoding="utf-8")
    if "llama_kv_tail_storage_plan_for" in iswa_source or "LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT" in iswa_source:
        raise AssertionError("iSWA wrapper must not override raw-cache representation planning")
    full_coverage_branch = iswa_source.split(
        "if ((is_swa_group && tail_native_exact_swa)", 1
    )[1].split("// Structured KVarN records", 1)[0]
    if ("is_swa_group ? type_k : tail_type" not in full_coverage_branch or
            "is_swa_group ? type_v : tail_type" not in full_coverage_branch):
        raise AssertionError("fully covered non-SWA KVarN windows must use exact body storage")

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
    if not re.search(
        r"#if defined\(GGML_CUDA_KVARN\)\s+"
        r'if \(strcmp\(name, "ggml_backend_kvarn_store_route_stats_reset"\).*?'
        r'if \(strcmp\(name, "ggml_backend_kvarn_store_route_stats_get"\).*?'
        r"#endif",
        cuda_backend,
        re.DOTALL,
    ):
        raise AssertionError("KVarN-off builds must not link store-route telemetry from the excluded kvarn.cu")

    all_quants_option_line = next((line for line in ggml_cmake.splitlines()
                                   if line.startswith("option(GGML_CUDA_FA_ALL_QUANTS ")), "")
    if not all_quants_option_line.endswith(" OFF)"):
        raise AssertionError("default CUDA builds must select only the default FA pair matrices")
    default_pairs_block = ggml_cmake.split("set(GGML_CUDA_KVARN_DEFAULT_PAIRS", 1)[1].split(")", 1)[0]
    default_pairs = default_pairs_block.split()
    if len(default_pairs) != 15:
        raise AssertionError("default CUDA build must compile all and only the 15 default KVarN pairs")

    ggml_header = (ROOT / "ggml/include/ggml.h").read_text(encoding="utf-8")
    cuda_fattn = (ROOT / "ggml/src/ggml-cuda/fattn.cu").read_text(encoding="utf-8")
    ggml_core = (ROOT / "ggml/src/ggml.c").read_text(encoding="utf-8")
    graph = (ROOT / "src/llama-graph.cpp").read_text(encoding="utf-8")
    cache_header = (ROOT / "src/llama-kv-cache.h").read_text(encoding="utf-8")
    route_header = (ROOT / "src/llama-kv-cache-tail.h").read_text(encoding="utf-8")
    if "bool has_body;" not in route_header:
        raise AssertionError("per-layer KV-tail execution descriptor does not own body presence")
    if "uint32_t body_execution_rows;" not in route_header:
        raise AssertionError("per-layer KV-tail execution descriptor does not own packed-body extent")
    if "bool has_current;" not in route_header:
        raise AssertionError("per-layer KV-tail execution descriptor does not own current-segment presence")
    if "virtual bool has_kv_body(int32_t il) const" not in cache_header:
        raise AssertionError("KV-cache graph interface exposes only component-wide body presence")
    if graph.count("has_kv_body(il)") < 4:
        raise AssertionError("standard and iSWA graph builders do not consume per-layer body presence")
    if "mixed_tail_native_preferred(il)" in graph:
        raise AssertionError(
            "backend implementation preference must not veto a validated KVarN attention operation")
    if graph.count("!mctx_cur->has_kv_body(il)") < 4:
        raise AssertionError(
            "KVarN full/iSWA graph builders do not distinguish bodyless native tails")
    if "get_tail_body_execution_stride()" not in graph:
        raise AssertionError("attention input planning still derives packed-body extent from persistent rows")
    if "ggml_backend_dev_supports_op" not in graph:
        raise AssertionError("native KV-tail planning is not validated against the final fused operation")
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
        kvarn_cache.find("expected complete-head axis 1"),
        kvarn_cache.find("metadata->finalize_tail_overlay_metadata()"),
        kvarn_cache.find('ggml_format_name(layer.k_tail, "cache_kvarn_k_tail_l%d"'),
    ]
    if min(kvarn_order) < 0 or kvarn_order != sorted(kvarn_order):
        raise AssertionError("KVarN must validate its complete-head meta split before allocating tail metadata/tensors")
    if "realized body uses a tensor/meta split buffer" in kvarn_cache:
        raise AssertionError("KVarN still rejects a valid upstream tensor/meta owner")
    for required in ("ggml_flash_attn_ext_add_kv_tail", "ggml_kv_tail_attention_merge"):
        if required not in ggml_header:
            raise AssertionError(f"ggml tail-attention contract lacks {required}")
        if required not in ggml_core:
            raise AssertionError(f"ggml core does not implement {required}")
    if "ggml_kv_tail_attention_merge" not in graph:
        raise AssertionError("model graph does not use native tail attention")
    if "ggml_cuda_flash_attn_ext_tail" not in cuda_fattn:
        raise AssertionError("CUDA lacks the native tail-attention dispatch")

    for required in (
        "ggml_kv_tail_attention_merge_segmented",
        "ggml_flash_attn_ext_set_kv_tail_bodyless",
    ):
        if required not in ggml_header:
            raise AssertionError(f"ggml compact segmented contract lacks {required}")
    if not re.search(r"#define\s+GGML_MAX_SRC\s+12\b", ggml_header):
        raise AssertionError("ggml compact segmented contract requires 12 source operands")
    if "ggml_kv_tail_attention_merge_segmented" not in graph:
        raise AssertionError("model graph does not attach graph-local current K/V")

    vulkan = (ROOT / "ggml/src/ggml-vulkan/ggml-vulkan.cpp").read_text(encoding="utf-8")
    vulkan_fattn_support = vulkan.split("case GGML_OP_FLASH_ATTN_EXT:", 1)[1].split(
        "case GGML_OP_FLASH_ATTN_EXT_BACK:", 1
    )[0]
    if "ggml_vk_kvarn_attn_tail_sources_supported(op)" not in vulkan_fattn_support:
        raise AssertionError("Vulkan KVarN final-node support does not validate compact current operands")
    if "pipeline_flash_attn_tail" not in vulkan or "descriptor_offsets_ab" not in vulkan:
        raise AssertionError("Vulkan standard attention does not own compact-current operands and descriptor offsets")
    vulkan_proc = vulkan.split("static void * ggml_backend_vk_reg_get_proc_address", 1)[1]
    if "ggml_backend_kv_tail_segmented_attention_supported" not in vulkan_proc:
        raise AssertionError("Vulkan does not advertise its implemented segmented KVarN attention matrix")
    vulkan_shader = (
        ROOT / "ggml/src/ggml-vulkan/vulkan-shaders/kvarn_flash_attn.comp"
    ).read_text(encoding="utf-8")
    for required in (
        "k_tail_current_addr",
        "v_tail_current_addr",
        "FLAG_TAIL_HISTORY_SHIFT",
        "TailCurrentRef",
        "if (!bodyless)",
    ):
        if required not in vulkan_shader:
            raise AssertionError(f"Vulkan segmented KVarN shader lacks {required}")
    vulkan_materialize_shader = (
        ROOT / "ggml/src/ggml-vulkan/vulkan-shaders/kvarn_materialize.comp"
    ).read_text(encoding="utf-8")
    for required in (
        "MODE_PREPARE_LIVE",
        "binding = 4",
        "data_live",
    ):
        if required not in vulkan_materialize_shader:
            raise AssertionError(
                f"Vulkan KVarN materialization must prepare live state once: missing {required}"
            )
    materialize_dispatch = vulkan.split(
        "static void ggml_vk_kvarn_materialize(", 1
    )[1].split("static void ggml_vk_mul_mat(", 1)[0]
    for required in (
        "ggml_pipeline_request_descriptor_sets(ctx, pipeline, 2)",
        "ggml_vk_sync_buffers(ctx, subctx)",
        "ctx->prealloc_y_need_sync = true",
    ):
        if required not in materialize_dispatch:
            raise AssertionError(
                f"Vulkan KVarN materialization live descriptor dispatch lacks {required}"
            )
    if '"src10", "src11"' not in vulkan:
        raise AssertionError("Vulkan graph debugging does not cover the retained 12-source tensor contract")
    if graph.count("if (tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE)") < 2 or graph.count(
            "ggml_concat(ctx0, k_tail, k_tail_current, 2)") < 2:
        raise AssertionError("non-native backends lack the bounded history/current composition route")
    if graph.count(
            "tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE &&\n"
            "            !kvarn_plan.native_attention") != 2:
        raise AssertionError(
            "KVarN full/iSWA graphs do not fail closed to the generic tail oracle "
            "outside the backend's bounded native query matrix")

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
    if "GGML_USE_HIP" in tail_support or "return false" in tail_support:
        raise AssertionError("HIP must use the shared segmented-current capability path")
    direct_tail_support = kvarn_dispatch.split(
        "bool ggml_cuda_flash_attn_ext_kvarn_direct_tail_supported(", 2
    )[2].split("\n}\n", 1)[0]
    if (
        cuda_fattn.count("ggml_cuda_flash_attn_ext_kvarn_direct_tail_supported") != 2
        or "dst->src[10] == nullptr" not in direct_tail_support
        or "GGML_KVARN_TEST_FORCE_PORTABLE_FATTN" not in direct_tail_support
        or "ggml_cuda_flash_attn_ext_kvarn_portable_supported" not in direct_tail_support
        or "ggml_cuda_flash_attn_ext_tail(ctx, dst)" not in cuda_fattn
    ):
        raise AssertionError("HIP KVarN current segments lack the shared bounded tail fallback")
    cuda_tail = (ROOT / "ggml/src/ggml-cuda/fattn-tail.cuh").read_text(encoding="utf-8")
    if "k_flash_attn_ext_tail_merge" in cuda_tail:
        raise AssertionError("serialized indexed tail merge kernel is still present")

    graph_header = (ROOT / "src/llama-graph.h").read_text(encoding="utf-8")
    if "self_tail_bias_read_idxs" not in graph_header or "build_attn_bias_tail" not in graph:
        raise AssertionError("query-specific tails must gather matching body attention bias rows")
    if graph.count("storage_kind == LLAMA_KV_TAIL_STORAGE_DISABLED") < 2:
        raise AssertionError(
            "tail identity must skip per-layer graph-reuse work when tail storage is disabled"
        )

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

    model = (ROOT / "src/llama-model.cpp").read_text(encoding="utf-8")
    if model.count("if (params.kv_tail_native_exact)") < 2:
        raise AssertionError("full-window KVarN selection must use the logical native-exact policy")
    if model.count("params.kv_tail_native_exact ? cparams.n_ctx : 0") < 2:
        raise AssertionError("full-window KVarN storage must retain the logical visibility window")
    cache = (ROOT / "src/llama-kv-cache.cpp").read_text(encoding="utf-8")
    if cache.count("route_spec.body_type_k = candidate") != 1 or cache.count(
            "route_spec.body_type_v = candidate") != 1:
        raise AssertionError("bodyless standard-tail anchors must match the realized exact type")
    empty_body = graph.split("static void build_empty_kv_body", 1)[1].split(
        "static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl", 1)[0]
    if "first_row_anchor(body_mask_template)" not in empty_body:
        raise AssertionError(
            "bodyless standard-tail masks must retain a graph dependency on the authoritative planner mask")
    if "body_bias_template" not in empty_body or "first_row_anchor(body_bias_template)" not in empty_body:
        raise AssertionError(
            "bodyless standard-tail biases must retain a graph dependency on the authoritative planner bias")
    if "self_tail_query_order_swa" not in (ROOT / "src/llama-graph.h").read_text(encoding="utf-8"):
        raise AssertionError("full and SWA cache groups must own independent exact-tail query-order inputs")
    if "set_tail_query_plan(self_tail_query_order_swa, self_tail_run_desc_swa" not in graph:
        raise AssertionError("the SWA exact-tail planner must populate its own query-order input")

    hybrid_setter = graph.split("void llm_graph_input_mem_hybrid::set_input", 1)[1].split(
        "bool llm_graph_input_mem_hybrid::can_reuse", 1)[0]
    if "inp_attn->set_input(ubatch)" not in hybrid_setter:
        raise AssertionError("hybrid attention wrapper must delegate every input, including exact tails")

    bench = (ROOT / "tools/llama-bench/llama-bench.cpp").read_text(encoding="utf-8")
    if "bench_device_memory_checkpoint" not in bench or "ggml_backend_dev_memory" not in bench:
        raise AssertionError("--kv-memory must use a backend-generic device memory checkpoint")
    if "bench_memory_device" not in bench or "inst.devices" not in bench:
        raise AssertionError("--kv-memory must checkpoint the explicitly selected benchmark device")
    if "CUDA KV memory telemetry is unavailable" in bench:
        raise AssertionError("--kv-memory must not reject non-CUDA backends with memory telemetry")
    if "cuda_memory_checkpoint != nullptr" not in bench:
        raise AssertionError("the CUDA checkpoint must remain the preferred synchronized CUDA route")
    for required in (
        "struct_size = sizeof(stats)",
        "route_stats_abi_version",
        "generic_shape_rejected",
        '"kvarn_route_generic_rejected"',
        "cuda_context_buffer_bytes",
        "cuda_non_kv_context_buffer_bytes",
        "cuda_runtime_overhead_bytes",
        "ggml_backend_dev_backend_reg(memory_dev)",
        '"kvarn_route_portable"',
        '"kvarn_route_materialize"',
        '"kvarn_route_compact_tail"',
        "ggml_backend_kv_memory_transient_stats_reset",
        "ggml_backend_kv_memory_transient_stats_get",
    ):
        if required not in bench:
            raise AssertionError(
                f"llama-bench backend telemetry is not device-matched and version-safe: missing {required}"
            )
    for required in (
        "struct ggml_vk_kv_memory_transient_stats",
        "ggml_backend_vk_kv_memory_transient_stats_reset",
        "ggml_backend_vk_kv_memory_transient_stats_get",
        '"ggml_backend_kv_memory_transient_stats_reset"',
        '"ggml_backend_kv_memory_transient_stats_get"',
        "ggml_vk_kv_memory_transient_stats_record_kvarn",
    ):
        if required not in vulkan:
            raise AssertionError(
                f"Vulkan does not account backend-private KVarN transient memory: missing {required}"
            )
    vulkan_memory = vulkan.split(
        "void ggml_backend_vk_get_device_memory(", 1
    )[1].split("static vk::PhysicalDeviceType", 1)[0]
    if "budget > usage ? budget - usage : 0" not in vulkan_memory:
        raise AssertionError(
            "Vulkan memory-budget accounting can underflow when driver heap usage exceeds budget"
        )
    if "std::min<vk::DeviceSize>(available, heap.size)" not in vulkan_memory:
        raise AssertionError(
            "Vulkan memory-budget accounting can report more free memory than the physical heap"
        )


if __name__ == "__main__":
    main()
