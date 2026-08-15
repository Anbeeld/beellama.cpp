#include "llama-kv-cache-tail.h"
#include "llama-kv-cells.h"

#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <stdexcept>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

static llama_kv_tail_identity id(uint32_t cell, uint64_t generation = 1) {
    return { 0, cell, generation };
}

int main() {
    const auto empty_slot_runs = llama_kv_tail_contiguous_slot_runs({});
    CHECK(empty_slot_runs.empty());

    const auto one_slot_run = llama_kv_tail_contiguous_slot_runs({ 3 });
    CHECK(one_slot_run.size() == 1);
    CHECK(one_slot_run[0].payload_begin == 0);
    CHECK(one_slot_run[0].slot_begin == 3);
    CHECK(one_slot_run[0].length == 1);

    const auto contiguous_slot_runs = llama_kv_tail_contiguous_slot_runs({ 2, 3, 4, 5 });
    CHECK(contiguous_slot_runs.size() == 1);
    CHECK(contiguous_slot_runs[0].payload_begin == 0);
    CHECK(contiguous_slot_runs[0].slot_begin == 2);
    CHECK(contiguous_slot_runs[0].length == 4);

    const auto wrapped_slot_runs = llama_kv_tail_contiguous_slot_runs({ 6, 7, 0, 1 });
    CHECK(wrapped_slot_runs.size() == 2);
    CHECK(wrapped_slot_runs[0].payload_begin == 0);
    CHECK(wrapped_slot_runs[0].slot_begin == 6);
    CHECK(wrapped_slot_runs[0].length == 2);
    CHECK(wrapped_slot_runs[1].payload_begin == 2);
    CHECK(wrapped_slot_runs[1].slot_begin == 0);
    CHECK(wrapped_slot_runs[1].length == 2);

    const std::vector<int32_t> fragmented_slots = { 4, 5, 2, 3, 7 };
    const auto fragmented_slot_runs = llama_kv_tail_contiguous_slot_runs(fragmented_slots);
    CHECK(fragmented_slot_runs.size() == 3);
    CHECK(fragmented_slot_runs[0].payload_begin == 0);
    CHECK(fragmented_slot_runs[0].slot_begin == 4);
    CHECK(fragmented_slot_runs[0].length == 2);
    CHECK(fragmented_slot_runs[1].payload_begin == 2);
    CHECK(fragmented_slot_runs[1].slot_begin == 2);
    CHECK(fragmented_slot_runs[1].length == 2);
    CHECK(fragmented_slot_runs[2].payload_begin == 4);
    CHECK(fragmented_slot_runs[2].slot_begin == 7);
    CHECK(fragmented_slot_runs[2].length == 1);
    uint32_t fragmented_payloads = 0;
    for (const auto & run : fragmented_slot_runs) {
        CHECK(run.payload_begin == fragmented_payloads);
        CHECK(run.length > 0);
        fragmented_payloads += run.length;
    }
    CHECK(fragmented_payloads == fragmented_slots.size());

    // Layer split is an ownership mapping, not a device-zero default. Exercise
    // two logical owners even on a one-GPU host and include graph and state
    // consumers in the invariant.
    auto layer0 = llama_kv_tail_plan_layer_ownership(0, 101, 101, true, true);
    auto layer1 = llama_kv_tail_plan_layer_ownership(1, 202, 202, true, true);
    CHECK(llama_kv_tail_validate_layer_ownership(layer0) == LLAMA_KV_TAIL_OWNERSHIP_OK);
    CHECK(llama_kv_tail_validate_layer_ownership(layer1) == LLAMA_KV_TAIL_OWNERSHIP_OK);
    CHECK(layer0.graph_write_owner == 101 && layer0.graph_read_owner == 101);
    CHECK(layer1.graph_write_owner == 202 && layer1.graph_read_owner == 202);
    CHECK(layer0.state_k_owner == 101 && layer0.state_v_owner == 101);
    CHECK(layer1.state_k_owner == 202 && layer1.state_v_owner == 202);
    layer1.state_k_owner = 101;
    CHECK(llama_kv_tail_validate_layer_ownership(layer1) == LLAMA_KV_TAIL_OWNERSHIP_STATE_K);
    layer1.state_k_owner = 202;
    layer1.shadow_v_owner = 101;
    CHECK(llama_kv_tail_validate_layer_ownership(layer1) == LLAMA_KV_TAIL_OWNERSHIP_SHADOW_V);
    layer1.shadow_v_owner = 202;
    layer1.body_k_meta_split = true;
    layer1.body_v_meta_split = true;
    CHECK(llama_kv_tail_validate_layer_ownership(layer1) == LLAMA_KV_TAIL_OWNERSHIP_OK);

    llama_kv_tail_route_requirements route_requirements;
    route_requirements.native_attention = true;
    auto route = llama_kv_tail_select_route(route_requirements);
    CHECK(route.supported && route.route == LLAMA_KV_TAIL_ROUTE_NATIVE);

    // Explicit attention bias disables the native attached-tail operation;
    // when every primitive remains available the complete route is generic.
    route_requirements.native_attention = false;
    route = llama_kv_tail_select_route(route_requirements);
    CHECK(route.supported && route.route == LLAMA_KV_TAIL_ROUTE_GENERIC);

    // CANN classifies the fused shadow operand as unsupported. A complete
    // route must reject it rather than scheduling a full-body CPU transfer.
    route_requirements.write_k = false;
    route = llama_kv_tail_select_route(route_requirements);
    CHECK(!route.supported);
    CHECK(route.missing_operation == LLAMA_KV_TAIL_OP_WRITE_K);
    route_requirements.write_k = true;

    route_requirements.exact_value = false;
    route = llama_kv_tail_select_route(route_requirements);
    CHECK(!route.supported);
    CHECK(route.missing_operation == LLAMA_KV_TAIL_OP_EXACT_VALUE);
    route_requirements.exact_value = true;

    const llama_kv_tail_route_capability supported_native {
        true, LLAMA_KV_TAIL_ROUTE_NATIVE, LLAMA_KV_TAIL_OP_NONE,
    };
    const llama_kv_tail_route_capability bf16_missing_write {
        false, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_WRITE_K,
    };
    const llama_kv_tail_route_capability f16_missing_value {
        false, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_EXACT_VALUE,
    };
    auto type_resolution = llama_kv_tail_resolve_type(
            GGML_TYPE_COUNT, GGML_TYPE_BF16, supported_native, supported_native);
    CHECK(type_resolution.supported && type_resolution.actual_type == GGML_TYPE_BF16);
    CHECK(!type_resolution.downgraded);
    type_resolution = llama_kv_tail_resolve_type(
            GGML_TYPE_COUNT, GGML_TYPE_BF16, bf16_missing_write, supported_native);
    CHECK(type_resolution.supported && type_resolution.actual_type == GGML_TYPE_F16);
    CHECK(type_resolution.downgraded);
    CHECK(type_resolution.missing_preferred == LLAMA_KV_TAIL_OP_WRITE_K);
    type_resolution = llama_kv_tail_resolve_type(
            GGML_TYPE_BF16, GGML_TYPE_BF16, bf16_missing_write, supported_native);
    CHECK(!type_resolution.supported && type_resolution.actual_type == GGML_TYPE_COUNT);
    type_resolution = llama_kv_tail_resolve_type(
            GGML_TYPE_F16, GGML_TYPE_BF16, bf16_missing_write, supported_native);
    CHECK(type_resolution.supported && type_resolution.actual_type == GGML_TYPE_F16);
    CHECK(!type_resolution.downgraded);
    type_resolution = llama_kv_tail_resolve_type(
            GGML_TYPE_COUNT, GGML_TYPE_BF16, bf16_missing_write, f16_missing_value);
    CHECK(!type_resolution.supported);
    CHECK(type_resolution.missing_preferred == LLAMA_KV_TAIL_OP_WRITE_K);
    CHECK(type_resolution.missing_fallback == LLAMA_KV_TAIL_OP_EXACT_VALUE);

    assert(!llama_kv_tail_sparse_body_capacity_safe(0, 0));
    assert(llama_kv_tail_sparse_body_capacity_safe(255, 256));
    assert(llama_kv_tail_sparse_body_capacity_safe(256, 256));
    assert(!llama_kv_tail_sparse_body_capacity_safe(257, 256));

    CHECK(llama_kv_tail_packed_body_stride(0, 256) == 0);
    CHECK(llama_kv_tail_packed_body_stride(33, 256) == 256);
    CHECK(llama_kv_tail_packed_body_stride(256, 256) == 256);
    CHECK(llama_kv_tail_packed_body_stride(257, 256) == 512);
    // Automatic resolution is architecture-agnostic for standard KV groups:
    // every applicable group requests 1024 and is capped by its own window.
    std::vector<llama_kv_tail_group_request> group_requests = {
        { 0, 65536, true },
    };
    auto resolved_groups = llama_kv_tail_resolve_groups(true, true, group_requests);
    CHECK(resolved_groups.valid);
    CHECK(resolved_groups.tokens == std::vector<uint32_t>({ 1024 }));

    group_requests = {
        { 0, 16384, true },
        { 0, 1024,  true },
    };
    resolved_groups = llama_kv_tail_resolve_groups(true, true, group_requests);
    CHECK(resolved_groups.tokens == std::vector<uint32_t>({ 1024, 1024 }));

    group_requests = {
        { 0, 384, true },       // small standard context
        { 0, 8192, false },     // recurrent/raw-special/non-standard group
    };
    resolved_groups = llama_kv_tail_resolve_groups(true, true, group_requests);
    CHECK(resolved_groups.tokens == std::vector<uint32_t>({ 384, 0 }));

    // Unknown model architecture does not matter when the group is ordinary
    // standard KV; applicability, not a model-name table, is authoritative.
    group_requests = { { 0, 2048, true } };
    resolved_groups = llama_kv_tail_resolve_groups(true, true, group_requests);
    CHECK(resolved_groups.tokens == std::vector<uint32_t>({ 1024 }));

    // Explicit values bypass auto while retaining per-group caps.
    group_requests = { { 1536, 4096, true }, { 2048, 512, true } };
    resolved_groups = llama_kv_tail_resolve_groups(false, true, group_requests);
    CHECK(resolved_groups.valid);
    CHECK(resolved_groups.tokens == std::vector<uint32_t>({ 1536, 512 }));

    // An incomplete explicit specification fails atomically; it cannot leave
    // one group enabled from a partially applied request.
    resolved_groups = llama_kv_tail_resolve_groups(false, false, group_requests);
    CHECK(!resolved_groups.valid);
    CHECK(resolved_groups.tokens == std::vector<uint32_t>({ 0, 0 }));

    // Single-sequence decode never has ragged sequence-membership levels, so
    // every write receives an arena destination and needs no discard sink.
    // Multi-sequence batches retain one ubatch-sized sink for absent levels.
    const auto single_layout = llama_kv_tail_layout_for(1024, 1, 512);
    CHECK(single_layout.arena_stride == 1536);
    CHECK(single_layout.sink_slots == 0);
    CHECK(single_layout.total_slots == 1536);

    const auto multi_layout = llama_kv_tail_layout_for(1024, 4, 512);
    CHECK(multi_layout.arena_stride == 1536);
    CHECK(multi_layout.sink_slots == 512);
    CHECK(multi_layout.total_slots == 6656);

    // Compact persistent history depends only on logical exact coverage and
    // the promised rollback horizon. The active ubatch affects descriptor
    // workspace, not per-layer payload rows.
    const auto compact_layout_128 = llama_kv_tail_compact_layout_for(128, 8, 4, 128);
    const auto compact_layout_512 = llama_kv_tail_compact_layout_for(128, 8, 4, 512);
    CHECK(compact_layout_128.history_stride == 136);
    CHECK(compact_layout_128.history_slots == 544);
    CHECK(compact_layout_128.rollback_tokens == 8);
    CHECK(compact_layout_128.attention_stride == 256);
    CHECK(compact_layout_512.history_stride == compact_layout_128.history_stride);
    CHECK(compact_layout_512.history_slots == compact_layout_128.history_slots);
    CHECK(compact_layout_512.attention_stride == 640);

    CHECK(llama_kv_tail_can_remove_suffix(-1, 0, -1, 8));
    CHECK(llama_kv_tail_can_remove_suffix(6, 7, -1, 8));
    CHECK(llama_kv_tail_can_remove_suffix(6, 6, -1, 8));
    CHECK(llama_kv_tail_can_remove_suffix(6, 0, -1, 8));
    CHECK(!llama_kv_tail_can_remove_suffix(15, 6, -1, 8));
    CHECK(!llama_kv_tail_can_remove_suffix(6, 6, 7, 8));

    // Representation is selected from visibility, ownership, and aggregate
    // byte cost rather than from a model or cache-role special case.
    const auto storage_request = [](
            uint32_t n_tokens,
            uint32_t visibility_window,
            uint64_t physical_body_rows,
            uint64_t promotion_bytes_per_row,
            uint64_t overlay_bytes_per_row,
            uint64_t requested_body_bytes_per_row = 5632,
            bool native_capable = true,
            bool already_exact = false,
            bool has_owned_body = true,
            bool has_shared_body = false,
            bool shadow_k_capable = true,
            bool shadow_v_capable = true,
            ggml_type body_type_k = GGML_TYPE_Q5_0,
            ggml_type body_type_v = GGML_TYPE_Q5_0) {
        return llama_kv_tail_storage_request {
            body_type_k, body_type_v, GGML_TYPE_BF16,
            n_tokens, n_tokens, 1, 512, visibility_window,
            physical_body_rows, requested_body_bytes_per_row,
            promotion_bytes_per_row, overlay_bytes_per_row,
            native_capable, already_exact, has_owned_body, has_shared_body,
            shadow_k_capable, shadow_v_capable,
        };
    };

    auto storage = llama_kv_tail_storage_plan_for(storage_request(0, 1024, 1536, 10752, 16384));
    CHECK(storage.kind == LLAMA_KV_TAIL_STORAGE_DISABLED);
    CHECK(storage.requested_tokens == 0);
    CHECK(storage.effective_tokens == 0);
    CHECK(storage.visibility_window == 1024);
    CHECK(storage.requested_body_type_k == GGML_TYPE_Q5_0);
    CHECK(storage.requested_body_type_v == GGML_TYPE_Q5_0);
    CHECK(storage.actual_body_type_k == GGML_TYPE_Q5_0);
    CHECK(storage.actual_body_type_v == GGML_TYPE_Q5_0);
    CHECK(!storage.shadow_k && !storage.shadow_v);
    CHECK(storage.physical_body_rows == 1536);
    CHECK(storage.layout.total_slots == 0);

    storage = llama_kv_tail_storage_plan_for(storage_request(512, 1024, 1536, 10752, 16384));
    CHECK(storage.kind == LLAMA_KV_TAIL_STORAGE_OVERLAY);
    CHECK(storage.requested_tokens == 512);
    CHECK(storage.effective_tokens == 512);
    CHECK(storage.visibility_window == 1024);
    CHECK(storage.shadow_k && storage.shadow_v);
    CHECK(storage.shadow_type_k == GGML_TYPE_BF16);
    CHECK(storage.shadow_type_v == GGML_TYPE_BF16);
    CHECK(storage.layout.arena_stride == 1024);
    CHECK(storage.layout.sink_slots == 0);
    CHECK(storage.requested_body_bytes == uint64_t(1536)*5632);
    CHECK(storage.actual_body_bytes == storage.requested_body_bytes);
    CHECK(storage.shadow_bytes == uint64_t(1024)*16384);
    CHECK(storage.promotion_increment == uint64_t(1536)*10752);
    CHECK(storage.overlay_increment == uint64_t(1024)*16384);
    CHECK(storage.has_owned_body);
    CHECK(!storage.has_shared_body);

    // Gemma's compact SWA topology: 1536 promoted rows cost less than a
    // 1536-row symmetric overlay and the requested span covers all visibility.
    storage = llama_kv_tail_storage_plan_for(storage_request(1024, 1024, 1536, 10752, 16384));
    CHECK(storage.kind == LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT);
    CHECK(storage.body_promoted);
    CHECK(storage.actual_body_type_k == GGML_TYPE_BF16);
    CHECK(storage.actual_body_type_v == GGML_TYPE_BF16);
    CHECK(!storage.shadow_k && !storage.shadow_v);
    CHECK(storage.actual_body_bytes == storage.requested_body_bytes + storage.promotion_increment);
    CHECK(storage.shadow_bytes == 0);
    CHECK(storage.promotion_increment == uint64_t(1536)*10752);
    CHECK(storage.overlay_increment == uint64_t(1536)*16384);

    // A full-sized SWA body is eligible by visibility but not by byte cost.
    storage = llama_kv_tail_storage_plan_for(storage_request(1024, 1024, 16384, 10752, 16384));
    CHECK(storage.kind == LLAMA_KV_TAIL_STORAGE_OVERLAY);

    auto compact_partial = storage_request(128, 1024, 1536, 10752, 16384);
    compact_partial.rollback_tokens = 8;
    compact_partial.compact_history_capable = true;
    compact_partial.compact_current_source_capable = true;
    compact_partial.compact_ordered_commit_capable = true;
    compact_partial.full_window_body_can_be_omitted = true;
    storage = llama_kv_tail_storage_plan_for(compact_partial);
    CHECK(storage.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_OVERLAY);
    CHECK(storage.compact_layout.history_stride == 136);
    CHECK(storage.compact_layout.history_slots == 136);
    CHECK(storage.compact_layout.rollback_tokens == 8);
    CHECK(storage.compact_history_bytes == uint64_t(136)*16384);
    CHECK(storage.compact_rollback_bytes == uint64_t(8)*16384);
    CHECK(storage.physical_body_rows == 1536);

    auto compact_partial_larger_ubatch = compact_partial;
    compact_partial_larger_ubatch.n_ubatch = 1024;
    const auto compact_larger_ubatch_plan = llama_kv_tail_storage_plan_for(compact_partial_larger_ubatch);
    CHECK(compact_larger_ubatch_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_OVERLAY);
    CHECK(compact_larger_ubatch_plan.compact_layout.history_slots == storage.compact_layout.history_slots);
    CHECK(compact_larger_ubatch_plan.compact_history_bytes == storage.compact_history_bytes);
    CHECK(compact_larger_ubatch_plan.compact_layout.attention_stride == 1152);

    auto compact_full = storage_request(1024, 1024, 1536, 10752, 16384);
    compact_full.rollback_tokens = 8;
    compact_full.compact_history_capable = true;
    compact_full.compact_current_source_capable = true;
    compact_full.compact_ordered_commit_capable = true;
    compact_full.full_window_body_can_be_omitted = true;
    storage = llama_kv_tail_storage_plan_for(compact_full);
    CHECK(storage.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT);
    CHECK(storage.compact_layout.history_stride == 1032);
    CHECK(storage.compact_layout.history_slots == 1032);
    CHECK(storage.compact_history_bytes == uint64_t(1032)*16384);
    CHECK(storage.actual_body_bytes == 0);
    CHECK(!storage.has_owned_body);

    auto graph_rejected = storage_request(512, 1024, 1536, 10752, 16384);
    graph_rejected.graph_consumes_exact_tail = false;
    graph_rejected.architecture = "deepseek2";
    graph_rejected.group_id = "mla@l0";
    bool graph_rejected_before_allocation = false;
    try {
        (void) llama_kv_tail_storage_plan_for(graph_rejected);
    } catch (const std::invalid_argument & error) {
        const std::string message = error.what();
        graph_rejected_before_allocation =
                message.find("deepseek2") != std::string::npos &&
                message.find("mla@l0") != std::string::npos &&
                message.find("requested N=512") != std::string::npos &&
                message.find("K-only MLA/DSA") != std::string::npos;
    }
    CHECK(graph_rejected_before_allocation);

    auto dsa_rejected = graph_rejected;
    dsa_rejected.architecture = "deepseek4";
    dsa_rejected.group_id = "dsa-raw@l0";
    bool dsa_rejected_before_allocation = false;
    try {
        (void) llama_kv_tail_storage_plan_for(dsa_rejected);
    } catch (const std::invalid_argument & error) {
        const std::string message = error.what();
        dsa_rejected_before_allocation =
                message.find("deepseek4") != std::string::npos &&
                message.find("dsa-raw@l0") != std::string::npos &&
                message.find("requested N=512") != std::string::npos &&
                message.find("K-only MLA/DSA") != std::string::npos;
    }
    CHECK(dsa_rejected_before_allocation);

    auto placement_rejected = storage_request(512, 1024, 1536, 10752, 16384);
    placement_rejected.overlay_placement_supported = false;
    bool placement_rejected_before_allocation = false;
    try {
        (void) llama_kv_tail_storage_plan_for(placement_rejected);
    } catch (const std::invalid_argument &) {
        placement_rejected_before_allocation = true;
    }
    CHECK(placement_rejected_before_allocation);

    auto native_without_overlay_consumer = storage_request(1024, 1024, 1536, 10752, 16384);
    native_without_overlay_consumer.graph_consumes_exact_tail = false;
    native_without_overlay_consumer.overlay_placement_supported = false;
    storage = llama_kv_tail_storage_plan_for(native_without_overlay_consumer);
    CHECK(storage.kind == LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT);

    // A shared quantized source cannot be promoted by the child context.
    storage = llama_kv_tail_storage_plan_for(storage_request(
            1024, 1024, 1536, 10752, 16384, 5632, false, false, false, true));
    CHECK(storage.kind == LLAMA_KV_TAIL_STORAGE_OVERLAY);
    CHECK(!storage.has_owned_body && storage.has_shared_body);

    // An already-exact body satisfies any requested suffix without an overlay.
    auto already_exact_compact_capable = storage_request(
            512, 1024, 1536, 0, 0, 16384, true, true, true, false,
            true, true, GGML_TYPE_BF16, GGML_TYPE_F16);
    already_exact_compact_capable.compact_history_capable = true;
    already_exact_compact_capable.compact_current_source_capable = true;
    already_exact_compact_capable.compact_ordered_commit_capable = true;
    storage = llama_kv_tail_storage_plan_for(already_exact_compact_capable);
    CHECK(storage.kind == LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT);
    CHECK(!storage.body_promoted);
    CHECK(storage.actual_body_type_k == GGML_TYPE_BF16);
    CHECK(storage.actual_body_type_v == GGML_TYPE_F16);

    // One-sided promotion preserves the side which was already exact.
    storage = llama_kv_tail_storage_plan_for(storage_request(
            1024, 1024, 1536, 4096, 8192, 12288, true, false, true, false,
            true, true, GGML_TYPE_Q5_0, GGML_TYPE_F16));
    CHECK(storage.kind == LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT);
    CHECK(storage.actual_body_type_k == GGML_TYPE_BF16);
    CHECK(storage.actual_body_type_v == GGML_TYPE_F16);

    // An already-exact F32 side remains F32 in a mixed overlay. It is an
    // aligned shadow, not a reason to leave one half of the exact pair absent.
    storage = llama_kv_tail_storage_plan_for(storage_request(
            512, 1024, 1536, 4096, 16384, 12288, true, false, true, false,
            true, true, GGML_TYPE_Q5_0, GGML_TYPE_F32));
    CHECK(storage.kind == LLAMA_KV_TAIL_STORAGE_OVERLAY);
    CHECK(storage.shadow_k && storage.shadow_v);
    CHECK(storage.shadow_type_k == GGML_TYPE_BF16);
    CHECK(storage.shadow_type_v == GGML_TYPE_F32);

    storage = llama_kv_tail_storage_plan_for(storage_request(
            512, 1024, 1536, 4096, 16384, 12288, true, false, true, false,
            true, true, GGML_TYPE_F32, GGML_TYPE_Q5_0));
    CHECK(storage.kind == LLAMA_KV_TAIL_STORAGE_OVERLAY);
    CHECK(storage.shadow_k && storage.shadow_v);
    CHECK(storage.shadow_type_k == GGML_TYPE_F32);
    CHECK(storage.shadow_type_v == GGML_TYPE_BF16);

    // Explicit selection never turns an invisible gap into a suffix. Causal
    // mode also excludes rollback records newer than the query; non-causal
    // mode may select them when the authoritative mask is finite.
    const std::vector<llama_kv_tail_mask_entry> mask_entries = {
        { 0, 10, 10 }, { 1, 11, 11 }, { 2, 12, 12 },
        { 3, 13, 13 }, { 4, 14, 14 },
    };
    const std::vector<uint8_t> finite = { 1, 0, 1, 0, 1 };
    std::vector<uint32_t> selected;
    llama_kv_tail_select_masked_entries(mask_entries, finite, 13, 2, true, selected);
    CHECK(selected == std::vector<uint32_t>({ 0, 2 }));
    llama_kv_tail_select_masked_entries(mask_entries, finite, 13, 2, false, selected);
    CHECK(selected == std::vector<uint32_t>({ 2, 4 }));
    llama_kv_tail_select_masked_entries(mask_entries, finite, 13, 0, false, selected);
    CHECK(selected.empty());

    const auto make_body_rows = []() {
        std::vector<llama_kv_tail_body_row> rows;
        for (uint32_t cell = 0; cell < 16; ++cell) {
            rows.push_back({ llama_pos(cell), cell, int32_t(cell) });
        }
        return rows;
    };
    const std::vector<llama_kv_tail_query_window> standard_queries = {
        { 0, 3 }, { 1, 7 },
    };
    auto standard_rows = make_body_rows();
    std::vector<uint8_t> row_scratch;
    llama_kv_tail_union_swa_rows(
            standard_rows, standard_queries, 4, LLAMA_SWA_TYPE_STANDARD, true,
            [](uint32_t query, uint32_t cell) {
                return !(query == 0 && cell == 2);
            }, row_scratch);
    std::vector<uint32_t> standard_cells;
    for (const auto & row : standard_rows) {
        standard_cells.push_back(row.cell);
    }
    CHECK(standard_cells == std::vector<uint32_t>({ 0, 1, 3, 4, 5, 6, 7 }));

    const std::vector<llama_kv_tail_query_window> chunked_queries = {
        { 0, 5 }, { 1, 7 },
    };
    auto chunked_rows = make_body_rows();
    llama_kv_tail_union_swa_rows(
            chunked_rows, chunked_queries, 4, LLAMA_SWA_TYPE_CHUNKED, true,
            [](uint32_t, uint32_t) { return true; }, row_scratch);
    std::vector<uint32_t> chunked_cells;
    for (const auto & row : chunked_rows) {
        chunked_cells.push_back(row.cell);
    }
    CHECK(chunked_cells == std::vector<uint32_t>({ 4, 5, 6, 7 }));

    auto noncausal_rows = make_body_rows();
    llama_kv_tail_union_swa_rows(
            noncausal_rows, { { 0, 3 } }, 4, LLAMA_SWA_TYPE_STANDARD, false,
            [](uint32_t, uint32_t) { return true; }, row_scratch);
    CHECK(noncausal_rows.size() == 16);

    // Planning the same query run as one ubatch or two partitions produces the
    // same packed-row union for both standard and chunked windows.
    const auto check_partition_invariance = [&](llama_swa_type type,
                                                 const std::vector<llama_kv_tail_query_window> & queries) {
        auto combined = make_body_rows();
        llama_kv_tail_union_swa_rows(combined, queries, 4, type, true,
                [](uint32_t, uint32_t) { return true; }, row_scratch);
        std::vector<bool> partition_union(16, false);
        for (const auto & query : queries) {
            auto partition = make_body_rows();
            llama_kv_tail_union_swa_rows(partition, { query }, 4, type, true,
                    [](uint32_t, uint32_t) { return true; }, row_scratch);
            for (const auto & row : partition) {
                partition_union[row.cell] = true;
            }
        }
        std::vector<uint32_t> partition_cells;
        for (uint32_t cell = 0; cell < partition_union.size(); ++cell) {
            if (partition_union[cell]) {
                partition_cells.push_back(cell);
            }
        }
        std::vector<uint32_t> combined_cells;
        for (const auto & row : combined) {
            combined_cells.push_back(row.cell);
        }
        CHECK(combined_cells == partition_cells);
    };
    check_partition_invariance(LLAMA_SWA_TYPE_STANDARD, standard_queries);
    check_partition_invariance(LLAMA_SWA_TYPE_CHUNKED, chunked_queries);

    // A structurally empty group cannot allocate or advertise exact coverage.
    storage = llama_kv_tail_storage_plan_for(storage_request(
            1024, 1024, 0, 0, 0, 0, false, false, false, false, false, false));
    CHECK(storage.kind == LLAMA_KV_TAIL_STORAGE_DISABLED);
    CHECK(storage.requested_tokens == 1024);
    CHECK(storage.effective_tokens == 0);

    // The plan preserves the uncapped request separately from its effective span.
    auto capped = storage_request(1024, 1024, 1536, 10752, 16384);
    capped.requested_tokens = 4096;
    storage = llama_kv_tail_storage_plan_for(capped);
    CHECK(storage.requested_tokens == 4096);
    CHECK(storage.effective_tokens == 1024);

    // Equal persistent cost chooses the simpler native graph.
    storage = llama_kv_tail_storage_plan_for(storage_request(1024, 1024, 1536, 4096, 4096));
    CHECK(storage.kind == LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT);

    bool layout_overflow = false;
    try {
        (void) llama_kv_tail_layout_for(std::numeric_limits<uint32_t>::max(), 1, 512);
    } catch (const std::overflow_error &) {
        layout_overflow = true;
    }
    CHECK(layout_overflow);

    bool storage_overflow = false;
    try {
        (void) llama_kv_tail_storage_plan_for(storage_request(
                1024, 1024, std::numeric_limits<uint64_t>::max(), 2, 1));
    } catch (const std::overflow_error &) {
        storage_overflow = true;
    }
    CHECK(storage_overflow);

    // Incremental per-sequence live-cell accounting covers membership,
    // removal, keep, shift eviction, and reset without a cache scan.
    llama_kv_cells cells;
    cells.resize(4);
    CHECK(cells.seq_size(0) == 0);
    cells.pos_set(0, 10);
    cells.seq_add(0, 0);
    cells.pos_set(1, 11);
    cells.seq_add(1, 0);
    cells.seq_add(1, 1);
    CHECK(cells.seq_size(0) == 2);
    CHECK(cells.seq_size(1) == 1);
    CHECK(!cells.seq_rm(1, 0));
    CHECK(cells.seq_size(0) == 1);
    CHECK(cells.seq_size(1) == 1);
    CHECK(!cells.seq_keep(0, 0));
    CHECK(cells.seq_size(0) == 1);
    CHECK(cells.pos_add(0, -20));
    CHECK(cells.seq_size(0) == 0);
    cells.rm(1);
    CHECK(cells.seq_size(1) == 0);
    cells.reset();
    CHECK(cells.seq_size(0) == 0 && cells.seq_size(1) == 0);

    // Arena ownership is per sequence: identical physical-cache identities in
    // two sequence memberships still receive distinct exact payload rows.
    llama_kv_tail_store arenas(2, 2, 8);
    const int32_t arena0_slot = arenas.commit(0, id(40), 0, 0);
    const int32_t arena1_slot = arenas.commit(1, id(40), 0, 1);
    CHECK(arena0_slot >= 0 && arena0_slot < 4);
    CHECK(arena1_slot >= 4 && arena1_slot < 8);
    CHECK(arena0_slot != arena1_slot);

    // Overflow replaces the true recency victim in place. It must not consume
    // another arena row and defer the release until after assignment.
    llama_kv_tail_store victim(2, 1, 4);
    const int32_t victim0 = victim.commit(0, id(41), 10, 0);
    victim.commit(0, id(42), 20, 1);
    const int32_t replacement = victim.commit(0, id(43), 30, 2);
    CHECK(replacement == victim0);

    // Per-sequence arenas require payload copies even when both logical
    // sequences share the same unified-cache stream.
    llama_kv_tail_store same_stream_copy(2, 2, 8);
    same_stream_copy.commit(0, id(44), 10, 0);
    same_stream_copy.commit(0, id(45), 11, 1);
    const auto same_stream_remap = same_stream_copy.seq_cp_remap(0, 1, 0, 0, 0, -1);
    CHECK(same_stream_remap.size() == 2);
    CHECK(same_stream_remap[0].src_slot != same_stream_remap[0].dst_slot);

    // A partial copy into a full destination preselects the newest N records
    // before allocating slots. This remains valid when R < 2N and scales with
    // retained metadata rather than per-layer payload bytes.
    constexpr uint32_t copy_n = 2048;
    llama_kv_tail_store full_copy(copy_n, 2, 4608); // R=2304 < 2N per sequence.
    for (uint32_t i = 0; i < copy_n; ++i) {
        full_copy.commit(0, { 0, i, 1 }, 1024 + int32_t(i), i);
        full_copy.commit(1, { 0, copy_n + i, 1 }, int32_t(i), copy_n + i);
    }
    const auto copy_start = std::chrono::steady_clock::now();
    const auto full_remap = full_copy.seq_cp_remap(0, 1, 0, 0, 1024, 3072);
    const auto copy_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - copy_start).count();
    CHECK(full_copy.coverage(1, copy_n).exact == copy_n);
    CHECK(full_remap.size() == copy_n);
    CHECK(copy_ms < 50.0);

    llama_kv_tail_store tail(3, 2, 12);

    tail.commit(0, id(0), 0, 0);
    tail.commit(0, id(1), 1, 1);
    tail.commit(0, id(2), 2, 2);
    tail.commit(0, id(3), 3, 3);

    const auto plan = tail.build_source_plan(0, { id(0), id(1), id(2), id(3) });
    CHECK(plan.size() == 4);
    CHECK(plan[0] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(plan[1] >= 0 && plan[2] >= 0 && plan[3] >= 0);
    CHECK(tail.coverage(0).state == LLAMA_KV_TAIL_COVERAGE_COMPLETE);
    CHECK(tail.coverage(0).requested == 3);
    CHECK(tail.coverage(0).exact == 3);

    // A branch shares direct exact shadows, but cannot manufacture an evicted one.
    tail.seq_cp(0, 1, 0, 4);
    const auto copied = tail.build_source_plan(1, { id(0), id(1), id(2), id(3) });
    CHECK(copied[0] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(copied[1] >= 0 && copied[2] >= 0 && copied[3] >= 0);

    // Recycling a physical cell invalidates the old generation for every sequence.
    tail.recycle(0, 3, 2);
    const auto stale = tail.build_source_plan(1, { id(1), id(2), id(3) });
    CHECK(stale[2] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(tail.coverage(1).state == LLAMA_KV_TAIL_COVERAGE_PARTIAL);

    tail.commit(0, id(3, 2), 4, 4);
    const auto refreshed = tail.build_source_plan(0, { id(1), id(2), id(3, 2) });
    CHECK(refreshed[0] >= 0 && refreshed[1] >= 0 && refreshed[2] >= 0);

    // Query-local selection never expands merely because more entries are in flight.
    tail.commit(0, id(4), 5, 5);
    tail.commit(0, id(5), 6, 6);
    const auto early = tail.build_source_plan(0, { id(0), id(1), id(2) });
    CHECK(early[0] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(early[1] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(early[2] == LLAMA_KV_TAIL_BODY_SLOT);

    tail.seq_rm(0, 4, 7);
    const auto after_rm = tail.build_source_plan(0, { id(1), id(2) });
    CHECK(after_rm[0] == LLAMA_KV_TAIL_BODY_SLOT && after_rm[1] == LLAMA_KV_TAIL_BODY_SLOT);

    // Cross-stream copies receive new physical identities and payload slots.
    // The returned slot pairs tell the cache which exact rows to copy.
    llama_kv_tail_store cross(2, 2, 8);
    cross.commit(0, { 0, 4, 7 }, 10, 10);
    cross.commit(0, { 0, 5, 8 }, 11, 11);
    const auto remap = cross.seq_cp_remap(0, 1, 0, 1, 0, -1);
    CHECK(remap.size() == 2);
    const auto cross_plan = cross.build_source_plan(1, { { 1, 4, 7 }, { 1, 5, 8 } });
    CHECK(cross_plan[0] >= 0 && cross_plan[1] >= 0);
    CHECK(remap[0].src_slot != remap[0].dst_slot);

    // A failed payload transaction must invalidate every possibly partial
    // destination instead of publishing stale exact rows.
    std::vector<int32_t> failed_dst;
    for (const auto & copy : remap) {
        failed_dst.push_back(copy.dst_slot);
    }
    cross.invalidate_slots(failed_dst, LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
    const auto failed_plan = cross.build_source_plan(1, { { 1, 4, 7 }, { 1, 5, 8 } });
    CHECK(failed_plan[0] == LLAMA_KV_TAIL_BODY_SLOT && failed_plan[1] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(cross.coverage(1, 2).degradation_flags & LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);

    // Rebuild the copied state for the remaining cell-local removal checks.
    (void) cross.seq_cp_remap(0, 1, 0, 1, 0, -1);

    // Cell-local removal must preserve another sequence's shared exact row.
    cross.seq_rm_cell(1, 1, 4);
    CHECK(cross.build_source_plan(1, { { 1, 4, 7 }, { 1, 5, 8 } })[0] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(cross.build_source_plan(0, { { 0, 4, 7 }, { 0, 5, 8 } })[0] >= 0);

    cross.mark_degraded(0, LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
    CHECK(cross.coverage(0).state == LLAMA_KV_TAIL_COVERAGE_PARTIAL);
    CHECK(cross.coverage(0).degradation_flags & LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
    cross.commit(0, { 0, 6, 9 }, 12, 12);
    cross.commit(0, { 0, 7, 10 }, 13, 13);
    CHECK(cross.coverage(0).state == LLAMA_KV_TAIL_COVERAGE_COMPLETE);

    // A partial sequence copy must combine degradation history. It must not
    // clear an existing destination reason merely because the source is clean.
    llama_kv_tail_store degraded_copy(2, 2, 8);
    degraded_copy.commit(0, id(0), 0, 0);
    degraded_copy.commit(0, id(1), 1, 1);
    degraded_copy.commit(1, id(2), 2, 2);
    degraded_copy.mark_degraded(1, LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE);
    degraded_copy.seq_cp(0, 1, 0, 1);
    CHECK(degraded_copy.coverage(1).degradation_flags & LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE);
    degraded_copy.mark_degraded(0, LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
    degraded_copy.seq_cp(0, 1, 0, 1);
    CHECK(degraded_copy.coverage(1).degradation_flags & LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE);
    CHECK(degraded_copy.coverage(1).degradation_flags & LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);

    // State v2 must preserve both the degradation cause and partial recovery
    // progress. Whole restores replace every sequence, while a sequence restore
    // remaps only the selected provenance entry to its destination.
    llama_kv_tail_store provenance_source(2, 2, 8);
    provenance_source.mark_degraded(0, LLAMA_KV_TAIL_DEGRADED_STATE_RESTORE);
    provenance_source.commit(0, id(6), 6, 6);
    const auto saved_provenance = provenance_source.snapshot_provenance();
    CHECK(saved_provenance.size() == 2);
    CHECK(saved_provenance[0].seq_id == 0);
    CHECK(saved_provenance[0].degradation_flags == LLAMA_KV_TAIL_DEGRADED_STATE_RESTORE);
    CHECK(saved_provenance[0].recovery_commits == 1);

    llama_kv_tail_store provenance_restored(2, 2, 8);
    provenance_restored.mark_degraded(1, LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE);
    provenance_restored.restore_provenance(saved_provenance);
    CHECK(provenance_restored.coverage(0).degradation_flags == LLAMA_KV_TAIL_DEGRADED_STATE_RESTORE);
    CHECK(provenance_restored.coverage(1).degradation_flags == 0);
    provenance_restored.commit(0, id(7), 7, 7);
    CHECK(provenance_restored.coverage(0).degradation_flags == 0);

    provenance_restored.mark_degraded(0, LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
    provenance_restored.restore_provenance(provenance_source.snapshot_provenance(0), 1);
    CHECK(provenance_restored.coverage(0).degradation_flags == LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
    CHECK(provenance_restored.coverage(1).degradation_flags == LLAMA_KV_TAIL_DEGRADED_STATE_RESTORE);

    // Deferred copies reserve private destination rows but do not publish
    // metadata until the payload transaction commits. Coverage and state-size
    // snapshots still describe the logical post-copy destination.
    llama_kv_tail_store transactional_copy(2, 2, 8);
    const int32_t source_slot = transactional_copy.commit(0, id(0), 0, 0);
    const int32_t old_destination_slot = transactional_copy.commit(1, id(4), 4, 4);
    const auto prepared = transactional_copy.prepare_seq_cp(0, 1, 0, 0, 0, -1);
    CHECK(transactional_copy.has_pending_seq_cp());
    CHECK(prepared.size() == 1 && prepared[0].src_slot == source_slot);
    CHECK(transactional_copy.build_source_plan(1, { id(4) })[0] == old_destination_slot);
    CHECK(transactional_copy.build_source_plan(1, { id(0) })[0] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(transactional_copy.coverage(1, 1).exact == 1);
    const auto pending_snapshot = transactional_copy.snapshot(1);
    CHECK(pending_snapshot.size() == 1 && pending_snapshot[0].identity == id(0));
    transactional_copy.commit_seq_cp();
    CHECK(!transactional_copy.has_pending_seq_cp());
    CHECK(transactional_copy.build_source_plan(1, { id(0) })[0] == prepared[0].dst_slot);

    (void) transactional_copy.prepare_seq_cp(0, 1, 0, 0, 0, -1);
    transactional_copy.cancel_seq_cp();
    CHECK(!transactional_copy.has_pending_seq_cp());
    CHECK(transactional_copy.build_source_plan(1, { id(0) })[0] == prepared[0].dst_slot);

    llama_kv_tail_store shared_transform(4, 2, 12);
    const llama_kv_tail_identity shared_id { 0, 5, 9 };
    shared_transform.commit(0, shared_id, 4, 4);
    shared_transform.commit(1, shared_id, 4, 4);
    shared_transform.seq_add(0, 4, 5, 2);
    CHECK(shared_transform.snapshot(0)[0].position == 6);
    CHECK(shared_transform.snapshot(1)[0].position == 6);
    CHECK(shared_transform.coverage(0).degradation_flags & LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
    CHECK(shared_transform.coverage(1).degradation_flags & LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
    shared_transform.seq_div(0, 6, 7, 2);
    CHECK(shared_transform.snapshot(0)[0].position == 3);
    CHECK(shared_transform.snapshot(1)[0].position == 3);

    // The rollback reserve keeps every query-local row in the current physical
    // ubatch until the next ubatch boundary.
    llama_kv_tail_store rollback(2, 1, 5);
    rollback.commit(0, id(0), 0, 0);
    rollback.commit(0, id(1), 1, 1);
    rollback.begin_batch();
    rollback.commit(0, id(2), 2, 2);
    rollback.commit(0, id(3), 3, 3);
    rollback.commit(0, id(4), 4, 4);
    const auto rollback_candidates = rollback.source_candidates(0);
    CHECK(rollback_candidates.size() == 5);
    for (size_t i = 0; i < rollback_candidates.size(); ++i) {
        CHECK(rollback_candidates[i].identity == id(uint32_t(i)));
    }
    const auto rollback_runs = rollback.source_runs(0);
    CHECK(rollback_runs.size() == 1);
    CHECK(rollback_runs[0].exact_offset == 0 && rollback_runs[0].stream == 0 &&
            rollback_runs[0].cell == 0 && rollback_runs[0].length == 5);
    const auto early_in_flight = rollback.build_source_plan(0, { id(0), id(1), id(2) });
    CHECK(early_in_flight[0] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(early_in_flight[1] >= 0 && early_in_flight[2] >= 0);
    const auto late_in_flight = rollback.build_source_plan(0, { id(0), id(1), id(2), id(3), id(4) });
    CHECK(late_in_flight[0] == LLAMA_KV_TAIL_BODY_SLOT && late_in_flight[1] == LLAMA_KV_TAIL_BODY_SLOT &&
            late_in_flight[2] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(late_in_flight[3] >= 0 && late_in_flight[4] >= 0);
    rollback.begin_batch();
    CHECK(rollback.active_slots().size() == 2);

    // Compact history retains N active rows plus exactly R rollback rows. A
    // suffix removal through R exposes the previous complete exact suffix;
    // R+1 is rejected before mutation.
    llama_kv_tail_store compact_rollback(2, 2, 1, 4, 0);
    compact_rollback.commit(0, id(0), 0, 0);
    compact_rollback.commit(0, id(1), 1, 1);
    compact_rollback.commit(0, id(2), 2, 2);
    compact_rollback.commit(0, id(3), 3, 3);
    CHECK(compact_rollback.retention() == 2);
    CHECK(compact_rollback.history_capacity() == 4);
    CHECK(compact_rollback.rollback_horizon() == 2);
    CHECK(compact_rollback.supports_suffix_rollback(0, 2));
    CHECK(!compact_rollback.supports_suffix_rollback(0, 3));
    compact_rollback.seq_rm(0, 2, -1);
    const auto compact_rollback_plan =
            compact_rollback.build_source_plan(0, { id(0), id(1) });
    CHECK(compact_rollback_plan[0] >= 0 && compact_rollback_plan[1] >= 0);
    CHECK(compact_rollback.coverage(0, 2).state == LLAMA_KV_TAIL_COVERAGE_COMPLETE);

    // A compact batch exposes the newest committed N rows plus graph-local
    // current rows, while its destination metadata remains transactional.
    compact_rollback.begin_batch();
    compact_rollback.commit(0, id(4), 4, 4, 0);
    compact_rollback.commit(0, id(5), 5, 5, 1);
    compact_rollback.commit(0, id(6), 6, 6, 2);
    const auto mixed_sources = compact_rollback.source_candidates(0);
    CHECK(mixed_sources.size() == 5);
    CHECK(mixed_sources[0].identity == id(0));
    CHECK(mixed_sources[1].identity == id(1));
    for (size_t i = 2; i < mixed_sources.size(); ++i) {
        CHECK(mixed_sources[i].identity == id(uint32_t(i + 2)));
        CHECK(mixed_sources[i].slot >= int32_t(compact_rollback.history_capacity()));
    }
    compact_rollback.finish_batch(false, false);
    CHECK(compact_rollback.snapshot(0).size() == 2);
    CHECK(compact_rollback.snapshot(0)[0].identity == id(0));
    CHECK(compact_rollback.snapshot(0)[1].identity == id(1));

    compact_rollback.begin_batch();
    compact_rollback.commit(0, id(4), 4, 4, 0);
    compact_rollback.finish_batch(false, true);
    CHECK(compact_rollback.snapshot(0).empty());
    CHECK(compact_rollback.coverage(0, 2).state == LLAMA_KV_TAIL_COVERAGE_NONE);

    // Recency follows position and insertion ordinal, not physical-cell order.
    llama_kv_tail_store ordered(2, 1, 6);
    ordered.commit(0, id(10), 30, 0);
    ordered.commit(0, id(11), 10, 1);
    ordered.commit(0, id(12), 30, 2);
    auto ordered_plan = ordered.build_source_plan(0, { id(10), id(11), id(12) });
    CHECK(ordered_plan[0] >= 0);
    CHECK(ordered_plan[1] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(ordered_plan[2] >= 0);
    ordered.seq_add(0, 30, 31, -25);
    ordered_plan = ordered.build_source_plan(0, { id(10), id(11), id(12) });
    CHECK(ordered_plan[0] >= 0 && ordered_plan[2] >= 0);
    CHECK(ordered_plan[1] == LLAMA_KV_TAIL_BODY_SLOT);
    ordered.seq_div(0, 0, -1, 5);
    CHECK(ordered.coverage(0).state == LLAMA_KV_TAIL_COVERAGE_PARTIAL);
    CHECK(ordered.coverage(0).degradation_flags & LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);

    // Exhaustion fails before publishing an identity or sequence membership.
    llama_kv_tail_store bounded(2, 1, 2);
    bounded.begin_batch();
    bounded.commit(0, id(20), 0, 0);
    bounded.commit(0, id(21), 1, 1);
    bool exhausted = false;
    try {
        bounded.commit(0, id(22), 2, 2);
    } catch (const std::runtime_error &) {
        exhausted = true;
    }
    CHECK(exhausted);
    CHECK(bounded.active_slots().size() == 2);
    CHECK(bounded.build_source_plan(0, { id(22) })[0] == LLAMA_KV_TAIL_BODY_SLOT);

    // Updating an existing identity can make it older than the retained
    // window. Returning the body sentinel must not dereference its released
    // shadow after trimming.
    llama_kv_tail_store duplicate(2, 1, 4);
    duplicate.begin_batch();
    duplicate.commit(0, id(30), 10, 0);
    duplicate.commit(0, id(31), 20, 1);
    duplicate.commit(0, id(32), 30, 2);
    CHECK(duplicate.commit(0, id(30), 0, 3) == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(duplicate.build_source_plan(0, { id(30), id(31), id(32) })[0] == LLAMA_KV_TAIL_BODY_SLOT);

    // State restore preserves the physical ring phase. Reallocating logically
    // identical exact rows from a later live cursor permutes the dense tail
    // reduction and can change a near-tied token on GPU backends.
    llama_kv_tail_store ring_source(4, 1, 6);
    ring_source.commit(0, id(40), 40, 40);
    ring_source.commit(0, id(41), 41, 41);
    ring_source.commit(0, id(42), 42, 42);
    const auto ring_snapshot = ring_source.snapshot(0);
    const uint32_t ring_cursor = ring_source.state_write_cursor(0);

    llama_kv_tail_store ring_restored(4, 1, 6);
    ring_restored.commit(0, id(90), 90, 90);
    ring_restored.commit(0, id(91), 91, 91);
    ring_restored.commit(0, id(92), 92, 92);
    ring_restored.seq_rm(0, -1, -1);
    for (const auto & entry : ring_snapshot) {
        CHECK(ring_restored.restore(
                0, entry.identity, entry.position, entry.insertion_ordinal,
                uint32_t(entry.slot)) == entry.slot);
    }
    ring_restored.restore_write_cursor(0, ring_cursor);
    CHECK(ring_restored.snapshot(0).size() == ring_snapshot.size());
    for (size_t i = 0; i < ring_snapshot.size(); ++i) {
        CHECK(ring_restored.snapshot(0)[i].slot == ring_snapshot[i].slot);
    }
    CHECK(ring_source.commit(0, id(43), 43, 43) ==
            ring_restored.commit(0, id(43), 43, 43));

    // The reference attention performs one global normalization while choosing
    // body or exact data independently for every visible entry.
    const std::vector<float> q = { 1.0f, 0.5f };
    const std::vector<float> body_k = { 1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f };
    const std::vector<float> body_v = { 1.0f, 2.0f, 10.0f, 20.0f, 100.0f, 200.0f };
    const std::vector<float> tail_k = { 0.25f, 1.5f };
    const std::vector<float> tail_v = { 30.0f, 60.0f };
    const auto ref = llama_kv_tail_attention_reference(
            q, body_k, body_v, tail_k, tail_v, { -1, 0, -1 }, 2, 2, 1.0f);

    const float e0 = std::exp(1.0f);
    const float e1 = std::exp(1.0f);
    const float e2 = std::exp(0.75f);
    const float z = e0 + e1 + e2;
    CHECK(std::fabs(ref[0] - (e0*1.0f + e1*30.0f + e2*100.0f)/z) < 1e-5f);
    CHECK(std::fabs(ref[1] - (e0*2.0f + e1*60.0f + e2*200.0f)/z) < 1e-5f);

    return 0;
}
