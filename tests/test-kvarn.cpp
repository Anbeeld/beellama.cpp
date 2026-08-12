#include "llama-kvarn.h"
#include "llama-kv-cache-kvarn.h"
#include "llama-memory.h"

#include "ggml-backend.h"
#include "../ggml/src/ggml-vulkan/fattn-kvarn-route-policy.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static void require(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "test-kvarn: %s\n", msg);
        std::abort();
    }
}

static void test_type_table() {
    const int supported_bits[] = { 2, 3, 4, 5, 6, 8 };

    require(llama_kvarn_type_count() == 37, "unexpected KVarN type count");

    const llama_kvarn_type_desc * disabled = llama_kvarn_type_desc_from_name("off");
    require(disabled != nullptr, "disabled type name did not parse");
    require(disabled->type == LLAMA_KVARN_TYPE_DISABLED, "disabled type enum mismatch");
    require(disabled->key_bits == 0 && disabled->value_bits == 0, "disabled bits mismatch");
    require(disabled->group == 128, "disabled group mismatch");

    for (int key_bits : supported_bits) {
        for (int value_bits : supported_bits) {
            const std::string name = "kvarn_k" + std::to_string(key_bits) + "v" + std::to_string(value_bits) + "_g128";
            const llama_kvarn_type_desc * desc = llama_kvarn_type_desc_from_name(name.c_str());
            require(desc != nullptr, "expected type name did not parse");
            require(desc->type != LLAMA_KVARN_TYPE_DISABLED && desc->type != LLAMA_KVARN_TYPE_INVALID, "parsed type enum mismatch");
            require(desc->key_bits == key_bits, "parsed key bits mismatch");
            require(desc->value_bits == value_bits, "parsed value bits mismatch");
            require(desc->group == 128, "parsed group mismatch");

            const llama_kvarn_type_desc * by_type = llama_kvarn_type_desc_from_type(desc->type);
            require(by_type != nullptr, "expected enum did not map to descriptor");
            require(std::string(by_type->name) == name, "enum descriptor name mismatch");
        }
    }

    require(llama_kvarn_type_desc_from_name("kvarn_k7v2_g128") == nullptr, "invalid type parsed");
}

static void test_attention_domain_policy() {
    const auto portable_decode = llama_kvarn_plan_attention(true, false, 16, 1);
    require(portable_decode.native_attention,
            "portable single-token decode must use native KVarN attention");
    require(portable_decode.domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED,
            "portable single-token decode must remain in the rotated domain");

    const auto portable_verification = llama_kvarn_plan_attention(true, false, 16, 16);
    require(portable_verification.native_attention,
            "portable verification within the advertised limit must remain native");

    const auto portable_prefill = llama_kvarn_plan_attention(true, false, 16, 17);
    require(!portable_prefill.native_attention,
            "portable large-query prefill must materialize for tiled backend attention");
    require(portable_prefill.domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED,
            "portable materialized prefill must remain in the rotated domain");

    const auto cuda_portable_prefill =
        llama_kvarn_plan_attention(true, false, UINT32_MAX, 1024);
    require(cuda_portable_prefill.native_attention,
            "portable CUDA prompt processing must remain native through the advertised limit");
    require(cuda_portable_prefill.domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED,
            "portable-only CUDA prompt processing must remain in the rotated domain");

    const auto cuda_prefill = llama_kvarn_plan_attention(true, true, 16, 17);
    require(cuda_prefill.native_attention,
            "backends with original-domain V support must retain native prefill");
    require(cuda_prefill.domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V,
            "native large-query prefill must use original-domain V");

    require(llama_kvarn_attention_domain(false, false, 0, 64) ==
                GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED,
            "materialized KVarN attention must remain in the rotated domain");
    require(llama_kvarn_attention_domain(true, false, 0, 64) ==
                GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED,
            "portable native KVarN attention must remain in the rotated domain");
    require(llama_kvarn_attention_domain(true, true, 16, 1) ==
                GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED,
            "single-token optimized KVarN decode must remain in the rotated domain");
    require(llama_kvarn_attention_domain(true, true, 16, 9) ==
                GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED,
            "DFlash verification must use the rotated KVarN decode domain");
    require(llama_kvarn_attention_domain(true, true, 16, 16) ==
                GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED,
            "full DFlash block verification must use the rotated KVarN decode domain");
    require(llama_kvarn_attention_domain(true, true, 16, 17) ==
                GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V,
            "large KVarN batches must retain original-domain V prefill");
    require(llama_kvarn_attention_domain(true, true, 0, 1) ==
                GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED,
            "a backend without an extended capability must retain true decode");
    require(llama_kvarn_attention_domain(true, true, 0, 2) ==
                GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V,
            "a backend without an extended capability must retain multi-row prefill");

    for (uint32_t n_q : { 1u, 16u, 17u, 256u, 512u }) {
        const auto hip_safe_first = llama_kvarn_plan_attention(
                true, false, UINT32_MAX, n_q);
        require(hip_safe_first.native_attention &&
                hip_safe_first.domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED,
                "HIP-like portable capability must keep every admitted query count in the rotated domain");

        const auto cuda_domain = llama_kvarn_plan_attention(true, true, 16, n_q);
        require(cuda_domain.native_attention,
                "CUDA-like KVarN capability unexpectedly materialized an admitted query count");
        require(cuda_domain.domain == (n_q <= 16 ?
                    GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED :
                    GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V),
                "CUDA-like KVarN domain threshold changed while correcting HIP policy");
    }
}

static void test_vulkan_decode_route_policy() {
    const auto decode = ggml_vk_fattn_kvarn_plan({
        16384, 1, 32, 4, 1, 82,
    });
    require(decode.gqa_group_size == 4,
            "Vulkan KVarN decode must reuse each KV reconstruction across GQA heads");
    require(decode.workgroups_y == 8,
            "Vulkan KVarN grouped-GQA workgroup count is incorrect");
    require(decode.split_k > 1,
            "long-context Vulkan KVarN decode must use split-K occupancy");

    const auto no_gqa = ggml_vk_fattn_kvarn_plan({
        16384, 1, 8, 8, 1, 16,
    });
    require(no_gqa.gqa_group_size == 1 && no_gqa.workgroups_y == 8,
            "Vulkan KVarN MHA must not group unrelated KV heads");
    require(no_gqa.split_k > 1,
            "under-occupied Vulkan KVarN MHA decode must use split-K");

    const auto populated = ggml_vk_fattn_kvarn_plan({
        2048, 16, 32, 4, 1, 16,
    });
    require(populated.split_k == 1,
            "well-populated Vulkan verification must avoid unnecessary split reduction");
}

static void set_test_env(const char * name, const char * value) {
#if defined(_WIN32)
    require(_putenv_s(name, value ? value : "") == 0, "failed to update test environment");
#else
    const int rc = value ? setenv(name, value, 1) : unsetenv(name);
    require(rc == 0, "failed to update test environment");
#endif
}

class scoped_test_env {
public:
    scoped_test_env(const char * name, const char * value) : name(name) {
        const char * previous = std::getenv(name);
        if (previous != nullptr) {
            had_previous = true;
            previous_value = previous;
        }
        set_test_env(name, value);
    }

    ~scoped_test_env() {
        set_test_env(name.c_str(), had_previous ? previous_value.c_str() : nullptr);
    }

private:
    std::string name;
    std::string previous_value;
    bool had_previous = false;
};

struct test_state_memory : public llama_memory_i {
    bool can_save;
    bool can_restore;

    test_state_memory(bool can_save, bool can_restore) : can_save(can_save), can_restore(can_restore) {}

    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override { return nullptr; }
    llama_memory_context_ptr init_full() override { return nullptr; }
    llama_memory_context_ptr init_update(llama_context *, bool) override { return nullptr; }
    bool get_can_shift() const override { return false; }
    void clear(bool) override {}
    bool seq_rm(llama_seq_id, llama_pos, llama_pos) override { return true; }
    bool seq_rm_cell(llama_seq_id, uint32_t) override { return true; }
    int cells_at_pos(llama_seq_id, llama_pos, uint32_t *, int) override { return 0; }
    void seq_cp(llama_seq_id, llama_seq_id, llama_pos, llama_pos) override {}
    void seq_keep(llama_seq_id) override {}
    void seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) override {}
    void seq_div(llama_seq_id, llama_pos, llama_pos, int) override {}
    llama_pos seq_pos_min(llama_seq_id) const override { return -1; }
    llama_pos seq_pos_max(llama_seq_id) const override { return -1; }
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override { return {}; }
    bool state_seq_can_save(llama_seq_id seq_id) const override { return seq_id >= 0 && can_save; }
    bool state_seq_can_restore(llama_seq_id seq_id) const override { return seq_id >= 0 && can_restore; }
    void state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const override {}
    void state_read(llama_io_read_i &, llama_seq_id, llama_state_seq_flags) override {}
};

struct test_composite_state_memory : public test_state_memory {
    const llama_memory_i & first;
    const llama_memory_i & second;

    test_composite_state_memory(const llama_memory_i & first, const llama_memory_i & second) :
        test_state_memory(true, true), first(first), second(second) {}

    bool state_seq_can_save(llama_seq_id seq_id) const override {
        return first.state_seq_can_save(seq_id) && second.state_seq_can_save(seq_id);
    }

    bool state_seq_can_restore(llama_seq_id seq_id) const override {
        return first.state_seq_can_restore(seq_id) && second.state_seq_can_restore(seq_id);
    }

    bool seq_rm_plan(
            llama_seq_id seq_id, llama_pos p0, llama_pos p1,
            llama_pos & planned_p0, llama_pos & planned_p1) const override {
        return llama_memory_seq_rm_plan_all(
                seq_id, p0, p1, { &first, &second }, planned_p0, planned_p1);
    }
};

struct test_planning_memory : public test_state_memory {
    llama_pos proposal;
    bool accepts_any_suffix;

    test_planning_memory(llama_pos proposal, bool accepts_any_suffix) :
        test_state_memory(true, true), proposal(proposal), accepts_any_suffix(accepts_any_suffix) {}

    bool can_seq_rm(llama_seq_id, llama_pos p0, llama_pos p1) const override {
        return p1 < 0 && (accepts_any_suffix || p0 == proposal);
    }

    bool seq_rm_plan(
            llama_seq_id, llama_pos, llama_pos p1,
            llama_pos & planned_p0, llama_pos & planned_p1) const override {
        if (p1 >= 0) {
            return false;
        }
        planned_p0 = proposal;
        planned_p1 = -1;
        return true;
    }
};

static void kvarn_composite_exclusivity_forwards() {
    const test_state_memory permissive(true, true);
    const test_state_memory contended_kvarn_like(false, false);
    const test_composite_state_memory composite(permissive, contended_kvarn_like);

    require(!composite.state_seq_can_save(0),
            "composite accepted a save rejected by its KVarN-like child");
    require(!composite.state_seq_can_restore(0),
            "composite accepted a restore rejected by its KVarN-like child");
}

static void kvarn_composite_removal_plan_forwards() {
    const test_planning_memory standard_like(5626, true);
    const test_planning_memory kvarn_like(5504, false);
    const test_composite_state_memory composite(standard_like, kvarn_like);
    llama_pos planned_p0 = -1;
    llama_pos planned_p1 = 0;

    require(composite.seq_rm_plan(0, 5626, -1, planned_p0, planned_p1),
            "composite rejected a common suffix removal plan");
    require(planned_p0 == 5504 && planned_p1 == -1,
            "composite did not choose the earliest child suffix boundary");
}

static void kvarn_unified_save_requires_exclusive_stream() {
    const std::array<llama_pos, 2> pos_max = { 255, 127 };
    const auto get_pos_max = [&](llama_seq_id seq_id) { return pos_max.at(size_t(seq_id)); };
    require(!llama_kvarn_stream_is_exclusive_for(1, pos_max.size(), 0, get_pos_max),
            "unified KVarN save accepted a stream containing another live sequence");
}

static void kvarn_unified_restore_requires_exclusive_stream() {
    const std::array<llama_pos, 2> pos_max = { 255, 127 };
    const auto get_pos_max = [&](llama_seq_id seq_id) { return pos_max.at(size_t(seq_id)); };
    require(!llama_kvarn_stream_is_exclusive_for(1, pos_max.size(), 1, get_pos_max),
            "unified KVarN restore accepted a stream containing another live sequence");

    const std::array<llama_pos, 2> exclusive_pos_max = { 255, -1 };
    const auto get_exclusive_pos_max = [&](llama_seq_id seq_id) { return exclusive_pos_max.at(size_t(seq_id)); };
    require(llama_kvarn_stream_is_exclusive_for(1, exclusive_pos_max.size(), 0, get_exclusive_pos_max),
            "unified KVarN restore rejected an exclusively owned stream");
    require(llama_kvarn_stream_is_exclusive_for(2, pos_max.size(), 1, get_pos_max),
            "non-unified KVarN restore rejected a sequence-owned stream");
}

static void kvarn_selective_state_owns_only_live_stage_rows() {
    const std::vector<uint32_t> selected = { 0, 3, 127, 128, 255, 256, 383, 384, 511, 640 };
    const auto rows = llama_kvarn_select_state_stage_cells(selected, 641, 3, 2, false);
    require(rows.size() == 4, "selective KVarN state copied sealed history rows");
    const std::array<llama_kvarn_state_stage_cell, 4> expected = {{
        { 0, 0 }, { 3, 3 }, { 127, 127 },
        { 640, 128 },
    }};
    for (size_t i = 0; i < expected.size(); ++i) {
        require(rows[i].source_cell == expected[i].source_cell &&
                rows[i].stage_row == expected[i].stage_row,
                "selective KVarN state stage mapping mismatch");
    }

    const auto swa_rows = llama_kvarn_select_state_stage_cells(
            { 0, 127, 128, 255, 256, 383, 384 }, 385, 3, 3, true);
    require(swa_rows.size() == 5 && swa_rows.front().source_cell == 128 &&
            swa_rows.back().source_cell == 384 && swa_rows.back().stage_row == 0,
            "selective SWA state did not retain exactly the live ring rows");

    const auto record_groups = llama_kvarn_select_state_record_groups(selected, rows, 8);
    require(record_groups == std::vector<uint32_t>({ 1, 2, 3 }),
            "selective KVarN state serialized a staged or stale record group");

    const std::vector<uint32_t> short_cells = { 0, 1, 127, 128 };
    const auto short_stage = llama_kvarn_select_state_stage_cells(short_cells, 129, 3, 2, false);
    require(llama_kvarn_select_state_record_groups(short_cells, short_stage, 8).empty(),
            "short selective KVarN state serialized stale compressed records");

    const std::vector<uint32_t> wrapped_cells = { 128, 255, 768, 769 };
    const std::vector<uint32_t> wrapped_stage_groups = { 6 };
    const auto wrapped_stage = llama_kvarn_select_state_stage_cells(
            wrapped_cells, 770, 3, 2, false, &wrapped_stage_groups);
    require(wrapped_stage.size() == 2 && wrapped_stage[0].source_cell == 768 &&
            wrapped_stage[1].source_cell == 769 && wrapped_stage[0].stage_row == 256,
            "selective KVarN state inferred stage provenance from physical record order");

    const std::vector<uint32_t> no_stage_groups;
    require(llama_kvarn_select_state_stage_cells(
                wrapped_cells, 770, 3, 2, false, &no_stage_groups).empty(),
            "selective KVarN state treated sealed records as live stage rows");
}

static void kvarn_compact_read_plan_skips_ownership_holes() {
    std::vector<uint32_t> occupied;
    for (uint32_t group : { 3u, 7u, 11u }) {
        for (uint32_t cell = group*128u; cell < (group + 1u)*128u; ++cell) {
            occupied.push_back(cell);
        }
    }

    const std::vector<uint32_t> pending = { 15u*128u, 15u*128u + 1u };
    const auto plan = llama_kvarn_compact_read_plan(occupied, pending, 4096, 256);

    require(plan.size() == 512,
            "compact KVarN read plan retained the sparse physical span");
    require(plan.front() == 3*128 && plan[383] == 12*128 - 1,
            "compact KVarN read plan changed occupied-cell ordering");
    require(plan[384] == 15*128 && plan[385] == 15*128 + 1,
            "compact KVarN read plan omitted pending cells");
    require(std::all_of(plan.begin() + 386, plan.end(),
                    [](int64_t cell) { return cell == -1; }),
            "compact KVarN read plan did not mark padding rows empty");

    const auto reordered = llama_kvarn_compact_read_plan(
            { 0, 128, 256, 384 }, { 1 }, 512, 256);
    require(reordered[0] == 0 && reordered[1] == 128 && reordered[2] == 256 &&
            reordered[3] == 384 && reordered[4] == 1,
            "compact KVarN read plan discarded logical caller ordering");

    const auto deduped = llama_kvarn_compact_read_plan(
            { 1, 2, 3 }, { 2, 3, 4 }, 128, 256);
    require(deduped.size() == 128 && deduped[0] == 1 && deduped[3] == 4 && deduped[4] == -1,
            "compact KVarN read plan did not deduplicate pending cells");
}

static void test_stage_policy() {
    // The reference keeps one incomplete group in F16, but the suffix-removal
    // contract advertised by llama_kvarn_can_remove_range reaches back into the
    // previous group from any position. That group's own F16 source therefore
    // has to stay resident, otherwise re-sealing a partially reopened group
    // pulls its surviving prefix from the newer group's aliased stage rows.
    require(llama_kvarn_non_swa_tail_groups(0, 0) == 2,
            "non-SWA KVarN must retain the reopenable previous group");
    require(llama_kvarn_non_swa_tail_groups(2048, 128) == 2,
            "reference KVarN tail must not scale with the scheduler batch");
    require(llama_kvarn_non_swa_tail_groups(2048, 512) == 2,
            "reference KVarN tail must not scale with the physical ubatch");
}

static void test_memory_stats_aggregation() {
    llama_kv_memory_stats first;
    first.global.k_payload_bytes = 10;
    first.global.v_payload_bytes = 20;
    first.global.exact_tail_bytes = 30;
    first.global.native_exact_bytes = 7;
    first.global.rollback_reserve_bytes = 9;
    first.global.transient_estimate_bytes = 11;
    first.global.staging_bytes = 40;
    first.global.metadata_bytes = 50;
    first.global.padding_bytes = 60;
    first.global.allocated_capacity_tokens = 4096;

    llama_kv_memory_stats second;
    second.swa.k_payload_bytes = 1;
    second.swa.v_payload_bytes = 2;
    second.swa.exact_tail_bytes = 3;
    second.swa.native_exact_bytes = 8;
    second.swa.rollback_reserve_bytes = 2;
    second.swa.transient_estimate_bytes = 3;
    second.swa.staging_bytes = 4;
    second.swa.metadata_bytes = 5;
    second.swa.padding_bytes = 6;
    second.swa.allocated_capacity_tokens = 1024;

    first.add(second);
    require(first.k_payload_bytes() == 11 && first.v_payload_bytes() == 22,
            "KV memory payload aggregation mismatch");
    require(first.exact_overlay_bytes() == 33 && first.native_exact_bytes() == 15 &&
            first.exact_history_bytes() == 48 && first.rollback_reserve_bytes() == 11 &&
            first.exact_tail_bytes() == 59 && first.transient_estimate_bytes() == 14 &&
            first.persistent_overhead_bytes() == 165,
            "KV memory overhead aggregation mismatch");
    require(first.resident_bytes() == 257,
            "KV memory resident total does not reconcile");
    require(first.allocated_capacity_tokens() == 4096,
            "KV memory capacity must report the largest attention domain, not their sum");
}

static void test_exact_tail_policy() {
    const auto omitted = llama_kvarn_tail_policy_for(0, 4096);
    require(omitted.raw_requested_tokens == 0, "omitted KVarN tail request provenance changed");
    require(omitted.requested_tokens == 128, "omitted KVarN tail must report the intrinsic 128-token request");
    require(omitted.effective_tokens == 128, "omitted KVarN tail must retain one exact group");
    require(omitted.exact_groups == 1, "omitted KVarN tail must retain one exact group");
    require(!omitted.native_exact, "partial KVarN tail must retain compressed body records");

    for (uint32_t requested : { 1u, 127u, 128u }) {
        const auto policy = llama_kvarn_tail_policy_for(requested, 4096);
        require(policy.raw_requested_tokens == requested, "explicit KVarN tail request provenance changed");
        require(policy.requested_tokens == 128, "explicit sub-group KVarN tail coverage target was not rounded");
        require(policy.effective_tokens == 128, "sub-group KVarN tail request did not round to one group");
        require(policy.exact_groups == 1, "sub-group KVarN tail request resolved to the wrong group count");
    }

    const auto rounded = llama_kvarn_tail_policy_for(129, 4096);
    require(rounded.effective_tokens == 256 && rounded.exact_groups == 2,
            "KVarN tail request did not round upward to complete groups");

    const auto capped = llama_kvarn_tail_policy_for(4096, 1000);
    require(capped.effective_tokens == 1000 && capped.exact_groups == 8 && capped.native_exact,
            "full-window KVarN tail request did not promote to native exact storage");

    const auto short_window = llama_kvarn_tail_policy_for(0, 64);
    require(short_window.effective_tokens == 64 && short_window.native_exact,
            "intrinsic KVarN tail did not cap and promote at a short window");

    for (uint32_t n_ubatch : { 1u, 127u, 128u, 129u, 512u, 513u }) {
        require(llama_kvarn_workspace_groups(n_ubatch) == 1,
                "boundary-completing KVarN store must keep one incomplete-group workspace slot");
        require(llama_kvarn_tail_policy_for(512, 4096).effective_tokens == 512,
                "physical ubatch changed the logical KVarN exact tail");
    }
}

static void test_tile_layout() {
    for (size_t i = 0; i < llama_kvarn_type_count(); ++i) {
        const llama_kvarn_type type = (llama_kvarn_type) i;
        if (type == LLAMA_KVARN_TYPE_DISABLED) {
            continue;
        }

        const llama_kvarn_type_desc * desc = llama_kvarn_type_desc_from_type(type);
        require(desc != nullptr, "layout type descriptor missing");

        const llama_kvarn_tile_layout layout = llama_kvarn_make_layout(128, 128, desc->key_bits, desc->value_bits);
        require(layout.k_payload_bytes == size_t(2048 * desc->key_bits), "K payload bytes mismatch");
        require(layout.v_payload_bytes == size_t(2048 * desc->value_bits), "V payload bytes mismatch");
        require(layout.tile_bytes == size_t(2048 * (desc->key_bits + desc->value_bits) + 1536), "tile bytes mismatch");
        require(layout.k_s_col_off == layout.k_payload_off + layout.k_payload_bytes, "K scale offset mismatch");
        require(layout.v_payload_off == layout.k_s_row_off + 128 * sizeof(uint16_t), "V payload offset mismatch");
        require(layout.v_s_col_off == layout.v_payload_off + layout.v_payload_bytes, "V scale offset mismatch");
        require(layout.tile_bytes % 8 == 0, "tile bytes not 8-byte aligned");
    }
}

static void test_head_dimension_slicing() {
    require(llama_kvarn_head_slices(128) == 1, "128-dim head should use one KVarN slice");
    require(llama_kvarn_head_slices(256) == 2, "256-dim head should use two KVarN slices");
    require(llama_kvarn_head_slices(512) == 4, "512-dim head should use four KVarN slices");
    require(llama_kvarn_head_slices(384) == 0, "384-dim head has no KVarN slice route");
    require(llama_kvarn_head_slices(64)  == 0, "64-dim head is not KVarN slice-compatible");
    require(llama_kvarn_head_slices(513) == 0, "non-128-multiple head is not KVarN slice-compatible");
}

static void test_runtime_validation() {
    llama_kvarn_runtime_requirements supported = {};
    supported.attention_supported = true;
    supported.head_dims_supported = true;
    supported.backend_ops_supported = true;
    supported.n_seq_max = 1;
    supported.kv_unified = false;

    for (size_t i = 0; i < llama_kvarn_type_count(); ++i) {
        const llama_kvarn_type type = (llama_kvarn_type) i;
        if (type == LLAMA_KVARN_TYPE_DISABLED) {
            continue;
        }

        const auto params = llama_kvarn_params_for_type(type);
        require(llama_kvarn_validate_runtime(params, supported) == nullptr, "valid runtime rejected");
    }

    auto invalid = llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128);
    invalid.key_bits = 3;
    require(llama_kvarn_validate_runtime(invalid, supported) != nullptr, "mismatched preset bits accepted");

    invalid = llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128);
    invalid.sink_tokens = 0;
    require(llama_kvarn_validate_runtime(invalid, supported) != nullptr, "unsupported sink tokens accepted");

    auto requirements = supported;
    requirements.attention_supported = false;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) != nullptr,
            "unsupported attention accepted");

    requirements = supported;
    requirements.head_dims_supported = false;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) != nullptr,
            "unsupported head dimension accepted");

    requirements = supported;
    requirements.backend_ops_supported = false;
    const char * backend_reason = llama_kvarn_validate_runtime(
            llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements);
    require(backend_reason != nullptr, "backend without KVarN store/materialize accepted");
    require(std::string(backend_reason) ==
            "KVarN requires a backend with KVarN store and materialization support",
            "backend without KVarN store/materialize returned an inaccurate diagnostic");

    requirements = supported;
    requirements.kv_unified = true;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) == nullptr,
            "unified single-sequence runtime rejected");

    requirements = supported;
    requirements.n_seq_max = 2;
    requirements.kv_unified = false;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) == nullptr,
            "non-unified multi-sequence runtime rejected");

    requirements.kv_unified = true;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) == nullptr,
            "unified multi-sequence runtime rejected");
}

static void test_remove_policy() {
    require(llama_kvarn_can_remove_range(-1, 0, -1, 128), "empty sequence removal rejected");
    require(llama_kvarn_can_remove_range(783, -1, -1, 128), "full sequence removal with negative range rejected");
    require(llama_kvarn_can_remove_range(783, 0, -1, 128), "full sequence removal from zero rejected");
    require(llama_kvarn_can_remove_range(783, 0, 784, 128), "explicit full sequence range removal rejected");
    require(!llama_kvarn_can_remove_range(783, 0, 640, 128), "old compressed partial removal accepted");
    require(llama_kvarn_can_remove_range(783, 640, -1, 128), "current/previous tail removal rejected");
}

static void iswa_nonunified_multislot_kvarn_policy() {
    require(llama_kvarn_iswa_policy_for(true, true, 2) ==
                    LLAMA_KVARN_ISWA_STANDARD_SWA_FALLBACK,
            "non-unified multi-slot iSWA did not select the standard-SWA fallback");
    require(llama_kvarn_iswa_policy_for(true, true, 1) ==
                    LLAMA_KVARN_ISWA_ALL_LAYERS,
            "single-slot non-unified iSWA KVarN was rejected");
}

static void kvarn_historical_suffix_plans_group_boundary() {
    const auto expect_plan = [](llama_pos p0, llama_pos p1, bool owned, bool expected,
                                llama_pos expected_p0, llama_pos expected_p1) {
        llama_pos planned_p0 = -99;
        llama_pos planned_p1 = -99;
        const bool actual = llama_kvarn_plan_remove_range(
                6143, p0, p1, KVAR_N_GROUP, owned, planned_p0, planned_p1);
        require(actual == expected, "KVarN historical suffix planner acceptance mismatch");
        if (actual) {
            require(planned_p0 == expected_p0 && planned_p1 == expected_p1,
                    "KVarN historical suffix planner boundary mismatch");
        }
    };

    expect_plan(5626, -1, true,  true, 5504, -1); // issue #67: 122 extra positions
    expect_plan(5504, -1, true,  true, 5504, -1); // aligned historical boundary
    expect_plan(5503, -1, true,  true, 5376, -1); // one before the boundary
    expect_plan(5505, -1, true,  true, 5504, -1); // one after the boundary
    expect_plan(6000, -1, false, true, 6000, -1); // live exact tail does not require ownership
    expect_plan(0,    -1, false, true,    0, -1); // complete removal
    expect_plan(-1,   -1, false, true,   -1, -1); // negative full-range convention
    expect_plan(5626, 5700, true, false,  -1, -1); // finite middle range is never widened
}

static void kvarn_swa_deep_rollback_plans_group_boundary() {
    llama_pos planned_p0 = -1;
    llama_pos planned_p1 = 0;
    require(llama_kvarn_plan_remove_range(
                1023, 300, -1, KVAR_N_GROUP, true, planned_p0, planned_p1),
            "deep SWA rollback did not produce a stage-safe removal plan");
    require(planned_p0 == 256 && planned_p1 == -1,
            "deep SWA rollback did not widen to the requested group's boundary");
}

static void kvarn_historical_suffix_rejects_contended_unified_stream() {
    llama_pos planned_p0 = -99;
    llama_pos planned_p1 = -99;
    require(!llama_kvarn_plan_remove_range(
                    6143, 5626, -1, KVAR_N_GROUP, false, planned_p0, planned_p1),
            "contended unified KVarN planned a historical suffix removal");
}

static void test_pack_roundtrip(int bits) {
    const int n = 257;
    std::vector<uint8_t> values(n);
    for (int i = 0; i < n; ++i) {
        values[i] = uint8_t((i * 7 + 3) & ((1 << bits) - 1));
    }

    std::vector<uint8_t> packed(llama_kvarn_packed_bytes(n, bits), 0);
    llama_kvarn_pack_bits(values.data(), n, bits, packed.data());

    for (int i = 0; i < n; ++i) {
        const uint8_t got = llama_kvarn_unpack_bits_value(packed.data(), i, bits);
        if (got != values[i]) {
            std::fprintf(stderr, "test-kvarn: %d-bit roundtrip mismatch at %d: got %u expected %u\n",
                    bits, i, unsigned(got), unsigned(values[i]));
            std::abort();
        }
    }
}

static void test_hadamard_roundtrip() {
    std::vector<float> values(128);
    std::vector<float> expected(128);
    for (int i = 0; i < 128; ++i) {
        values[i] = std::sin(float(i) * 0.19f) + float(i - 64) * 0.002f;
    }
    expected = values;

    llama_kvarn_hadamard_128(values.data());
    llama_kvarn_hadamard_128(values.data());

    for (int i = 0; i < 128; ++i) {
        require(std::fabs(values[i] - expected[i]) < 1e-5f, "Hadamard roundtrip mismatch");
    }
}

static float tile_rmse(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size(), "RMSE shape mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = double(a[i]) - double(b[i]);
        sum += diff * diff;
    }
    return float(std::sqrt(sum / a.size()));
}

static void test_tile_quantization(llama_kvarn_type type) {
    const auto * desc = llama_kvarn_type_desc_from_type(type);
    require(desc != nullptr, "quantization type descriptor missing");

    const auto layout = llama_kvarn_make_layout(128, 128, desc->key_bits, desc->value_bits);
    std::vector<float> k(128 * 128);
    std::vector<float> v(128 * 128);
    for (int r = 0; r < 128; ++r) {
        for (int c = 0; c < 128; ++c) {
            k[r * 128 + c] =
                std::sin(float(r) * 0.071f) +
                std::cos(float(c) * 0.113f) +
                float((r * 17 + c * 13) % 29 - 14) * 0.015f;
            v[r * 128 + c] =
                std::cos(float(r) * 0.057f) -
                std::sin(float(c) * 0.091f) +
                float((r * 11 + c * 19) % 31 - 15) * 0.012f;
        }
    }

    std::vector<uint8_t> record(layout.tile_bytes, 0);
    llama_kvarn_quantize_k_tile(k.data(), 16, desc->key_bits, layout, record.data());
    llama_kvarn_quantize_v_tile(v.data(), 16, desc->value_bits, layout, record.data());

    std::vector<float> k_dequant(k.size());
    std::vector<float> v_dequant(v.size());
    llama_kvarn_dequantize_k_tile(record.data(), desc->key_bits, layout, k_dequant.data());
    llama_kvarn_dequantize_v_tile(record.data(), desc->value_bits, layout, v_dequant.data());

    for (size_t i = 0; i < k.size(); ++i) {
        require(std::isfinite(k_dequant[i]), "K dequant produced non-finite value");
        require(std::isfinite(v_dequant[i]), "V dequant produced non-finite value");
    }

    const float max_rmse[] = { 0.0f, 0.0f, 0.40f, 0.22f, 0.12f, 0.08f, 0.05f, 0.0f, 0.025f };
    require(tile_rmse(k, k_dequant) < max_rmse[desc->key_bits], "K tile RMSE too high");
    require(tile_rmse(v, v_dequant) < max_rmse[desc->value_bits], "V tile RMSE too high");
}

// Proof that KVarN rotated-domain attention is algebraically equivalent to the
// original-domain decode path, using only the CPU reference quant/dequant. Let R = the
// normalized WHT-128 (symmetric involution: R^2 = I, verified separately by
// test_hadamard_roundtrip). KVarN compressed records store K_rot = R*K and
// V_rot = R*V. Live fp16 K/V stage rows are rotated-domain so decode can read
// stage and records through the same hot path. Reference decode reconstructs
// X_orig = R*dequant(record_or_stage). Rotated-domain attention
// skips that inverse-WHT and instead rotates the query / inverse-rotates the output:
//   K:  Q . K_orig[:,c]        == (R Q) . K_rot[:,c]
//   V:  sum_t w[t] V_orig[t,:] == R ( sum_t w[t] V_rot[t,:] )
static void test_rotated_domain_equivalence() {
    const int bits = 4; // kvarn4
    const auto layout = llama_kvarn_make_layout(128, 128, bits, bits);

    std::vector<float> k(128 * 128);
    std::vector<float> v(128 * 128);
    for (int r = 0; r < 128; ++r) {
        for (int c = 0; c < 128; ++c) {
            k[r * 128 + c] = std::sin(float(r) * 0.071f) + std::cos(float(c) * 0.113f) +
                             float((r * 17 + c * 13) % 29 - 14) * 0.015f;
            v[r * 128 + c] = std::cos(float(r) * 0.057f) - std::sin(float(c) * 0.091f) +
                             float((r * 11 + c * 19) % 31 - 15) * 0.012f;
        }
    }

    std::vector<uint8_t> k_record(layout.tile_bytes, 0);
    std::vector<uint8_t> v_record(layout.tile_bytes, 0);
    llama_kvarn_quantize_k_tile(k.data(), 16, bits, layout, k_record.data());
    llama_kvarn_quantize_v_tile(v.data(), 16, bits, layout, v_record.data());

    std::vector<float> k_rot(128 * 128); // tile[dim*128 + token]
    std::vector<float> v_rot(128 * 128); // tile[token*128 + dim]
    llama_kvarn_dequantize_k_tile(k_record.data(), bits, layout, k_rot.data());
    llama_kvarn_dequantize_v_tile(v_record.data(), bits, layout, v_rot.data());

    // ---- K side: scores ----
    std::vector<float> q(128);
    for (int d = 0; d < 128; ++d) {
        q[d] = std::sin(float(d) * 0.037f) + 0.25f * std::cos(float(d) * 0.0131f);
    }
    std::vector<float> rq = q;
    llama_kvarn_hadamard_128(rq.data()); // R q

    float k_max_abs = 0.0f, k_max_diff = 0.0f;
    for (int c = 0; c < 128; ++c) {
        std::array<float, 128> kcol;                         // K_rot[:,c]
        for (int d = 0; d < 128; ++d) kcol[d] = k_rot[d * 128 + c];
        std::array<float, 128> korig = kcol;                 // K_orig[:,c] = R * K_rot[:,c]
        llama_kvarn_hadamard_128(korig.data());

        double ref = 0.0, rot = 0.0;
        for (int d = 0; d < 128; ++d) {
            ref += double(q[d]) * double(korig[d]);          // Q . K_orig
            rot += double(rq[d]) * double(kcol[d]);          // (R Q) . K_rot
        }
        k_max_abs  = std::max(k_max_abs, std::fabs(float(ref)));
        k_max_diff = std::max(k_max_diff, std::fabs(float(ref - rot)));
    }
    require(k_max_diff < 1e-3f * (1.0f + k_max_abs), "K rotated-domain score mismatch");

    // ---- V side: weighted output ----
    std::vector<float> w(128);
    double wsum = 0.0;
    for (int t = 0; t < 128; ++t) { w[t] = 0.5f + 0.5f * std::sin(float(t) * 0.083f) + 0.01f * float(t); wsum += w[t]; }
    for (int t = 0; t < 128; ++t) w[t] = float(w[t] / wsum);

    std::array<float, 128> ref_o = {}; // sum_t w[t] * R(V_rot[t,:])
    for (int t = 0; t < 128; ++t) {
        std::array<float, 128> vorig;
        for (int d = 0; d < 128; ++d) vorig[d] = v_rot[t * 128 + d];
        llama_kvarn_hadamard_128(vorig.data());
        for (int d = 0; d < 128; ++d) ref_o[d] += w[t] * vorig[d];
    }
    std::array<float, 128> o_rot = {}; // R( sum_t w[t] * V_rot[t,:] )
    for (int t = 0; t < 128; ++t) {
        for (int d = 0; d < 128; ++d) o_rot[d] += w[t] * v_rot[t * 128 + d];
    }
    llama_kvarn_hadamard_128(o_rot.data());

    float v_max_abs = 0.0f, v_max_diff = 0.0f;
    for (int d = 0; d < 128; ++d) {
        v_max_abs  = std::max(v_max_abs, std::fabs(ref_o[d]));
        v_max_diff = std::max(v_max_diff, std::fabs(ref_o[d] - o_rot[d]));
    }
    require(v_max_diff < 1e-3f * (1.0f + v_max_abs), "V rotated-domain output mismatch");
}

static ggml_backend_t init_test_backend(enum ggml_backend_dev_type device_type, bool required) {
    const char * backend_name = std::getenv("GGML_KVARN_TEST_BACKEND");
    const bool use_named_gpu = backend_name != nullptr && backend_name[0] != '\0' && device_type == GGML_BACKEND_DEVICE_TYPE_GPU;

    ggml_backend_t backend = use_named_gpu ?
        ggml_backend_init_by_name(backend_name, nullptr) :
        ggml_backend_init_by_type(device_type, nullptr);
    if (backend == nullptr && !required) {
        return nullptr;
    }
    require(backend != nullptr, use_named_gpu ? "failed to initialize GGML_KVARN_TEST_BACKEND" : "failed to initialize requested backend");
    return backend;
}

static void apply_reference_kvarn_wht_head(float * values, int head_width) {
    require(head_width == 128 || head_width == 256 || head_width == 512, "reference KVarN WHT invalid head width");
    const int slices = head_width / 128;

    for (int slice = 0; slice < slices; ++slice) {
        llama_kvarn_hadamard_128(values + slice * 128);
    }
    if (slices == 1) {
        return;
    }

    const float scale = slices == 2 ? 0.7071067811865475f : 0.5f;
    for (int d = 0; d < 128; ++d) {
        float x[4] = {};
        for (int slice = 0; slice < slices; ++slice) {
            x[slice] = values[slice * 128 + d];
        }
        for (int stride = 1; stride < slices; stride <<= 1) {
            for (int base = 0; base < slices; base += 2 * stride) {
                for (int i = 0; i < stride; ++i) {
                    const float a = x[base + i];
                    const float b = x[base + stride + i];
                    x[base + i] = a + b;
                    x[base + stride + i] = a - b;
                }
            }
        }
        for (int slice = 0; slice < slices; ++slice) {
            values[slice * 128 + d] = x[slice] * scale;
        }
    }
}

static void test_kvarn_wht_op(
        enum ggml_backend_dev_type device_type, bool required, int head_width,
        ggml_type input_type = GGML_TYPE_F32) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "KVarN WHT: failed to initialize ggml context");

    constexpr int n_tokens = 7;
    constexpr int n_heads = 3;
    ggml_tensor * input = ggml_new_tensor_3d(ctx, input_type, head_width, n_heads, n_tokens);
    ggml_tensor * output = ggml_kvarn_wht(ctx, input, head_width);

    if (!ggml_backend_supports_op(backend, output)) {
        require(!required, "required backend does not support KVarN WHT");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "KVarN WHT: failed to allocate tensors");

    std::vector<float> src(head_width * n_heads * n_tokens);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = std::sin(float(i) * 0.013f) + std::cos(float(i) * 0.021f) + float(int(i % 17) - 8) * 0.007f;
    }
    std::vector<uint8_t> input_data(ggml_nbytes(input));
    std::vector<float> ref(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        if (input_type == GGML_TYPE_F32) {
            ((float *) input_data.data())[i] = src[i];
            ref[i] = src[i];
        } else if (input_type == GGML_TYPE_F16) {
            const ggml_fp16_t value = ggml_fp32_to_fp16(src[i]);
            ((ggml_fp16_t *) input_data.data())[i] = value;
            ref[i] = ggml_fp16_to_fp32(value);
        } else {
            require(input_type == GGML_TYPE_BF16, "KVarN WHT: invalid test input type");
            const ggml_bf16_t value = ggml_fp32_to_bf16(src[i]);
            ((ggml_bf16_t *) input_data.data())[i] = value;
            ref[i] = ggml_bf16_to_fp32(value);
        }
    }
    for (int group = 0; group < n_heads * n_tokens; ++group) {
        apply_reference_kvarn_wht_head(ref.data() + size_t(group) * head_width, head_width);
    }

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size());
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "KVarN WHT: graph compute failed");

    std::vector<uint8_t> output_data(ggml_nbytes(output));
    ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size());
    std::vector<float> got(src.size());
    for (size_t i = 0; i < got.size(); ++i) {
        if (input_type == GGML_TYPE_F32) {
            got[i] = ((const float *) output_data.data())[i];
        } else if (input_type == GGML_TYPE_F16) {
            got[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) output_data.data())[i]);
        } else {
            got[i] = ggml_bf16_to_fp32(((const ggml_bf16_t *) output_data.data())[i]);
        }
    }

    double mse = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = double(got[i]) - double(ref[i]);
        mse += diff * diff;
        max_diff = std::max(max_diff, std::fabs(diff));
    }
    const double rmse = std::sqrt(mse / double(got.size()));
    const double tolerance = input_type == GGML_TYPE_F32 ? 2e-5 :
        (input_type == GGML_TYPE_F16 ? 2e-3 : 2e-2);
    if (!std::isfinite(rmse) || rmse > tolerance || max_diff > 4*tolerance) {
        std::fprintf(stderr, "KVarN WHT: head_width=%d type=%s rmse=%g max_diff=%g\n",
                head_width, ggml_type_name(input_type), rmse, max_diff);
        require(false, "KVarN WHT output mismatch");
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static float test_kvarn_record_value(const uint8_t * record, int bits, bool value, int token, int dim) {
    const size_t payload_bytes = llama_kvarn_packed_bytes(128 * 128, bits);
    const size_t scale_axis_off = payload_bytes;
    const size_t zp_axis_off = scale_axis_off + 128 * sizeof(ggml_fp16_t);
    const size_t other_axis_off = zp_axis_off + 128 * sizeof(ggml_fp16_t);
    const int row = value ? token : dim;
    const int col = value ? dim : token;
    ggml_fp16_t scale_fp16;
    ggml_fp16_t zp_fp16;
    ggml_fp16_t other_fp16;
    std::memcpy(&scale_fp16, record + scale_axis_off + row * sizeof(scale_fp16), sizeof(scale_fp16));
    std::memcpy(&zp_fp16, record + zp_axis_off + row * sizeof(zp_fp16), sizeof(zp_fp16));
    std::memcpy(&other_fp16, record + other_axis_off + col * sizeof(other_fp16), sizeof(other_fp16));
    const float scale = ggml_fp16_to_fp32(scale_fp16);
    const float zp = ggml_fp16_to_fp32(zp_fp16);
    const float other = ggml_fp16_to_fp32(other_fp16);
    const uint8_t q = llama_kvarn_unpack_bits_value(record, row * 128 + col, bits);
    return (float(q) * scale + zp) * other;
}

static std::vector<ggml_fp16_t> test_kvarn_reference_decode(
        const ggml_tensor * records,
        const ggml_tensor * stage,
        const std::vector<int64_t> & indices,
        int n_kv,
        int stream_start,
        int n_stream,
        int bits,
        bool value,
        int stage_groups,
        bool emit_rotated = false,
        bool swa = false,
        int head_slices = 1) {
    require(records->type == GGML_TYPE_I8, "reference decode records type mismatch");
    require(stage->type == GGML_TYPE_F16, "reference decode stage type mismatch");
    require(stage->ne[0] == 128, "reference decode stage width mismatch");
    require(stage_groups >= 2, "reference decode invalid stage_groups");
    require(stage->ne[2] % (128 * stage_groups) == 0, "reference decode stage shape mismatch");
    const int n_heads = (int) stage->ne[1];
    require(head_slices == 1 || head_slices == 2 || head_slices == 4,
            "reference decode invalid head_slices");
    require(n_heads % head_slices == 0, "reference decode head_slices mismatch");
    const int total_streams = (int) (stage->ne[2] / (128 * stage_groups));
    require(total_streams > 0, "reference decode total stream mismatch");
    require(records->ne[1] == n_heads, "reference decode head count mismatch");
    require(records->ne[2] % total_streams == 0, "reference decode record shape mismatch");
    require(stream_start >= 0 && n_stream > 0 && stream_start + n_stream <= total_streams,
            "reference decode stream range mismatch");
    if (swa) {
        require((int) indices.size() >= n_kv, "reference decode SWA indices too short");
    }

    const int groups_per_stream = (int) (records->ne[2] / total_streams);
    int tail_groups = stage_groups - 1;
    const int explicit_tail_groups = stage != nullptr ? stage->op_params[8] : 0;
    if (explicit_tail_groups > 0) {
        tail_groups = explicit_tail_groups;
    }
    std::vector<ggml_fp16_t> stage_data(ggml_nelements(stage));
    std::vector<uint8_t> record_data(ggml_nbytes(records));
    ggml_backend_tensor_get(stage, stage_data.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());

    std::vector<int64_t> live_groups(n_stream, 0);
    for (int64_t idx : indices) {
        if (idx < 0) {
            require(swa, "reference decode negative non-SWA index");
            continue;
        }
        const int64_t group_global = idx / 128;
        if (swa) {
            live_groups[0] = std::max(live_groups[0], group_global);
        } else {
            const int64_t stream = group_global / groups_per_stream;
            if (stream >= stream_start && stream < stream_start + n_stream) {
                const int64_t group = group_global - stream * groups_per_stream;
                live_groups[stream - stream_start] = std::max(live_groups[stream - stream_start], group);
            }
        }
    }

    std::vector<ggml_fp16_t> output((size_t) 128 * n_heads * n_kv * n_stream, ggml_fp32_to_fp16(0.0f));
    for (int out_stream = 0; out_stream < n_stream; ++out_stream) {
        const int stream = stream_start + out_stream;
        const int64_t live_group = live_groups[out_stream];
        const int64_t stage_base = (int64_t) stream * 128 * stage_groups;
        const int64_t stage_begin = swa
            ? (live_group >= (tail_groups - 1) ? live_group - (tail_groups - 1) : 0)
            : 0;
        for (int cell = 0; cell < n_kv; ++cell) {
            const int64_t abs_pos = swa ? indices[cell] : cell;
            if (abs_pos < 0) {
                continue;
            }
            const int64_t group = abs_pos / 128;
            const int64_t pos = abs_pos % 128;
            for (int h = 0; h < n_heads; ++h) {
                std::array<float, 128> values = {};
                bool values_original = false;
                bool from_stage;
                bool from_record;
                int64_t stage_pos = 0;
                int64_t record_group = 0;
                if (swa) {
                    from_stage  = group >= stage_begin && group <= live_group;
                    from_record = !from_stage && group >= 0 && group < stage_begin &&
                                  (live_group - group) < groups_per_stream + tail_groups;
                    stage_pos    = stage_base + (group % stage_groups) * 128 + pos;
                    record_group = (int64_t) stream * groups_per_stream + (group % groups_per_stream);
                } else {
                    from_stage  = group == 0 ||
                                  (group > 0 && group <= live_group &&
                                   group + (tail_groups - 1) >= live_group);
                    from_record = !from_stage && group < live_group;
                    stage_pos    = stage_base + (group == 0 ? pos : 128 + ((group - 1) % tail_groups) * 128 + pos);
                    record_group = (int64_t) stream * groups_per_stream + group;
                }
                if (from_stage) {
                    require(stage_pos >= 0 && stage_pos < stage->ne[2], "reference decode stage offset out of range");
                    for (int d = 0; d < 128; ++d) {
                        const size_t off = (size_t) d + (size_t) h * 128 + (size_t) stage_pos * 128 * n_heads;
                        values[d] = ggml_fp16_to_fp32(stage_data[off]);
                    }
                    values_original = false;
                } else if (from_record) {
                    require(record_group >= 0 && record_group < records->ne[2], "reference decode record offset out of range");
                    const size_t record_off = ((size_t) record_group * n_heads + h) * (size_t) records->ne[0];
                    const uint8_t * record = record_data.data() + record_off;
                    for (int d = 0; d < 128; ++d) {
                        values[d] = test_kvarn_record_value(record, bits, value, (int) pos, d);
                    }
                }
                if (emit_rotated == values_original && head_slices == 1) {
                    llama_kvarn_hadamard_128(values.data());
                }
                for (int d = 0; d < 128; ++d) {
                    const size_t out_off = (size_t) d + (size_t) h * 128 +
                        (size_t) cell * 128 * n_heads + (size_t) out_stream * 128 * n_heads * n_kv;
                    output[out_off] = ggml_fp32_to_fp16(values[d]);
                }
            }
        }
    }
    if (!emit_rotated && head_slices > 1) {
        const int head_width = 128 * head_slices;
        std::vector<float> head_values(head_width);
        for (int out_stream = 0; out_stream < n_stream; ++out_stream) {
            for (int cell = 0; cell < n_kv; ++cell) {
                for (int logical_head = 0; logical_head < n_heads / head_slices; ++logical_head) {
                    for (int slice = 0; slice < head_slices; ++slice) {
                        const int h = logical_head * head_slices + slice;
                        for (int d = 0; d < 128; ++d) {
                            const size_t off = (size_t) d + (size_t) h * 128 +
                                (size_t) cell * 128 * n_heads + (size_t) out_stream * 128 * n_heads * n_kv;
                            head_values[slice * 128 + d] = ggml_fp16_to_fp32(output[off]);
                        }
                    }

                    apply_reference_kvarn_wht_head(head_values.data(), head_width);

                    for (int slice = 0; slice < head_slices; ++slice) {
                        const int h = logical_head * head_slices + slice;
                        for (int d = 0; d < 128; ++d) {
                            const size_t off = (size_t) d + (size_t) h * 128 +
                                (size_t) cell * 128 * n_heads + (size_t) out_stream * 128 * n_heads * n_kv;
                            output[off] = ggml_fp32_to_fp16(head_values[slice * 128 + d]);
                        }
                    }
                }
            }
        }
    }
    return output;
}

static std::vector<float> test_kvarn_reference_decode_f32(
        const ggml_tensor * records,
        const ggml_tensor * stage,
        const std::vector<int64_t> & indices,
        int n_kv,
        int stream_start,
        int n_stream,
        int bits,
        bool value,
        int stage_groups,
        bool emit_rotated = false,
        bool swa = false,
        int head_slices = 1) {
    std::vector<ggml_fp16_t> output_f16 = test_kvarn_reference_decode(
            records, stage, indices, n_kv, stream_start, n_stream, bits, value, stage_groups, emit_rotated, swa, head_slices);
    std::vector<float> output(output_f16.size());
    ggml_fp16_to_fp32_row(output_f16.data(), output.data(), output.size());
    return output;
}

static void require_close_f16_rmse(
        const std::vector<ggml_fp16_t> & actual,
        const std::vector<ggml_fp16_t> & expected,
        float                            rmse_limit,
        const char *                     message);

static void test_cache_ops(
        enum ggml_backend_dev_type device_type,
        bool required,
        int bits,
        int head_slices = 1) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize ggml context");

    constexpr int n_tokens = 385;
    const int n_heads = head_slices;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 384);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, 4);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false, 3);
    stored->op_params[5] = head_slices;
    ggml_tensor * materialized = ggml_kvarn_materialize(
            ctx, records, stored, indices, n_tokens, 0, 1, bits, false, 3);
    materialized->op_params[5] = head_slices;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, materialized);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate KVarN tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            for (int d = 0; d < 128; ++d) {
                const int full_d = h * 128 + d;
                input[(t * n_heads + h) * 128 + d] =
                    std::sin(float(full_d) * 0.071f) +
                    std::cos(float(t) * 0.037f + float(h) * 0.11f) +
                    float((full_d * 13 + t * 17) % 31 - 15) * 0.01f;
            }
        }
    }
    std::vector<int64_t> idx(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        idx[i] = i;
    }
    std::vector<uint8_t> zeros(ggml_nbytes(stage) + ggml_nbytes(records), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, zeros.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, zeros.data(), 0, ggml_nbytes(records));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "KVarN graph compute failed");

    const std::vector<float> output = test_kvarn_reference_decode_f32(
            records, stored, idx, n_tokens, 0, 1, bits, false, 3, false, false, head_slices);
    std::vector<ggml_fp16_t> materialized_data(ggml_nelements(materialized));
    ggml_backend_tensor_get(materialized, materialized_data.data(), 0, ggml_nbytes(materialized));
    require(materialized_data.size() == output.size(), "materialized KVarN output shape mismatch");
    float materialize_max_diff = 0.0f;
    size_t materialize_max_index = 0;
    for (size_t i = 0; i < materialized_data.size(); ++i) {
        const float actual = ggml_fp16_to_fp32(materialized_data[i]);
        const float diff = std::abs(actual - output[i]);
        if (diff > materialize_max_diff) {
            materialize_max_diff = diff;
            materialize_max_index = i;
        }
    }
    if (materialize_max_diff >= 2e-3f) {
        std::fprintf(stderr, "KVarN materialize mismatch: bits=%d index=%zu max_diff=%g actual=%g expected=%g\n",
                bits, materialize_max_index, materialize_max_diff,
                ggml_fp16_to_fp32(materialized_data[materialize_max_index]), output[materialize_max_index]);
        require(false, "materialized KVarN output mismatch");
    }

    double sink_error = 0.0;
    double compressed_error = 0.0;
    double previous_tail_error = 0.0;
    double live_tail_error = 0.0;
    for (int t = 0; t < n_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            const double diff = double(input[t * n_heads * 128 + d]) - double(output[t * n_heads * 128 + d]);
            if (t < 128) {
                sink_error += diff * diff;
            } else if (t < 256) {
                compressed_error += diff * diff;
            } else if (t < 384) {
                previous_tail_error += diff * diff;
            } else {
                live_tail_error += diff * diff;
            }
        }
    }
    sink_error = std::sqrt(sink_error / (128 * 128));
    compressed_error = std::sqrt(compressed_error / (128 * 128));
    previous_tail_error = std::sqrt(previous_tail_error / (128 * 128));
    live_tail_error = std::sqrt(live_tail_error / 128);
    require(sink_error < 0.01, "sink reconstruction error too high");
    require(compressed_error < 0.25, "compressed reconstruction error too high");
    require(previous_tail_error < 0.01, "previous tail reconstruction error too high");
    require(live_tail_error < 0.01, "live tail reconstruction error too high");

    std::vector<uint8_t> record_data(ggml_nbytes(records));
    ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());
    require(std::any_of(record_data.begin(), record_data.end(), [](uint8_t v) { return v != 0; }),
            "completed group was not flushed");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void test_cache_ops_multi_stream(enum ggml_backend_dev_type device_type, bool required, int bits) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize ggml context");

    constexpr int n_stream = 2;
    constexpr int kv_size = 512;
    constexpr int n_groups_per_stream = kv_size / 128;
    constexpr int n_tokens_per_stream = 385;
    constexpr int n_tokens = n_tokens_per_stream * n_stream;
    constexpr int n_heads = 1;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 384 * n_stream);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream * n_stream);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false, 3);
    ggml_tensor * materialized = ggml_kvarn_materialize(
            ctx, records, stored, indices, n_tokens_per_stream, 0, n_stream, bits, false, 3);
    materialized->op_params[4] = 1;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, materialized);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate multi-stream KVarN tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            for (int d = 0; d < 128; ++d) {
                input[(s * n_tokens_per_stream + t) * 128 + d] =
                    std::sin(float(d) * 0.071f + float(s) * 0.31f) +
                    std::cos(float(t) * 0.037f + float(s) * 0.23f) +
                    float((d * 13 + t * 17 + s * 19) % 31 - 15) * 0.01f;
            }
        }
    }
    std::vector<int64_t> idx(n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            idx[s * n_tokens_per_stream + t] = int64_t(s * kv_size + t);
        }
    }
    std::vector<uint8_t> zeros(std::max(ggml_nbytes(stage), ggml_nbytes(records)), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, zeros.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, zeros.data(), 0, ggml_nbytes(records));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "multi-stream KVarN graph compute failed");

    const std::vector<float> output = test_kvarn_reference_decode_f32(
            records, stored, idx, n_tokens_per_stream, 0, n_stream, bits, false, 3);
    const std::vector<ggml_fp16_t> rotated_reference = test_kvarn_reference_decode(
            records, stored, idx, n_tokens_per_stream, 0, n_stream, bits, false, 3, true);
    std::vector<ggml_fp16_t> rotated_actual(ggml_nelements(materialized));
    ggml_backend_tensor_get(materialized, rotated_actual.data(), 0, ggml_nbytes(materialized));
    require_close_f16_rmse(rotated_reference, rotated_actual, 2e-3f,
            "multi-stream rotated KVarN materialization mismatch");

    for (int s = 0; s < n_stream; ++s) {
        double sink_error = 0.0;
        double compressed_error = 0.0;
        double previous_tail_error = 0.0;
        double live_tail_error = 0.0;
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            for (int d = 0; d < 128; ++d) {
                const size_t input_off = size_t(s * n_tokens_per_stream + t) * 128 + d;
                const size_t output_off = size_t(s * n_tokens_per_stream + t) * 128 + d;
                const double diff = double(input[input_off]) - double(output[output_off]);
                if (t < 128) {
                    sink_error += diff * diff;
                } else if (t < 256) {
                    compressed_error += diff * diff;
                } else if (t < 384) {
                    previous_tail_error += diff * diff;
                } else {
                    live_tail_error += diff * diff;
                }
            }
        }
        sink_error = std::sqrt(sink_error / (128 * 128));
        compressed_error = std::sqrt(compressed_error / (128 * 128));
        previous_tail_error = std::sqrt(previous_tail_error / (128 * 128));
        live_tail_error = std::sqrt(live_tail_error / 128);
        require(sink_error < 0.01, "multi-stream sink reconstruction error too high");
        require(compressed_error < 0.25, "multi-stream compressed reconstruction error too high");
        require(previous_tail_error < 0.01, "multi-stream previous tail reconstruction error too high");
        require(live_tail_error < 0.01, "multi-stream live tail reconstruction error too high");
    }

    std::vector<uint8_t> record_data(ggml_nbytes(records));
    ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());
    const size_t stream_record_bytes = size_t(record_bytes) * n_groups_per_stream * n_heads;
    for (int s = 0; s < n_stream; ++s) {
        const auto begin = record_data.begin() + ptrdiff_t(s * stream_record_bytes);
        const auto end = begin + ptrdiff_t(stream_record_bytes);
        require(std::any_of(begin, end, [](uint8_t v) { return v != 0; }),
                "multi-stream completed group was not flushed");
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// SWA sliding-window ring: write more tiles than the record ring holds so old
// slots are reused, then decode the live window. SWA has no permanent sink slot,
// so op_params[8] makes every stage group part of the local F16 tail. The older
// in-window tile comes from records
// whose ring slots were reused — a ring/seal bug would surface stale tiles and
// blow up the error.
static void test_cache_ops_swa(enum ggml_backend_dev_type device_type, bool required) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    const int bits = 4;
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "swa: failed to init ctx");

    constexpr int n_heads = 1;
    constexpr int stage_groups = 4;
    constexpr int tail_groups = 4;         // explicit SWA no-sink tail: tail == stage
    constexpr int gps = 1;                 // deduplicated record ring for the one sealed window tile
    constexpr int n_tiles = 10;            // tiles 0..9 -> ring wraps (tile 6 reuses tile 0's slot)
    constexpr int n_tokens = n_tiles * 128;
    constexpr int window_base = 5 * 128;   // window covers tiles 5..9
    constexpr int n_kv = 5 * 128;          // tile 5 sealed; tiles 6..9 live in staging
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, gps);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false, stage_groups);
    stored->op_params[4] = 1; // SWA ring store
    stored->op_params[8] = tail_groups;
    ggml_tensor * mat_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv);
    ggml_tensor * materialized = ggml_kvarn_materialize(
            ctx, records, stored, mat_indices, n_kv, 0, 1, bits, false, stage_groups);
    materialized->op_params[6] = 1;
    materialized->op_params[8] = tail_groups;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, materialized);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "swa: failed to allocate tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            input[t * 128 + d] =
                std::sin(float(d) * 0.071f) +
                std::cos(float(t) * 0.0037f) +
                float((d * 13 + t * 17) % 31 - 15) * 0.01f;
        }
    }
    std::vector<int64_t> idx(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        idx[i] = i; // absolute token position
    }
    std::vector<int64_t> mat_idx(n_kv);
    for (int cell = 0; cell < n_kv; ++cell) {
        mat_idx[cell] = window_base + cell; // window covers tiles 6..9
    }
    std::vector<uint8_t> zeros(ggml_nbytes(stage) + ggml_nbytes(records), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(mat_indices, mat_idx.data(), 0, ggml_nbytes(mat_indices));
    ggml_backend_tensor_set(stage, zeros.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, zeros.data(), 0, ggml_nbytes(records));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "swa: graph compute failed");

    const std::vector<float> output = test_kvarn_reference_decode_f32(
            records, stored, mat_idx, n_kv, 0, 1, bits, false, stage_groups, false, true);
    std::vector<ggml_fp16_t> materialized_data(ggml_nelements(materialized));
    ggml_backend_tensor_get(materialized, materialized_data.data(), 0, ggml_nbytes(materialized));
    for (size_t i = 0; i < output.size(); ++i) {
        require(std::abs(ggml_fp16_to_fp32(materialized_data[i]) - output[i]) < 2e-3f,
                "SWA KVarN materialization mismatch");
    }

    double sealed_error = 0.0; // tile 5 -> reused ring slot 0
    double live_error = 0.0;   // tiles 6..9 -> fp16 staging
    for (int cell = 0; cell < n_kv; ++cell) {
        const int abs_pos = window_base + cell;
        for (int d = 0; d < 128; ++d) {
            const double diff = double(input[abs_pos * 128 + d]) - double(output[cell * 128 + d]);
            if (cell < 128) {
                sealed_error += diff * diff;
            } else {
                live_error += diff * diff;
            }
        }
    }
    sealed_error = std::sqrt(sealed_error / (128 * 128));
    live_error = std::sqrt(live_error / (4 * 128 * 128));
    require(sealed_error < 0.25, "swa: sealed (wrapped) tile reconstruction error too high");
    require(live_error < 0.02, "swa: live tail reconstruction error too high");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static std::vector<ggml_fp16_t> test_store_reference_output(
        ggml_backend_t backend,
        int            bits,
        bool           value,
        int            n_stream,
        int            n_heads,
        int            n_tokens_per_stream,
        int            start_idx,
        bool           discontinuous_indices = false,
        bool           seed_stage = true,
        int            stage_groups = 3,
        int            head_slices = 1,
        int            striped_group_stride = 0,
        bool           eager_records = false) {
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize KVarN store parity context");

    const int logical_last = start_idx + n_tokens_per_stream - 1;
    const int striped_last = striped_group_stride > 0 ?
            ((logical_last / 128) * striped_group_stride + striped_group_stride - 1) * 128 +
                    logical_last % 128 : logical_last;
    const int n_kv = striped_last + 1 + (discontinuous_indices ? 1 : 0);
    const int n_groups_per_stream = std::max(4, (n_kv + 127) / 128);
    const int n_tokens = n_tokens_per_stream * n_stream;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups * n_stream);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream * n_stream);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, value, stage_groups);
    stored->op_params[3] = n_tokens_per_stream;
    stored->op_params[5] = head_slices;
    stored->op_params[9] = eager_records ? 1 : 0;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate KVarN store parity tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            for (int h = 0; h < n_heads; ++h) {
                for (int d = 0; d < 128; ++d) {
                    input[((s * n_tokens_per_stream + t) * n_heads + h) * 128 + d] =
                        std::sin(float(d) * 0.071f + float(h) * 0.13f + float(s) * 0.31f) +
                        std::cos(float(t) * 0.037f + float(h) * 0.11f + float(s) * 0.23f) +
                        float((d * 13 + h * 7 + t * 17 + s * 19) % 31 - 15) * 0.01f;
                }
            }
        }
    }

    std::vector<int64_t> idx(n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            const int logical_idx = start_idx + t;
            const int local_idx = striped_group_stride > 0 ?
                    ((logical_idx / 128) * striped_group_stride + striped_group_stride - 1) * 128 +
                            logical_idx % 128 :
                    logical_idx + (discontinuous_indices && t >= n_tokens_per_stream / 2 ? 1 : 0);
            idx[s * n_tokens_per_stream + t] = int64_t(s * n_groups_per_stream * 128 + local_idx);
        }
    }

    std::vector<uint8_t> record_zeros(ggml_nbytes(records), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    if (seed_stage) {
        std::vector<ggml_fp16_t> stage_data(ggml_nelements(stage));
        for (size_t i = 0; i < stage_data.size(); ++i) {
            const float f = std::sin(float(i) * 0.017f) + std::cos(float(i) * 0.011f);
            stage_data[i] = ggml_fp32_to_fp16(f);
        }
        ggml_backend_tensor_set(stage, stage_data.data(), 0, ggml_nbytes(stage));
    } else {
        std::vector<uint8_t> stage_zeros(ggml_nbytes(stage), 0);
        ggml_backend_tensor_set(stage, stage_zeros.data(), 0, ggml_nbytes(stage));
    }
    ggml_backend_tensor_set(records, record_zeros.data(), 0, record_zeros.size());

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "KVarN store path graph compute failed");

    std::vector<ggml_fp16_t> output = test_kvarn_reference_decode(
            records, stored, idx, n_kv, 0, n_stream, bits, value, stage_groups, false, false, head_slices);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return output;
}

static std::vector<ggml_fp16_t> test_store_segmented_output(
        ggml_backend_t backend,
        int            bits,
        bool           value,
        int            n_heads,
        int            head_slices,
        int            total_tokens,
        int            segment_tokens,
        int            stage_groups,
        bool           eager_records = false,
        bool           swa = false) {
    require(total_tokens > 0 && segment_tokens > 0 && total_tokens % segment_tokens == 0,
            "segmented store route requires even segments");
    require(n_heads > 0 && n_heads % head_slices == 0,
            "segmented store route invalid head slicing");

    ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "segmented store route: failed to initialize context");

    constexpr int n_stream = 1;
    const int n_segments = total_tokens / segment_tokens;
    const int groups_per_stream = std::max(64, (total_tokens + 127) / 128 + stage_groups + 2);
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups * n_stream);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, groups_per_stream * n_stream);

    std::vector<ggml_tensor *> current_segments;
    std::vector<ggml_tensor *> index_segments;
    current_segments.reserve(n_segments);
    index_segments.reserve(n_segments);

    ggml_tensor * stored = stage;
    for (int segment = 0; segment < n_segments; ++segment) {
        ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, segment_tokens);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, segment_tokens);
        current_segments.push_back(current);
        index_segments.push_back(indices);

        stored = ggml_kvarn_store(ctx, current, indices, stored, records, bits, 16, value, stage_groups);
        stored->op_params[3] = segment_tokens;
        stored->op_params[5] = head_slices;
        stored->op_params[9] = eager_records ? 1 : 0;
        stored->op_params[4] = swa ? 1 : 0;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "segmented store route: failed to allocate tensors");

    std::vector<uint8_t> stage_zeros(ggml_nbytes(stage), 0);
    std::vector<uint8_t> record_zeros(ggml_nbytes(records), 0);
    ggml_backend_tensor_set(stage, stage_zeros.data(), 0, stage_zeros.size());
    ggml_backend_tensor_set(records, record_zeros.data(), 0, record_zeros.size());

    for (int segment = 0; segment < n_segments; ++segment) {
        const int base_token = segment * segment_tokens;
        std::vector<float> input((size_t) 128 * n_heads * segment_tokens);
        std::vector<int64_t> indices(segment_tokens);
        for (int t = 0; t < segment_tokens; ++t) {
            const int abs_t = base_token + t;
            indices[t] = abs_t;
            for (int h = 0; h < n_heads; ++h) {
                const int logical_head = h / head_slices;
                const int slice = h % head_slices;
                for (int d = 0; d < 128; ++d) {
                    const int full_d = slice * 128 + d;
                    input[((size_t) t * n_heads + h) * 128 + d] =
                        0.74f * std::sin(float(full_d) * 0.011f + float(abs_t) * 0.019f + float(logical_head) * 0.037f) +
                        0.19f * std::cos(float(full_d) * 0.017f - float(abs_t) * 0.013f + float(logical_head) * 0.029f) +
                        float((full_d * 5 + h * 11 + abs_t * 7) % 41 - 20) * 0.006f;
                }
            }
        }
        ggml_backend_tensor_set(current_segments[segment], input.data(), 0, ggml_nbytes(current_segments[segment]));
        ggml_backend_tensor_set(index_segments[segment], indices.data(), 0, ggml_nbytes(index_segments[segment]));
    }

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "segmented store route: graph compute failed");

    std::vector<int64_t> full_indices(total_tokens);
    for (int t = 0; t < total_tokens; ++t) {
        full_indices[t] = t;
    }
    std::vector<ggml_fp16_t> output = test_kvarn_reference_decode(
            records, stored, full_indices, total_tokens, 0, n_stream, bits, value, stage_groups, false, swa, head_slices);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return output;
}

static void require_close_f16_rmse(
        const std::vector<ggml_fp16_t> & actual,
        const std::vector<ggml_fp16_t> & expected,
        float                            rmse_limit,
        const char *                     message) {
    require(actual.size() == expected.size(), "f16 RMSE parity size mismatch");
    double mse = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = double(ggml_fp16_to_fp32(actual[i])) - double(ggml_fp16_to_fp32(expected[i]));
        mse += diff * diff;
        max_diff = std::max(max_diff, std::fabs(diff));
    }

    const double rmse = std::sqrt(mse / double(actual.size()));
    if (!std::isfinite(rmse) || rmse > rmse_limit) {
        std::fprintf(stderr,
                "KVarN CPU/CUDA parity mismatch: rmse=%g max_diff=%g limit=%g\n",
                rmse, max_diff, double(rmse_limit));
        require(false, message);
    }
}

static void fill_hadamard_matrix_128(std::vector<float> & data) {
    constexpr int n = 128;
    data.assign(n * n, 0.0f);
    data[0] = 1.0f / std::sqrt(float(n));

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                const float val = data[i * n + j];

                data[(i + s) * n + j]       =  val;
                data[i * n + (j + s)]       =  val;
                data[(i + s) * n + (j + s)] = -val;
            }
        }
    }
}

static ggml_tensor * apply_hadamard_128(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * rot) {
    const int64_t n = rot->ne[0];
    ggml_tensor * res = nullptr;

    if (!ggml_is_contiguous(cur)) {
        res = ggml_cont_2d(ctx, cur, n, ggml_nelements(cur) / n);
    } else {
        res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur) / n);
    }
    res = ggml_mul_mat(ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    return ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
}

static ggml_tensor * apply_kvarn_wht_head(ggml_context * ctx, ggml_tensor * cur, int head_dim) {
    return ggml_kvarn_wht(ctx, cur, head_dim);
}

static std::vector<float> test_native_flash_attention_output(
        ggml_backend_t backend,
        bool           native_view,
        bool           rotate_graph,
        int            head_dim,
        int            bits_k,
        int            bits_v,
        int            n_q,
        int            n_q_heads = 1,
        int            n_kv_heads = 1,
        int            n_kv = 512,
        int            stage_groups = 3,
        bool           swa = false,
        std::vector<float> * body_meta_output = nullptr,
        bool           force_generic = false,
        int            exact_tail_tokens = 0,
        bool           original_value_domain = false,
        ggml_type      exact_tail_type = GGML_TYPE_F16,
        int            exact_tail_current_tokens = 0,
        bool           exact_tail_bodyless = false,
        bool           production_query_layout = false) {
    ggml_init_params params = {
        /*.mem_size   =*/ 32 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "native FA: failed to initialize ggml context");

    constexpr int n_stream   = 1;
    const int slices = head_dim / 128;
    const int record_heads = n_kv_heads * slices;
    const int groups_per_stream = std::max(4, (n_kv + 127) / 128);
    const int k_record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits_k) + 3 * 128 * sizeof(ggml_fp16_t));
    const int v_record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits_v) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * q_in = production_query_layout ?
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_q_heads, n_q, n_stream) :
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_q, n_q_heads, n_stream);
    const bool use_q_rot = native_view && rotate_graph;
    const bool use_output_rot = use_q_rot && !original_value_domain;
    ggml_tensor * q = use_q_rot ? apply_kvarn_wht_head(ctx, q_in, head_dim) : q_in;
    if (production_query_layout) {
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
    }
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv * n_stream);
    ggml_tensor * current_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, record_heads, n_kv * n_stream);
    ggml_tensor * current_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, record_heads, n_kv * n_stream);
    ggml_tensor * k_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, record_heads, 128 * stage_groups * n_stream);
    ggml_tensor * v_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, record_heads, 128 * stage_groups * n_stream);
    ggml_tensor * k_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, k_record_bytes, record_heads, groups_per_stream * n_stream);
    ggml_tensor * v_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, v_record_bytes, record_heads, groups_per_stream * n_stream);

    ggml_tensor * stored_k = ggml_kvarn_store(ctx, current_k, indices, k_stage, k_records, bits_k, 16, false, stage_groups);
    ggml_tensor * stored_v = ggml_kvarn_store(ctx, current_v, indices, v_stage, v_records, bits_v, 16, true,  stage_groups);
    stored_k->op_params[3] = n_kv;
    stored_v->op_params[3] = n_kv;
    stored_k->op_params[5] = slices;
    stored_v->op_params[5] = slices;
    if (native_view && n_kv >= 3*128) {
        using workspace_fn = size_t (*)(ggml_backend_dev_t, const ggml_tensor *);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        auto * workspace_y = reg ? reinterpret_cast<workspace_fn>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_kvarn_workspace_y_size")) : nullptr;
        require(workspace_y == nullptr || workspace_y(dev, stored_k) > 0,
                "KVarN eager-store backend workspace estimate is missing");
        if (workspace_y != nullptr) {
            stored_k->op_params[3] = 0;
            require(workspace_y(dev, stored_k) > 0,
                    "KVarN reserve estimate must cover a full ubatch without runtime slot hints");
            stored_k->op_params[3] = n_kv;
        }
    }
    if (swa) {
        stored_k->op_params[4] = 1;
        stored_v->op_params[4] = 1;
    }

    ggml_tensor * k_ref = native_view ? nullptr : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, n_kv_heads, n_kv, n_stream);
    ggml_tensor * v_ref = native_view ? nullptr : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, n_kv_heads, n_kv, n_stream);
    ggml_tensor * k = native_view ?
        ggml_kvarn_view(ctx, k_records, stored_k, indices, n_kv, 0, n_stream, bits_k, false, stage_groups) : k_ref;
    ggml_tensor * v = native_view ?
        ggml_kvarn_view(ctx, v_records, stored_v, indices, n_kv, 0, n_stream, bits_v, true,  stage_groups) : v_ref;

    if (native_view && swa) {
        k->op_params[6] = 1;
        v->op_params[6] = 1;
    }

    if (native_view && slices > 1) {
        k = ggml_reshape_4d(ctx, k, head_dim, n_kv_heads, n_kv, n_stream);
        v = ggml_reshape_4d(ctx, v, head_dim, n_kv_heads, n_kv, n_stream);
    }
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, 1, n_stream);
    ggml_tensor * sinks = force_generic ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_q_heads) : nullptr;
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / std::sqrt(float(head_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(out, sinks);
    if (native_view) {
        out->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN] =
            rotate_graph ? (original_value_domain ?
                GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V :
                GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED) :
            GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL;
    }
    ggml_tensor * body_meta = nullptr;
    if (body_meta_output != nullptr && native_view) {
        body_meta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, n_q_heads, n_q, n_stream);
        out->src[8] = body_meta;
    }

    ggml_tensor * k_tail_storage = nullptr;
    ggml_tensor * v_tail_storage = nullptr;
    ggml_tensor * tail_mask = nullptr;
    ggml_tensor * query_order = nullptr;
    ggml_tensor * run_desc = nullptr;
    ggml_tensor * k_tail_current_storage = nullptr;
    ggml_tensor * v_tail_current_storage = nullptr;
    if (exact_tail_tokens > 0) {
        require(body_meta_output == nullptr,
                "native FA: exact-tail attention owns the body metadata output");
        require(exact_tail_type == GGML_TYPE_F16 ||
                exact_tail_type == GGML_TYPE_BF16,
                "native FA: unsupported exact-tail test type");
        require(exact_tail_current_tokens >= 0 &&
                exact_tail_current_tokens < exact_tail_tokens,
                "native FA: invalid compact current-tail width");
        const int history_tail_tokens = exact_tail_tokens - exact_tail_current_tokens;
        const int arena_stride = exact_tail_current_tokens > 0 ?
                history_tail_tokens : int(GGML_PAD(exact_tail_tokens + n_q, 256));
        k_tail_storage = ggml_new_tensor_4d(
                ctx, exact_tail_type, head_dim, n_kv_heads, arena_stride, n_stream);
        v_tail_storage = ggml_new_tensor_4d(
                ctx, exact_tail_type, head_dim, n_kv_heads, arena_stride, n_stream);
        ggml_tensor * k_tail_source = use_q_rot ?
            apply_kvarn_wht_head(ctx, k_tail_storage, head_dim) : k_tail_storage;
        ggml_tensor * v_tail_source = use_output_rot ?
            apply_kvarn_wht_head(ctx, v_tail_storage, head_dim) : v_tail_storage;
        ggml_tensor * k_tail = ggml_permute(ctx, k_tail_source, 0, 2, 1, 3);
        ggml_tensor * v_tail = ggml_permute(ctx, v_tail_source, 0, 2, 1, 3);
        ggml_tensor * k_tail_current = nullptr;
        ggml_tensor * v_tail_current = nullptr;
        if (exact_tail_current_tokens > 0) {
            k_tail_current_storage = ggml_new_tensor_4d(
                    ctx, exact_tail_type, head_dim, n_kv_heads, exact_tail_current_tokens, 1);
            v_tail_current_storage = ggml_new_tensor_4d(
                    ctx, exact_tail_type, head_dim, n_kv_heads, exact_tail_current_tokens, 1);
            ggml_tensor * k_current_source = use_q_rot ?
                apply_kvarn_wht_head(ctx, k_tail_current_storage, head_dim) : k_tail_current_storage;
            ggml_tensor * v_current_source = use_output_rot ?
                apply_kvarn_wht_head(ctx, v_tail_current_storage, head_dim) : v_tail_current_storage;
            k_tail_current = ggml_permute(ctx, k_current_source, 0, 2, 1, 3);
            v_tail_current = ggml_permute(ctx, v_current_source, 0, 2, 1, 3);
        }
        tail_mask = ggml_new_tensor_4d(
                ctx, GGML_TYPE_F16, exact_tail_tokens, n_q, 1, n_stream);
        query_order = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_q, n_stream);
        run_desc = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6 + exact_tail_tokens, n_stream);
        if (k_tail_current != nullptr) {
            out = ggml_kv_tail_attention_merge_segmented(
                    ctx, out, k_tail, v_tail, k_tail_current, v_tail_current,
                    tail_mask, query_order, run_desc);
            ggml_flash_attn_ext_set_kv_tail_history_slots(out, history_tail_tokens);
        } else {
            out = ggml_kv_tail_attention_merge(
                    ctx, out, k_tail, v_tail, tail_mask, query_order, run_desc);
        }
        if (exact_tail_bodyless) {
            ggml_flash_attn_ext_set_kv_tail_bodyless(out);
        }
        if (native_view && k_tail_current != nullptr) {
            require(ggml_backend_supports_op(backend, out),
                    "native compact segmented KVarN tail was rejected by its backend");
        }
    }
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    if (use_output_rot) {
        out = apply_kvarn_wht_head(ctx, out, head_dim);
    }

    ggml_cgraph * store_graph = nullptr;
    if (!native_view) {
        store_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(store_graph, stored_k);
        ggml_build_forward_expand(store_graph, stored_v);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "native FA: failed to allocate tensors");

    std::vector<float> q_data((size_t) head_dim * n_q * n_q_heads * n_stream);
    for (int qh = 0; qh < n_q_heads; ++qh) {
        for (int iq = 0; iq < n_q; ++iq) {
            for (int d = 0; d < head_dim; ++d) {
                q_data[((size_t) qh * n_q + iq) * head_dim + d] =
                    0.09f * std::sin(float(d) * 0.017f + float(iq) * 0.13f + float(qh) * 0.021f) +
                    0.07f * std::cos(float(d) * 0.031f - float(iq) * 0.05f + float(qh) * 0.033f);
            }
        }
    }

    std::vector<float> k_data((size_t) 128 * record_heads * n_kv * n_stream);
    std::vector<float> v_data(k_data.size());
    for (int t = 0; t < n_kv; ++t) {
        for (int h = 0; h < n_kv_heads; ++h) {
            for (int slice = 0; slice < slices; ++slice) {
                const int record_head = h * slices + slice;
                for (int d = 0; d < 128; ++d) {
                    const int full_d = slice * 128 + d;
                    const size_t off = ((size_t) t * record_heads + record_head) * 128 + d;
                    k_data[off] =
                        0.80f * std::sin(float(full_d) * 0.011f + float(t) * 0.021f) +
                        0.10f * std::cos(float(t) * 0.009f + float(h) * 0.17f);
                    v_data[off] =
                        0.75f * std::cos(float(full_d) * 0.013f - float(t) * 0.019f) +
                        0.08f * std::sin(float(t) * 0.015f + float(h) * 0.23f);
                }
            }
        }
    }

    std::vector<int64_t> idx(n_kv * n_stream);
    for (int i = 0; i < n_kv * n_stream; ++i) {
        idx[i] = i;
    }

    std::vector<ggml_fp16_t> mask_data((size_t) n_kv * n_q * n_stream);
    for (int iq = 0; iq < n_q; ++iq) {
        for (int ikv = 0; ikv < n_kv; ++ikv) {
            mask_data[(size_t) iq * n_kv + ikv] = ggml_fp32_to_fp16(
                    !exact_tail_bodyless && ikv <= iq + n_kv - n_q ? 0.0f : -INFINITY);
        }
    }

    std::vector<uint8_t> k_stage_zeros(ggml_nbytes(k_stage), 0);
    std::vector<uint8_t> v_stage_zeros(ggml_nbytes(v_stage), 0);
    std::vector<uint8_t> k_record_zeros(ggml_nbytes(k_records), 0);
    std::vector<uint8_t> v_record_zeros(ggml_nbytes(v_records), 0);

    ggml_backend_tensor_set(q_in, q_data.data(), 0, ggml_nbytes(q_in));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(current_k, k_data.data(), 0, ggml_nbytes(current_k));
    ggml_backend_tensor_set(current_v, v_data.data(), 0, ggml_nbytes(current_v));
    ggml_backend_tensor_set(k_stage, k_stage_zeros.data(), 0, k_stage_zeros.size());
    ggml_backend_tensor_set(v_stage, v_stage_zeros.data(), 0, v_stage_zeros.size());
    ggml_backend_tensor_set(k_records, k_record_zeros.data(), 0, k_record_zeros.size());
    ggml_backend_tensor_set(v_records, v_record_zeros.data(), 0, v_record_zeros.size());
    ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask));

    if (exact_tail_tokens > 0) {
        std::vector<float> k_tail_data(ggml_nelements(k_tail_storage), 0.0f);
        std::vector<float> v_tail_data(ggml_nelements(v_tail_storage), 0.0f);
        std::vector<float> k_tail_current_data(
                k_tail_current_storage ? ggml_nelements(k_tail_current_storage) : 0, 0.0f);
        std::vector<float> v_tail_current_data(
                v_tail_current_storage ? ggml_nelements(v_tail_current_storage) : 0, 0.0f);
        const int history_tail_tokens = exact_tail_tokens - exact_tail_current_tokens;
        for (int t = 0; t < exact_tail_tokens; ++t) {
            for (int h = 0; h < n_kv_heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    const int source_t = t < history_tail_tokens ? t : t - history_tail_tokens;
                    const size_t off = ((size_t) source_t * n_kv_heads + h) * head_dim + d;
                    const float k_value =
                        0.67f * std::sin(float(d) * 0.014f + float(t) * 0.023f) +
                        0.11f * std::cos(float(t) * 0.007f + float(h) * 0.19f);
                    const float v_value =
                        0.63f * std::cos(float(d) * 0.018f - float(t) * 0.017f) +
                        0.09f * std::sin(float(t) * 0.012f + float(h) * 0.29f);
                    if (t < history_tail_tokens) {
                        k_tail_data[off] = k_value;
                        v_tail_data[off] = v_value;
                    } else {
                        k_tail_current_data[off] = k_value;
                        v_tail_current_data[off] = v_value;
                    }
                }
            }
        }
        std::vector<ggml_fp16_t> tail_mask_data(ggml_nelements(tail_mask), ggml_fp32_to_fp16(-INFINITY));
        for (int iq = 0; iq < n_q; ++iq) {
            const int last_visible = exact_tail_tokens - n_q + iq;
            for (int t = 0; t < exact_tail_tokens; ++t) {
                if (t <= last_visible &&
                        (exact_tail_bodyless ||
                         exact_tail_current_tokens == 0 ||
                         t >= history_tail_tokens)) {
                    tail_mask_data[(size_t) iq * exact_tail_tokens + t] = ggml_fp32_to_fp16(0.0f);
                }
            }
        }
        std::vector<int32_t> query_order_data(n_q);
        std::iota(query_order_data.begin(), query_order_data.end(), 0);
        std::vector<int32_t> run_desc_data(6 + exact_tail_tokens, -1);
        run_desc_data[0] = 0;
        run_desc_data[1] = 0;
        run_desc_data[2] = n_q;
        run_desc_data[3] = 1;
        run_desc_data[4] = exact_tail_tokens;
        run_desc_data[5] = -1;
        for (int t = 0; t < exact_tail_tokens; ++t) {
            run_desc_data[6 + t] = t;
        }
        const auto upload_tail = [](ggml_tensor * tensor, const std::vector<float> & data) {
            if (tensor->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> converted(data.size());
                for (size_t i = 0; i < data.size(); ++i) {
                    converted[i] = ggml_fp32_to_fp16(data[i]);
                }
                ggml_backend_tensor_set(
                    tensor, converted.data(), 0, converted.size() * sizeof(converted[0]));
            } else {
                std::vector<ggml_bf16_t> converted(data.size());
                for (size_t i = 0; i < data.size(); ++i) {
                    converted[i] = ggml_fp32_to_bf16(data[i]);
                }
                ggml_backend_tensor_set(
                    tensor, converted.data(), 0, converted.size() * sizeof(converted[0]));
            }
        };
        upload_tail(k_tail_storage, k_tail_data);
        upload_tail(v_tail_storage, v_tail_data);
        if (k_tail_current_storage != nullptr) {
            upload_tail(k_tail_current_storage, k_tail_current_data);
            upload_tail(v_tail_current_storage, v_tail_current_data);
        }
        ggml_backend_tensor_set(tail_mask, tail_mask_data.data(), 0, ggml_nbytes(tail_mask));
        ggml_backend_tensor_set(query_order, query_order_data.data(), 0, ggml_nbytes(query_order));
        ggml_backend_tensor_set(run_desc, run_desc_data.data(), 0, ggml_nbytes(run_desc));
    }

    if (!native_view) {
        require(ggml_backend_graph_compute(backend, store_graph) == GGML_STATUS_SUCCESS,
                "native FA: reference store graph compute failed");
        const std::vector<ggml_fp16_t> k_ref_data = test_kvarn_reference_decode(
                k_records, stored_k, idx, n_kv, 0, n_stream, bits_k, false, stage_groups, false, swa, slices);
        const std::vector<ggml_fp16_t> v_ref_data = test_kvarn_reference_decode(
                v_records, stored_v, idx, n_kv, 0, n_stream, bits_v, true, stage_groups, false, swa, slices);
        ggml_backend_tensor_set(k_ref, k_ref_data.data(), 0, ggml_nbytes(k_ref));
        ggml_backend_tensor_set(v_ref, v_ref_data.data(), 0, ggml_nbytes(v_ref));
    }

    if (sinks != nullptr) {
        std::vector<float> sink_data(size_t(n_q_heads), -INFINITY);
        ggml_backend_tensor_set(sinks, sink_data.data(), 0, ggml_nbytes(sinks));
    }

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            native_view ? "native FA: native-view graph compute failed" : "native FA: reference graph compute failed");

    std::vector<float> output(ggml_nelements(out));
    ggml_backend_tensor_get(out, output.data(), 0, ggml_nbytes(out));
    if (body_meta_output != nullptr) {
        require(native_view, "native FA: body metadata reference requires a native CUDA view");
        body_meta_output->resize(ggml_nelements(body_meta));
        ggml_backend_tensor_get(body_meta, body_meta_output->data(), 0, ggml_nbytes(body_meta));
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return output;
}

static std::vector<float> test_native_flash_attention_segmented_output(
        ggml_backend_t backend,
        bool           native_view,
        int            head_dim,
        int            bits_k,
        int            bits_v,
        int            n_q,
        int            n_q_heads,
        int            n_kv_heads,
        int            n_segments,
        int            segment_tokens,
        int            stage_groups,
        bool           swa = false,
        int            n_kv_override = 0,
        bool           ring_view = false,
        int            sliding_window = 0,
        float          attn_scale = 0.0f,
        bool           scramble_view = false,
        int            tail_groups_override = 0,
        bool           mixed_domain = false,
        bool           eager_records = false,
        bool           require_body_meta = false) {
    ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "segmented native FA: failed to initialize ggml context");

    constexpr int n_stream = 1;
    const int n_total_tokens = n_segments * segment_tokens;
    const int n_kv = n_kv_override > 0 ? n_kv_override : n_total_tokens;
    require(n_kv <= n_total_tokens, "segmented native FA: view cannot exceed stored tokens");
    require(n_q <= segment_tokens, "segmented native FA: q width must fit in final segment");
    require(segment_tokens > 0 && (segment_tokens % 128) == 0, "segmented native FA: segment size must be tile aligned");
    const int slices = head_dim / 128;
    const int record_heads = n_kv_heads * slices;
    const int tail_groups = tail_groups_override > 0 ? tail_groups_override : stage_groups - 1;
    require(tail_groups > 0 && tail_groups <= stage_groups,
            "segmented native FA: invalid tail_groups override");
    const int groups_per_stream = std::max(4, (n_kv + 127) / 128 + (swa ? 2 : 0));
    const int k_record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits_k) + 3 * 128 * sizeof(ggml_fp16_t));
    const int v_record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits_v) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * q_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_q, n_q_heads, n_stream);
    ggml_tensor * q = mixed_domain ? apply_kvarn_wht_head(ctx, q_in, head_dim) : q_in;
    ggml_tensor * k_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, record_heads, 128 * stage_groups * n_stream);
    ggml_tensor * v_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, record_heads, 128 * stage_groups * n_stream);
    ggml_tensor * k_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, k_record_bytes, record_heads, groups_per_stream * n_stream);
    ggml_tensor * v_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, v_record_bytes, record_heads, groups_per_stream * n_stream);
    ggml_tensor * view_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv * n_stream);

    std::vector<ggml_tensor *> current_k_segments;
    std::vector<ggml_tensor *> current_v_segments;
    std::vector<ggml_tensor *> indices_segments;
    current_k_segments.reserve(n_segments);
    current_v_segments.reserve(n_segments);
    indices_segments.reserve(n_segments);

    ggml_tensor * stored_k = k_stage;
    ggml_tensor * stored_v = v_stage;
    for (int segment = 0; segment < n_segments; ++segment) {
        ggml_tensor * current_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, record_heads, segment_tokens * n_stream);
        ggml_tensor * current_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, record_heads, segment_tokens * n_stream);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, segment_tokens * n_stream);
        current_k_segments.push_back(current_k);
        current_v_segments.push_back(current_v);
        indices_segments.push_back(indices);

        stored_k = ggml_kvarn_store(ctx, current_k, indices, stored_k, k_records, bits_k, 16, false, stage_groups);
        stored_v = ggml_kvarn_store(ctx, current_v, indices, stored_v, v_records, bits_v, 16, true,  stage_groups);
        stored_k->op_params[3] = segment_tokens;
        stored_v->op_params[3] = segment_tokens;
        stored_k->op_params[5] = slices;
        stored_v->op_params[5] = slices;
        stored_k->op_params[9] = eager_records ? 1 : 0;
        stored_v->op_params[9] = eager_records ? 1 : 0;
        if (swa) {
            stored_k->op_params[4] = 1;
            stored_v->op_params[4] = 1;
        }
        if (tail_groups_override > 0) {
            stored_k->op_params[8] = tail_groups;
            stored_v->op_params[8] = tail_groups;
        }
    }

    ggml_tensor * k_ref = native_view ? nullptr : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, n_kv_heads, n_kv, n_stream);
    ggml_tensor * v_ref = native_view ? nullptr : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, n_kv_heads, n_kv, n_stream);
    ggml_tensor * k = native_view ?
        ggml_kvarn_view(ctx, k_records, stored_k, view_indices, n_kv, 0, n_stream, bits_k, false, stage_groups) : k_ref;
    ggml_tensor * v = native_view ?
        ggml_kvarn_view(ctx, v_records, stored_v, view_indices, n_kv, 0, n_stream, bits_v, true,  stage_groups) : v_ref;
    if (native_view && swa) {
        k->op_params[6] = 1;
        v->op_params[6] = 1;
    }
    if (native_view && tail_groups_override > 0) {
        k->op_params[8] = tail_groups;
        v->op_params[8] = tail_groups;
    }

    if (native_view && slices > 1) {
        k = ggml_reshape_4d(ctx, k, head_dim, n_kv_heads, n_kv, n_stream);
        v = ggml_reshape_4d(ctx, v, head_dim, n_kv_heads, n_kv, n_stream);
    }
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, 1, n_stream);
    const float scale = attn_scale > 0.0f ? attn_scale : 1.0f / std::sqrt(float(head_dim));
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    if (native_view) {
        out->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN] = mixed_domain ?
            GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V :
            GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL;
    }
    ggml_tensor * body_meta = nullptr;
    if (require_body_meta) {
        require(native_view, "segmented native FA: body metadata requires a native view");
        body_meta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, n_q_heads, n_q, n_stream);
        out->src[8] = body_meta;
    }
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    ggml_cgraph * store_graph = nullptr;
    if (!native_view) {
        store_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(store_graph, stored_k);
        ggml_build_forward_expand(store_graph, stored_v);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "segmented native FA: failed to allocate tensors");

    std::vector<float> q_data((size_t) head_dim * n_q * n_q_heads * n_stream);
    for (int qh = 0; qh < n_q_heads; ++qh) {
        for (int iq = 0; iq < n_q; ++iq) {
            const int abs_q = n_total_tokens - n_q + iq;
            for (int d = 0; d < head_dim; ++d) {
                q_data[((size_t) qh * n_q + iq) * head_dim + d] =
                    0.08f * std::sin(float(d) * 0.019f + float(abs_q) * 0.037f + float(qh) * 0.023f) +
                    0.06f * std::cos(float(d) * 0.041f - float(abs_q) * 0.029f + float(qh) * 0.017f);
            }
        }
    }

    std::vector<std::vector<float>> k_segment_data;
    std::vector<std::vector<float>> v_segment_data;
    std::vector<std::vector<int64_t>> idx_segment_data;
    k_segment_data.reserve(n_segments);
    v_segment_data.reserve(n_segments);
    idx_segment_data.reserve(n_segments);

    for (int segment = 0; segment < n_segments; ++segment) {
        const int base_token = segment * segment_tokens;
        std::vector<float> k_data((size_t) 128 * record_heads * segment_tokens * n_stream);
        std::vector<float> v_data(k_data.size());
        std::vector<int64_t> idx(segment_tokens * n_stream);

        for (int t = 0; t < segment_tokens; ++t) {
            const int abs_t = base_token + t;
            idx[t] = int64_t(abs_t);
            for (int h = 0; h < n_kv_heads; ++h) {
                for (int slice = 0; slice < slices; ++slice) {
                    const int record_head = h * slices + slice;
                    for (int d = 0; d < 128; ++d) {
                        const int full_d = slice * 128 + d;
                        const size_t off = ((size_t) t * record_heads + record_head) * 128 + d;
                        k_data[off] =
                            0.82f * std::sin(float(full_d) * 0.010f + float(abs_t) * 0.023f) +
                            0.09f * std::cos(float(abs_t) * 0.007f + float(h) * 0.19f);
                        v_data[off] =
                            0.73f * std::cos(float(full_d) * 0.012f - float(abs_t) * 0.017f) +
                            0.10f * std::sin(float(abs_t) * 0.013f + float(h) * 0.29f);
                    }
                }
            }
        }

        k_segment_data.push_back(std::move(k_data));
        v_segment_data.push_back(std::move(v_data));
        idx_segment_data.push_back(std::move(idx));
    }

    std::vector<int64_t> view_idx(n_kv * n_stream);
    const int view_base = n_total_tokens - n_kv;
    if (ring_view) {
        const int last = n_total_tokens - 1;
        for (int i = 0; i < n_kv * n_stream; ++i) {
            int abs_pos = last - ((last - i) % n_kv + n_kv) % n_kv;
            if (abs_pos < view_base) {
                abs_pos -= n_kv;
            }
            view_idx[i] = abs_pos >= view_base ? int64_t(abs_pos) : int64_t(-1);
        }
    } else {
        for (int i = 0; i < n_kv * n_stream; ++i) {
            view_idx[i] = int64_t(view_base + i);
        }
    }
    if (scramble_view && n_kv >= 128) {
        std::swap(view_idx[1], view_idx[2]);
        std::swap(view_idx[65], view_idx[66]);
    }

    std::vector<ggml_fp16_t> mask_data((size_t) n_kv * n_q * n_stream);
    for (int iq = 0; iq < n_q; ++iq) {
        const int abs_q = n_total_tokens - n_q + iq;
        for (int ikv = 0; ikv < n_kv; ++ikv) {
            const bool visible = view_idx[ikv] <= abs_q &&
                (sliding_window <= 0 || abs_q - view_idx[ikv] < sliding_window);
            mask_data[(size_t) iq * n_kv + ikv] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
        }
    }

    std::vector<uint8_t> k_stage_zeros(ggml_nbytes(k_stage), 0);
    std::vector<uint8_t> v_stage_zeros(ggml_nbytes(v_stage), 0);
    std::vector<uint8_t> k_record_zeros(ggml_nbytes(k_records), 0);
    std::vector<uint8_t> v_record_zeros(ggml_nbytes(v_records), 0);

    ggml_backend_tensor_set(q_in, q_data.data(), 0, ggml_nbytes(q_in));
    ggml_backend_tensor_set(k_stage, k_stage_zeros.data(), 0, k_stage_zeros.size());
    ggml_backend_tensor_set(v_stage, v_stage_zeros.data(), 0, v_stage_zeros.size());
    ggml_backend_tensor_set(k_records, k_record_zeros.data(), 0, k_record_zeros.size());
    ggml_backend_tensor_set(v_records, v_record_zeros.data(), 0, v_record_zeros.size());
    ggml_backend_tensor_set(view_indices, view_idx.data(), 0, ggml_nbytes(view_indices));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask));

    for (int segment = 0; segment < n_segments; ++segment) {
        ggml_backend_tensor_set(current_k_segments[segment], k_segment_data[segment].data(), 0, ggml_nbytes(current_k_segments[segment]));
        ggml_backend_tensor_set(current_v_segments[segment], v_segment_data[segment].data(), 0, ggml_nbytes(current_v_segments[segment]));
        ggml_backend_tensor_set(indices_segments[segment], idx_segment_data[segment].data(), 0, ggml_nbytes(indices_segments[segment]));
    }

    if (!native_view) {
        require(ggml_backend_graph_compute(backend, store_graph) == GGML_STATUS_SUCCESS,
                "segmented native FA: reference store graph compute failed");
        const std::vector<ggml_fp16_t> k_ref_data = test_kvarn_reference_decode(
                k_records, stored_k, view_idx, n_kv, 0, n_stream, bits_k, false, stage_groups, mixed_domain, swa, slices);
        const std::vector<ggml_fp16_t> v_ref_data = test_kvarn_reference_decode(
                v_records, stored_v, view_idx, n_kv, 0, n_stream, bits_v, true, stage_groups, false, swa, slices);
        ggml_backend_tensor_set(k_ref, k_ref_data.data(), 0, ggml_nbytes(k_ref));
        ggml_backend_tensor_set(v_ref, v_ref_data.data(), 0, ggml_nbytes(v_ref));
    }

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            native_view ? "segmented native FA: native-view graph compute failed" : "segmented native FA: reference graph compute failed");

    std::vector<float> output(ggml_nelements(out));
    ggml_backend_tensor_get(out, output.data(), 0, ggml_nbytes(out));
    if (body_meta) {
        std::vector<float> meta(ggml_nelements(body_meta));
        ggml_backend_tensor_get(body_meta, meta.data(), 0, ggml_nbytes(body_meta));
        for (size_t row = 0; row < meta.size()/2; ++row) {
            require(std::isfinite(meta[2*row]) && std::isfinite(meta[2*row + 1]) && meta[2*row + 1] > 0.0f,
                    "segmented native FA: body max/rowsum metadata was not emitted");
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return output;
}

static void require_close_f32_rmse(
        const std::vector<float> & actual,
        const std::vector<float> & expected,
        float                      rmse_limit,
        const char *               message) {
    require(actual.size() == expected.size(), "f32 RMSE parity size mismatch");
    double mse = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "f32 parity output contained non-finite value");
        const double diff = double(actual[i]) - double(expected[i]);
        mse += diff * diff;
        max_diff = std::max(max_diff, std::fabs(diff));
    }

    const double rmse = std::sqrt(mse / double(actual.size()));
    if (!std::isfinite(rmse) || rmse > rmse_limit) {
        std::fprintf(stderr,
                "KVarN native FA parity mismatch: rmse=%g max_diff=%g limit=%g\n",
                rmse, max_diff, double(rmse_limit));
        require(false, message);
    }
}

static void require_segmented_raw_roundtrip(
        ggml_backend_t backend,
        int            head_dim,
        int            bits,
        bool           value,
        int            n_heads,
        int            n_segments,
        int            segment_tokens,
        int            stage_groups,
        float          rmse_limit,
        const char *   message,
        bool           swa = false,
        int            n_kv_override = 0,
        bool           ring_view = false,
        int            tail_groups_override = 0) {
    ggml_init_params params = {
        /*.mem_size   =*/ 32 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "segmented raw roundtrip: failed to initialize ggml context");

    const int n_tokens = n_segments * segment_tokens;
    const int n_kv = n_kv_override > 0 ? n_kv_override : n_tokens;
    require(n_kv <= n_tokens, "segmented raw roundtrip: view cannot exceed stored tokens");
    const int slices = head_dim / 128;
    const int record_heads = n_heads * slices;
    const int tail_groups = tail_groups_override > 0 ? tail_groups_override : stage_groups - 1;
    require(tail_groups > 0 && tail_groups <= stage_groups,
            "segmented raw roundtrip: invalid tail_groups override");
    const int groups_per_stream = std::max(4, (swa ? (n_kv + 127) / 128 + 2 : (n_tokens + 127) / 128));
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, record_heads, 128 * stage_groups);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, record_heads, groups_per_stream);

    std::vector<ggml_tensor *> currents;
    std::vector<ggml_tensor *> indices_tensors;
    currents.reserve(n_segments);
    indices_tensors.reserve(n_segments);

    ggml_tensor * stored = stage;
    for (int segment = 0; segment < n_segments; ++segment) {
        ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, record_heads, segment_tokens);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, segment_tokens);
        currents.push_back(current);
        indices_tensors.push_back(indices);

        stored = ggml_kvarn_store(ctx, current, indices, stored, records, bits, 16, value, stage_groups);
        stored->op_params[3] = segment_tokens;
        stored->op_params[5] = slices;
        if (swa) {
            stored->op_params[4] = 1;
        }
        if (tail_groups_override > 0) {
            stored->op_params[8] = tail_groups;
        }
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "segmented raw roundtrip: failed to allocate tensors");

    std::vector<float> expected((size_t) n_tokens * n_heads * head_dim);
    std::vector<std::vector<float>> segment_data;
    std::vector<std::vector<int64_t>> idx_data;
    segment_data.reserve(n_segments);
    idx_data.reserve(n_segments);

    for (int segment = 0; segment < n_segments; ++segment) {
        const int base_token = segment * segment_tokens;
        std::vector<float> data((size_t) 128 * record_heads * segment_tokens);
        std::vector<int64_t> idx(segment_tokens);
        for (int t = 0; t < segment_tokens; ++t) {
            const int abs_t = base_token + t;
            idx[t] = int64_t(abs_t);
            for (int h = 0; h < n_heads; ++h) {
                for (int slice = 0; slice < slices; ++slice) {
                    const int record_head = h * slices + slice;
                    for (int d = 0; d < 128; ++d) {
                        const int full_d = slice * 128 + d;
                        const float x =
                            0.67f * std::sin(float(full_d) * 0.013f + float(abs_t) * 0.017f + float(h) * 0.07f) +
                            0.21f * std::cos(float(full_d) * 0.037f - float(abs_t) * 0.011f + float(h) * 0.13f);
                        data[((size_t) t * record_heads + record_head) * 128 + d] = x;
                        expected[((size_t) abs_t * n_heads + h) * head_dim + full_d] = x;
                    }
                }
            }
        }
        segment_data.push_back(std::move(data));
        idx_data.push_back(std::move(idx));
    }

    std::vector<uint8_t> stage_zeros(ggml_nbytes(stage), 0);
    std::vector<uint8_t> record_zeros(ggml_nbytes(records), 0);
    ggml_backend_tensor_set(stage, stage_zeros.data(), 0, stage_zeros.size());
    ggml_backend_tensor_set(records, record_zeros.data(), 0, record_zeros.size());
    for (int segment = 0; segment < n_segments; ++segment) {
        ggml_backend_tensor_set(currents[segment], segment_data[segment].data(), 0, ggml_nbytes(currents[segment]));
        ggml_backend_tensor_set(indices_tensors[segment], idx_data[segment].data(), 0, ggml_nbytes(indices_tensors[segment]));
    }

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "segmented raw roundtrip: store graph compute failed");

    std::vector<int64_t> view_idx;
    if (swa) {
        view_idx.resize(n_kv);
        const int view_base = n_tokens - n_kv;
        if (ring_view) {
            const int last = n_tokens - 1;
            for (int i = 0; i < n_kv; ++i) {
                int abs_pos = last - ((last - i) % n_kv + n_kv) % n_kv;
                if (abs_pos < view_base) {
                    abs_pos -= n_kv;
                }
                view_idx[i] = abs_pos >= view_base ? int64_t(abs_pos) : int64_t(-1);
            }
        } else {
            for (int i = 0; i < n_kv; ++i) {
                view_idx[i] = int64_t(view_base + i);
            }
        }
    }

    const std::vector<ggml_fp16_t> decoded = test_kvarn_reference_decode(
            records, stored, swa ? view_idx : idx_data.back(), n_kv, 0, 1, bits, value, stage_groups, false, swa, slices);
    double mse = 0.0;
    double max_diff = 0.0;
    size_t n_checked = 0;
    for (int t = 0; t < n_kv; ++t) {
        const int expected_token = swa ? int(view_idx[t]) : t;
        if (expected_token < 0) {
            continue;
        }
        for (int h = 0; h < n_heads; ++h) {
            for (int slice = 0; slice < slices; ++slice) {
                const int record_head = h * slices + slice;
                for (int d = 0; d < 128; ++d) {
                    const int full_d = slice * 128 + d;
                    const size_t decoded_off = (size_t) d + (size_t) record_head * 128 +
                        (size_t) t * 128 * record_heads;
                    const size_t expected_off = ((size_t) expected_token * n_heads + h) * head_dim + full_d;
                    const double diff = double(ggml_fp16_to_fp32(decoded[decoded_off])) - double(expected[expected_off]);
                    mse += diff * diff;
                    ++n_checked;
                    max_diff = std::max(max_diff, std::fabs(diff));
                }
            }
        }
    }
    require(n_checked > 0, "segmented raw roundtrip: no cells checked");

    const double rmse = std::sqrt(mse / double(n_checked));
    if (!std::isfinite(rmse) || rmse > rmse_limit) {
        std::fprintf(stderr,
                "segmented raw KVarN roundtrip mismatch: rmse=%g max_diff=%g limit=%g\n",
                rmse, max_diff, double(rmse_limit));
        require(false, message);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static bool backend_supports_kvarn_flash_attention_shape(ggml_backend_t backend, int head_dim) {
    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "support gate: failed to initialize ggml context");

    constexpr int n_q          = 4;
    constexpr int n_kv         = 128;
    constexpr int n_stream     = 1;
    constexpr int stage_groups = 3;
    const int slices = head_dim / 128;
    const int record_heads = slices;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, 4) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_q, 1, n_stream);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv);
    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, record_heads, n_kv);
    ggml_tensor * k_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, record_heads, 128 * stage_groups);
    ggml_tensor * v_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, record_heads, 128 * stage_groups);
    ggml_tensor * k_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, record_heads, 1);
    ggml_tensor * v_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, record_heads, 1);
    ggml_tensor * stored_k = ggml_kvarn_store(ctx, current, indices, k_stage, k_records, 4, 16, false, stage_groups);
    ggml_tensor * stored_v = ggml_kvarn_store(ctx, current, indices, v_stage, v_records, 4, 16, true,  stage_groups);
    stored_k->op_params[5] = slices;
    stored_v->op_params[5] = slices;
    ggml_tensor * k = ggml_kvarn_view(ctx, k_records, stored_k, indices, n_kv, 0, n_stream, 4, false, stage_groups);
    ggml_tensor * v = ggml_kvarn_view(ctx, v_records, stored_v, indices, n_kv, 0, n_stream, 4, true,  stage_groups);
    if (slices > 1) {
        k = ggml_reshape_4d(ctx, k, head_dim, 1, n_kv, n_stream);
        v = ggml_reshape_4d(ctx, v, head_dim, 1, n_kv, n_stream);
    }
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, 1, n_stream);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / std::sqrt(float(head_dim)), 0.0f, 0.0f);
    out->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN] =
        GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED;

    const bool supported = ggml_backend_supports_op(backend, out);
    ggml_free(ctx);
    return supported;
}

static void require_attention_meta_close(
        const std::vector<float> & actual,
        const std::vector<float> & expected,
        const char *               message) {
    require(actual.size() == expected.size() && actual.size() % 2 == 0,
            "attention metadata parity size mismatch");
    for (size_t row = 0; row < actual.size()/2; ++row) {
        const float actual_max = actual[2*row + 0];
        const float actual_sum = actual[2*row + 1];
        const float expected_max = expected[2*row + 0];
        const float expected_sum = expected[2*row + 1];
        require(std::isfinite(actual_max) && std::isfinite(expected_max),
                "attention metadata maximum is non-finite");
        require(std::isfinite(actual_sum) && std::isfinite(expected_sum) &&
                actual_sum > 0.0f && expected_sum > 0.0f,
                "attention metadata denominator is not positive and finite");
        const float max_error = std::fabs(actual_max - expected_max);
        const float sum_relative_error = std::fabs(actual_sum - expected_sum) / expected_sum;
        if (max_error > 2e-3f || sum_relative_error > 5e-4f) {
            std::fprintf(stderr,
                    "KVarN metadata parity mismatch: row=%zu max_error=%g rowsum_relative_error=%g\n",
                    row, double(max_error), double(sum_relative_error));
            require(false, message);
        }
    }
}

struct test_kvarn_route_stats {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t route_families;
    uint32_t reserved;
    uint64_t decode_split;
    uint64_t decode_vector;
    uint64_t generic_mma;
    uint64_t prompt_prefill;
    uint64_t portable_native;
    uint64_t amd_generic_mma;
    uint64_t amd_decode_split;
    uint64_t amd_decode_vector;
    uint64_t materialize_fallback;
    uint64_t split_reduce;
    uint64_t direct_entry;
    uint64_t compact_tail_entry;
    uint64_t generic_shape_rejected;
};

static test_kvarn_route_stats make_test_kvarn_route_stats(uint32_t abi_version = 2) {
    test_kvarn_route_stats stats = {};
    stats.struct_size = sizeof(stats);
    stats.abi_version = abi_version;
    return stats;
}

using test_kvarn_route_stats_reset_fn = void (*)();
using test_kvarn_route_stats_get_fn = void (*)(test_kvarn_route_stats *);
using test_kvarn_capabilities_fn = bool (*)(
        ggml_backend_dev_t, ggml_backend_kvarn_capabilities *);
using test_kvarn_rotated_max_query_tokens_fn = uint32_t (*)(ggml_backend_dev_t);
using test_kv_tail_segmented_supported_fn = bool (*)(
        ggml_type, ggml_type, ggml_type, ggml_type, int64_t, int64_t);
using test_kvarn_tail_supported_fn = bool (*)(
        ggml_backend_dev_t, ggml_type, ggml_type, ggml_type, ggml_type, int64_t, int64_t);
using test_kvarn_mixed_tail_preferred_fn = bool (*)(ggml_backend_dev_t);

struct test_kvarn_store_route_stats {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t headwide_workspace;
    uint64_t headwide_monolithic;
    uint64_t single_slice_workspace;
    uint64_t direct_store;
    uint64_t high_shared_fallback;
    uint64_t low_shared_store;
};

static test_kvarn_store_route_stats make_test_kvarn_store_route_stats() {
    test_kvarn_store_route_stats stats = {};
    stats.struct_size = sizeof(stats);
    stats.abi_version = 1;
    return stats;
}

using test_kvarn_store_route_stats_reset_fn = void (*)();
using test_kvarn_store_route_stats_get_fn = void (*)(test_kvarn_store_route_stats *);

static std::pair<test_kvarn_route_stats_reset_fn, test_kvarn_route_stats_get_fn>
get_kvarn_route_stats_fns(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto reset = reg ? reinterpret_cast<test_kvarn_route_stats_reset_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_kvarn_route_stats_reset")) : nullptr;
    auto get = reg ? reinterpret_cast<test_kvarn_route_stats_get_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_kvarn_route_stats_get")) : nullptr;
    return { reset, get };
}

static std::pair<test_kvarn_store_route_stats_reset_fn, test_kvarn_store_route_stats_get_fn>
get_kvarn_store_route_stats_fns(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto reset = reg ? reinterpret_cast<test_kvarn_store_route_stats_reset_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_kvarn_store_route_stats_reset")) : nullptr;
    auto get = reg ? reinterpret_cast<test_kvarn_store_route_stats_get_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_kvarn_store_route_stats_get")) : nullptr;
    return { reset, get };
}

static void test_native_flash_attention_support_gates() {
    ggml_backend_t cpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    require( backend_supports_kvarn_flash_attention_shape(cpu_backend, 128),
            "CPU backend rejected native 128-dim KVarN FlashAttention");
    require( backend_supports_kvarn_flash_attention_shape(cpu_backend, 256),
            "CPU backend rejected native 256-dim KVarN FlashAttention");
    require( backend_supports_kvarn_flash_attention_shape(cpu_backend, 512),
            "CPU backend rejected native 512-dim KVarN FlashAttention");
    require(!backend_supports_kvarn_flash_attention_shape(cpu_backend, 384),
            "CPU backend accepted unsupported 384-dim KVarN FlashAttention");
    ggml_backend_free(cpu_backend);

    ggml_backend_t gpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_GPU, false);
    if (gpu_backend == nullptr) {
        return;
    }

    if (backend_supports_kvarn_flash_attention_shape(gpu_backend, 128)) {
        require( backend_supports_kvarn_flash_attention_shape(gpu_backend, 256),
                "native KVarN FlashAttention rejected supported 256-dim heads");
        require( backend_supports_kvarn_flash_attention_shape(gpu_backend, 512),
                "native KVarN FlashAttention rejected supported 512-dim heads");
        require(!backend_supports_kvarn_flash_attention_shape(gpu_backend, 384),
                "GPU backend accepted unsupported 384-dim KVarN view FlashAttention as ordinary F16");
    }

    ggml_backend_free(gpu_backend);
}

static void test_native_flash_attention_portable_backend(
        ggml_backend_t backend,
        ggml_backend_t reference_backend,
        const char * backend_label) {
    const bool trace = std::getenv("GGML_KVARN_TEST_TRACE_NATIVE") != nullptr;
    for (int head_dim : { 128, 256, 512 }) {
        for (int n_q : { 1, 4, 32 }) {
            if (trace) {
                std::fprintf(stderr, "native trace: %s D%d nq=%d\n", backend_label, head_dim, n_q);
                std::fflush(stderr);
            }
            const std::vector<float> expected = test_native_flash_attention_output(
                    reference_backend, false, false, head_dim, 4, 3, n_q,
                    6, 1, 512, 5);
            const std::vector<float> actual = test_native_flash_attention_output(
                    backend, true, true, head_dim, 4, 3, n_q,
                    6, 1, 512, 5);
            require_close_f32_rmse(actual, expected, 1e-2f,
                    "portable native KVarN FlashAttention differs from materialized reference");
        }
    }

    for (int bits_k : { 2, 4, 6, 8 }) {
        for (int bits_v : { 3, 5, 8 }) {
            if (trace) {
                std::fprintf(stderr, "native trace: %s D256 k%d/v%d\n", backend_label, bits_k, bits_v);
                std::fflush(stderr);
            }
            const std::vector<float> expected = test_native_flash_attention_output(
                    reference_backend, false, false, 256, bits_k, bits_v, 1,
                    6, 1, 1024, 5);
            const std::vector<float> actual = test_native_flash_attention_output(
                    backend, true, true, 256, bits_k, bits_v, 1,
                    6, 1, 1024, 5);
            require_close_f32_rmse(actual, expected, 1e-2f,
                    "portable mixed-bit KVarN FlashAttention differs from materialized reference");
        }
    }

    {
        if (trace) {
            std::fprintf(stderr, "native trace: %s SWA\n", backend_label);
            std::fflush(stderr);
        }
        const std::vector<float> expected = test_native_flash_attention_output(
                reference_backend, false, false, 256, 4, 4, 4,
                2, 1, 768, 4, true);
        const std::vector<float> actual = test_native_flash_attention_output(
                backend, true, true, 256, 4, 4, 4,
                2, 1, 768, 4, true);
        require_close_f32_rmse(actual, expected, 1e-2f,
                "portable SWA KVarN FlashAttention differs from materialized reference");
    }

    {
        if (trace) {
            std::fprintf(stderr, "native trace: %s sinks\n", backend_label);
            std::fflush(stderr);
        }
        const std::vector<float> expected = test_native_flash_attention_output(
                reference_backend, false, false, 256, 4, 4, 4,
                6, 1, 512, 5, false, nullptr, true);
        const std::vector<float> actual = test_native_flash_attention_output(
                backend, true, true, 256, 4, 4, 4,
                6, 1, 512, 5, false, nullptr, true);
        require_close_f32_rmse(actual, expected, 1e-2f,
                "portable KVarN FlashAttention with sinks differs from materialized reference");
    }

    for (ggml_type exact_type : { GGML_TYPE_F16, GGML_TYPE_BF16 }) {
        if (trace) {
            std::fprintf(stderr, "native trace: %s exact tail %s\n",
                    backend_label, ggml_type_name(exact_type));
            std::fflush(stderr);
        }
        const std::vector<float> expected = test_native_flash_attention_output(
                reference_backend, false, false, 256, 4, 3, 4,
                6, 1, 512, 5, false, nullptr, false, 128, false, exact_type);
        const std::vector<float> actual = test_native_flash_attention_output(
                backend, true, true, 256, 4, 3, 4,
                6, 1, 512, 5, false, nullptr, false, 128, false, exact_type);
        require_close_f32_rmse(actual, expected, 1e-2f,
                "portable KVarN FlashAttention with an exact tail differs from materialized reference");
    }

    if (std::strcmp(backend_label, "GPU") == 0 &&
            std::getenv("GGML_KVARN_TEST_FORCE_PORTABLE_CAPABILITY") != nullptr) {
        for (int n_q : { 64, 256, 1024 }) {
            if (trace) {
                std::fprintf(stderr, "native trace: %s portable prefill nq=%d\n",
                        backend_label, n_q);
                std::fflush(stderr);
            }
            const std::vector<float> expected = test_native_flash_attention_output(
                    reference_backend, false, false, 256, 4, 4, n_q,
                    2, 1, 256, 3, false, nullptr, false, 128);
            const std::vector<float> actual = test_native_flash_attention_output(
                    backend, true, true, 256, 4, 4, n_q,
                    2, 1, 256, 3, false, nullptr, false, 128);
            require_close_f32_rmse(actual, expected, 1e-2f,
                    "portable prompt-sized KVarN body-plus-tail attention differs from materialized reference");
        }

        for (ggml_type exact_type : { GGML_TYPE_F16, GGML_TYPE_BF16 }) {
            if (trace) {
                std::fprintf(stderr, "native trace: %s portable compact body-plus-current tail %s\n",
                        backend_label, ggml_type_name(exact_type));
                std::fflush(stderr);
            }
            const std::vector<float> expected = test_native_flash_attention_output(
                    reference_backend, false, false, 256, 5, 4, 4,
                    16, 2, 512, 3, false, nullptr, false, 128,
                    false, exact_type, 4, false, true);
            const std::vector<float> actual = test_native_flash_attention_output(
                    backend, true, true, 256, 5, 4, 4,
                    16, 2, 512, 3, false, nullptr, false, 128,
                    false, exact_type, 4, false, true);
            require_close_f32_rmse(actual, expected, 1e-2f,
                    "portable compact KVarN body-plus-current tail differs from materialized reference");
        }
    }

    for (int head_dim : { 128, 256, 512 }) {
        for (int n_q : { 1, 2, 8, 16 }) {
            for (ggml_type exact_type : { GGML_TYPE_F16, GGML_TYPE_BF16 }) {
                const int current_tokens = std::max(n_q, 4);
                if (trace) {
                    std::fprintf(stderr,
                            "native trace: %s compact bodyless tail D%d nq=%d %s\n",
                            backend_label, head_dim, n_q, ggml_type_name(exact_type));
                    std::fflush(stderr);
                }
                const std::vector<float> expected = test_native_flash_attention_output(
                        reference_backend, false, false, head_dim, 4, 3, n_q,
                        6, 1, 512, 5, false, nullptr, false, 128,
                        false, exact_type, current_tokens, true, true);
                const std::vector<float> actual = test_native_flash_attention_output(
                        backend, true, true, head_dim, 4, 3, n_q,
                        6, 1, 512, 5, false, nullptr, false, 128,
                        false, exact_type, current_tokens, true, true);
                require_close_f32_rmse(actual, expected, 1e-2f,
                        "portable compact bodyless segmented KVarN tail differs from materialized reference");
            }
        }
    }

    std::printf("test-kvarn: %s direct KVarN FlashAttention parity OK\n", backend_label);
}

static void test_native_flash_attention_cpu() {
    ggml_backend_t cpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    test_native_flash_attention_portable_backend(cpu_backend, cpu_backend, "CPU");
    ggml_backend_free(cpu_backend);
}

static void test_native_flash_attention_gpu() {
    ggml_backend_t gpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_GPU, false);
    if (gpu_backend == nullptr) {
        return;
    }
    if (!backend_supports_kvarn_flash_attention_shape(gpu_backend, 128)) {
        // The standard-tail regression build intentionally compiles without
        // descriptor-native KVarN FA templates. Its capability gate must
        // reject the route before any backend allocation is attempted.
        ggml_backend_free(gpu_backend);
        return;
    }
    ggml_backend_t cpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_CPU, true);

    const auto [route_stats_reset, route_stats_get] = get_kvarn_route_stats_fns(gpu_backend);
    const ggml_backend_dev_t gpu_device = ggml_backend_get_device(gpu_backend);
    const char * actual_backend = gpu_device ? ggml_backend_dev_name(gpu_device) : nullptr;
    const bool expect_vulkan_route_stats = actual_backend != nullptr &&
        std::strncmp(actual_backend, "Vulkan", 6) == 0;
    const uint32_t route_stats_abi_version = expect_vulkan_route_stats ? 1u : 2u;
    bool hip_safe_first = false;
    int hip_physical_wave_size = 0;
    require(!expect_vulkan_route_stats ||
            (route_stats_reset != nullptr && route_stats_get != nullptr),
            "Vulkan KVarN backend omitted ABI-v2 route telemetry");
    if (expect_vulkan_route_stats) {
        ggml_backend_dev_t dev = ggml_backend_get_device(gpu_backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        auto segmented_supported = reg ? reinterpret_cast<test_kv_tail_segmented_supported_fn>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_kv_tail_segmented_attention_supported")) : nullptr;
        auto kvarn_tail_supported = reg ? reinterpret_cast<test_kvarn_tail_supported_fn>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_kvarn_tail_attention_supported")) : nullptr;
        auto mixed_tail_preferred = reg ? reinterpret_cast<test_kvarn_mixed_tail_preferred_fn>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_kvarn_mixed_tail_native_preferred")) : nullptr;
        require(segmented_supported != nullptr && kvarn_tail_supported != nullptr &&
                mixed_tail_preferred != nullptr,
                "Vulkan omitted a compact segmented KVarN capability predicate");
        require(!mixed_tail_preferred(dev),
                "Vulkan must not prefer its slower native compressed-body/tail mixture");
        for (int64_t d : { 128, 256, 512 }) {
            for (ggml_type exact_type : { GGML_TYPE_F16, GGML_TYPE_BF16 }) {
                require(segmented_supported(
                            GGML_TYPE_F16, GGML_TYPE_F16,
                            exact_type, exact_type, d, d) &&
                        kvarn_tail_supported(
                            dev, GGML_TYPE_F16, GGML_TYPE_F16,
                            exact_type, exact_type, d, d),
                        "Vulkan compact segmented and KVarN capability predicates disagree");
            }
        }
        require(segmented_supported(
                    GGML_TYPE_Q4_0, GGML_TYPE_Q4_0,
                    GGML_TYPE_F16, GGML_TYPE_F16, 256, 256) &&
                !segmented_supported(
                    GGML_TYPE_F16, GGML_TYPE_F16,
                    GGML_TYPE_F16, GGML_TYPE_F16, 384, 384),
                "Vulkan compact segmented capability disagrees with the bounded standard-tail matrix");
    }
    if (route_stats_reset != nullptr && route_stats_get != nullptr) {
        route_stats_get(nullptr);
        test_kvarn_route_stats undersized = make_test_kvarn_route_stats(route_stats_abi_version);
        undersized.struct_size -= expect_vulkan_route_stats ?
            2 * sizeof(undersized.compact_tail_entry) : sizeof(undersized.generic_shape_rejected);
        undersized.route_families = 0xa5a5a5a5u;
        route_stats_get(&undersized);
        require(undersized.route_families == 0xa5a5a5a5u,
                "KVarN route telemetry wrote an undersized caller structure");
        test_kvarn_route_stats wrong_version = make_test_kvarn_route_stats(route_stats_abi_version);
        wrong_version.abi_version += 1;
        wrong_version.route_families = 0x5a5a5a5au;
        route_stats_get(&wrong_version);
        require(wrong_version.route_families == 0x5a5a5a5au,
                "KVarN route telemetry wrote a caller structure with an unsupported ABI version");
        route_stats_reset();
        ggml_backend_dev_t dev = ggml_backend_get_device(gpu_backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        auto rotated_max_query_tokens = reg ? reinterpret_cast<test_kvarn_rotated_max_query_tokens_fn>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_kvarn_native_rotated_max_query_tokens")) : nullptr;
        auto get_capabilities = reg ? reinterpret_cast<test_kvarn_capabilities_fn>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_kvarn_capabilities")) : nullptr;
        require(rotated_max_query_tokens != nullptr,
                "native KVarN backend omitted its rotated query-batch capability");
        require(rotated_max_query_tokens(dev) >= 16,
                "native KVarN backend does not cover a complete DFlash verification block");
        if (expect_vulkan_route_stats) {
            require(rotated_max_query_tokens(dev) == UINT32_MAX,
                    "Vulkan must advertise the query widths accepted by its final portable KVarN operation");
        }
        require(get_capabilities != nullptr || expect_vulkan_route_stats,
                "native CUDA/HIP KVarN backend omitted its versioned capability record");
        if (get_capabilities != nullptr) {
            ggml_backend_kvarn_capabilities undersized_capabilities = {};
            undersized_capabilities.struct_size =
                sizeof(undersized_capabilities) - sizeof(undersized_capabilities.minimum_dynamic_shared_bytes);
            undersized_capabilities.abi_version = GGML_BACKEND_KVARN_CAPABILITIES_ABI_VERSION;
            undersized_capabilities.route_families = 0xa5a5a5a5u;
            require(!get_capabilities(dev, &undersized_capabilities) &&
                    undersized_capabilities.route_families == 0xa5a5a5a5u,
                    "KVarN capability query wrote an undersized caller structure");
            ggml_backend_kvarn_capabilities wrong_capability_version = {};
            wrong_capability_version.struct_size = sizeof(wrong_capability_version);
            wrong_capability_version.abi_version =
                GGML_BACKEND_KVARN_CAPABILITIES_ABI_VERSION + 1;
            wrong_capability_version.route_families = 0x5a5a5a5au;
            require(!get_capabilities(dev, &wrong_capability_version) &&
                    wrong_capability_version.route_families == 0x5a5a5a5au,
                    "KVarN capability query accepted an unknown ABI version");
            ggml_backend_kvarn_capabilities capabilities = {};
            capabilities.struct_size = sizeof(capabilities);
            capabilities.abi_version = GGML_BACKEND_KVARN_CAPABILITIES_ABI_VERSION;
            require(get_capabilities(dev, &capabilities),
                    "native KVarN backend rejected the current capability ABI");
            require(capabilities.store_materialize &&
                    capabilities.portable_direct_body &&
                    capabilities.portable_integrated_tail_f16 &&
                    capabilities.portable_integrated_tail_bf16 &&
                    capabilities.minimum_dynamic_shared_bytes > 0,
                    "native KVarN backend capability record omitted its portable body-plus-tail contract");
            hip_safe_first = capabilities.specialized_generic_mma &&
                !capabilities.original_v_domain;
            hip_physical_wave_size = capabilities.physical_warp_size;
            if (hip_safe_first) {
                require(!capabilities.specialized_decode_split &&
                        !capabilities.specialized_decode_vector,
                        "HIP advertised a CUDA-only specialized KVarN decode route");
            }
            if (std::getenv("GGML_KVARN_TEST_FORCE_PORTABLE_CAPABILITY") != nullptr) {
                require(!capabilities.specialized_generic_mma &&
                        !capabilities.specialized_decode_split &&
                        !capabilities.specialized_decode_vector &&
                        !capabilities.original_v_domain &&
                        capabilities.rotated_query_max_portable == UINT32_MAX &&
                        capabilities.rotated_query_max_specialized == 0 &&
                        rotated_max_query_tokens(dev) == UINT32_MAX,
                        "simulated pre-Turing CUDA capability did not remain portable-only");
            }
        }
    }
    test_native_flash_attention_portable_backend(gpu_backend, cpu_backend, "GPU");
    if (std::getenv("GGML_KVARN_TEST_AMD_ROUTE_BOUNDARIES_ONLY") != nullptr) {
        require(route_stats_reset != nullptr && route_stats_get != nullptr,
                "AMD route-boundary validation requires KVarN route telemetry");
        require(hip_safe_first &&
                (hip_physical_wave_size == 32 || hip_physical_wave_size == 64),
                "AMD route-boundary validation did not find a safe-first HIP wave32/wave64 device");

        const char * attestation = std::getenv("GGML_KVARN_AMD_RUNTIME_ATTESTATION");
        const bool attested_wave = attestation != nullptr &&
            ((hip_physical_wave_size == 32 && std::strcmp(attestation, "rdna-wave32") == 0) ||
             (hip_physical_wave_size == 64 && std::strcmp(attestation, "cdna-wave64") == 0));
        require(attested_wave,
                "AMD runtime attestation does not match the physical wave reported by the backend");

        const auto require_amd_case = [&](int head_dim, int n_q, int gqa,
                                          int tail_tokens, ggml_type exact_type,
                                          const char * message) {
            const std::vector<float> expected = test_native_flash_attention_output(
                    cpu_backend, false, false, head_dim, 4, 3, n_q,
                    gqa, 1, 513, 5, false, nullptr, false,
                    tail_tokens, false, exact_type);
            route_stats_reset();
            const std::vector<float> actual = test_native_flash_attention_output(
                    gpu_backend, true, true, head_dim, 4, 3, n_q,
                    gqa, 1, 513, 5, false, nullptr, true,
                    tail_tokens, false, exact_type);
            test_kvarn_route_stats stats = make_test_kvarn_route_stats(route_stats_abi_version);
            route_stats_get(&stats);

            require_close_f32_rmse(actual, expected, 1e-2f, message);
            require(stats.materialize_fallback == 0,
                    "AMD route-boundary case materialized the KVarN body");
            require(stats.decode_split == 0 && stats.amd_decode_split == 0 &&
                    stats.decode_vector == 0 && stats.amd_decode_vector == 0,
                    "AMD route-boundary case entered a CUDA-only specialized decode route");
            const bool known_invalid_generic = hip_physical_wave_size == 32 ?
                head_dim > 128 : head_dim > 256;
            if (known_invalid_generic) {
                require(stats.generic_shape_rejected > 0 && stats.portable_native > 0 &&
                        stats.generic_mma == 0 && stats.prompt_prefill == 0,
                        "known-invalid AMD MMA shape did not fall through to portable KVarN attention");
            } else {
                require(stats.generic_mma + stats.prompt_prefill + stats.portable_native > 0,
                        "AMD route-boundary case did not execute an optimized or portable native route");
                require(stats.generic_shape_rejected == 0 || stats.portable_native > 0,
                        "AMD generic rejection did not continue to portable KVarN attention");
            }
        };

        for (int head_dim : { 128, 256, 512 }) {
            for (int n_q : { 1, 2, 4, 8, 16, 17, 256, 511, 512 }) {
                require_amd_case(head_dim, n_q, 6, 0, GGML_TYPE_F16,
                        "AMD KVarN route-boundary output differs from the materialized oracle");
            }
        }
        for (int gqa : { 1, 2, 4, 6, 8, 16 }) {
            require_amd_case(128, 17, gqa, 0, GGML_TYPE_F16,
                    "AMD D128 GQA route-boundary output differs from the materialized oracle");
        }
        for (int head_dim : { 256, 512 }) {
            for (int n_q : { 17, 256 }) {
                for (ggml_type exact_type : { GGML_TYPE_F16, GGML_TYPE_BF16 }) {
                    require_amd_case(head_dim, n_q, 6, 128, exact_type,
                            "AMD KVarN body-plus-tail output differs from the materialized oracle");
                }
            }
        }
        std::printf("test-kvarn: AMD runtime validation attestation=%s physical_wave=%d OK\n",
                attestation, hip_physical_wave_size);
        ggml_backend_free(cpu_backend);
        ggml_backend_free(gpu_backend);
        return;
    }
    if (hip_safe_first) {
        // The general suite below contains CUDA's validated original-V window
        // and specialized-route assertions. HIP remains in the rotated domain;
        // its complete shape matrix is exercised by the attested subset above.
        ggml_backend_free(cpu_backend);
        ggml_backend_free(gpu_backend);
        return;
    }
    if (route_stats_reset == nullptr || route_stats_get == nullptr ||
            std::getenv("GGML_KVARN_TEST_FORCE_PORTABLE_FATTN") != nullptr) {
        // Vulkan and other portable GPU backends intentionally do not expose
        // CUDA's specialized-route counters. A forced portable CUDA run also
        // deliberately bypasses those routes.
        ggml_backend_free(cpu_backend);
        ggml_backend_free(gpu_backend);
        return;
    }
    test_kvarn_route_stats route_capabilities = make_test_kvarn_route_stats(route_stats_abi_version);
    route_stats_get(&route_capabilities);
    require(route_capabilities.direct_entry > 0,
            "KVarN route telemetry did not observe the direct attention entry");
    constexpr uint32_t specialized_route_families = (1u << 1) | (1u << 2) | (1u << 3);
    if ((route_capabilities.route_families & specialized_route_families) == 0) {
        // Portable-only Vulkan/HIP/MUSA devices expose counters for diagnostics,
        // but must not be asserted against CUDA/AMD matrix-route expectations.
        require((route_capabilities.route_families & (1u << 0)) != 0,
                "portable-only KVarN backend did not advertise its portable route family");
        require(route_capabilities.portable_native > 0,
                "portable-only KVarN backend did not report its native route");
        require(route_capabilities.direct_entry > 0,
                "portable-only KVarN backend did not report its direct body-plus-tail entry");
        require(route_capabilities.compact_tail_entry > 0,
                "portable-only KVarN backend did not report its compact body-plus-current-tail entry");
        require(route_capabilities.materialize_fallback == 0,
                "portable-only compact KVarN tail unexpectedly materialized");
        ggml_backend_free(cpu_backend);
        ggml_backend_free(gpu_backend);
        return;
    }
    require(route_capabilities.compact_tail_entry > 0,
            "KVarN route telemetry did not observe the exact/compact-tail body entry");

    const auto require_metadata_case = [&](int head_dim, int n_q, int n_q_heads,
                                           int n_kv_heads, int bits, bool swa,
                                           uint64_t min_split, uint64_t min_vector,
                                           bool expect_tiled_mma,
                                           const char * message) {
        std::vector<float> generic_meta;
        std::vector<float> actual_meta;
        const std::vector<float> expected = test_native_flash_attention_output(
                cpu_backend, false, false, head_dim, bits, bits, n_q, n_q_heads, n_kv_heads,
                1024, 5, swa);
        route_stats_reset();
        const std::vector<float> generic = test_native_flash_attention_output(
                gpu_backend, true, true, head_dim, bits, bits, n_q, n_q_heads, n_kv_heads,
                1024, 5, swa, &generic_meta, true);
        test_kvarn_route_stats generic_stats = make_test_kvarn_route_stats(route_stats_abi_version);
        route_stats_get(&generic_stats);
        route_stats_reset();
        const std::vector<float> actual = test_native_flash_attention_output(
                gpu_backend, true, true, head_dim, bits, bits, n_q, n_q_heads, n_kv_heads,
                1024, 5, swa, &actual_meta);
        test_kvarn_route_stats stats = make_test_kvarn_route_stats(route_stats_abi_version);
        route_stats_get(&stats);

        if (!swa) {
            require_close_f32_rmse(actual, expected, 1e-2f, message);
        }
        require_close_f32_rmse(actual, generic, 1e-4f,
                "specialized KVarN body output differs from generic KVarN reference");
        require_attention_meta_close(actual_meta, generic_meta,
                "specialized KVarN body metadata differs from generic KVarN reference");
        require(generic_stats.generic_mma > 0 && generic_stats.decode_split == 0 && generic_stats.decode_vector == 0,
                "neutral-sink metadata reference did not exercise generic KVarN MMA");
        if (expect_tiled_mma) {
            require(stats.generic_mma > 0 && stats.decode_split == 0 && stats.decode_vector == 0,
                    "multi-query KVarN decode did not exercise tiled native MMA");
        } else {
            require(stats.decode_split >= min_split && stats.decode_vector >= min_vector,
                    "metadata-capable KVarN decode did not exercise the required specialized route");
            require(stats.generic_mma == 0,
                    "eligible single-query KVarN decode fell back to generic MMA");
        }
        require(actual_meta.size() == size_t(2 * n_q_heads * n_q),
                "specialized KVarN body metadata has the wrong layout");
        for (size_t i = 0; i < actual_meta.size(); i += 2) {
            require(std::isfinite(actual_meta[i]), "specialized KVarN body maximum is non-finite");
            require(std::isfinite(actual_meta[i + 1]) && actual_meta[i + 1] > 0.0f,
                    "specialized KVarN body denominator is not positive and finite");
        }
    };

    require_metadata_case(256, 1, 6, 1, 4, false, 1, 0, false,
            "Qwen-like D256 metadata-capable split output differs from reference");
    require_metadata_case(512, 1, 16, 1, 4, false, 1, 0, false,
            "Gemma-like D512 metadata-capable split output differs from reference");
    require_metadata_case(256, 1, 2, 1, 4, true, 0, 1, false,
            "Gemma-like D256 SWA metadata-capable vector output differs from reference");
    for (int n_q = 2; n_q <= 16; ++n_q) {
        require_metadata_case(256, n_q, 6, 1, 4, false, 0, 0, true,
                "multi-token metadata-capable tiled MMA output differs from reference");
    }
    require_metadata_case(256, 9, 6, 1, 6, false, 0, 0, true,
            "KVarN6 DFlash-sized tiled MMA output differs from reference");
    require_metadata_case(256, 16, 6, 1, 6, false, 0, 0, true,
            "KVarN6 full-block tiled MMA output differs from reference");

    const auto require_exact_tail_case = [&](int head_dim, int n_q, int n_q_heads,
                                              int n_kv_heads, bool swa, int tail_tokens,
                                              uint64_t min_split, uint64_t min_vector,
                                              bool expect_tiled_mma,
                                              const char * message) {
        route_stats_reset();
        const std::vector<float> generic = test_native_flash_attention_output(
                gpu_backend, true, true, head_dim, 4, 4, n_q, n_q_heads, n_kv_heads,
                1024, 5, swa, nullptr, true, tail_tokens);
        test_kvarn_route_stats generic_stats = make_test_kvarn_route_stats(route_stats_abi_version);
        route_stats_get(&generic_stats);
        route_stats_reset();
        const std::vector<float> actual = test_native_flash_attention_output(
                gpu_backend, true, true, head_dim, 4, 4, n_q, n_q_heads, n_kv_heads,
                1024, 5, swa, nullptr, false, tail_tokens);
        test_kvarn_route_stats stats = make_test_kvarn_route_stats(route_stats_abi_version);
        route_stats_get(&stats);

        require_close_f32_rmse(actual, generic, 1e-4f, message);
        require(generic_stats.generic_mma > 0 &&
                generic_stats.decode_split == 0 && generic_stats.decode_vector == 0,
                "exact-tail reference did not exercise generic KVarN MMA");
        if (expect_tiled_mma) {
            require(stats.generic_mma > 0 && stats.decode_split == 0 && stats.decode_vector == 0,
                    "multi-query exact-tail KVarN body did not exercise tiled native MMA");
        } else {
            require(stats.decode_split >= min_split && stats.decode_vector >= min_vector,
                    "exact-tail KVarN body did not exercise the required specialized route");
            require(stats.generic_mma == 0,
                    "eligible single-query exact-tail KVarN body fell back to generic MMA");
        }
    };

    require_exact_tail_case(256, 1, 6, 1, false, 128, 1, 0, false,
            "intrinsic-128 exact-tail merge differs from generic KVarN reference");
    require_exact_tail_case(256, 1, 6, 1, false, 1024, 1, 0, false,
            "requested-1024 exact-tail merge differs from generic KVarN reference");
    // A request of 129 is rounded by policy to this 256-token effective tail.
    require_exact_tail_case(256, 1, 6, 1, false, 256, 1, 0, false,
            "rounded-256 exact-tail merge differs from generic KVarN reference");
    require_exact_tail_case(512, 1, 16, 1, false, 128, 1, 0, false,
            "D512 exact-tail merge differs from generic KVarN reference");
    require_exact_tail_case(256, 1, 2, 1, true, 128, 0, 1, false,
            "D256 SWA vector exact-tail merge differs from generic KVarN reference");
    for (int n_q = 2; n_q <= 16; ++n_q) {
        require_exact_tail_case(256, n_q, 6, 1, false, 128, 0, 0, true,
                "speculative exact-tail tiled MMA output differs from generic KVarN reference");
    }

    for (int head_dim : { 128, 256, 512 }) {
        constexpr int n_q_native = 4;
        const std::vector<float> expected = test_native_flash_attention_output(cpu_backend, false, false, head_dim, 4, 3, n_q_native);
        const std::vector<float> actual   = test_native_flash_attention_output(gpu_backend, true,  true,  head_dim, 4, 3, n_q_native);
        require_close_f32_rmse(actual, expected, 1e-2f,
                "native KVarN FlashAttention output differs from CPU reference decode");

        constexpr int n_q_prefill = 64;
        const std::vector<float> expected_prefill = test_native_flash_attention_output(cpu_backend, false, false, head_dim, 4, 3, n_q_prefill);
        const std::vector<float> actual_prefill   = test_native_flash_attention_output(gpu_backend, true,  false, head_dim, 4, 3, n_q_prefill);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "large-prefill original-domain KVarN FlashAttention differs from CPU reference");
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_q_prefill = 128;
        constexpr int n_q_heads = 6;
        constexpr int n_kv_heads = 1;
        constexpr int n_segments = 4;
        constexpr int segment_tokens = 128;
        constexpr int stage_groups = 2;
        const std::vector<float> expected_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, false, head_dim, 4, 4, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, false, 0, false, 0, 0.0f,
                false, 0, true, true);
        const std::vector<float> actual_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, true, head_dim, 4, 4, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, false, 0, false, 0, 0.0f,
                false, 0, true, true, true);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "eager segmented KVarN FlashAttention differs from materialized reference");
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_q_heads = 6;
        constexpr int n_kv_heads = 1;
        constexpr int n_kv = 4096;
        constexpr int stage_groups = 25;
        for (int n_q_prefill : { 512, 1024 }) {
            const std::vector<float> expected_prefill = test_native_flash_attention_output(
                    cpu_backend, false, false, head_dim, 4, 4, n_q_prefill, n_q_heads, n_kv_heads, n_kv, stage_groups);
            const std::vector<float> actual_prefill = test_native_flash_attention_output(
                    gpu_backend, true, false, head_dim, 4, 4, n_q_prefill, n_q_heads, n_kv_heads, n_kv, stage_groups);
            require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                    "large-prefill multi-head original-domain KVarN FlashAttention differs from CPU reference");
        }
    }

    {
        constexpr int head_dim = 512;
        constexpr int n_q_prefill = 256;
        constexpr int n_q_heads = 32;
        constexpr int n_kv_heads = 4;
        constexpr int n_segments = 16;
        constexpr int segment_tokens = 256;
        constexpr int stage_groups = 19;
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, false, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "segmented D512 KVarN8 K roundtrip differs from original-domain input");
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, true, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "segmented D512 KVarN8 V roundtrip differs from original-domain input");
        const std::vector<float> expected_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, false, head_dim, 4, 4, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups);
        const std::vector<float> actual_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, true, head_dim, 4, 4, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "segmented Gemma-style original-domain KVarN FlashAttention differs from materialized reference");
    }

    {
        constexpr int head_dim = 512;
        constexpr int n_q_prefill = 256;
        constexpr int n_q_heads = 8;
        constexpr int n_kv_heads = 1;
        constexpr int n_segments = 64;
        constexpr int segment_tokens = 256;
        constexpr int stage_groups = 19;
        constexpr float gemma_attn_scale = 1.0f;
        const std::vector<float> expected_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, false, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, false, 0, false, 0, gemma_attn_scale);
        const std::vector<float> actual_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, true, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, false, 0, false, 0, gemma_attn_scale);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "deep D512 original-domain KVarN8 FlashAttention differs from materialized reference");
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_q_prefill = 256;
        constexpr int n_q_heads = 32;
        constexpr int n_kv_heads = 16;
        constexpr int n_segments = 8;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 1280;
        constexpr int stage_groups = 3;
        const std::vector<float> expected_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, false, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window);
        const std::vector<float> actual_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, true, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "segmented Gemma-style SWA original-domain KVarN FlashAttention differs from materialized reference");
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_q_prefill = 256;
        constexpr int n_q_heads = 32;
        constexpr int n_kv_heads = 16;
        constexpr int n_segments = 64;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 1280;
        constexpr int stage_groups = 2;
        constexpr int tail_groups = 2;
        constexpr float gemma_attn_scale = 1.0f;
        const std::vector<float> expected_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, false, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window, false, 0,
                gemma_attn_scale, false, tail_groups, true);
        const std::vector<float> actual_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, true, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window, false, 0,
                gemma_attn_scale, false, tail_groups, true);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "tail-equals-stage SWA original-domain KVarN FlashAttention differs from materialized reference");
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_kv_heads = 16;
        constexpr int n_segments = 64;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 1280;
        constexpr int stage_groups = 3;
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, false, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "wrapped SWA D256 KVarN8 K roundtrip differs from original-domain input",
                true, n_swa_window, true);
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, true, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "wrapped SWA D256 KVarN8 V roundtrip differs from original-domain input",
                true, n_swa_window, true);
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_kv_heads = 16;
        constexpr int n_segments = 64;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 1280;
        constexpr int stage_groups = 2;
        constexpr int tail_groups = 2;
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, false, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "tail-equals-stage SWA D256 KVarN8 K roundtrip differs from original-domain input",
                true, n_swa_window, true, tail_groups);
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, true, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "tail-equals-stage SWA D256 KVarN8 V roundtrip differs from original-domain input",
                true, n_swa_window, true, tail_groups);
    }

    {
        // stage_groups=4 gives tail_groups=3, which is the direct SWA safety guard.
        constexpr int head_dim = 256;
        constexpr int n_kv_heads = 4;
        constexpr int n_segments = 8;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 768;
        constexpr int stage_groups = 4;
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 4, false, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-1f,
                "direct SWA D256 KVarN4 K roundtrip differs from original-domain input",
                true, n_swa_window, true);
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 4, true, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-1f,
                "direct SWA D256 KVarN4 V roundtrip differs from original-domain input",
                true, n_swa_window, true);
    }

    {
        constexpr int head_dim = 512;
        constexpr int n_kv_heads = 1;
        constexpr int n_segments = 64;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 1280;
        constexpr int stage_groups = 3;
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, false, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "wrapped SWA D512 KVarN8 K roundtrip differs from original-domain input",
                true, n_swa_window, true);
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, true, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "wrapped SWA D512 KVarN8 V roundtrip differs from original-domain input",
                true, n_swa_window, true);
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_q_prefill = 256;
        constexpr int n_q_heads = 32;
        constexpr int n_kv_heads = 16;
        constexpr int n_segments = 64;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 1280;
        constexpr int n_swa_visible = 1024;
        constexpr int stage_groups = 3;
        constexpr float gemma_attn_scale = 1.0f;
        const std::vector<float> expected_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, false, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window, true, n_swa_visible, gemma_attn_scale);
        const std::vector<float> actual_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, true, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window, true, n_swa_visible, gemma_attn_scale);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "wrapped Gemma-style SWA original-domain KVarN FlashAttention differs from materialized reference");
    }

    {
        constexpr int head_dim = 512;
        constexpr int n_q_prefill = 128;
        constexpr int n_q_heads = 4;
        constexpr int n_kv_heads = 1;
        constexpr int n_segments = 8;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 768;
        constexpr int stage_groups = 3;
        constexpr float gemma_attn_scale = 1.0f;
        const std::vector<float> expected_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, false, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window, false, 0, gemma_attn_scale, true);
        const std::vector<float> actual_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, true, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window, false, 0, gemma_attn_scale, true);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "scrambled SWA D512 original-domain KVarN FlashAttention differs from materialized reference");
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_q_decode = 1;
        constexpr int n_q_heads = 6;
        constexpr int n_kv_heads = 1;
        constexpr int stage_groups = 5;
        for (int bits_k : { 2, 3, 4, 5, 6, 8 }) {
            for (int bits_v : { 2, 3, 4, 5, 6, 8 }) {
                constexpr int n_kv_matrix = 1024;
                const std::vector<float> expected_decode = test_native_flash_attention_output(
                        cpu_backend, false, false, head_dim, bits_k, bits_v, n_q_decode, n_q_heads, n_kv_heads, n_kv_matrix, stage_groups);
                const std::vector<float> actual_decode = test_native_flash_attention_output(
                        gpu_backend, true, true, head_dim, bits_k, bits_v, n_q_decode, n_q_heads, n_kv_heads, n_kv_matrix, stage_groups);
                require_close_f32_rmse(actual_decode, expected_decode, 1e-2f,
                        "single-token GQA mixed-bit native KVarN FlashAttention output differs from CPU reference decode");
            }
        }

        constexpr int n_kv_deep = 4096;
        const std::vector<float> expected_deep = test_native_flash_attention_output(
                cpu_backend, false, false, head_dim, 4, 4, n_q_decode, n_q_heads, n_kv_heads, n_kv_deep, stage_groups);
        const std::vector<float> actual_deep = test_native_flash_attention_output(
                gpu_backend, true, true, head_dim, 4, 4, n_q_decode, n_q_heads, n_kv_heads, n_kv_deep, stage_groups);
        require_close_f32_rmse(actual_deep, expected_deep, 1e-2f,
                "single-token GQA deep K4/V4 native KVarN FlashAttention output differs from CPU reference decode");
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);
}

static void test_native_flash_attention_prefill_route_parity() {
    ggml_backend_t gpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_GPU, false);
    if (gpu_backend == nullptr) {
        return;
    }
    if (!backend_supports_kvarn_flash_attention_shape(gpu_backend, 256)) {
        ggml_backend_free(gpu_backend);
        return;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(gpu_backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto get_capabilities = reg ? reinterpret_cast<test_kvarn_capabilities_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_kvarn_capabilities")) : nullptr;
    if (get_capabilities != nullptr) {
        ggml_backend_kvarn_capabilities capabilities = {};
        capabilities.struct_size = sizeof(capabilities);
        capabilities.abi_version = GGML_BACKEND_KVARN_CAPABILITIES_ABI_VERSION;
        if (get_capabilities(dev, &capabilities) &&
                capabilities.specialized_generic_mma && !capabilities.original_v_domain) {
            ggml_backend_free(gpu_backend);
            return;
        }
    }

    const auto require_route_parity = [&](int bits, int n_kv, int tail_candidates, const char * message) {
        const std::vector<float> windowed = test_native_flash_attention_output(
                gpu_backend, true, true, 256, bits, bits, 512, 6, 1,
                n_kv, 3, false, nullptr, false, tail_candidates, true);
        std::vector<float> generic;
        {
            scoped_test_env disable_window("GGML_KVARN_WINDOW", "0");
            generic = test_native_flash_attention_output(
                    gpu_backend, true, true, 256, bits, bits, 512, 6, 1,
                    n_kv, 3, false, nullptr, false, tail_candidates, true);
        }
        require_close_f32_rmse(generic, windowed, 1e-4f, message);
    };

    require_route_parity(4, 512, 128,
            "generic and windowed KVarN prefill routes disagree with an exact tail");
    // A 512-token serving ubatch needs the union of the configured exact tail
    // and its in-flight rows: 1024 -> 1536 candidates, 2048 -> 2560.  These
    // long-tail geometries exercise the regular two-FA merge omitted by the
    // original 128-candidate regression for the determinism fix.
    for (int bits : { 5, 8 }) {
        require_route_parity(bits, 4096, 1536,
                "generic and windowed KVarN prefill routes disagree for a 1024-token serving tail");
        require_route_parity(bits, 4096, 2560,
                "generic and windowed KVarN prefill routes disagree for a 2048-token serving tail");
    }

    ggml_backend_free(gpu_backend);
}

static void test_store_paths_gpu() {
    ggml_backend_t gpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_GPU, false);
    if (gpu_backend == nullptr) {
        return;
    }
    ggml_backend_t cpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    const auto [store_stats_reset, store_stats_get] = get_kvarn_store_route_stats_fns(gpu_backend);
    if (store_stats_reset != nullptr) {
        store_stats_reset();
    }

    for (int bits : { 2, 3, 4, 5, 6, 8 }) {
        for (bool value : { false, true }) {
            const std::vector<ggml_fp16_t> cuda_output = test_store_reference_output(
                    gpu_backend, bits, value, 2, 2, 385, 64);
            const std::vector<ggml_fp16_t> cpu_output = test_store_reference_output(
                    cpu_backend, bits, value, 2, 2, 385, 64);
            require_close_f16_rmse(cuda_output, cpu_output, 1e-1f, "KVarN CUDA store output differs from CPU reference");
        }
    }

    {
        const std::vector<ggml_fp16_t> cuda_output = test_store_reference_output(
                gpu_backend, 4, false, 1, 2, 65, 63);
        const std::vector<ggml_fp16_t> cpu_output = test_store_reference_output(
                cpu_backend, 4, false, 1, 2, 65, 63);
        require_close_f16_rmse(cuda_output, cpu_output, 1e-1f,
                "KVarN CUDA high-shared fallback output differs from CPU reference");
    }

    {
        scoped_test_env force_low_shared("GGML_KVARN_TEST_FORCE_LOWSHMEM", "1");
        const std::vector<ggml_fp16_t> cuda_output = test_store_reference_output(
                gpu_backend, 4, false, 1, 2, 65, 63);
        const std::vector<ggml_fp16_t> cpu_output = test_store_reference_output(
                cpu_backend, 4, false, 1, 2, 65, 63);
        require_close_f16_rmse(cuda_output, cpu_output, 1e-1f,
                "KVarN CUDA low-shared fallback output differs from CPU reference");
    }

    for (int bits : { 2, 3, 4, 5, 6, 8 }) {
        for (bool value : { false, true }) {
            const std::vector<ggml_fp16_t> cuda_output = test_store_reference_output(
                    gpu_backend, bits, value, 1, 2, 512, 200);
            const std::vector<ggml_fp16_t> cpu_output = test_store_reference_output(
                    cpu_backend, bits, value, 1, 2, 512, 200);
            require_close_f16_rmse(cuda_output, cpu_output, 1e-1f, "KVarN CUDA split workspace store output differs from CPU reference");
        }
    }

    for (bool value : { false, true }) {
        const std::vector<ggml_fp16_t> workspace_output = test_store_segmented_output(
                gpu_backend, 4, value, 16, 2, 4096, 512, 7);
        const std::vector<ggml_fp16_t> fallback_output = test_store_segmented_output(
                gpu_backend, 4, value, 16, 2, 4096, 256, 7);
        require(workspace_output == fallback_output,
                "KVarN CUDA D256 workspace ubatch store output is not segmentation-independent");
    }

    for (bool value : { false, true }) {
        const std::vector<ggml_fp16_t> workspace_output = test_store_segmented_output(
                gpu_backend, 4, value, 16, 2, 4096, 512, 7, true);
        const std::vector<ggml_fp16_t> fallback_output = test_store_segmented_output(
                gpu_backend, 4, value, 16, 2, 4096, 256, 7, true);
        require(workspace_output == fallback_output,
                "KVarN CUDA eager workspace store output is not segmentation-independent");
    }

    for (bool value : { false, true }) {
        const std::vector<ggml_fp16_t> workspace_output = test_store_segmented_output(
                gpu_backend, 4, value, 16, 2, 4096, 512, 7, true, true);
        const std::vector<ggml_fp16_t> fallback_output = test_store_segmented_output(
                gpu_backend, 4, value, 16, 2, 4096, 256, 7, true, true);
        require(workspace_output == fallback_output,
                "KVarN CUDA eager SWA workspace store output is not segmentation-independent");
    }

    for (int bits : { 2, 3, 4, 5, 6, 8 }) {
        for (bool value : { false, true }) {
            const std::vector<ggml_fp16_t> cuda_output = test_store_reference_output(
                    gpu_backend, bits, value, 1, 2, 16, 504, false, true, 4);
            const std::vector<ggml_fp16_t> cpu_output = test_store_reference_output(
                    cpu_backend, bits, value, 1, 2, 16, 504, false, true, 4);
            require_close_f16_rmse(cuda_output, cpu_output, 1e-1f, "KVarN CUDA direct-flush store output differs from CPU reference");
        }
    }

    for (bool value : { false, true }) {
        const std::vector<ggml_fp16_t> cuda_output = test_store_reference_output(
                gpu_backend, 4, value, 1, 2, 512, 0, false, false, 9, 2, 4, true);
        const std::vector<ggml_fp16_t> cpu_output = test_store_reference_output(
                cpu_backend, 4, value, 1, 2, 512, 0, false, false, 9, 2, 4, true);
        require_close_f16_rmse(cuda_output, cpu_output, 1e-1f,
                "KVarN CUDA striped-group workspace store output differs from CPU reference");
    }

    for (bool value : { false, true }) {
        const std::vector<ggml_fp16_t> cuda_output = test_store_reference_output(
                gpu_backend, 4, value, 1, 2, 385, 64, true, false);
        const std::vector<ggml_fp16_t> cpu_output = test_store_reference_output(
                cpu_backend, 4, value, 1, 2, 385, 64, true, false);
        require_close_f16_rmse(cuda_output, cpu_output, 1e-1f, "KVarN CUDA stale workspace hint fallback output differs from CPU reference");
    }

    if (store_stats_get != nullptr) {
        test_kvarn_store_route_stats stats = make_test_kvarn_store_route_stats();
        store_stats_get(&stats);
        require(stats.headwide_workspace + stats.headwide_monolithic > 0,
                "KVarN GPU store tests did not exercise a head-wide store route");
        require(stats.single_slice_workspace > 0,
                "KVarN GPU store tests did not exercise the single-slice workspace route");
        require(stats.direct_store > 0,
                "KVarN GPU store tests did not exercise the direct store route");
        require(stats.high_shared_fallback > 0,
                "KVarN GPU store tests did not exercise the high-shared fallback route");
        require(stats.low_shared_store > 0,
                "KVarN GPU store tests did not exercise the low-shared fallback route");
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);
}

// Validates the GPU/CPU rotated reference decode: emitting K_rot (skip the
// inverse-WHT) and then applying R on the host must reproduce normal original-
// domain decode (X_orig = R*K_rot). The same store feeds both, so this
// holds for sink/record/stage groups alike.
static void test_rotated_decode_transform_consistency(enum ggml_backend_dev_type device_type, bool required) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    const int bits = 4;
    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "rotated parity: failed to init ctx");

    constexpr int n_stream = 1;
    constexpr int kv_size = 512;
    constexpr int n_groups_per_stream = kv_size / 128;
    constexpr int n_tokens = 385;
    constexpr int n_heads = 2;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 384 * n_stream);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream * n_stream);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false, 3);
    ggml_tensor * materialized = ggml_kvarn_materialize(
            ctx, records, stored, indices, n_tokens, 0, n_stream, bits, false, 3);
    materialized->op_params[4] = 1;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, materialized);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "rotated parity: failed to allocate tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            for (int d = 0; d < 128; ++d) {
                input[(size_t(t) * n_heads + h) * 128 + d] =
                    std::sin(float(d) * 0.07f + float(h) * 0.13f) + std::cos(float(t) * 0.037f) +
                    float((d * 13 + h * 7 + t * 17) % 31 - 15) * 0.01f;
            }
        }
    }
    std::vector<int64_t> idx(n_tokens);
    for (int t = 0; t < n_tokens; ++t) idx[t] = t;
    std::vector<uint8_t> zeros(std::max(ggml_nbytes(stage), ggml_nbytes(records)), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, zeros.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, zeros.data(), 0, ggml_nbytes(records));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "rotated parity: graph compute failed");

    std::vector<ggml_fp16_t> orig_h = test_kvarn_reference_decode(
            records, stored, idx, n_tokens, 0, n_stream, bits, false, 3);
    std::vector<ggml_fp16_t> rot_h(ggml_nelements(materialized));
    ggml_backend_tensor_get(materialized, rot_h.data(), 0, ggml_nbytes(materialized));

    double sum_sq = 0.0;
    double max_diff = 0.0;
    size_t count = 0;
    std::array<float, 128> buf;
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            const size_t base = (size_t(t) * n_heads + h) * 128;
            for (int d = 0; d < 128; ++d) buf[d] = ggml_fp16_to_fp32(rot_h[base + d]);
            llama_kvarn_hadamard_128(buf.data());
            for (int d = 0; d < 128; ++d) {
                const float ref = ggml_fp16_to_fp32(orig_h[base + d]);
                require(std::isfinite(ref) && std::isfinite(buf[d]), "rotated decode transform produced non-finite value");
                const double diff = double(ref) - double(buf[d]);
                sum_sq += diff * diff;
                max_diff = std::max(max_diff, std::fabs(diff));
                ++count;
            }
        }
    }
    const double rmse = std::sqrt(sum_sq / std::max<size_t>(count, 1));
    require(rmse <= 5e-4, "rotated decode inverse-WHT RMSE too high");
    require(max_diff <= 2e-3, "rotated decode inverse-WHT max error too high");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// W2 dynamic-stage-depth test: verifies the store path honors
// stage_groups carried in op_params[7] instead of the legacy three-slot stride.
// Writes 768 tokens (6 groups) through a 5-deep stage (tail_groups=4)
// and checks reconstruction of sink, compressed, previous-tail, and live-tail
// groups. Writing 6 groups with stage_groups=5 forces group 5 to reuse transient
// slot 1, flushing the completed group 1 to records — exercising the dynamic
// tail_groups flush predicate (group > tail_groups instead of group > 2) and
// the slot reuse modulo (1 + ((group - 1) % tail_groups)).
static void test_cache_ops_dynamic_stage(enum ggml_backend_dev_type device_type, bool required, int bits) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "dynamic-stage: failed to initialize ggml context");

    constexpr int n_tokens = 768;     // 6 complete groups (0..5)
    constexpr int n_heads   = 1;
    constexpr int stage_groups = 5;   // tail_groups = 4
    constexpr int tail_groups  = stage_groups - 1;
    // 8 record groups per stream (kv_size = 1024) — enough to hold 6 groups with
    // room for the flush ring to grow without collision.
    constexpr int n_groups_per_stream = 8;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage    = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups);
    ggml_tensor * records  = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false, stage_groups);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "dynamic-stage: failed to allocate tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            input[t * 128 + d] =
                std::sin(float(d) * 0.071f) +
                std::cos(float(t) * 0.037f) +
                float((d * 13 + t * 17) % 31 - 15) * 0.01f;
        }
    }
    std::vector<int64_t> idx(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        idx[i] = i;
    }
    std::vector<uint8_t> zeros(ggml_nbytes(stage) + ggml_nbytes(records), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, zeros.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, zeros.data(), 0, ggml_nbytes(records));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "dynamic-stage: graph compute failed");

    const std::vector<float> output = test_kvarn_reference_decode_f32(
            records, stored, idx, n_tokens, 0, 1, bits, false, stage_groups);

    // Group decomposition for n_tokens=768, stage_groups=5, tail_groups=4:
    //   group 0: tokens   0..127  (permanent sink, slot 0)
    //   group 1: tokens 128..255  (transient slot 1, flushed to records when group 5 begins)
    //   group 2: tokens 256..383  (transient slot 2)
    //   group 3: tokens 384..511  (transient slot 3)
    //   group 4: tokens 512..639  (transient slot 4)
    //   group 5: tokens 640..767  (reuses transient slot 1, group 1 must be in records)
    // live_group after processing = 5. Stage holds groups 2..5 (tail_groups=4 groups);
    // group 1 comes from records; group 0 is the sink.
    double sink_error = 0.0;          // group 0
    double compressed_error = 0.0;    // group 1 (flushed to records)
    double stage_transit_error = 0.0; // groups 2, 3, 4, 5 (in stage transient slots)
    for (int t = 0; t < n_tokens; ++t) {
        const int group = t / 128;
        for (int d = 0; d < 128; ++d) {
            const double diff = double(input[t * 128 + d]) - double(output[t * 128 + d]);
            if (group == 0) {
                sink_error += diff * diff;
            } else if (group == 1) {
                compressed_error += diff * diff;
            } else {
                stage_transit_error += diff * diff;
            }
        }
    }
    sink_error          = std::sqrt(sink_error          / (128 * 128));
    compressed_error   = std::sqrt(compressed_error    / (128 * 128));
    stage_transit_error = std::sqrt(stage_transit_error / (128 * 512));
    require(sink_error < 0.01,          "dynamic-stage: sink reconstruction error too high");
    require(compressed_error < 0.25,   "dynamic-stage: compressed reconstruction error too high");
    require(stage_transit_error < 0.01, "dynamic-stage: in-stage transient reconstruction error too high");

    // Group 1 must have been flushed to records when group 5 began (reusing slot 1).
    std::vector<uint8_t> record_data(ggml_nbytes(records));
    ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());
    const size_t group1_off = size_t(1) * record_bytes;
    require(std::any_of(record_data.begin() + ptrdiff_t(group1_off),
                        record_data.begin() + ptrdiff_t(group1_off + record_bytes),
                        [](uint8_t v) { return v != 0; }),
            "dynamic-stage: completed group 1 was not flushed to records");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// W2 unaligned-start persistence test: first store a prefix into stage/records,
// then run a second store whose first absolute index is inside an already-open
// non-sink group. Reference decode covers the whole logical range, while the
// live-group input is only the second store's indices, matching the production
// non-SWA contract.
static void test_unaligned_start(enum ggml_backend_dev_type device_type, bool required,
                                  int start_offset, int n_tokens, int stage_groups) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "unaligned-start: failed to initialize ggml context");

    // Start the second store inside group 1 so the unaligned group is non-sink.
    // For stage_groups 3 and 5, these cases also cross enough boundaries to
    // reuse a transient slot and flush group 1 from the second store.
    const int second_start = 128 + start_offset;
    const int total_tokens = second_start + n_tokens;
    constexpr int n_heads = 1;
    const int tail_groups = stage_groups - 1;
    const int bits = 4;
    const int last_group = (total_tokens - 1) / 128;
    const int n_groups_per_stream = std::max(8, last_group + 1);
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current_prefix = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, second_start);
    ggml_tensor * indices_prefix = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, second_start);
    ggml_tensor * current        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices        = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage    = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups);
    ggml_tensor * records  = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream);

    ggml_tensor * stored_prefix = ggml_kvarn_store(ctx, current_prefix, indices_prefix, stage, records, bits, 16, false, stage_groups);
    stored_prefix->op_params[3] = second_start;
    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stored_prefix, records, bits, 16, false, stage_groups);
    stored->op_params[3] = n_tokens;
    // Non-SWA reference decode uses output cell t as absolute position t. The indices
    // tensor is still the second store's index range, which supplies the current
    // live group exactly as production graph construction does.

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "unaligned-start: failed to allocate tensors");

    auto sample = [](int abs_pos, int d) {
        return std::sin(float(d) * 0.071f) +
            std::cos(float(abs_pos) * 0.037f) +
            float((d * 13 + abs_pos * 17) % 31 - 15) * 0.01f;
    };
    std::vector<float> prefix_input(128 * n_heads * second_start);
    std::vector<float> input(128 * n_heads * n_tokens);
    std::vector<float> expected(128 * n_heads * total_tokens);
    for (int t = 0; t < total_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            const float value = sample(t, d);
            expected[t * 128 + d] = value;
            if (t < second_start) {
                prefix_input[t * 128 + d] = value;
            } else {
                input[(t - second_start) * 128 + d] = value;
            }
        }
    }
    std::vector<int64_t> idx_prefix(second_start);
    for (int i = 0; i < second_start; ++i) {
        idx_prefix[i] = int64_t(i);
    }
    std::vector<int64_t> idx(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        idx[i] = int64_t(second_start + i);
    }
    require(idx.front() == second_start && (idx.front() % 128) == start_offset,
            "unaligned-start: second store did not start at the requested unaligned absolute position");
    std::vector<uint8_t> stage_zeros(ggml_nbytes(stage), 0);
    std::vector<uint8_t> record_zeros(ggml_nbytes(records), 0);

    ggml_backend_tensor_set(current_prefix, prefix_input.data(), 0, ggml_nbytes(current_prefix));
    ggml_backend_tensor_set(indices_prefix, idx_prefix.data(), 0, ggml_nbytes(indices_prefix));
    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, stage_zeros.data(), 0, stage_zeros.size());
    ggml_backend_tensor_set(records, record_zeros.data(), 0, record_zeros.size());

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "unaligned-start: graph compute failed");

    const std::vector<float> output = test_kvarn_reference_decode_f32(
            records, stored, idx, total_tokens, 0, 1, bits, false, stage_groups);

    // Verify reconstruction across prefix and second store, including the group
    // partially filled by the prefix and completed by the second store.
    double mse = 0.0;
    double max_diff = 0.0;
    for (int t = 0; t < total_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            const double diff = double(expected[t * 128 + d]) - double(output[t * 128 + d]);
            mse += diff * diff;
            max_diff = std::max(max_diff, std::fabs(diff));
        }
    }
    const double rmse = std::sqrt(mse / double(total_tokens * 128));
    if (!std::isfinite(rmse) || rmse >= 0.30) {
        std::fprintf(stderr, "unaligned-start: reconstruction RMSE too high (start=%d abs=%d n=%d sg=%d rmse=%g max=%g)\n",
                start_offset, second_start, n_tokens, stage_groups, rmse, max_diff);
        require(false, "unaligned-start: reconstruction RMSE too high");
    }
    if (max_diff >= 2.0) {
        std::fprintf(stderr, "unaligned-start: max reconstruction error too high (start=%d abs=%d n=%d sg=%d max=%g)\n",
                start_offset, second_start, n_tokens, stage_groups, max_diff);
        require(false, "unaligned-start: max reconstruction error too high");
    }

    if (last_group > tail_groups) {
        std::vector<uint8_t> record_data(ggml_nbytes(records));
        ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());
        const int flushed_group = last_group - tail_groups;
        const size_t off = size_t(flushed_group) * size_t(record_bytes);
        require(std::any_of(record_data.begin() + ptrdiff_t(off),
                            record_data.begin() + ptrdiff_t(off + record_bytes),
                            [](uint8_t v) { return v != 0; }),
                "unaligned-start: second store did not flush the reused transient slot");
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// Eager unaligned-start coverage at the production non-SWA stage depth
// (stage_groups=2 => tail_groups=1). The second store is large enough to take
// the CUDA/Vulkan bulk workspace path (>= 3 * 128 tokens) and begins inside an
// already-open non-sink group. That straddling group completes inside the
// second store, so it must reach `records`.
static void test_eager_unaligned_start(enum ggml_backend_dev_type device_type, bool required,
                                       int start_offset, int n_tokens, int bits) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    const int stage_groups = int(llama_kvarn_non_swa_tail_groups(0, 0) + 1);
    constexpr int n_heads = 1;

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "eager-unaligned: failed to initialize ggml context");

    const int second_start = 128 + start_offset;
    const int total_tokens = second_start + n_tokens;
    const int straddling_group = second_start / 128;
    const int last_group = (total_tokens - 1) / 128;
    const int n_groups_per_stream = std::max(8, last_group + 1);
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    require(start_offset > 0 && start_offset < 128, "eager-unaligned: start must be inside a group");
    require((straddling_group + 1) * 128 <= total_tokens,
            "eager-unaligned: straddling group must complete inside the second store");

    ggml_tensor * current_prefix = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, second_start);
    ggml_tensor * indices_prefix = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, second_start);
    ggml_tensor * current        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices        = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage    = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups);
    ggml_tensor * records  = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream);

    ggml_tensor * stored_prefix = ggml_kvarn_store(ctx, current_prefix, indices_prefix, stage, records, bits, 16, false, stage_groups);
    stored_prefix->op_params[3] = second_start;
    stored_prefix->op_params[9] = 1; // eager records, as the KVarN cache always sets
    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stored_prefix, records, bits, 16, false, stage_groups);
    stored->op_params[3] = n_tokens;
    stored->op_params[9] = 1;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "eager-unaligned: failed to allocate tensors");

    auto sample = [](int abs_pos, int d) {
        return std::sin(float(d) * 0.071f) +
            std::cos(float(abs_pos) * 0.037f) +
            float((d * 13 + abs_pos * 17) % 31 - 15) * 0.01f;
    };
    std::vector<float> prefix_input(128 * n_heads * second_start);
    std::vector<float> input(128 * n_heads * n_tokens);
    std::vector<float> expected(128 * n_heads * total_tokens);
    for (int t = 0; t < total_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            const float value = sample(t, d);
            expected[t * 128 + d] = value;
            if (t < second_start) {
                prefix_input[t * 128 + d] = value;
            } else {
                input[(t - second_start) * 128 + d] = value;
            }
        }
    }
    std::vector<int64_t> idx_prefix(second_start);
    for (int i = 0; i < second_start; ++i) {
        idx_prefix[i] = int64_t(i);
    }
    std::vector<int64_t> idx(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        idx[i] = int64_t(second_start + i);
    }
    std::vector<uint8_t> stage_zeros(ggml_nbytes(stage), 0);
    std::vector<uint8_t> record_zeros(ggml_nbytes(records), 0);

    ggml_backend_tensor_set(current_prefix, prefix_input.data(), 0, ggml_nbytes(current_prefix));
    ggml_backend_tensor_set(indices_prefix, idx_prefix.data(), 0, ggml_nbytes(indices_prefix));
    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, stage_zeros.data(), 0, stage_zeros.size());
    ggml_backend_tensor_set(records, record_zeros.data(), 0, record_zeros.size());

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "eager-unaligned: graph compute failed");

    // Direct check: the straddling group completed in this store, so its record
    // must have been written.
    std::vector<uint8_t> record_data(ggml_nbytes(records));
    ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());
    const size_t off = size_t(straddling_group) * size_t(record_bytes);
    const bool straddling_written = std::any_of(
            record_data.begin() + ptrdiff_t(off),
            record_data.begin() + ptrdiff_t(off + record_bytes),
            [](uint8_t v) { return v != 0; });
    // Investigation aid: KVARN_EAGER_UNALIGNED_SOFT=1 reports the whole matrix
    // instead of aborting on the first failure.
    const bool soft = std::getenv("KVARN_EAGER_UNALIGNED_SOFT") != nullptr;
    if (soft) {
        std::fprintf(stderr, "eager-unaligned[%s]: start=%3d n=%3d bits=%d group=%d record=%s\n",
                device_type == GGML_BACKEND_DEVICE_TYPE_CPU ? "CPU" : "GPU",
                start_offset, n_tokens, bits, straddling_group,
                straddling_written ? "WRITTEN" : "UNWRITTEN <-- LOST");
    }
    if (!straddling_written && !soft) {
        std::fprintf(stderr,
                "eager-unaligned: group %d completed in the store but its record is unwritten "
                "(start=%d abs=%d n=%d bits=%d)\n",
                straddling_group, start_offset, second_start, n_tokens, bits);
        require(false, "eager-unaligned: completed straddling group was never quantized");
    }
    if (!straddling_written) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    const std::vector<float> output = test_kvarn_reference_decode_f32(
            records, stored, idx, total_tokens, 0, 1, bits, false, stage_groups);

    double mse = 0.0;
    double max_diff = 0.0;
    double group_mse = 0.0;
    for (int t = 0; t < total_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            const double diff = double(expected[t * 128 + d]) - double(output[t * 128 + d]);
            mse += diff * diff;
            max_diff = std::max(max_diff, std::fabs(diff));
            if (t / 128 == straddling_group) {
                group_mse += diff * diff;
            }
        }
    }
    const double rmse = std::sqrt(mse / double(total_tokens * 128));
    const double group_rmse = std::sqrt(group_mse / double(128 * 128));
    if (!std::isfinite(rmse) || rmse >= 0.30 || group_rmse >= 0.30) {
        std::fprintf(stderr,
                "eager-unaligned: reconstruction error too high (start=%d abs=%d n=%d bits=%d "
                "rmse=%g straddling-group-%d rmse=%g max=%g)\n",
                start_offset, second_start, n_tokens, bits, rmse, straddling_group, group_rmse, max_diff);
        require(false, "eager-unaligned: reconstruction error too high");
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// Investigation: does a partial rollback into an already-sealed group leak the
// newer group's stage rows into that group's record?
//
// Sequence (all stores are small, so they take the per-token path that decode
// and speculative rollback use):
//   1. seal group 1 completely            -> stage slot 1 holds group 1
//   2. write the first `p` tokens of group 2 -> slot 1 positions [0,p) now hold group 2
//   3. roll back and re-complete group 1 from offset q > p
// Step 3 rewrites slot 1 positions [q,128) and re-seals group 1 from the whole
// slot, so positions [0,p) are sourced from group 2 if the leak is real.
//
// The control run performs steps 1 and 3 only. Group 1's record must be
// byte-identical between the two runs; any difference is the leak.
static std::vector<uint8_t> kvarn_reseal_after_rollback_record(
        ggml_backend_t backend, bool write_next_group, int p, int q, int bits, int stage_groups) {
    constexpr int n_heads = 1;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_init_params params = { /*.mem_size =*/ 8 * 1024 * 1024, nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "reseal: failed to initialize ggml context");

    const int n_next = write_next_group ? p : 0;
    const int n_tail = 128 - q;

    ggml_tensor * g1     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, 128);
    ggml_tensor * g1_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 128);
    ggml_tensor * nxt     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, std::max(1, n_next));
    ggml_tensor * nxt_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, std::max(1, n_next));
    ggml_tensor * tail     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tail);
    ggml_tensor * tail_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tail);
    ggml_tensor * stage   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, 4);

    auto mk_store = [&](ggml_tensor * cur, ggml_tensor * idx, ggml_tensor * prev) {
        ggml_tensor * s = ggml_kvarn_store(ctx, cur, idx, prev, records, bits, 16, false, stage_groups);
        s->op_params[3] = (int32_t) cur->ne[2];
        s->op_params[9] = 1; // eager records, as the cache always sets
        return s;
    };
    ggml_tensor * chain = mk_store(g1, g1_idx, stage);
    if (write_next_group) {
        chain = mk_store(nxt, nxt_idx, chain);
    }
    chain = mk_store(tail, tail_idx, chain);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, chain);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "reseal: failed to allocate tensors");

    // Group 1 rows, the next group's rows, and the refilled tail are all
    // deliberately far apart so a leak cannot be mistaken for quantization noise.
    auto orig = [](int pos, int d) { return std::sin(float(pos) * 0.031f) + std::cos(float(d) * 0.017f); };
    auto next = [](int pos, int d) { return 6.0f + 3.0f * std::sin(float(pos * 7 + d) * 0.11f); };
    auto refill = [](int pos, int d) { return -4.0f + std::cos(float(pos * 3 + d) * 0.09f); };

    std::vector<float>   g1_data(128 * 128), nxt_data(128 * std::max(1, n_next)), tail_data(128 * n_tail);
    std::vector<int64_t> g1_i(128), nxt_i(std::max(1, n_next)), tail_i(n_tail);
    for (int t = 0; t < 128; ++t) {
        g1_i[t] = 128 + t;
        for (int d = 0; d < 128; ++d) { g1_data[t * 128 + d] = orig(t, d); }
    }
    for (int t = 0; t < n_next; ++t) {
        nxt_i[t] = 256 + t;
        for (int d = 0; d < 128; ++d) { nxt_data[t * 128 + d] = next(t, d); }
    }
    if (n_next == 0) { nxt_i[0] = -1; }
    for (int t = 0; t < n_tail; ++t) {
        tail_i[t] = 128 + q + t;
        for (int d = 0; d < 128; ++d) { tail_data[t * 128 + d] = refill(q + t, d); }
    }

    std::vector<uint8_t> zeros_stage(ggml_nbytes(stage), 0);
    std::vector<uint8_t> zeros_rec(ggml_nbytes(records), 0);
    ggml_backend_tensor_set(stage, zeros_stage.data(), 0, zeros_stage.size());
    ggml_backend_tensor_set(records, zeros_rec.data(), 0, zeros_rec.size());
    ggml_backend_tensor_set(g1, g1_data.data(), 0, ggml_nbytes(g1));
    ggml_backend_tensor_set(g1_idx, g1_i.data(), 0, ggml_nbytes(g1_idx));
    ggml_backend_tensor_set(nxt, nxt_data.data(), 0, ggml_nbytes(nxt));
    ggml_backend_tensor_set(nxt_idx, nxt_i.data(), 0, ggml_nbytes(nxt_idx));
    ggml_backend_tensor_set(tail, tail_data.data(), 0, ggml_nbytes(tail));
    ggml_backend_tensor_set(tail_idx, tail_i.data(), 0, ggml_nbytes(tail_idx));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "reseal: graph compute failed");

    std::vector<uint8_t> all(ggml_nbytes(records));
    ggml_backend_tensor_get(records, all.data(), 0, all.size());
    std::vector<uint8_t> group1(all.begin() + record_bytes, all.begin() + 2 * record_bytes);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return group1;
}

static void test_reseal_after_partial_rollback(enum ggml_backend_dev_type device_type, bool required) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }
    const char * dev = device_type == GGML_BACKEND_DEVICE_TYPE_CPU ? "CPU" : "GPU";
    constexpr int bits = 6;
    const int q = 126;

    // Expected rotated-domain contents of group 1 after the rollback: the
    // surviving original prefix, then the refilled tail.
    auto orig   = [](int pos, int d) { return std::sin(float(pos) * 0.031f) + std::cos(float(d) * 0.017f); };
    auto refill = [](int pos, int d) { return -4.0f + std::cos(float(pos * 3 + d) * 0.09f); };
    std::vector<float> expect(128 * 128);
    for (int pos = 0; pos < 128; ++pos) {
        std::array<float, 128> row{};
        for (int d = 0; d < 128; ++d) { row[d] = pos < q ? orig(pos, d) : refill(pos, d); }
        llama_kvarn_hadamard_128(row.data());
        for (int d = 0; d < 128; ++d) { expect[pos * 128 + d] = row[d]; }
    }

    auto rmse_over = [&](const std::vector<uint8_t> & rec, int pos0, int pos1) {
        double se = 0.0; int n = 0;
        for (int pos = pos0; pos < pos1; ++pos) {
            for (int d = 0; d < 128; ++d) {
                const double got = test_kvarn_record_value(rec.data(), bits, false, pos, d);
                const double dif = got - double(expect[pos * 128 + d]);
                se += dif * dif; ++n;
            }
        }
        return n > 0 ? std::sqrt(se / double(n)) : 0.0;
    };

    // Track the production stage depth rather than hard-coding it, so this test
    // follows llama_kvarn_non_swa_tail_groups if that policy ever changes.
    const int stage_groups = int(llama_kvarn_non_swa_tail_groups(0, 0)) + 1;

    for (int p : { 1, 4, 16 }) {
        const auto control = kvarn_reseal_after_rollback_record(backend, false, p, q, bits, stage_groups);
        const auto leaked  = kvarn_reseal_after_rollback_record(backend, true,  p, q, bits, stage_groups);
        size_t diff = 0;
        for (size_t i = 0; i < control.size(); ++i) {
            diff += control[i] != leaked[i] ? 1 : 0;
        }
        if (diff != 0) {
            std::fprintf(stderr,
                    "reseal-rollback[%s]: stage_groups=%d p=%2d -> %zu/%zu record bytes differ | "
                    "leaked rows [0,%d) rmse %.4f -> %.4f | untouched rows [%d,%d) rmse %.4f -> %.4f\n",
                    dev, stage_groups, p, diff, control.size(),
                    p, rmse_over(control, 0, p), rmse_over(leaked, 0, p),
                    p, q, rmse_over(control, p, q), rmse_over(leaked, p, q));
            require(false, "reseal-rollback: reopening a sealed group leaked the newer group's stage rows");
        }
    }
    ggml_backend_free(backend);
}

static void test_eager_completed_record(enum ggml_backend_dev_type device_type, bool required) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "eager-record: failed to initialize ggml context");

    constexpr int bits = 4;
    constexpr int n_heads = 1;
    constexpr int stage_groups = 2;
    constexpr int prefix_tokens = 64;
    constexpr int suffix_tokens = 64;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * prefix = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, prefix_tokens);
    ggml_tensor * prefix_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, prefix_tokens);
    ggml_tensor * suffix = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, suffix_tokens);
    ggml_tensor * suffix_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, suffix_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, 4);

    ggml_tensor * stored_prefix = ggml_kvarn_store(
            ctx, prefix, prefix_idxs, stage, records, bits, 16, false, stage_groups);
    stored_prefix->op_params[9] = 1;
    ggml_tensor * stored = ggml_kvarn_store(
            ctx, suffix, suffix_idxs, stored_prefix, records, bits, 16, false, stage_groups);
    stored->op_params[9] = 1;
    ggml_tensor * decode_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 257);
    ggml_tensor * materialized = ggml_kvarn_materialize(
            ctx, records, stored, decode_indices, 256, 0, 1, bits, false, stage_groups);
    materialized->op_params[9] = 1;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, materialized);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "eager-record: failed to allocate tensors");

    std::vector<float> prefix_data(128 * prefix_tokens);
    std::vector<float> suffix_data(128 * suffix_tokens);
    std::vector<int64_t> prefix_idx(prefix_tokens);
    std::vector<int64_t> suffix_idx(suffix_tokens);
    for (int t = 0; t < prefix_tokens; ++t) {
        prefix_idx[t] = 128 + t;
        for (int d = 0; d < 128; ++d) {
            prefix_data[t * 128 + d] = std::sin(float(128 + t) * 0.03f) + std::cos(float(d) * 0.07f);
        }
    }
    for (int t = 0; t < suffix_tokens; ++t) {
        suffix_idx[t] = 128 + prefix_tokens + t;
        for (int d = 0; d < 128; ++d) {
            suffix_data[t * 128 + d] = std::sin(float(192 + t) * 0.03f) + std::cos(float(d) * 0.07f);
        }
    }
    std::vector<uint8_t> zeros_stage(ggml_nbytes(stage), 0);
    std::vector<uint8_t> zeros_records(ggml_nbytes(records), 0);
    std::vector<int64_t> decode_idxs(257);
    for (size_t i = 0; i < decode_idxs.size(); ++i) {
        decode_idxs[i] = int64_t(i);
    }
    ggml_backend_tensor_set(prefix, prefix_data.data(), 0, ggml_nbytes(prefix));
    ggml_backend_tensor_set(prefix_idxs, prefix_idx.data(), 0, ggml_nbytes(prefix_idxs));
    ggml_backend_tensor_set(suffix, suffix_data.data(), 0, ggml_nbytes(suffix));
    ggml_backend_tensor_set(suffix_idxs, suffix_idx.data(), 0, ggml_nbytes(suffix_idxs));
    ggml_backend_tensor_set(decode_indices, decode_idxs.data(), 0, ggml_nbytes(decode_indices));
    ggml_backend_tensor_set(stage, zeros_stage.data(), 0, zeros_stage.size());
    ggml_backend_tensor_set(records, zeros_records.data(), 0, zeros_records.size());

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "eager-record: graph compute failed");
    std::vector<uint8_t> record_data(ggml_nbytes(records));
    ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());
    const size_t group1 = size_t(record_bytes);
    require(std::any_of(record_data.begin() + ptrdiff_t(group1),
                        record_data.begin() + ptrdiff_t(group1 + record_bytes),
                        [](uint8_t v) { return v != 0; }),
            "eager-record: completed group was not materialized at its closing token");

    // Advance the logical live group in the reference view so group 1 is read
    // from its eagerly committed record instead of the transient stage slot.
    const std::vector<ggml_fp16_t> decoded = test_kvarn_reference_decode(
            records, stored, decode_idxs, 256, 0, 1, bits, false, stage_groups);
    std::vector<ggml_fp16_t> materialized_data(ggml_nelements(materialized));
    ggml_backend_tensor_get(materialized, materialized_data.data(), 0, ggml_nbytes(materialized));
    double eager_record_mse = 0.0;
    double eager_materialized_mse = 0.0;
    for (int t = 0; t < 128; ++t) {
        for (int d = 0; d < 128; ++d) {
            const float expected = t < prefix_tokens
                ? prefix_data[t * 128 + d]
                : suffix_data[(t - prefix_tokens) * 128 + d];
            const float actual = ggml_fp16_to_fp32(decoded[(128 + t) * 128 + d]);
            const double diff = double(actual) - double(expected);
            eager_record_mse += diff * diff;
            const double materialized_diff =
                double(ggml_fp16_to_fp32(materialized_data[(128 + t) * 128 + d])) - double(expected);
            eager_materialized_mse += materialized_diff * materialized_diff;
        }
    }
    const double eager_record_rmse = std::sqrt(eager_record_mse / double(128 * 128));
    require(eager_record_rmse < 0.25,
            "eager-record: completed record reconstruction error too high");
    require(std::sqrt(eager_materialized_mse / double(128 * 128)) < 0.25,
            "eager-record: materialized completed record reconstruction error too high");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static ggml_backend_meta_split_state test_kvarn_meta_split(
        const ggml_tensor * tensor, void *) {
    if (std::strcmp(tensor->name, "meta_current") == 0 ||
            std::strcmp(tensor->name, "meta_current_next") == 0 ||
            std::strcmp(tensor->name, "meta_stage") == 0 ||
            std::strcmp(tensor->name, "meta_records") == 0) {
        ggml_backend_meta_split_state result = {
            GGML_BACKEND_SPLIT_AXIS_1, { 0 }, { 1 }, 1
        };
        // Two complete heads over three devices: the first shard is a valid
        // zero-head no-op and the remaining devices own one head each.
        result.ne[0] = 0;
        result.ne[1] = 1;
        result.ne[2] = 1;
        return result;
    }
    return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, { 0 }, { 1 }, 1 };
}

static void test_meta_kvarn_zero_head_shard() {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    require(cpu != nullptr, "meta KVarN: CPU device unavailable");
    ggml_backend_dev_t devices[] = { cpu, cpu, cpu };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(
            devices, 3, test_kvarn_meta_split, nullptr);
    require(meta_dev != nullptr, "meta KVarN: failed to create meta device");
    ggml_backend_t backend = ggml_backend_dev_init(meta_dev, nullptr);
    require(backend != nullptr, "meta KVarN: failed to create backend");

    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * static_ctx = ggml_init(params);
    require(static_ctx != nullptr, "meta KVarN: failed to create static context");
    ggml_context * input_ctx = ggml_init(params);
    require(input_ctx != nullptr, "meta KVarN: failed to create input context");

    constexpr int n_heads = 2;
    constexpr int n_tokens = 128;
    constexpr int stage_groups = 3;
    constexpr int bits = 4;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) +
            3 * 128 * sizeof(ggml_fp16_t));
    ggml_tensor * current = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * current_next = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    // Index graph inputs can remain ordinary scheduler-owned tensors in
    // production. Split-state recursion must retain the owning meta context
    // while following this non-meta source.
    ggml_tensor * indices = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups);
    ggml_tensor * records = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_I8, record_bytes, n_heads, 4);
    ggml_set_name(current, "meta_current");
    ggml_set_name(current_next, "meta_current_next");
    ggml_set_name(stage, "meta_stage");
    ggml_set_name(records, "meta_records");

    ggml_backend_buffer_t static_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            static_ctx, ggml_backend_dev_buffer_type(meta_dev));
    require(static_buffer != nullptr, "meta KVarN: failed to allocate split tensors");
    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            input_ctx, ggml_backend_dev_buffer_type(cpu));
    require(input_buffer != nullptr, "meta KVarN: failed to allocate mirrored input tensors");

    ggml_context * compute_ctx = ggml_init(params);
    require(compute_ctx != nullptr, "meta KVarN: failed to create compute context");
    ggml_tensor * stored = ggml_kvarn_store(
            compute_ctx, current, indices, stage, records, bits, 16, false, stage_groups);
    ggml_tensor * materialized = ggml_kvarn_materialize(
            compute_ctx, records, stored, indices, n_tokens, 0, 1,
            bits, false, stage_groups);
    ggml_cgraph * graph = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(graph, materialized);
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(
            GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(cpu_backend != nullptr, "meta KVarN: failed to initialize scheduler fallback");
    ggml_backend_t backends[] = { backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
            backends, nullptr, 2, 32, false, true);
    require(sched != nullptr && ggml_backend_sched_alloc_graph(sched, graph),
            "meta KVarN: failed to allocate scheduled graph");

    std::vector<float> input(size_t(128) * n_heads * n_tokens);
    for (int token = 0; token < n_tokens; ++token) {
        for (int head = 0; head < n_heads; ++head) {
            for (int dim = 0; dim < 128; ++dim) {
                input[(size_t(token)*n_heads + head)*128 + dim] =
                        std::sin(float(token)*0.03f) +
                        std::cos(float(dim)*0.05f) + float(head)*0.25f;
            }
        }
    }
    std::vector<int64_t> index_data(n_tokens);
    std::iota(index_data.begin(), index_data.end(), int64_t(0));
    std::vector<float> input_next(input.size());
    for (size_t i = 0; i < input_next.size(); ++i) {
        input_next[i] = input[i] + 2.0f;
    }
    std::vector<uint8_t> stage_zeros(ggml_nbytes(stage), 0);
    std::vector<uint8_t> record_zeros(ggml_nbytes(records), 0);
    ggml_backend_tensor_memset(records, 0xa5, 0, ggml_nbytes(records));
    std::vector<uint8_t> memset_records(ggml_nbytes(records), 0);
    ggml_backend_tensor_get(records, memset_records.data(), 0, memset_records.size());
    require(std::all_of(memset_records.begin(), memset_records.end(),
                    [](uint8_t value) { return value == 0xa5; }),
            "meta KVarN: split tensor memset did not cover every logical record byte");
    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(current_next, input_next.data(), 0, ggml_nbytes(current_next));
    ggml_backend_tensor_set(indices, index_data.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, stage_zeros.data(), 0, stage_zeros.size());
    ggml_backend_tensor_set(records, record_zeros.data(), 0, record_zeros.size());
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
            "meta KVarN: split store/materialize graph failed");

    std::vector<ggml_fp16_t> output(ggml_nelements(materialized));
    ggml_backend_tensor_get(materialized, output.data(), 0, ggml_nbytes(materialized));
    double max_error = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        max_error = std::max(max_error,
                std::fabs(double(ggml_fp16_to_fp32(output[i])) - input[i]));
    }
    require(max_error < 0.002,
            "meta KVarN: zero-head split changed lossless staged values");

    // The scheduler deliberately reuses the parent graph identity while the
    // graph-local KVarN node changes its persistent input. A stale projected
    // meta node would continue reading `current` here. This is the lifecycle
    // exercised by decode graphs whose graph-local tensors are rebuilt in
    // place between tokens.
    stored->src[0] = current_next;
    ggml_backend_tensor_set(stage, stage_zeros.data(), 0, stage_zeros.size());
    ggml_backend_tensor_set(records, record_zeros.data(), 0, record_zeros.size());
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
            "meta KVarN: rebuilt split graph failed");
    ggml_backend_tensor_get(materialized, output.data(), 0, ggml_nbytes(materialized));
    max_error = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        max_error = std::max(max_error,
                std::fabs(double(ggml_fp16_to_fp32(output[i])) - input_next[i]));
    }
    require(max_error < 0.005,
            "meta KVarN: projected graph retained a stale graph-local source");

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);
    ggml_free(compute_ctx);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(input_ctx);
    ggml_backend_buffer_free(static_buffer);
    ggml_free(static_ctx);
    ggml_backend_free(backend);
}

int main() {
    ggml_backend_load_all();

    if (std::getenv("GGML_KVARN_TEST_AMD_ROUTE_BOUNDARIES_ONLY") != nullptr) {
        test_native_flash_attention_support_gates();
        test_native_flash_attention_gpu();
        std::printf("test-kvarn: AMD route-boundary matrix OK\n");
        return 0;
    }

    if (std::getenv("GGML_KVARN_TEST_PORTABLE_NATIVE_ONLY") != nullptr) {
        test_native_flash_attention_support_gates();
        test_native_flash_attention_cpu();
        test_native_flash_attention_gpu();
        std::printf("test-kvarn: portable native attention OK\n");
        return 0;
    }

    if (std::getenv("GGML_KVARN_TEST_PREFILL_PARITY_ONLY") != nullptr) {
        test_native_flash_attention_prefill_route_parity();
        std::printf("test-kvarn: prefill route parity OK\n");
        return 0;
    }

    kvarn_composite_exclusivity_forwards();
    kvarn_composite_removal_plan_forwards();
    kvarn_unified_save_requires_exclusive_stream();
    kvarn_unified_restore_requires_exclusive_stream();
    kvarn_selective_state_owns_only_live_stage_rows();
    kvarn_compact_read_plan_skips_ownership_holes();
    test_type_table();
    test_attention_domain_policy();
    test_vulkan_decode_route_policy();
    test_stage_policy();
    test_memory_stats_aggregation();
    test_meta_kvarn_zero_head_shard();
    test_exact_tail_policy();
    test_tile_layout();
    test_head_dimension_slicing();
    test_runtime_validation();
    iswa_nonunified_multislot_kvarn_policy();
    test_remove_policy();
    kvarn_historical_suffix_rejects_contended_unified_stream();
    kvarn_historical_suffix_plans_group_boundary();
    kvarn_swa_deep_rollback_plans_group_boundary();
    test_pack_roundtrip(2);
    test_pack_roundtrip(3);
    test_pack_roundtrip(4);
    test_pack_roundtrip(5);
    test_pack_roundtrip(6);
    test_pack_roundtrip(8);
    test_hadamard_roundtrip();
    test_rotated_domain_equivalence();
    for (int head_width : { 128, 256, 512 }) {
        test_kvarn_wht_op(GGML_BACKEND_DEVICE_TYPE_CPU, true,  head_width);
        test_kvarn_wht_op(GGML_BACKEND_DEVICE_TYPE_GPU, false, head_width);
        for (ggml_type input_type : { GGML_TYPE_F16, GGML_TYPE_BF16 }) {
            test_kvarn_wht_op(GGML_BACKEND_DEVICE_TYPE_CPU, true, head_width, input_type);
            test_kvarn_wht_op(GGML_BACKEND_DEVICE_TYPE_GPU, false, head_width, input_type);
        }
    }

    for (size_t i = 0; i < llama_kvarn_type_count(); ++i) {
        const llama_kvarn_type type = (llama_kvarn_type) i;
        if (type != LLAMA_KVARN_TYPE_DISABLED) {
            test_tile_quantization(type);
        }
    }

    for (int bits : { 3, 5, 6, 8 }) {
        test_cache_ops(GGML_BACKEND_DEVICE_TYPE_CPU, true, bits);
        test_cache_ops(GGML_BACKEND_DEVICE_TYPE_GPU, false, bits);
    }
    for (int head_slices : { 2, 4 }) {
        test_cache_ops(GGML_BACKEND_DEVICE_TYPE_CPU, true, 4, head_slices);
        test_cache_ops(GGML_BACKEND_DEVICE_TYPE_GPU, false, 4, head_slices);
    }
    // W2 dynamic stage-depth coverage: stage_groups=5 (tail_groups=4).
    test_cache_ops_dynamic_stage(GGML_BACKEND_DEVICE_TYPE_CPU, true, 4);
    test_cache_ops_dynamic_stage(GGML_BACKEND_DEVICE_TYPE_GPU, false, 4);
    test_eager_completed_record(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    test_eager_completed_record(GGML_BACKEND_DEVICE_TYPE_GPU, false);
    test_reseal_after_partial_rollback(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    test_reseal_after_partial_rollback(GGML_BACKEND_DEVICE_TYPE_GPU, false);
    // W2 unaligned-start coverage: first persist a prefix, then run a second
    // store whose first index is 128 + {1, 64, 127}. The 256/512 cases force a
    // transient slot reuse/flush from the second store; all cases decode
    // the full prefix + second-store range from the same stage/records.
    for (int start : { 1, 64, 127 }) {
        test_unaligned_start(GGML_BACKEND_DEVICE_TYPE_CPU, true,  start, 256, 3);
        test_unaligned_start(GGML_BACKEND_DEVICE_TYPE_GPU, false, start, 256, 3);
        test_unaligned_start(GGML_BACKEND_DEVICE_TYPE_CPU, true,  start, 512, 5);
        test_unaligned_start(GGML_BACKEND_DEVICE_TYPE_GPU, false, start, 512, 5);
        test_unaligned_start(GGML_BACKEND_DEVICE_TYPE_CPU, true,  start, 513, 6);
        test_unaligned_start(GGML_BACKEND_DEVICE_TYPE_GPU, false, start, 513, 6);
    }
    // Eager unaligned-start coverage at the production stage depth (stage_groups=2).
    // 512/535 tokens take the GPU bulk workspace path; 256 stays on the per-token path.
    for (int start : { 1, 64, 88, 127 }) {
        for (int n : { 256, 512, 535 }) {
            test_eager_unaligned_start(GGML_BACKEND_DEVICE_TYPE_CPU, true,  start, n, 6);
            test_eager_unaligned_start(GGML_BACKEND_DEVICE_TYPE_GPU, false, start, n, 6);
        }
    }
    test_cache_ops_multi_stream(GGML_BACKEND_DEVICE_TYPE_CPU, true, 6);
    test_cache_ops_multi_stream(GGML_BACKEND_DEVICE_TYPE_GPU, false, 6);
    test_cache_ops_swa(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    test_cache_ops_swa(GGML_BACKEND_DEVICE_TYPE_GPU, false); // CUDA SWA ring parity
    test_store_paths_gpu();
    test_native_flash_attention_support_gates();
    test_native_flash_attention_cpu();
    test_native_flash_attention_gpu();
    test_native_flash_attention_prefill_route_parity();
    test_rotated_decode_transform_consistency(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    test_rotated_decode_transform_consistency(GGML_BACKEND_DEVICE_TYPE_GPU, false);

    std::printf("test-kvarn: all tests OK\n");
    return 0;
}
