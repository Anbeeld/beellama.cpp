#pragma once

#include <algorithm>
#include <cstdint>

// Match generic FA topology normalization, including its unknown-device fallback.
inline uint32_t ggml_vk_kvarn_shader_core_count(
        uint32_t core_count, bool intel, bool xe2) {
    return core_count ? core_count * (intel && !xe2 ? 2u : 1u) : 16u;
}

inline bool ggml_vk_kvarn_store_parallel_preferred(
        uint32_t tokens_per_stream, uint32_t head_workgroups,
        uint32_t active_streams, uint32_t shader_core_count) {
    // MODE_MONOLITHIC dispatches one CTA per logical head and processes
    // every stream and head slice serially. Only that existing grid can make
    // record/stream/slice parallelism unnecessary; do not count parallelism
    // that MODE_FLUSH would introduce as already present in the serial path.
    return tokens_per_stream > 128u && active_streams > 0u &&
        uint64_t(head_workgroups) < uint64_t(shader_core_count) * 2u;
}

inline bool ggml_vk_kvarn_wht_full_head_preferred(
        bool nvidia, uint32_t head_width, uint32_t subgroup_size) {
    // Both variants dispatch exactly one workgroup per row. Preserve the
    // measured NVIDIA route; elsewhere avoid adding subgroups without adding
    // independent rows. Unknown subgroup topology conservatively uses slices.
    if (nvidia) {
        return true;
    }
    if (subgroup_size == 0) {
        return false;
    }
    return (head_width + subgroup_size - 1u) / subgroup_size <=
        (128u + subgroup_size - 1u) / subgroup_size;
}

struct ggml_vk_fattn_kvarn_plan_input {
    uint32_t n_kv;
    uint32_t n_query;
    uint32_t n_query_heads;
    uint32_t n_kv_heads;
    uint32_t n_stream;
    uint32_t shader_core_count;
};

struct ggml_vk_fattn_kvarn_plan_result {
    uint32_t gqa_group_size;
    uint32_t workgroups_y;
    uint32_t split_k;
};

inline ggml_vk_fattn_kvarn_plan_result ggml_vk_fattn_kvarn_plan(
        const ggml_vk_fattn_kvarn_plan_input & input) {
    const uint32_t gqa = input.n_query_heads / input.n_kv_heads;
    const uint32_t gqa_group_size = std::min(gqa, 4u);
    const uint32_t gqa_groups = (gqa + gqa_group_size - 1) / gqa_group_size;
    const uint32_t workgroups_y = input.n_kv_heads * gqa_groups;
    const uint32_t workgroups = std::max(
        1u, input.n_query * workgroups_y * input.n_stream);
    const uint32_t target_workgroups = std::max(1u, input.shader_core_count) * 2u;
    uint32_t split_k = std::max(1u,
        (target_workgroups + workgroups - 1) / workgroups);
    const uint32_t record_groups = std::max(1u, (input.n_kv + 127u) / 128u);
    split_k = std::min(split_k, record_groups);
    return { gqa_group_size, workgroups_y, split_k };
}
