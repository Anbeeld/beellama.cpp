#pragma once

#include <algorithm>
#include <cstdint>

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
