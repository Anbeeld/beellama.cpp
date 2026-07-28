#include "llama-kv-cache-placement.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "check failed at line " << __LINE__ << ": " #condition "\n"; \
        return 1; \
    } \
} while (0)

static ggml_backend_meta_split_state mirrored_split(const ggml_tensor *, void *) {
    return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, { 0 }, { 1 }, 1 };
}

int main() {
    ggml_backend_load_all();

    // Tensor-parallel cache payloads use the upstream meta buffer as their
    // single physical owner. Every persistent component derives its split
    // contract from a typed role rather than a collection of unrelated regexes.
    const auto standard_k = llama_kv_cache_component_from_name("cache_k_l17");
    CHECK(standard_k.valid);
    CHECK(standard_k.role == LLAMA_KV_CACHE_COMPONENT_STANDARD_K);
    CHECK(standard_k.layer_id == 17);
    CHECK(standard_k.split_axis == 0);

    const auto standard_v_tail = llama_kv_cache_component_from_name("cache_v_tail_l3");
    CHECK(standard_v_tail.valid);
    CHECK(standard_v_tail.role == LLAMA_KV_CACHE_COMPONENT_STANDARD_V_TAIL);
    CHECK(standard_v_tail.layer_id == 3);
    CHECK(standard_v_tail.split_axis == 0);

    const auto kvarn_k_records = llama_kv_cache_component_from_name("cache_kvarn_k_records_l9");
    CHECK(kvarn_k_records.valid);
    CHECK(kvarn_k_records.role == LLAMA_KV_CACHE_COMPONENT_KVARN_K_RECORDS);
    CHECK(kvarn_k_records.layer_id == 9);
    CHECK(kvarn_k_records.split_axis == 1);

    const auto kvarn_v_stage = llama_kv_cache_component_from_name("cache_kvarn_v_stage_l4_s2");
    CHECK(kvarn_v_stage.valid);
    CHECK(kvarn_v_stage.role == LLAMA_KV_CACHE_COMPONENT_KVARN_V_STAGE);
    CHECK(kvarn_v_stage.layer_id == 4);
    CHECK(kvarn_v_stage.split_axis == 1);

    const auto kvarn_k_tail = llama_kv_cache_component_from_name("cache_kvarn_k_tail_l5");
    CHECK(kvarn_k_tail.valid);
    CHECK(kvarn_k_tail.role == LLAMA_KV_CACHE_COMPONENT_KVARN_K_TAIL);
    CHECK(kvarn_k_tail.layer_id == 5);
    CHECK(kvarn_k_tail.split_axis == 0);

    CHECK(!llama_kv_cache_component_from_name("cache_k_idx_l1").valid);
    CHECK(!llama_kv_cache_component_from_name("cache_kvarn_k_stage_bad").valid);
    CHECK(!llama_kv_cache_component_from_name("cache_k_l4294967296").valid);

    // Qwen3.6 has four KV heads. Split ratios are rounded only at complete
    // head boundaries and may legitimately leave a shard empty.
    CHECK((llama_tensor_split_counts(4, { 1.0f, 1.0f }, 1) ==
            std::vector<int64_t> { 2, 2 }));
    CHECK((llama_tensor_split_counts(4, { 3.0f, 1.0f }, 1) ==
            std::vector<int64_t> { 3, 1 }));
    const auto zero_head_shards = llama_tensor_split_counts(
            2, { 1.0f, 1.0f, 1.0f, 1.0f }, 1);
    CHECK(zero_head_shards.size() == 4);
    CHECK(std::count(zero_head_shards.begin(), zero_head_shards.end(), int64_t(0)) == 2);
    CHECK(std::accumulate(zero_head_shards.begin(), zero_head_shards.end(), int64_t(0)) == 2);
    CHECK((llama_tensor_split_counts(4, { 0.0f, 0.0f }, 1) ==
            std::vector<int64_t> { 2, 2 }));
    CHECK((llama_tensor_split_counts(10, { 1.0f, 1.0f, 1.0f }, 4) ==
            std::vector<int64_t> { 0, 4, 6 }));

    bool rejected = false;
    try {
        (void) llama_tensor_split_counts(4, { 1.0f, -1.0f }, 1);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    CHECK(rejected);
    rejected = false;
    try {
        (void) llama_tensor_split_counts(4, {
                1.0f, std::numeric_limits<float>::quiet_NaN() }, 1);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    CHECK(rejected);
    rejected = false;
    try {
        (void) llama_tensor_split_counts(4, {}, 1);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    CHECK(rejected);
    rejected = false;
    try {
        (void) llama_tensor_split_counts(4, { 1.0f }, 0);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    CHECK(rejected);
    rejected = false;
    try {
        (void) llama_tensor_split_counts(-1, { 1.0f }, 1);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    CHECK(rejected);

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    CHECK(cpu != nullptr);
    ggml_backend_dev_t logical_devices[] = { cpu, cpu };
    ggml_backend_dev_t meta = ggml_backend_meta_device(
            logical_devices, 2, mirrored_split, nullptr);
    CHECK(ggml_backend_dev_is_meta(meta));
    CHECK(ggml_backend_meta_device_count(meta) == 2);
    CHECK(ggml_backend_meta_device_get(meta, 0) == cpu);
    CHECK(ggml_backend_meta_device_get(meta, 1) == cpu);
    CHECK(!ggml_backend_dev_is_meta(cpu));

    ggml_init_params params = { 4096, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);
    ggml_tensor * leaf = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    CHECK(leaf->op == GGML_OP_NONE);
    CHECK(ggml_backend_dev_supports_op(meta, leaf));
    if (ggml_backend_dev_t cuda = ggml_backend_dev_by_name("CUDA0")) {
        ggml_backend_dev_t cuda_devices[] = { cuda, cuda };
        ggml_backend_dev_t cuda_meta = ggml_backend_meta_device(
                cuda_devices, 2, mirrored_split, nullptr);
        CHECK(ggml_backend_dev_supports_op(cuda_meta, leaf));
    }
    ggml_free(ctx);
    return 0;
}
