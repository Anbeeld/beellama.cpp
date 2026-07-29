#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum llama_kv_cache_component_role {
    LLAMA_KV_CACHE_COMPONENT_UNKNOWN,
    LLAMA_KV_CACHE_COMPONENT_STANDARD_K,
    LLAMA_KV_CACHE_COMPONENT_STANDARD_V,
    LLAMA_KV_CACHE_COMPONENT_STANDARD_K_TAIL,
    LLAMA_KV_CACHE_COMPONENT_STANDARD_V_TAIL,
    LLAMA_KV_CACHE_COMPONENT_KVARN_K_RECORDS,
    LLAMA_KV_CACHE_COMPONENT_KVARN_V_RECORDS,
    LLAMA_KV_CACHE_COMPONENT_KVARN_K_STAGE,
    LLAMA_KV_CACHE_COMPONENT_KVARN_V_STAGE,
    LLAMA_KV_CACHE_COMPONENT_KVARN_K_TAIL,
    LLAMA_KV_CACHE_COMPONENT_KVARN_V_TAIL,
};

// Typed adapter between cache-owned component roles and upstream's tensor-name
// split callback. Persistent payload is split along complete KV heads: standard
// rows and exact tails use axis 0, while KVarN records/stages use their explicit
// sliced-head axis 1.
struct llama_kv_cache_component {
    bool valid;
    llama_kv_cache_component_role role;
    uint32_t layer_id;
    int split_axis;
};

llama_kv_cache_component llama_kv_cache_component_from_name(const std::string & name);

// Return per-device element counts using the cumulative-ratio convention used
// by tensor-parallel placement. Non-final boundaries are rounded down to the
// requested granularity; empty shards and a shorter final remainder are valid.
// Throws std::invalid_argument for invalid dimensions, granularity, or weights.
std::vector<int64_t> llama_tensor_split_counts(
        int64_t n_elements,
        const std::vector<float> & weights,
        int64_t granularity);
