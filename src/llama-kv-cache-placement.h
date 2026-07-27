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

// Return per-device head counts using the same cumulative-ratio convention as
// the upstream meta device. Empty shards are valid when devices outnumber heads.
std::vector<uint32_t> llama_kv_cache_head_split(
        uint32_t n_heads,
        const std::vector<float> & weights);
