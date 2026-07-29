#include "llama-kv-cache-placement.h"

#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

bool parse_cache_component_suffix(
        const std::string & name,
        const char * prefix,
        bool allow_stream,
        uint32_t & layer_id) {
    const size_t prefix_size = std::strlen(prefix);
    if (name.compare(0, prefix_size, prefix) != 0 || prefix_size == name.size()) {
        return false;
    }
    size_t pos = prefix_size;
    uint64_t parsed = 0;
    const size_t digits_begin = pos;
    while (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
        parsed = parsed*10 + uint64_t(name[pos] - '0');
        if (parsed > UINT32_MAX) {
            return false;
        }
        ++pos;
    }
    if (pos == digits_begin) {
        return false;
    }
    if (pos != name.size()) {
        if (!allow_stream || name.compare(pos, 2, "_s") != 0) {
            return false;
        }
        pos += 2;
        const size_t stream_begin = pos;
        while (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
            ++pos;
        }
        if (pos == stream_begin || pos != name.size()) {
            return false;
        }
    }
    layer_id = uint32_t(parsed);
    return true;
}

} // namespace

llama_kv_cache_component llama_kv_cache_component_from_name(const std::string & name) {
    struct component_pattern {
        const char * prefix;
        llama_kv_cache_component_role role;
        int split_axis;
        bool allow_stream;
    };
    static const component_pattern patterns[] = {
        { "cache_kvarn_k_records_l", LLAMA_KV_CACHE_COMPONENT_KVARN_K_RECORDS, 1, true  },
        { "cache_kvarn_v_records_l", LLAMA_KV_CACHE_COMPONENT_KVARN_V_RECORDS, 1, true  },
        { "cache_kvarn_k_stage_l",   LLAMA_KV_CACHE_COMPONENT_KVARN_K_STAGE,   1, true  },
        { "cache_kvarn_v_stage_l",   LLAMA_KV_CACHE_COMPONENT_KVARN_V_STAGE,   1, true  },
        { "cache_kvarn_k_tail_l",    LLAMA_KV_CACHE_COMPONENT_KVARN_K_TAIL,    0, false },
        { "cache_kvarn_v_tail_l",    LLAMA_KV_CACHE_COMPONENT_KVARN_V_TAIL,    0, false },
        { "cache_k_tail_l",          LLAMA_KV_CACHE_COMPONENT_STANDARD_K_TAIL, 0, false },
        { "cache_v_tail_l",          LLAMA_KV_CACHE_COMPONENT_STANDARD_V_TAIL, 0, false },
        { "cache_k_l",               LLAMA_KV_CACHE_COMPONENT_STANDARD_K,      0, false },
        { "cache_v_l",               LLAMA_KV_CACHE_COMPONENT_STANDARD_V,      0, false },
    };
    for (const auto & pattern : patterns) {
        uint32_t layer_id = 0;
        if (parse_cache_component_suffix(name, pattern.prefix, pattern.allow_stream, layer_id)) {
            return { true, pattern.role, layer_id, pattern.split_axis };
        }
    }
    return { false, LLAMA_KV_CACHE_COMPONENT_UNKNOWN, 0, -1 };
}

std::vector<int64_t> llama_tensor_split_counts(
        int64_t n_elements,
        const std::vector<float> & weights,
        int64_t granularity) {
    if (n_elements < 0) {
        throw std::invalid_argument("tensor split element count must be non-negative");
    }
    if (granularity <= 0) {
        throw std::invalid_argument("tensor split granularity must be positive");
    }
    if (weights.empty()) {
        throw std::invalid_argument("tensor split requires at least one device");
    }
    double total = 0.0;
    for (float weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0f) {
            throw std::invalid_argument("tensor split weights must be finite and non-negative");
        }
        total += weight;
    }
    std::vector<int64_t> result(weights.size(), 0);
    int64_t low = 0;
    double cumulative = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i];
        const bool is_last = i + 1 == weights.size();
        int64_t high = is_last ? n_elements :
                total == 0.0 ?
                    int64_t(static_cast<long double>(n_elements)*(i + 1)/weights.size()) :
                    int64_t(static_cast<long double>(n_elements)*cumulative/total);
        if (!is_last) {
            high -= high % granularity;
        }
        result[i] = high - low;
        low = high;
    }
    return result;
}
