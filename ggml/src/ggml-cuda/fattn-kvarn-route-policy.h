#pragma once

#include <cstdint>
#include <limits>

constexpr int GGML_CUDA_FATTN_KVARN_SPECIALIZED_DECODE_MAX_Q = 16;
constexpr int GGML_CUDA_FATTN_KVARN_PORTABLE_THREADS = 128;
constexpr uint32_t GGML_CUDA_FATTN_KVARN_PORTABLE_MAX_Q =
    std::numeric_limits<uint32_t>::max();

enum ggml_cuda_fattn_kvarn_head_dim : uint32_t {
    GGML_CUDA_FATTN_KVARN_HEAD_DIM_128 = 1u << 0,
    GGML_CUDA_FATTN_KVARN_HEAD_DIM_256 = 1u << 1,
    GGML_CUDA_FATTN_KVARN_HEAD_DIM_512 = 1u << 2,
};

enum ggml_cuda_fattn_kvarn_route {
    GGML_CUDA_FATTN_KVARN_ROUTE_DECODE_SPLIT,
    GGML_CUDA_FATTN_KVARN_ROUTE_DECODE_VECTOR,
    GGML_CUDA_FATTN_KVARN_ROUTE_GENERIC_MMA,
    GGML_CUDA_FATTN_KVARN_ROUTE_PROMPT_PREFILL,
};

enum ggml_cuda_fattn_kvarn_backend {
    GGML_CUDA_FATTN_KVARN_BACKEND_CUDA,
    GGML_CUDA_FATTN_KVARN_BACKEND_HIP,
    GGML_CUDA_FATTN_KVARN_BACKEND_MUSA,
};

enum ggml_cuda_fattn_kvarn_route_family : uint32_t {
    GGML_CUDA_FATTN_KVARN_FAMILY_PORTABLE_NATIVE = 1u << 0,
    GGML_CUDA_FATTN_KVARN_FAMILY_GENERIC_MMA     = 1u << 1,
    GGML_CUDA_FATTN_KVARN_FAMILY_DECODE_SPLIT    = 1u << 2,
    GGML_CUDA_FATTN_KVARN_FAMILY_DECODE_VECTOR   = 1u << 3,
};

struct ggml_cuda_fattn_kvarn_capability_input {
    ggml_cuda_fattn_kvarn_backend backend;
    int  physical_wave_size;
    bool matrix_mma;
    bool kvarn_instances;
    int  max_threads_per_block;
    uint64_t shared_memory_per_block;
    uint64_t minimum_dynamic_shared_bytes;
};

struct ggml_cuda_fattn_kvarn_capabilities {
    bool store_materialize;
    bool generic_mma;
    bool decode_split;
    bool decode_vector;
    bool portable_native;
    bool portable_tail_f16;
    bool portable_tail_bf16;
    bool specialized_routes;
    bool original_v_domain;
    uint32_t route_families;
    uint32_t rotated_query_max_portable;
    uint32_t rotated_query_max_specialized;
    uint32_t supported_head_dims;
    uint64_t minimum_dynamic_shared_bytes;
    int physical_wave_size;
};

inline ggml_cuda_fattn_kvarn_capabilities ggml_cuda_fattn_kvarn_select_capabilities(
        const ggml_cuda_fattn_kvarn_capability_input & input) {
    const bool physical_wave_supported =
        input.physical_wave_size == 32 || input.physical_wave_size == 64;
    const bool portable_wave_supported =
        input.backend == GGML_CUDA_FATTN_KVARN_BACKEND_CUDA ?
            input.physical_wave_size == 32 : physical_wave_supported;
    const bool portable_hardware =
        portable_wave_supported &&
        input.max_threads_per_block >= GGML_CUDA_FATTN_KVARN_PORTABLE_THREADS &&
        input.shared_memory_per_block >= input.minimum_dynamic_shared_bytes;

    ggml_cuda_fattn_kvarn_capabilities result = {};
    result.minimum_dynamic_shared_bytes = input.minimum_dynamic_shared_bytes;
    result.physical_wave_size = input.physical_wave_size;
    result.store_materialize = input.kvarn_instances && portable_hardware;
    result.portable_native = result.store_materialize;
    result.portable_tail_f16 = result.portable_native;
    result.portable_tail_bf16 = result.portable_native;
    if (input.backend == GGML_CUDA_FATTN_KVARN_BACKEND_CUDA) {
        result.generic_mma = input.matrix_mma && result.store_materialize;
        result.decode_split = result.generic_mma;
        result.decode_vector = result.generic_mma;
    } else if (input.backend == GGML_CUDA_FATTN_KVARN_BACKEND_HIP) {
        result.generic_mma =
            input.matrix_mma && result.store_materialize && physical_wave_supported;
        result.decode_split = result.generic_mma;
        // The SWA vector kernel is still CUDA-warp tuned. HIP uses split decode
        // or generic MMA until a physical-wave vector route proves worthwhile.
        result.decode_vector = false;
    }
    // MUSA intentionally remains portable-native. Its compiler consumes these
    // shared sources, but it does not provide the AMD/NVIDIA MMA contracts used
    // by the KVarN matrix loaders.

    result.specialized_routes =
        result.generic_mma || result.decode_split || result.decode_vector;
    result.original_v_domain = result.generic_mma;
    result.route_families =
        (result.portable_native ? GGML_CUDA_FATTN_KVARN_FAMILY_PORTABLE_NATIVE : 0u) |
        (result.generic_mma ? GGML_CUDA_FATTN_KVARN_FAMILY_GENERIC_MMA : 0u) |
        (result.decode_split ? GGML_CUDA_FATTN_KVARN_FAMILY_DECODE_SPLIT : 0u) |
        (result.decode_vector ? GGML_CUDA_FATTN_KVARN_FAMILY_DECODE_VECTOR : 0u);
    result.rotated_query_max_portable =
        result.portable_native ? GGML_CUDA_FATTN_KVARN_PORTABLE_MAX_Q : 0u;
    result.rotated_query_max_specialized =
        result.specialized_routes ? GGML_CUDA_FATTN_KVARN_SPECIALIZED_DECODE_MAX_Q : 0u;
    result.supported_head_dims = result.portable_native || result.specialized_routes ?
        GGML_CUDA_FATTN_KVARN_HEAD_DIM_128 |
        GGML_CUDA_FATTN_KVARN_HEAD_DIM_256 |
        GGML_CUDA_FATTN_KVARN_HEAD_DIM_512 : 0u;
    return result;
}

struct ggml_cuda_fattn_kvarn_route_input {
    int  head_dim;
    int  n_q;
    int  gqa;
    int  k_bits;
    int  v_bits;
    bool swa;
    bool body_meta_requested;
    bool vector_eligible;
    bool split_eligible;
    bool prompt_prefill;
};

// Optional softmax metadata is an output contract, not a route constraint.
// Eligibility is computed by the shape/domain-specific dispatch helpers.
inline ggml_cuda_fattn_kvarn_route ggml_cuda_fattn_kvarn_select_route(
        const ggml_cuda_fattn_kvarn_route_input & input) {
    if (input.prompt_prefill) {
        return GGML_CUDA_FATTN_KVARN_ROUTE_PROMPT_PREFILL;
    }
    if (input.vector_eligible) {
        return GGML_CUDA_FATTN_KVARN_ROUTE_DECODE_VECTOR;
    }
    // Split decode parallelizes one query over the KV sequence. Reusing it for
    // speculative verification repeats K/V decoding for every query and grows
    // its partial output with n_q * n_splits. The native MMA path instead tiles
    // the short query batch and reuses each decoded K/V tile across those rows.
    if (input.n_q == 1 && input.split_eligible) {
        return GGML_CUDA_FATTN_KVARN_ROUTE_DECODE_SPLIT;
    }
    return GGML_CUDA_FATTN_KVARN_ROUTE_GENERIC_MMA;
}

// The regular MMA matrix tops out at 64 query/head columns. A 16-token
// speculative verification block with GQA > 4 therefore reconstructs each
// compressed K/V tile more than once. Use the 128-column fused case only when
// it removes that duplicate work and the backend has confirmed that the
// concrete kernel fits and can occupy the device.
inline bool ggml_cuda_fattn_kvarn_use_wide_mma(
    int n_q,
    int gqa,
    bool wide_kernel_supported) {
    return wide_kernel_supported && n_q > 8 &&
        n_q <= GGML_CUDA_FATTN_KVARN_SPECIALIZED_DECODE_MAX_Q && gqa > 4;
}
