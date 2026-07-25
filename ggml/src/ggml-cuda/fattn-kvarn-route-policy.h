#pragma once

#include <cstdint>

constexpr int GGML_CUDA_FATTN_KVARN_DECODE_MAX_Q = 16;

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
    bool fast_decode_instances;
};

struct ggml_cuda_fattn_kvarn_capabilities {
    bool generic_mma;
    bool decode_split;
    bool decode_vector;
    bool portable_native;
    bool specialized_routes;
    uint32_t route_families;
};

inline ggml_cuda_fattn_kvarn_capabilities ggml_cuda_fattn_kvarn_select_capabilities(
        const ggml_cuda_fattn_kvarn_capability_input & input) {
    const bool physical_wave_supported =
        input.physical_wave_size == 32 || input.physical_wave_size == 64;

    ggml_cuda_fattn_kvarn_capabilities result = {};
    result.portable_native = input.fast_decode_instances;
    if (input.backend == GGML_CUDA_FATTN_KVARN_BACKEND_CUDA) {
        result.generic_mma = input.matrix_mma && input.fast_decode_instances;
        result.decode_split = input.matrix_mma && input.fast_decode_instances;
        result.decode_vector = input.matrix_mma && input.fast_decode_instances;
    } else if (input.backend == GGML_CUDA_FATTN_KVARN_BACKEND_HIP) {
        result.generic_mma =
            input.matrix_mma && input.fast_decode_instances && physical_wave_supported;
        result.decode_split = result.generic_mma && input.fast_decode_instances;
        // The SWA vector kernel is still CUDA-warp tuned. HIP uses split decode
        // or generic MMA until a physical-wave vector route proves worthwhile.
        result.decode_vector = false;
    }
    // MUSA intentionally remains portable-native. Its compiler consumes these
    // shared sources, but it does not provide the AMD/NVIDIA MMA contracts used
    // by the KVarN matrix loaders.

    result.specialized_routes =
        result.generic_mma || result.decode_split || result.decode_vector;
    result.route_families =
        (result.portable_native ? GGML_CUDA_FATTN_KVARN_FAMILY_PORTABLE_NATIVE : 0u) |
        (result.generic_mma ? GGML_CUDA_FATTN_KVARN_FAMILY_GENERIC_MMA : 0u) |
        (result.decode_split ? GGML_CUDA_FATTN_KVARN_FAMILY_DECODE_SPLIT : 0u) |
        (result.decode_vector ? GGML_CUDA_FATTN_KVARN_FAMILY_DECODE_VECTOR : 0u);
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
    return wide_kernel_supported && n_q > 8 && n_q <= GGML_CUDA_FATTN_KVARN_DECODE_MAX_Q && gqa > 4;
}
