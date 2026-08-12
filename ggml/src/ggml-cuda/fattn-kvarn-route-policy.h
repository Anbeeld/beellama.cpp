#pragma once

#include <cstdint>
#include <limits>

#define GGML_CUDA_FATTN_KVARN_OPERATION_POLICY 1

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
    GGML_CUDA_FATTN_KVARN_ROUTE_PORTABLE_NATIVE,
    GGML_CUDA_FATTN_KVARN_ROUTE_UNAVAILABLE,
};

enum ggml_cuda_fattn_kvarn_amd_mma_arch {
    GGML_CUDA_FATTN_KVARN_AMD_NONE,
    GGML_CUDA_FATTN_KVARN_AMD_RDNA_WMMA,
    GGML_CUDA_FATTN_KVARN_AMD_CDNA_MFMA,
};

enum ggml_cuda_fattn_kvarn_mma_eligibility {
    GGML_CUDA_FATTN_KVARN_MMA_ELIGIBLE,
    GGML_CUDA_FATTN_KVARN_MMA_NO_FAMILY,
    GGML_CUDA_FATTN_KVARN_MMA_INVALID_COLUMNS,
    GGML_CUDA_FATTN_KVARN_MMA_TILE_TOO_SMALL,
    GGML_CUDA_FATTN_KVARN_MMA_RDNA_SINGLE_GQA_COLUMN,
    GGML_CUDA_FATTN_KVARN_MMA_HEAD_DIM_UNSUPPORTED,
};

struct ggml_cuda_fattn_kvarn_amd_mma_input {
    ggml_cuda_fattn_kvarn_amd_mma_arch arch;
    int head_dim;
    int ncols1;
    int ncols2;
};

// Host mirror of the final device-side invariants in flash_attn_ext_f16.
// Device guards remain in place so a future caller cannot turn an invalid
// AMD template into executable code by bypassing this route policy.
inline ggml_cuda_fattn_kvarn_mma_eligibility ggml_cuda_fattn_kvarn_amd_mma_eligibility(
        const ggml_cuda_fattn_kvarn_amd_mma_input & input) {
    if (input.arch == GGML_CUDA_FATTN_KVARN_AMD_NONE) {
        return GGML_CUDA_FATTN_KVARN_MMA_NO_FAMILY;
    }
    if (input.ncols1 <= 0 || input.ncols2 <= 0) {
        return GGML_CUDA_FATTN_KVARN_MMA_INVALID_COLUMNS;
    }
    if (input.head_dim <= 0 ||
            (input.arch == GGML_CUDA_FATTN_KVARN_AMD_RDNA_WMMA && input.head_dim > 128) ||
            (input.arch == GGML_CUDA_FATTN_KVARN_AMD_CDNA_MFMA && input.head_dim > 256)) {
        return GGML_CUDA_FATTN_KVARN_MMA_HEAD_DIM_UNSUPPORTED;
    }
    if (input.ncols1 * input.ncols2 < 16) {
        return GGML_CUDA_FATTN_KVARN_MMA_TILE_TOO_SMALL;
    }
    if (input.arch == GGML_CUDA_FATTN_KVARN_AMD_RDNA_WMMA && input.ncols2 == 1) {
        return GGML_CUDA_FATTN_KVARN_MMA_RDNA_SINGLE_GQA_COLUMN;
    }
    return GGML_CUDA_FATTN_KVARN_MMA_ELIGIBLE;
}

inline ggml_cuda_fattn_kvarn_route ggml_cuda_fattn_kvarn_select_fallback_route(
        bool prompt_prefill,
        bool generic_shape_eligible,
        bool portable_eligible) {
    if (generic_shape_eligible) {
        return prompt_prefill ? GGML_CUDA_FATTN_KVARN_ROUTE_PROMPT_PREFILL :
            GGML_CUDA_FATTN_KVARN_ROUTE_GENERIC_MMA;
    }
    return portable_eligible ? GGML_CUDA_FATTN_KVARN_ROUTE_PORTABLE_NATIVE :
        GGML_CUDA_FATTN_KVARN_ROUTE_UNAVAILABLE;
}

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
        // Split decode uses NVIDIA ldmatrix plus m16n8 MMA fragments, and the
        // SWA vector kernel is CUDA-warp tuned. HIP uses shape-gated generic
        // WMMA/MFMA or portable direct-record attention instead.
        result.decode_split = false;
        result.decode_vector = false;
    }
    // MUSA intentionally remains portable-native. Its compiler consumes these
    // shared sources, but it does not provide the AMD/NVIDIA MMA contracts used
    // by the KVarN matrix loaders.

    result.specialized_routes =
        result.generic_mma || result.decode_split || result.decode_vector;
    // On CUDA, generic MMA is valid for every graph-admitted original-V
    // shape. HIP's family bit is only an inventory claim: RDNA/CDNA template
    // support is operation-specific, so HIP stays on the unbounded rotated
    // portable domain until the bounded original-V window matrix is proven on
    // both physical wave sizes.
    result.original_v_domain =
        input.backend == GGML_CUDA_FATTN_KVARN_BACKEND_CUDA && result.generic_mma;
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

inline bool ggml_cuda_fattn_kvarn_body_shape_supported(
        const ggml_cuda_fattn_kvarn_capabilities & capabilities,
        int64_t d_k,
        int64_t d_v) {
    uint32_t head_dim = 0;
    if (d_k != d_v) {
        return false;
    }
    switch (d_k) {
        case 128: head_dim = GGML_CUDA_FATTN_KVARN_HEAD_DIM_128; break;
        case 256: head_dim = GGML_CUDA_FATTN_KVARN_HEAD_DIM_256; break;
        case 512: head_dim = GGML_CUDA_FATTN_KVARN_HEAD_DIM_512; break;
        default: return false;
    }
    return capabilities.portable_native &&
        (capabilities.supported_head_dims & head_dim) != 0;
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
