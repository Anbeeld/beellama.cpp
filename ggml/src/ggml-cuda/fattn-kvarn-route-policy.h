#pragma once

constexpr int GGML_CUDA_FATTN_KVARN_DECODE_MAX_Q = 16;

enum ggml_cuda_fattn_kvarn_route {
    GGML_CUDA_FATTN_KVARN_ROUTE_DECODE_SPLIT,
    GGML_CUDA_FATTN_KVARN_ROUTE_DECODE_VECTOR,
    GGML_CUDA_FATTN_KVARN_ROUTE_GENERIC_MMA,
    GGML_CUDA_FATTN_KVARN_ROUTE_PROMPT_PREFILL,
};

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
