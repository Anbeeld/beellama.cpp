#pragma once

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
    if (input.split_eligible) {
        return GGML_CUDA_FATTN_KVARN_ROUTE_DECODE_SPLIT;
    }
    return GGML_CUDA_FATTN_KVARN_ROUTE_GENERIC_MMA;
}
