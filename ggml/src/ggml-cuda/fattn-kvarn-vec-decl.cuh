#pragma once

#include "fattn-mma-kvarn-decode-decl.cuh"

static inline int ggml_cuda_fattn_kvarn_vec_tokens_per_split() {
    const char * value = getenv("GGML_KVARN_VEC_TOKENS");
    if (value != nullptr) {
        const int tokens = atoi(value);
        if (tokens == 8 || tokens == 16 || tokens == 32) {
            return tokens;
        }
    }
    return 16;
}

// Bounded GQA sharing for the proven D256 SWA low-parallelism decode path.
template<int D>
constexpr int ggml_cuda_fattn_kvarn_vec_max_gqa() {
    static_assert(D == 256, "KVarN vec production route is currently proven only for D256");
    return 2;
}

template<int D, int K_BITS, int V_BITS>
void ggml_cuda_fattn_kvarn_vec_launch(const ggml_cuda_fattn_kvarn_decode_args & args);

#define DECL_FATTN_KVARN_VEC_CASE(D, K_BITS, V_BITS) \
    template void ggml_cuda_fattn_kvarn_vec_launch<D, K_BITS, V_BITS>( \
        const ggml_cuda_fattn_kvarn_decode_args & args)
