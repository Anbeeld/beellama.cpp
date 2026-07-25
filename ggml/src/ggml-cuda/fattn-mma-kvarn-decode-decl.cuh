#pragma once

#include "common.cuh"
#include "fattn-mma-kvarn.cuh"

struct ggml_cuda_fattn_kvarn_decode_geometry {
    bool use_split;
    int  split_tokens;
    int  nwarps;
    int  gqa_per_block;
    int  n_splits;
    int  n_gqa_blocks;
    int  max_blocks_per_sm;
    int  wave_efficiency_percent;
    int  n_waves;
};

struct ggml_cuda_fattn_kvarn_decode_args {
    const char * Q;
    const ggml_cuda_fattn_kvarn_desc * k_descs;
    const ggml_cuda_fattn_kvarn_desc * v_descs;
    const char * mask;
    float * partial;
    float2 * partial_meta;
    float * dst;
    float2 * dst_meta;
    float scale;
    float logit_softcap;
    int64_t nb01;
    int64_t nb02;
    int64_t nb03;
    int64_t nb30;
    int64_t nb31;
    int64_t nb33;
    int ne33;
    int n_kv;
    int n_q;
    int n_q_heads;
    int n_kv_heads;
    int n_stream;
    int gqa_ratio;
    int gqa_per_block;
    int n_gqa_blocks;
    int n_splits;
    int split_tokens;
    int nwarps;
    int wave_size;
    cudaStream_t stream;
};

template<int D, int K_BITS, int V_BITS>
ggml_cuda_fattn_kvarn_decode_geometry ggml_cuda_fattn_kvarn_decode_select(
        int device,
        int n_kv,
        int n_q,
        int n_q_heads,
        int n_kv_heads,
        int n_stream);

template<int D, int K_BITS, int V_BITS>
void ggml_cuda_fattn_kvarn_decode_launch(const ggml_cuda_fattn_kvarn_decode_args & args);

#define DECL_FATTN_KVARN_DECODE_CASE(D, K_BITS, V_BITS) \
    template ggml_cuda_fattn_kvarn_decode_geometry ggml_cuda_fattn_kvarn_decode_select<D, K_BITS, V_BITS>( \
        int device, int n_kv, int n_q, int n_q_heads, int n_kv_heads, int n_stream); \
    template void ggml_cuda_fattn_kvarn_decode_launch<D, K_BITS, V_BITS>( \
        const ggml_cuda_fattn_kvarn_decode_args & args)

#define EXTERN_DECL_FATTN_KVARN_DECODE_CASE(D, K_BITS, V_BITS) \
    extern template ggml_cuda_fattn_kvarn_decode_geometry ggml_cuda_fattn_kvarn_decode_select<D, K_BITS, V_BITS>( \
        int device, int n_kv, int n_q, int n_q_heads, int n_kv_heads, int n_stream); \
    extern template void ggml_cuda_fattn_kvarn_decode_launch<D, K_BITS, V_BITS>( \
        const ggml_cuda_fattn_kvarn_decode_args & args)

#define DECL_FATTN_KVARN_DECODE_ALL_V(D, K_BITS) \
    EXTERN_DECL_FATTN_KVARN_DECODE_CASE(D, K_BITS, 2); \
    EXTERN_DECL_FATTN_KVARN_DECODE_CASE(D, K_BITS, 3); \
    EXTERN_DECL_FATTN_KVARN_DECODE_CASE(D, K_BITS, 4); \
    EXTERN_DECL_FATTN_KVARN_DECODE_CASE(D, K_BITS, 5); \
    EXTERN_DECL_FATTN_KVARN_DECODE_CASE(D, K_BITS, 6); \
    EXTERN_DECL_FATTN_KVARN_DECODE_CASE(D, K_BITS, 8)

#define DECL_FATTN_KVARN_DECODE_ALL_KV(D) \
    DECL_FATTN_KVARN_DECODE_ALL_V(D, 2); \
    DECL_FATTN_KVARN_DECODE_ALL_V(D, 3); \
    DECL_FATTN_KVARN_DECODE_ALL_V(D, 4); \
    DECL_FATTN_KVARN_DECODE_ALL_V(D, 5); \
    DECL_FATTN_KVARN_DECODE_ALL_V(D, 6); \
    DECL_FATTN_KVARN_DECODE_ALL_V(D, 8)

DECL_FATTN_KVARN_DECODE_ALL_KV(128);
DECL_FATTN_KVARN_DECODE_ALL_KV(256);
DECL_FATTN_KVARN_DECODE_ALL_KV(512);

#undef DECL_FATTN_KVARN_DECODE_ALL_KV
#undef DECL_FATTN_KVARN_DECODE_ALL_V
#undef EXTERN_DECL_FATTN_KVARN_DECODE_CASE
