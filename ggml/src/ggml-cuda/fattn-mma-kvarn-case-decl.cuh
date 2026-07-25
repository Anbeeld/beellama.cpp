#pragma once

#include "fattn-mma-kvarn.cuh"

void ggml_cuda_fattn_kvarn_init_descs(
        const ggml_cuda_fattn_kvarn_plan & plan,
        ggml_cuda_fattn_kvarn_desc * k_desc,
        ggml_cuda_fattn_kvarn_desc * v_desc,
        int k_original_domain,
        int v_original_domain,
        cudaStream_t stream);

static inline enum ggml_flash_attn_ext_kvarn_domain ggml_cuda_fattn_kvarn_domain(const ggml_tensor * dst) {
    return (enum ggml_flash_attn_ext_kvarn_domain) ggml_get_op_params_i32(
            dst, GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN);
}

static inline bool ggml_cuda_fattn_kvarn_k_original_domain(const ggml_tensor * dst) {
    switch (ggml_cuda_fattn_kvarn_domain(dst)) {
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL:
            return true;
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED:
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V:
            return false;
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO:
            return dst->src[0]->ne[1] > 1;
    }

    return dst->src[0]->ne[1] > 1;
}

static inline bool ggml_cuda_fattn_kvarn_v_original_domain(const ggml_tensor * dst) {
    switch (ggml_cuda_fattn_kvarn_domain(dst)) {
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL:
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V:
            return true;
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED:
            return false;
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO:
            return dst->src[0]->ne[1] > 1;
    }

    return dst->src[0]->ne[1] > 1;
}

static inline bool ggml_cuda_fattn_kvarn_rotated_decode_domain(const ggml_tensor * dst) {
    switch (ggml_cuda_fattn_kvarn_domain(dst)) {
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL:
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V:
            return false;
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED:
            return true;
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO:
            return dst->src[0]->ne[1] == 1;
    }

    return dst->src[0]->ne[1] == 1;
}

static inline const char * ggml_cuda_fattn_kvarn_domain_name(const ggml_tensor * dst) {
    switch (ggml_cuda_fattn_kvarn_domain(dst)) {
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL:
            return "original_prefill";
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED:
            return "rotated";
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V:
            return "rotated_k_original_v";
        case GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO:
            return "auto";
    }

    return "unknown";
}

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_kvarn_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define DECL_FATTN_MMA_KVARN_CASE_EXTERN(DKQ, DV, ncols1, ncols2)                  \
    extern template void ggml_cuda_flash_attn_ext_mma_kvarn_case                  \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(DKQ, DV, ncols)     \
    DECL_FATTN_MMA_KVARN_CASE_EXTERN(DKQ, DV, (ncols)/ 1, 1);    \
    DECL_FATTN_MMA_KVARN_CASE_EXTERN(DKQ, DV, (ncols)/ 2, 2);    \
    DECL_FATTN_MMA_KVARN_CASE_EXTERN(DKQ, DV, (ncols)/ 4, 4);    \
    DECL_FATTN_MMA_KVARN_CASE_EXTERN(DKQ, DV, (ncols)/ 8, 8);    \

DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(128, 128,  8)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(128, 128, 16)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(128, 128, 32)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(128, 128, 64)

DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(256, 256,  8)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(256, 256, 16)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(256, 256, 32)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(256, 256, 64)

DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(512, 512,  8)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(512, 512, 16)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(512, 512, 32)
DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2(512, 512, 64)

#if !defined(GGML_USE_MUSA)
DECL_FATTN_MMA_KVARN_CASE_EXTERN(128, 128, 16, 8);
DECL_FATTN_MMA_KVARN_CASE_EXTERN(256, 256, 16, 8);

template <int DKQ, int DV, int ncols1, int ncols2>
bool ggml_cuda_fattn_kvarn_wide_mma_supported(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst);

extern template bool ggml_cuda_fattn_kvarn_wide_mma_supported<128, 128, 16, 8>(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst);
extern template bool ggml_cuda_fattn_kvarn_wide_mma_supported<256, 256, 16, 8>(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst);
#endif // !defined(GGML_USE_MUSA)

#undef DECL_FATTN_MMA_KVARN_CASE_ALL_NCOLS2
#undef DECL_FATTN_MMA_KVARN_CASE_EXTERN
