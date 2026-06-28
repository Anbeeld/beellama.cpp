#pragma once

template <int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_turbo_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, ncols1, ncols2, tK, tV)                       \
    template void ggml_cuda_flash_attn_ext_mma_turbo_case                                \
    <DKQ, DV, ncols1, ncols2, tK, tV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_NCOLS2(DKQ, DV, ncols, tK) \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/1, 1, tK, tK);      \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/2, 2, tK, tK);      \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/4, 4, tK, tK);      \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, (ncols)/8, 8, tK, tK);      \

#define DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(DKQ, DV, ncols)        \
    DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_NCOLS2(DKQ, DV, ncols, GGML_TYPE_TURBO4_0) \
    DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_NCOLS2(DKQ, DV, ncols, GGML_TYPE_TURBO3_0) \
    DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_NCOLS2(DKQ, DV, ncols, GGML_TYPE_TURBO2_0) \

DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(128, 128,  8)
DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(128, 128, 16)
DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(128, 128, 32)
DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(128, 128, 64)
DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(256, 256,  8)
DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(256, 256, 16)
DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(256, 256, 32)
DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES(256, 256, 64)

#undef DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_TYPES
#undef DECL_FATTN_MMA_TURBO_STRAIGHT_CASE_ALL_NCOLS2
#undef DECL_FATTN_MMA_TURBO_CASE
