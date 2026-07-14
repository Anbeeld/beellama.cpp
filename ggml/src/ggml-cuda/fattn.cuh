#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_tail_supported(
        ggml_type body_k, ggml_type body_v, ggml_type tail_k, ggml_type tail_v, int64_t d_k, int64_t d_v);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

const char * ggml_cuda_fa_build_policy();

bool ggml_cuda_fa_pair_compiled(ggml_type type_K, ggml_type type_V);
