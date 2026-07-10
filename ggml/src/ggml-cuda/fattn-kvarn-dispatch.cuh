#pragma once

#include "common.cuh"

bool ggml_cuda_flash_attn_ext_kvarn_uses_views(
        const ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_kvarn_supported(
        int device,
        const ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_kvarn(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);
