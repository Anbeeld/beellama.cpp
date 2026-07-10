#pragma once

#include "common.cuh"

void ggml_cuda_op_kvarn_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
