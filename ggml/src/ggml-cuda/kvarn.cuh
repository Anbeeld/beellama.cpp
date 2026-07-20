#include "common.cuh"

void ggml_cuda_op_kvarn_store(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_kvarn_materialize(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

size_t ggml_cuda_kvarn_required_shared_bytes();
size_t ggml_cuda_kvarn_low_shared_bytes();

void ggml_cuda_kvarn_profile_dump();
