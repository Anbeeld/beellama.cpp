#pragma once

#include "common.cuh"

constexpr uint32_t GGML_CUDA_KVARN_STORE_ROUTE_STATS_ABI_VERSION = 2;

struct ggml_cuda_kvarn_store_route_stats {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t headwide_workspace;
    uint64_t headwide_monolithic;
    uint64_t single_slice_workspace;
    uint64_t direct_store;
    uint64_t high_shared_fallback;
    uint64_t low_shared_store;
    uint64_t sealer_128;
    uint64_t sealer_256;
    uint64_t sealer_candidates;
};

void ggml_cuda_op_kvarn_store(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_kvarn_materialize(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

size_t ggml_cuda_kvarn_required_shared_bytes();
size_t ggml_cuda_kvarn_low_shared_bytes();

void ggml_cuda_kvarn_profile_dump();
void ggml_cuda_kvarn_store_route_stats_reset();
void ggml_cuda_kvarn_store_route_stats_get(ggml_cuda_kvarn_store_route_stats * stats);
