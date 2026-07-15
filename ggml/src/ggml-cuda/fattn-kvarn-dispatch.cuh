#pragma once

#include "common.cuh"

#include <cstdint>

struct ggml_cuda_fattn_kvarn_route_stats {
    uint64_t decode_split;
    uint64_t decode_vector;
    uint64_t generic_mma;
    uint64_t prompt_prefill;
};

struct ggml_cuda_kv_memory_transient_stats {
    uint64_t kvarn_descriptor_bytes;
    uint64_t kvarn_partial_output_bytes;
    uint64_t kvarn_partial_meta_bytes;
    uint64_t kvarn_total_bytes;
    uint64_t tail_body_meta_bytes;
    uint64_t tail_exact_meta_bytes;
    uint64_t tail_pack_bytes;
    uint64_t tail_body_output_bytes;
    uint64_t tail_exact_output_bytes;
    uint64_t tail_plan_input_bytes;
    uint64_t tail_total_bytes;
};

void ggml_cuda_fattn_kvarn_route_stats_reset();
void ggml_cuda_fattn_kvarn_route_stats_get(ggml_cuda_fattn_kvarn_route_stats * stats);

void ggml_cuda_kv_memory_transient_stats_reset();
void ggml_cuda_kv_memory_transient_stats_get(ggml_cuda_kv_memory_transient_stats * stats);
void ggml_cuda_kv_memory_transient_stats_record_kvarn(
        uint64_t descriptor_bytes,
        uint64_t partial_output_bytes,
        uint64_t partial_meta_bytes,
        uint64_t total_bytes);
void ggml_cuda_kv_memory_transient_stats_record_tail(
        uint64_t body_meta_bytes,
        uint64_t exact_meta_bytes,
        uint64_t pack_bytes,
        uint64_t body_output_bytes,
        uint64_t exact_output_bytes,
        uint64_t plan_input_bytes,
        uint64_t total_bytes);

bool ggml_cuda_flash_attn_ext_kvarn_uses_views(
        const ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_kvarn_supported(
        int device,
        const ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_kvarn(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);
