#pragma once

#include "common.cuh"
#include "fattn-kvarn-route-policy.h"

#include <cstdint>

constexpr uint32_t GGML_CUDA_FATTN_KVARN_ROUTE_STATS_ABI_VERSION = 1;

struct ggml_cuda_fattn_kvarn_route_stats {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t route_families;
    uint32_t reserved;
    uint64_t decode_split;
    uint64_t decode_vector;
    uint64_t generic_mma;
    uint64_t prompt_prefill;
    uint64_t portable_native;
    uint64_t amd_generic_mma;
    uint64_t amd_decode_split;
    uint64_t amd_decode_vector;
    uint64_t materialize_fallback;
    uint64_t split_reduce;
    uint64_t direct_entry;
    uint64_t compact_tail_entry;
};

enum ggml_cuda_fattn_kvarn_entry_path {
    GGML_CUDA_FATTN_KVARN_ENTRY_DIRECT,
    GGML_CUDA_FATTN_KVARN_ENTRY_COMPACT_TAIL,
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

ggml_cuda_fattn_kvarn_capabilities ggml_cuda_fattn_kvarn_device_capabilities(int device);

void ggml_cuda_fattn_kvarn_route_stats_reset();
void ggml_cuda_fattn_kvarn_route_stats_get(ggml_cuda_fattn_kvarn_route_stats * stats);
uint32_t ggml_cuda_fattn_kvarn_decode_max_q();

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

bool ggml_cuda_flash_attn_ext_kvarn_portable_supported(
        int device,
        const ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_kvarn_direct_tail_supported(
        int device,
        const ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_kvarn(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        ggml_cuda_fattn_kvarn_entry_path entry_path = GGML_CUDA_FATTN_KVARN_ENTRY_DIRECT);
