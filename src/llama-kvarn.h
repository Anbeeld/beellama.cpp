#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>

struct llama_kvarn_type_desc {
    llama_kvarn_type type;
    const char * name;
    int key_bits;
    int value_bits;
    int group;
};

struct llama_kvarn_tile_layout {
    size_t k_payload_off;
    size_t v_payload_off;
    size_t k_s_col_off;
    size_t k_zp_off;
    size_t k_s_row_off;
    size_t v_s_col_off;
    size_t v_s_row_off;
    size_t v_zp_off;

    size_t k_payload_bytes;
    size_t v_payload_bytes;
    size_t tile_bytes;
};

struct llama_kvarn_runtime_requirements {
    bool attention_supported;
    bool head_dims_supported;
    bool kv_offload;
    bool native_backend_supported;
    uint32_t n_seq_max;
    bool kv_unified;
};

size_t llama_kvarn_type_count();

const llama_kvarn_type_desc * llama_kvarn_type_desc_from_name(const char * name);
const llama_kvarn_type_desc * llama_kvarn_type_desc_from_type(llama_kvarn_type type);

llama_kvarn_params llama_kvarn_params_with_min_bits(llama_kvarn_params params, int min_bits);

llama_kvarn_tile_layout llama_kvarn_make_layout(int head_dim, int group, int key_bits, int value_bits);

int  llama_kvarn_head_slices(int head_dim);
bool llama_kvarn_head_dim_supported(int head_dim);

size_t  llama_kvarn_packed_bytes(int n_values, int bits);
void    llama_kvarn_pack_bits(const uint8_t * values, int n_values, int bits, uint8_t * dst);
uint8_t llama_kvarn_unpack_bits_value(const uint8_t * src, int index, int bits);

const char * llama_kvarn_validate_runtime(
        const llama_kvarn_params & params,
        const llama_kvarn_runtime_requirements & requirements);

bool llama_kvarn_can_remove_range(llama_pos pos_max, llama_pos p0, llama_pos p1, uint32_t group);

void llama_kvarn_hadamard_128(float * values);

void llama_kvarn_quantize_k_tile(
        const float * tile,
        int sinkhorn_iters,
        int bits,
        const llama_kvarn_tile_layout & layout,
        uint8_t * record);

void llama_kvarn_quantize_v_tile(
        const float * tile,
        int sinkhorn_iters,
        int bits,
        const llama_kvarn_tile_layout & layout,
        uint8_t * record);

void llama_kvarn_dequantize_k_tile(
        const uint8_t * record,
        int bits,
        const llama_kvarn_tile_layout & layout,
        float * tile);

void llama_kvarn_dequantize_v_tile(
        const uint8_t * record,
        int bits,
        const llama_kvarn_tile_layout & layout,
        float * tile);
