#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>

constexpr uint32_t KVAR_N_GROUP = 128;

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
    bool backend_ops_supported;
    uint32_t n_seq_max;
    bool kv_unified;
};

enum llama_kvarn_iswa_policy {
    LLAMA_KVARN_ISWA_DISABLED,
    LLAMA_KVARN_ISWA_ALL_LAYERS,
    LLAMA_KVARN_ISWA_STANDARD_SWA_FALLBACK,
    LLAMA_KVARN_ISWA_UNSUPPORTED,
};

llama_kvarn_iswa_policy llama_kvarn_iswa_policy_for(
        bool enabled,
        bool has_swa,
        uint32_t n_seq_max,
        bool unified,
        bool fail_if_unsupported);

size_t llama_kvarn_type_count();

const llama_kvarn_type_desc * llama_kvarn_type_desc_from_name(const char * name);
const llama_kvarn_type_desc * llama_kvarn_type_desc_from_type(llama_kvarn_type type);

llama_kvarn_tile_layout llama_kvarn_make_layout(int head_dim, int group, int key_bits, int value_bits);

int  llama_kvarn_head_slices(int head_dim);
bool llama_kvarn_head_dim_supported(int head_dim);

struct llama_kvarn_attention_plan {
    bool native_attention;
    enum ggml_flash_attn_ext_kvarn_domain domain;
};

llama_kvarn_attention_plan llama_kvarn_plan_attention(
        bool native_attention,
        bool native_original_v,
        uint32_t native_rotated_max_query_tokens,
        uint32_t n_query_tokens);

enum ggml_flash_attn_ext_kvarn_domain llama_kvarn_attention_domain(
        bool native_attention,
        bool native_original_v,
        uint32_t native_rotated_max_query_tokens,
        uint32_t n_query_tokens);

size_t  llama_kvarn_packed_bytes(int n_values, int bits);
void    llama_kvarn_pack_bits(const uint8_t * values, int n_values, int bits, uint8_t * dst);
uint8_t llama_kvarn_unpack_bits_value(const uint8_t * src, int index, int bits);

const char * llama_kvarn_validate_runtime(
        const llama_kvarn_params & params,
        const llama_kvarn_runtime_requirements & requirements);

bool llama_kvarn_can_remove_range(llama_pos pos_max, llama_pos p0, llama_pos p1, uint32_t group);
bool llama_kvarn_plan_remove_range(
        llama_pos pos_max,
        llama_pos p0,
        llama_pos p1,
        uint32_t group,
        bool stream_owned,
        llama_pos & planned_p0,
        llama_pos & planned_p1);

template<typename SeqPosMax>
bool llama_kvarn_stream_is_exclusive_for(
        uint32_t n_stream,
        size_t n_seq_max,
        llama_seq_id seq_id,
        const SeqPosMax & seq_pos_max) {
    if (n_stream == 0 || seq_id < 0 || size_t(seq_id) >= n_seq_max) {
        return false;
    }
    if (n_stream > 1) {
        return true;
    }
    for (size_t other = 0; other < n_seq_max; ++other) {
        if (other != size_t(seq_id) && seq_pos_max(llama_seq_id(other)) >= 0) {
            return false;
        }
    }
    return true;
}

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
