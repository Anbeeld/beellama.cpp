#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <vector>

constexpr uint32_t KVAR_N_GROUP = 128;
// Stage indices carry both the logical record cell and, when present, the
// host-selected physical F16 slot. A zero high word retains the legacy
// stateless mapping for non-unified callers; explicit slots are one-based in
// the packed representation so slot zero remains distinguishable.
constexpr int64_t llama_kvarn_encode_store_cell(uint32_t cell, uint32_t stage_slot) {
    return int64_t((uint64_t(stage_slot) + 1u) << 32u | uint64_t(cell));
}

constexpr int64_t llama_kvarn_encode_stage_cell(uint32_t cell) {
    return -int64_t(cell) - 2;
}

constexpr int64_t llama_kvarn_encode_stage_cell(uint32_t cell, uint32_t stage_slot) {
    return -llama_kvarn_encode_store_cell(cell, stage_slot) - 2;
}

constexpr uint64_t llama_kvarn_index_payload(int64_t index) {
    return index < -1 ? uint64_t(-(index + 2)) : uint64_t(index);
}

constexpr uint32_t llama_kvarn_decode_cell(int64_t index) {
    return uint32_t(llama_kvarn_index_payload(index));
}

constexpr int32_t llama_kvarn_decode_stage_slot(int64_t index) {
    const uint32_t encoded = uint32_t(llama_kvarn_index_payload(index) >> 32u);
    return encoded == 0 ? -1 : int32_t(encoded - 1u);
}

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
};

llama_kvarn_iswa_policy llama_kvarn_iswa_policy_for(
        bool enabled,
        bool has_swa,
        uint32_t n_seq_max);

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

struct llama_kvarn_state_stage_cell {
    uint32_t source_cell;
    uint32_t stage_row;
};

// Group ownership includes reservations made earlier in the same planning
// transaction. This prevents a fresh physical record group from being assigned
// to different logical sequence sets before cell metadata is committed.
bool llama_kvarn_group_owner_compatible(
        uint32_t group_used,
        bool group_mixed,
        const std::vector<llama_seq_id> & current_owners,
        const std::vector<llama_seq_id> & desired_owners);

// Reconciles process-local physical F16 ownership with a logical live-group
// set. Existing valid owners remain stable, dead owners release immediately,
// and failure is atomic when physical capacity is insufficient.
bool llama_kvarn_reconcile_stage_slots(
        const std::vector<uint32_t> & live_groups,
        uint32_t group_capacity,
        uint32_t stage_slot_capacity,
        std::vector<int32_t> & group_slots);

std::vector<llama_kvarn_state_stage_cell> llama_kvarn_select_state_stage_cells(
        const std::vector<uint32_t> & source_cells,
        uint32_t live_cell_max_p1,
        uint32_t stage_groups,
        uint32_t tail_groups,
        bool swa,
        const std::vector<uint32_t> * staged_groups = nullptr,
        const std::vector<int32_t> * group_stage_slots = nullptr);

std::vector<uint32_t> llama_kvarn_select_state_record_groups(
        const std::vector<uint32_t> & source_cells,
        const std::vector<llama_kvarn_state_stage_cell> & stage_cells,
        uint32_t groups_per_stream);

// Builds the dense physical-cell index used by KVarN attention. Unified record
// ownership can leave holes in the physical arena; those holes are storage
// ownership, not attention rows.
std::vector<int64_t> llama_kvarn_compact_read_plan(
        const std::vector<uint32_t> & occupied_cells,
        const std::vector<uint32_t> & pending_cells,
        uint32_t capacity,
        uint32_t padding);

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
