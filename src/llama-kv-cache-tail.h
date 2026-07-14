#pragma once

#include "llama.h"

#include <cstdint>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

constexpr int32_t LLAMA_KV_TAIL_BODY_SLOT = -1;

struct llama_kv_tail_identity {
    uint32_t stream;
    uint32_t cell;
    uint64_t generation;

    bool operator==(const llama_kv_tail_identity & other) const {
        return stream == other.stream && cell == other.cell && generation == other.generation;
    }
};

struct llama_kv_tail_coverage {
    llama_kv_tail_coverage_state state;
    uint32_t requested;
    uint32_t exact;
    uint32_t degradation_flags;
};

struct llama_kv_tail_slot_copy {
    int32_t src_slot;
    int32_t dst_slot;
};

struct llama_kv_tail_snapshot_entry {
    llama_seq_id seq_id;
    llama_kv_tail_identity identity;
    llama_pos position;
    uint64_t insertion_ordinal;
    int32_t slot;
};

struct llama_kv_tail_source_run {
    uint32_t exact_offset;
    uint32_t stream;
    uint32_t cell;
    uint32_t length;
};

struct llama_kv_tail_layout {
    uint32_t arena_stride;
    uint32_t sink_slots;
    uint32_t total_slots;
};

// Reserve a discard sink only when a batch can contain ragged sequence
// memberships. With one sequence every write has exactly one arena target.
llama_kv_tail_layout llama_kv_tail_layout_for(
        uint32_t n_tokens,
        uint32_t n_seq_max,
        uint32_t n_ubatch);

enum llama_kv_tail_storage_kind {
    LLAMA_KV_TAIL_STORAGE_DISABLED,
    LLAMA_KV_TAIL_STORAGE_OVERLAY,
    LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT,
};

struct llama_kv_tail_storage_request {
    ggml_type body_type_k;
    ggml_type body_type_v;
    ggml_type exact_type;
    uint32_t n_tokens;
    uint32_t n_seq_max;
    uint32_t n_ubatch;
    uint32_t visibility_window;
    uint64_t physical_body_rows;
    uint64_t promotion_bytes_per_row;
    uint64_t overlay_bytes_per_row;
    bool native_capable;
    bool already_exact;
};

struct llama_kv_tail_storage_plan {
    llama_kv_tail_storage_kind kind;
    ggml_type body_type_k;
    ggml_type body_type_v;
    llama_kv_tail_layout layout;
    uint64_t promotion_bytes;
    uint64_t overlay_bytes;
    bool body_promoted;
};

llama_kv_tail_storage_plan llama_kv_tail_storage_plan_for(
        const llama_kv_tail_storage_request & request);

// Backend-neutral ownership and source-selection metadata for one physical
// standard-cache group. Payload tensors are owned by llama_kv_cache; this class
// assigns their compact slot indices and never reconstructs exact data.
class llama_kv_tail_store {
public:
    // Three-argument form is retained for focused tests and treats all rows as
    // equally divided per-sequence arenas with no write-sink slab.
    llama_kv_tail_store(uint32_t n_tokens, uint32_t n_seq_max, uint32_t n_slots);
    llama_kv_tail_store(
            uint32_t n_tokens,
            uint32_t n_seq_max,
            uint32_t arena_stride,
            uint32_t sink_slots);

    int32_t commit(
            llama_seq_id seq_id,
            llama_kv_tail_identity identity,
            llama_pos position,
            uint64_t insertion_ordinal);

    void begin_batch();
    void recycle(uint32_t stream, uint32_t cell, uint64_t next_generation);
    void clear();
    void mark_degraded(llama_seq_id seq_id, uint32_t flags);
    void invalidate_slots(const std::vector<int32_t> & slots, uint32_t flags);
    void seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1);
    std::vector<llama_kv_tail_slot_copy> seq_cp_remap(
            llama_seq_id src,
            llama_seq_id dst,
            uint32_t src_stream,
            uint32_t dst_stream,
            llama_pos p0,
            llama_pos p1);
    void seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1);
    void seq_rm_cell(llama_seq_id seq_id, uint32_t stream, uint32_t cell);
    void seq_keep(llama_seq_id seq_id);
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift);
    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int divisor);

    std::vector<int32_t> build_source_plan(
            llama_seq_id seq_id,
            const std::vector<llama_kv_tail_identity> & visible) const;

    llama_kv_tail_coverage coverage(llama_seq_id seq_id, uint32_t available = UINT32_MAX) const;
    std::vector<int32_t> body_indices(uint32_t kv_size) const;
    std::vector<std::pair<int32_t, llama_kv_tail_identity>> active_slots() const;
    std::vector<llama_kv_tail_snapshot_entry> source_candidates(llama_seq_id seq_id) const;
    std::vector<llama_kv_tail_source_run> source_runs(llama_seq_id seq_id) const;
    uint32_t retention() const { return n_tokens; }
    std::vector<llama_kv_tail_snapshot_entry> snapshot(llama_seq_id seq_id = -1) const;

private:
    struct identity_hash {
        size_t operator()(const llama_kv_tail_identity & value) const;
    };

    struct exact_entry {
        llama_kv_tail_identity identity;
        llama_pos position;
        uint64_t insertion_ordinal;
        int32_t slot;
    };

    const uint32_t n_tokens;
    const uint32_t arena_stride;
    const uint32_t sink_slots;
    const uint32_t n_slots;
    using entry_list = std::list<exact_entry>;
    std::vector<entry_list> sequences;
    // Each sequence can own at most one exact record for a physical KV cell.
    // Keep that identity lookup incremental so a token commit/recycle does not
    // scan the full retained tail.
    std::vector<std::unordered_map<uint64_t, entry_list::iterator>> entry_by_cell;
    std::vector<std::vector<bool>> slot_used;
    std::vector<uint32_t> write_cursors;
    std::vector<uint32_t> degradation;
    std::vector<uint32_t> recovery_commits;
    bool in_batch = false;

    bool valid_seq(llama_seq_id seq_id) const;
    static uint64_t cell_key(uint32_t stream, uint32_t cell);
    void rebuild_index(llama_seq_id seq_id);
    void erase_entry(llama_seq_id seq_id, entry_list::iterator entry, bool release_slot = true);
    int32_t acquire(llama_seq_id seq_id);
    void release(llama_seq_id seq_id, int32_t slot);
    void trim(llama_seq_id seq_id);
};

std::vector<float> llama_kv_tail_attention_reference(
        const std::vector<float> & query,
        const std::vector<float> & body_k,
        const std::vector<float> & body_v,
        const std::vector<float> & tail_k,
        const std::vector<float> & tail_v,
        const std::vector<int32_t> & source_slots,
        uint32_t key_dim,
        uint32_t value_dim,
        float scale);
