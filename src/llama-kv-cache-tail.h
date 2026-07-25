#pragma once

#include "llama.h"
#include "llama-hparams.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <list>
#include <optional>
#include <stdexcept>
#include <string>
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

struct llama_kv_tail_slot_run {
    uint32_t payload_begin;
    int32_t  slot_begin;
    uint32_t length;
};

std::vector<llama_kv_tail_slot_run> llama_kv_tail_contiguous_slot_runs(
        const std::vector<int32_t> & slots);

struct llama_kv_tail_snapshot_entry {
    llama_seq_id seq_id;
    llama_kv_tail_identity identity;
    llama_pos position;
    uint64_t insertion_ordinal;
    int32_t slot;
};

struct llama_kv_tail_provenance {
    llama_seq_id seq_id;
    uint32_t degradation_flags;
    uint32_t recovery_commits;
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

struct llama_kv_tail_compact_layout {
    uint32_t history_stride;
    uint32_t history_slots;
    uint32_t rollback_tokens;
    uint32_t attention_stride;
};

// Reserve a discard sink only when a batch can contain ragged sequence
// memberships. With one sequence every write has exactly one arena target.
llama_kv_tail_layout llama_kv_tail_layout_for(
        uint32_t n_tokens,
        uint32_t n_seq_max,
        uint32_t n_ubatch);

// Persistent compact history contains only the active exact suffix and the
// explicitly promised rollback reserve. Current-ubatch rows are graph inputs,
// so they affect descriptor workspace but never persistent payload capacity.
llama_kv_tail_compact_layout llama_kv_tail_compact_layout_for(
        uint32_t n_tokens,
        uint32_t rollback_tokens,
        uint32_t n_seq_max,
        uint32_t n_ubatch);

// Compact-tail suffix removal is an operation contract, not a report of how
// many cells changed. A suffix beginning beyond the current end is therefore
// an accepted idempotent no-op.
bool llama_kv_tail_can_remove_suffix(
        llama_pos pos_max,
        llama_pos p0,
        llama_pos p1,
        uint32_t rollback_tokens);

enum llama_kv_tail_storage_kind {
    LLAMA_KV_TAIL_STORAGE_DISABLED,
    LLAMA_KV_TAIL_STORAGE_OVERLAY,
    LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT,
    LLAMA_KV_TAIL_STORAGE_COMPACT_OVERLAY,
    LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT,
};

enum llama_kv_tail_route : int {
    LLAMA_KV_TAIL_ROUTE_NONE,
    LLAMA_KV_TAIL_ROUTE_NATIVE,
    LLAMA_KV_TAIL_ROUTE_GENERIC,
};

enum llama_kv_tail_operation {
    LLAMA_KV_TAIL_OP_NONE,
    LLAMA_KV_TAIL_OP_WRITE_K,
    LLAMA_KV_TAIL_OP_WRITE_V,
    LLAMA_KV_TAIL_OP_GATHER_K,
    LLAMA_KV_TAIL_OP_GATHER_V,
    LLAMA_KV_TAIL_OP_BODY_SCORE,
    LLAMA_KV_TAIL_OP_BODY_VALUE,
    LLAMA_KV_TAIL_OP_EXACT_SCORE,
    LLAMA_KV_TAIL_OP_EXACT_VALUE,
    LLAMA_KV_TAIL_OP_GENERIC_MERGE,
    LLAMA_KV_TAIL_OP_NATIVE_ATTENTION,
};

struct llama_kv_tail_route_requirements {
    bool write_k = true;
    bool write_v = true;
    bool gather_k = true;
    bool gather_v = true;
    bool body_score = true;
    bool body_value = true;
    bool exact_score = true;
    bool exact_value = true;
    bool generic_merge = true;
    bool native_attention = false;
};

struct llama_kv_tail_route_capability {
    bool supported;
    llama_kv_tail_route route;
    llama_kv_tail_operation missing_operation;
};

struct llama_kv_tail_type_resolution {
    bool supported;
    ggml_type actual_type;
    bool downgraded;
    llama_kv_tail_operation missing_preferred;
    llama_kv_tail_operation missing_fallback;
};

llama_kv_tail_route_capability llama_kv_tail_select_route(
        const llama_kv_tail_route_requirements & requirements);

const char * llama_kv_tail_operation_name(llama_kv_tail_operation operation);

llama_kv_tail_type_resolution llama_kv_tail_resolve_type(
        ggml_type requested,
        ggml_type family_default,
        const llama_kv_tail_route_capability & bf16,
        const llama_kv_tail_route_capability & f16);

struct llama_kv_tail_layer_route {
    uint32_t layer_id;
    std::string backend;
    ggml_type body_type_k;
    ggml_type body_type_v;
    ggml_type exact_type_k;
    ggml_type exact_type_v;
    bool v_transposed;
    bool causal_attn;
    bool swa;
    bool explicit_bias;
    bool has_body;
    bool has_current;
    uint32_t body_execution_rows;
    ggml_backend_dev_t owner;
    llama_kv_tail_route_capability capability;
};

// Backend-neutral ownership contract for one persistent KV layer. Owner tokens
// are deliberately opaque: production code derives them from the realized
// backend buffers while synthetic tests can use stable logical device IDs.
// A zero shadow owner means that side has no exact overlay tensor.
struct llama_kv_tail_layer_ownership {
    uint32_t layer_id;
    uintptr_t body_k_owner;
    uintptr_t body_v_owner;
    uintptr_t shadow_k_owner;
    uintptr_t shadow_v_owner;
    uintptr_t graph_write_owner;
    uintptr_t graph_read_owner;
    uintptr_t state_k_owner;
    uintptr_t state_v_owner;
    bool body_k_meta_split;
    bool body_v_meta_split;
};

enum llama_kv_tail_ownership_error {
    LLAMA_KV_TAIL_OWNERSHIP_OK,
    LLAMA_KV_TAIL_OWNERSHIP_META_SPLIT_K,
    LLAMA_KV_TAIL_OWNERSHIP_META_SPLIT_V,
    LLAMA_KV_TAIL_OWNERSHIP_SHADOW_K,
    LLAMA_KV_TAIL_OWNERSHIP_SHADOW_V,
    LLAMA_KV_TAIL_OWNERSHIP_GRAPH_WRITE,
    LLAMA_KV_TAIL_OWNERSHIP_GRAPH_READ,
    LLAMA_KV_TAIL_OWNERSHIP_STATE_K,
    LLAMA_KV_TAIL_OWNERSHIP_STATE_V,
};

llama_kv_tail_layer_ownership llama_kv_tail_plan_layer_ownership(
        uint32_t layer_id,
        uintptr_t body_k_owner,
        uintptr_t body_v_owner,
        bool shadow_k,
        bool shadow_v);

llama_kv_tail_ownership_error llama_kv_tail_validate_layer_ownership(
        const llama_kv_tail_layer_ownership & ownership);

struct llama_kv_tail_group_request {
    uint32_t requested_tokens;
    uint32_t effective_window;
    bool applicable_standard_kv;
};

struct llama_kv_tail_group_resolution {
    bool valid;
    std::vector<uint32_t> tokens;
};

// Resolve the model-bound group configuration atomically. Automatic mode is
// intentionally architecture-agnostic for this release: every applicable
// standard KV group requests 1024 exact tokens, capped by its own window.
llama_kv_tail_group_resolution llama_kv_tail_resolve_groups(
        bool automatic,
        bool explicit_complete,
        const std::vector<llama_kv_tail_group_request> & groups);

// Sparse body packing must remain valid for every future occupancy of a
// cached graph. It is therefore safe only when the complete physical stream,
// not merely its currently live rows, fits in the reserved arena.
bool llama_kv_tail_sparse_body_capacity_safe(
        uint32_t physical_stream_rows,
        uint32_t arena_stride);

// The packed body is graph-local and may be padded to a backend execution
// alignment without changing the persistent compact-history capacity.
uint32_t llama_kv_tail_packed_body_stride(
        uint64_t logical_rows,
        uint32_t alignment);

struct llama_kv_tail_storage_request {
    ggml_type requested_body_type_k;
    ggml_type requested_body_type_v;
    ggml_type exact_type;
    uint32_t requested_tokens;
    uint32_t effective_tokens;
    uint32_t n_seq_max;
    uint32_t n_ubatch;
    uint32_t visibility_window;
    uint64_t physical_body_rows;
    uint64_t requested_body_bytes_per_row;
    uint64_t promotion_bytes_per_row;
    uint64_t overlay_bytes_per_row;
    bool native_capable;
    bool already_exact;
    bool has_owned_body;
    bool has_shared_body;
    bool shadow_k_capable;
    bool shadow_v_capable;
    bool graph_consumes_exact_tail = true;
    bool overlay_placement_supported = true;
    const char * architecture = nullptr;
    const char * group_id = nullptr;
    uint32_t rollback_tokens = 0;
    bool compact_history_capable = false;
    bool compact_current_source_capable = false;
    bool compact_ordered_commit_capable = false;
    bool full_window_body_can_be_omitted = false;
};

struct llama_kv_tail_storage_plan {
    llama_kv_tail_storage_kind kind;
    uint32_t requested_tokens;
    uint32_t effective_tokens;
    uint32_t visibility_window;
    ggml_type requested_body_type_k;
    ggml_type requested_body_type_v;
    ggml_type actual_body_type_k;
    ggml_type actual_body_type_v;
    bool shadow_k;
    bool shadow_v;
    ggml_type shadow_type_k;
    ggml_type shadow_type_v;
    uint64_t physical_body_rows;
    llama_kv_tail_layout layout;
    uint64_t requested_body_bytes;
    uint64_t actual_body_bytes;
    uint64_t shadow_bytes;
    uint64_t promotion_increment;
    uint64_t overlay_increment;
    bool has_owned_body;
    bool has_shared_body;
    bool body_promoted;
    // Native-exact storage needs no overlay consumer. For an overlay this is
    // the model-layer declaration captured before allocation and is retained
    // so coverage cannot claim completeness for an unconsumed payload.
    bool graph_consumes_exact_tail = true;
    std::vector<llama_kv_tail_layer_route> layer_routes = {};
    llama_kv_tail_compact_layout compact_layout = { 0, 0, 0, 0 };
    uint64_t compact_history_bytes = 0;
    uint64_t compact_rollback_bytes = 0;
};

llama_kv_tail_storage_plan llama_kv_tail_storage_plan_for(
        const llama_kv_tail_storage_request & request);

struct llama_kv_tail_mask_entry {
    uint32_t exact;
    uint32_t cell;
    llama_pos position;
};

// Select the newest finite exact entries for one query. The result contains
// indices into entries in ascending exact order, which permits callers to
// coalesce adjacent physical rows without accidentally selecting masked gaps.
void llama_kv_tail_select_masked_entries(
        const std::vector<llama_kv_tail_mask_entry> & entries,
        const std::vector<uint8_t> & finite,
        llama_pos query_position,
        uint32_t retention,
        bool causal_attn,
        std::vector<uint32_t> & selected);

struct llama_kv_tail_body_row {
    llama_pos position;
    uint32_t cell;
    int32_t flat;
};

struct llama_kv_tail_query_window {
    uint32_t mask_row;
    llama_pos position;
};

// Reduce ordered live body rows to the union of finite SWA candidates. The
// predicate is evaluated only within position-window intervals, keeping work
// proportional to the candidate union rather than queries*physical-cache.
template <typename IsFinite>
void llama_kv_tail_union_swa_rows(
        std::vector<llama_kv_tail_body_row> & rows,
        const std::vector<llama_kv_tail_query_window> & queries,
        uint32_t n_swa,
        llama_swa_type swa_type,
        bool causal_attn,
        IsFinite is_finite,
        std::vector<uint8_t> & selected_rows) {
    if (n_swa == 0 || swa_type == LLAMA_SWA_TYPE_NONE) {
        throw std::invalid_argument("SWA row union requires a non-zero window");
    }

    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
        return a.position < b.position || (a.position == b.position && a.cell < b.cell);
    });
    selected_rows.assign(rows.size(), 0);
    const auto clamp_pos = [](int64_t value) {
        return llama_pos(std::max<int64_t>(std::numeric_limits<llama_pos>::min(),
                std::min<int64_t>(std::numeric_limits<llama_pos>::max(), value)));
    };

    for (const auto & query : queries) {
        const int64_t p1 = query.position;
        int64_t lo64 = std::numeric_limits<llama_pos>::min();
        int64_t hi64 = std::numeric_limits<llama_pos>::max();
        switch (swa_type) {
            case LLAMA_SWA_TYPE_STANDARD:
                lo64 = p1 - int64_t(n_swa) + 1;
                break;
            case LLAMA_SWA_TYPE_CHUNKED:
                lo64 = (p1/int64_t(n_swa))*int64_t(n_swa);
                break;
            case LLAMA_SWA_TYPE_SYMMETRIC:
                lo64 = p1 - int64_t(n_swa/2);
                hi64 = p1 + int64_t(n_swa/2);
                break;
            case LLAMA_SWA_TYPE_NONE:
                throw std::logic_error("unreachable SWA planner mode");
        }
        if (causal_attn) {
            hi64 = std::min(hi64, p1);
        }
        const llama_pos lo = clamp_pos(lo64);
        const llama_pos hi = clamp_pos(hi64);
        const auto first = std::lower_bound(rows.begin(), rows.end(), lo,
                [](const auto & row, llama_pos pos) { return row.position < pos; });
        const auto last = std::upper_bound(first, rows.end(), hi,
                [](llama_pos pos, const auto & row) { return pos < row.position; });
        for (auto it = first; it != last; ++it) {
            const size_t index = size_t(it - rows.begin());
            if (is_finite(query.mask_row, it->cell)) {
                selected_rows[index] = 1;
            }
        }
    }

    size_t out = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (selected_rows[i]) {
            rows[out++] = rows[i];
        }
    }
    rows.resize(out);
}

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
    llama_kv_tail_store(
            uint32_t n_tokens,
            uint32_t rollback_tokens,
            uint32_t n_seq_max,
            uint32_t history_stride,
            uint32_t sink_slots);

    int32_t commit(
            llama_seq_id seq_id,
            llama_kv_tail_identity identity,
            llama_pos position,
            uint64_t insertion_ordinal,
            uint32_t current_row = UINT32_MAX);

    void begin_batch();
    void finish_batch(bool success, bool payload_may_be_modified);
    bool has_batch_transaction() const { return batch_transaction.has_value(); }
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
    std::vector<llama_kv_tail_slot_copy> prepare_seq_cp(
            llama_seq_id src,
            llama_seq_id dst,
            uint32_t src_stream,
            uint32_t dst_stream,
            llama_pos p0,
            llama_pos p1);
    void commit_seq_cp();
    void cancel_seq_cp();
    bool has_pending_seq_cp() const { return pending_seq_cp.has_value(); }
    llama_seq_id pending_seq_cp_destination() const;
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
    uint32_t history_capacity() const { return history_limit; }
    uint32_t rollback_horizon() const { return rollback_tokens; }
    bool supports_suffix_rollback(llama_seq_id seq_id, uint32_t n) const;
    std::vector<llama_kv_tail_snapshot_entry> snapshot(llama_seq_id seq_id = -1) const;
    std::vector<llama_kv_tail_provenance> snapshot_provenance(llama_seq_id seq_id = -1) const;
    void restore_provenance(
            const std::vector<llama_kv_tail_provenance> & provenance,
            llama_seq_id dest_seq_id = -1);

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
    using entry_list = std::list<exact_entry>;

    struct pending_seq_cp_state {
        llama_seq_id src;
        llama_seq_id dst;
        std::vector<exact_entry> destination;
        std::vector<llama_kv_tail_slot_copy> copies;
        std::vector<int32_t> acquired_slots;
        uint32_t degradation_flags;
        uint32_t recovery_commits;
    };

    struct batch_transaction_state {
        std::vector<entry_list> sequences;
        std::vector<std::vector<bool>> slot_used;
        std::vector<uint32_t> write_cursors;
        std::vector<uint32_t> degradation;
        std::vector<uint32_t> recovery_commits;
        std::vector<entry_list> current;
        std::vector<bool> affected;
    };

    const uint32_t n_tokens;
    const uint32_t rollback_tokens;
    const uint32_t history_limit;
    const uint32_t arena_stride;
    const uint32_t n_slots;
    const bool compact_storage;
    std::vector<entry_list> sequences;
    // Each sequence can own at most one exact record for a physical KV cell.
    // Keep that identity lookup incremental so a token commit/recycle does not
    // scan the full retained tail.
    std::vector<std::unordered_map<uint64_t, entry_list::iterator>> entry_by_cell;
    std::vector<std::vector<bool>> slot_used;
    std::vector<uint32_t> write_cursors;
    std::vector<uint32_t> degradation;
    std::vector<uint32_t> recovery_commits;
    std::optional<pending_seq_cp_state> pending_seq_cp;
    std::optional<batch_transaction_state> batch_transaction;
    bool in_batch = false;

    bool valid_seq(llama_seq_id seq_id) const;
    static uint64_t cell_key(uint32_t stream, uint32_t cell);
    void rebuild_index(llama_seq_id seq_id);
    void erase_entry(llama_seq_id seq_id, entry_list::iterator entry, bool release_slot = true);
    int32_t acquire(llama_seq_id seq_id);
    void release(llama_seq_id seq_id, int32_t slot);
    void trim(llama_seq_id seq_id, uint32_t limit);
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
