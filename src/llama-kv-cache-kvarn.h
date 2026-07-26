#pragma once

#include "llama-kv-cache.h"
#include "llama-kvarn.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

struct llama_hparams;
struct llama_model;

bool llama_kvarn_backend_supports_native_ops(ggml_backend_dev_t dev);
bool llama_kvarn_backend_supports_ops(ggml_backend_dev_t dev);
bool llama_kvarn_backend_native_attention_uses_original_v(ggml_backend_dev_t dev);
uint32_t llama_kvarn_backend_native_rotated_max_query_tokens(ggml_backend_dev_t dev);
bool llama_kvarn_backend_mixed_tail_native_preferred(ggml_backend_dev_t dev);

struct llama_kvarn_tail_policy {
    uint32_t raw_requested_tokens;
    uint32_t requested_tokens;
    uint32_t effective_tokens;
    uint32_t exact_groups;
    bool native_exact;
};

inline llama_kvarn_tail_policy llama_kvarn_tail_policy_for(
        uint32_t raw_requested_tokens,
        uint32_t effective_window) {
    constexpr uint32_t group_size = 128;
    if (effective_window == 0) {
        return { raw_requested_tokens, 0, 0, 0, true };
    }

    const uint32_t intrinsic = std::min(group_size, effective_window);
    const uint64_t rounded = raw_requested_tokens == 0 ? 0 :
        ((uint64_t(raw_requested_tokens) + group_size - 1u) / group_size) * group_size;
    const uint32_t explicit_tokens = uint32_t(std::min<uint64_t>(rounded, effective_window));
    const uint32_t effective_tokens = std::max(intrinsic, explicit_tokens);
    return {
        raw_requested_tokens,
        effective_tokens,
        effective_tokens,
        (effective_tokens + group_size - 1u) / group_size,
        effective_tokens == effective_window,
    };
}

// Completed groups are committed eagerly. Only the currently incomplete group
// needs persistent quantization workspace, regardless of physical ubatch size.
inline uint32_t llama_kvarn_workspace_groups(uint32_t n_ubatch) {
    GGML_UNUSED(n_ubatch);
    return 1;
}

inline uint32_t llama_kvarn_non_swa_tail_groups(uint32_t n_batch, uint32_t n_ubatch) {
    GGML_UNUSED(n_batch);
    GGML_UNUSED(n_ubatch);

    // Reference KVarN keeps only the currently incomplete 128-token group in
    // F16. Completed groups are quantized before the next group is evaluated.
    return 1;
}

class llama_kv_cache_kvarn;

class llama_kv_cache_kvarn_context : public llama_kv_cache_context {
public:
    llama_kv_cache_kvarn_context(
            llama_kv_cache_kvarn * cache,
            llama_memory_context_ptr base,
            llama_context * update_lctx = nullptr);

    bool next() override;
    bool apply() override;
    void graph_compute_start() override;
    void graph_compute_finish(ggml_status compute_status) override;

    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    uint32_t get_n_kv() const override;
    llama_kv_cache * get_kv() const override;
    const llama_kv_cache::slot_info & current_sinfo() const override;

    ggml_type type_k() const override;
    ggml_type type_v() const override;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const override;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const override;
    ggml_tensor * get_k_tail(ggml_context * ctx, int32_t il) const override;
    ggml_tensor * get_v_tail(ggml_context * ctx, int32_t il) const override;
    uint32_t get_tail_slots() const override;
    ggml_type get_tail_type() const override;
    uint32_t get_tail_tokens() const override;
    uint32_t get_tail_arena_stride() const override;
    uint32_t get_tail_attention_stride(uint32_t n_query_tokens = 0) const override;
    uint32_t get_tail_body_execution_stride() const override;
    uint32_t get_tail_body_execution_rows(int32_t il) const override;
    bool has_compact_tail() const override;
    bool has_kv_body() const override;
    bool has_kv_body(int32_t il) const override;
    bool has_tail_current(int32_t il) const override;
    ggml_backend_dev_t get_tail_backend(int32_t il) const override;
    llama_kv_tail_storage_kind get_tail_storage_kind() const override;
    uint32_t get_tail_rollback_tokens() const override;
    llama_kv_tail_route get_tail_route(int32_t il) const override;
    const llama_kv_tail_layer_route * get_tail_layer_route(int32_t il) const override;
    bool get_tail_explicit_bias(int32_t il) const override;
    bool can_pack_tail_body(const llama_ubatch & ubatch) const override;
    ggml_tensor * get_k_native(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_native(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_k_for_attention(ggml_context * ctx, int32_t il, bool native_attention) const;
    ggml_tensor * get_v_for_attention(ggml_context * ctx, int32_t il, bool native_attention) const;
    bool uses_native_attention(int32_t il) const;
    bool mixed_tail_native_preferred(int32_t il) const;
    bool native_attention_uses_original_v(int32_t il) const;
    uint32_t native_rotated_max_query_tokens(int32_t il) const;

    // SWA sliding-window ring: per-cell absolute positions for KVarN reads.
    // Built as a graph input sized [n_kv]; set on the host from cells.pos_get(cell).
    ggml_tensor * build_input_kvarn_rot(ggml_context * ctx, int n_rot) const;
    void set_input_kvarn_rot(ggml_tensor * dst) const;
    ggml_tensor * build_input_kvarn_mat_idxs(ggml_context * ctx) const;
    void set_input_kvarn_mat_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_mat_idxs(ggml_tensor * idxs) const { mat_idxs = idxs; }

    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const override;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const override;
    ggml_tensor * cpy_k_with_tail(
            ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs,
            ggml_tensor * tail_idxs, int32_t il) const override;
    ggml_tensor * cpy_v_with_tail(
            ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs,
            ggml_tensor * tail_idxs, int32_t il) const override;
    ggml_tensor * cpy_k_tail(
            ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * tail_idxs,
            int32_t il, ggml_tensor * dependency = nullptr) const override;
    ggml_tensor * cpy_v_tail(
            ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * tail_idxs,
            int32_t il, ggml_tensor * dependency = nullptr) const override;

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override;
    ggml_tensor * build_input_tail_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override;
    ggml_tensor * build_input_tail_body_idxs(ggml_context * ctx) const override;
    ggml_tensor * build_input_k_rot(ggml_context * ctx) const override;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const override;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_tail_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_tail_body_idxs(ggml_tensor * dst) const override;
    void set_input_k_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_v_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_k_shift(ggml_tensor * dst) const override;
    void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const override;
    void set_input_kq_mask_tail(
            ggml_tensor * body, ggml_tensor * exact,
            ggml_tensor * read_idxs, ggml_tensor * body_read_idxs, ggml_tensor * bias_read_idxs,
            const llama_ubatch * ubatch, bool causal_attn) const override;
    void set_input_tail_body_plan(
            ggml_tensor * query_order, ggml_tensor * run_desc,
            ggml_tensor * body_mask, const llama_ubatch * ubatch, bool causal_attn) const override;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_k_rot(ggml_tensor * dst) const override;
    void set_input_v_rot(ggml_tensor * dst) const override;
    void set_input_k_rot_backend(ggml_tensor * dst) const override;
    void set_input_v_rot_backend(ggml_tensor * dst) const override;

private:
    llama_kv_cache_context * base() const;

    llama_kv_cache_kvarn * cache;
    llama_memory_context_ptr base_ctx;
    llama_context * update_lctx;

    mutable std::unordered_map<int32_t, ggml_tensor *> stored_k;
    mutable std::unordered_map<int32_t, ggml_tensor *> stored_v;
    mutable ggml_tensor * mat_idxs = nullptr; // SWA per-cell absolute positions for views/materialization
};

class llama_kv_cache_kvarn : public llama_memory_i {
public:
    llama_kv_cache_kvarn(
            const llama_model & model,
            const llama_hparams & hparams,
            llama_kvarn_params params,
            bool offload,
            bool unified,
            uint32_t kv_size,
            uint32_t n_seq_max,
            uint32_t n_batch,
            uint32_t n_ubatch,
            uint32_t n_pad = 1,
            uint32_t n_swa = 0,
            llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE,
            const layer_filter_cb & filter = nullptr,
            const layer_reuse_cb & reuse = nullptr,
            uint32_t tail_tokens = 0,
            ggml_type tail_type = GGML_TYPE_F16,
            uint32_t tail_tokens_requested = UINT32_MAX,
            uint32_t tail_rollback_tokens = 0);

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    uint32_t get_kv_n_stream() const override;
    uint32_t get_kv_size() const override;
    llama_memory_context_ptr init_kv_batch(const std::vector<llama_ubatch> & ubatches) override;

    bool get_can_shift() const override;
    seq_rm_capability get_seq_rm_capability() const override;

    void clear(bool data) override;
    bool can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const override;
    bool seq_rm_plan(
            llama_seq_id seq_id, llama_pos p0, llama_pos p1,
            llama_pos & planned_p0, llama_pos & planned_p1) const override;
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    bool seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) override;
    int cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) override;
    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    GGML_NORETURN void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    GGML_NORETURN void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    llama_kv_memory_stats kv_memory_stats() const override;
    ggml_type get_kv_tail_type() const override { return exact_tail_type; }
    uint32_t get_kv_tail_group_count() const override { return 1; }
    bool get_kv_tail_coverage(
            uint32_t group_index,
            llama_seq_id seq_id,
            llama_kv_tail_coverage_info & out) const override;
    void reset_kv_tail_planner_timing() override;
    uint64_t get_kv_tail_planner_timing_ns() const override;

    bool requires_state_for_partial_restore() const override;
    bool state_seq_can_save(llama_seq_id seq_id) const override;
    bool state_seq_can_restore(llama_seq_id seq_id) const override;
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read(llama_io_read_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    llama_kv_cache * get_metadata_cache() const;
    int32_t mapped_layer_id(int32_t il) const;
    llama_kv_tail_route get_tail_route(int32_t il) const;
    bool get_tail_explicit_bias(int32_t il) const;
    ggml_type get_tail_type() const { return exact_tail_type; }
    bool has_pending_stream_copies() const;
    bool stream_is_exclusive_for(llama_seq_id seq_id) const;
    bool apply_pending_stream_copies(llama_context * lctx);
    bool is_swa() const { return swa; }
    bool uses_native_attention(int32_t il) const;
    bool mixed_tail_native_preferred(int32_t il) const;
    bool native_attention_uses_original_v(int32_t il) const;
    uint32_t native_rotated_max_query_tokens(int32_t il) const;

    // Reference-faithful staging keeps one incomplete 128-token group lossless.
    // Completed records are committed eagerly, so physical ubatch size does not
    // change the logical exact suffix.
    //   non-SWA tail_groups = 1
    //   SWA tail_groups     = 2 physical wrap-safety slots
    //   stage_groups        = tail_groups + 1 for non-SWA, tail_groups for SWA
    // The +1 is only the permanent sink slot for non-SWA. SWA has no sink slot,
    // so both F16 slots are physical quantization workspace. The SWA record ring
    // carries the active ubatch span because early query rows can still attend
    // older window groups after later rows advance the ring.
    uint32_t get_stage_groups() const { return stage_groups; }
    uint32_t get_tail_groups()  const { return tail_groups; }

    ggml_tensor * store(
            ggml_context * ctx,
            ggml_tensor * current,
            ggml_tensor * indices,
            int32_t il,
            const llama_kv_cache::slot_info & sinfo,
            bool value) const;
    ggml_tensor * view(
            ggml_context * ctx,
            ggml_tensor * stored,
            int32_t il,
            uint32_t n_kv,
            const llama_kv_cache::slot_info & sinfo,
            bool value,
            ggml_tensor * mat_idxs = nullptr) const;
    ggml_tensor * materialize(
            ggml_context * ctx,
            ggml_tensor * stored,
            int32_t il,
            uint32_t n_kv,
            const llama_kv_cache::slot_info & sinfo,
            bool value,
            ggml_tensor * mat_idxs = nullptr) const;
    ggml_tensor * get_tail(ggml_context * ctx, int32_t il, bool value) const;
    ggml_tensor * store_tail(
            ggml_context * ctx, ggml_tensor * current, ggml_tensor * indices,
            int32_t il, bool value, ggml_tensor * dependency = nullptr) const;

private:
    struct layer {
        uint32_t il;
        uint32_t n_head_kv;
        uint32_t head_dim_k;
        uint32_t head_dim_v;
        uint32_t k_slices;
        uint32_t v_slices;
        bool native_attention;
        bool mixed_tail_native;
        bool native_original_v;
        uint32_t native_rotated_max_query_tokens;
        ggml_tensor * k_records;
        ggml_tensor * v_records;
        ggml_tensor * k_stage;
        ggml_tensor * v_stage;
        ggml_tensor * k_tail;
        ggml_tensor * v_tail;
        std::vector<ggml_tensor *> k_records_stream;
        std::vector<ggml_tensor *> v_records_stream;
        std::vector<ggml_tensor *> k_stage_stream;
        std::vector<ggml_tensor *> v_stage_stream;
    };

    const layer & layer_for(int32_t il) const;
    std::unique_ptr<llama_kv_cache> make_metadata_cache() const;
    bool can_remove(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;
    void copy_kvarn_stream(uint32_t stream_src, uint32_t stream_dst);

    const llama_model & model;
    const llama_hparams & hparams;
    const llama_kvarn_params params;
    const uint32_t n_stream;
    const uint32_t n_seq_max;
    const uint32_t kv_size;
    // Declaration order matters: stage_groups depends on tail_groups, so
    // tail_groups must be declared first (C++ initializes members in order).
    const uint32_t tail_groups;   // non-SWA scheduler span; SWA fixed local tail
    const uint32_t stage_groups;   // F16 stage depth (non-SWA sink + tail; SWA tail only)
    const bool swa;
    const uint32_t n_groups_per_stream;
    const uint32_t exact_tail_tokens;
    const uint32_t metadata_n_pad;
    const uint32_t metadata_n_swa;
    const llama_swa_type metadata_swa_type;
    const uint32_t metadata_n_ubatch;
    const uint32_t exact_tail_tokens_requested;
    const ggml_type exact_tail_type_requested;
    ggml_type exact_tail_type;

    std::unique_ptr<llama_kv_cache> metadata;
    std::vector<layer> layers;
    std::unordered_map<int32_t, int32_t> map_layer_ids;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;
    llama_kv_cache::stream_copy_info pending_stream_copies;
};
