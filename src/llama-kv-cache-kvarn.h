#pragma once

#include "llama-kv-cache.h"
#include "llama-kvarn.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

struct llama_hparams;
struct llama_model;

bool llama_kvarn_backend_supports_native_ops(ggml_backend_dev_t dev);

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

    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    uint32_t get_n_kv() const override;
    llama_kv_cache * get_kv() const override;
    const llama_kv_cache::slot_info & current_sinfo() const override;

    ggml_type type_k() const override;
    ggml_type type_v() const override;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const override;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const override;
    ggml_tensor * get_k_native(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_native(ggml_context * ctx, int32_t il) const;

    // SWA sliding-window ring: per-cell absolute positions for native KVarN views.
    // Built as a graph input sized [n_kv]; set on the host from cells.pos_get(cell).
    ggml_tensor * build_input_kvarn_rot(ggml_context * ctx, int n_rot) const;
    void set_input_kvarn_rot(ggml_tensor * dst) const;
    ggml_tensor * build_input_kvarn_mat_idxs(ggml_context * ctx) const;
    void set_input_kvarn_mat_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_mat_idxs(ggml_tensor * idxs) const { mat_idxs = idxs; }

    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const override;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const override;

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override;
    ggml_tensor * build_input_k_rot(ggml_context * ctx) const override;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const override;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_k_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_v_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_k_shift(ggml_tensor * dst) const override;
    void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const override;
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
    mutable ggml_tensor * mat_idxs = nullptr; // SWA per-cell absolute positions for native KVarN views
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
            const layer_reuse_cb & reuse = nullptr);

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

    void clear(bool data) override;
    bool can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const override;
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

    bool requires_state_for_partial_restore() const override;
    bool state_seq_restore_requires_exclusive_kv_stream() const override;
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read(llama_io_read_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    llama_kv_cache * get_metadata_cache() const;
    int32_t mapped_layer_id(int32_t il) const;
    bool has_pending_stream_copies() const;
    bool apply_pending_stream_copies(llama_context * lctx);
    bool is_swa() const { return swa; }

    // Reference-faithful staging keeps one incomplete 128-token group lossless.
    // Use n_ubatch <= KVAR_N_GROUP so completed groups are not flushed before
    // queries from the same physical ubatch have consumed them.
    //   non-SWA tail_groups = 1
    //   SWA tail_groups     = KVAR_N_SWA_TAIL_GROUPS
    //   stage_groups        = tail_groups + 1 for non-SWA, tail_groups for SWA
    // The +1 is only the permanent sink slot for non-SWA. SWA has no sink slot,
    // so all F16 stage groups are live local tail groups. Its record ring also
    // carries the active ubatch span because early query rows can still attend
    // older window groups after later rows have flushed newer groups into the
    // ring. KVarN cache state is versioned to carry stage_groups/tail_groups so
    // restore can validate the saved layout against the current cache. The W2
    // gate rejects any layout mismatch; remap for differing save/restore ubatch
    // is future.
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

private:
    struct layer {
        uint32_t il;
        uint32_t n_head_kv;
        uint32_t head_dim_k;
        uint32_t head_dim_v;
        uint32_t k_slices;
        uint32_t v_slices;
        ggml_tensor * k_records;
        ggml_tensor * v_records;
        ggml_tensor * k_stage;
        ggml_tensor * v_stage;
        std::vector<ggml_tensor *> k_records_stream;
        std::vector<ggml_tensor *> v_records_stream;
        std::vector<ggml_tensor *> k_stage_stream;
        std::vector<ggml_tensor *> v_stage_stream;
    };

    const layer & layer_for(int32_t il) const;
    bool can_remove(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;
    void copy_kvarn_stream(uint32_t stream_src, uint32_t stream_dst);

    const llama_hparams & hparams;
    const llama_kvarn_params params;
    const uint32_t n_stream;
    // Declaration order matters: stage_groups depends on tail_groups, so
    // tail_groups must be declared first (C++ initializes members in order).
    const uint32_t tail_groups;   // non-SWA scheduler span; SWA fixed local tail
    const uint32_t stage_groups;   // F16 stage depth (non-SWA sink + tail; SWA tail only)
    const bool swa;
    const uint32_t n_groups_per_stream;

    std::unique_ptr<llama_kv_cache> metadata;
    std::vector<layer> layers;
    std::unordered_map<int32_t, int32_t> map_layer_ids;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;
    llama_kv_cache::stream_copy_info pending_stream_copies;
};
