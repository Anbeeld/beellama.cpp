#pragma once

#include "llama-kv-cache.h"

#include <vector>

//
// llama_kv_cache_iswa
//

// utilizes two instances of llama_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

class llama_kv_cache_iswa : public llama_memory_i {
public:
    llama_kv_cache_iswa(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_batch,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
          llama_kvarn_params   kvarn = llama_kvarn_default_params(),
                     uint32_t   tail_tokens = 0,
                     uint32_t   tail_tokens_swa = 0,
                    ggml_type   tail_type = GGML_TYPE_F16,
                     uint32_t   tail_tokens_requested = UINT32_MAX,
                     uint32_t   tail_tokens_swa_requested = UINT32_MAX,
                     uint32_t   tail_rollback_tokens = 0,
                         bool   tail_native_exact_swa = false);

    // DSV4 uses a projected hparams view for its raw iSWA cache.  Keep this
    // explicit overload so KVarN support does not erase that upstream need.
    llama_kv_cache_iswa(
            const llama_model & model,
            const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_batch,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
          llama_kvarn_params   kvarn = llama_kvarn_default_params(),
                     uint32_t   tail_tokens = 0,
                     uint32_t   tail_tokens_swa = 0,
                    ggml_type   tail_type = GGML_TYPE_F16,
                     uint32_t   tail_tokens_requested = UINT32_MAX,
                     uint32_t   tail_tokens_swa_requested = UINT32_MAX,
                     uint32_t   tail_rollback_tokens = 0,
                         bool   tail_native_exact_swa = false);

    ~llama_kv_cache_iswa() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    uint32_t get_kv_n_stream() const override;
    llama_memory_context_ptr init_kv_batch(const std::vector<llama_ubatch> & ubatches) override;

    bool get_can_shift() const override;
    seq_rm_capability get_seq_rm_capability() const override;

    void clear(bool data) override;

    bool can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const override;
    bool seq_rm_plan(
            llama_seq_id seq_id, llama_pos p0, llama_pos p1,
            llama_pos & planned_p0, llama_pos & planned_p1) const override;
    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    bool seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) override;

    int cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) override;

    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    llama_kv_memory_stats kv_memory_stats() const override;
    ggml_type get_kv_tail_type() const override;
    uint32_t get_kv_tail_group_count() const override;
    bool get_kv_tail_coverage(uint32_t group_index, llama_seq_id seq_id,
            llama_kv_tail_coverage_info & out) const override;
    void reset_kv_tail_planner_timing() override;
    uint64_t get_kv_tail_planner_timing_ns() const override;

    // state write/load

    bool requires_state_for_partial_restore() const override;
    bool state_seq_can_save(llama_seq_id seq_id) const override;
    bool state_seq_can_restore(llama_seq_id seq_id) const override;
    bool state_seq_can_save(llama_seq_id seq_id, llama_state_seq_flags flags) const override;
    bool state_seq_can_restore(llama_seq_id seq_id, llama_state_seq_flags flags) const override;
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache_iswa specific API
    //

    llama_memory_i * get_base() const;
    llama_memory_i * get_swa () const;

private:
    const bool unified;

    std::unique_ptr<llama_memory_i> kv_base;
    std::unique_ptr<llama_memory_i> kv_swa;
};

class llama_kv_cache_iswa_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    llama_kv_cache_iswa_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv);

    // used to create an update context
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv,
            llama_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    llama_kv_cache_iswa_context(
            llama_memory_context_ptr ctx_base_in,
            llama_memory_context_ptr ctx_swa_in,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_iswa_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;
    void graph_compute_start() override;
    void graph_compute_finish(ggml_status status) override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_iswa_context specific API
    //

    const llama_kv_cache_context * get_base() const;
    const llama_kv_cache_context * get_swa()  const;

private:
    //llama_kv_cache_iswa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_base;
    const llama_memory_context_ptr ctx_swa;

    const llama_memory_status status;
};
