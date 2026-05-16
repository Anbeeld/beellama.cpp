#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-arch.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <unordered_map>
#include <vector>

static inline bool llama_dflash_gpu_hidden_supported_arch(llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_GEMMA4:
            return true;
        default:
            return false;
    }
}

static inline bool llama_dflash_gpu_tape_supported_arch(llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
            return true;
        default:
            return false;
    }
}

// DFlash hidden capture is indexed by unique logical seq/slot.
// In single-slot server decode, the ubatch may represent tokens as multiple
// internal lanes with n_seq_tokens == 1 while the logical DFlash slot
// owns the whole ubatch. For capture sizing, single unique-seq means
// "copy all ubatch tokens per slot".
static inline int llama_dflash_capture_tokens_per_seq(uint32_t n_tokens, uint32_t n_seq_tokens, uint32_t n_seqs_unq) {
    return n_seqs_unq > 1 ? (int) n_seq_tokens : (int) n_tokens;
}

static inline bool llama_dflash_suppress_callback_for_prefill_ubatch_for_test(
        bool prefill_plan_active,
        bool use_prefill_staging,
        bool has_intersection) {
    return prefill_plan_active && use_prefill_staging && !has_intersection;
}

struct llama_memory_recurrent;

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// DFlash: hidden state buffer for captured layer activations
struct dflash_layer_hidden_buf {
    std::vector<float> data;
    int64_t n_embd = 0;
    int64_t n_tokens = 0;
};

// DFlash: tape recording data for one recurrent layer
struct dflash_tape_layer {
    std::vector<float> k;          // [S_k * H_k * n_tokens] after l2_norm
    std::vector<float> v;          // [S_v * H_v * n_tokens]
    std::vector<float> gate;       // [H_v * n_tokens] pre-exp
    std::vector<float> beta;       // [H_v * n_tokens] pre-sigmoid
    std::vector<float> qkv_mixed;  // [conv_channels * n_tokens * n_seqs] for conv state rebuild
    int64_t S_k = 0, H_k = 0, S_v = 0, H_v = 0;
    int64_t conv_channels = 0;
    int n_tokens = 0;
    // per-seq metadata for multi-seq verify QKV scatter
    int n_seqs = 1;
    llama_seq_id seq_ids[LLAMA_DFLASH_MAX_SLOTS] = {};
};

// GPU-resident tape: persistent tensors that the graph writes into directly (no eval callback sync)
struct dflash_tape_gpu_layer {
    ggml_tensor * k    = nullptr;  // [S_k, H_k, max_tokens]
    ggml_tensor * v    = nullptr;  // [S_v, H_v, max_tokens]
    ggml_tensor * gate = nullptr;  // [1, H_v, max_tokens]
    ggml_tensor * beta = nullptr;  // [1, H_v, max_tokens]
    ggml_tensor * qkv  = nullptr;  // [conv_channels, max_tokens]
};

struct dflash_tape_gpu {
    std::vector<dflash_tape_gpu_layer> layers;  // one per recurrent layer
    std::vector<int32_t> layer_ids;             // model layer indices → tape index mapping
    ggml_backend_buffer_t buf = nullptr;
    ggml_context * ctx = nullptr;               // owns the tensor descriptors
    int max_tokens = 0;                         // allocated capacity
    int n_tokens = 0;                           // actual tokens recorded this pass

    ~dflash_tape_gpu() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

struct dflash_hidden_gpu {
    std::vector<ggml_tensor *> layers;  // one [n_embd, max_tokens] tensor per captured layer
    std::vector<int32_t> layer_ids;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context * ctx = nullptr;
    int64_t n_embd = 0;
    int max_tokens = 0;
    int n_tokens = 0;

    ~dflash_hidden_gpu() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

enum dflash_tape_type {
    DFLASH_TAPE_K    = 0,
    DFLASH_TAPE_V    = 1,
    DFLASH_TAPE_GATE = 2,
    DFLASH_TAPE_BETA = 3,
    DFLASH_TAPE_QKV  = 4,
};

// DDTree: tree attention mask for verification
struct llama_tree_mask {
    bool active = false;
    int n_tree_tokens = 0;         // number of tree tokens (root + nodes)
    std::vector<uint8_t> visibility;  // [n² row-major] true = can attend
};

struct dflash_prefill_capture_plan {
    bool active = false;

    llama_seq_id seq_id = -1;

    int32_t capture_begin = 0;
    int32_t capture_end   = 0;

    int32_t n_tokens = 0;

    int32_t n_written = 0;
};

// DFlash: eval callback data for hidden state capture + tape recording
struct dflash_capture_data {
    // Logical per-view capture gate. When false, the eval callback is not
    // installed and graph builds skip hidden outputs. Layer IDs, GPU buffers,
    // tape metadata, and profile counters are preserved across toggles.
    bool capture_active = true;

    // hidden state capture (for drafter conditioning)
    std::vector<int32_t> layer_ids;           // layer indices to capture
    std::vector<std::string> tensor_names;    // pre-formatted "l_out-{id}" names
    std::unordered_map<std::string, size_t> hidden_name_idx; // name → index for O(1) lookup
    // pointer to context's layer_hiddens (outer: per-slot, inner: per-captured-layer)
    std::vector<std::vector<dflash_layer_hidden_buf>> * hiddens;

    // tape recording (for DeltaNet state rollback)
    bool tape_enabled = false;
    bool gpu_capture_enabled = true;
    std::vector<int32_t> recurrent_layer_ids;       // model layer indices that are DeltaNet
    std::unordered_map<std::string, std::pair<int, int>> tape_name_map;  // name → (layer_idx, type)
    std::vector<dflash_tape_layer> tape_layers;     // one per recurrent layer (CPU fallback)

    // GPU-resident tape: graph writes directly to these tensors (no eval callback sync).
    // One entry per slot for multi-slot DFlash (see --spec-dflash-max-slots). For single-slot
    // (default), `tapes` has size 1 and `active_tape_idx` is always 0 — behavior is
    // byte-identical to the pre-multi-slot singleton.
    std::vector<std::unique_ptr<dflash_tape_gpu>> tapes;
    std::vector<std::unique_ptr<dflash_hidden_gpu>> hidden_gpu;
    std::vector<std::unique_ptr<dflash_hidden_gpu>> prefill_gpu;  // large staging buffer for suffix prefill
    int prefill_gpu_max_tokens = 0;  // allocation capacity of prefill_gpu
    dflash_prefill_capture_plan prefill_plan;  // active capture window plan
    int active_tape_idx = 0;

    // Active ubatch for the in-flight process_ubatch() call. The eval callback
    // reads ubatch->n_seqs_unq / ubatch->seq_id to route hidden-state captures
    // to layer_hiddens[seq] (per-token scatter under multi-seq ubatches).
    // ggml's scheduler serializes callbacks within a graph compute, so this
    // pointer is safe to read without synchronization.
    const llama_ubatch * ubatch = nullptr;

    // Reused scratch for the multi-seq scatter path (avoid per-ubatch alloc).
    std::vector<float> scatter_buf;

    // Opt-in DFlash profiling (GGML_DFLASH_PROFILE=1).
    bool profile = false;
    bool multi_gpu_capture_fallback_logged = false;
    bool multi_gpu_replay_fallback_logged = false;
    uint64_t profile_decode_us = 0;
    uint64_t profile_raw_logits_us = 0;
    uint64_t profile_raw_logits_bytes = 0;
    uint64_t profile_reduced_logits_us = 0;
    uint64_t profile_reduced_logits_bytes = 0;
    uint64_t profile_cb_ask = 0;
    uint64_t profile_cb_hidden_ask = 0;
    uint64_t profile_cb_tape_ask = 0;
    uint64_t profile_cb_qkv_ask = 0;
    uint64_t profile_cb_read = 0;
    uint64_t profile_cb_hidden_read = 0;
    uint64_t profile_cb_tape_read = 0;
    uint64_t profile_cb_qkv_read = 0;
    uint64_t profile_replay_wait_us = 0;
    uint64_t profile_conv_gpu_us = 0;
    uint64_t profile_conv_read_wait_us = 0;
    uint64_t profile_conv_cpu_us = 0;
    uint64_t profile_conv_write_wait_us = 0;
    std::unordered_map<std::string, uint64_t> profile_cb_names;

    dflash_tape_gpu * active_tape() const {
        return (active_tape_idx >= 0 && active_tape_idx < (int) tapes.size())
                   ? tapes[active_tape_idx].get()
                   : nullptr;
    }

    dflash_hidden_gpu * active_hidden_gpu() const {
        return (active_tape_idx >= 0 && active_tape_idx < (int) hidden_gpu.size())
                   ? hidden_gpu[active_tape_idx].get()
                   : nullptr;
    }

    std::vector<dflash_layer_hidden_buf> * slot_hiddens(int slot) const {
        if (!hiddens || slot < 0 || slot >= (int) hiddens->size()) {
            return nullptr;
        }
        return &(*hiddens)[slot];
    }

    std::vector<dflash_layer_hidden_buf> * active_slot_hiddens() const {
        return slot_hiddens(active_tape_idx);
    }

    // persistent GPU buffer for tape replay (avoids per-call alloc/free)
    ggml_backend_buffer_t replay_buf = nullptr;
    size_t replay_buf_size = 0;

    // S2: pre-allocated zeros buffer for Q input (avoids per-call alloc+zero)
    std::vector<float> replay_zeros;

    // async tape replay state (GDN launched, waiting for sync before conv rebuild)
    bool replay_pending = false;
    ggml_backend_t replay_gpu_backend = nullptr;
    ggml_context * replay_graph_ctx = nullptr;
    ggml_backend_event_t replay_event = nullptr;  // P2-11: CUDA event for fine-grained sync
    bool replay_direct_gpu = false;
    const void * replay_sync_ptr = nullptr;
    std::vector<const void *> replay_sync_ptrs;
    int replay_n_accepted = 0;
    int32_t replay_cell_idx = -1;
    llama_seq_id replay_seq_id = 0;
    llama_memory_recurrent * replay_mem_recurrent = nullptr;

    ~dflash_capture_data() {
        if (replay_graph_ctx) {
            ggml_free(replay_graph_ctx);
        }
        if (replay_buf) {
            ggml_backend_buffer_free(replay_buf);
        }
        if (replay_event) {
            ggml_backend_event_free(replay_event);
        }
    }
};

struct dflash_kv_cache_data {
    llama_dflash_kv_cache_view view;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_buffer_t update_buf = nullptr;
    size_t update_buf_size = 0;

    std::vector<ggml_tensor *> k_ring;
    std::vector<ggml_tensor *> v_ring;

    int ring_size = 0;
    int write_pos = 0;
    int n_filled = 0;
    int n_elem = 0;

    using write_d2d_fn_t = bool (*)(void *, const void *, int, int, int, int);
    using append_d2d_fn_t = bool (*)(void *, const void *, int, int, int, int);
    using prepare_ptr_fn_t = bool (*)(const void *);
    using sync_ptr_fn_t = bool (*)(const void *);
    using interleave_fn_t = bool (*)(const void *, void *, int, int, int, int, int);
    write_d2d_fn_t fn_write_d2d = nullptr;
    append_d2d_fn_t fn_append_d2d = nullptr;
    append_d2d_fn_t fn_append_d2d_no_check = nullptr;
    prepare_ptr_fn_t fn_prepare_ptr = nullptr;
    sync_ptr_fn_t fn_sync_ptr = nullptr;
    interleave_fn_t fn_interleave = nullptr;

    ~dflash_kv_cache_data() {
        if (update_buf) {
            ggml_backend_buffer_free(update_buf);
        }
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};
struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    int32_t * get_logits_argmax();
    int32_t * get_logits_argmax_ith(int32_t i);
    int32_t   get_logits_argmax_n();
    int32_t   get_logits_argmax_k();
    float   * get_logits_argmax_probs();  // log-probs of top-K tokens (when temp > 0)
    float   * get_logits_argmax_probs_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_causal_attn(bool value);
    void set_warmup(bool value);
    bool resize_recurrent_memory(uint32_t new_n_seq_max, bool expand);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

    // DFlash hidden state accessors
    float * get_layer_hidden(int layer_idx);
    int64_t get_layer_hidden_n_tokens(int layer_idx) const;
    int64_t get_layer_hidden_n_embd(int layer_idx) const;
    int32_t get_n_layer_hiddens() const;

    // DFlash: configure hidden state capture layers
    void set_dflash_capture(const int32_t * layer_ids, int32_t n_layers);
    void set_dflash_capture_active(bool active);
    void set_dflash_gpu_capture(bool enabled);
    void set_dflash_sample_temp(float temp);
    void set_dflash_topk(int k);
    void set_dflash_verify_logits(bool enabled, int top_k);
    void set_dflash_consume_reduced(bool enabled);
    void set_dflash_n_slots(int n);

    // DFlash: reset hidden-state capture for a fresh decode() call so the
    // eval callback accumulates across this call's ubatches
    void dflash_reset_hidden_capture();

    // DFlash: enable/disable tape recording for DeltaNet state rollback
    void set_tape_recording(bool enable);
    void dflash_ensure_recurrent_setup();

    // DFlash: allocate GPU-resident tape buffer for graph-embedded recording.
    // n_slots > 1 allocates per-slot buffers so concurrent slots (llama-server -np > 1)
    // don't clobber each other's tape entries. The single-arg overload keeps legacy
    // callers single-slot.
    void allocate_tape_gpu(int max_tokens) { allocate_tape_gpu(1, max_tokens); }
    void allocate_tape_gpu(int n_slots, int max_tokens);
    void allocate_hidden_gpu(int n_slots, int max_tokens);
    // DFlash: allocate per-slot GPU staging buffers for suffix-prefill hidden capture.
    // Unlike hidden_gpu (sized for LLAMA_DFLASH_MAX_VERIFY_TOKENS = 25), this is sized
    // for one ubatch worth of tokens (typically 128-512). Lazily allocated on first
    // suffix prefill. Returns true if allocation succeeded or buffers already exist.
    bool allocate_prefill_gpu(int n_slots, int max_tokens);

    // DFlash: select which slot's tape the next llama_decode() writes into.
    // Must be called before each decode when multi-slot tape is in use.
    // No-op when n_slots == 1. Invalidates graph reuse if the slot changes.
    void set_active_dflash_slot(int slot_idx);

    // DFlash: replay tape data to reconstruct DeltaNet state for n_accepted tokens
    void tape_replay(llama_seq_id seq_id, int n_accepted);
    void tape_replay_sync();
    bool tape_replay_conv_gpu(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted);
    bool tape_replay_conv_gpu(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted, bool advance_pos);
    bool tape_replay_gdn_direct_gpu(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted);
    bool tape_replay_gdn_direct_from_cpu_tape(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted);
    bool tape_replay_conv_gpu_from_cpu_tape(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted, llama_seq_id seq_id);
    void tape_replay_conv(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted, llama_seq_id seq_id = 0);
    void tape_replay_cpu(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted);

    // DFlash: complete rollback for hybrid models (KV trim + recurrent restore + tape replay)
    void dflash_rollback(llama_seq_id seq_id, llama_seq_id seq_backup, int n_past_before, int n_accepted);

    // DFlash: prepare DeltaNet state for branch verification (recurrent restore + tape replay, no KV touch)
    void dflash_prepare_branch(llama_seq_id seq_id, llama_seq_id seq_backup, int depth);

    // DFlash: set cross data for drafter context
    void set_cross_data(const float * data, int64_t n_embd, int64_t n_tokens);

    // DFlash multi-slot: stash cross data keyed by seq_id. Multiple slots can
    // each set their own buffer before a batched drafter decode. seq_id < 0
    // routes to the legacy single-slot path (set_cross_data).
    void set_cross_data_seq(llama_seq_id seq_id, const float * data, int64_t n_embd, int64_t n_tokens);

    // DFlash GPU ring: allocate ring on GPU backend, returns opaque handle
    void * init_cross_ring_gpu(int n_layers, int n_embd, int ring_size);
    bool cross_ring_gpu_write_hidden(void * handle, int layer, int ring_pos, int src_offset, int n_tokens, int n_embd);
    // DFlash GPU: copy from prefill staging buffer into the cross ring (D2D).
    // slot: which DFlash slot's prefill buffer to read from.
    bool prefill_gpu_write_hidden(void * handle, int slot, int layer, int ring_pos, int src_offset, int n_tokens, int n_embd);

    // DFlash GPU ring: set GPU device pointer as cross data source (D2D path)
    using set_tensor_d2d_fn_t = void (*)(void *, const void *, size_t, size_t);
    void set_cross_data_gpu(llama_seq_id seq_id, const void * d_staging, int cross_len,
                            int n_layers, int n_embd, set_tensor_d2d_fn_t fn_d2d);

    bool dflash_kv_cache_init(int ctx_size);
    void dflash_kv_cache_reset();
    bool dflash_kv_cache_update(int n_tokens);
    bool dflash_kv_cache_update_gpu(const void * d_hidden, int n_tokens, int n_layers, int n_embd_layer, set_tensor_d2d_fn_t fn_d2d);
    bool dflash_kv_cache_prepare(int ctx_window);

    // DDTree: set/clear tree attention mask for verification
    void set_tree_mask(const uint8_t * visibility, int n_tree_tokens);
    void clear_tree_mask();

    // DDTree: tree-mode parent IDs for SSM kernels
    void set_tree_parent_ids(const int32_t * parents, int n_tokens);
    void clear_tree_parent_ids();

    // DFlash: allocate persistent intermediate buffers for tree verify
    void allocate_tree_buffers(int max_tree_tokens);

    // DDTree: rollback SSM state to accepted token from intermediates
    void tree_rollback(llama_seq_id seq_id, llama_seq_id seq_backup, int commit_n, const int32_t * parents);
    void set_tree_seq0_count(int n) { tree_bufs.n_seq0_tokens = n; }

    void dflash_prefill_capture_begin(llama_seq_id seq_id, int32_t capture_begin, int32_t capture_end);
    void dflash_prefill_capture_end();
    bool dflash_prefill_capture_info(llama_seq_id seq_id, int32_t * n_tokens, int32_t * n_written) const;

    // DFlash: check if prefill GPU staging buffers have captured data from the last decode.
    bool prefill_gpu_active() const {
        if (!dflash_capture) return false;
        for (auto & pf : dflash_capture->prefill_gpu) {
            if (pf && pf->n_tokens > 0) return true;
        }
        return false;
    }

    // DFlash: get the number of tokens captured into the prefill GPU staging
    // buffer for the given slot. Returns 0 if slot is invalid or not active.
    int64_t prefill_gpu_n_tokens(int slot) const {
        if (!dflash_capture || slot < 0 || slot >= (int) dflash_capture->prefill_gpu.size()) return 0;
        const auto & plan = dflash_capture->prefill_plan;
        if (plan.seq_id == slot && plan.n_written > 0) {
            return std::min(plan.n_written, plan.n_tokens);
        }
        auto * pf = dflash_capture->prefill_gpu[slot].get();
        return pf ? pf->n_tokens : 0;
    }

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;
    llm_graph_cb graph_get_cb_name_only() const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<llama_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // GPU argmax/topk results (1-dimensional: [K * n_outputs])
    std::vector<int32_t> logits_argmax_buf;
    std::vector<float>   logits_argmax_prob_buf;  // log-probs of top-K tokens (when temp > 0)
    int32_t logits_argmax_count = 0;
    int32_t logits_argmax_k = 1;  // K value (1 = argmax, >1 = top-K)

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // DFlash: captured hidden states (outer: per-slot matching dflash_capture->tapes,
    // inner: per-captured-layer). Single-slot default is 1 × n_capture_layers.
    std::vector<std::vector<dflash_layer_hidden_buf>> layer_hiddens;

    std::unique_ptr<dflash_capture_data> dflash_capture;

    // DDTree: tree attention mask (set before verification decode, cleared after)
    llama_tree_mask tree_mask;

    // DDTree: tree-mode parent IDs and persistent SSM intermediate buffers
    struct {
        bool active = false;
        bool disabled = false;                     // multi-GPU: tree ops can't span devices
        int n_tokens = 0;
        int n_seq0_tokens = 0;                     // tokens on seq_id=0 in last batch (for pos rollback)
        std::vector<int32_t> parent_ids_cpu;       // [max_tree_tokens] host copy
        ggml_backend_buffer_t buffer = nullptr;    // GPU allocation for all intermediates + parent_ids
        ggml_context * ggml_ctx = nullptr;         // context owning tensor metadata
        ggml_tensor * parent_ids_gpu = nullptr;    // [max_tree_tokens] i32 on GPU
        std::vector<ggml_tensor *> ssm_intermediates; // per recurrent layer: [S_v*S_v*H*max_tokens] f16
        int max_tree_tokens = 0;
    } tree_bufs;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    bool dflash_kv_cache_multi_gpu_fallback_logged = false;
    std::unique_ptr<dflash_kv_cache_data> dflash_kv_cache;

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
