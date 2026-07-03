#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

struct dflash_tape_gpu;
struct dflash_hidden_gpu;

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool flash_attn_required;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    llama_kvarn_params kvarn = {
        /*.type                =*/ LLAMA_KVARN_TYPE_DISABLED,
        /*.key_bits            =*/ 0,
        /*.value_bits          =*/ 0,
        /*.swa_key_bits        =*/ 0,
        /*.swa_value_bits      =*/ 0,
        /*.group               =*/ 128,
        /*.sinkhorn_iters      =*/ 16,
        /*.sink_tokens         =*/ 128,
        /*.fail_if_unsupported =*/ true,
    };

    // DFlash: target layer indices to capture hidden states from (empty = disabled)
    std::vector<int> dflash_capture_layers;

    // DFlash: drafter sampling temperature (0 = greedy argmax, >0 = Gumbel sampling)
    float dflash_sample_temp = 0.0f;

    // DFlash: top-K candidates per position (1 = argmax only, >1 = tree branching)
    int dflash_topk = 1;

    // DFlash target verifier: emit compact per-output logits for the server fast path.
    // top_k = 1 means greedy argmax. top_k > 1 emits top-K ids plus raw logits
    // so the server can run the sampler chain on K candidates instead of
    // copying the full vocabulary row to the CPU.
    bool dflash_verify_logits = false;
    int  dflash_verify_topk = 1;
    bool dflash_reduced_consumer_active = false;

    // DFlash: cross-attention window in tokens (how many target hidden states the drafter sees)
    int dflash_cross_ctx = 512;

    // DFlash drafter: number of concurrent slots the batched drafter graph is reserved
    // for. ctx_len in the drafter graph = dflash_n_slots * dflash_cross_ctx,
    // and drafter n_tokens reservation = dflash_n_slots * block_size. Set on the
    // drafter context (not the target) via llama_set_dflash_n_slots(). Default 1
    // (single-slot) so the drafter graph stays narrow when no batching is configured.
    // Capped at LLAMA_DFLASH_MAX_SLOTS.
    int dflash_n_slots = 1;

    // GPU-resident tape for DeltaNet rollback (graph writes directly, no eval callback sync).
    // tape_gpu is non-null when GPU tape is enabled (backward compat sentinel).
    dflash_tape_gpu * tape_gpu = nullptr;

    // Per-seq tape pointers for multi-seq verify batching.
    // tape_gpu_seqs[s] = tape for ubatch seq index s (0..tape_gpu_n_seqs-1).
    // Populated by the decode loop before each process_ubatch().
    dflash_tape_gpu * tape_gpu_seqs[LLAMA_DFLASH_MAX_SLOTS] = {};
    int tape_gpu_n_seqs = 0;

    // DFlash GPU-resident hidden capture for verifier decodes. When non-empty,
    // supported target graphs copy l_out tensors directly to these per-slot buffers.
    dflash_hidden_gpu * hidden_gpu_seqs[LLAMA_DFLASH_MAX_SLOTS] = {};
    int hidden_gpu_n_seqs = 0;

    // DFlash GPU-resident hidden staging for suffix-prefill capture. Larger than
    // hidden_gpu_seqs (sized for one ubatch) and used only during prefill when
    // capture_active is true and the ubatch exceeds LLAMA_DFLASH_MAX_VERIFY_TOKENS.
    // When prefill_gpu_n_seqs > 0, hidden_gpu_n_seqs is 0 and vice versa.
    dflash_hidden_gpu * prefill_gpu_seqs[LLAMA_DFLASH_MAX_SLOTS] = {};
    int prefill_gpu_n_seqs = 0;

    bool    dflash_prefill_capture_active = false;
    int32_t dflash_prefill_src_offset = 0;
    int32_t dflash_prefill_dst_offset = 0;
    int32_t dflash_prefill_n_tokens   = 0;
    int32_t dflash_prefill_src_offsets[LLAMA_DFLASH_MAX_SLOTS] = {};
    int32_t dflash_prefill_dst_offsets[LLAMA_DFLASH_MAX_SLOTS] = {};
    int32_t dflash_prefill_n_tokens_seqs[LLAMA_DFLASH_MAX_SLOTS] = {};

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
