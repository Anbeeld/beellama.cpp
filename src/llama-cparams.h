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
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    enum llama_pooling_type pooling_type;

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

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};
