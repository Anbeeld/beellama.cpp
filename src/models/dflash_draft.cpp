#include "models.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <vector>

// Max cross-attention context for DFlash drafter (caps VRAM growth).
// Override with GGML_DFLASH_MAX_CTX env var. 0 = unlimited.
static int64_t dflash_max_cross_ctx() {
    static const int64_t val = [] {
        const char * e = getenv("GGML_DFLASH_MAX_CTX");
        return e ? (int64_t) atoi(e) : (int64_t) 4096;
    }();
    return val;
}

static int64_t dflash_draft_ctx_len(const llama_cross * cross, const llama_cparams & cparams) {
    const int n_slots = std::clamp(cparams.dflash_n_slots, 1, (int) LLAMA_DFLASH_MAX_SLOTS);
    const int64_t per_slot_ctx = cparams.dflash_cross_ctx > 0 ? cparams.dflash_cross_ctx : (int64_t) LLAMA_DFLASH_PER_SLOT_CTX;

    if (n_slots == 1) {
        int64_t ctx_len = (cross && cross->n_enc > 0) ? cross->n_enc : per_slot_ctx;
        const int64_t max_ctx = dflash_max_cross_ctx();
        if (max_ctx > 0 && ctx_len > max_ctx) {
            ctx_len = max_ctx;
        }
        if (ctx_len > per_slot_ctx) {
            ctx_len = per_slot_ctx;
        }
        return ctx_len;
    }

    return (int64_t) n_slots * per_slot_ctx;
}

enum dflash_kv_cache_mode {
    DFLASH_KV_CACHE_OFF,
    DFLASH_KV_CACHE_BOTH,
    DFLASH_KV_CACHE_K_ONLY,
    DFLASH_KV_CACHE_V_ONLY,
};

static dflash_kv_cache_mode dflash_kv_cache_mode_env() {
    static const dflash_kv_cache_mode mode = [] {
        const char * mode_env = getenv("GGML_DFLASH_KV_CACHE_MODE");
        if (!mode_env || mode_env[0] == '\0') {
            return DFLASH_KV_CACHE_BOTH;
        }
        if (std::strcmp(mode_env, "off") == 0 ||
                std::strcmp(mode_env, "none") == 0 ||
                std::strcmp(mode_env, "disabled") == 0) {
            return DFLASH_KV_CACHE_OFF;
        }
        if (std::strcmp(mode_env, "k") == 0 ||
                std::strcmp(mode_env, "k-only") == 0) {
            return DFLASH_KV_CACHE_K_ONLY;
        }
        if (std::strcmp(mode_env, "v") == 0 ||
                std::strcmp(mode_env, "v-only") == 0) {
            return DFLASH_KV_CACHE_V_ONLY;
        }
        return DFLASH_KV_CACHE_BOTH;
    }();
    return mode;
}

static bool dflash_kv_cache_ready_for_window(const llama_cross * cross, int64_t ctx_len) {
    const auto * kv_cache = cross ? cross->dflash_kv_cache : nullptr;
    if (!kv_cache || dflash_kv_cache_mode_env() == DFLASH_KV_CACHE_OFF) {
        return false;
    }

    const int64_t n_cache_needed = cross ? std::min((int64_t) cross->n_enc_real, ctx_len) : 0;
    if (kv_cache->ctx_len < ctx_len ||
            kv_cache->n_filled < n_cache_needed ||
            kv_cache->ring_size < ctx_len ||
            kv_cache->n_layers <= 0 ||
            (int) kv_cache->k_ring.size() < kv_cache->n_layers ||
            (int) kv_cache->v_ring.size() < kv_cache->n_layers) {
        return false;
    }

    for (int il = 0; il < kv_cache->n_layers; ++il) {
        if (kv_cache->k_ring[il] == nullptr || kv_cache->v_ring[il] == nullptr) {
            return false;
        }
    }

    return true;
}

// DFlash drafter custom graph input
// Holds the target hidden states, context positions, and asymmetric non-causal attention mask
class llm_graph_input_dflash : public llm_graph_input_i {
public:
    llm_graph_input_dflash(const llama_cross * cross, int64_t ctx_len, int64_t n_block, uint32_t n_swa, bool use_kv_cache)
        : cross(cross), ctx_len(ctx_len), n_block(n_block), use_kv_cache(use_kv_cache), n_swa(n_swa) {}

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * target_hidden     = nullptr; // [n_target_features, ctx_len]
    ggml_tensor * pos_ctx           = nullptr; // [ctx_len]
    ggml_tensor * pos_q_rebased     = nullptr; // [n_block]
    ggml_tensor * kq_mask           = nullptr; // [ctx_len + n_block, n_block, 1, 1]
    ggml_tensor * kq_mask_cnv       = nullptr;
    // Only allocated when hparams.is_swa_any(); same shape as kq_mask
    ggml_tensor * kq_mask_swa       = nullptr;
    ggml_tensor * kq_mask_swa_cnv   = nullptr;

    const llama_cross * cross;
    int64_t ctx_len;
    int64_t n_block;
    bool use_kv_cache;
    uint32_t n_swa;
};

class llm_graph_input_dflash_update : public llm_graph_input_i {
public:
    llm_graph_input_dflash_update(const llama_cross * cross, int64_t n_tokens)
        : cross(cross), n_tokens(n_tokens) {}

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * target_hidden = nullptr; // [n_target_features, n_tokens]

    const llama_cross * cross;
    int64_t n_tokens;
};

bool llm_graph_input_dflash::can_reuse(const llm_graph_params & params) {
    if (params.cross != cross) {
        return false;
    }
    if ((int64_t) params.ubatch.n_tokens != n_block) {
        return false;
    }
    if (params.hparams.n_swa != n_swa) {
        return false;
    }
    if (dflash_draft_ctx_len(params.cross, params.cparams) != ctx_len) {
        return false;
    }
    if (dflash_kv_cache_ready_for_window(params.cross, ctx_len) != use_kv_cache) {
        return false;
    }

    const int n_seqs = params.ubatch.n_seqs_unq > 1 ? (int) params.ubatch.n_seqs_unq : 1;
    if (n_seqs > LLAMA_DFLASH_MAX_SLOTS) {
        return false;
    }
    if (n_seqs > 1 && (ctx_len % n_seqs != 0 || n_block % n_seqs != 0)) {
        return false;
    }

    return true;
}

bool llm_graph_input_dflash_update::can_reuse(const llm_graph_params & params) {
    return params.cross == cross && (int64_t) params.ubatch.n_tokens == n_tokens;
}

void llm_graph_input_dflash_update::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (!target_hidden) {
        return;
    }

    const float * src_data = nullptr;
    const void *  src_gpu  = nullptr;
    auto          fn_d2d   = cross ? cross->fn_set_tensor_d2d : nullptr;
    int64_t       src_real = 0;
    int64_t       n_feat   = 0;

    if (cross) {
        if (cross->dflash_kv_update_gpu) {
            n_feat = cross->dflash_kv_update_n_embd;
            src_gpu = cross->dflash_kv_update_gpu;
            src_real = cross->dflash_kv_update_n_enc_real;
            fn_d2d = cross->dflash_kv_update_fn_set_tensor_d2d;
        } else if (cross->v_embd_gpu) {
            n_feat = cross->n_embd;
            src_gpu = cross->v_embd_gpu;
            src_real = cross->v_embd_gpu_n_enc_real;
        } else if (!cross->v_embd.empty()) {
            n_feat = cross->n_embd;
            src_data = cross->v_embd.data();
            src_real = cross->n_enc_real;
        }
    }

    const int64_t n_copy = std::min(src_real, n_tokens);
    const size_t copy_bytes = (size_t) n_feat * (size_t) n_copy * sizeof(float);
    const size_t tensor_bytes = ggml_nbytes(target_hidden);

    if (n_copy > 0 && n_feat == target_hidden->ne[0] && (src_gpu || src_data)) {
        const size_t actual_bytes = std::min(copy_bytes, tensor_bytes);
        if (src_gpu && fn_d2d) {
            fn_d2d(target_hidden->data, src_gpu, 0, actual_bytes);
        } else {
            ggml_backend_tensor_set(target_hidden, src_data, 0, actual_bytes);
        }
        if (actual_bytes < tensor_bytes) {
            ggml_backend_tensor_memset(target_hidden, 0, actual_bytes, tensor_bytes - actual_bytes);
        }
    } else {
        ggml_backend_tensor_memset(target_hidden, 0, 0, tensor_bytes);
    }
}

void llm_graph_input_dflash::set_input(const llama_ubatch * ubatch) {
    const int n_seqs = (ubatch && ubatch->n_seqs_unq > 1) ? (int) ubatch->n_seqs_unq : 1;

    if (n_seqs == 1) {
        // === Single-slot path ===
        // resolve cross data for the active seq (GPU path preferred, CPU fallback)
        const float * src_data  = nullptr;
        const void *  src_gpu   = nullptr;
        int64_t       src_n_enc  = 0;
        int64_t       src_n_real = 0;
        if (cross) {
            llama_seq_id active_seq = -1;
            if (ubatch && ubatch->n_seqs_unq > 0 && ubatch->seq_id_unq) {
                active_seq = ubatch->seq_id_unq[0];
            }
            if (active_seq >= 0) {
                auto it = cross->v_embd_per_seq.find(active_seq);
                if (it != cross->v_embd_per_seq.end()) {
                    if (it->second.v_embd_gpu) {
                        src_gpu    = it->second.v_embd_gpu;
                        src_n_enc  = it->second.n_enc;
                        src_n_real = it->second.v_embd_gpu_n_enc_real;
                    } else if (!it->second.v_embd.empty()) {
                        src_data   = it->second.v_embd.data();
                        src_n_enc  = it->second.n_enc;
                        src_n_real = it->second.n_enc_real;
                    }
                }
            }
            if (!src_data && !src_gpu) {
                if (cross->v_embd_gpu) {
                    src_gpu    = cross->v_embd_gpu;
                    src_n_enc  = cross->n_enc;
                    src_n_real = cross->v_embd_gpu_n_enc_real;
                } else if (!cross->v_embd.empty()) {
                    src_data   = cross->v_embd.data();
                    src_n_enc  = cross->n_enc;
                    src_n_real = cross->n_enc_real;
                }
            }
        }

        // Sliding window: if src has more tokens than ctx_len, take the most recent
        const int64_t src_real = src_n_real > 0 ? src_n_real : 0;
        const int64_t n_copy  = std::min(src_real, ctx_len);
        const int64_t win_off = (src_real > ctx_len) ? (src_real - ctx_len) : 0;

        if (target_hidden && target_hidden->buffer && target_hidden->data && (src_data || src_gpu) && n_copy > 0) {
            const int64_t n_feat = cross->n_embd;
            const size_t copy_bytes  = (size_t) n_feat * (size_t) n_copy * sizeof(float);
            const size_t tensor_bytes = ggml_nbytes(target_hidden);
            const size_t actual_bytes = std::min(copy_bytes, tensor_bytes);

            if (src_gpu && cross->fn_set_tensor_d2d) {
                // GPU D2D path
                const void * gpu_src = (const char *)src_gpu + (size_t)win_off * n_feat * sizeof(float);
                cross->fn_set_tensor_d2d(target_hidden->data, gpu_src, 0, actual_bytes);
            } else {
                // CPU H2D path (fallback)
                const float * src = src_data + win_off * n_feat;
                ggml_backend_tensor_set(target_hidden, src, 0, actual_bytes);
            }
            if (copy_bytes < tensor_bytes) {
                ggml_backend_tensor_memset(target_hidden, 0, copy_bytes, tensor_bytes - copy_bytes);
            }
        } else if (target_hidden && target_hidden->buffer && target_hidden->data) {
            ggml_backend_tensor_memset(target_hidden, 0, 0, ggml_nbytes(target_hidden));
        }

        const int64_t n_real = n_copy;

        if (pos_ctx && pos_ctx->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(pos_ctx->buffer));
            int32_t * data = (int32_t *) pos_ctx->data;
            for (int64_t i = 0; i < ctx_len; ++i) {
                data[i] = (i < n_real) ? (int32_t) i : 0;
            }
        }

        if (pos_q_rebased && pos_q_rebased->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(pos_q_rebased->buffer));
            int32_t * data = (int32_t *) pos_q_rebased->data;
            const bool have_pos = (ubatch != nullptr) && (ubatch->pos != nullptr)
                               && ((int64_t) ubatch->n_tokens >= n_block);
            for (int64_t q = 0; q < n_block; ++q) {
                data[q] = have_pos ? (int32_t) (ubatch->pos[q] - win_off) : (int32_t) (n_real + q);
            }
        }

        if (kq_mask && kq_mask->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask->buffer));
            float * data = (float *) kq_mask->data;
            const int64_t n_kv = ctx_len + n_block;
            for (int64_t q = 0; q < n_block; ++q) {
                for (int64_t k = 0; k < n_kv; ++k) {
                    if (k >= n_real && k < ctx_len) {
                        data[q * n_kv + k] = -INFINITY;
                    } else {
                        data[q * n_kv + k] = 0.0f;
                    }
                }
            }
        }

        if (kq_mask_swa && kq_mask_swa->buffer && n_swa > 0) {
            GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask_swa->buffer));
            float * data = (float *) kq_mask_swa->data;
            const int64_t n_kv   = ctx_len + n_block;
            const int32_t window = (int32_t) n_swa;
            const bool    have_pos = (ubatch != nullptr) && (ubatch->pos != nullptr)
                                   && ((int64_t) ubatch->n_tokens >= n_block);
            for (int64_t q = 0; q < n_block; ++q) {
                const int32_t q_pos = have_pos ? (int32_t) (ubatch->pos[q] - win_off) : (int32_t) (n_real + q);
                for (int64_t k = 0; k < n_kv; ++k) {
                    float v = 0.0f;
                    if (k < n_real) {
                        if (q_pos - (int32_t) k > window) v = -INFINITY;
                    } else if (k < ctx_len) {
                        v = -INFINITY;
                    } else {
                        const int64_t b_k = k - ctx_len;
                        if (b_k > q) v = -INFINITY;
                    }
                    data[q * n_kv + k] = v;
                }
            }
        }
    } else {
        // === Multi-slot batched draft path ===
        // Pack each slot's cross data at per-slot offsets in target_hidden.
        // Build per-slot isolating masks so each slot's block queries only
        // attend to that slot's cross keys and block keys.
        GGML_ASSERT(ctx_len % n_seqs == 0);
        GGML_ASSERT(n_block % n_seqs == 0);
        const int per_slot_ctx    = (int)(ctx_len / n_seqs);
        const int n_seq_tokens    = (int)(n_block / n_seqs);
        const size_t n_feat       = cross ? (size_t) cross->n_embd : 0;

        // collect per-slot cross data
        struct { const float * data; int64_t n_real; } slot_info[LLAMA_DFLASH_MAX_SLOTS] = {};
        for (int s = 0; s < n_seqs && s < LLAMA_DFLASH_MAX_SLOTS; s++) {
            llama_seq_id seq = ubatch->seq_id_unq[s];
            if (!cross) { continue; }
            auto it = cross->v_embd_per_seq.find(seq);
            if (it != cross->v_embd_per_seq.end() && !it->second.v_embd.empty()) {
                slot_info[s] = { it->second.v_embd.data(), it->second.n_enc_real };
            }
        }

        // pack target_hidden (f16): slot s at offset [s * per_slot_ctx, (s+1) * per_slot_ctx)
        // Sliding window: if a slot has more tokens than per_slot_ctx, take the most recent.
        int64_t slot_win_off[LLAMA_DFLASH_MAX_SLOTS] = {};
        int64_t slot_n_copy[LLAMA_DFLASH_MAX_SLOTS]  = {};
        for (int s = 0; s < n_seqs && s < LLAMA_DFLASH_MAX_SLOTS; s++) {
            const int64_t nr = slot_info[s].n_real > 0 ? slot_info[s].n_real : 0;
            slot_n_copy[s] = std::min(nr, (int64_t) per_slot_ctx);
            slot_win_off[s] = (nr > per_slot_ctx) ? (nr - per_slot_ctx) : 0;
        }

        if (target_hidden && n_feat > 0) {
            ggml_backend_tensor_memset(target_hidden, 0, 0, ggml_nbytes(target_hidden));
            for (int s = 0; s < n_seqs; s++) {
                if (!slot_info[s].data || slot_n_copy[s] <= 0) { continue; }
                const float * src = slot_info[s].data + slot_win_off[s] * n_feat;
                const size_t copy_bytes = n_feat * (size_t) slot_n_copy[s] * sizeof(float);
                const size_t dst_offset = (size_t) s * (size_t) per_slot_ctx * n_feat * sizeof(float);
                ggml_backend_tensor_set(target_hidden, src, dst_offset, copy_bytes);
            }
        }

        // pos_ctx: per-slot position patterns in each drafter window.
        if (pos_ctx && pos_ctx->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(pos_ctx->buffer));
            int32_t * data = (int32_t *) pos_ctx->data;
            for (int s = 0; s < n_seqs; s++) {
                const int64_t nc  = slot_n_copy[s];
                const int64_t off = (int64_t) s * per_slot_ctx;
                for (int64_t i = 0; i < per_slot_ctx; i++) {
                    data[off + i] = (i < nc) ? (int32_t) i : 0;
                }
            }
        }

        if (pos_q_rebased && pos_q_rebased->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(pos_q_rebased->buffer));
            int32_t * data = (int32_t *) pos_q_rebased->data;
            const bool have_pos = (ubatch != nullptr) && (ubatch->pos != nullptr)
                               && ((int64_t) ubatch->n_tokens >= n_block);
            for (int64_t q = 0; q < n_block; ++q) {
                const int qs = (int)(q / n_seq_tokens);
                const int ql = (int)(q % n_seq_tokens);
                const int64_t full_nr = slot_info[qs].n_real > 0 ? slot_info[qs].n_real : 0;
                data[q] = have_pos
                    ? (int32_t) (ubatch->pos[q] - slot_win_off[qs])
                    : (int32_t) (full_nr + ql);
            }
        }

        // kq_mask: per-slot isolation — query in slot S sees only slot S's
        // cross keys (real only) and slot S's block keys (causal).
        if (kq_mask && kq_mask->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask->buffer));
            float * data = (float *) kq_mask->data;
            const int64_t n_kv = ctx_len + n_block;
            for (int64_t q = 0; q < n_block; q++) {
                const int qs = (int)(q / n_seq_tokens);
                const int ql = (int)(q % n_seq_tokens);
                const int64_t nc = slot_n_copy[qs];
                for (int64_t k = 0; k < n_kv; k++) {
                    float v = -INFINITY;
                    if (k < ctx_len) {
                        const int ks = (int)(k / per_slot_ctx);
                        const int kl = (int)(k % per_slot_ctx);
                        if (ks == qs && kl < nc) { v = 0.0f; }
                    } else {
                        const int bi = (int)(k - ctx_len);
                        const int ks = bi / n_seq_tokens;
                        const int kl = bi % n_seq_tokens;
                        if (ks == qs && kl <= ql) { v = 0.0f; }
                    }
                    data[q * n_kv + k] = v;
                }
            }
        }

        // kq_mask_swa: per-slot isolation + sliding window on cross keys
        if (kq_mask_swa && kq_mask_swa->buffer && n_swa > 0) {
            GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask_swa->buffer));
            float * data = (float *) kq_mask_swa->data;
            const int64_t n_kv   = ctx_len + n_block;
            const int32_t window = (int32_t) n_swa;
            const bool have_pos  = (ubatch->pos != nullptr)
                                 && ((int64_t) ubatch->n_tokens >= n_block);
            for (int64_t q = 0; q < n_block; q++) {
                const int qs = (int)(q / n_seq_tokens);
                const int ql = (int)(q % n_seq_tokens);
                const int64_t nc = slot_n_copy[qs];
                const int64_t full_nr = slot_info[qs].n_real > 0 ? slot_info[qs].n_real : 0;
                const int32_t q_pos = have_pos ? (int32_t) (ubatch->pos[q] - slot_win_off[qs]) : (int32_t)(full_nr + ql);
                for (int64_t k = 0; k < n_kv; k++) {
                    float v = -INFINITY;
                    if (k < ctx_len) {
                        const int ks = (int)(k / per_slot_ctx);
                        const int kl = (int)(k % per_slot_ctx);
                        if (ks == qs && kl < nc && q_pos - (int32_t) kl <= window) {
                            v = 0.0f;
                        }
                    } else {
                        const int bi = (int)(k - ctx_len);
                        const int ks = bi / n_seq_tokens;
                        const int kl = bi % n_seq_tokens;
                        if (ks == qs && kl <= ql) { v = 0.0f; }
                    }
                    data[q * n_kv + k] = v;
                }
            }
        }
    }
}

llm_build_dflash_draft::llm_build_dflash_draft(
        const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int64_t n_target_features = hparams.dflash_n_target_features;

    // Drafter graph shape:
    //   n_slots == 1: ctx_len = cross->n_enc (power-of-2 bucket of actual data length),
    //                 capped by cparams.dflash_cross_ctx and dflash_max_cross_ctx().
    //                 set_cross_data triggers sched_need_reserve when the bucket changes,
    //                 so the graph re-reserves at each bucket boundary. This matches the
    //                 original single-slot path and keeps throughput unchanged.
    //   n_slots >= 2: ctx_len = n_slots × cparams.dflash_cross_ctx (fixed). The shared
    //                 drafter ctx services multiple slots whose bucket-of-n_enc would
    //                 otherwise thrash sched_need_reserve as different slots write data
    //                 of different lengths. Fixed width avoids that thrash; multi-slot
    //                 users pay flat n_slots × dflash_cross_ctx attention cost.
    const int64_t ctx_len = dflash_draft_ctx_len(cross, cparams);

    const int64_t n_kv_total = ctx_len + n_tokens;
    const bool use_kv_cache_graph = dflash_kv_cache_ready_for_window(cross, ctx_len);

    // --- DFlash-specific inputs ---
    const bool have_swa = hparams.is_swa_any();
    const bool need_swa_mask = have_swa && hparams.n_swa > 0 && (int64_t) hparams.n_swa < n_kv_total;
    auto inp_dflash = std::make_unique<llm_graph_input_dflash>(cross, ctx_len, n_tokens, hparams.n_swa, use_kv_cache_graph);

    // concatenated target hidden states [n_target_features, ctx_len]
    inp_dflash->target_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_target_features, ctx_len);
    ggml_set_input(inp_dflash->target_hidden);
    cb(inp_dflash->target_hidden, "dflash_target_hidden", -1);

    // context positions for K RoPE [ctx_len]
    inp_dflash->pos_ctx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ctx_len);
    ggml_set_input(inp_dflash->pos_ctx);
    cb(inp_dflash->pos_ctx, "dflash_pos_ctx", -1);

    // drafted-block positions rebased into the drafter window for Q/noise-K RoPE
    inp_dflash->pos_q_rebased = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_dflash->pos_q_rebased);
    cb(inp_dflash->pos_q_rebased, "dflash_pos_q_rebased", -1);

    // asymmetric non-causal mask [n_kv_total, n_tokens, 1, 1] — full-attention layers
    inp_dflash->kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_total, n_tokens, 1, 1);
    ggml_set_input(inp_dflash->kq_mask);
    inp_dflash->kq_mask_cnv = cparams.flash_attn
        ? ggml_cast(ctx0, inp_dflash->kq_mask, GGML_TYPE_F16)
        : inp_dflash->kq_mask;

    if (need_swa_mask) {
        inp_dflash->kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_total, n_tokens, 1, 1);
        ggml_set_input(inp_dflash->kq_mask_swa);
        cb(inp_dflash->kq_mask_swa, "dflash_kq_mask_swa", -1);
        inp_dflash->kq_mask_swa_cnv = cparams.flash_attn
            ? ggml_cast(ctx0, inp_dflash->kq_mask_swa, GGML_TYPE_F16)
            : inp_dflash->kq_mask_swa;
    }

    ggml_tensor * kq_mask_full  = inp_dflash->kq_mask_cnv;
    ggml_tensor * kq_mask_swa   = inp_dflash->kq_mask_swa_cnv; // may be null if no SWA
    ggml_tensor * pos_ctx       = inp_dflash->pos_ctx;
    ggml_tensor * pos_q_rebased = inp_dflash->pos_q_rebased;
    ggml_tensor * target_hidden = inp_dflash->target_hidden;

    res->add_input(std::move(inp_dflash));

    // --- Embedding ---
    // tok_embd/output may be nullptr during graph reservation (shared from target at runtime)
    // Use Q4_0 placeholder to avoid 4.8 GB F32 allocation during reservation
    ggml_tensor * tok_embd_use = model.tok_embd;
    if (!tok_embd_use) {
        tok_embd_use = ggml_new_tensor_2d(ctx0, GGML_TYPE_Q4_0, n_embd, model.vocab.n_tokens());
    }
    ggml_tensor * inpL = build_inp_embd(tok_embd_use);
    ggml_tensor * inp_out_ids = n_outputs < n_tokens ? build_inp_out_ids() : nullptr;

    // block positions rebased into the drafter's sliding window
    ggml_tensor * inp_pos = pos_q_rebased;

    // --- Fusion layer: project concatenated target hidden states ---
    ggml_tensor * fused_target = build_lora_mm(model.dflash_fc, target_hidden);
    fused_target = build_norm(fused_target, model.dflash_hidden_norm, nullptr, LLM_NORM_RMS, -1);
    cb(fused_target, "fused_target", -1);

    // --- Transformer layers ---
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        ggml_tensor * kq_mask = (hparams.is_swa(il) && kq_mask_swa) ? kq_mask_swa : kq_mask_full;

        ggml_tensor * cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // --- KV-injection attention ---
        {
            // Q from drafter hidden only
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                                 n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            // K from drafter (noise tokens)
            ggml_tensor * Kcur_noise = build_lora_mm(model.layers[il].wk, cur);
            Kcur_noise = ggml_reshape_3d(ctx0, Kcur_noise, n_embd_head, n_head_kv, n_tokens);
            Kcur_noise = build_norm(Kcur_noise, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Kcur_noise = ggml_rope_ext(ctx0, Kcur_noise, inp_pos, nullptr,
                                       n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                       ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur_noise, "Kcur_noise", il);

            const auto * kv_cache = cross ? cross->dflash_kv_cache : nullptr;
            const dflash_kv_cache_mode kv_cache_mode = dflash_kv_cache_mode_env();
            const bool use_kv_cache =
                use_kv_cache_graph &&
                kv_cache != nullptr &&
                kv_cache_mode != DFLASH_KV_CACHE_OFF &&
                il < kv_cache->n_layers &&
                (int) kv_cache->k_ring.size() > il &&
                (int) kv_cache->v_ring.size() > il &&
                kv_cache->k_ring[il] != nullptr &&
                kv_cache->v_ring[il] != nullptr;
            const bool use_k_cache = use_kv_cache &&
                (kv_cache_mode == DFLASH_KV_CACHE_BOTH || kv_cache_mode == DFLASH_KV_CACHE_K_ONLY);
            const bool use_v_cache = use_kv_cache &&
                (kv_cache_mode == DFLASH_KV_CACHE_BOTH || kv_cache_mode == DFLASH_KV_CACHE_V_ONLY);

            // K from target (context features)
            ggml_tensor * Kcur_ctx = nullptr;
            if (use_k_cache) {
                auto * k_ring = kv_cache->k_ring[il];
                const int64_t write_pos = kv_cache->n_filled >= ctx_len
                    ? (kv_cache->write_pos % ctx_len)
                    : 0;
                if (write_pos > 0) {
                    const int64_t tail_len = ctx_len - write_pos;
                    ggml_tensor * k_tail = ggml_view_3d(ctx0, k_ring,
                        n_embd_head, n_head_kv, tail_len,
                        k_ring->nb[1], k_ring->nb[2], (size_t) write_pos * k_ring->nb[2]);
                    ggml_tensor * k_head = ggml_view_3d(ctx0, k_ring,
                        n_embd_head, n_head_kv, write_pos,
                        k_ring->nb[1], k_ring->nb[2], 0);
                    Kcur_ctx = ggml_concat(ctx0, k_tail, k_head, 2);
                    cb(Kcur_ctx, "Kcur_ctx_cache_ordered", il);
                } else {
                    Kcur_ctx = ggml_view_3d(ctx0, k_ring,
                        n_embd_head, n_head_kv, ctx_len,
                        k_ring->nb[1], k_ring->nb[2], 0);
                    cb(Kcur_ctx, "Kcur_ctx_cache_pre_rope", il);
                }
                Kcur_ctx = ggml_rope_ext(ctx0, Kcur_ctx, pos_ctx, nullptr,
                                         n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                         ext_factor, attn_factor, beta_fast, beta_slow);
            } else {
                Kcur_ctx = build_lora_mm(model.layers[il].wk, fused_target);
                Kcur_ctx = ggml_reshape_3d(ctx0, Kcur_ctx, n_embd_head, n_head_kv, ctx_len);
                Kcur_ctx = build_norm(Kcur_ctx, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
                Kcur_ctx = ggml_rope_ext(ctx0, Kcur_ctx, pos_ctx, nullptr,
                                         n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                         ext_factor, attn_factor, beta_fast, beta_slow);
            }
            cb(Kcur_ctx, use_k_cache ? "Kcur_ctx_cache" : "Kcur_ctx", il);

            // V from drafter (noise tokens)
            ggml_tensor * Vcur_noise = build_lora_mm(model.layers[il].wv, cur);
            Vcur_noise = ggml_reshape_3d(ctx0, Vcur_noise, n_embd_head, n_head_kv, n_tokens);
            cb(Vcur_noise, "Vcur_noise", il);

            // V from target (context features)
            ggml_tensor * Vcur_ctx = nullptr;
            if (use_v_cache) {
                auto * v_ring = kv_cache->v_ring[il];
                const int64_t write_pos = kv_cache->n_filled >= ctx_len
                    ? (kv_cache->write_pos % ctx_len)
                    : 0;
                if (write_pos > 0) {
                    const int64_t tail_len = ctx_len - write_pos;
                    ggml_tensor * v_tail = ggml_view_3d(ctx0, v_ring,
                        n_embd_head, n_head_kv, tail_len,
                        v_ring->nb[1], v_ring->nb[2], (size_t) write_pos * v_ring->nb[2]);
                    ggml_tensor * v_head = ggml_view_3d(ctx0, v_ring,
                        n_embd_head, n_head_kv, write_pos,
                        v_ring->nb[1], v_ring->nb[2], 0);
                    Vcur_ctx = ggml_concat(ctx0, v_tail, v_head, 2);
                    cb(Vcur_ctx, "Vcur_ctx_cache_ordered", il);
                } else {
                    Vcur_ctx = ggml_view_3d(ctx0, v_ring,
                        n_embd_head, n_head_kv, ctx_len,
                        v_ring->nb[1], v_ring->nb[2], 0);
                }
            } else {
                Vcur_ctx = build_lora_mm(model.layers[il].wv, fused_target);
                Vcur_ctx = ggml_reshape_3d(ctx0, Vcur_ctx, n_embd_head, n_head_kv, ctx_len);
            }
            cb(Vcur_ctx, use_v_cache ? "Vcur_ctx_cache" : "Vcur_ctx", il);

            // concatenate K: [ctx, noise] along sequence dim (dim 2)
            ggml_tensor * Kcur = ggml_concat(ctx0, Kcur_ctx, Kcur_noise, 2);
            cb(Kcur, "Kcur", il);

            // concatenate V: [ctx, noise] along sequence dim (dim 2)
            ggml_tensor * Vcur = ggml_concat(ctx0, Vcur_ctx, Vcur_noise, 2);
            cb(Vcur, "Vcur", il);

            // prevent reordering
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);

            // asymmetric attention: Q [head_dim, n_head, n_tokens]
            //                       K [head_dim, n_head_kv, n_kv_total]
            //                    mask [n_kv_total, n_tokens, 1, 1]
            cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, kq_mask, nullptr, nullptr,
                                 1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "kqv_out", il);

            // output projection
            cur = build_lora_mm(model.layers[il].wo, cur);
        }

        // residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        // post-attention RMSNorm
        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        // SwiGLU FFN
        cur = build_ffn(cur,
            model.layers[il].ffn_up,   nullptr, nullptr,
            model.layers[il].ffn_gate, nullptr, nullptr,
            model.layers[il].ffn_down, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // FFN residual
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    // final RMSNorm
    if (inp_out_ids) {
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        cb(inpL, "result_output_rows", -1);
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head — may be nullptr during reservation (shared from target at runtime)
    // Use Q4_0 placeholder to avoid 4.8 GB F32 allocation during reservation
    ggml_tensor * output_use = model.output;
    if (!output_use) {
        output_use = ggml_new_tensor_2d(ctx0, GGML_TYPE_Q4_0, n_embd, model.vocab.n_tokens());
    }
    cur = build_lora_mm(output_use, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    // GPU top-K or argmax — avoids 15.9MB logits transfer + CPU scan for DFlash draft
    const float sample_temp = cparams.dflash_sample_temp;
    static std::atomic<uint64_t> gumbel_counter{1};
    const uint64_t seed = (sample_temp > 0.0f) ? gumbel_counter.fetch_add(1) : 0;
    const int topk = cparams.dflash_topk;
    if (topk > 1) {
        res->t_logits_argmax = ggml_topk_ext(ctx0, cur, topk, sample_temp, seed);
    } else {
        res->t_logits_argmax = ggml_argmax_ext(ctx0, cur, sample_temp, seed);
    }

    ggml_build_forward_expand(gf, res->t_logits_argmax);
}

llm_build_dflash_kv_update::llm_build_dflash_kv_update(
        const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int64_t n_target_features = hparams.dflash_n_target_features;
    GGML_ASSERT(n_tokens > 0);

    auto inp_update = std::make_unique<llm_graph_input_dflash_update>(cross, n_tokens);
    inp_update->target_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_target_features, n_tokens);
    ggml_set_input(inp_update->target_hidden);
    cb(inp_update->target_hidden, "dflash_kv_update_hidden", -1);

    ggml_tensor * target_hidden = inp_update->target_hidden;
    res->add_input(std::move(inp_update));

    ggml_tensor * fused_target = build_lora_mm(model.dflash_fc, target_hidden);
    fused_target = build_norm(fused_target, model.dflash_hidden_norm, nullptr, LLM_NORM_RMS, -1);
    cb(fused_target, "dflash_kv_update_fused", -1);

    res->dflash_k_update.reserve(n_layer);
    res->dflash_v_update.reserve(n_layer);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * Kcur_ctx = build_lora_mm(model.layers[il].wk, fused_target);
        Kcur_ctx = ggml_reshape_3d(ctx0, Kcur_ctx, n_embd_head, n_head_kv, n_tokens);
        Kcur_ctx = build_norm(Kcur_ctx, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Kcur_ctx, "dflash_kv_update_k", il);
        res->dflash_k_update.push_back(Kcur_ctx);
        ggml_build_forward_expand(gf, Kcur_ctx);

        ggml_tensor * Vcur_ctx = build_lora_mm(model.layers[il].wv, fused_target);
        Vcur_ctx = ggml_reshape_3d(ctx0, Vcur_ctx, n_embd_head, n_head_kv, n_tokens);
        cb(Vcur_ctx, "dflash_kv_update_v", il);
        res->dflash_v_update.push_back(Vcur_ctx);
        ggml_build_forward_expand(gf, Vcur_ctx);
    }
}
