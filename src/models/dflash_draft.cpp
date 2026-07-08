#include "models.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <vector>

void llama_model_dflash_draft::load_arch_hparams(llama_model_loader & ml) {
    auto & hparams = this->hparams;

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (arch == LLM_ARCH_DFLASH) {
        // Canonical upstream DFlash uses literal, un-prefixed metadata keys.
        ml.get_key("dflash.block_size",        hparams.dflash_block_size,        false);
        ml.get_key("dflash.mask_token_id",     hparams.dflash_mask_token_id,     false);
        ml.get_key("dflash.n_target_features", hparams.dflash_n_target_features, false);
    } else {
        ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE,        hparams.dflash_block_size,        false);
        ml.get_key(LLM_KV_DFLASH_MASK_TOKEN_ID,     hparams.dflash_mask_token_id,     false);
        ml.get_key(LLM_KV_DFLASH_N_TARGET_FEATURES, hparams.dflash_n_target_features, false);
    }

    {
        const std::string key = arch == LLM_ARCH_DFLASH ? "dflash.target_layer_ids" : ml.llm_kv(LLM_KV_DFLASH_TARGET_LAYER_IDS);
        const int64_t kid = gguf_find_key(ml.metadata, key.c_str());
        if (kid >= 0) {
            const size_t n = gguf_get_arr_n(ml.metadata, kid);
            hparams.dflash_n_target_layers = std::min((uint32_t) n, (uint32_t) 8);
            const void * data = gguf_get_arr_data(ml.metadata, kid);
            for (uint32_t i = 0; i < hparams.dflash_n_target_layers; ++i) {
                hparams.dflash_target_layer_ids[i] = ((const uint32_t *) data)[i];
            }
        }
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer, false);
    }
}

void llama_model_dflash_draft::load_arch_tensors(llama_model_loader & ml) {
    GGML_UNUSED(ml);
    LLAMA_LOAD_LOCALS;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    const llm_tensor dflash_fc_tensor =
        arch == LLM_ARCH_DFLASH ? LLM_TENSOR_DFLASH_UPSTREAM_FC : LLM_TENSOR_DFLASH_FC;
    const llm_tensor dflash_hidden_norm_tensor =
        arch == LLM_ARCH_DFLASH ? LLM_TENSOR_DFLASH_UPSTREAM_HIDDEN_NORM : LLM_TENSOR_DFLASH_HIDDEN_NORM;
    const llm_tensor dflash_layer_norm_tensor =
        arch == LLM_ARCH_DFLASH ? LLM_TENSOR_FFN_NORM : LLM_TENSOR_ATTN_POST_NORM;

    dflash_fc          = create_tensor(tn(dflash_fc_tensor,          "weight"), {(int64_t) hparams.dflash_n_target_features, n_embd}, 0);
    dflash_hidden_norm = create_tensor(tn(dflash_hidden_norm_tensor, "weight"), {n_embd}, 0);

    for (int i = 0; i < (int) hparams.n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(dflash_layer_norm_tensor, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {0}, TENSOR_NOT_REQUIRED);
        if (layer.ssm_conv1d) {
            const int64_t d_inner = layer.ssm_conv1d->ne[1];

            layer.ssm_in   = create_tensor(tn(LLM_TENSOR_SSM_IN,   "weight", i), {n_embd, d_inner},     TENSOR_NOT_REQUIRED);
            layer.ssm_out  = create_tensor(tn(LLM_TENSOR_SSM_OUT,  "weight", i), {d_inner / 2, n_embd}, TENSOR_NOT_REQUIRED);
            layer.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {d_inner / 2},         TENSOR_NOT_REQUIRED);

            layer.ssm_dt       = create_tensor(tn(LLM_TENSOR_SSM_DT,       "weight", i), {d_inner / 2, d_inner / 2}, TENSOR_NOT_REQUIRED);
            layer.ssm_a        = create_tensor(tn(LLM_TENSOR_SSM_A,        "weight", i), {d_inner / 2},              TENSOR_NOT_REQUIRED);
            layer.ssm_d        = create_tensor(tn(LLM_TENSOR_SSM_D,        "weight", i), {d_inner / 2},              TENSOR_NOT_REQUIRED);
            layer.ssm_dt_b     = create_tensor(tn(LLM_TENSOR_SSM_DT,       "bias",   i), {d_inner / 2},              TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,   "bias",   i), {d_inner},                  TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash_draft::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DFLASH_KV_UPDATE) {
        return std::make_unique<llm_build_dflash_kv_update>(*this, params);
    }
    return std::make_unique<llm_build_dflash_draft>(*this, params);
}

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

template <typename Fn>
static void dflash_fill_kq_mask(ggml_tensor * mask, const Fn & fn) {
    GGML_ASSERT(ggml_backend_buffer_is_host(mask->buffer));

    if (mask->type == GGML_TYPE_F16) {
        fn((ggml_fp16_t *) mask->data);
    } else {
        GGML_ASSERT(mask->type == GGML_TYPE_F32);
        fn((float *) mask->data);
    }
}

template <typename T>
static void dflash_set_mask_value(T * data, int64_t index, float value) {
    data[index] = llama_cast<T>(value);
}

static ggml_tensor * dflash_mul_mat_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur) / n);
    res = ggml_mul_mat(ctx, rot, res);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

class llm_graph_input_attn_kv_backend final : public llm_graph_input_attn_kv {
public:
    llm_graph_input_attn_kv_backend(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx,
            bool rebind_base_from_iswa) :
        llm_graph_input_attn_kv(hparams, cparams, mctx),
        rebind_base_from_iswa(rebind_base_from_iswa) {
    }

    void set_input(const llama_ubatch * ubatch) override {
        mctx->set_input_k_idxs_backend(self_k_idxs, ubatch);
        mctx->set_input_v_idxs_backend(self_v_idxs, ubatch);

        if (self_k_rot) {
            mctx->set_input_k_rot_backend(self_k_rot);
        }

        if (self_v_rot) {
            mctx->set_input_v_rot_backend(self_v_rot);
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        const auto * mctx_cur = rebind_base_from_iswa
            ? static_cast<const llama_kv_cache_iswa_context *>(params.mctx)->get_base()
            : static_cast<const llama_kv_cache_context *>(params.mctx);
        this->mctx = mctx_cur;

        bool ok = true;
        ok &= mctx_cur != nullptr;
        ok &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
        return ok;
    }

    const bool rebind_base_from_iswa = false;
};

class llm_graph_input_attn_kv_commit_backend final : public llm_graph_input_attn_kv {
public:
    llm_graph_input_attn_kv_commit_backend(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx) :
        llm_graph_input_attn_kv(hparams, cparams, mctx),
        kv(mctx ? mctx->get_kv() : nullptr),
        sinfo(mctx ? mctx->current_sinfo() : llama_kv_cache::slot_info()) {
    }

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(kv != nullptr);
        kv->set_input_k_idxs_backend(self_k_idxs, ubatch, sinfo);
        kv->set_input_v_idxs_backend(self_v_idxs, ubatch, sinfo);

        if (self_k_rot) {
            kv->set_input_k_rot_backend(self_k_rot);
        }

        if (self_v_rot) {
            kv->set_input_v_rot_backend(self_v_rot);
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        GGML_UNUSED(params);
        return false;
    }

    const llama_kv_cache * kv = nullptr;
    llama_kv_cache::slot_info sinfo;
};

static llm_graph_input_attn_kv * dflash_build_base_attn_input(
        llm_graph_result * res,
        ggml_context * ctx0,
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_ubatch & ubatch,
        const llama_kv_cache_context * base_ctx,
        bool rebind_base_from_iswa) {
    auto inp = std::make_unique<llm_graph_input_attn_kv_backend>(hparams, cparams, base_ctx, rebind_base_from_iswa);

    inp->self_k_idxs = base_ctx->build_input_k_idxs(ctx0, ubatch);
    inp->self_v_idxs = base_ctx->build_input_v_idxs(ctx0, ubatch);

    inp->self_k_rot = base_ctx->build_input_k_rot(ctx0);
    inp->self_v_rot = base_ctx->build_input_v_rot(ctx0);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

static llm_graph_input_attn_kv_commit_backend * dflash_build_commit_attn_input(
        llm_graph_result * res,
        ggml_context * ctx0,
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_ubatch & ubatch,
        const llama_kv_cache_context * base_ctx) {
    auto inp = std::make_unique<llm_graph_input_attn_kv_commit_backend>(hparams, cparams, base_ctx);

    inp->self_k_idxs = base_ctx->build_input_k_idxs(ctx0, ubatch);
    inp->self_v_idxs = base_ctx->build_input_v_idxs(ctx0, ubatch);

    inp->self_k_rot = base_ctx->build_input_k_rot(ctx0);
    inp->self_v_rot = base_ctx->build_input_v_rot(ctx0);

    return (llm_graph_input_attn_kv_commit_backend *) res->add_input(std::move(inp));
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

static bool dflash_input_shape_debug() {
    static const bool v = [] {
        const char * e = getenv("GGML_DFLASH_INPUT_DEBUG");
        return e && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return v;
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
                        src_n_real = it->second.v_embd_gpu_n_enc_real;
                    } else if (!it->second.v_embd.empty()) {
                        src_data   = it->second.v_embd.data();
                        src_n_real = it->second.n_enc_real;
                    }
                }
            }
            if (!src_data && !src_gpu) {
                if (cross->v_embd_gpu) {
                    src_gpu    = cross->v_embd_gpu;
                    src_n_real = cross->v_embd_gpu_n_enc_real;
                } else if (!cross->v_embd.empty()) {
                    src_data   = cross->v_embd.data();
                    src_n_real = cross->n_enc_real;
                }
            }
        }

        // Sliding window: if src has more tokens than ctx_len, take the most recent
        const int64_t src_real = src_n_real > 0 ? src_n_real : 0;
        const int64_t n_copy  = std::min(src_real, ctx_len);
        const int64_t win_off = (src_real > ctx_len) ? (src_real - ctx_len) : 0;

        const bool skip_target_hidden_upload =
            use_kv_cache && dflash_kv_cache_mode_env() == DFLASH_KV_CACHE_BOTH;

        if (!skip_target_hidden_upload &&
                target_hidden && target_hidden->buffer && target_hidden->data &&
                (src_data || src_gpu) && n_copy > 0) {
            const int64_t n_feat = cross ? cross->n_embd : 0;
            const size_t tensor_bytes = ggml_nbytes(target_hidden);

            if (n_feat != target_hidden->ne[0]) {
                if (dflash_input_shape_debug()) {
                    fprintf(stderr, "dflash input: feature mismatch single-slot n_feat=%lld target_hidden_ne0=%lld ctx_len=%lld n_copy=%lld\n",
                        (long long) n_feat,
                        (long long) target_hidden->ne[0],
                        (long long) ctx_len,
                        (long long) n_copy);
                }
                ggml_backend_tensor_memset(target_hidden, 0, 0, tensor_bytes);
            } else {
                const size_t copy_bytes  = (size_t) n_feat * (size_t) n_copy * sizeof(float);
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
            }
        } else if (!skip_target_hidden_upload && target_hidden && target_hidden->buffer && target_hidden->data) {
            ggml_backend_tensor_memset(target_hidden, 0, 0, ggml_nbytes(target_hidden));
        }

        const int64_t n_real = n_copy;

        const bool have_pos = (ubatch != nullptr) && (ubatch->pos != nullptr)
                           && ((int64_t) ubatch->n_tokens >= n_block);
        const int32_t q_pos_base = have_pos && n_block > 0
            ? (int32_t) ubatch->pos[0]
            : (int32_t) n_real;
        const int32_t ctx_pos_base = q_pos_base - (int32_t) n_real;

        if (pos_ctx && pos_ctx->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(pos_ctx->buffer));
            int32_t * data = (int32_t *) pos_ctx->data;
            for (int64_t i = 0; i < ctx_len; ++i) {
                data[i] = (i < n_real) ? (ctx_pos_base + (int32_t) i) : 0;
            }
        }

        if (pos_q_rebased && pos_q_rebased->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(pos_q_rebased->buffer));
            int32_t * data = (int32_t *) pos_q_rebased->data;
            for (int64_t q = 0; q < n_block; ++q) {
                data[q] = have_pos ? (int32_t) ubatch->pos[q] : (ctx_pos_base + (int32_t) n_real + (int32_t) q);
            }
        }

        if (kq_mask && kq_mask->buffer) {
            dflash_fill_kq_mask(kq_mask, [&](auto * data) {
                const int64_t n_kv = ctx_len + n_block;
                for (int64_t q = 0; q < n_block; ++q) {
                    for (int64_t k = 0; k < n_kv; ++k) {
                        const float v = (k >= n_real && k < ctx_len) ? -INFINITY : 0.0f;
                        dflash_set_mask_value(data, q * n_kv + k, v);
                    }
                }
            });
        }

        if (kq_mask_swa && kq_mask_swa->buffer && n_swa > 0) {
            dflash_fill_kq_mask(kq_mask_swa, [&](auto * data) {
                const int64_t n_kv   = ctx_len + n_block;
                const int32_t window = (int32_t) n_swa;
                for (int64_t q = 0; q < n_block; ++q) {
                    const int32_t q_pos = have_pos ? (int32_t) ubatch->pos[q] : (ctx_pos_base + (int32_t) n_real + (int32_t) q);
                    for (int64_t k = 0; k < n_kv; ++k) {
                        float v = 0.0f;
                        if (k < n_real) {
                            const int32_t k_pos = ctx_pos_base + (int32_t) k;
                            if (q_pos - k_pos >= window) v = -INFINITY;
                        } else if (k < ctx_len) {
                            v = -INFINITY;
                        } else {
                            const int64_t b_k = k - ctx_len;
                            if (b_k > q) v = -INFINITY;
                        }
                        dflash_set_mask_value(data, q * n_kv + k, v);
                    }
                }
            });
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

        // collect per-slot cross data. Each slot may be CPU-backed or GPU-backed.
        struct slot_cross_info {
            const float * data = nullptr;
            const void  * gpu  = nullptr;
            int64_t       n_real = 0;
        };
        slot_cross_info slot_info[LLAMA_DFLASH_MAX_SLOTS] = {};

        for (int s = 0; s < n_seqs && s < LLAMA_DFLASH_MAX_SLOTS; s++) {
            if (!cross) {
                continue;
            }

            const llama_seq_id seq = ubatch->seq_id_unq[s];
            auto it = cross->v_embd_per_seq.find(seq);
            if (it == cross->v_embd_per_seq.end()) {
                continue;
            }

            if (it->second.v_embd_gpu) {
                slot_info[s].gpu = it->second.v_embd_gpu;
                slot_info[s].n_real = it->second.v_embd_gpu_n_enc_real;
            } else if (!it->second.v_embd.empty()) {
                slot_info[s].data = it->second.v_embd.data();
                slot_info[s].n_real = it->second.n_enc_real;
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

        if (target_hidden && target_hidden->buffer && target_hidden->data && n_feat > 0) {
            ggml_backend_tensor_memset(target_hidden, 0, 0, ggml_nbytes(target_hidden));

            if ((int64_t) n_feat == target_hidden->ne[0]) {
                for (int s = 0; s < n_seqs; s++) {
                    if (slot_n_copy[s] <= 0) {
                        continue;
                    }

                    const size_t copy_bytes =
                        n_feat * (size_t) slot_n_copy[s] * sizeof(float);
                    const size_t dst_offset =
                        (size_t) s * (size_t) per_slot_ctx * n_feat * sizeof(float);

                    if (slot_info[s].gpu && cross && cross->fn_set_tensor_d2d) {
                        const void * gpu_src =
                            (const char *) slot_info[s].gpu +
                            (size_t) slot_win_off[s] * n_feat * sizeof(float);

                        cross->fn_set_tensor_d2d(
                            target_hidden->data,
                            gpu_src,
                            dst_offset,
                            copy_bytes);
                    } else if (slot_info[s].data) {
                        const float * src =
                            slot_info[s].data + slot_win_off[s] * n_feat;

                        ggml_backend_tensor_set(
                            target_hidden,
                            src,
                            dst_offset,
                            copy_bytes);
                    } else if (dflash_input_shape_debug()) {
                        fprintf(stderr,
                            "dflash input: missing cross data for multi-slot seq_index=%d n_real=%lld n_copy=%lld\n",
                            s,
                            (long long) slot_info[s].n_real,
                            (long long) slot_n_copy[s]);
                    }
                }
            } else if (dflash_input_shape_debug()) {
                fprintf(stderr, "dflash input: feature mismatch multi-slot n_feat=%zu target_hidden_ne0=%lld n_seqs=%d per_slot_ctx=%d\n",
                    n_feat,
                    (long long) target_hidden->ne[0],
                    n_seqs,
                    per_slot_ctx);
            }
        }

        // pos_ctx: per-slot absolute positions for the active drafter window.
        int32_t slot_ctx_pos_base[LLAMA_DFLASH_MAX_SLOTS] = {};
        const bool have_pos = (ubatch != nullptr) && (ubatch->pos != nullptr)
                           && ((int64_t) ubatch->n_tokens >= n_block);
        for (int s = 0; s < n_seqs && s < LLAMA_DFLASH_MAX_SLOTS; s++) {
            const int64_t nc = slot_n_copy[s];
            const int64_t full_nr = slot_info[s].n_real > 0 ? slot_info[s].n_real : 0;
            const int32_t q_pos_base = have_pos ? (int32_t) ubatch->pos[s * n_seq_tokens] : (int32_t) full_nr;
            slot_ctx_pos_base[s] = q_pos_base - (int32_t) nc;
        }
        if (pos_ctx && pos_ctx->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(pos_ctx->buffer));
            int32_t * data = (int32_t *) pos_ctx->data;
            for (int s = 0; s < n_seqs; s++) {
                const int64_t nc  = slot_n_copy[s];
                const int64_t off = (int64_t) s * per_slot_ctx;
                for (int64_t i = 0; i < per_slot_ctx; i++) {
                    data[off + i] = (i < nc) ? (slot_ctx_pos_base[s] + (int32_t) i) : 0;
                }
            }
        }

        if (pos_q_rebased && pos_q_rebased->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(pos_q_rebased->buffer));
            int32_t * data = (int32_t *) pos_q_rebased->data;
            for (int64_t q = 0; q < n_block; ++q) {
                const int qs = (int)(q / n_seq_tokens);
                const int ql = (int)(q % n_seq_tokens);
                const int64_t full_nr = slot_info[qs].n_real > 0 ? slot_info[qs].n_real : 0;
                data[q] = have_pos
                    ? (int32_t) ubatch->pos[q]
                    : (slot_ctx_pos_base[qs] + (int32_t) full_nr + ql);
            }
        }

        // kq_mask: per-slot isolation — query in slot S sees only slot S's
        // cross keys (real only) and slot S's block keys (causal).
        if (kq_mask && kq_mask->buffer) {
            dflash_fill_kq_mask(kq_mask, [&](auto * data) {
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
                        dflash_set_mask_value(data, q * n_kv + k, v);
                    }
                }
            });
        }

        // kq_mask_swa: per-slot isolation + sliding window on cross keys
        if (kq_mask_swa && kq_mask_swa->buffer && n_swa > 0) {
            dflash_fill_kq_mask(kq_mask_swa, [&](auto * data) {
                const int64_t n_kv   = ctx_len + n_block;
                const int32_t window = (int32_t) n_swa;
                for (int64_t q = 0; q < n_block; q++) {
                    const int qs = (int)(q / n_seq_tokens);
                    const int ql = (int)(q % n_seq_tokens);
                    const int64_t nc = slot_n_copy[qs];
                    const int64_t full_nr = slot_info[qs].n_real > 0 ? slot_info[qs].n_real : 0;
                    const int32_t q_pos = have_pos ? (int32_t) ubatch->pos[q] : (slot_ctx_pos_base[qs] + (int32_t) full_nr + ql);
                    for (int64_t k = 0; k < n_kv; k++) {
                        float v = -INFINITY;
                        if (k < ctx_len) {
                            const int ks = (int)(k / per_slot_ctx);
                            const int kl = (int)(k % per_slot_ctx);
                            const int32_t k_pos = slot_ctx_pos_base[ks] + kl;
                            if (ks == qs && kl < nc && q_pos - k_pos < window) {
                                v = 0.0f;
                            }
                        } else {
                            const int bi = (int)(k - ctx_len);
                            const int ks = bi / n_seq_tokens;
                            const int kl = bi % n_seq_tokens;
                            if (ks == qs && kl <= ql) { v = 0.0f; }
                        }
                        dflash_set_mask_value(data, q * n_kv + k, v);
                    }
                }
            });
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
    const ggml_type type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
    inp_dflash->kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_kv_total, n_tokens, 1, 1);
    ggml_set_input(inp_dflash->kq_mask);
    inp_dflash->kq_mask_cnv = inp_dflash->kq_mask;

    if (need_swa_mask) {
        inp_dflash->kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_kv_total, n_tokens, 1, 1);
        ggml_set_input(inp_dflash->kq_mask_swa);
        cb(inp_dflash->kq_mask_swa, "dflash_kq_mask_swa", -1);
        inp_dflash->kq_mask_swa_cnv = inp_dflash->kq_mask_swa;
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

    // Keep absolute target positions so RoPE tracks the live suffix instead of
    // restarting from zero when the cross window slides.
    ggml_tensor * inp_pos = pos_q_rebased;

    // Full-attention draft layers consume accepted-prefix K/V from the normal
    // drafter KV cache. Sliding layers still read the current target-hidden
    // window directly because their context is already bounded by SWA.
    llm_graph_input_attn_kv * inp_attn_kv_full = nullptr;
    if (mctx) {
        const bool rebind_base_from_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;
        const llama_kv_cache_context * base_ctx =
            rebind_base_from_iswa
                ? static_cast<const llama_kv_cache_iswa_context *>(mctx)->get_base()
                : static_cast<const llama_kv_cache_context *>(mctx);
        inp_attn_kv_full = dflash_build_base_attn_input(res, ctx0, hparams, cparams, ubatch, base_ctx, rebind_base_from_iswa);
    }

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

        // --- Attention ---
        {
            // Q from drafter hidden only
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                                 n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            // K/V from drafter proposal tokens
            ggml_tensor * Kcur_noise = build_lora_mm(model.layers[il].wk, cur);
            Kcur_noise = ggml_reshape_3d(ctx0, Kcur_noise, n_embd_head, n_head_kv, n_tokens);
            Kcur_noise = build_norm(Kcur_noise, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Kcur_noise = ggml_rope_ext(ctx0, Kcur_noise, inp_pos, nullptr,
                                       n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                       ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur_noise, "Kcur_noise", il);

            ggml_tensor * Vcur_noise = build_lora_mm(model.layers[il].wv, cur);
            Vcur_noise = ggml_reshape_3d(ctx0, Vcur_noise, n_embd_head, n_head_kv, n_tokens);
            cb(Vcur_noise, "Vcur_noise", il);

            if (inp_attn_kv_full && !hparams.is_swa(il)) {
                const float kq_scale = 1.0f / sqrtf(float(n_embd_head));
                cur = build_attn(inp_attn_kv_full,
                                 model.layers[il].wo, nullptr, nullptr,
                                 Qcur, Kcur_noise, Vcur_noise,
                                 nullptr, nullptr, nullptr, kq_scale, il);
            } else {
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

    // GPU top-K or argmax — avoids 15.9MB logits transfer + CPU scan for DFlash draft.
    // Only built when the output device backend implements the extended argmax
    // semantics (CUDA); otherwise leave t_logits_argmax unset so the drafter
    // falls back to scanning the full logits on the CPU.
    if (cparams.dflash_graph_argmax) {
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
    } else {
        ggml_build_forward_expand(gf, cur);
    }
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

    ggml_tensor * inp_pos = nullptr;
    llm_graph_input_attn_kv_commit_backend * inp_attn_kv = nullptr;
    if (mctx != nullptr) {
        inp_pos = build_inp_pos();
        inp_attn_kv = dflash_build_commit_attn_input(
            res, ctx0, hparams, cparams, ubatch, static_cast<const llama_kv_cache_context *>(mctx));
    }

    ggml_tensor * fused_target = build_lora_mm(model.dflash_fc, target_hidden);
    fused_target = build_norm(fused_target, model.dflash_hidden_norm, nullptr, LLM_NORM_RMS, -1);
    cb(fused_target, "dflash_kv_update_fused", -1);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * Kcur_ctx = build_lora_mm(model.layers[il].wk, fused_target);
        Kcur_ctx = ggml_reshape_3d(ctx0, Kcur_ctx, n_embd_head, n_head_kv, n_tokens);
        Kcur_ctx = build_norm(Kcur_ctx, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * Vcur_ctx = build_lora_mm(model.layers[il].wv, fused_target);
        Vcur_ctx = ggml_reshape_3d(ctx0, Vcur_ctx, n_embd_head, n_head_kv, n_tokens);

        if (mctx == nullptr) {
            cb(Kcur_ctx, "dflash_kv_update_k", il);
            res->dflash_k_update.push_back(Kcur_ctx);
            ggml_build_forward_expand(gf, Kcur_ctx);

            cb(Vcur_ctx, "dflash_kv_update_v", il);
            res->dflash_v_update.push_back(Vcur_ctx);
            ggml_build_forward_expand(gf, Vcur_ctx);
            continue;
        }

        if (hparams.is_swa(il)) {
            continue;
        }

        Kcur_ctx = ggml_rope_ext(ctx0, Kcur_ctx, inp_pos, nullptr,
                                 n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Kcur_ctx, "dflash_kv_commit_k", il);
        cb(Vcur_ctx, "dflash_kv_commit_v", il);

        GGML_ASSERT(inp_attn_kv != nullptr);
        if (inp_attn_kv->self_k_rot) {
            Kcur_ctx = dflash_mul_mat_aux(ctx0, Kcur_ctx, inp_attn_kv->self_k_rot);
        }
        if (inp_attn_kv->self_v_rot) {
            Vcur_ctx = dflash_mul_mat_aux(ctx0, Vcur_ctx, inp_attn_kv->self_v_rot);
        }
        ggml_build_forward_expand(gf, inp_attn_kv->kv->cpy_k(ctx0, Kcur_ctx, inp_attn_kv->get_k_idxs(), il, inp_attn_kv->sinfo));
        ggml_build_forward_expand(gf, inp_attn_kv->kv->cpy_v(ctx0, Vcur_ctx, inp_attn_kv->get_v_idxs(), il, inp_attn_kv->sinfo));
    }
}
