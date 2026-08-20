#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-sampler.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-kvarn.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>

// dedup helpers

static void llm_flash_attn_ext_set_kvarn_domain(
        ggml_tensor * cur,
        enum ggml_flash_attn_ext_kvarn_domain domain) {
    while (cur != nullptr &&
            (cur->op == GGML_OP_RESHAPE ||
             cur->op == GGML_OP_PERMUTE ||
             cur->op == GGML_OP_TRANSPOSE ||
             cur->op == GGML_OP_VIEW ||
             cur->op == GGML_OP_CONT) &&
            cur->src[0] != nullptr) {
        cur = cur->src[0];
    }

    if (cur != nullptr && cur->op == GGML_OP_FLASH_ATTN_EXT) {
        cur->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN] = (int32_t) domain;
    }
}

static ggml_tensor * ggml_kvarn_wht_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        int64_t       head_width) {
    GGML_ASSERT(head_width == 128 || head_width == 256 || head_width == 512);
    if (!ggml_is_contiguous(cur)) {
        cur = ggml_cont(ctx, cur);
    }
    return ggml_kvarn_wht(ctx, cur, (int) head_width);
}

static ggml_tensor * llm_kvarn_rot_for_dim(
        ggml_tensor * rot_128,
        ggml_tensor * rot_256,
        ggml_tensor * rot_512,
        int64_t       head_dim) {
    switch (head_dim) {
        case 128: return rot_128;
        case 256: return rot_256;
        case 512: return rot_512;
        default:  return nullptr;
    }
}

static void llm_kvarn_set_rot_inputs(
        const llama_kv_cache_kvarn_context * kvarn,
        ggml_tensor * rot_128,
        ggml_tensor * rot_256,
        ggml_tensor * rot_512) {
    GGML_ASSERT(kvarn != nullptr);
    if (rot_128 && rot_128->buffer) {
        kvarn->set_input_kvarn_rot(rot_128);
    }
    if (rot_256 && rot_256->buffer) {
        kvarn->set_input_kvarn_rot(rot_256);
    }
    if (rot_512 && rot_512->buffer) {
        kvarn->set_input_kvarn_rot(rot_512);
    }
}

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    // flash attention requires an f16 mask
    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

// impl

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_embd_h::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens = ubatch->n_tokens;

    if (ubatch->token) {
        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    } else {
        // note: mtmd embedding input goes through here
        GGML_ASSERT(ubatch->embd);
        GGML_ASSERT(n_embd == embd->ne[0]);

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }

    // TODO: extend llama_ubatch to differentiate between token embeddings and hidden states
    //       for now, we assume that the hidden state is always provided as an embedding
    //       ref: https://github.com/ggml-org/llama.cpp/pull/23643
    if (ubatch->embd) {
        GGML_ASSERT(n_embd == h->ne[0]);

        ggml_backend_tensor_set(h, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }
}

bool llm_graph_input_embd_h::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   && embd->ne[1]   == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (h      && h->ne[1]      == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        if (ubatch->token && n_pos_per_embd == 4) {
            // in case we're using M-RoPE with text tokens, convert the 1D positions to 4D
            // the 3 first dims are the same, and 4th dim is all 0
            std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
            // copy the first dimension
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[               i] = ubatch->pos[i];
                pos_data[    n_tokens + i] = ubatch->pos[i];
                pos_data[2 * n_tokens + i] = ubatch->pos[i];
                pos_data[3 * n_tokens + i] = 0; // 4th dim is 0
            }
            ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
        } else {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
        }
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_tokens; ++i) {
                data[j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings   &&
       (cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK )) {

        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL)) // qwen3 reranking & embedding models use last token
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();

    res &= s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= s_copy_extra->ne[0] == mctx->get_n_rs() - params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    return res;
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

template <typename T>
static void print_mask(const T * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = llama_cast<float>(data[i * n_kv + j]);
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = ubatch->n_tokens;
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](auto * data, int64_t ne, int n_swa, llama_swa_type swa_type) {
        using T = std::remove_reference_t<decltype(*data)>;
        std::fill(data, data + ne, llama_cast<T>(-INFINITY));

        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos    p1 = ubatch->pos[i1];

            const uint64_t idst = i1*n_kv;

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0    = ubatch->pos[i0];

                // mask different sequences
                if (s0 != s1) {
                    continue;
                }

                // mask future tokens
                if (cparams.causal_attn && p0 > p1) {
                    continue;
                }

                // apply SWA if any
                if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                    continue;
                }

                data[idst + i0] = llama_cast<T>(hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f);
            }
        }

        if (debug) {
            print_mask(data, n_tokens, n_kv, n_swa, swa_type);
        }
    };

    GGML_ASSERT(self_kq_mask);
    GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));
    if (self_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    } else {
        fill_mask((float       *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
        if (self_kq_mask_swa->type == GGML_TYPE_F16) {
            fill_mask((ggml_fp16_t *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        } else {
            fill_mask((float       *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        }
    }
}

static bool tail_query_plan_has_explicit_tokens(const llama_ubatch & ubatch) {
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] <= 0 || ubatch.seq_id[i] == nullptr) {
            return false;
        }
    }
    return true;
}

static llama_seq_id tail_query_plan_primary_seq(
        const llama_ubatch & ubatch, uint32_t token, bool explicit_tokens) {
    if (explicit_tokens) {
        return ubatch.seq_id[token][0];
    }
    // graph_reserve() deliberately creates a lightweight equal-sequence
    // ubatch: only its synthetic sequence descriptors are initialized. The
    // token layout is still the normal stream-major equal-sequence layout.
    GGML_ASSERT(ubatch.equal_seqs() && ubatch.n_seq_tokens > 0 && ubatch.seq_id_unq != nullptr);
    const uint32_t stream = token/ubatch.n_seq_tokens;
    GGML_ASSERT(stream < ubatch.n_seqs_unq);
    return ubatch.seq_id_unq[stream];
}

static std::pair<int64_t, int64_t> tail_query_plan_shape(const llama_ubatch & ubatch) {
    const bool explicit_tokens = tail_query_plan_has_explicit_tokens(ubatch);
    std::map<llama_seq_id, int64_t> counts;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        ++counts[tail_query_plan_primary_seq(ubatch, i, explicit_tokens)];
    }
    int64_t q_max = 0;
    for (const auto & entry : counts) {
        q_max = std::max(q_max, entry.second);
    }
    return { q_max, int64_t(counts.size()) };
}

static void set_tail_query_plan(
        ggml_tensor * query_order, ggml_tensor * run_desc, const llama_ubatch & ubatch) {
    if (!query_order || !run_desc || !query_order->buffer || !run_desc->buffer) {
        return;
    }
    GGML_ASSERT(ggml_backend_buffer_is_host(query_order->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(run_desc->buffer));
    GGML_ASSERT(query_order->type == GGML_TYPE_I32 && run_desc->type == GGML_TYPE_I32 && run_desc->ne[0] >= 4);
    const bool explicit_tokens = tail_query_plan_has_explicit_tokens(ubatch);
    GGML_ASSERT(explicit_tokens);
    std::map<llama_seq_id, std::vector<int32_t>> queries;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        queries[tail_query_plan_primary_seq(ubatch, i, explicit_tokens)].push_back(int32_t(i));
    }
    GGML_ASSERT(query_order->ne[1] == int64_t(queries.size()) && run_desc->ne[1] == int64_t(queries.size()));
    auto * order = static_cast<int32_t *>(query_order->data);
    auto * desc = static_cast<int32_t *>(run_desc->data);
    std::fill(order, order + ggml_nelements(query_order), -1);
    std::fill(desc, desc + ggml_nelements(run_desc), -1);
    const int64_t desc_stride = run_desc->ne[0];
    std::vector<std::pair<llama_seq_id, int64_t>> layout;
    layout.reserve(queries.size());
    int64_t active = 0;
    for (const auto & entry : queries) {
        GGML_ASSERT(int64_t(entry.second.size()) <= query_order->ne[0]);
        std::copy(entry.second.begin(), entry.second.end(), order + active*query_order->ne[0]);
        layout.emplace_back(entry.first, int64_t(entry.second.size()));
        ++active;
    }
    for (int64_t ia = 0; ia < int64_t(layout.size()); ++ia) {
        int64_t run = 1;
        while (ia + run < int64_t(layout.size()) &&
                layout[ia + run].first == layout[ia].first + run &&
                layout[ia + run].second == layout[ia].second) {
            ++run;
        }
        desc[desc_stride*ia + 0] = layout[ia].first;
        desc[desc_stride*ia + 1] = int32_t(ia*query_order->ne[0]);
        desc[desc_stride*ia + 2] = int32_t(layout[ia].second);
        desc[desc_stride*ia + 3] = int32_t(run);
    }
}

llm_graph_kv_tail_identity llm_graph_kv_tail_identity::capture(
        const llama_hparams & hparams,
        const llama_kv_cache_context * mctx) {
    GGML_ASSERT(mctx != nullptr);
    llm_graph_kv_tail_identity result;
    result.storage_kind = uint32_t(mctx->get_tail_storage_kind());
    result.exact_type = mctx->get_tail_type();
    result.retention_tokens = mctx->get_tail_tokens();
    result.rollback_tokens = mctx->get_tail_rollback_tokens();
    result.arena_stride = mctx->get_tail_arena_stride();
    result.storage_slots = mctx->get_tail_slots();
    result.compact = mctx->has_compact_tail();
    if (result.storage_kind == LLAMA_KV_TAIL_STORAGE_DISABLED) {
        return result;
    }
    result.has_body.reserve(hparams.n_layer());
    result.has_current.reserve(hparams.n_layer());
    result.body_execution_rows.reserve(hparams.n_layer());
    result.routes.reserve(hparams.n_layer());
    result.explicit_bias.reserve(hparams.n_layer());
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        const auto route = mctx->get_tail_route(int32_t(il));
        result.routes.push_back(route);
        result.has_body.push_back(route == LLAMA_KV_TAIL_ROUTE_NONE || mctx->has_kv_body(int32_t(il)));
        result.has_current.push_back(
                route != LLAMA_KV_TAIL_ROUTE_NONE && mctx->has_tail_current(int32_t(il)));
        result.body_execution_rows.push_back(
                route == LLAMA_KV_TAIL_ROUTE_NONE ? 0 : mctx->get_tail_body_execution_rows(int32_t(il)));
        result.explicit_bias.push_back(
                route != LLAMA_KV_TAIL_ROUTE_NONE && mctx->get_tail_explicit_bias(int32_t(il)));
    }
    return result;
}

bool llm_graph_kv_tail_identity::matches(
        const llama_hparams & hparams,
        const llama_kv_cache_context * mctx) const {
    if (mctx == nullptr || storage_kind != uint32_t(mctx->get_tail_storage_kind()) ||
            exact_type != mctx->get_tail_type() ||
            retention_tokens != mctx->get_tail_tokens() ||
            rollback_tokens != mctx->get_tail_rollback_tokens() ||
            arena_stride != mctx->get_tail_arena_stride() ||
            storage_slots != mctx->get_tail_slots() ||
            compact != mctx->has_compact_tail()) {
        return false;
    }
    if (storage_kind == LLAMA_KV_TAIL_STORAGE_DISABLED) {
        return true;
    }
    if (has_body.size() != hparams.n_layer() ||
            has_current.size() != hparams.n_layer() ||
            body_execution_rows.size() != hparams.n_layer() ||
            routes.size() != hparams.n_layer() || explicit_bias.size() != hparams.n_layer()) {
        return false;
    }
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        const auto route = mctx->get_tail_route(int32_t(il));
        const bool current_has_body = route == LLAMA_KV_TAIL_ROUTE_NONE || mctx->has_kv_body(int32_t(il));
        const bool current_has_current =
                route != LLAMA_KV_TAIL_ROUTE_NONE && mctx->has_tail_current(int32_t(il));
        const uint32_t current_body_rows =
                route == LLAMA_KV_TAIL_ROUTE_NONE ? 0 : mctx->get_tail_body_execution_rows(int32_t(il));
        const bool current_explicit_bias =
                route != LLAMA_KV_TAIL_ROUTE_NONE && mctx->get_tail_explicit_bias(int32_t(il));
        if (has_body[il] != current_has_body || has_current[il] != current_has_current ||
                body_execution_rows[il] != current_body_rows ||
                routes[il] != route || explicit_bias[il] != current_explicit_bias) {
            return false;
        }
    }
    return true;
}

llm_graph_input_attn_kv_iswa::llm_graph_input_attn_kv_iswa(
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_iswa_context * mctx) :
    hparams(hparams),
    cparams(cparams),
    mctx(mctx),
    base_tail_identity(llm_graph_kv_tail_identity::capture(hparams, mctx->get_base())),
    swa_tail_identity(llm_graph_kv_tail_identity::capture(hparams, mctx->get_swa())) {
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);
    mctx->set_input_tail_idxs(self_tail_idxs, ubatch);
    set_tail_query_plan(self_tail_query_order, self_tail_run_desc, *ubatch);
    if (self_kvarn_mat_idxs && self_kvarn_mat_idxs->buffer) {
        const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx);
        GGML_ASSERT(kvarn != nullptr);
        kvarn->set_input_kvarn_mat_idxs(self_kvarn_mat_idxs, ubatch);
    }

    // the mask is left unallocated when the graph only stores K/V without attending
    // (e.g. DFlash's KV-injection pass)
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
        mctx->set_input_kq_mask_tail(self_kq_mask, self_kq_mask_tail,
                self_tail_read_idxs, self_tail_body_read_idxs, self_tail_bias_read_idxs,
                ubatch, cparams.causal_attn);
        mctx->set_input_tail_body_plan(
                self_tail_query_order, self_tail_run_desc, self_kq_mask,
                ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->set_input_v_rot(self_v_rot);
    }

    if ((self_kvarn_rot_128 && self_kvarn_rot_128->buffer) ||
            (self_kvarn_rot_256 && self_kvarn_rot_256->buffer) ||
            (self_kvarn_rot_512 && self_kvarn_rot_512->buffer)) {
        const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx);
        GGML_ASSERT(kvarn != nullptr);
        llm_kvarn_set_rot_inputs(kvarn,
                self_kvarn_rot_128, self_kvarn_rot_256, self_kvarn_rot_512);
    }
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;
    res &= tail_identity.matches(hparams, mctx);
    const auto [q_max, n_active] = tail_query_plan_shape(params.ubatch);
    res &= self_tail_query_order == nullptr ||
            (self_tail_query_order->ne[0] == q_max && self_tail_query_order->ne[1] == n_active);
    const uint32_t tail_attention_stride = mctx->get_tail_attention_stride(uint32_t(q_max));
    const int64_t tail_desc_stride = 6 + tail_attention_stride +
            (mctx->can_pack_tail_body(params.ubatch) ? mctx->get_tail_body_execution_stride() : 0);
    res &= self_tail_run_desc == nullptr ||
            (self_tail_run_desc->ne[0] == tail_desc_stride && self_tail_run_desc->ne[1] == n_active);

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    res &= self_tail_idxs == nullptr || self_tail_idxs->ne[0] == params.ubatch.n_tokens;
    res &= self_kvarn_mat_idxs == nullptr || self_kvarn_mat_idxs->ne[0] == mctx->get_n_kv();
    res &= self_tail_read_idxs == nullptr ||
            (self_tail_read_idxs->ne[0] == tail_attention_stride &&
             self_tail_read_idxs->ne[1] == params.ubatch.n_tokens);
    res &= self_tail_body_read_idxs == nullptr ||
            (self_tail_body_read_idxs->ne[0] == tail_attention_stride &&
             self_tail_body_read_idxs->ne[1] == params.ubatch.n_tokens);
    res &= self_tail_bias_read_idxs == nullptr ||
            (self_tail_bias_read_idxs->ne[0] == tail_attention_stride &&
             self_tail_bias_read_idxs->ne[1] == params.ubatch.n_tokens);
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);
    res &= self_kq_mask_tail == nullptr ||
            (self_kq_mask_tail->ne[0] == tail_attention_stride &&
             self_kq_mask_tail->ne[1] == self_kq_mask->ne[1] &&
             self_kq_mask_tail->ne[3] == self_kq_mask->ne[3]);

    if (self_kvarn_mat_idxs && self_kvarn_mat_idxs->buffer) {
        const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx);
        GGML_ASSERT(kvarn != nullptr);
        kvarn->set_mat_idxs(self_kvarn_mat_idxs);
    }

    return res;
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

llm_graph_input_attn_kv_msa::llm_graph_input_attn_kv_msa(
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_msa_context * mctx) :
    llm_graph_input_attn_kv(hparams, cparams, mctx->get_base()),
    mctx_msa(mctx) {
}

void llm_graph_input_attn_kv_msa::set_input(const llama_ubatch * ubatch) {
    llm_graph_input_attn_kv::set_input(ubatch);

    if (self_k_idxs_idx) {
        mctx_msa->get_idx()->set_input_k_idxs(self_k_idxs_idx, ubatch);
    }
}

bool llm_graph_input_attn_kv_msa::can_reuse(const llm_graph_params & params) {
    mctx_msa = static_cast<const llama_kv_cache_msa_context *>(params.mctx);

    // the parent class operates on the base cache context
    this->mctx = mctx_msa->get_base();

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    if (self_k_idxs_idx) {
        res &= self_k_idxs_idx->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(self_kq_mask, this->mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    mctx->get_mla()->set_input_kq_mask(self_kq_mask_mla, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
}

bool llm_graph_input_attn_k_dsa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs_mla->ne[0] == params.ubatch.n_tokens;
    res &= self_k_idxs_lid->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask_mla, mctx->get_mla(), params.ubatch, params.cparams);
    res &= can_reuse_kq_mask(self_kq_mask_lid, mctx->get_lid(), params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    set_tail_query_plan(self_tail_query_order, self_tail_run_desc, *ubatch);
    set_tail_query_plan(self_tail_query_order_swa, self_tail_run_desc_swa, *ubatch);
    const auto set_group = [&](const llama_kv_cache_context * group, bool swa) {
        ggml_tensor * k_idxs = swa ? self_k_idxs_swa : self_k_idxs;
        ggml_tensor * v_idxs = swa ? self_v_idxs_swa : self_v_idxs;
        ggml_tensor * tail_idxs = get_tail_idxs(swa);
        ggml_tensor * mask = swa ? self_kq_mask_swa : self_kq_mask;
        ggml_tensor * kvarn_mat_idxs = swa ? self_kvarn_mat_idxs_swa : self_kvarn_mat_idxs;
        if (k_idxs && k_idxs->buffer) {
            group->set_input_k_idxs(k_idxs, ubatch);
            if (v_idxs) {
                group->set_input_v_idxs(v_idxs, ubatch);
            }
        }
        if (tail_idxs && tail_idxs->buffer) {
            group->set_input_tail_idxs(tail_idxs, ubatch);
        }
        if (kvarn_mat_idxs && kvarn_mat_idxs->buffer) {
            const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(group);
            GGML_ASSERT(kvarn != nullptr);
            kvarn->set_input_kvarn_mat_idxs(kvarn_mat_idxs, ubatch);
        }
        if (mask && mask->buffer) {
            group->set_input_kq_mask(mask, ubatch, cparams.causal_attn);
            group->set_input_kq_mask_tail(mask, get_kq_mask_tail(swa),
                    get_tail_read_idxs(swa), get_tail_body_read_idxs(swa),
                    get_tail_bias_read_idxs(swa), ubatch, cparams.causal_attn);
            group->set_input_tail_body_plan(
                    self_tail_query_order, get_tail_run_desc(swa), mask,
                    ubatch, cparams.causal_attn);
        }
    };
    set_group(mctx->get_base(), false);
    set_group(mctx->get_swa(), true);

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->get_base()->set_input_v_rot(self_v_rot);
    }

    if ((self_kvarn_rot_128 && self_kvarn_rot_128->buffer) ||
            (self_kvarn_rot_256 && self_kvarn_rot_256->buffer) ||
            (self_kvarn_rot_512 && self_kvarn_rot_512->buffer)) {
        const auto * kvarn_base = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx->get_base());
        const auto * kvarn_swa  = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx->get_swa());
        GGML_ASSERT(kvarn_base != nullptr || kvarn_swa != nullptr);
        llm_kvarn_set_rot_inputs(kvarn_base ? kvarn_base : kvarn_swa,
                self_kvarn_rot_128, self_kvarn_rot_256, self_kvarn_rot_512);
    }

    if (self_kvarn_mat_idxs_swa && self_kvarn_mat_idxs_swa->buffer) {
        const auto * kvarn_swa = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx->get_swa());
        GGML_ASSERT(kvarn_swa != nullptr);
        kvarn_swa->set_input_kvarn_mat_idxs(self_kvarn_mat_idxs_swa, ubatch);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }

    if (self_v_rot_swa && self_v_rot_swa->buffer) {
        mctx->get_swa()->set_input_v_rot(self_v_rot_swa);
    }
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;
    res &= base_tail_identity.matches(hparams, mctx->get_base());
    res &= swa_tail_identity.matches(hparams, mctx->get_swa());
    // plain locals instead of structured bindings: the lambda below captures
    // them, which C++17 does not allow for structured bindings
    const auto tail_shape = tail_query_plan_shape(params.ubatch);
    const int64_t q_max    = tail_shape.first;
    const int64_t n_active = tail_shape.second;
    res &= self_tail_query_order == nullptr ||
            (self_tail_query_order->ne[0] == q_max && self_tail_query_order->ne[1] == n_active);
    res &= self_tail_query_order_swa == nullptr ||
            (self_tail_query_order_swa->ne[0] == q_max && self_tail_query_order_swa->ne[1] == n_active);
    const auto reuse_tail = [&](const llama_kv_cache_context * group, bool swa) {
        const uint32_t attention_stride = group->get_tail_attention_stride(uint32_t(q_max));
        ggml_tensor * read_idxs = get_tail_read_idxs(swa);
        ggml_tensor * body_idxs = get_tail_body_read_idxs(swa);
        ggml_tensor * bias_idxs = get_tail_bias_read_idxs(swa);
        ggml_tensor * tail_mask = get_kq_mask_tail(swa);
        ggml_tensor * body_mask = swa ? self_kq_mask_swa : self_kq_mask;
        bool valid = true;
        for (ggml_tensor * indices : { read_idxs, body_idxs, bias_idxs }) {
            valid &= indices == nullptr ||
                    (indices->ne[0] == attention_stride &&
                     indices->ne[1] == params.ubatch.n_tokens);
        }
        valid &= tail_mask == nullptr ||
                (tail_mask->ne[0] == attention_stride &&
                 tail_mask->ne[1] == body_mask->ne[1] && tail_mask->ne[3] == body_mask->ne[3]);
        const int64_t desc_stride = 6 + attention_stride +
                (group->can_pack_tail_body(params.ubatch) ? group->get_tail_body_execution_stride() : 0);
        ggml_tensor * run_desc = get_tail_run_desc(swa);
        valid &= run_desc == nullptr ||
                (run_desc->ne[0] == desc_stride && run_desc->ne[1] == n_active);
        return valid;
    };

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }
    res &= reuse_tail(mctx->get_base(), false);
    res &= reuse_tail(mctx->get_swa(), true);

    // Re-bind the SWA KVarN view indices to the new per-batch context before
    // graph construction asks it for native K/V views.
    if (self_kvarn_mat_idxs_swa && self_kvarn_mat_idxs_swa->buffer) {
        if (const auto * kvarn_swa = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx->get_swa())) {
            kvarn_swa->set_mat_idxs(self_kvarn_mat_idxs_swa);
        }
    }
    if (self_kvarn_mat_idxs && self_kvarn_mat_idxs->buffer) {
        if (const auto * kvarn_base = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx->get_base())) {
            kvarn_base->set_mat_idxs(self_kvarn_mat_idxs);
        }
    }

    return res;
}

void llm_graph_input_attn_k_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }
}

bool llm_graph_input_attn_k_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

static void dsv4_set_i64(ggml_tensor * dst, const std::vector<int64_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_i32(ggml_tensor * dst, const std::vector<int32_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_kq_mask(
        ggml_tensor * dst,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);
    GGML_ASSERT(dst->ne[0] == plan.n_kv);
    GGML_ASSERT(dst->ne[1] == (int64_t) n_tokens/n_stream);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == n_stream);
    GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) n_tokens);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    if (dst->type == GGML_TYPE_F32) {
        float * data = (float *) dst->data;

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? 0.0f : -INFINITY;
            }
        }
    } else if (dst->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *) dst->data;
        const ggml_fp16_t fp16_ninf = llama_cast<ggml_fp16_t>(-INFINITY);
        const ggml_fp16_t fp16_zero = llama_cast<ggml_fp16_t>(0.0f);

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? fp16_zero : fp16_ninf;
            }
        }
    }
}

static ggml_tensor * dsv4_build_raw_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);

    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool dsv4_can_reuse_raw_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static bool dsv4_compress_debug() {
    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    return debug;
}

static void dsv4_set_comp_inputs(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        bool debug,
        uint32_t n_tokens,
        int64_t n_stream) {
    dsv4_set_i32(inp.state_pos, plan.state_pos);
    dsv4_set_i32(inp.state_persist_src_idxs, plan.state_persist_src_idxs);
    dsv4_set_i32(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs);
    dsv4_set_i32(inp.state_restore_src_idxs, plan.state_restore_src_idxs);
    dsv4_set_i32(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs);
    dsv4_set_i32(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs);
    dsv4_set_i32(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs);
    dsv4_set_i32(inp.state_read_idxs, plan.state_read_idxs);
    dsv4_set_i64(inp.state_write_idxs, plan.state_write_idxs);
    dsv4_set_i32(inp.state_write_pos, plan.state_write_pos);
    dsv4_set_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    if (debug || dsv4_compress_debug()) {
        LLAMA_LOG_INFO("%s: %s n_tokens=%u, n_stream=%d, state_persist_dst=%s, state_write_pos=%s\n",
                __func__, name, n_tokens, (int) n_stream,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_pos).c_str());
    }
}

static bool dsv4_can_reuse_tensor_1d(ggml_tensor * t, int64_t ne0) {
    return (t == nullptr && ne0 == 0) || (t != nullptr && t->ne[0] == ne0);
}

static bool dsv4_can_reuse_kq_mask(
        ggml_tensor * t,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (plan.n_kv == 0) {
        return t == nullptr;
    }

    GGML_ASSERT(n_stream > 0);

    return t != nullptr &&
           t->ne[0] == plan.n_kv &&
           t->ne[1] == (int64_t) n_tokens/n_stream &&
           t->ne[2] == 1 &&
           t->ne[3] == n_stream;
}

static bool dsv4_can_reuse_comp_input(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    bool res = true;
    res &= dsv4_can_reuse_tensor_1d(inp.state_pos, plan.state_pos.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_src_idxs, plan.state_persist_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_src_idxs, plan.state_restore_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_read_idxs, plan.state_read_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_idxs, plan.state_write_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_pos, plan.state_write_pos.size());
    res &= dsv4_can_reuse_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    return res;
}

static ggml_tensor * dsv4_build_input_1d(
        ggml_context * ctx,
        ggml_type type,
        int64_t ne0,
        const std::string & name) {
    if (ne0 == 0) {
        return nullptr;
    }

    ggml_tensor * res = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(res);
    ggml_set_name(res, name.c_str());

    return res;
}

static void dsv4_build_comp_inputs(
        ggml_context * ctx,
        llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        const llama_cparams & cparams,
        int64_t n_stream) {
    inp.state_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_pos.size(), std::string("dsv4_") + name + "_state_pos");
    inp.state_persist_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_src_idxs.size(), std::string("dsv4_") + name + "_state_persist_src_idxs");
    inp.state_persist_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_dst_idxs.size(), std::string("dsv4_") + name + "_state_persist_dst_idxs");
    inp.state_restore_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_src_idxs.size(), std::string("dsv4_") + name + "_state_restore_src_idxs");
    inp.state_restore_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_dst_idxs.size(), std::string("dsv4_") + name + "_state_restore_dst_idxs");
    inp.state_snapshot_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_src_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_src_idxs");
    inp.state_snapshot_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_dst_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_dst_idxs");
    inp.state_read_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_read_idxs.size(), std::string("dsv4_") + name + "_state_read_idxs");
    inp.state_write_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I64, plan.state_write_idxs.size(), std::string("dsv4_") + name + "_state_write_idxs");
    inp.state_write_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_write_pos.size(), std::string("dsv4_") + name + "_state_write_pos");

    if (plan.n_kv > 0) {
        const int64_t n_tokens = (int64_t) plan.n_visible.size();

        GGML_ASSERT(n_stream > 0);
        GGML_ASSERT(n_tokens%n_stream == 0);

        inp.kq_mask = ggml_new_tensor_4d(ctx, (strcmp(name, "lid") != 0 && cparams.flash_attn) || (strcmp(name, "lid") == 0 && cparams.fused_lid) ? GGML_TYPE_F16 : GGML_TYPE_F32, plan.n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp.kq_mask);
        ggml_set_name(inp.kq_mask, (std::string("dsv4_") + name + "_kq_mask").c_str());
    }
}

void llm_graph_input_dsv4_raw::set_input(const llama_ubatch * ubatch) {
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs);
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
        mctx->set_input_kq_mask_tail(self_kq_mask, self_kq_mask_tail,
                self_tail_read_idxs, self_tail_body_read_idxs, self_tail_bias_read_idxs,
                ubatch, cparams.causal_attn);
    }

    mctx->set_input_tail_idxs(self_tail_idxs, ubatch);

    if (self_k_rot) {
        mctx->set_input_k_rot(self_k_rot);
    }
}

void llm_graph_input_dsv4::set_input(const llama_ubatch * ubatch) {
    const auto & plan_csa = mctx->get_csa_plan(*ubatch);
    const auto & plan_hca = mctx->get_hca_plan(*ubatch);
    const auto & plan_lid = mctx->get_lid_plan(*ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    inp_raw->mctx = mctx->get_raw();
    inp_raw->set_input(ubatch);

    dsv4_set_comp_inputs(inp_csa, plan_csa, "csa", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_hca, plan_hca, "hca", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_lid, plan_lid, "lid", debug > 0, ubatch->n_tokens, n_stream);

    if (inp_csa.k_rot && inp_csa.k_rot->buffer) {
        mctx->get_csa()->set_input_k_rot(inp_csa.k_rot);
    }

    if (inp_hca.k_rot && inp_hca.k_rot->buffer) {
        mctx->get_hca()->set_input_k_rot(inp_hca.k_rot);
    }

    if (inp_lid.k_rot && inp_lid.k_rot->buffer) {
        mctx->get_lid()->set_input_k_rot(inp_lid.k_rot);
    }
}

bool llm_graph_input_dsv4::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

    this->mctx = mctx;
    inp_raw->mctx = mctx->get_raw();

    bool res = true;

    const auto & plan_csa = mctx->get_csa_plan(params.ubatch);
    const auto & plan_hca = mctx->get_hca_plan(params.ubatch);
    const auto & plan_lid = mctx->get_lid_plan(params.ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    const auto * raw_ctx = mctx->get_raw();
    inp_raw->mctx = raw_ctx;
    const auto [tail_q_max, tail_n_active] = tail_query_plan_shape(params.ubatch);
    GGML_UNUSED(tail_n_active);
    const uint32_t tail_attention_stride = raw_ctx->get_tail_attention_stride(uint32_t(tail_q_max));

    if (inp_raw->self_k_idxs && inp_raw->self_k_idxs->buffer) {
        res &= inp_raw->self_k_idxs->ne[0] == raw_ctx->get_n_write();
    }
    res &= inp_raw->self_tail_idxs == nullptr || inp_raw->self_tail_idxs->ne[0] == raw_ctx->get_n_write();
    res &= inp_raw->self_tail_read_idxs == nullptr ||
            (inp_raw->self_tail_read_idxs->ne[0] == tail_attention_stride &&
             inp_raw->self_tail_read_idxs->ne[1] == params.ubatch.n_tokens);
    res &= inp_raw->self_tail_body_read_idxs == nullptr ||
            (inp_raw->self_tail_body_read_idxs->ne[0] == tail_attention_stride &&
             inp_raw->self_tail_body_read_idxs->ne[1] == params.ubatch.n_tokens);
    res &= inp_raw->self_tail_bias_read_idxs == nullptr ||
            (inp_raw->self_tail_bias_read_idxs->ne[0] == tail_attention_stride &&
             inp_raw->self_tail_bias_read_idxs->ne[1] == params.ubatch.n_tokens);
    if (inp_raw->self_kq_mask && inp_raw->self_kq_mask->buffer) {
        res &= dsv4_can_reuse_raw_kq_mask(inp_raw->self_kq_mask, raw_ctx, params.ubatch, n_stream);
    }
    res &= inp_raw->self_kq_mask_tail == nullptr ||
            (inp_raw->self_kq_mask_tail->ne[0] == tail_attention_stride &&
             inp_raw->self_kq_mask_tail->ne[1] == inp_raw->self_kq_mask->ne[1] &&
             inp_raw->self_kq_mask_tail->ne[3] == inp_raw->self_kq_mask->ne[3]);

    res &= dsv4_can_reuse_comp_input(inp_csa, plan_csa, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_hca, plan_hca, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_lid, plan_lid, params.ubatch.n_tokens, n_stream);

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    const auto fill_mask = [&](auto * data) {
        using T = std::remove_reference_t<decltype(*data)>;
        for (int i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first");
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[i*n_enc + j] = llama_cast<T>(f);
            }
        }
    };

    if (cross_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) cross_kq_mask->data);
    } else {
        fill_mask((float *) cross_kq_mask->data);
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    inp_attn->mctx = mctx->get_attn();
    inp_attn->set_input(ubatch);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;
    inp_attn->mctx = mctx->get_attn();

    bool res = true;

    const auto [tail_q_max, tail_n_active] = tail_query_plan_shape(params.ubatch);
    res &= inp_attn->self_tail_query_order == nullptr ||
            (inp_attn->self_tail_query_order->ne[0] == tail_q_max &&
             inp_attn->self_tail_query_order->ne[1] == tail_n_active);
    const uint32_t tail_attention_stride = mctx->get_attn()->get_tail_attention_stride(uint32_t(tail_q_max));
    const int64_t tail_desc_stride = 6 + tail_attention_stride +
            (mctx->get_attn()->can_pack_tail_body(params.ubatch) ?
                    mctx->get_attn()->get_tail_body_execution_stride() : 0);
    res &= inp_attn->self_tail_run_desc == nullptr ||
            (inp_attn->self_tail_run_desc->ne[0] == tail_desc_stride &&
             inp_attn->self_tail_run_desc->ne[1] == tail_n_active);

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);
    res &= inp_attn->self_tail_read_idxs == nullptr ||
            (inp_attn->self_tail_read_idxs->ne[0] == tail_attention_stride &&
             inp_attn->self_tail_read_idxs->ne[1] == params.ubatch.n_tokens);
    res &= inp_attn->self_tail_body_read_idxs == nullptr ||
            (inp_attn->self_tail_body_read_idxs->ne[0] == tail_attention_stride &&
             inp_attn->self_tail_body_read_idxs->ne[1] == params.ubatch.n_tokens);
    res &= inp_attn->self_tail_bias_read_idxs == nullptr ||
            (inp_attn->self_tail_bias_read_idxs->ne[0] == tail_attention_stride &&
             inp_attn->self_tail_bias_read_idxs->ne[1] == params.ubatch.n_tokens);
    res &= inp_attn->self_kq_mask_tail == nullptr ||
            (inp_attn->self_kq_mask_tail->ne[0] == tail_attention_stride &&
             inp_attn->self_kq_mask_tail->ne[1] == inp_attn->self_kq_mask->ne[1] &&
             inp_attn->self_kq_mask_tail->ne[3] == inp_attn->self_kq_mask->ne[3]);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_mem_hybrid_iswa::set_input(const llama_ubatch * ubatch) {
    inp_attn->mctx = mctx->get_attn();
    inp_attn->set_input(ubatch);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // plain locals instead of structured bindings: the lambda below captures
    // them, which C++17 does not allow for structured bindings
    const auto tail_shape = tail_query_plan_shape(params.ubatch);
    const int64_t tail_q_max    = tail_shape.first;
    const int64_t tail_n_active = tail_shape.second;
    res &= inp_attn->self_tail_query_order == nullptr ||
            (inp_attn->self_tail_query_order->ne[0] == tail_q_max &&
             inp_attn->self_tail_query_order->ne[1] == tail_n_active);
    res &= inp_attn->self_tail_query_order_swa == nullptr ||
            (inp_attn->self_tail_query_order_swa->ne[0] == tail_q_max &&
             inp_attn->self_tail_query_order_swa->ne[1] == tail_n_active);

    const auto * attn_ctx = mctx->get_attn();
    inp_attn->mctx = attn_ctx;
    const auto reuse_tail = [&](const llama_kv_cache_context * group, bool swa) {
        const uint32_t attention_stride = group->get_tail_attention_stride(uint32_t(tail_q_max));
        bool valid = true;
        for (ggml_tensor * indices : {
                inp_attn->get_tail_read_idxs(swa), inp_attn->get_tail_body_read_idxs(swa),
                inp_attn->get_tail_bias_read_idxs(swa) }) {
            valid &= indices == nullptr ||
                    (indices->ne[0] == attention_stride &&
                     indices->ne[1] == params.ubatch.n_tokens);
        }
        ggml_tensor * tail_mask = inp_attn->get_kq_mask_tail(swa);
        ggml_tensor * body_mask = swa ? inp_attn->self_kq_mask_swa : inp_attn->self_kq_mask;
        valid &= tail_mask == nullptr ||
                (tail_mask->ne[0] == attention_stride &&
                 tail_mask->ne[1] == body_mask->ne[1] && tail_mask->ne[3] == body_mask->ne[3]);
        const int64_t desc_stride = 6 + attention_stride +
                (group->can_pack_tail_body(params.ubatch) ? group->get_tail_body_execution_stride() : 0);
        ggml_tensor * run_desc = inp_attn->get_tail_run_desc(swa);
        valid &= run_desc == nullptr ||
                (run_desc->ne[0] == desc_stride && run_desc->ne[1] == tail_n_active);
        return valid;
    };

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, attn_ctx->get_base(), params.ubatch, params.cparams);

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask_swa, attn_ctx->get_swa(), params.ubatch, params.cparams);
    res &= reuse_tail(attn_ctx->get_base(), false);
    res &= reuse_tail(attn_ctx->get_swa(), true);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_h_nextn     = nullptr;

    t_layer_inp.resize(LLAMA_MAX_LAYERS + 1);
    std::fill(t_layer_inp.begin(), t_layer_inp.end(), nullptr);

    t_sampled.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();

    params = {};

    inputs.clear();
    fused_nodes.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

void llm_graph_result::set_outputs(const llm_graph_params & params) {
    if (t_logits != nullptr) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    if (t_h_nextn != nullptr) {
        ggml_set_output(t_h_nextn);
    }
    {
        const auto & embeddings_layer_inp = params.cparams.embeddings_layer_inp;
        for (size_t il = 0; il < embeddings_layer_inp.size(); ++il) {
            if (embeddings_layer_inp[il]) {
                GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null");
                ggml_set_output(t_layer_inp[il]);
            }
        }
    }
    for (auto * tensor : t_sampled) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_sampled_probs) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_sampled_logits) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_candidates) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::add_fused_node(llm_graph_fused_node result) {
    fused_nodes.push_back(result);
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer()),
    n_layer_nextn    (hparams.n_layer_nextn),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    samplers         (params.samplers),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}



ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);

    if (w_s) {
        const int64_t n_expert = w_s->ne[0];
        const int64_t n_tokens = cur->ne[2];
        ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, ids);
        res = ggml_mul(ctx0, res, s);
    }
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}


llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const {
    const int64_t n_embd_q  = n_embd_head * n_head;
    const int64_t n_embd_kv = n_embd_head * n_head_kv;

    ggml_tensor * Qcur, * Kcur, * Vcur;

    if (layer.wqkv) {
        // fused QKV path
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
        cb(qkv, "wqkv", il);
        if (layer.wqkv_b) {
            qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
            cb(qkv, "wqkv_b", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(qkv, "wqkv_clamped", il);
        }
        Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], 0);
        Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q));
        Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q + n_embd_kv));
    } else {
        // separate Q/K/V path
        Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        cb(Qcur, "Qcur", il);
        if (layer.wq_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
            cb(Qcur, "Qcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Qcur, "Qcur_clamped", il);
        }
        Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        cb(Kcur, "Kcur", il);
        if (layer.wk_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
            cb(Kcur, "Kcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Kcur, "Kcur_clamped", il);
        }
        Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        cb(Vcur, "Vcur", il);
        if (layer.wv_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
            cb(Vcur, "Vcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Vcur, "Vcur_clamped", il);
        }
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    }

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    return { Qcur, Kcur, Vcur };
}


ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il) const {
    // NVFP4 support is currently restricted to
    // 1) LORA absence (*_s would be applied after LORA residual, which is incorrect)
    // 2) bias absense (*_s would be applied after bias addition, which is incorrect)
    // TODO: disambiguate LLM-architectural scales (which use *_s) from NVFP4 scale_2 (which also uses *_s currently)
    auto has_lora = [this](ggml_tensor * w) {
        if (!w) {
            return false;
        }
        for (const auto & lora : *loras) {
            if (lora.first->get_weight(w) != nullptr) {
                return true;
            }
        }
        return false;
    };

    GGML_ASSERT(!up_s   || !up_b   || !up   || up->type   != GGML_TYPE_NVFP4);
    GGML_ASSERT(!gate_s || !gate_b || !gate || gate->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!down_s || !down_b || !down || down->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!up_s   || !up   || up->type   != GGML_TYPE_NVFP4 || !has_lora(up));
    GGML_ASSERT(!gate_s || !gate || gate->type != GGML_TYPE_NVFP4 || !has_lora(gate));
    GGML_ASSERT(!down_s || !down || down->type != GGML_TYPE_NVFP4 || !has_lora(down));

    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                        cb(tmp, "ffn_up_clamped", il);

                        if (arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_clamp(ctx0, cur, -INFINITY, limit);
                            cb(cur, "ffn_gate_clamped", il);
                            cur = ggml_swiglu_split(ctx0, cur, tmp);
                        } else {
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, tmp);
                        }
                        cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            if (gate && type_gate == LLM_FFN_PAR) {
                // same alpha/limit constants as gpt-oss
                const float alpha = 1.702f;
                const float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, tmp, alpha, limit);
                cb(cur, "ffn_swiglu_oai", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                GGML_ABORT("LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        case LLM_FFN_SITU:
            GGML_ABORT("not yet supported");
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s,
        selected_experts_in
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
        }
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = selected_experts_in;
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;

    if (gate_up_exps) {
        // merged gate_up path: one mul_mat_id, then split into gate and up views
        ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts, up_exps_s); // [n_ff*2, n_expert_used, n_tokens]
        cb(gate_up, "ffn_moe_gate_up", il);

        if (up_exps_s) {
            cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
            cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = build_lora_mm_id(up_exps, cur, selected_experts, up_exps_s); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        if (up_exps_s) {
            cb(up, "ffn_moe_up_scaled", il);
        }

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
            cb(up, "ffn_moe_up_biased", il);
        }

        if (gate_exps) {
            cur = build_lora_mm_id(gate_exps, cur, selected_experts, gate_exps_s); // [n_ff, n_expert_used, n_tokens]
            cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_s) {
            cb(cur, "ffn_moe_gate_scaled", il);
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
            cb(cur, "ffn_moe_gate_biased", il);
        }
    }

    const bool has_gate = gate_exps || gate_up_exps;

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_exp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        up = ggml_clamp(ctx0, up, -limit, limit);
                        cb(up, "ffn_moe_up_clamped", il);

                        if (arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_clamp(ctx0, cur, -INFINITY, limit);
                            cb(cur, "ffn_moe_gate_clamped", il);
                            cur = ggml_swiglu_split(ctx0, cur, up);
                        } else {
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_moe_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_moe_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, up);
                        }
                        cb(cur, "ffn_moe_swiglu_limited", il);
                        break;
                    }
                }
            }

            if (has_gate) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_SITU:
            {
                // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
                GGML_ASSERT(has_gate);
                const float beta = hparams.situ_beta;
                const float lb   = hparams.situ_linear_beta;

                ggml_tensor * act = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/beta)), beta);
                act = ggml_mul(ctx0, act, ggml_sigmoid(ctx0, cur));
                if (lb > 0.0f) {
                    up = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, up, 1.0f/lb)), lb);
                }
                cur = ggml_mul(ctx0, act, up);
                cb(cur, "ffn_moe_situ", il);
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (has_gate) {
                // TODO: add support for gated squared relu
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    experts = build_lora_mm_id(down_exps, cur, selected_experts, down_exps_s); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    if (down_exps_s) {
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    // note: here we explicitly use hparams.n_expert_used instead of n_expert_used
    //       to avoid potentially a large number of add nodes during warmup
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14753
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

        ggml_build_forward_expand(gf, moe_out);
    }

    if (hparams.n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    cb(inp->tokens, "inp_tokens", -1);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    // select one of the 2 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 2> inps;

    // token embeddings path (ubatch.token != nullptr)
    {
        auto & cur = inps[0];

        cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, inp->tokens)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }
    }

    // vector embeddings path (ubatch.embd != nullptr)
    {
        auto & cur = inps[1];

        cur = inp->embd;
    }

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    // For Granite architecture
    // NOTE: For deepstack models, only apply scale to token inputs (ie text-only input).
    //  Raw embeddings are assumed to be multimodal inputs that should not be scaled.
    if (hparams.f_embedding_scale != 0.0f && (ubatch.token || hparams.n_deepstack_layers == 0)) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "attn_scale");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interfere with
    //       features that require constant topology such as pipeline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc  : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_bias_tail(
        ggml_tensor * kq_b, ggml_tensor * bias_read_idxs, ggml_tensor * kq_mask_tail) const {
    if (!kq_b || !kq_mask_tail) {
        return nullptr;
    }

    GGML_ASSERT(bias_read_idxs && bias_read_idxs->type == GGML_TYPE_I32);
    GGML_ASSERT(bias_read_idxs->ne[0] == kq_mask_tail->ne[0]);
    GGML_ASSERT(bias_read_idxs->ne[1] == kq_mask_tail->ne[1]*kq_mask_tail->ne[3]);
    GGML_ASSERT(kq_b->ne[2] == kq_mask_tail->ne[1] && kq_b->ne[3] == kq_mask_tail->ne[3]);

    // Repack [body, head, query, stream] into vectors of heads indexed by the
    // flattened (body, query, stream) row selected during mask preparation.
    ggml_tensor * bias_rows = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 1, 0, 2, 3));
    bias_rows = ggml_reshape_2d(ctx0, bias_rows, kq_b->ne[1],
            kq_b->ne[0]*kq_b->ne[2]*kq_b->ne[3]);
    ggml_tensor * flat_idxs = ggml_reshape_1d(ctx0, bias_read_idxs, ggml_nelements(bias_read_idxs));
    ggml_tensor * tail = ggml_get_rows(ctx0, bias_rows, flat_idxs);
    tail = ggml_reshape_4d(ctx0, tail, kq_b->ne[1], kq_mask_tail->ne[0],
            kq_mask_tail->ne[1], kq_mask_tail->ne[3]);
    return ggml_permute(ctx0, tail, 1, 2, 0, 3);
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
               float   kq_scale,
                 int   il,
         ggml_tensor * k_tail,
         ggml_tensor * v_tail,
         ggml_tensor * kq_mask_tail,
         ggml_tensor * kq_b_tail,
         ggml_tensor * tail_read_idxs,
         ggml_tensor * tail_query_order,
         ggml_tensor * tail_run_desc,
         llama_kv_tail_route tail_route,
         enum ggml_flash_attn_ext_kvarn_domain kvarn_domain,
         ggml_tensor * k_tail_current,
         ggml_tensor * v_tail_current,
              uint32_t tail_history_slots,
                   bool tail_bodyless,
        ggml_tensor ** final_attn_op) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];
    ggml_tensor * q_tail_batched = k_tail && !tail_read_idxs ? ggml_reshape_4d(
            ctx0, q, q->ne[0], q->ne[1], 1, q->ne[2]*q->ne[3]) : nullptr;
    if (q_tail_batched) {
        // Tail records are gathered per query, so the query index is the
        // outer batch axis while GQA heads retain the regular matmul axis.
        q_tail_batched = ggml_permute(ctx0, q_tail_batched, 0, 2, 1, 3);
    }

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);
    if (k_tail || v_tail || kq_mask_tail) {
        GGML_ASSERT(k_tail && v_tail && kq_mask_tail);
        k_tail = ggml_permute(ctx0, k_tail, 0, 2, 1, 3);
        v_tail = ggml_permute(ctx0, v_tail, 0, 2, 1, 3);
    }
    if (k_tail_current || v_tail_current) {
        GGML_ASSERT(k_tail_current && v_tail_current && k_tail);
        k_tail_current = ggml_permute(ctx0, k_tail_current, 0, 2, 1, 3);
        v_tail_current = ggml_permute(ctx0, v_tail_current, 0, 2, 1, 3);
    }

    ggml_tensor * cur;

    const bool use_native_tail = k_tail != nullptr && tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE;
    GGML_ASSERT(!k_tail || tail_route != LLAMA_KV_TAIL_ROUTE_NONE);
    GGML_ASSERT(!tail_read_idxs || use_native_tail);
    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr && (k_tail == nullptr || use_native_tail);
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        if (kvarn_domain != GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO) {
            cur->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN] = (int32_t) kvarn_domain;
        }
        if (use_native_tail) {
            GGML_ASSERT(tail_query_order && tail_run_desc);
            cur = k_tail_current ?
                    ggml_kv_tail_attention_merge_segmented(
                            ctx0, cur, k_tail, v_tail, k_tail_current, v_tail_current,
                            kq_mask_tail, tail_query_order, tail_run_desc) :
                    ggml_kv_tail_attention_merge(
                            ctx0, cur, k_tail, v_tail, kq_mask_tail, tail_query_order, tail_run_desc);
            if (tail_bodyless) {
                ggml_flash_attn_ext_set_kv_tail_bodyless(cur);
            }
            if (k_tail_current) {
                ggml_flash_attn_ext_set_kv_tail_history_slots(cur, int32_t(tail_history_slots));
            }
        }
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);

        if (final_attn_op) {
            *final_attn_op = cur;
        }

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        GGML_ASSERT(!tail_read_idxs && "indexed KV tails require native backend attention support");
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        cb(kq, "kq", il);

        const int64_t n_kv_body = kq->ne[0];
        if (k_tail) {
            ggml_tensor * kq_tail = ggml_mul_mat(ctx0, k_tail, q_tail_batched);
            ggml_mul_mat_set_prec(kq_tail, GGML_PREC_F32);
            kq_tail = ggml_reshape_4d(ctx0, kq_tail,
                    kq_tail->ne[0], q->ne[2], q->ne[1], q->ne[3]);
            kq_tail = ggml_permute(ctx0, kq_tail, 0, 2, 1, 3);
            kq = ggml_concat(ctx0, kq, kq_tail, 0);
            kq_mask = ggml_concat(ctx0, kq_mask, kq_mask_tail, 0);
            if (kq_b) {
                GGML_ASSERT(kq_b_tail);
                kq_b = ggml_concat(ctx0, kq_b, kq_b_tail, 0);
            }
        }

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        const bool quant_v_body = v_tail && ggml_is_quantized(v->type);
        if (!v_trans && !quant_v_body) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv;
        if (v_tail) {
            const int64_t n_kv_tail = kq->ne[0] - n_kv_body;
            ggml_tensor * weights_body = ggml_view_4d(ctx0, kq,
                    n_kv_body, kq->ne[1], kq->ne[2], kq->ne[3],
                    kq->nb[1], kq->nb[2], kq->nb[3], 0);
            ggml_tensor * weights_tail = ggml_view_4d(ctx0, kq,
                    n_kv_tail, kq->ne[1], kq->ne[2], kq->ne[3],
                    kq->nb[1], kq->nb[2], kq->nb[3], n_kv_body*kq->nb[0]);
            v_tail = ggml_cont(ctx0, ggml_transpose(ctx0, v_tail));
            ggml_tensor * body_out = quant_v_body ?
                    ggml_out_prod(ctx0, v, ggml_transpose(ctx0, weights_body)) :
                    ggml_mul_mat(ctx0, v, weights_body);
            weights_tail = ggml_cont(ctx0, ggml_permute(ctx0, weights_tail, 0, 2, 1, 3));
            weights_tail = ggml_reshape_4d(ctx0, weights_tail,
                    weights_tail->ne[0], 1, weights_tail->ne[1],
                    weights_tail->ne[2]*weights_tail->ne[3]);
            ggml_tensor * tail_out = ggml_mul_mat(ctx0, v_tail, weights_tail);
            tail_out = ggml_reshape_4d(ctx0, tail_out,
                    tail_out->ne[0], body_out->ne[2], body_out->ne[1], body_out->ne[3]);
            tail_out = ggml_permute(ctx0, tail_out, 0, 2, 1, 3);
            kqv = ggml_add(ctx0, body_out, tail_out);
        } else {
            kqv = ggml_mul_mat(ctx0, v, kq);
        }
        cb(kqv, "kqv", il);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

static void validate_native_tail_operation(
        const llama_kv_cache_context * mctx, int32_t il, const ggml_tensor * op) {
    const auto * route = mctx->get_tail_layer_route(il);
    if (!route || route->capability.route != LLAMA_KV_TAIL_ROUTE_NATIVE) {
        return;
    }
    if (!op || op->op != GGML_OP_FLASH_ATTN_EXT) {
        throw std::logic_error(format(
                "KV tail layer %d advertised native attention but built no final FlashAttention operation", il));
    }
    const bool has_current = op->src[10] != nullptr || op->src[11] != nullptr;
    if ((op->src[10] == nullptr) != (op->src[11] == nullptr) ||
            has_current != route->has_current) {
        throw std::logic_error(format(
                "KV tail layer %d current-segment descriptor does not match the final operation", il));
    }
    if (!op->src[1] || !op->src[2] || !op->src[5] || !op->src[6] ||
            op->src[1]->type != route->body_type_k || op->src[2]->type != route->body_type_v ||
            op->src[5]->type != route->exact_type_k || op->src[6]->type != route->exact_type_v ||
            route->has_body != mctx->has_kv_body(il) ||
            route->body_execution_rows != mctx->get_tail_body_execution_rows(il)) {
        throw std::logic_error(format(
                "KV tail layer %d execution descriptor does not match final tensor types or extents", il));
    }
    auto * dev = route->owner;
    if (!dev || !ggml_backend_dev_supports_op(dev, op)) {
        throw std::runtime_error(format(
                "KV tail layer %d native operation is unsupported by its planned backend %s; "
                "refusing scheduler fallback",
                il, dev ? ggml_backend_dev_name(dev) : "unknown"));
    }
    LLAMA_LOG_DEBUG("KV tail final: layer=%d dev=%s route=native body=%s current=%s "
            "body_type=%s/%s exact_type=%s/%s body_rows=%u supported=1\n",
            il, ggml_backend_dev_name(dev), mctx->has_kv_body(il) ? "present" : "bodyless",
            has_current ? "present" : "absent",
            ggml_type_name(op->src[1]->type), ggml_type_name(op->src[2]->type),
            ggml_type_name(op->src[5]->type), ggml_type_name(op->src[6]->type),
            mctx->get_tail_body_execution_rows(il));
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);

    inp->self_kq_mask_cnv = inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static void build_attn_inp_tail(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_kv_cache_context * mctx,
             ggml_tensor * kq_mask,
             ggml_tensor *& tail_idxs,
             ggml_tensor *& tail_read_idxs,
             ggml_tensor *& tail_body_read_idxs,
             ggml_tensor *& tail_bias_read_idxs,
             ggml_tensor *& kq_mask_tail,
             ggml_tensor *& tail_query_order,
             ggml_tensor *& tail_run_desc,
                    bool   pack_sparse_body) {
    tail_idxs = mctx->build_input_tail_idxs(ctx0, ubatch);
    if (mctx->get_tail_tokens() == 0) {
        return;
    }
    const auto [q_max, n_active] = tail_query_plan_shape(ubatch);
    GGML_ASSERT(q_max > 0 && n_active > 0);
    const uint32_t arena_stride = mctx->get_tail_arena_stride();
    const uint32_t body_execution_stride = mctx->get_tail_body_execution_stride();
    const uint32_t attention_stride = mctx->get_tail_attention_stride(uint32_t(q_max));
    GGML_ASSERT(attention_stride >= mctx->get_tail_tokens());
    GGML_ASSERT(mctx->has_compact_tail() || arena_stride >= attention_stride);
    tail_read_idxs = ggml_new_tensor_2d(
            ctx0, GGML_TYPE_I32, attention_stride, ubatch.n_tokens);
    tail_body_read_idxs = ggml_new_tensor_2d(
            ctx0, GGML_TYPE_I32, attention_stride, ubatch.n_tokens);
    tail_bias_read_idxs = ggml_new_tensor_2d(
            ctx0, GGML_TYPE_I32, attention_stride, ubatch.n_tokens);
    kq_mask_tail = ggml_new_tensor_4d(ctx0, kq_mask->type,
            attention_stride, kq_mask->ne[1], 1, kq_mask->ne[3]);
    for (ggml_tensor * input : {
            tail_read_idxs, tail_body_read_idxs, tail_bias_read_idxs, kq_mask_tail }) {
        ggml_set_input(input);
    }
    if (!tail_query_order) {
        tail_query_order = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, q_max, n_active);
        ggml_set_input(tail_query_order);
    }
    if (!tail_run_desc) {
        const bool sparse_body = pack_sparse_body && mctx->can_pack_tail_body(ubatch);
        GGML_ASSERT(!sparse_body || body_execution_stride >= arena_stride);
        const int64_t desc_stride = sparse_body ?
                int64_t(6 + attention_stride + body_execution_stride) : int64_t(6 + attention_stride);
        tail_run_desc = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, desc_stride, n_active);
        ggml_set_input(tail_run_desc);
        LLAMA_LOG_DEBUG("%s: standard KV tail body route=%s physical-capacity-bound=%u\n",
                __func__, sparse_body ? "sparse-packed" : "ordinary", body_execution_stride);
    }
}

static ggml_tensor * prepare_compact_tail_current(
        ggml_context * ctx0, ggml_tensor * persistent, ggml_tensor * current) {
    GGML_ASSERT(persistent && current);
    GGML_ASSERT(persistent->ne[0] == current->ne[0] &&
            persistent->ne[1] == current->ne[1] &&
            persistent->ne[3] == 1 && current->ne[3] == 1);
    if (current->type != persistent->type) {
        current = ggml_cast(ctx0, current, persistent->type);
    }
    current = ggml_cont(ctx0, current);
    current = ggml_reshape_4d(ctx0, current,
            persistent->ne[0], persistent->ne[1], current->ne[2], 1);
    return current;
}

static void build_empty_kv_body(
        ggml_context * ctx0,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_type body_type_k,
        ggml_type body_type_v,
        ggml_tensor * body_mask_template,
        ggml_tensor * body_bias_template,
        ggml_tensor *& k,
        ggml_tensor *& v,
        ggml_tensor *& body_mask,
        ggml_tensor *& body_bias) {
    GGML_ASSERT(k_cur && v_cur && body_mask_template);
    GGML_ASSERT((body_type_k == GGML_TYPE_F16 || body_type_k == GGML_TYPE_BF16) &&
                (body_type_v == GGML_TYPE_F16 || body_type_v == GGML_TYPE_BF16));
    const int64_t n_stream = body_mask_template->ne[3];
    const auto zero_anchor = [&](ggml_type type, const ggml_tensor * current) {
        ggml_tensor * anchor = ggml_fill(ctx0, ggml_new_tensor_4d(
                ctx0, GGML_TYPE_F16, current->ne[0], current->ne[1], 1, n_stream), 0.0f);
        return type == GGML_TYPE_F16 ? anchor : ggml_cast(ctx0, anchor, type);
    };
    const auto first_row_anchor = [&](ggml_tensor * source) {
        ggml_tensor * row = ggml_view_4d(ctx0, source,
                1, source->ne[1], source->ne[2], source->ne[3],
                source->nb[1], source->nb[2], source->nb[3], 0);
        return ggml_cont(ctx0, row);
    };
    k = zero_anchor(body_type_k, k_cur);
    v = zero_anchor(body_type_v, v_cur);
    // Keep the authoritative planner inputs live even though the body itself
    // is empty. Otherwise graph allocation prunes them and set_input() cannot
    // populate the exact-tail mask and run descriptor derived from them.
    body_mask = ggml_fill(ctx0, first_row_anchor(body_mask_template), -INFINITY);
    body_bias = body_bias_template ?
            ggml_fill(ctx0, first_row_anchor(body_bias_template), 0.0f) : nullptr;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
        build_attn_inp_tail(ctx0, ubatch, mctx_cur, inp->self_kq_mask,
                inp->self_tail_idxs, inp->self_tail_read_idxs, inp->self_tail_body_read_idxs,
                inp->self_tail_bias_read_idxs, inp->self_kq_mask_tail,
                inp->self_tail_query_order, inp->self_tail_run_desc, true);
    }

    inp->self_k_rot = mctx_cur->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->build_input_v_rot(ctx0);
    if (const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx_cur)) {
        inp->self_kvarn_rot_128 = kvarn->build_input_kvarn_rot(ctx0, 128);
        inp->self_kvarn_rot_256 = kvarn->build_input_kvarn_rot(ctx0, 256);
        inp->self_kvarn_rot_512 = kvarn->build_input_kvarn_rot(ctx0, 512);
        if (kvarn->uses_compact_read_indices()) {
            inp->self_kvarn_mat_idxs = kvarn->build_input_kvarn_mat_idxs(ctx0);
            kvarn->set_mat_idxs(inp->self_kvarn_mat_idxs);
        }
    }

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(v_mla == nullptr);

    const auto * mctx_cur = inp->mctx;
    const auto * kvarn_ctx = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx_cur);
    const bool use_kvarn = kvarn_ctx != nullptr;
    // Backend preferences choose an implementation inside the final operation;
    // they must not veto direct KVarN attention and force full materialization.
    // validate_native_tail_operation() proves the actual attached-tail shape.
    const bool kvarn_native_attention = use_kvarn &&
        kvarn_ctx->uses_native_attention(il);
    const auto kvarn_plan = use_kvarn ? llama_kvarn_plan_attention(
        kvarn_native_attention,
        kvarn_ctx->native_attention_uses_original_v(il),
        kvarn_ctx->native_rotated_max_query_tokens(il),
        (uint32_t) q_cur->ne[2]) : llama_kvarn_attention_plan {
            false, GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO };
    const auto kvarn_domain = kvarn_plan.domain;
    const bool use_kvarn_rotated_domain = use_kvarn &&
        kvarn_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED;
    const bool use_kvarn_mixed_domain = use_kvarn &&
        kvarn_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V;
    const bool use_kvarn_q_rot = use_kvarn_rotated_domain || use_kvarn_mixed_domain;

    if (use_kvarn) {
        GGML_ASSERT(llama_kvarn_head_dim_supported((int) q_cur->ne[0]));
        GGML_ASSERT(inp->self_k_rot == nullptr);
        GGML_ASSERT(inp->self_v_rot == nullptr);
        GGML_ASSERT(!use_kvarn_q_rot || llm_kvarn_rot_for_dim(
                inp->self_kvarn_rot_128, inp->self_kvarn_rot_256,
                inp->self_kvarn_rot_512, q_cur->ne[0]) != nullptr);
    }

    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    ggml_tensor * k_tail_written = nullptr;
    ggml_tensor * v_tail_written = nullptr;
    const bool compact_tail = mctx_cur->has_compact_tail();

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        if (compact_tail) {
            if (ggml_tensor * written = mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il)) {
                ggml_build_forward_expand(gf, written);
            }
        } else {
            k_tail_written = mctx_cur->cpy_k_with_tail(ctx0, k_cur, k_idxs, inp->self_tail_idxs, il);
            if (k_tail_written) {
                ggml_build_forward_expand(gf, k_tail_written);
            } else {
                ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
                if ((k_tail_written = mctx_cur->cpy_k_tail(ctx0, k_cur, inp->self_tail_idxs, il))) {
                    ggml_build_forward_expand(gf, k_tail_written);
                }
            }
        }
        if (compact_tail) {
            if (ggml_tensor * written = mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il)) {
                ggml_build_forward_expand(gf, written);
            }
        } else {
            v_tail_written = mctx_cur->cpy_v_with_tail(ctx0, v_cur, v_idxs, inp->self_tail_idxs, il);
            if (v_tail_written) {
                ggml_build_forward_expand(gf, v_tail_written);
            } else {
                ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
                if ((v_tail_written = mctx_cur->cpy_v_tail(ctx0, v_cur, inp->self_tail_idxs, il))) {
                    ggml_build_forward_expand(gf, v_tail_written);
                }
            }
        }
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = use_kvarn ?
        kvarn_ctx->get_k_for_attention(ctx0, il, kvarn_plan.native_attention) :
        mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = use_kvarn ?
        kvarn_ctx->get_v_for_attention(ctx0, il, kvarn_plan.native_attention) :
        mctx_cur->get_v(ctx0, il);
    ggml_tensor * kvarn_rot = use_kvarn ? llm_kvarn_rot_for_dim(
            inp->self_kvarn_rot_128, inp->self_kvarn_rot_256,
            inp->self_kvarn_rot_512, q->ne[0]) : nullptr;

    if (use_kvarn_q_rot) {
        GGML_ASSERT(q->type == GGML_TYPE_F32);
        GGML_ASSERT(kvarn_rot != nullptr);
        q = ggml_kvarn_wht_aux(ctx0, q, kvarn_rot->ne[0]);
    }

    // The SET_ROWS results alias the complete persistent shadow tensors and
    // carry the write dependency needed by same-graph attention reads.
    ggml_tensor * k_tail = k_tail_written ? k_tail_written : mctx_cur->get_k_tail(ctx0, il);
    ggml_tensor * v_tail = v_tail_written ? v_tail_written : mctx_cur->get_v_tail(ctx0, il);
    const bool gather_k_tail = k_tail != nullptr;
    const bool gather_v_tail = v_tail != nullptr;
    ggml_tensor * tail_read_idxs = inp->get_tail_read_idxs();
    llama_kv_tail_route tail_route = mctx_cur->get_tail_route(il);
    // Vulkan and other portable KVarN backends advertise a bounded rotated
    // query width. Outside that matrix the body is materialized, and the exact
    // tail must use the same explicit generic oracle instead of attaching
    // segmented sources to a standard FlashAttention operation that the
    // backend did not advertise.
    if (use_kvarn && tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE &&
            !kvarn_plan.native_attention) {
        tail_route = LLAMA_KV_TAIL_ROUTE_GENERIC;
    }
    if (tail_route != LLAMA_KV_TAIL_ROUTE_NONE) {
        GGML_ASSERT(mctx_cur->get_tail_explicit_bias(il) == (kq_b != nullptr) &&
                "KV tail route bias contract does not match the model graph");
    }
    ggml_tensor * k_tail_current = nullptr;
    ggml_tensor * v_tail_current = nullptr;
    if (compact_tail) {
        k_tail_current = prepare_compact_tail_current(ctx0, k_tail, k_cur);
        v_tail_current = prepare_compact_tail_current(ctx0, v_tail, v_cur);
        if (tail_route != LLAMA_KV_TAIL_ROUTE_NATIVE) {
            k_tail = ggml_concat(ctx0, k_tail, k_tail_current, 2);
            v_tail = ggml_concat(ctx0, v_tail, v_tail_current, 2);
            k_tail_current = nullptr;
            v_tail_current = nullptr;
        }
    }
    const bool use_indexed_tail = gather_k_tail && gather_v_tail && tail_read_idxs &&
            tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE;
    if (!use_indexed_tail && !use_kvarn && mctx_cur->get_tail_tokens() > 0) {
        if (!k_tail) {
            k_tail = mctx_cur->get_k_tail_fallback(ctx0, il, inp->self_tail_body_read_idxs);
        }
        if (!v_tail) {
            v_tail = mctx_cur->get_v_tail_fallback(ctx0, il, inp->self_tail_body_read_idxs);
        }
    }
    if (k_tail && !use_indexed_tail) {
        GGML_ASSERT(v_tail && tail_read_idxs);
        auto gather_tail = [&](ggml_tensor * value) {
            const int64_t n_embd_head = value->ne[0];
            const int64_t n_heads = value->ne[1];
            value = ggml_reshape_2d(ctx0, value, n_embd_head*n_heads, value->ne[2]);
            ggml_tensor * read_idxs = ggml_reshape_1d(ctx0, tail_read_idxs,
                    tail_read_idxs->ne[0]*ubatch.n_tokens);
            value = ggml_get_rows_as(ctx0, value, read_idxs, value->type);
            return ggml_reshape_4d(ctx0, value, n_embd_head, n_heads,
                    tail_read_idxs->ne[0], ubatch.n_tokens);
        };
        if (gather_k_tail) {
            k_tail = gather_tail(k_tail);
        }
        if (gather_v_tail) {
            v_tail = gather_tail(v_tail);
        }
    }
    if (use_kvarn && k_tail) {
        GGML_ASSERT(v_tail && kvarn_rot);
        k_tail = ggml_kvarn_wht_aux(ctx0, k_tail, kvarn_rot->ne[0]);
        if (use_kvarn_rotated_domain) {
            v_tail = ggml_kvarn_wht_aux(ctx0, v_tail, kvarn_rot->ne[0]);
        }
        if (k_tail_current) {
            k_tail_current = ggml_kvarn_wht_aux(ctx0, k_tail_current, kvarn_rot->ne[0]);
            if (use_kvarn_rotated_domain) {
                v_tail_current = ggml_kvarn_wht_aux(ctx0, v_tail_current, kvarn_rot->ne[0]);
            }
        }
    }
    ggml_tensor * kq_b_tail = build_attn_bias_tail(
            kq_b, inp->get_tail_bias_read_idxs(), inp->get_kq_mask_tail());
    ggml_tensor * body_mask = kq_mask;
    ggml_tensor * body_bias = kq_b;
    if (!mctx_cur->has_kv_body(il)) {
        GGML_ASSERT(compact_tail && !use_kvarn);
        const auto * route = mctx_cur->get_tail_layer_route(il);
        GGML_ASSERT(route);
        build_empty_kv_body(ctx0, k_cur, v_cur, route->body_type_k, route->body_type_v, kq_mask, kq_b,
                k, v, body_mask, body_bias);
    }
    ggml_tensor * final_tail_op = nullptr;
    ggml_tensor * cur = build_attn_mha(q, k, v, body_bias, body_mask, sinks, v_mla, kq_scale, il,
            k_tail, v_tail, inp->get_kq_mask_tail(), kq_b_tail,
            use_indexed_tail ? tail_read_idxs : nullptr,
            (use_indexed_tail || use_kvarn) ? inp->get_tail_query_order() : nullptr,
            (use_indexed_tail || use_kvarn) ? inp->get_tail_run_desc() : nullptr,
            tail_route,
            kvarn_domain,
            k_tail_current, v_tail_current,
            mctx_cur->get_tail_slots(), !mctx_cur->has_kv_body(il), &final_tail_op);
    if (use_kvarn) {
        llm_flash_attn_ext_set_kvarn_domain(cur, kvarn_domain);
    }
    if (tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE) {
        validate_native_tail_operation(mctx_cur, il, final_tail_op);
    }
    if (compact_tail) {
        if (ggml_tensor * written = mctx_cur->cpy_k_tail(
                    ctx0, k_cur, inp->self_tail_idxs, il, cur)) {
            ggml_build_forward_expand(gf, written);
        }
        if (ggml_tensor * written = mctx_cur->cpy_v_tail(
                    ctx0, v_cur, inp->self_tail_idxs, il, cur)) {
            ggml_build_forward_expand(gf, written);
        }
    }
    cb(cur, "kqv_out", il);

    if (use_kvarn_rotated_domain) {
        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(kvarn_rot != nullptr);
        cur = ggml_kvarn_wht_aux(ctx0, cur, kvarn_rot->ne[0]);
    } else if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_dsa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        ggml_tensor * top_k,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx->get_mla();

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs_mla();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask_mla();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_top_k, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
    auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;

    const auto * mctx_iswa = inp->mctx;
    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();
    const auto * kvarn_ctx = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx_cur);
    const bool use_kvarn = kvarn_ctx != nullptr;
    const bool kvarn_native_attention = use_kvarn &&
        kvarn_ctx->uses_native_attention(il);
    const auto kvarn_plan = use_kvarn ? llama_kvarn_plan_attention(
        kvarn_native_attention,
        kvarn_ctx->native_attention_uses_original_v(il),
        kvarn_ctx->native_rotated_max_query_tokens(il),
        (uint32_t) q_cur->ne[2]) : llama_kvarn_attention_plan {
            false, GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO };
    const auto kvarn_domain = kvarn_plan.domain;
    const bool use_kvarn_rotated_domain = use_kvarn &&
        kvarn_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED;
    const bool use_kvarn_mixed_domain = use_kvarn &&
        kvarn_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V;
    const bool use_kvarn_q_rot = use_kvarn_rotated_domain || use_kvarn_mixed_domain;

    if (use_kvarn) {
        GGML_ASSERT(llama_kvarn_head_dim_supported((int) q_cur->ne[0]));
        GGML_ASSERT(k_rot == nullptr);
        GGML_ASSERT(v_rot == nullptr);
        GGML_ASSERT(!use_kvarn_q_rot || llm_kvarn_rot_for_dim(
                inp->self_kvarn_rot_128, inp->self_kvarn_rot_256,
                inp->self_kvarn_rot_512, q_cur->ne[0]) != nullptr);
    }

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = llama_mul_mat_hadamard(ctx0, v_cur, v_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    ggml_tensor * k_tail_written = nullptr;
    ggml_tensor * v_tail_written = nullptr;
    const bool compact_tail = mctx_cur->has_compact_tail();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        if (compact_tail) {
            if (ggml_tensor * written = mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il)) {
                ggml_build_forward_expand(gf, written);
            }
        } else {
            k_tail_written = mctx_cur->cpy_k_with_tail(
                    ctx0, k_cur, k_idxs, inp->get_tail_idxs(is_swa), il);
            if (k_tail_written) {
                ggml_build_forward_expand(gf, k_tail_written);
            } else {
                ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
                if ((k_tail_written = mctx_cur->cpy_k_tail(ctx0, k_cur, inp->get_tail_idxs(is_swa), il))) {
                    ggml_build_forward_expand(gf, k_tail_written);
                }
            }
        }
    }

    if (v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        if (compact_tail) {
            if (ggml_tensor * written = mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il)) {
                ggml_build_forward_expand(gf, written);
            }
        } else {
            v_tail_written = mctx_cur->cpy_v_with_tail(
                    ctx0, v_cur, v_idxs, inp->get_tail_idxs(is_swa), il);
            if (v_tail_written) {
                ggml_build_forward_expand(gf, v_tail_written);
            } else {
                ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
                if ((v_tail_written = mctx_cur->cpy_v_tail(ctx0, v_cur, inp->get_tail_idxs(is_swa), il))) {
                    ggml_build_forward_expand(gf, v_tail_written);
                }
            }
        }
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = use_kvarn ?
        kvarn_ctx->get_k_for_attention(ctx0, il, kvarn_plan.native_attention) :
        mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = use_kvarn ?
        kvarn_ctx->get_v_for_attention(ctx0, il, kvarn_plan.native_attention) :
        mctx_cur->get_v(ctx0, il);
    ggml_tensor * kvarn_rot = use_kvarn ? llm_kvarn_rot_for_dim(
            inp->self_kvarn_rot_128, inp->self_kvarn_rot_256,
            inp->self_kvarn_rot_512, q->ne[0]) : nullptr;

    if (use_kvarn_q_rot) {
        GGML_ASSERT(q->type == GGML_TYPE_F32);
        GGML_ASSERT(kvarn_rot != nullptr);
        q = ggml_kvarn_wht_aux(ctx0, q, kvarn_rot->ne[0]);
    }

    ggml_tensor * k_tail = k_tail_written ? k_tail_written : mctx_cur->get_k_tail(ctx0, il);
    ggml_tensor * v_tail = v_tail_written ? v_tail_written : mctx_cur->get_v_tail(ctx0, il);
    const bool gather_k_tail = k_tail != nullptr;
    const bool gather_v_tail = v_tail != nullptr;
    ggml_tensor * tail_read_idxs = inp->get_tail_read_idxs(is_swa);
    llama_kv_tail_route tail_route = mctx_cur->get_tail_route(il);
    // Keep the iSWA route decision identical to the non-iSWA path above.
    if (use_kvarn && tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE &&
            !kvarn_plan.native_attention) {
        tail_route = LLAMA_KV_TAIL_ROUTE_GENERIC;
    }
    if (tail_route != LLAMA_KV_TAIL_ROUTE_NONE) {
        GGML_ASSERT(mctx_cur->get_tail_explicit_bias(il) == (kq_b != nullptr) &&
                "KV tail route bias contract does not match the model graph");
    }
    ggml_tensor * k_tail_current = nullptr;
    ggml_tensor * v_tail_current = nullptr;
    if (compact_tail) {
        GGML_ASSERT(k_cur && v_cur);
        k_tail_current = prepare_compact_tail_current(ctx0, k_tail, k_cur);
        v_tail_current = prepare_compact_tail_current(ctx0, v_tail, v_cur);
        if (tail_route != LLAMA_KV_TAIL_ROUTE_NATIVE) {
            k_tail = ggml_concat(ctx0, k_tail, k_tail_current, 2);
            v_tail = ggml_concat(ctx0, v_tail, v_tail_current, 2);
            k_tail_current = nullptr;
            v_tail_current = nullptr;
        }
    }
    const bool use_indexed_tail = gather_k_tail && gather_v_tail && tail_read_idxs &&
            tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE;
    if (!use_indexed_tail && !use_kvarn && mctx_cur->get_tail_tokens() > 0) {
        if (!k_tail) {
            k_tail = mctx_cur->get_k_tail_fallback(ctx0, il, inp->get_tail_body_read_idxs(is_swa));
        }
        if (!v_tail) {
            v_tail = mctx_cur->get_v_tail_fallback(ctx0, il, inp->get_tail_body_read_idxs(is_swa));
        }
    }
    if (k_tail && !use_indexed_tail) {
        GGML_ASSERT(v_tail && tail_read_idxs);
        const auto gather_tail = [&](ggml_tensor * value) {
            const int64_t n_embd_head = value->ne[0];
            const int64_t n_heads = value->ne[1];
            value = ggml_reshape_2d(ctx0, value, n_embd_head*n_heads, value->ne[2]);
            ggml_tensor * read_idxs = ggml_reshape_1d(ctx0, tail_read_idxs,
                    tail_read_idxs->ne[0]*ubatch.n_tokens);
            value = ggml_get_rows_as(ctx0, value, read_idxs, value->type);
            return ggml_reshape_4d(ctx0, value, n_embd_head, n_heads,
                    tail_read_idxs->ne[0], ubatch.n_tokens);
        };
        if (gather_k_tail) {
            k_tail = gather_tail(k_tail);
        }
        if (gather_v_tail) {
            v_tail = gather_tail(v_tail);
        }
    }
    if (use_kvarn && k_tail) {
        GGML_ASSERT(v_tail && kvarn_rot);
        k_tail = ggml_kvarn_wht_aux(ctx0, k_tail, kvarn_rot->ne[0]);
        if (use_kvarn_rotated_domain) {
            v_tail = ggml_kvarn_wht_aux(ctx0, v_tail, kvarn_rot->ne[0]);
        }
        if (k_tail_current) {
            k_tail_current = ggml_kvarn_wht_aux(ctx0, k_tail_current, kvarn_rot->ne[0]);
            if (use_kvarn_rotated_domain) {
                v_tail_current = ggml_kvarn_wht_aux(ctx0, v_tail_current, kvarn_rot->ne[0]);
            }
        }
    }
    ggml_tensor * kq_b_tail = build_attn_bias_tail(
            kq_b, inp->get_tail_bias_read_idxs(is_swa), inp->get_kq_mask_tail(is_swa));
    ggml_tensor * body_mask = kq_mask;
    ggml_tensor * body_bias = kq_b;
    if (!mctx_cur->has_kv_body(il)) {
        GGML_ASSERT(compact_tail && !use_kvarn && is_swa);
        const auto * route = mctx_cur->get_tail_layer_route(il);
        GGML_ASSERT(route);
        build_empty_kv_body(ctx0, k_cur, v_cur, route->body_type_k, route->body_type_v, kq_mask, kq_b,
                k, v, body_mask, body_bias);
    }
    ggml_tensor * final_tail_op = nullptr;
    ggml_tensor * cur = build_attn_mha(q, k, v, body_bias, body_mask, sinks, v_mla, kq_scale, il,
            k_tail, v_tail, inp->get_kq_mask_tail(is_swa), kq_b_tail,
            use_indexed_tail ? tail_read_idxs : nullptr,
            (use_indexed_tail || use_kvarn) ? inp->get_tail_query_order(is_swa) : nullptr,
            (use_indexed_tail || use_kvarn) ? inp->get_tail_run_desc(is_swa) : nullptr,
            tail_route,
            kvarn_domain,
            k_tail_current, v_tail_current,
            mctx_cur->get_tail_slots(), !mctx_cur->has_kv_body(il), &final_tail_op);
    if (use_kvarn) {
        llm_flash_attn_ext_set_kvarn_domain(cur, kvarn_domain);
    }
    if (tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE) {
        validate_native_tail_operation(mctx_cur, il, final_tail_op);
    }
    if (compact_tail) {
        if (ggml_tensor * written = mctx_cur->cpy_k_tail(
                    ctx0, k_cur, inp->get_tail_idxs(is_swa), il, cur)) {
            ggml_build_forward_expand(gf, written);
        }
        if (ggml_tensor * written = mctx_cur->cpy_v_tail(
                    ctx0, v_cur, inp->get_tail_idxs(is_swa), il, cur)) {
            ggml_build_forward_expand(gf, written);
        }
    }
    cb(cur, "kqv_out", il);

    if (use_kvarn_rotated_domain) {
        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(kvarn_rot != nullptr);
        cur = ggml_kvarn_wht_aux(ctx0, cur, kvarn_rot->ne[0]);
    } else if (v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    const auto * mctx_iswa = inp->mctx;
    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // MLA-style attention: the cached K is used as V
    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (k_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, k_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_k_dsa * llm_graph_context::build_attn_inp_k_dsa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_mla = build_attn_inp_kq_mask(ctx0, mctx_cur->get_mla(), ubatch, cparams);
        inp->self_kq_mask_mla_cnv = inp->self_kq_mask_mla;
    }

    {
        inp->self_k_idxs_lid = mctx_cur->get_lid()->build_input_k_idxs(ctx0, ubatch);

        // ensure that mask type matches fused lightning indexer use (requires f16 mask)
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = cparams.fused_lid;

        inp->self_kq_mask_lid = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);
        inp->self_kq_mask_lid_cnv = inp->self_kq_mask_lid;

        inp->self_k_rot_lid = mctx_cur->get_lid()->build_input_k_rot(ctx0);
    }

    return (llm_graph_input_attn_k_dsa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_kv_msa * llm_graph_context::build_attn_inp_kv_msa(bool msa_enabled) const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_msa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_msa>(hparams, cparams, mctx_cur);

    const auto * mctx_base = mctx_cur->get_base();
    const auto * mctx_idx  = mctx_cur->get_idx();

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_base->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_base->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_base, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_base->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_base->build_input_v_rot(ctx0);

    if (msa_enabled) {
        inp->self_k_idxs_idx = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    }

    return (llm_graph_input_attn_kv_msa *) res->add_input(std::move(inp));
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
        build_attn_inp_tail(ctx0, ubatch, mctx_cur->get_base(), inp->self_kq_mask,
                inp->self_tail_idxs, inp->self_tail_read_idxs, inp->self_tail_body_read_idxs,
                inp->self_tail_bias_read_idxs, inp->self_kq_mask_tail,
                inp->self_tail_query_order, inp->self_tail_run_desc, true);
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
        build_attn_inp_tail(ctx0, ubatch, mctx_cur->get_swa(), inp->self_kq_mask_swa,
                inp->self_tail_idxs_swa, inp->self_tail_read_idxs_swa,
                inp->self_tail_body_read_idxs_swa, inp->self_tail_bias_read_idxs_swa,
                inp->self_kq_mask_tail_swa, inp->self_tail_query_order_swa, inp->self_tail_run_desc_swa, true);
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->get_base()->build_input_v_rot(ctx0);
    if (const auto * kvarn_base = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx_cur->get_base())) {
        inp->self_kvarn_rot_128 = kvarn_base->build_input_kvarn_rot(ctx0, 128);
        inp->self_kvarn_rot_256 = kvarn_base->build_input_kvarn_rot(ctx0, 256);
        inp->self_kvarn_rot_512 = kvarn_base->build_input_kvarn_rot(ctx0, 512);
        if (kvarn_base->uses_compact_read_indices()) {
            inp->self_kvarn_mat_idxs = kvarn_base->build_input_kvarn_mat_idxs(ctx0);
            kvarn_base->set_mat_idxs(inp->self_kvarn_mat_idxs);
        }
    }

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);
    inp->self_v_rot_swa = mctx_cur->get_swa()->build_input_v_rot(ctx0);
    if (const auto * kvarn_swa = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx_cur->get_swa())) {
        if (inp->self_kvarn_rot_128 == nullptr) {
            inp->self_kvarn_rot_128 = kvarn_swa->build_input_kvarn_rot(ctx0, 128);
            inp->self_kvarn_rot_256 = kvarn_swa->build_input_kvarn_rot(ctx0, 256);
            inp->self_kvarn_rot_512 = kvarn_swa->build_input_kvarn_rot(ctx0, 512);
        }
        inp->self_kvarn_mat_idxs_swa = kvarn_swa->build_input_kvarn_mat_idxs(ctx0);
        // Native K/V views are built before set_input() populates the tensor.
        kvarn_swa->set_mat_idxs(inp->self_kvarn_mat_idxs_swa);
    }

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_iswa * llm_graph_context::build_attn_inp_k_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);

    return (llm_graph_input_attn_k_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_dsv4 * llm_graph_context::build_inp_dsv4() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);
    const auto * raw_ctx  = mctx_cur->get_raw();

    auto inp_raw = std::make_unique<llm_graph_input_dsv4_raw>(cparams, raw_ctx);

    const int64_t n_stream = mctx_cur->get_csa_plan(ubatch).n_stream;

    GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "DSV4 expects SWA raw cache");

    inp_raw->self_k_idxs = raw_ctx->build_input_k_idxs(ctx0, ubatch);
    inp_raw->self_tail_idxs = raw_ctx->build_input_tail_idxs(ctx0, ubatch);
    inp_raw->self_kq_mask = dsv4_build_raw_kq_mask(ctx0, raw_ctx, ubatch, cparams, n_stream);
    inp_raw->self_kq_mask_cnv = inp_raw->self_kq_mask;
    if (raw_ctx->get_tail_tokens() > 0) {
        const auto [tail_q_max, tail_n_active] = tail_query_plan_shape(ubatch);
        GGML_UNUSED(tail_n_active);
        const uint32_t attention_stride = raw_ctx->get_tail_attention_stride(uint32_t(tail_q_max));
        inp_raw->self_tail_read_idxs = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_I32, attention_stride, ubatch.n_tokens);
        ggml_set_input(inp_raw->self_tail_read_idxs);
        inp_raw->self_tail_body_read_idxs = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_I32, attention_stride, ubatch.n_tokens);
        ggml_set_input(inp_raw->self_tail_body_read_idxs);
        inp_raw->self_tail_bias_read_idxs = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_I32, attention_stride, ubatch.n_tokens);
        ggml_set_input(inp_raw->self_tail_bias_read_idxs);
        inp_raw->self_kq_mask_tail = ggml_new_tensor_4d(ctx0, inp_raw->self_kq_mask->type,
                attention_stride, inp_raw->self_kq_mask->ne[1], 1, inp_raw->self_kq_mask->ne[3]);
        ggml_set_input(inp_raw->self_kq_mask_tail);
    }

    inp_raw->self_k_rot = raw_ctx->build_input_k_rot(ctx0);
    auto inp = std::make_unique<llm_graph_input_dsv4>(cparams, std::move(inp_raw), mctx_cur);

    dsv4_build_comp_inputs(ctx0, inp->inp_csa, mctx_cur->get_csa_plan(ubatch), "csa", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_hca, mctx_cur->get_hca_plan(ubatch), "hca", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_lid, mctx_cur->get_lid_plan(ubatch), "lid", cparams, n_stream);
    inp->inp_csa.k_rot = mctx_cur->get_csa()->build_input_k_rot(ctx0);
    inp->inp_hca.k_rot = mctx_cur->get_hca()->build_input_k_rot(ctx0);
    inp->inp_lid.k_rot = mctx_cur->get_lid()->build_input_k_rot(ctx0);

    return (llm_graph_input_dsv4 *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy_main,
        ggml_tensor * state_copy_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    GGML_UNUSED(rs_size);
    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // {state_size, rs_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);

    inp->s_copy_main  = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_extra = ggml_view_1d(ctx0, inp->s_copy, n_rs - n_seqs, n_seqs * inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    return build_rs(s, inp->s_copy_main, inp->s_copy_extra, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::build_inp_mem_hybrid_iswa() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    // build iswa attention input
    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
        inp_attn->self_kq_mask_cnv = inp_attn->self_kq_mask;
        build_attn_inp_tail(ctx0, ubatch, attn_ctx->get_base(), inp_attn->self_kq_mask,
                inp_attn->self_tail_idxs, inp_attn->self_tail_read_idxs,
                inp_attn->self_tail_body_read_idxs, inp_attn->self_tail_bias_read_idxs,
                inp_attn->self_kq_mask_tail, inp_attn->self_tail_query_order, inp_attn->self_tail_run_desc, true);
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
        inp_attn->self_kq_mask_swa_cnv = inp_attn->self_kq_mask_swa;
        build_attn_inp_tail(ctx0, ubatch, attn_ctx->get_swa(), inp_attn->self_kq_mask_swa,
                inp_attn->self_tail_idxs_swa, inp_attn->self_tail_read_idxs_swa,
                inp_attn->self_tail_body_read_idxs_swa, inp_attn->self_tail_bias_read_idxs_swa,
                inp_attn->self_kq_mask_tail_swa, inp_attn->self_tail_query_order_swa,
                inp_attn->self_tail_run_desc_swa, true);
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_iswa *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_2_b,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || !(dense_2 || dense_2_b || dense_3)) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    if (dense_2) {
        cur = ggml_mul_mat(ctx0, dense_2, cur);
    }
    if (dense_2_b) {
        cur = ggml_add(ctx0, cur, dense_2_b);
    }
    if (dense_3) {
        cur = ggml_mul_mat(ctx0, dense_3, cur);
    }
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b,
        ggml_tensor * cls_norm) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                if (arch == LLM_ARCH_MODERN_BERT) {
                    // modern bert gte reranker builds mean first then applies prediction head and classifier
                    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
                    ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    if (arch == LLM_ARCH_MODERN_BERT) {
                        cur = ggml_gelu(ctx0, cur);
                    } else {
                        cur = ggml_tanh(ctx0, cur);
                    }
                    if (cls_norm) {
                        // head norm
                        cur = build_norm(cur, cls_norm, NULL, LLM_NORM, -1);
                    }
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    res->add_input(std::move(inp_sampling));

    std::map<llama_seq_id, std::vector<uint32_t>> sampling_rows;
    uint32_t n_rows = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.output[i]) {
            sampling_rows[ubatch.seq_id[i][0]].push_back(n_rows++);
        }
    }

    res->t_sampled.resize(n_rows, nullptr);
    res->t_sampled_probs.resize(n_rows, nullptr);
    res->t_sampled_logits.resize(n_rows, nullptr);
    res->t_candidates.resize(n_rows, nullptr);

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    // add a dummy row to keep the single-output graph static regardless of active samplers
    // multi-output graphs can still vary with the number of output rows
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);

    for (const auto & entry : samplers) {
        if (entry.second->iface->backend_reset) {
            entry.second->iface->backend_reset(entry.second);
        }
    }

    static const std::vector<uint32_t> dummy_row = { 0 };

    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);

        // inactive samplers always work on the first row
        const bool active = it != sampling_rows.end();
        const auto & rows = active ? it->second : dummy_row;
        const int i_out   = active ? 1          : 0;

        for (uint32_t i = 0; i < rows.size(); ++i) {
            ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], rows[i] * logits_t->nb[1]);
            ggml_format_name(logits_seq, "logits_seq_%d_%u", seq_id, i);

            struct llama_sampler_data data = {
                /*.logits       =*/ logits_seq,
                /*.probs        =*/ nullptr,
                /*.sampled      =*/ nullptr,
                /*.candidates   =*/ nullptr,
            };

            assert(sampler->iface->backend_apply);
            sampler->iface->backend_apply(sampler, ctx0, gf, &data);

            if (data.sampled != nullptr) {
                if (active) {
                    res->t_sampled[rows[i]] = data.sampled;
                }
                outs[1] = data.sampled;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.probs != nullptr) {
                if (active) {
                    res->t_sampled_probs[rows[i]] = data.probs;
                }
                outs[1] = data.probs;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.logits != nullptr) {
                if (active) {
                    res->t_sampled_logits[rows[i]] = data.logits;
                }
                outs[1] = data.logits;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.candidates != nullptr) {
                if (active) {
                    res->t_candidates[rows[i]] = data.candidates;
                }
                outs[1] = data.candidates;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }
        }
    }

    // TODO: Call backend_accept after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);
        if (it == sampling_rows.end()) {
            continue;
        }

        for (uint32_t row : it->second) {
            ggml_tensor * selected_token = res->t_sampled[row];
            if (selected_token != nullptr && sampler->iface->backend_accept) {
                sampler->iface->backend_accept(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
