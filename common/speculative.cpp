#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "suffix-tree.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <cinttypes>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,
    COMMON_SPECULATIVE_TYPE_SUFFIX,
    COMMON_SPECULATIVE_TYPE_COPYSPEC,
    COMMON_SPECULATIVE_TYPE_RECYCLE,
    COMMON_SPECULATIVE_TYPE_DFLASH
};

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"suffix",        COMMON_SPECULATIVE_TYPE_SUFFIX},
    {"copyspec",      COMMON_SPECULATIVE_TYPE_COPYSPEC},
    {"recycle",       COMMON_SPECULATIVE_TYPE_RECYCLE},
    {"dflash",        COMMON_SPECULATIVE_TYPE_DFLASH}
};

static bool common_dflash_profile_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_PROFILE");
        return env && std::atoi(env) != 0;
    }();
    return enabled;
}

static bool common_dflash_kv_cache_disabled() {
    static const bool disabled = [] {
        const char * env = std::getenv("GGML_DFLASH_DISABLE_KV_CACHE");
        if (env && std::atoi(env) != 0) {
            return true;
        }

        const char * mode = std::getenv("GGML_DFLASH_KV_CACHE_MODE");
        return mode && (std::strcmp(mode, "off") == 0 ||
                        std::strcmp(mode, "none") == 0 ||
                        std::strcmp(mode, "disabled") == 0);
    }();
    return disabled;
}

static bool common_dflash_gpu_ring_env_enabled() {
    const char * env = std::getenv("GGML_DFLASH_GPU_RING");
    return env == nullptr || std::atoi(env) != 0;
}

static bool common_dflash_force_cpu_cross_ring() {
    static const bool v = [] {
        const char * e = getenv("GGML_DFLASH_FORCE_CPU_CROSS");
        return e && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return v;
}

static bool common_dflash_log_contract_verbose() {
    static const bool v = [] {
        const char * e = getenv("GGML_DFLASH_VERBOSE_CONTRACT");
        return e && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return v;
}

static bool common_dflash_gpu_ring_allowed(llama_context * ctx_tgt, llama_context * ctx_dft) {
    if (!common_dflash_gpu_ring_env_enabled()) {
        LOG_INF("dflash: GPU cross ring disabled by GGML_DFLASH_GPU_RING=0; using CPU hidden capture\n");
        return false;
    }

    const int32_t n_tgt_devices = ctx_tgt ? llama_model_n_devices(llama_get_model(ctx_tgt)) : 1;
    const int32_t n_dft_devices = ctx_dft ? llama_model_n_devices(llama_get_model(ctx_dft)) : 1;
    if (n_tgt_devices > 1 || n_dft_devices > 1) {
        LOG_INF("dflash: multi-GPU placement detected (target=%d devices, drafter=%d devices); disabling GPU cross ring and graph hidden capture\n",
                n_tgt_devices, n_dft_devices);
        return false;
    }

    return true;
}

common_dflash_ring_write common_dflash_ring_write_plan(int ring_size, int ring_pos, int n_tokens) {
    if (ring_size <= 0 || n_tokens <= 0) {
        return { 0, 0, 0 };
    }

    int normalized_pos = ring_pos % ring_size;
    if (normalized_pos < 0) {
        normalized_pos += ring_size;
    }

    if (n_tokens <= ring_size) {
        return { normalized_pos, n_tokens, 0 };
    }

    const int skip = n_tokens - ring_size;
    normalized_pos = (normalized_pos + skip) % ring_size;
    return { normalized_pos, ring_size, skip };
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.
    size_t n_ext_calls  = 0; // number of times extend() produced tokens
    size_t n_ext_tokens = 0; // total tokens added via extend()

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_state(enum common_speculative_type type) : type(type) {}

    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result,
            std::vector<float> * draft_log_probs = nullptr) = 0;

    virtual void draft_tree(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            int tree_budget,
            common_speculative_tree & tree) {
        // default: flat draft, no tree
        llama_tokens result;
        draft(params, prompt_tgt, id_last, result);
        tree.n_nodes = (int)result.size();
        tree.tokens = std::move(result);
        tree.parents.resize(tree.n_nodes + 1);
        tree.parents[0] = -1;
        for (int i = 0; i < tree.n_nodes; ++i) {
            tree.parents[i + 1] = i; // linear chain
            tree.depths.push_back(i + 1);
        }
        // build child_maps + visibility for linear chain
        tree.child_maps.resize(tree.n_nodes + 1);
        for (int i = 0; i < tree.n_nodes; ++i) {
            tree.child_maps[i][tree.tokens[i]] = i + 1;
        }
        int n = tree.n_nodes + 1;
        tree.visibility.assign(n * n, false);
        for (int i = 0; i < n; ++i) {
            // each node sees itself and all ancestors
            int cur = i;
            while (cur >= 0) {
                tree.visibility[i * n + cur] = true;
                cur = tree.parents[cur];
            }
        }
        GGML_UNUSED(tree_budget);
    }

    // extend an existing draft with additional tokens (e.g. CopySpec appending after DFlash)
    // called after a primary impl has produced a non-empty result
    virtual void extend(
            const common_params_speculative & /*params*/,
            const llama_tokens & /*prompt_tgt*/,
            llama_token /*id_last*/,
            llama_tokens & /*result*/) {}

    virtual void accept(uint16_t n_accepted) = 0;

    // called after verification decode with logits still in ctx
    // batch_tokens: tokens that were in the batch [id_last, draft0, draft1, ...]
    // n_accepted: number of decoded batch rows to commit (root + accepted draft tokens)
    virtual void update_logits(llama_context * /*ctx*/, const llama_tokens & /*batch_tokens*/, int /*n_accepted*/) {}

    // tree variant: accept specific capture-buffer indices instead of a contiguous block.
    // capture_indices[0] = id_last (root), capture_indices[1..] = accepted tree node indices.
    virtual void update_logits_by_indices(llama_context * /*ctx*/, const std::vector<int> & /*capture_indices*/) {}

    // flush captured hidden states into the ring buffer during prefill.
    // called after each llama_decode sub-batch so checkpoint splits don't
    // lose hidden state context between sub-batches.
    // src_offset and n_tokens: which contiguous span of the capture buffer
    // to write.  When n_tokens=0, write the entire capture buffer (legacy).
    // Returns the number of tokens actually written to the ring.
    virtual int flush_prefill(int src_offset = 0, int n_tokens = 0) { return 0; }

    // Enable/disable target hidden capture during prefill.
    // No-op in base class. DFlash overrides to toggle llama_set_dflash_capture.
    virtual void set_prefill_capture_enabled(bool /*enabled*/) {}

    // save/restore ring buffer state for checkpoint persistence.
    // allows hidden states captured during prefill to survive checkpoint restore.
    virtual size_t ring_state_size() const { return 0; }
    virtual void ring_state_save(uint8_t * /*buf*/, size_t /*size*/) const {}
    virtual bool ring_state_load(const uint8_t * /*buf*/, size_t /*size*/) { return false; }

    // identify which server slot (seq_id on both ctx_tgt and shared ctx_dft) this
    // state owns. Default 0 (single-slot). Non-DFlash impls ignore this.
    virtual void set_seq_id(llama_seq_id /*seq_id*/) {}

    // prepare cross-attention data for batched drafting on a shared ctx_dft.
    // Returns cross_len (position offset for batch tokens), or -1 if not ready.
    virtual int prepare_batch_draft(llama_context * /*ctx_dft*/) { return -1; }

    virtual int32_t n_max(const common_params_speculative & params) const = 0;
    virtual int32_t n_min(const common_params_speculative & params) const = 0;
};

struct common_speculative_checkpoint {
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;

    int64_t   n_tokens = 0;

    std::vector<uint8_t> data;

    size_t size() const {
        return data.size();
    }
};

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    bool use_ckpt = false;
    common_speculative_checkpoint ckpt;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements,
            bool use_ckpt)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
        , use_ckpt(use_ckpt)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;

        {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_perf_context_print(ctx_dft);

        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & /*prompt*/) override {
    }

    size_t create_checkpoint(int n_tokens_prompt) {
        int slot_id = 0;
        const size_t checkpoint_size = llama_state_seq_get_size_ext(ctx_dft, slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

        ckpt.pos_min  = llama_memory_seq_pos_min(llama_get_memory(ctx_dft), slot_id);
        ckpt.pos_max  = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), slot_id);
        ckpt.n_tokens = n_tokens_prompt;
        ckpt.data.resize(checkpoint_size);

        const size_t n = llama_state_seq_get_data_ext(ctx_dft, ckpt.data.data(), checkpoint_size, slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (n != checkpoint_size) {
            GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", checkpoint_size, n);
        }

        LOG_DBG("%s: pos_min = %d, pos_max = %d, size = %.3f MiB\n", __func__,
                ckpt.pos_min, ckpt.pos_max, (float) ckpt.data.size() / 1024 / 1024);
        return n;
    }

    size_t restore_checkpoint() {
        int slot_id = 0;
        LOG_DBG("%s: pos_min = %d, pos_max = %d\n", __func__, ckpt.pos_min, ckpt.pos_max);
        const size_t n = llama_state_seq_set_data_ext(ctx_dft, ckpt.data.data(), ckpt.size(), slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (n != ckpt.size()) {
            GGML_ABORT("%s: failed to restore context checkpoint (pos_min=%d, pos_max=%d, size=%zu",
                        __func__, ckpt.pos_min, ckpt.pos_max, ckpt.size());
        }
        llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, ckpt.pos_max + 1, -1);

        return n;
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result,
            std::vector<float> * draft_log_probs = nullptr) override {
        GGML_UNUSED(draft_log_probs);
        const auto & sparams = params.draft;

        auto * spec = this;

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        auto * mem_dft = llama_get_memory(ctx_dft);

        int reuse_i = 0; // index of part to be reused in prompt_dft
        int reuse_n = 0; // length of part to be reused in prompt_dft

        const int n_ctx = llama_n_ctx(ctx_dft) - sparams.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);

            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        if (use_ckpt && i_start > 0) {
            LOG_WRN("%s: context shift is not supported with checkpoint-based contexts - skipping\n", __func__);
            return;
        }

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                   i       + cur < (int) prompt_dft.size() &&
                   prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }

            if (use_ckpt) {
                break;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, #prompt_dft = %zu, #prompt_cur = %zu\n",
                __func__, reuse_i, reuse_n, prompt_dft.size(), prompt_cur.size());
        if (use_ckpt && ckpt.n_tokens > reuse_n) {
            LOG_DBG("%s: checkpoint (n_tokens = %d) is outdated -> delete it\n", __func__, (int) ckpt.n_tokens);

            reuse_i = 0;
            reuse_n = 0;

            ckpt = {};
        }

        result.clear();
        result.reserve(sparams.n_max);

        if (reuse_n == 0 || (use_ckpt && reuse_i > 0)) {
            llama_memory_clear(mem_dft, false);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int64_t) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (sparams.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            if (reuse_i > 0) {
                GGML_ASSERT(!use_ckpt);

                bool is_removed = llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
                if (!is_removed) {
                    LOG_ERR("%s: llama_memory_seq_rm failed, reuse_i=%d\n", __func__, reuse_i);
                    return;
                }
                llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                if (use_ckpt) {
                    if (ckpt.n_tokens > 0) {
                        LOG_DBG("%s: restoring checkpoint, reuse_n=%d, prompt_dft.size=%zu\n", __func__, reuse_n, prompt_dft.size());
                        restore_checkpoint();
                        reuse_n = ckpt.n_tokens;
                        prompt_dft.resize(reuse_n);
                    }
                } else {
                    const bool is_removed = llama_memory_seq_rm(mem_dft, 0, reuse_n, -1);
                    if (!is_removed) {
                        LOG_ERR("%s: llama_memory_seq_rm failed, reuse_n=%d, prompt_dft.size=%zu\n", __func__, reuse_n, prompt_dft.size());
                        return;
                    }
                    prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
                }
            }
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
        }

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());
            LOG_DBG("%s: draft prompt batch: %d tokens\n", __func__, batch.n_tokens);

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0 && ret != 1) {
                LOG_WRN("%s: llama_decode returned %d, prompt_cur.size=%zu\n",
                        __func__, ret, prompt_cur.size());
            }

            if (use_ckpt) {
                create_checkpoint(prompt_dft.size());
            }
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        //LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0 && ret != 1) {
            LOG_WRN("%s: llama_decode returned %d, prompt_cur.size=%zu, prompt_dft.size=%zu\n",
                    __func__, ret, prompt_cur.size(), prompt_dft.size());
        }

        common_sampler_reset(smpl);

        // sample n_draft tokens from the draft model
        for (int i = 0; i < sparams.n_max; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            // add drafted token for each sequence
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, id, true);

            // only collect very high-confidence draft tokens
            if (cur_p->data[0].p < sparams.p_min) {
                break;
            }

            result.push_back(id);

            if (sparams.n_max <= (int) result.size()) {
                break;
            }

            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d, prompt_cur.size=%zu, prompt_dft.size=%zu\n",
                        __func__, i, ret, prompt_cur.size(), prompt_dft.size());
            }

            prompt_dft.push_back(id);
        }

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t) sparams.n_max) {
                result.resize(sparams.n_max);
            }
        }

        if (result.size() < (size_t) sparams.n_min) {
            result.clear();
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.draft.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.draft.n_min;
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

struct common_speculative_state_eagle3 : public common_speculative_state {
    common_speculative_state_eagle3(enum common_speculative_type type) : common_speculative_state(type) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens,
            std::vector<float> * draft_log_probs = nullptr) override {
        // TODO: implement
        GGML_UNUSED(params);
        GGML_UNUSED(prompt_tgt);
        GGML_UNUSED(id_last);
        GGML_UNUSED(draft_tokens);
        GGML_UNUSED(draft_log_probs);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.draft.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.draft.n_min;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_config config)
        : common_speculative_state(type), config(config) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result,
            std::vector<float> * draft_log_probs = nullptr) override {

        result = common_ngram_simple_draft(config, prompt_tgt, id_last);
        GGML_UNUSED(params);
        GGML_UNUSED(draft_log_probs);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return config.size_mgram;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return config.size_mgram;
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map config;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map config)
        : common_speculative_state(type), config(std::move(config)) {}

    void begin(const llama_tokens & prompt) override {
        common_ngram_map_begin(config, prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result,
            std::vector<float> * draft_log_probs = nullptr) override {
        common_ngram_map_draft(config, prompt_tgt, id_last, result);
        GGML_UNUSED(params);
        GGML_UNUSED(draft_log_probs);
    }

    void accept(uint16_t n_accepted) override {
        common_ngram_map_accept(config, n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return config.size_value;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return config.size_value;
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result,
            std::vector<float> * draft_log_probs = nullptr) override {
        GGML_UNUSED(draft_log_probs);
        const auto & sparams = params.ngram_mod;

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + sparams.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

        for (int i = 0; i < sparams.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < sparams.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            if (f_acc < 0.5) {
                n_low++;
                if (n_low >= 3) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);
                    }

                    mod.reset();
                    n_low = 0;
                    i_last = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.ngram_mod.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.ngram_mod.n_min;
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result,
            std::vector<float> * draft_log_probs = nullptr) override {
        GGML_UNUSED(params);
        GGML_UNUSED(draft_log_probs);

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last); // add the last token

            // Update context ngram cache with new prompt_tgt:
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            cache_size = prompt_tgt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted) override {
        // TODO: noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return n_draft;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return 0;
    }
};

struct common_speculative_state_suffix : public common_speculative_state {
    SuffixTree tree;
    static constexpr int SEQ_ID = 1;

    int32_t max_depth;
    int32_t n_draft_max;
    float   spec_factor;
    float   spec_offset;
    float   min_prob;

    size_t tree_size = 0;  // number of tokens fed to the tree (prompt_tgt.size() + 1)

    common_speculative_state_suffix(
            const enum common_speculative_type type,
            int32_t max_depth,
            int32_t n_draft_max,
            float   spec_factor,
            float   spec_offset,
            float   min_prob)
        : common_speculative_state(type)
        , tree(max_depth)
        , max_depth(max_depth)
        , n_draft_max(n_draft_max)
        , spec_factor(spec_factor)
        , spec_offset(spec_offset)
        , min_prob(min_prob)
    {}

    void begin(const llama_tokens & prompt) override {
        tree = SuffixTree(max_depth);
        tree_size = 0;
        if (!prompt.empty()) {
            tree.extend(SEQ_ID, prompt.data(), prompt.size());
            tree_size = prompt.size();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result,
            std::vector<float> * draft_log_probs = nullptr) override {
        GGML_UNUSED(params);
        GGML_UNUSED(draft_log_probs);

        // feed new tokens to suffix tree (same pattern as ngram_cache)
        if (tree_size < prompt_tgt.size() + 1) {
            for (size_t j = tree_size; j < prompt_tgt.size(); ++j) {
                tree.append(SEQ_ID, prompt_tgt[j]);
            }
            tree.append(SEQ_ID, id_last);
            tree_size = prompt_tgt.size() + 1;
        }

        // build full context for pattern matching
        std::vector<int32_t> context;
        context.reserve(prompt_tgt.size() + 1);
        for (size_t i = 0; i < prompt_tgt.size(); i++) {
            context.push_back(prompt_tgt[i]);
        }
        context.push_back(id_last);

        if (context.size() < 2) { return; }

        SuffixDraft draft = tree.speculate(
            context.data(), context.size(),
            n_draft_max, spec_factor, spec_offset, min_prob, false);

        for (size_t i = 0; i < draft.token_ids.size(); i++) {
            result.push_back(draft.token_ids[i]);
        }
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.n_max;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return 0;
    }
};

// CopySpec: draft by copying matching subsequences from the prompt context.
// Builds a rolling-hash index of all gamma-length windows in the prompt.
// On each draft call, hashes the last gamma tokens of output and looks up matches.
struct common_speculative_state_copyspec : public common_speculative_state {
    static constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    static constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

    int32_t gamma; // window size for matching

    // hash of gamma-length window -> position after the window in the prompt
    std::unordered_multimap<uint64_t, int32_t> index;
    llama_tokens prompt_tokens;

    int32_t original_prompt_size = 0;
    bool has_model_drafter = false; // true when paired with DFlash/draft (apply primary threshold)

    common_speculative_state_copyspec(enum common_speculative_type type, int32_t gamma)
        : common_speculative_state(type)
        , gamma(gamma)
    {}

    static uint64_t hash_window(const llama_token * tokens, int32_t len) {
        uint64_t h = FNV_OFFSET;
        for (int32_t i = 0; i < len; i++) {
            h ^= (uint64_t)(uint32_t)tokens[i];
            h *= FNV_PRIME;
        }
        return h;
    }

    void begin(const llama_tokens & prompt) override {
        index.clear();
        prompt_tokens = prompt;
        original_prompt_size = (int32_t)prompt.size();
        if ((int32_t)prompt.size() <= gamma) {
            return;
        }
        for (int32_t i = 0; i <= (int32_t)prompt.size() - gamma; i++) {
            uint64_t h = hash_window(prompt.data() + i, gamma);
            index.emplace(h, i + gamma);
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result,
            std::vector<float> * draft_log_probs = nullptr) override {
        GGML_UNUSED(draft_log_probs);
        // build the full context (prompt_tgt + id_last)
        const int32_t ctx_len = (int32_t)prompt_tgt.size() + 1;
        if (ctx_len < gamma) {
            return;
        }

        // hash the last gamma tokens of context
        std::vector<llama_token> window(gamma);
        const int32_t start = ctx_len - gamma;
        for (int32_t i = 0; i < gamma; i++) {
            const int32_t pos = start + i;
            window[i] = (pos < (int32_t)prompt_tgt.size()) ? prompt_tgt[pos] : id_last;
        }
        uint64_t h = hash_window(window.data(), gamma);

        // find longest match in prompt
        int32_t best_pos = -1;
        int32_t best_avail = 0; // uncapped available tokens after match
        auto range = index.equal_range(h);
        for (auto it = range.first; it != range.second; ++it) {
            int32_t pos = it->second;
            // verify hash match (collision check)
            if (pos < gamma || pos > (int32_t)prompt_tokens.size()) {
                continue;
            }
            bool match = true;
            for (int32_t j = 0; j < gamma; j++) {
                if (prompt_tokens[pos - gamma + j] != window[j]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                continue;
            }
            int32_t avail = (int32_t)prompt_tokens.size() - pos;
            if (avail > best_avail) {
                best_avail = avail;
                best_pos = pos;
            }
        }

        if (best_pos < 0) {
            return;
        }

        // when paired with a model-based drafter, only fire as primary if the match
        // has enough original prompt tokens to justify skipping the model drafter
        if (has_model_drafter) {
            const int32_t avail_in_orig = std::max(0, original_prompt_size - best_pos);
            if (avail_in_orig < 2 * params.n_max) {
                return;
            }
        }

        const int32_t draft_len = std::min(params.n_max, best_avail);
        for (int32_t i = 0; i < draft_len; i++) {
            result.push_back(prompt_tokens[best_pos + i]);
        }
    }

    // Extend an existing draft by looking for suffix matches at the end of (prompt + draft)
    void extend(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        if (result.empty() || index.empty()) {
            return;
        }

        // build full context: prompt_tgt + id_last + result
        // hash the last gamma tokens of this extended context
        const int32_t ctx_len = (int32_t)prompt_tgt.size() + 1 + (int32_t)result.size();
        if (ctx_len < gamma) {
            return;
        }

        std::vector<llama_token> window(gamma);
        const int32_t start = ctx_len - gamma;
        for (int32_t i = 0; i < gamma; i++) {
            const int32_t pos = start + i;
            if (pos < (int32_t)prompt_tgt.size()) {
                window[i] = prompt_tgt[pos];
            } else if (pos == (int32_t)prompt_tgt.size()) {
                window[i] = id_last;
            } else {
                window[i] = result[pos - (int32_t)prompt_tgt.size() - 1];
            }
        }
        uint64_t h = hash_window(window.data(), gamma);

        // find longest match
        int32_t best_pos = -1;
        int32_t best_len = 0;
        const int32_t max_ext = params.n_max - (int32_t)result.size();
        if (max_ext <= 0) {
            return;
        }

        auto range = index.equal_range(h);
        for (auto it = range.first; it != range.second; ++it) {
            int32_t pos = it->second;
            if (pos < gamma || pos > (int32_t)prompt_tokens.size()) {
                continue;
            }
            bool match = true;
            for (int32_t j = 0; j < gamma; j++) {
                if (prompt_tokens[pos - gamma + j] != window[j]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                continue;
            }
            int32_t avail = std::min(max_ext, (int32_t)prompt_tokens.size() - pos);
            if (avail > best_len) {
                best_len = avail;
                best_pos = pos;
            }
        }

        if (best_pos < 0) {
            return;
        }

        for (int32_t i = 0; i < best_len; i++) {
            result.push_back(prompt_tokens[best_pos + i]);
        }
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }

    // incrementally extend index with accepted tokens
    void update_logits(llama_context * /*ctx*/, const llama_tokens & batch_tokens, int n_accepted) override {
        // batch_tokens = [id_last, draft0, draft1, ...], n_accepted of which were accepted
        for (int i = 0; i < n_accepted && i < (int)batch_tokens.size(); i++) {
            prompt_tokens.push_back(batch_tokens[i]);
            // add new gamma-length window ending at this position
            if ((int32_t)prompt_tokens.size() >= gamma) {
                int32_t start = (int32_t)prompt_tokens.size() - gamma;
                uint64_t h = hash_window(prompt_tokens.data() + start, gamma);
                index.emplace(h, (int32_t)prompt_tokens.size());
            }
        }
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.n_max;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return 0;
    }
};

// Token Recycling: adjacency matrix tracking top-k successors per token.
// Seeded from observed bigrams, then updated from model logits after each
// verification decode. Logit-based entries have much higher scores and
// dominate the adjacency matrix after the first few iterations.
struct common_speculative_state_recycle : public common_speculative_state {
    int32_t k; // top-k successors per token

    // adjacency: token -> vector of (score, successor) pairs, sorted by score descending
    // scores: bigram observations use small integer counts (1, 2, ...),
    //         logit-derived entries use logit values (typically 10-30+ for top tokens)
    std::unordered_map<llama_token, std::vector<std::pair<float, llama_token>>> adj;

    size_t n_fed = 0;
    int32_t n_vocab = 0;

    common_speculative_state_recycle(enum common_speculative_type type, int32_t k)
        : common_speculative_state(type)
        , k(k)
    {}

    void set_successors(llama_token tok, const float * logits, int32_t vocab_size) {
        // partial sort to find top-k logits
        std::vector<std::pair<float, llama_token>> top(k, std::make_pair(-INFINITY, (llama_token)-1));
        for (int32_t i = 0; i < vocab_size; i++) {
            if (logits[i] > top[k-1].first) {
                top[k-1] = std::make_pair(logits[i], (llama_token)i);
                // bubble up
                for (int32_t j = k-2; j >= 0; j--) {
                    if (top[j+1].first > top[j].first) {
                        std::swap(top[j], top[j+1]);
                    } else {
                        break;
                    }
                }
            }
        }
        // remove unfilled slots
        while (!top.empty() && top.back().second < 0) {
            top.pop_back();
        }
        adj[tok] = std::move(top);
    }

    void add_bigram(llama_token a, llama_token b) {
        auto & succs = adj[a];
        for (size_t i = 0; i < succs.size(); i++) {
            if (succs[i].second == b) {
                succs[i].first += 1.0f;
                while (i > 0 && succs[i].first > succs[i-1].first) {
                    std::swap(succs[i], succs[i-1]);
                    i--;
                }
                return;
            }
        }
        if ((int32_t)succs.size() < k) {
            succs.push_back(std::make_pair(1.0f, b));
        }
    }

    void begin(const llama_tokens & prompt) override {
        adj.clear();
        n_fed = 0;
        for (size_t i = 0; i + 1 < prompt.size(); i++) {
            add_bigram(prompt[i], prompt[i + 1]);
        }
        n_fed = prompt.size();
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result,
            std::vector<float> * draft_log_probs = nullptr) override {
        GGML_UNUSED(draft_log_probs);
        // feed new bigrams from generated tokens
        if (n_fed < prompt_tgt.size() + 1) {
            size_t start = (n_fed > 0) ? n_fed - 1 : 0;
            for (size_t i = start; i < prompt_tgt.size(); i++) {
                llama_token next = (i + 1 < prompt_tgt.size()) ? prompt_tgt[i + 1] : id_last;
                add_bigram(prompt_tgt[i], next);
            }
            n_fed = prompt_tgt.size() + 1;
        }

        // greedy walk through adjacency matrix
        llama_token cur = id_last;
        for (int32_t i = 0; i < params.n_max; i++) {
            auto it = adj.find(cur);
            if (it == adj.end() || it->second.empty()) {
                break;
            }
            cur = it->second[0].second;
            result.push_back(cur);
        }
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }

    void update_logits(llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) override {
        if (n_vocab == 0) {
            const llama_model * model = llama_get_model(ctx);
            n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        }
        // update adjacency from logits for each position that had logits computed
        // batch_tokens[i] is the token at position i; logits at i predict its successor
        const int n_positions = std::min(n_accepted, (int)batch_tokens.size());
        for (int i = 0; i < n_positions; i++) {
            const float * logits = llama_get_logits_ith(ctx, i);
            if (logits) {
                set_successors(batch_tokens[i], logits, n_vocab);
            }
        }
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.n_max;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return 0;
    }
};

static std::string common_dflash_layer_ids_to_string(const std::vector<int32_t> & ids) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
            ss << ",";
        }
        ss << ids[i];
    }
    ss << "]";
    return ss.str();
}

static bool common_dflash_layer_ids_unique(const std::vector<int32_t> & ids) {
    std::set<int32_t> seen;
    for (const int32_t id : ids) {
        if (!seen.insert(id).second) {
            return false;
        }
    }
    return true;
}

// DFlash block-diffusion speculative decoding
// Uses an external drafter model conditioned on target hidden states via KV injection

enum class dflash_capture_source {
    cpu_hidden,         // eval callback CPU path
    verify_gpu_hidden,  // small 25-token verify buffers
    prefill_gpu_hidden, // large ubatch-sized prefill staging buffers
};

struct common_speculative_state_dflash : public common_speculative_state {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;
    llama_model   * model_dft;
    bool            owns_ctx_dft; // when false, ctx_dft is externally owned (shared across slots)
    llama_seq_id    seq_id = 0;   // which server slot this state owns

    int block_size;
    llama_token mask_token_id;
    int n_target_layers;
    int n_embd;
    int n_target_features;

    // Ring buffer for target hidden states — fixed memory regardless of context length
    // Stores last RING_SIZE tokens per layer in circular fashion
    static constexpr int RING_SIZE = 4096;

    // ring_buf[layer][slot * n_embd ... (slot+1) * n_embd - 1], slot = pos % RING_SIZE
    std::vector<std::vector<float>> ring_buf; // [n_target_layers][RING_SIZE * n_embd]
    int ring_write_pos = 0;    // next write slot (0..RING_SIZE-1)
    int ring_filled = 0;       // how many valid slots (0..RING_SIZE)
    int committed_len = 0;     // total tokens committed (unbounded counter)
    bool prefill_flushed = false; // true if flush_prefill() produced tokens during this request
    bool prefill_flush_called = false; // true if flush_prefill() was called at all
    int  prefill_flush_requested = 0; // total tokens requested across all flush_prefill() calls
    int  prefill_flush_written = 0;    // total tokens actually written across all flush_prefill() calls
    bool prefill_suffix_seen = false; // true if set_prefill_capture_enabled(true) was called during this request

    // Interleaved cross-attention buffer — rebuilt from ring on each draft call
    // Only holds cross_ctx tokens worth of data
    std::vector<float> cross_buf;
    std::vector<float> gpu_restore_staging;

    // A2: sliding window limit for drafter context (0 = unlimited)
    int cross_ctx; // from cparams, default 512

    // GPU cross-attention ring (nullptr = CPU fallback)
    void * gpu_ring_handle = nullptr;
    bool profile = common_dflash_profile_enabled();
    bool kv_cache_init_attempted = false;
    bool kv_cache_enabled = false;

    // Adaptive draft length tracking
    int n_low_accept = 0;
    int n_draft_last = 0;
    int adaptive_n_draft = -1; // -1 = use default

    // Validate that target hidden capture produced expected shapes.
    // Returns true if shapes match, false if mismatched (logs warning).
    bool validate_target_hiddens(const char * where) {
        const int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots != n_target_layers) {
            LOG_WRN("dflash: %s hidden slot count mismatch: got=%d expected=%d\n",
                    where, (int) n_slots, n_target_layers);
            return false;
        }

        for (int i = 0; i < n_slots; ++i) {
            const int64_t h_embd = llama_get_layer_hidden_n_embd(ctx_tgt, i);
            const int64_t h_tok  = llama_get_layer_hidden_n_tokens(ctx_tgt, i);
            if (h_embd != n_embd || h_tok < 0) {
                LOG_WRN("dflash: %s hidden[%d] shape mismatch: embd=%lld expected=%d tokens=%lld\n",
                        where, i, (long long) h_embd, n_embd, (long long) h_tok);
                return false;
            }
        }

        return true;
    }

    // Stored target capture layer IDs for toggling prefill hidden capture on/off.
    std::vector<int32_t> capture_layers;
    bool target_capture_enabled = true;
    bool gpu_capture_available = false;

    // build interleaved cross-attention data from ring buffer (GPU or CPU path)
    int build_cross_data(llama_context * ctx) {
        LOG_DBG("DFLASH_DBG build_cross_data: ring_write_pos=%d ring_filled=%d committed_len=%d cross_ctx=%d gpu=%d\n",
            ring_write_pos, ring_filled, committed_len, cross_ctx, gpu_ring_handle ? 1 : 0);
        if (gpu_ring_handle) {
            int gpu_write_pos = ring_write_pos % cross_ctx;
            int gpu_filled = std::min(ring_filled, cross_ctx);
            llama_dflash_cross_ring_gpu_set_cross(ctx, gpu_ring_handle, seq_id,
                gpu_write_pos, gpu_filled, n_target_layers, n_embd, cross_ctx);
            return gpu_filled;
        }
        int cross_len = std::min(ring_filled, cross_ctx > 0 ? cross_ctx : ring_filled);
        cross_buf.resize((size_t)n_target_features * cross_len);
        int read_start = (ring_write_pos - cross_len + RING_SIZE) % RING_SIZE;
        for (int t = 0; t < cross_len; ++t) {
            int slot = (read_start + t) % RING_SIZE;
            for (int layer = 0; layer < n_target_layers; ++layer) {
                memcpy(&cross_buf[(size_t)(layer * n_embd) + (size_t)t * n_target_features],
                       ring_buf[layer].data() + (size_t)slot * n_embd,
                       n_embd * sizeof(float));
            }
        }
        llama_set_cross_data_seq(ctx, seq_id, cross_buf.data(), n_target_features, cross_len);
        return cross_len;
    }

    void update_drafter_kv_cache(int n_written) {
        if (!gpu_ring_handle || n_written <= 0) {
            return;
        }
        if (common_dflash_kv_cache_disabled()) {
            return;
        }

        if (!kv_cache_init_attempted) {
            kv_cache_init_attempted = true;
            kv_cache_enabled = llama_dflash_kv_cache_init(ctx_dft, cross_ctx);
            if (kv_cache_enabled) {
                LOG_INF("dflash: drafter K/V projection cache enabled (%d-token window)\n", cross_ctx);
            } else {
                LOG_WRN("dflash: drafter K/V projection cache unavailable; using full-window K/V projection\n");
            }
        }
        if (!kv_cache_enabled) {
            return;
        }

        const int n_update = std::min(n_written, cross_ctx);
        const int gpu_write_pos = ring_write_pos % cross_ctx;
        const int gpu_filled = std::min(ring_filled, cross_ctx);

        const int64_t t_start = profile ? ggml_time_us() : 0;
        const bool ok = llama_dflash_kv_cache_update_from_ring(ctx_dft, gpu_ring_handle,
                gpu_write_pos, gpu_filled, n_target_layers, n_embd, n_update);
        if (profile) {
            LOG_INF("dflash profile: kv_cache_update requested=%d update=%d ok=%d time=%.3f ms ring_pos=%d filled=%d committed=%d\n",
                    n_written, n_update, ok ? 1 : 0, (ggml_time_us() - t_start) / 1e3,
                    gpu_write_pos, gpu_filled, committed_len);
        }

        if (!ok) {
            static bool warned = false;
            if (!warned) {
                LOG_WRN("dflash: drafter K/V projection cache update failed; falling back to full-window projection\n");
                warned = true;
            }
            llama_dflash_kv_cache_reset(ctx_dft);
            kv_cache_enabled = false;
        }
    }

    llama_batch batch_dft;

    common_speculative_state_dflash(
            llama_context * ctx_tgt_,
            llama_context * ctx_dft_,
            llama_model   * model_dft_,
            bool            owns_ctx_dft_ = true,
            int             cross_ctx_ = 512)
        : common_speculative_state(COMMON_SPECULATIVE_TYPE_DFLASH)
        , ctx_tgt(ctx_tgt_)
        , ctx_dft(ctx_dft_)
        , model_dft(model_dft_)
        , owns_ctx_dft(owns_ctx_dft_)
        , cross_ctx(cross_ctx_ > 0 ? cross_ctx_ : 512)
    {
        block_size        = llama_model_dflash_block_size(model_dft_);
        mask_token_id     = (llama_token) llama_model_dflash_mask_token_id(model_dft_);
        n_target_layers   = llama_model_dflash_n_target_layers(model_dft_);
        n_embd            = llama_model_n_embd(model_dft_);
        n_target_features = llama_model_dflash_n_target_features(model_dft_);

        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        const int target_n_layer = model_tgt ? llama_model_n_layer(model_tgt) : 0;
        const int target_n_embd  = model_tgt ? llama_model_n_embd(model_tgt)  : 0;

        auto fail_contract = [&](const std::string & why) {
            throw std::runtime_error(string_format(
                "dflash: invalid drafter/target contract: %s "
                "(block_size=%d mask_token=%d n_target_layers=%d draft_n_embd=%d "
                "target_n_layer=%d target_n_embd=%d n_target_features=%d)",
                why.c_str(),
                block_size,
                (int) mask_token_id,
                n_target_layers,
                n_embd,
                target_n_layer,
                target_n_embd,
                n_target_features));
        };

        if (!model_tgt) {
            fail_contract("target model is null");
        }
        if (block_size <= 1) {
            fail_contract("block_size must be greater than 1");
        }
        if (mask_token_id < 0) {
            fail_contract("mask_token_id must be non-negative");
        }
        if (n_target_layers <= 0) {
            fail_contract("n_target_layers must be positive");
        }
        if (n_embd <= 0 || target_n_embd <= 0) {
            fail_contract("embedding sizes must be positive");
        }
        if (n_embd != target_n_embd) {
            fail_contract("drafter n_embd must match target hidden size");
        }
        if (n_target_features != n_embd * n_target_layers) {
            fail_contract("n_target_features must equal n_embd * n_target_layers");
        }

        std::vector<int32_t> capture_layers(n_target_layers);
        const int n_read = llama_model_dflash_target_layer_ids(model_dft_, capture_layers.data(), n_target_layers);
        if (n_read != n_target_layers) {
            fail_contract(string_format("target_layer_ids read count mismatch: read %d expected %d", n_read, n_target_layers));
        }
        if (!common_dflash_layer_ids_unique(capture_layers)) {
            fail_contract("target_layer_ids contain duplicates");
        }
        for (const int32_t il : capture_layers) {
            if (il < 0 || il >= target_n_layer) {
                fail_contract(string_format("target_layer_id %d is outside target layer range [0,%d)", il, target_n_layer));
            }
        }

        LOG_INF("dflash: contract ok: block_size=%d mask_token=%d target_layer_ids=%s "
                "n_target_layers=%d n_embd=%d n_target_features=%d target_layers=%d cross_ctx=%d\n",
                block_size,
                (int) mask_token_id,
                common_dflash_layer_ids_to_string(capture_layers).c_str(),
                n_target_layers,
                n_embd,
                n_target_features,
                target_n_layer,
                cross_ctx);

        {
            const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
            const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft_);
            const int vocab_tgt_n = vocab_tgt ? llama_vocab_n_tokens(vocab_tgt) : 0;
            const int vocab_dft_n = vocab_dft ? llama_vocab_n_tokens(vocab_dft) : 0;

            int32_t capture_min = capture_layers.empty() ? -1 : capture_layers[0];
            int32_t capture_max = capture_layers.empty() ? -1 : capture_layers[0];
            for (int32_t il : capture_layers) {
                capture_min = std::min(capture_min, il);
                capture_max = std::max(capture_max, il);
            }

            LOG_INF("dflash: target/drafter info: target_ctx_train=%d target_vocab=%d drafter_vocab=%d vocab_match=%d capture_min=%d capture_max=%d\n",
                    llama_model_n_ctx_train(model_tgt),
                    vocab_tgt_n,
                    vocab_dft_n,
                    vocab_tgt_n == vocab_dft_n ? 1 : 0,
                    capture_min,
                    capture_max);
        }

        ring_buf.resize(n_target_layers);
        for (int i = 0; i < n_target_layers; ++i) {
            ring_buf[i].resize((size_t)RING_SIZE * n_embd, 0.0f);
        }

        // tok_embd/output sharing must happen BEFORE context creation
        // (done in speculative-simple.cpp before common_speculative_init)

        // configure target context to capture hidden states
        capture_layers.resize(n_target_layers);
        llama_model_dflash_target_layer_ids(model_dft_, capture_layers.data(), n_target_layers);
        llama_set_dflash_capture(ctx_tgt, capture_layers.data(), n_target_layers);
        target_capture_enabled = true;

        batch_dft = llama_batch_init(block_size, 0, 1);

        // Try to allocate the GPU ring only when both contexts can safely share
        // single-backend cross tensors. Otherwise keep the target eval callback
        // active so the CPU ring fallback has hidden states to consume.
        const bool gpu_ring_allowed = common_dflash_gpu_ring_allowed(ctx_tgt, ctx_dft);
        const bool gpu_ring_forced_off = common_dflash_force_cpu_cross_ring();
        const bool gpu_ring_requested = gpu_ring_allowed && !gpu_ring_forced_off;

        llama_set_dflash_gpu_capture(ctx_tgt, gpu_ring_requested);

        LOG_INF("dflash: GPU hidden capture policy: allowed=%d forced_cpu=%d requested=%d target_devices=%d drafter_devices=%d\n",
                gpu_ring_allowed ? 1 : 0,
                gpu_ring_forced_off ? 1 : 0,
                gpu_ring_requested ? 1 : 0,
                llama_model_n_devices(llama_get_model(ctx_tgt)),
                llama_model_n_devices(model_dft));

        if (gpu_ring_requested) {
            gpu_ring_handle = llama_dflash_cross_ring_gpu_init(ctx_dft, n_target_layers, n_embd, cross_ctx);
        }
        if (gpu_ring_forced_off) {
            LOG_INF("dflash: GPU cross ring forced off by GGML_DFLASH_FORCE_CPU_CROSS; using CPU hidden capture\n");
        }
        if (gpu_ring_handle) {
            LOG_INF("dflash: GPU cross ring enabled (%d layers x %d slots x %d embd)\n",
                    n_target_layers, cross_ctx, n_embd);
            gpu_capture_available = true;
        } else if (gpu_ring_requested) {
            llama_set_dflash_gpu_capture(ctx_tgt, false);
            LOG_WRN("dflash: GPU cross ring unavailable; using CPU hidden capture\n");
        }

    }

    ~common_speculative_state_dflash() override {
        llama_dflash_cross_ring_gpu_free(gpu_ring_handle);
        llama_batch_free(batch_dft);
        if (owns_ctx_dft) {
            llama_free(ctx_dft);
        }
    }

    void set_seq_id(llama_seq_id seq_id_) override {
        seq_id = seq_id_;
    }

    void set_prefill_capture_enabled(bool enabled) override {
        if (target_capture_enabled == enabled) {
            return;
        }

        target_capture_enabled = enabled;

        if (enabled) {
            // Restore logical capture activity.  The layer configuration,
            // GPU buffers, and tape metadata were preserved across the
            // temporary disable, so no re-allocation is needed.
            llama_set_dflash_capture_active(ctx_tgt, true);
            prefill_suffix_seen = true;
            if (profile) {
                LOG_INF("dflash prefill capture: enabled hidden capture gpu=%d layers=%d\n",
                        gpu_capture_available ? 1 : 0, (int) capture_layers.size());
            }
        } else {
            // Logically disable hidden capture without destroying GPU
            // buffers, tape metadata, or layer configuration.
            llama_set_dflash_capture_active(ctx_tgt, false);
            if (profile) {
                LOG_INF("dflash prefill capture: disabled hidden capture for non-suffix prompt chunk\n");
            }
        }
    }

    // prepare cross-attention data for batched draft decode.
    // Returns cross_len (position offset for tokens), or -1 if no committed tokens.
    int prepare_batch_draft(llama_context * ctx_dft_ext) override {
        if (committed_len == 0) {
            return -1;
        }

        return build_cross_data(ctx_dft_ext);
    }

    // called after initial prefill — extract hidden states from target
    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);

        // Invariant: if suffix prefill was seen, verify that flush_prefill()
        // was actually called and that tokens reached the ring.
        if (prefill_suffix_seen && !prefill_flush_called) {
            LOG_ERR("dflash: prefill suffix was scheduled but flush_prefill() was never called (ring_filled=%d committed_len=%d)\n",
                ring_filled, committed_len);
        }
        if (prefill_suffix_seen && prefill_flush_called && prefill_flush_written <= 0) {
            LOG_ERR("dflash: prefill suffix flush was called but wrote 0/%d tokens (ring_filled=%d committed_len=%d)\n",
                prefill_flush_requested, ring_filled, committed_len);
        }
        if (prefill_suffix_seen && prefill_flush_written > 0 && ring_filled <= 0) {
            LOG_ERR("dflash: prefill suffix flush wrote %d tokens but ring is empty (ring_filled=%d committed_len=%d)\n",
                prefill_flush_written, ring_filled, committed_len);
        }

        if (prefill_flushed) {
            // ring was already populated incrementally by flush_prefill() calls
            // during checkpoint-split prefill — nothing to do
            prefill_flushed = false;
            prefill_flush_called = false;
            prefill_flush_requested = 0;
            prefill_flush_written = 0;
            prefill_suffix_seen = false;
            llama_dflash_prefill_capture_end(ctx_tgt);
            return;
        }
        if (!validate_target_hiddens("begin")) {
            return;
        }
        capture_target_hiddens();
    }

    int flush_prefill(int src_offset = 0, int n_tokens = 0) override {
        llama_dflash_set_active_slot(ctx_tgt, seq_id);

        prefill_flush_called = true;
        prefill_flush_requested += n_tokens;

        if (!validate_target_hiddens("flush_prefill")) {
            return 0;
        }

        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0) return 0;

        // Determine capture source: GPU prefill staging or CPU callback.
        // Source-first: check prefill GPU before CPU hidden, because in
        // GPU-staging mode the CPU callback was not installed and
        // llama_get_n_layer_hiddens() returns 0.
        const bool use_prefill_gpu = llama_dflash_prefill_gpu_active(ctx_tgt);

        dflash_capture_source source;
        int64_t captured = 0;
        int32_t n_src_layers = 0;
        int offset = 0;

        if (use_prefill_gpu) {
            source = dflash_capture_source::prefill_gpu_hidden;
            n_src_layers = n_target_layers;

            // For GPU prefill staging, the buffer is window-relative:
            // tokens were accumulated starting at offset 0 in the staging buffer.
            // Use capture plan info for the accumulated count.
            int32_t planned = 0;
            int32_t written = 0;
            if (llama_dflash_prefill_capture_info(ctx_tgt, seq_id, &planned, &written)) {
                captured = written;
            } else {
                captured = llama_dflash_prefill_gpu_n_tokens(ctx_tgt, seq_id);
            }

            // GPU staging flush is window-relative — always flush from offset 0.
            offset = 0;

            if (n_tokens > 0 && captured < n_tokens) {
                LOG_WRN("dflash prefill flush incomplete GPU capture: captured=%lld requested=%d seq=%d\n",
                    (long long) captured, n_tokens, seq_id);
            }
        } else {
            source = dflash_capture_source::cpu_hidden;
            n_src_layers = llama_get_n_layer_hiddens(ctx_tgt);
            if (n_src_layers == 0) {
                if (profile) {
                    LOG_INF("dflash prefill flush skipped: source=cpu reason=no-layer-slots n_src_layers=0\n");
                }
                return 0;
            }
            captured = llama_get_layer_hidden_n_tokens(ctx_tgt, 0);

            // CPU hidden path uses the original src_offset which is
            // outer-batch-relative into the CPU hidden buffer.
            offset = n_tokens > 0 ? src_offset : 0;
        }

        if (captured <= 0) {
            if (profile) {
                LOG_INF("dflash prefill flush skipped: source=%s reason=no-captured-tokens captured=0 n_tokens_arg=%d src_offset=%d\n",
                    source == dflash_capture_source::prefill_gpu_hidden ? "prefill_gpu" : "cpu",
                    n_tokens, src_offset);
            }
            return 0;
        }

        // When n_tokens=0, flush the entire capture buffer (legacy path).
        // When n_tokens>0, flush only the span [offset, offset+n_tokens).
        int to_write = n_tokens > 0 ? n_tokens : (int)captured;

        // Clamp to available captured tokens.
        if (offset < 0) offset = 0;
        if (offset + to_write > (int)captured) {
            to_write = std::max(0, (int)captured - offset);
        }
        if (to_write <= 0) {
            if (profile) {
                LOG_INF("dflash prefill flush skipped: source=%s reason=empty-clamped-span captured=%lld offset=%d to_write=%d\n",
                    source == dflash_capture_source::prefill_gpu_hidden ? "prefill_gpu" : "cpu",
                    (long long)captured, offset, to_write);
            }
            return 0;
        }

        if (profile) {
            LOG_INF("dflash prefill flush: source=%s captured=%lld offset=%d n_tokens=%d to_write=%d n_src_layers=%d prefill_flushed=%d ring_write_pos=%d ring_filled=%d committed_len=%d gpu=%d\n",
                source == dflash_capture_source::prefill_gpu_hidden ? "prefill_gpu" : "cpu",
                (long long)captured, offset, n_tokens, to_write, n_src_layers, (int)prefill_flushed, ring_write_pos, ring_filled, committed_len,
                gpu_ring_handle ? 1 : 0);
        }

        if (!prefill_flushed) {
            // first flush for this request — reset ring
            ring_write_pos = 0;
            ring_filled = 0;
            committed_len = 0;
            llama_dflash_kv_cache_reset(ctx_dft);
        }

        const int actual_written = ring_write(to_write, offset, true, source);
        committed_len += actual_written;
        update_drafter_kv_cache(actual_written);
        prefill_flushed = true;
        prefill_flush_written += actual_written;
        return actual_written;
    }

    // Ring state serialization for checkpoint persistence.
    // Format: [ring_write_pos:i32][ring_filled:i32][committed_len:i32]
    //         [n_target_layers:i32][n_embd:i32][n_entries:i32]
    //         [layer_0 data: n_entries * n_embd * f32] ...

    size_t ring_state_size() const override {
        int n_entries = std::min(ring_filled, RING_SIZE);
        return 6 * sizeof(int32_t) +
               (size_t)n_entries * n_embd * sizeof(float) * n_target_layers;
    }

    void ring_state_save(uint8_t * buf, size_t size) const override {
        int n_entries = std::min(ring_filled, RING_SIZE);
        size_t expected = 6 * sizeof(int32_t) +
                          (size_t)n_entries * n_embd * sizeof(float) * n_target_layers;
        if (size < expected) return;

        int32_t * hdr = (int32_t *)buf;
        hdr[0] = ring_write_pos;
        hdr[1] = ring_filled;
        hdr[2] = committed_len;
        hdr[3] = n_target_layers;
        hdr[4] = n_embd;
        hdr[5] = n_entries;

        uint8_t * dst = buf + 6 * sizeof(int32_t);
        size_t layer_bytes = (size_t)n_entries * n_embd * sizeof(float);

        for (int l = 0; l < n_target_layers; ++l) {
            memcpy(dst, ring_buf[l].data(), layer_bytes);
            dst += layer_bytes;
        }
    }

    bool ring_state_load(const uint8_t * buf, size_t size) override {
        if (size < 6 * sizeof(int32_t)) return false;

        const int32_t * hdr = (const int32_t *)buf;
        int saved_write_pos = hdr[0];
        int saved_filled    = hdr[1];
        int saved_committed = hdr[2];
        int saved_layers    = hdr[3];
        int saved_embd      = hdr[4];
        int saved_entries    = hdr[5];

        if (saved_layers != n_target_layers || saved_embd != n_embd) {
            LOG_WRN("dflash: ring state mismatch: layers %d/%d, embd %d/%d\n",
                    saved_layers, n_target_layers, saved_embd, n_embd);
            return false;
        }

        if (saved_write_pos < 0 || saved_write_pos >= RING_SIZE ||
            saved_filled < 0 || saved_filled > RING_SIZE ||
            saved_entries < 0 || saved_entries != saved_filled ||
            saved_committed < saved_filled) {
            LOG_WRN("dflash: ring state corrupt: write_pos=%d, filled=%d, committed=%d, entries=%d\n",
                    saved_write_pos, saved_filled, saved_committed, saved_entries);
            return false;
        }

        size_t layer_bytes = (size_t)saved_entries * n_embd * sizeof(float);
        if (size < 6 * sizeof(int32_t) + layer_bytes * n_target_layers) return false;

        ring_write_pos = saved_write_pos;
        ring_filled = saved_filled;
        committed_len = saved_committed;

        LOG_WRN("DFLASH_DBG ring_state_load: write_pos=%d filled=%d committed=%d entries=%d gpu=%d\n",
            saved_write_pos, saved_filled, saved_committed, saved_entries, gpu_ring_handle ? 1 : 0);

        const uint8_t * src = buf + 6 * sizeof(int32_t);
        for (int l = 0; l < n_target_layers; ++l) {
            memcpy(ring_buf[l].data(), src, layer_bytes);
            src += layer_bytes;
        }

        // sync GPU ring with restored CPU ring — batch per layer to avoid N*L individual H2D calls
        if (gpu_ring_handle) {
            int gpu_entries = std::min(ring_filled, cross_ctx);
            const size_t layer_floats = (size_t) gpu_entries * n_embd;
            gpu_restore_staging.resize(layer_floats * n_target_layers);
            for (int l = 0; l < n_target_layers; ++l) {
                float * tmp = gpu_restore_staging.data() + layer_floats * l;
                for (int t = 0; t < gpu_entries; ++t) {
                    int cpu_slot = (ring_write_pos - gpu_entries + t + RING_SIZE) % RING_SIZE;
                    memcpy(tmp + (size_t)t * n_embd,
                           ring_buf[l].data() + (size_t)cpu_slot * n_embd,
                           n_embd * sizeof(float));
                }
                int gpu_pos = ((ring_write_pos - gpu_entries) % cross_ctx + cross_ctx) % cross_ctx;
                llama_dflash_cross_ring_gpu_write(gpu_ring_handle, l, gpu_pos,
                    tmp, gpu_entries, n_embd);
            }
            llama_dflash_cross_ring_gpu_synchronize(gpu_ring_handle);
            llama_dflash_kv_cache_reset(ctx_dft);
            update_drafter_kv_cache(gpu_entries);
        }

        // mark as flushed so subsequent flush_prefill() calls from suffix
        // decoding APPEND to the restored ring instead of resetting it
        prefill_flushed = true;

        return true;
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result,
            std::vector<float> * draft_log_probs = nullptr) override {
        GGML_UNUSED(prompt_tgt);

        const int n_draft_base = adaptive_n_draft > 0 ? adaptive_n_draft : (block_size - 1);
        const int n_draft = std::min(n_draft_base, params.n_max);
        if (committed_len == 0) {
            return;
        }

        const int64_t t0 = ggml_time_us();

        int cross_len = build_cross_data(ctx_dft);

        const int64_t t1 = ggml_time_us();

        // build drafter batch: [id_last, mask, mask, ..., mask]
        // positions are relative to the context window fed to the drafter
        // batch size adapts to n_draft+1 (saves compute when n_max < block_size-1)
        const int batch_len = n_draft + 1;
        common_batch_clear(batch_dft);
        common_batch_add(batch_dft, id_last, cross_len, { seq_id }, true);
        for (int i = 1; i < batch_len; ++i) {
            common_batch_add(batch_dft, mask_token_id, cross_len + i, { seq_id }, true);
        }

        const int64_t t2 = ggml_time_us();

        // run drafter forward pass
        int ret = llama_decode(ctx_dft, batch_dft);
        if (ret != 0) {
            LOG_ERR("dflash: drafter decode failed with %d\n", ret);
            return;
        }

        const int64_t t3 = ggml_time_us();

        // read argmax tokens for positions 1..batch_len-1 (skip position 0 = staged_first)
        {
            int32_t * argmax = llama_get_logits_argmax(ctx_dft);
            float * argmax_probs = llama_get_logits_argmax_probs(ctx_dft);
            const int K_flat = llama_get_logits_argmax_k(ctx_dft);
            if (argmax) {
                // GPU argmax path — only 64-128 bytes transferred instead of 15.9MB
                for (int i = 1; i < batch_len && (int) result.size() < n_draft; ++i) {
                    if (argmax_probs && params.p_min > 0.0f && i > 1) {
                        float log_prob = argmax_probs[i * K_flat];
                        float log_p_min = logf(params.p_min);
                        if (log_prob < log_p_min) {
                            LOG_DBG("dflash: early stop at position %d/%d (prob %.3f < p_min %.3f)\n",
                                    i, batch_len, expf(log_prob), params.p_min);
                            break;
                        }
                    }
                    result.push_back((llama_token) argmax[i * K_flat]);
                    if (draft_log_probs && argmax_probs) {
                        draft_log_probs->push_back(argmax_probs[i * K_flat]);
                    }
                }
            } else {
                // fallback: CPU argmax over full vocab
                const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
                for (int i = 1; i < batch_len && (int) result.size() < n_draft; ++i) {
                    float * logits = llama_get_logits_ith(ctx_dft, i);
                    if (!logits) {
                        break;
                    }
                    llama_token best = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
                    result.push_back(best);
                    if (draft_log_probs) {
                        draft_log_probs->push_back(0.0f); // no prob info in fallback
                    }
                }
            }
        }

        const int64_t t4 = ggml_time_us();

        n_draft_last = (int) result.size();

        if (profile) {
            const llama_perf_context_data perf_dft = llama_perf_context(ctx_dft);
            LOG_INF("dflash profile: draft ctx=%d cross_len=%d n_draft=%d produced=%d "
                    "cross=%.3f ms batch=%.3f ms decode=%.3f ms argmax=%.3f ms total=%.3f ms gpu_ring=%d graph_reuse=%d\n",
                    committed_len, cross_len, n_draft, (int) result.size(),
                    (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t4 - t3) / 1e3,
                    (t4 - t0) / 1e3, gpu_ring_handle ? 1 : 0, perf_dft.n_reused);
        } else {
            LOG_DBG("dflash draft breakdown (ctx=%d): concat=%.1fms cross=%.1fms decode=%.1fms argmax=%.1fms total=%.1fms\n",
                    committed_len,
                    (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t4 - t3) / 1e3, (t4 - t0) / 1e3);
        }
    }

    void accept(uint16_t n_accepted) override {
        if (n_draft_last > 0) {
            float f_acc = (float) n_accepted / (float) n_draft_last;
            if (f_acc < 0.3f) {
                n_low_accept++;
                if (n_low_accept >= 3) {
                    int base = adaptive_n_draft > 0 ? adaptive_n_draft : (block_size - 1);
                    adaptive_n_draft = std::max(1, base / 2);
                    LOG_DBG("dflash: low acceptance streak (%d) — reducing draft to %d\n",
                            n_low_accept, adaptive_n_draft);
                    n_low_accept = 0;
                }
            } else {
                n_low_accept = 0;
                if (f_acc > 0.6f && adaptive_n_draft > 0) {
                    adaptive_n_draft = std::min(block_size - 1, adaptive_n_draft + 1);
                }
            }
        }
    }

    void draft_tree(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            int tree_budget,
            common_speculative_tree & tree) override {
        const int n_draft = std::min((int) params.n_max, block_size - 1);
        if (n_draft <= 0 || committed_len == 0) {
            return;
        }

        // run drafter forward pass (same as flat draft)
        // --- begin shared draft setup ---
        const int64_t t0 = ggml_time_us();

        int cross_len = build_cross_data(ctx_dft);

        common_batch_clear(batch_dft);
        common_batch_add(batch_dft, id_last, cross_len, { seq_id }, true);
        for (int i = 1; i < block_size; ++i) {
            common_batch_add(batch_dft, mask_token_id, cross_len + i, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch_dft);
        if (ret != 0) {
            LOG_ERR("dflash: drafter decode failed with %d\n", ret);
            return;
        }

        const int64_t t_gpu = ggml_time_us();
        // --- end shared draft setup ---

        const int draft_horizon = std::min(n_draft, block_size - 1);

        const int K = llama_get_logits_argmax_k(ctx_dft);
        const bool can_branch = (K > 1);

        // n_max_base is the user's original --draft-max before adaptive DM reduction.
        // It determines whether the user intended branches:
        //   tree_budget <= n_max_base → pure chain (user didn't allocate extra for branches)
        //   tree_budget >  n_max_base → branches (user explicitly allocated beyond main path)
        // When adaptive DM reduces n_max, branch_budget scales proportionally — branches
        // shrink when the model is less confident, matching adaptive DM's intent.
        const int n_max_original = (params.n_max_base > 0) ? params.n_max_base : params.n_max;
        const int original_horizon = std::min(n_max_original, block_size - 1);

        int branch_budget = 0;
        if (can_branch && tree_budget > original_horizon) {
            const int base_branch_budget = tree_budget - original_horizon;
            const float ratio = (float)base_branch_budget / original_horizon;
            branch_budget = std::min(
                (int)std::round(draft_horizon * ratio),
                std::max(0, tree_budget - draft_horizon));
            branch_budget = std::max(0, branch_budget);
        }

        const int main_path_len = std::min(draft_horizon, std::max(1, tree_budget - branch_budget));
        const int depth_limit = draft_horizon;

        // Use GPU argmax/topk for tree building
        int32_t * argmax = llama_get_logits_argmax(ctx_dft);
        if (!argmax) {
            LOG_ERR("draft_tree: no GPU argmax available\n");
            return;
        }
        float * argmax_probs = llama_get_logits_argmax_probs(ctx_dft);

        // Build tree using best-first heap expansion with chain-seed backbone
        tree.tokens.clear();
        tree.parents.clear();
        tree.depths.clear();
        tree.child_maps.clear();
        tree.visibility.clear();
        tree.log_probs.clear();

        tree.parents.push_back(-1); // root parent
        tree.child_maps.push_back({}); // root child_map
        tree.n_nodes = 0;
        tree.main_path_len = 0;

        // Chain-seed: pre-insert greedy backbone (top-1 at each depth)
        // Only insert up to main_path_len nodes; reserve remaining budget for branches
        {
            int parent = 0;
            for (int d = 1; d <= depth_limit && d <= main_path_len && tree.n_nodes < tree_budget; ++d) {
                llama_token token_id = (llama_token) argmax[d * K];
                float log_prob = argmax_probs ? argmax_probs[d * K + 0] : -INFINITY;

                int current_idx = tree.n_nodes + 1;
                tree.tokens.push_back(token_id);
                tree.parents.push_back(parent);
                tree.depths.push_back(d);
                tree.log_probs.push_back(log_prob);
                tree.child_maps.push_back({});
                tree.child_maps[parent][token_id] = current_idx;
                tree.n_nodes++;

                parent = current_idx;
            }
            tree.main_path_len = tree.n_nodes;
        }

        // Best-first expansion using log-prob heap (DDTree Algorithm 1)
        if (K > 1 && branch_budget > 0 && argmax_probs) {
            // Heap entry: (cumulative_log_prob, parent_tree_idx, depth, rank)
            struct heap_entry {
                float  log_w;
                int    parent_idx;
                int    depth;  // 1-based position in draft sequence
                int    rank;   // rank within top-K at this depth
                bool operator<(const heap_entry & o) const { return log_w < o.log_w; }
            };

            std::priority_queue<heap_entry> heap;

            // Seed heap: siblings of the main chain at each depth
            // Only seed depths where main path has a node (1..main_path_len)
            float cum_log_prob = 0.0f;
            for (int d = 1; d <= main_path_len; ++d) {
                float sibling_lp = cum_log_prob + argmax_probs[d * K + 1];
                // Parent of sibling at depth d is the main path node at depth d-1.
                // Main path nodes are 1-indexed: depth1=node1, depth2=node2, etc.
                int sibling_parent = (d == 1) ? 0 : d - 1;
                heap.push({sibling_lp, sibling_parent, d, 1});
                cum_log_prob += argmax_probs[d * K + 0];
            }

            while (!heap.empty() && (tree.n_nodes - tree.main_path_len) < branch_budget) {
                auto top = heap.top();
                heap.pop();

                llama_token token_id = (llama_token) argmax[top.depth * K + top.rank];
                if (token_id < 0) continue;
                if (tree.child_maps[top.parent_idx].count(token_id)) continue;

                int current_idx = tree.n_nodes + 1;
                tree.tokens.push_back(token_id);
                tree.parents.push_back(top.parent_idx);
                tree.depths.push_back(top.depth);
                tree.log_probs.push_back(argmax_probs ? argmax_probs[top.depth * K + top.rank] : -INFINITY);
                tree.child_maps.push_back({});
                tree.child_maps[top.parent_idx][token_id] = current_idx;
                tree.n_nodes++;

                // Push sibling: same depth, next rank
                if (top.rank + 1 < K) {
                    float sib_lp = top.log_w - argmax_probs[top.depth * K + top.rank]
                                             + argmax_probs[top.depth * K + top.rank + 1];
                    heap.push({sib_lp, top.parent_idx, top.depth, top.rank + 1});
                }

                // Push child: extend this branch one depth deeper (rank 0)
                if (top.depth < depth_limit) {
                    int child_depth = top.depth + 1;
                    float child_lp = top.log_w + argmax_probs[child_depth * K + 0];
                    heap.push({child_lp, current_idx, child_depth, 0});
                }
            }
        } else if (K > 1 && branch_budget > 0) {
            // Fallback without log-probs: uniform sibling addition
            // Add siblings at main path depths only (same rule as heap seeding)
            for (int d = 1; d <= main_path_len && (tree.n_nodes - tree.main_path_len) < branch_budget; ++d) {
                int parent_idx = (d == 1) ? 0 : d - 1;
                for (int ki = 1; ki < K && (tree.n_nodes - tree.main_path_len) < branch_budget; ++ki) {
                    llama_token alt_token = (llama_token) argmax[d * K + ki];
                    if (alt_token < 0) continue;
                    if (tree.child_maps[parent_idx].count(alt_token)) continue;

                    int current_idx = tree.n_nodes + 1;
                    tree.tokens.push_back(alt_token);
                    tree.parents.push_back(parent_idx);
                    tree.depths.push_back(d);
                    tree.log_probs.push_back(-INFINITY);
                    tree.child_maps.push_back({});
                    tree.child_maps[parent_idx][alt_token] = current_idx;
                    tree.n_nodes++;
                }
            }
        }

        // build visibility matrix [(n_nodes+1) × (n_nodes+1)]
        int n = tree.n_nodes + 1;
        tree.visibility.assign(n * n, false);
        tree.visibility[0] = true; // root sees itself
        for (int i = 1; i < n; ++i) {
            int parent = tree.parents[i];
            // inherit parent's visibility row
            for (int j = 0; j < i; ++j) {
                tree.visibility[i * n + j] = tree.visibility[parent * n + j];
            }
            tree.visibility[i * n + i] = true; // see itself
        }

        const int64_t t1 = ggml_time_us();
        const int64_t gpu_ms = (t_gpu - t0) / 1000;
        const int64_t cpu_ms = (t1 - t_gpu) / 1000;
        LOG_INF("ddtree: built tree with %d nodes (%d main + %d branch, budget %d) in %.1fms (gpu=%.0fms cpu=%.0fms)\n",
                tree.n_nodes, tree.main_path_len, tree.n_nodes - tree.main_path_len,
                tree_budget, (t1 - t0) / 1e3, (double)gpu_ms, (double)cpu_ms);

        GGML_UNUSED(prompt_tgt);
    }

    // called after target verification decode — capture and append new hidden states
    void update_logits(llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) override {
        GGML_UNUSED(ctx);
        GGML_UNUSED(batch_tokens);
        // In this path n_accepted means committed hidden-state rows, not output-token count.
        // [id_last, draft0, ..., draftN-1] => root + accepted draft tokens.
        // Boundary stops pass root + accepted draft tokens even when no bonus token was sampled.
        append_target_hiddens(n_accepted);
    }

    // tree variant: write specific capture-buffer indices to the ring
    void update_logits_by_indices(llama_context * ctx, const std::vector<int> & capture_indices) override {
        GGML_UNUSED(ctx);
        llama_dflash_set_active_slot(ctx_tgt, seq_id);

        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0 || capture_indices.empty()) {
            return;
        }

        const int actual_written = ring_write_by_indices(capture_indices);
        committed_len += actual_written;
        update_drafter_kv_cache(actual_written);
    }

private:
    void log_ring_profile(const char * name, int n_tokens, int actual_written,
            int64_t cpu_copy_us, int64_t gpu_enqueue_us, int64_t gpu_sync_us) const {
        if (!profile) {
            return;
        }
        LOG_INF("dflash profile: %s requested=%d written=%d cpu_copy=%.3f ms gpu_enqueue=%.3f ms gpu_sync=%.3f ms ring_filled_before=%d committed_before=%d gpu=%d\n",
                name, n_tokens, actual_written,
                cpu_copy_us / 1e3, gpu_enqueue_us / 1e3, gpu_sync_us / 1e3,
                ring_filled, committed_len, gpu_ring_handle ? 1 : 0);
    }

    // write n_tokens into ring buffer from captured hidden states
    // write n_tokens from the capture buffer into the ring, starting at
    // src_offset in the capture buffer. wraps circularly in the ring.
    // source determines whether data comes from CPU callback, verify GPU, or
    // prefill GPU staging buffers.
    int ring_write(int n_tokens, int src_offset = 0, bool force_cpu_ring = false,
                   dflash_capture_source source = dflash_capture_source::cpu_hidden) {
        if (n_tokens <= 0) return 0;

        const bool use_prefill_gpu = (source == dflash_capture_source::prefill_gpu_hidden);
        // For prefill GPU staging, the CPU callback was not installed so
        // llama_get_n_layer_hiddens() returns 0. Use n_target_layers instead
        // so the D2D loop iterates over all target layers.
        const int32_t n_src_layers = use_prefill_gpu
            ? n_target_layers
            : llama_get_n_layer_hiddens(ctx_tgt);
        int actual_written = n_tokens;
        bool first_layer = true;
        bool any_layer = false;
        for (int layer = 0; layer < n_target_layers && layer < n_src_layers; ++layer) {
            float * data = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);

            // When using prefill GPU staging, the CPU callback was not installed
            // so data/ntok are 0. Skip the CPU sizing loop; actual_written is set
            // by the caller's n_tokens argument (the span from should_flush_dflash_prefill).
            if (use_prefill_gpu && !data) {
                any_layer = true;
                continue;
            }

            if ((!data && !gpu_ring_handle) || ntok <= 0) continue;

            int to_write = std::min(n_tokens, std::max(0, (int)ntok - src_offset));

            if (to_write < n_tokens) {
                LOG_WRN("DFLASH_DBG ring_write MISMATCH: requested=%d actual=%d ntok=%lld src_offset=%d ring_write_pos=%d ring_filled=%d committed_len=%d\n",
                    n_tokens, to_write, (long long)ntok, src_offset, ring_write_pos, ring_filled, committed_len);
            }
            if (first_layer) {
                actual_written = to_write;
                first_layer = false;
            } else {
                actual_written = std::min(actual_written, to_write);
            }
            any_layer = true;
        }
        if (!any_layer || actual_written <= 0) return 0;

        bool gpu_upload_queued = false;
        bool gpu_d2d_failed = false;
        int64_t cpu_copy_us = 0;
        int64_t gpu_enqueue_us = 0;
        for (int layer = 0; layer < n_target_layers && layer < n_src_layers; ++layer) {
            float * data = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t embd = llama_get_layer_hidden_n_embd(ctx_tgt, layer);

            // When using prefill GPU staging, skip per-layer CPU checks and
            // go straight to D2D upload for each layer.
            if (use_prefill_gpu && !data) {
                embd = n_embd;
            } else {
                int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);
                if ((!data && !gpu_ring_handle) || ntok <= 0) continue;
            }

            // CPU ring write: only when we have CPU data and either forcing CPU
            // ring or no GPU ring is available. In prefill GPU mode, we have no
            // CPU data, so the CPU ring is not kept in sync with this write.
            if ((force_cpu_ring || !gpu_ring_handle) && data) {
                const int64_t t_start = profile ? ggml_time_us() : 0;
                for (int t = 0; t < actual_written; ++t) {
                    int slot = (ring_write_pos + t) % RING_SIZE;
                    memcpy(ring_buf[layer].data() + (size_t)slot * embd,
                           data + (size_t)(src_offset + t) * embd,
                           embd * sizeof(float));
                }
                if (profile) {
                    cpu_copy_us += ggml_time_us() - t_start;
                }
            }

            // GPU ring upload
            if (gpu_ring_handle) {
                const auto plan = common_dflash_ring_write_plan(cross_ctx, ring_write_pos, actual_written);
                if (plan.n_tokens > 0) {
                    const int64_t t_start = profile ? ggml_time_us() : 0;
                    bool used_d2d = false;
                    if (!data) {
                        // No CPU hidden data — try GPU buffer D2D paths.
                        if (use_prefill_gpu) {
                            // Prefill staging: try prefill D2D first, then verify fallback.
                            used_d2d = llama_dflash_prefill_gpu_write_hidden(gpu_ring_handle, ctx_tgt, seq_id, layer, plan.ring_pos,
                                src_offset + plan.src_token_offset, plan.n_tokens, embd);
                            if (!used_d2d) {
                                used_d2d = llama_dflash_cross_ring_gpu_write_hidden(gpu_ring_handle, ctx_tgt, layer, plan.ring_pos,
                                    src_offset + plan.src_token_offset, plan.n_tokens, embd);
                            }
                        } else {
                            // Verify: try verify hidden D2D first, then prefill fallback.
                            used_d2d = llama_dflash_cross_ring_gpu_write_hidden(gpu_ring_handle, ctx_tgt, layer, plan.ring_pos,
                                src_offset + plan.src_token_offset, plan.n_tokens, embd);
                            if (!used_d2d) {
                                used_d2d = llama_dflash_prefill_gpu_write_hidden(gpu_ring_handle, ctx_tgt, seq_id, layer, plan.ring_pos,
                                    src_offset + plan.src_token_offset, plan.n_tokens, embd);
                            }
                        }
                        gpu_d2d_failed = gpu_d2d_failed || !used_d2d;
                    }
                    if (!used_d2d && data) {
                        llama_dflash_cross_ring_gpu_write(gpu_ring_handle, layer, plan.ring_pos,
                            data + (size_t)(src_offset + plan.src_token_offset) * embd,
                            plan.n_tokens, embd);
                    }
                    if (profile) {
                        gpu_enqueue_us += ggml_time_us() - t_start;
                    }
                    gpu_upload_queued = true;
                }
            }
        }
        if (gpu_d2d_failed) {
            LOG_WRN("dflash: GPU hidden D2D ring write failed; accepted hiddens were not appended\n");
            return 0;
        }
        int64_t gpu_sync_us = 0;
        if (gpu_upload_queued) {
            // Hidden capture storage is reused on the next target decode; keep
            // async H2D uploads from reading freed or overwritten vectors.
            const int64_t t_start = profile ? ggml_time_us() : 0;
            llama_dflash_cross_ring_gpu_synchronize(gpu_ring_handle);
            if (profile) {
                gpu_sync_us += ggml_time_us() - t_start;
            }
        }
        log_ring_profile("ring_write", n_tokens, actual_written, cpu_copy_us, gpu_enqueue_us, gpu_sync_us);
        ring_write_pos = (ring_write_pos + actual_written) % RING_SIZE;
        ring_filled = std::min(ring_filled + actual_written, RING_SIZE);
        return actual_written;
    }

    // tree variant: write specific capture-buffer indices to the ring.
    // indices may be non-contiguous (e.g. when tree walk skips siblings).
    int ring_write_by_indices(const std::vector<int> & indices) {
        const int n_tokens = (int)indices.size();
        if (n_tokens <= 0) return 0;

        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        int actual_written = n_tokens;
        bool first_layer = true;
        bool any_layer = false;
        for (int layer = 0; layer < n_target_layers && layer < n_slots; ++layer) {
            float * data = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);
            if (!data || ntok <= 0) continue;

            int wrote_this_layer = 0;
            for (int t = 0; t < n_tokens; ++t) {
                int src_idx = indices[t];
                if (src_idx < 0 || src_idx >= (int)ntok) {
                    break;
                }
                wrote_this_layer++;
            }
            if (wrote_this_layer < n_tokens) {
                LOG_WRN("DFLASH_DBG ring_write_by_indices MISMATCH: requested=%d prefix=%d ntok=%lld ring_write_pos=%d ring_filled=%d committed_len=%d\n",
                    n_tokens, wrote_this_layer, (long long)ntok, ring_write_pos, ring_filled, committed_len);
            }
            if (first_layer) {
                actual_written = wrote_this_layer;
                first_layer = false;
            } else {
                actual_written = std::min(actual_written, wrote_this_layer);
            }
            any_layer = true;
        }

        if (!any_layer || actual_written <= 0) return 0;

        bool gpu_upload_queued = false;
        int64_t cpu_copy_us = 0;
        int64_t gpu_enqueue_us = 0;
        for (int layer = 0; layer < n_target_layers && layer < n_slots; ++layer) {
            float * data = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t embd = llama_get_layer_hidden_n_embd(ctx_tgt, layer);
            int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);
            if (!data || ntok <= 0) continue;

            for (int t = 0; t < actual_written; ++t) {
                int src_idx = indices[t];
                int ring_slot = (ring_write_pos + t) % RING_SIZE;
                const int64_t t_cpu_start = profile ? ggml_time_us() : 0;
                memcpy(ring_buf[layer].data() + (size_t)ring_slot * embd,
                       data + (size_t)src_idx * embd,
                       embd * sizeof(float));
                if (profile) {
                    cpu_copy_us += ggml_time_us() - t_cpu_start;
                }
                if (gpu_ring_handle) {
                    int gpu_pos = (ring_write_pos + t) % cross_ctx;
                    const int64_t t_gpu_start = profile ? ggml_time_us() : 0;
                    llama_dflash_cross_ring_gpu_write(gpu_ring_handle, layer, gpu_pos,
                        data + (size_t)src_idx * embd, 1, embd);
                    if (profile) {
                        gpu_enqueue_us += ggml_time_us() - t_gpu_start;
                    }
                    gpu_upload_queued = true;
                }
            }
        }
        int64_t gpu_sync_us = 0;
        if (gpu_upload_queued) {
            // Hidden capture storage is reused on the next target decode; keep
            // async H2D uploads from reading freed or overwritten vectors.
            const int64_t t_start = profile ? ggml_time_us() : 0;
            llama_dflash_cross_ring_gpu_synchronize(gpu_ring_handle);
            if (profile) {
                gpu_sync_us += ggml_time_us() - t_start;
            }
        }
        log_ring_profile("ring_write_by_indices", n_tokens, actual_written, cpu_copy_us, gpu_enqueue_us, gpu_sync_us);
        ring_write_pos = (ring_write_pos + actual_written) % RING_SIZE;
        ring_filled = std::min(ring_filled + actual_written, RING_SIZE);
        return actual_written;
    }

    // called after initial prefill — grab all hidden states
    void capture_target_hiddens() {
        llama_dflash_set_active_slot(ctx_tgt, seq_id);

        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0) return;

        int64_t n_tokens = llama_get_layer_hidden_n_tokens(ctx_tgt, 0);
        if (n_tokens <= 0) return;

        LOG_DBG("DFLASH_DBG capture_target_hiddens: n_tokens=%lld ring_write_pos=%d ring_filled=%d committed_len=%d\n",
            (long long)n_tokens, ring_write_pos, ring_filled, committed_len);

        // only keep last RING_SIZE tokens if prompt exceeds ring capacity
        int start_offset = std::max(0, (int)n_tokens - RING_SIZE);
        int to_store = (int)n_tokens - start_offset;

        ring_write_pos = 0;
        ring_filled = 0;
        llama_dflash_kv_cache_reset(ctx_dft);
        const int actual_written = ring_write(to_store, start_offset, true);
        committed_len = start_offset + actual_written;
        update_drafter_kv_cache(actual_written);
    }

    // called after each verification decode — append only the accepted tokens' hidden states
    void append_target_hiddens(int n_accepted) {
        llama_dflash_set_active_slot(ctx_tgt, seq_id);

        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0 || n_accepted <= 0) {
            return;
        }

        // DEBUG: log capture buffer vs n_accepted to detect counter mismatch
        {
            int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, 0);
            int64_t embd = llama_get_layer_hidden_n_embd(ctx_tgt, 0);
            const float * data = llama_get_layer_hidden(ctx_tgt, 0);
            float first5[5] = {}, last5[5] = {};
            if (data && ntok > 0 && embd >= 5) {
                memcpy(first5, data, 5 * sizeof(float));
                memcpy(last5, data + (ntok - 1) * embd + embd - 5, 5 * sizeof(float));
            }
            LOG_DBG("DFLASH_DBG append_target_hiddens: n_accepted=%d ntok=%lld embd=%lld ring_write_pos=%d ring_filled=%d committed_len=%d layer0_first=[%.4f,%.4f,%.4f,%.4f,%.4f] last=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                n_accepted, (long long)ntok, (long long)embd, ring_write_pos, ring_filled, committed_len,
                first5[0], first5[1], first5[2], first5[3], first5[4],
                last5[0], last5[1], last5[2], last5[3], last5[4]);
        }

        const int actual_written = ring_write(n_accepted);
        committed_len += actual_written;
        update_drafter_kv_cache(actual_written);
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.n_min;
    }
};

struct common_speculative {
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states

    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        case COMMON_SPECULATIVE_TYPE_SUFFIX:        return "suffix";
        case COMMON_SPECULATIVE_TYPE_COPYSPEC:      return "copyspec";
        case COMMON_SPECULATIVE_TYPE_RECYCLE:       return "recycle";
        case COMMON_SPECULATIVE_TYPE_DFLASH:        return "dflash";
        default:                                    return "unknown";
    }
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

// initialization of the speculative decoding system
//
llama_context * common_speculative_create_ctx_dft(const common_params_speculative & params, int dflash_n_slots) {
    if (!params.model_dft) {
        return nullptr;
    }
    llama_context_params cparams_dft = params.cparams_dft;
    cparams_dft.dflash_n_slots = dflash_n_slots;
    cparams_dft.dflash_cross_ctx = params.dflash_cross_ctx;
    llama_context * ctx_dft = llama_init_from_model(params.model_dft, cparams_dft);
    if (ctx_dft == nullptr) {
        LOG_ERR("%s", "failed to create draft context\n");
        return nullptr;
    }
    // Set topk/temp BEFORE first decode (graph reservation caches topology on K=1 shape,
    // but we need to invalidate so the actual decode rebuilds with correct K)
    if (params.draft_topk > 1) {
        llama_set_dflash_topk(ctx_dft, params.draft_topk);
        LOG_INF("dflash: top-K=%d enabled for tree branching\n", params.draft_topk);
    }
    if (params.sample_temp > 0.0f) {
        llama_set_dflash_sample_temp(ctx_dft, params.sample_temp);
    }

    // warmup the draft context with an empty run so the first real decode isn't slow
    {
        const llama_vocab * vocab_dft = llama_model_get_vocab(llama_get_model(ctx_dft));

        llama_token bos = llama_vocab_bos(vocab_dft);
        llama_token eos = llama_vocab_eos(vocab_dft);

        llama_token tmp[2];
        int n_tmp = 0;
        if (bos != LLAMA_TOKEN_NULL) { tmp[n_tmp++] = bos; }
        if (eos != LLAMA_TOKEN_NULL) { tmp[n_tmp++] = eos; }
        if (n_tmp == 0) { tmp[n_tmp++] = 0; }

        llama_set_warmup(ctx_dft, true);
        int ret = llama_decode(ctx_dft, llama_batch_get_one(tmp, n_tmp));
        if (ret != 0) {
            LOG_WRN("%s: draft warmup decode failed: %d (non-fatal)\n", __func__, ret);
        }

        llama_memory_t mem_dft = llama_get_memory(ctx_dft);
        if (mem_dft) {
            llama_memory_clear(mem_dft, true);
        }
        llama_synchronize(ctx_dft);
        llama_perf_context_reset(ctx_dft);
        llama_set_warmup(ctx_dft, false);

        LOG_INF("%s: draft model warmup complete\n", __func__);
    }

    return ctx_dft;
}

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        llama_context             * ctx_dft_shared) {
    // When ctx_dft_shared is provided, use it non-owning so multiple
    // common_speculative instances can share one drafter context.
    const bool owns_ctx_dft = (ctx_dft_shared == nullptr);
    llama_context * ctx_dft = ctx_dft_shared;
    if (ctx_dft == nullptr && params.model_dft) {
        ctx_dft = common_speculative_create_ctx_dft(params);
    }
    if (ctx_dft == nullptr && params.draft.model) {
        ctx_dft = llama_init_from_model(params.draft.model, params.draft.cparams);
        if (ctx_dft == nullptr) {
            return nullptr;
        }
    }

    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        bool has_draft = !params.draft.mparams.path.empty();
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3

        bool has_ngram_cache   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        bool has_ngram_simple  = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        bool has_ngram_map_k   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        bool has_ngram_map_k4v = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        bool has_ngram_mod     = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        bool has_suffix        = (params.type == COMMON_SPECULATIVE_TYPE_SUFFIX);
        bool has_copyspec      = (params.type == COMMON_SPECULATIVE_TYPE_COPYSPEC);
        bool has_recycle       = (params.type == COMMON_SPECULATIVE_TYPE_RECYCLE);
        bool has_dflash        = (params.type == COMMON_SPECULATIVE_TYPE_DFLASH);

        // DFlash uses --model-draft but is NOT a standard draft model
        if (has_dflash) {
            has_draft = false;
        }

        // In a more complex implementation we could use the same implementation but with different parameters.
        // This was initially used in PR-18471 but removed to simplify the code.
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            auto & sparams = params.ngram_mod;

            if (!sparams.obj) {
                sparams.obj = std::make_shared<common_ngram_mod>(sparams.n_match, 4*1024*1024);

                LOG_INF("%s: initialized ngram_mod with n_match=%d, size=%zu (%.3f MB)\n", __func__,
                        sparams.n_match, sparams.obj->size(), (float)(sparams.obj->size_bytes())/1024/1024);

                if (sparams.n_match < 16) {
                    LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                            "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, sparams.n_match);
                }
            }

            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_copyspec) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_COPYSPEC, params));
        }
        if (has_recycle) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_RECYCLE, params));
        }
        if (has_suffix) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_SUFFIX, params));
        }
        if (has_dflash) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DFLASH, params));
        }
        if (has_draft) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DFLASH: {
                GGML_ASSERT(ctx_dft != nullptr);
                // topk and temp already set on ctx_dft above (before impls loop)
                impls.push_back(std::make_unique<common_speculative_state_dflash>(
                    ctx_tgt,
                    ctx_dft,
                    params.model_dft,
                    owns_ctx_dft,
                    params.dflash_cross_ctx
                ));
                if (owns_ctx_dft) {
                    ctx_dft = nullptr; // ownership transferred to the state
                }
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                const bool use_ckpt = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.draft.replacements,
                    /* .use_ckpt     = */ use_ckpt
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.type));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type  = */ config.type,
                    /* .state = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    get_common_ngram_map(config.type, config.params.ngram_map_k)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod.obj);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod.obj));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(params.ngram_cache.lookup_cache_static, params.ngram_cache.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_SUFFIX: {
                impls.push_back(std::make_unique<common_speculative_state_suffix>(
                    config.type,
                    config.params.suffix_max_depth,
                    config.params.n_max,
                    config.params.suffix_spec_factor,
                    config.params.suffix_spec_offset,
                    config.params.suffix_min_prob
                ));
                LOG_INF("%s: suffix tree speculative decoding (max_depth=%d, factor=%.1f, min_prob=%.2f)\n",
                    __func__, config.params.suffix_max_depth,
                    config.params.suffix_spec_factor, config.params.suffix_min_prob);
                break;
            }
            case COMMON_SPECULATIVE_TYPE_COPYSPEC: {
                impls.push_back(std::make_unique<common_speculative_state_copyspec>(
                    config.type,
                    config.params.copyspec_gamma
                ));
                LOG_INF("%s: copyspec speculative decoding (gamma=%d)\n",
                    __func__, config.params.copyspec_gamma);
                break;
            }
            case COMMON_SPECULATIVE_TYPE_RECYCLE: {
                impls.push_back(std::make_unique<common_speculative_state_recycle>(
                    config.type,
                    config.params.recycle_k
                ));
                LOG_INF("%s: token recycling speculative decoding (k=%d)\n",
                    __func__, config.params.recycle_k);
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    // if a model-based drafter exists, tell CopySpec to only fire as primary for long matches
    bool has_model_impl = false;
    for (auto & impl : impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH ||
            impl->type == COMMON_SPECULATIVE_TYPE_DRAFT ||
            impl->type == COMMON_SPECULATIVE_TYPE_EAGLE3) {
            has_model_impl = true;
            break;
        }
    }
    if (has_model_impl) {
        for (auto & impl : impls) {
            if (impl->type == COMMON_SPECULATIVE_TYPE_COPYSPEC) {
                static_cast<common_speculative_state_copyspec *>(impl.get())->has_model_drafter = true;
            }
        }
    }

    auto * result = new common_speculative {
        /* .impls     = */ std::move(impls),
        /* .curr_impl = */ nullptr,
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt);
        impl->n_call_begin++;
    }
}

void common_speculative_set_seq_id(common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->set_seq_id(seq_id);
    }
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last,
        std::vector<float> * draft_log_probs) {
    llama_tokens result;

    spec->curr_impl = nullptr; // reset current implementation

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(params, prompt_tgt, id_last, result, draft_log_probs);
            impl->n_call_draft++;
        }

        {
            const int n_min = impl->n_min(params);

            if (!result.empty() && (int) result.size() < n_min) {
                LOG_DBG("%s: ignoring small draft: %d < %d\n", __func__, (int) result.size(), n_min);
                result.clear();
            }
        }

        if (!result.empty()) {
            LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                    common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                    impl.get()->n_call_draft, result.size());

            spec->curr_impl = impl.get(); // set current implementation for stats
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();

            break; // we have a draft, so break out of the loop and return it.
        }
    }

    // try extension impls (e.g. CopySpec appending suffix matches after DFlash draft)
    if (!result.empty()) {
        for (auto & impl : spec->impls) {
            const size_t pre = result.size();
            impl->extend(params, prompt_tgt, id_last, result);
            if (result.size() > pre) {
                impl->n_ext_calls++;
                impl->n_ext_tokens += result.size() - pre;
                LOG_DBG("%s: extended draft by %zu tokens (%s)\n", __func__,
                    result.size() - pre, common_speculative_type_to_str(impl->type).c_str());
            }
        }
    }

    return result;
}

// Batched DFlash draft: prepare cross data for all specs,
// build one combined multi-seq batch, decode once, distribute results.
void common_speculative_draft_batch(
        std::vector<common_speculative *> & specs,
        llama_context                     * ctx_dft,
        const common_params_speculative   & params,
        const std::vector<llama_token>    & id_last_per_spec,
        std::vector<llama_tokens>         & result_per_spec,
        std::vector<std::vector<float>>   * log_probs_per_spec) {
    const int n_specs = (int) specs.size();
    result_per_spec.clear();
    result_per_spec.resize(n_specs);
    if (log_probs_per_spec) {
        log_probs_per_spec->clear();
        log_probs_per_spec->resize(n_specs);
    }

    if (n_specs == 0 || !ctx_dft) {
        return;
    }

    const llama_model * model_dft  = llama_get_model(ctx_dft);
    const int block_size           = llama_model_dflash_block_size(model_dft);
    const int n_draft              = std::min(block_size - 1, params.n_max);
    const int batch_len            = n_draft + 1;
    const llama_token mask_tok     = (llama_token) llama_model_dflash_mask_token_id(model_dft);

    const int64_t t0 = ggml_time_us();

    // Phase 1: prepare cross-attention data per spec
    struct ready_slot {
        common_speculative_state * impl;
        int           cross_len;
        llama_seq_id  seq_id;
        int           spec_idx;
    };
    std::vector<ready_slot> ready;
    ready.reserve(n_specs);

    for (int s = 0; s < n_specs; s++) {
        for (auto & impl : specs[s]->impls) {
            if (impl->type != COMMON_SPECULATIVE_TYPE_DFLASH) {
                continue;
            }

            const int cross_len = impl->prepare_batch_draft(ctx_dft);
            if (cross_len < 0) {
                break;
            }

            auto * dfl = static_cast<common_speculative_state_dflash *>(impl.get());
            ready.push_back({ impl.get(), cross_len, dfl->seq_id, s });
            break;
        }
    }

    if (ready.empty()) {
        return;
    }

    const int n_ready = (int) ready.size();

    // Phase 2: set drafter graph width
    llama_set_dflash_n_slots(ctx_dft, n_ready);

    const int64_t t1 = ggml_time_us();

    // Phase 3: build combined batch — each spec's tokens tagged with its seq_id
    llama_batch batch = llama_batch_init(n_ready * batch_len, 0, 1);

    for (const auto & rs : ready) {
        common_batch_add(batch, id_last_per_spec[rs.spec_idx], rs.cross_len, { rs.seq_id }, true);
        for (int i = 1; i < batch_len; i++) {
            common_batch_add(batch, mask_tok, rs.cross_len + i, { rs.seq_id }, true);
        }
    }

    // Phase 4: single decode for all specs
    const int ret = llama_decode(ctx_dft, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        LOG_ERR("dflash batch: decode failed with %d\n", ret);
        return;
    }

    const int64_t t2 = ggml_time_us();

    // read per-spec argmax results
    int32_t * argmax  = llama_get_logits_argmax(ctx_dft);
    float   * argmax_probs = llama_get_logits_argmax_probs(ctx_dft);
    const int K_flat  = llama_get_logits_argmax_k(ctx_dft);

    for (int r = 0; r < n_ready; r++) {
        auto & rs     = ready[r];
        auto & result = result_per_spec[rs.spec_idx];
        std::vector<float> * log_probs = log_probs_per_spec ? &(*log_probs_per_spec)[rs.spec_idx] : nullptr;
        const int offset = r * batch_len;

        if (argmax) {
            for (int i = 1; i < batch_len && (int) result.size() < n_draft; i++) {
                if (argmax_probs && params.p_min > 0.0f && i > 1) {
                    float log_prob = argmax_probs[(offset + i) * K_flat];
                    if (log_prob < logf(params.p_min)) {
                        break;
                    }
                }
                result.push_back((llama_token) argmax[(offset + i) * K_flat]);
                if (log_probs && argmax_probs) {
                    log_probs->push_back(argmax_probs[(offset + i) * K_flat]);
                }
            }
        } else {
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
            for (int i = 1; i < batch_len && (int) result.size() < n_draft; i++) {
                float * logits = llama_get_logits_ith(ctx_dft, offset + i);
                if (!logits) {
                    break;
                }
                llama_token best = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
                result.push_back(best);
            }
        }

        // update stats + adaptive draft tracking
        rs.impl->n_call_draft++;
        auto * dfl = static_cast<common_speculative_state_dflash *>(rs.impl);
        dfl->n_draft_last = (int) result.size();
        if (!result.empty()) {
            rs.impl->n_gen_drafts++;
            rs.impl->n_gen_tokens += result.size();
            specs[rs.spec_idx]->curr_impl = rs.impl;
        }
    }

    const int64_t t3 = ggml_time_us();

    LOG_DBG("dflash batch draft (%d specs): prepare=%.1fms decode=%.1fms argmax=%.1fms total=%.1fms\n",
            n_ready, (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t3 - t0) / 1e3);
}

common_speculative_tree common_speculative_draft_tree(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt_tgt,
        llama_token id_last,
        int tree_budget) {
    common_speculative_tree tree;

    spec->curr_impl = nullptr;

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft_tree(params, prompt_tgt, id_last, tree_budget, tree);
            impl->n_call_draft++;
        }

        if (tree.n_nodes > 0) {
            spec->curr_impl = impl.get();
            impl->n_gen_drafts++;
            impl->n_gen_tokens += tree.n_nodes;
            break;
        }
    }

    return tree;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (n_accepted == 0) {
        return;
    }

    common_speculative_state * impl = spec->curr_impl;

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(n_accepted);
        impl->n_call_accept++;
    }
}

void common_speculative_update_logits(common_speculative * spec, llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->update_logits(ctx, batch_tokens, n_accepted);
    }
}

void common_speculative_update_logits_by_indices(common_speculative * spec, llama_context * ctx, const std::vector<int> & capture_indices) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->update_logits_by_indices(ctx, capture_indices);
    }
}

int common_speculative_flush_prefill(common_speculative * spec, int src_offset, int n_tokens) {
    if (spec == nullptr) {
        return 0;
    }
    int total_written = 0;
    for (auto & impl : spec->impls) {
        total_written += impl->flush_prefill(src_offset, n_tokens);
    }
    return total_written;
}

void common_speculative_set_prefill_capture_enabled(common_speculative * spec, bool enabled) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->set_prefill_capture_enabled(enabled);
    }
}

size_t common_speculative_ring_state_size(const common_speculative * spec) {
    if (spec == nullptr) return 0;
    size_t total = 0;
    for (auto & impl : spec->impls) {
        total += impl->ring_state_size();
    }
    return total;
}

void common_speculative_ring_state_save(const common_speculative * spec, uint8_t * buf, size_t size) {
    if (spec == nullptr) return;
    for (auto & impl : spec->impls) {
        size_t impl_size = impl->ring_state_size();
        if (impl_size > 0 && impl_size <= size) {
            impl->ring_state_save(buf, impl_size);
            buf += impl_size;
            size -= impl_size;
        }
    }
}

bool common_speculative_ring_state_load(common_speculative * spec, const uint8_t * buf, size_t size) {
    if (spec == nullptr) return false;
    for (auto & impl : spec->impls) {
        if (impl->ring_state_load(buf, size)) {
            return true;
        }
    }
    return false;
}

int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }

    int32_t n_max = 0;
    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, impl->n_max(params));
    }

    return n_max;
}

int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }

    int32_t n_min = 0;
    for (const auto & impl : spec->impls) {
        n_min = std::max(n_min, impl->n_min(params));
    }

    return n_min;
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_ext;
        if (impl->n_ext_calls > 0) {
            str_ext = ", ext calls = " + std::to_string(impl->n_ext_calls) +
                      ", ext tokens = " + std::to_string(impl->n_ext_tokens);
        }

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str(),
                str_ext.c_str());
    }
}
