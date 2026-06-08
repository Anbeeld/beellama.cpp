
#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-adaptive-dm.h"
#include "server-loop-guard.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "fit.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "src/dflash-profile.h"
#include "src/llama-ext.h"
#include "src/llama-memory.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cinttypes>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <filesystem>
#include <random>
#include <cmath>
#include <set>
#include <utility>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

server_dflash_recurrent_rollback_plan server_context_dflash_recurrent_rollback_plan(
        const common_params_speculative & speculative,
        bool target_recurrent_or_hybrid) {
    server_dflash_recurrent_rollback_plan plan;

    if (speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH || !target_recurrent_or_hybrid) {
        return plan;
    }

    plan.needs_backup_sequences = true;
    plan.needs_attention_backup_streams = speculative.branch_budget > 0;
    return plan;
}

static bool dflash_server_profile_enabled(uint32_t flags) {
    return dflash_profile_enabled(flags);
}

static bool dflash_server_crash_trace_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_CRASH_TRACE");
        return env && std::atoi(env) != 0;
    }();
    return enabled;
}

static bool dflash_verify_padding_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_DFLASH_VERIFY_PAD");
        return env && env[0] != '\0' && strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool dflash_shared_drafter_batch_disabled() {
    static const bool disabled = [] {
        const char * env = getenv("GGML_DFLASH_SHARED_DRAFT_BATCH");
        return env && env[0] != '\0' && strcmp(env, "0") == 0;
    }();
    return disabled;
}

struct server_model_arch_probe {
    bool known = false;
    bool recurrent_or_hybrid = false;
    std::string name;
};

static bool server_arch_name_is_recurrent_or_hybrid(const std::string & name) {
    static const std::set<std::string> recurrent_or_hybrid = {
        "mamba",
        "mamba2",
        "rwkv6",
        "rwkv6qwen2",
        "rwkv7",
        "arwkv7",
        "jamba",
        "falcon-h1",
        "plamo2",
        "granitehybrid",
        "lfm2",
        "lfm2moe",
        "nemotron_h",
        "nemotron_h_moe",
        "qwen3next",
        "kimi-linear",
        "qwen35",
        "qwen35moe",
    };

    return recurrent_or_hybrid.find(name) != recurrent_or_hybrid.end();
}

static server_model_arch_probe server_probe_model_arch(const std::string & path_model) {
    server_model_arch_probe result;

    if (path_model.empty()) {
        return result;
    }

    ggml_context * ctx_meta = nullptr;
    gguf_init_params gguf_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &ctx_meta,
    };

    gguf_context * ctx_gguf = gguf_init_from_file(path_model.c_str(), gguf_params);
    if (!ctx_gguf) {
        if (ctx_meta) {
            ggml_free(ctx_meta);
        }
        return result;
    }

    const int64_t key = gguf_find_key(ctx_gguf, "general.architecture");
    if (key >= 0 && gguf_get_kv_type(ctx_gguf, key) == GGUF_TYPE_STRING) {
        const char * arch_name = gguf_get_val_str(ctx_gguf, key);
        if (arch_name) {
            result.known = true;
            result.name = arch_name;
            result.recurrent_or_hybrid = server_arch_name_is_recurrent_or_hybrid(result.name);
        }
    }

    gguf_free(ctx_gguf);
    if (ctx_meta) {
        ggml_free(ctx_meta);
    }
    return result;
}

static bool server_model_is_dflash_drafter(const llama_model * model) {
    return model &&
        llama_model_dflash_block_size(model) > 1 &&
        llama_model_dflash_n_target_layers(model) > 0 &&
        llama_model_dflash_n_target_features(model) > 0;
}

static bool server_backend_dev_type_is_gpu(enum ggml_backend_dev_type type) {
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

static bool server_backend_dev_is_gpu(ggml_backend_dev_t dev) {
    return dev && server_backend_dev_type_is_gpu(ggml_backend_dev_type(dev));
}

static bool server_backend_dev_is_dflash_shared_output_compatible(ggml_backend_dev_t dev) {
    if (!dev) {
        return false;
    }

    const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
    return server_backend_dev_type_is_gpu(type) || type == GGML_BACKEND_DEVICE_TYPE_META;
}

static const char * dflash_backend_extended_argmax_block_reason(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    const char * name = reg ? ggml_backend_reg_name(reg) : nullptr;
    if (name && strcmp(name, "MTL") == 0) {
        return "metal-argmax-ext";
    }
    return "argmax-ext-backend";
}

static bool dflash_backend_dev_supports_extended_argmax(ggml_backend_dev_t dev) {
    if (!dev) {
        return false;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return name &&
        (strcmp(name, "CUDA") == 0 ||
         strcmp(name, "ROCm") == 0 ||
         strcmp(name, "MUSA") == 0);
}

static bool dflash_target_reduced_verify_supported(const llama_context * ctx, const char ** reason) {
    const llama_model * model = ctx ? llama_get_model(ctx) : nullptr;
    if (!model) {
        if (reason) {
            *reason = "no-target-model";
        }
        return false;
    }

    ggml_backend_dev_t output_dev = llama_model_dev_output(model);
    if (dflash_backend_dev_supports_extended_argmax(output_dev)) {
        return true;
    }

    if (output_dev && ggml_backend_dev_type(output_dev) == GGML_BACKEND_DEVICE_TYPE_META) {
        const int32_t n_devices = llama_model_n_devices(model);
        bool found_execution_device = false;
        for (int32_t i = 0; i < n_devices; ++i) {
            ggml_backend_dev_t dev = llama_model_get_device(model, i);
            if (!server_backend_dev_is_gpu(dev)) {
                continue;
            }
            found_execution_device = true;
            if (!dflash_backend_dev_supports_extended_argmax(dev)) {
                if (reason) {
                    *reason = dflash_backend_extended_argmax_block_reason(dev);
                }
                return false;
            }
        }
        if (found_execution_device) {
            return true;
        }
    }

    if (reason) {
        *reason = dflash_backend_extended_argmax_block_reason(output_dev);
    }
    return false;
}

static bool server_model_supports_device_buffer(const llama_model * model, ggml_backend_dev_t dev) {
    if (!model || !dev) {
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    if (!buft) {
        return false;
    }

    const int32_t n_devices = llama_model_n_devices(model);
    for (int32_t i = 0; i < n_devices; ++i) {
        ggml_backend_dev_t model_dev = llama_model_get_device(model, i);
        if (model_dev && ggml_backend_dev_supports_buft(model_dev, buft)) {
            return true;
        }
    }
    return false;
}

static ggml_backend_dev_t server_single_gpu_device_from_list(const std::vector<ggml_backend_dev_t> & devices) {
    if (devices.size() != 2 || devices[0] == nullptr || devices[1] != nullptr) {
        return nullptr;
    }

    if (!server_backend_dev_is_gpu(devices[0])) {
        return nullptr;
    }

    return devices[0];
}

static bool server_dflash_single_explicit_draft_device(const common_params & params, ggml_backend_dev_t & dev) {
    dev = nullptr;
    if (params.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH || !params.speculative.has_dft()) {
        return false;
    }

    dev = server_single_gpu_device_from_list(params.speculative.draft.devices);
    return dev != nullptr;
}

static bool server_tail_pos_is_in_code_fence(
        const std::string & text,
        size_t              pos) {
    bool in_fence = false;
    size_t search = 0;
    while (search < pos) {
        const size_t fence = text.find("```", search);
        if (fence == std::string::npos || fence >= pos) {
            break;
        }
        in_fence = !in_fence;
        search = fence + 3;
    }
    return in_fence;
}

static bool server_tail_tool_marker_has_boundary(
        const std::string & text,
        size_t              pos) {
    if (pos == 0) {
        return true;
    }

    const char prev = text[pos - 1];
    return prev == '\n' || prev == '\r' || prev == '\t' || prev == ' ' || prev == '>';
}

static bool server_tail_has_tool_call_marker(const std::string & text, size_t scan_from) {
    static const char * markers[] = {
        "<tool_call",
        "</tool_call",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    };

    const size_t tail_pos = text.size() > 256 ? text.size() - 256 : 0;
    const size_t start = std::max(scan_from, tail_pos);
    for (const char * marker : markers) {
        size_t pos = text.find(marker, start);
        while (pos != std::string::npos) {
            if (server_tail_tool_marker_has_boundary(text, pos) &&
                    !server_tail_pos_is_in_code_fence(text, pos)) {
                return true;
            }
            pos = text.find(marker, pos + 1);
        }
    }
    return false;
}

struct log_norm_cache {
    std::vector<float> cache;
    float inv_temp;
    int32_t n_vocab;
    struct llama_context * ctx;

    log_norm_cache(struct llama_context * ctx_, float temp, int32_t max_batch_idx = -1)
        : cache(max_batch_idx >= 0 ? (size_t) max_batch_idx + 1 : 0, -INFINITY)
        , inv_temp(temp > 0.0f ? 1.0f / temp : 1.0f)
        , n_vocab(llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_))))
        , ctx(ctx_) {}

    float get(int32_t batch_idx) {
        if (batch_idx < 0) {
            return -FLT_MAX;
        }
        if (batch_idx < (int32_t) cache.size() && !std::isinf(cache[batch_idx])) {
            return cache[batch_idx];
        }
        const float * logits = llama_get_logits_ith(ctx, batch_idx);
        if (!logits) {
            if (batch_idx >= (int32_t) cache.size()) {
                cache.resize(batch_idx + 1, -INFINITY);
            }
            cache[batch_idx] = -FLT_MAX;
            return -FLT_MAX;
        }
        float mx = -FLT_MAX;
        double se = 0.0;
        for (int32_t i = 0; i < n_vocab; i++) {
            float s = logits[i] * inv_temp;
            if (s > mx) {
                se = se * expf(mx - s) + 1.0;
                mx = s;
            } else {
                se += expf(s - mx);
            }
        }
        float ln = mx + (float) log(se);
        if (batch_idx >= (int32_t) cache.size()) {
            cache.resize(batch_idx + 1, -INFINITY);
        }
        cache[batch_idx] = ln;
        return ln;
    }
};

static std::vector<llama_token> speculative_reject_sample(
        struct common_sampler  * smpl,
        struct llama_context   * ctx_tgt,
        const llama_tokens     & draft,
        const std::vector<int> & idxs,
        const std::vector<float> & draft_log_probs,
        float                    temp,
        std::mt19937           & rng,
        int32_t                & n_exact,
        int32_t                & n_prob_accept,
        int32_t                & n_reject,
        int32_t                & n_no_prob,
        const common_sampler_accept_callback & on_accept = {}) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(draft.size() + 1);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    const float inv_temp = (temp > 0.0f) ? (1.0f / temp) : 1.0f;
    const int32_t max_batch_idx = idxs.empty() ? -1 :
        *std::max_element(idxs.begin(), idxs.end());
    log_norm_cache ln_cache(ctx_tgt, temp, max_batch_idx);
    const bool grammar_active_at_start = common_sampler_has_active_grammar(smpl);
    bool stopped_by_callback = false;
    auto accept = [&](llama_token id) {
        if (on_accept) {
            const auto info = common_sampler_accept_with_info(smpl, id, true);
            result.push_back(id);
            return on_accept(info);
        }

        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        return true;
    };

    for (size_t i = 0; i < draft.size(); i++) {
        const llama_token target_token = common_sampler_sample(smpl, ctx_tgt, idxs[i]);

        if (target_token == draft[i]) {
            n_exact++;
            if (!accept(target_token)) {
                stopped_by_callback = true;
                break;
            }
            if (common_sampler_stops_speculative_accept(smpl, grammar_active_at_start)) {
                break;
            }
            continue;
        }

        if (i >= draft_log_probs.size()) {
            n_no_prob++;
            (void) accept(target_token);
            break;
        }

        const float log_norm = ln_cache.get(idxs[i]);
        const float log_prob = (log_norm > -FLT_MAX) ? llama_get_logits_ith(ctx_tgt, idxs[i])[draft[i]] * inv_temp - log_norm : -FLT_MAX;
        const float target_prob = (log_prob > -FLT_MAX) ? expf(log_prob) : 0.0f;

        const float q_log = draft_log_probs[i];
        float accept_prob = (target_prob > 0.0f && std::isfinite(q_log) && q_log <= 0.0f) ? expf(log_prob - q_log) : 0.0f;
        if (accept_prob > 1.0f) {
            accept_prob = 1.0f;
        }

        if (uniform(rng) < accept_prob) {
            n_prob_accept++;
            if (!accept(draft[i])) {
                stopped_by_callback = true;
                break;
            }
            if (common_sampler_stops_speculative_accept(smpl, grammar_active_at_start)) {
                break;
            }
        } else {
            n_reject++;
            (void) accept(target_token);
            break;
        }
    }

    if (!stopped_by_callback && result.size() == draft.size() && !common_sampler_stops_speculative_accept(smpl, grammar_active_at_start)) {
        const llama_token bonus = common_sampler_sample(smpl, ctx_tgt, idxs[draft.size()]);
        (void) accept(bonus);
    }

    return result;
}

static bool speculative_flat_result_has_bonus(
        const llama_tokens          & ids,
        const llama_tokens          & draft,
        const struct common_sampler * smpl) {
    if (ids.empty()) {
        return false;
    }

    if (!common_sampler_blocks_speculative(smpl) || ids.size() > draft.size()) {
        return true;
    }

    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != draft[i]) {
            return true;
        }
    }

    return false;
}

struct dflash_reduced_verify_plan {
    bool enabled = false;
    int top_k = 1;
    const char * reason = "disabled";
};

static bool dflash_reduced_sampler_chain_supported(
        const common_params_sampling & sampling,
        bool                           stochastic,
        const char                  ** reason) {
    auto reject = [&](const char * why) {
        if (reason) {
            *reason = why;
        }
        return false;
    };

    bool saw_top_k = false;
    for (const auto sampler : sampling.samplers) {
        switch (sampler) {
            case COMMON_SAMPLER_TYPE_NONE:
                break;
            case COMMON_SAMPLER_TYPE_PENALTIES:
                if (!(sampling.penalty_repeat == 1.0f && sampling.penalty_freq == 0.0f && sampling.penalty_present == 0.0f)) {
                    return reject("penalties");
                }
                break;
            case COMMON_SAMPLER_TYPE_DRY:
                if (sampling.dry_multiplier != 0.0f && sampling.dry_penalty_last_n != 0) {
                    return reject("dry");
                }
                break;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                if (sampling.top_n_sigma >= 0.0f) {
                    return reject("top-n-sigma");
                }
                break;
            case COMMON_SAMPLER_TYPE_TOP_K:
                saw_top_k = true;
                break;
            case COMMON_SAMPLER_TYPE_TYPICAL_P:
                if (sampling.typ_p < 1.0f) {
                    if (!stochastic) {
                        return reject("typical");
                    }
                    if (!saw_top_k) {
                        return reject("sampler-order");
                    }
                }
                break;
            case COMMON_SAMPLER_TYPE_TOP_P:
                if (stochastic && sampling.top_p < 1.0f && !saw_top_k) {
                    return reject("sampler-order");
                }
                break;
            case COMMON_SAMPLER_TYPE_MIN_P:
                if (stochastic && sampling.min_p > 0.0f && !saw_top_k) {
                    return reject("sampler-order");
                }
                break;
            case COMMON_SAMPLER_TYPE_XTC:
                if (sampling.xtc_probability > 0.0f) {
                    return reject("xtc");
                }
                break;
            case COMMON_SAMPLER_TYPE_TEMPERATURE:
                break;
            case COMMON_SAMPLER_TYPE_INFILL:
                return reject("infill");
            case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                return reject("adaptive");
        }
    }

    if (stochastic && !saw_top_k) {
        return reject("top-k-sampler");
    }

    return true;
}

static dflash_reduced_verify_plan dflash_select_reduced_verify_plan(
        const common_params_sampling    & sampling,
        const common_params_speculative & speculative,
        bool                              use_rejection,
        bool                              has_tree) {
    dflash_reduced_verify_plan plan;

    if (has_tree) {
        plan.reason = "tree";
        return plan;
    }
    if (use_rejection) {
        plan.reason = "rejection";
        return plan;
    }
    if (sampling.n_probs > 0) {
        plan.reason = "prob-reporting";
        return plan;
    }
    if (!sampling.grammar.empty() && !sampling.grammar_lazy) {
        plan.reason = "grammar";
        return plan;
    }
    if (sampling.has_logit_bias() || sampling.ignore_eos) {
        plan.reason = "logit-bias";
        return plan;
    }
    if (!(sampling.penalty_repeat == 1.0f && sampling.penalty_freq == 0.0f && sampling.penalty_present == 0.0f)) {
        plan.reason = "penalties";
        return plan;
    }
    if (sampling.dry_multiplier != 0.0f && sampling.dry_penalty_last_n != 0) {
        plan.reason = "dry";
        return plan;
    }
    if (sampling.top_n_sigma >= 0.0f) {
        plan.reason = "top-n-sigma";
        return plan;
    }
    if (sampling.xtc_probability > 0.0f) {
        plan.reason = "xtc";
        return plan;
    }
    if (sampling.mirostat != 0) {
        plan.reason = "mirostat";
        return plan;
    }
    if (sampling.adaptive_target >= 0.0f) {
        plan.reason = "adaptive";
        return plan;
    }
    if (sampling.dynatemp_range > 0.0f) {
        plan.reason = "dynamic-temp";
        return plan;
    }
    if (sampling.reasoning_budget_tokens >= 0) {
        plan.reason = "finite-reasoning-budget";
        return plan;
    }
    if (speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH) {
        plan.reason = "not-dflash";
        return plan;
    }

    const bool stochastic = sampling.temp > 0.0f;
    const char * sampler_reason = nullptr;
    if (!dflash_reduced_sampler_chain_supported(sampling, stochastic, &sampler_reason)) {
        plan.reason = sampler_reason;
        return plan;
    }

    if (sampling.temp <= 0.0f) {
        plan.enabled = true;
        plan.top_k = 1;
        plan.reason = "greedy";
        return plan;
    }

    if (sampling.top_k <= 0) {
        plan.reason = "unbounded-top-k";
        return plan;
    }
    if (sampling.top_k > 64) {
        plan.reason = "top-k-too-large";
        return plan;
    }

    plan.enabled = true;
    plan.top_k = std::max(1, sampling.top_k);
    plan.reason = "top-k";
    return plan;
}

static std::vector<llama_token> dflash_sample_reduced_verify(
        struct common_sampler * smpl,
        struct llama_context  * ctx,
        const std::vector<int> & idxs,
        const llama_tokens    & draft,
        int                     top_k,
        const common_sampler_accept_callback & on_accept = {}) {
    if ((int) idxs.size() != (int) draft.size() + 1 || top_k <= 0) {
        return {};
    }
    if (llama_get_logits_argmax_k(ctx) != top_k ||
            llama_get_logits_argmax_n(ctx) < (int32_t) idxs.size()) {
        return {};
    }

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = model ? llama_model_get_vocab(model) : nullptr;
    const int32_t n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;
    if (n_vocab <= 0) {
        return {};
    }

    std::vector<llama_token> candidate_ids((size_t) idxs.size() * (size_t) top_k);
    std::vector<float> candidate_logits((size_t) idxs.size() * (size_t) top_k);

    for (size_t row = 0; row < idxs.size(); ++row) {
        const int32_t * ids = llama_get_logits_argmax_ith(ctx, idxs[row]);
        const float * logits = llama_get_logits_argmax_probs_ith(ctx, idxs[row]);
        if (!ids || !logits) {
            return {};
        }
        for (int k = 0; k < top_k; ++k) {
            const int32_t token = ids[k];
            if (token < 0 || token >= n_vocab) {
                return {};
            }
            candidate_ids[row * (size_t) top_k + (size_t) k] = (llama_token) token;
            candidate_logits[row * (size_t) top_k + (size_t) k] = logits[k];
        }
    }

    return common_sampler_sample_reduced_and_accept_n(
        smpl, candidate_ids.data(), candidate_logits.data(), (int32_t) idxs.size(), top_k, draft, on_accept);
}

static bool server_reasoning_budget_state_is_reasoning(common_reasoning_budget_state state) {
    return state == REASONING_BUDGET_COUNTING ||
           state == REASONING_BUDGET_WAITING_UTF8 ||
           state == REASONING_BUDGET_FORCING;
}

static bool server_accept_info_is_reasoning(const common_sampler_accept_info & info) {
    return server_reasoning_budget_state_is_reasoning(info.reasoning_state_before) ||
           server_reasoning_budget_state_is_reasoning(info.reasoning_state_after);
}

static std::string server_loop_guard_reason_to_string(const server_loop_guard_result & result) {
    std::string reason = result.kind.empty() ? "unknown" : result.kind;
    if (result.period > 0) {
        reason += string_format(" period=%d", result.period);
    }
    if (result.coverage > 0) {
        reason += string_format(" coverage=%d", result.coverage);
    }
    if (result.score > 0.0f) {
        reason += string_format(" score=%.3f", (double) result.score);
    }
    return reason;
}

constexpr int HTTP_POLLING_SECONDS = 1;

static uint32_t server_n_outputs_max(const common_params & params) {
    const uint32_t n_batch  = params.n_batch;

    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return n_batch;
    }

    const auto spec_n_max = [](int32_t n) -> uint32_t {
        return (uint32_t) std::max(0, n);
    };
    uint32_t n_outputs_per_seq = 1 + spec_n_max(common_speculative_n_max(&params.speculative));

    const auto dflash_outputs_per_seq = [&]() -> uint32_t {
        const uint32_t n_max_base = spec_n_max(params.speculative.n_max_base > 0 ?
                params.speculative.n_max_base : params.speculative.n_max);
        uint32_t n_nodes = spec_n_max(params.speculative.n_max);

        if (params.speculative.branch_budget > 0 && n_max_base > 0) {
            const uint32_t topk = (uint32_t) std::max(1, params.speculative.draft_topk);
            const uint32_t tree_budget = std::min<uint32_t>(
                    n_max_base + spec_n_max(params.speculative.branch_budget),
                    n_max_base * topk);
            n_nodes = std::max(n_nodes, tree_budget);
        }

        return 1 + n_nodes;
    };

    if (params.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH)) {
        n_outputs_per_seq = std::max<uint32_t>(n_outputs_per_seq, dflash_outputs_per_seq());
    }

    // Draft model present without an explicit speculative type: init can auto-enable
    // draft-simple, so reserve room for the verify batch. Bee can also auto-detect
    // DFlash after the target context is already created, which needs the DFlash cap.
    if (params.speculative.has_dft()) {
        n_outputs_per_seq = std::max<uint32_t>(n_outputs_per_seq, 1 + spec_n_max(params.speculative.draft.n_max));
        if (params.speculative.type() == COMMON_SPECULATIVE_TYPE_NONE) {
            n_outputs_per_seq = std::max<uint32_t>(n_outputs_per_seq, dflash_outputs_per_seq());
        }
    }

    const uint64_t n_outputs = (uint64_t) params.n_parallel * n_outputs_per_seq;

    return std::max<uint32_t>(1, std::min<uint64_t>(n_batch, n_outputs));
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

struct server_slot : server_adaptive_dm_state {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    // multimodal
    mtmd_context * mctx = nullptr;

    // speculative decoding
    common_speculative_ptr spec;
    common_speculative * spec_shared = nullptr; // non-owning

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    std::vector<int32_t> spec_pad_i_batch;
    common_prompt_checkpoint spec_ckpt;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;
    std::string stop_detail;
    server_loop_guard loop_guard;
    int32_t loop_guard_interventions = 0;
    bool loop_guard_triggered = false;
    std::string loop_guard_action;
    std::string loop_guard_reason;
    bool reasoning_tool_marker_logged = false;
    int32_t reasoning_output_tokens = 0;
    int32_t visible_output_tokens = 0;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    void prompt_save(server_prompt_cache & prompt_cache) const {
        GGML_ASSERT(prompt.data.size() == 0);

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_WRN(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear(bool allow_processing) {
        if (!allow_processing) {
            GGML_ASSERT(!is_processing());
        }

        SLT_INF(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        common_context_seq_rm(ctx_tgt, id, -1, -1);
        if (ctx_dft) {
            common_context_seq_rm(ctx_dft, id, -1, -1);
        }

        prompt.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token
    llama_tokens drafted;
    common_speculative_tree draft_tree;
    bool has_draft_tree = false;
    std::vector<float> draft_log_probs;
    std::mt19937 reject_rng;
    int32_t n_reject_exact = 0;
    int32_t n_reject_prob_accept = 0;
    int32_t n_reject_reject = 0;
    int32_t n_reject_no_prob = 0;

    // stats
    size_t n_sent_text = 0; // number of sent text character

    int64_t t_print_last = 0;
    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing = 0.0; // ms
    double t_token_generation = 0.0;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted
    int64_t dflash_cycle_count = 0;
    int64_t dflash_requested_total = 0;
    int64_t dflash_produced_total = 0;
    int64_t dflash_accepted_total = 0;
    std::array<int64_t, 5> dflash_accept_hist = {};
    std::array<std::array<int64_t, 5>, 5> dflash_accept_hist_by_ctx = {};

    // Hybrid model: recurrent state backup for speculative decoding
    bool has_draft_backup = false;
    bool has_recurrent_only_backup = false;
    llama_seq_id seq_id_backup = -1;
    int  n_tokens_before_draft = 0; // prompt token count before draft tokens were added
    llama_pos n_pos_before_draft = 0; // KV position before draft tokens (accounts for mmproj position expansion)

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens_cache = 0;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stop_detail    = "";
        loop_guard.reset();
        loop_guard_interventions = 0;
        loop_guard_triggered = false;
        loop_guard_action = "";
        loop_guard_reason = "";
        reasoning_tool_marker_logged = false;
        reasoning_output_tokens = 0;
        visible_output_tokens = 0;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_pad_i_batch.clear();
            spec_ckpt.clear();
        }
        drafted.clear();
        draft_tree = common_speculative_tree();
        has_draft_tree = false;
        draft_log_probs.clear();
        n_reject_exact = 0;
        n_reject_prob_accept = 0;
        n_reject_reject = 0;
        n_reject_no_prob = 0;
        spec_i_batch.clear();
        spec_pad_i_batch.clear();
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        // Clear per-request speculative decoding stats. Adaptive controller
        // state is reset or preserved when the next task selects this slot,
        // based on prompt continuity.
        n_draft_total = 0;
        n_draft_accepted = 0;
        dflash_cycle_count = 0;
        dflash_requested_total = 0;
        dflash_produced_total = 0;
        dflash_accepted_total = 0;
        dflash_accept_hist.fill(0);
        for (auto & bucket : dflash_accept_hist_by_ctx) {
            bucket.fill(0);
        }
        has_draft_backup = false;
        has_recurrent_only_backup = false;
        seq_id_backup = -1;
        n_tokens_before_draft = 0;
        n_pos_before_draft = 0;

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd() || (get_spec() && common_speculative_need_embd(get_spec()));
    }

    bool need_embd_nextn() const {
        GGML_ASSERT(task);
        return get_spec() && common_speculative_need_embd_nextn(get_spec());
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return spec || spec_shared;
    }

    common_speculative * get_spec() const {
        return spec ? spec.get() : spec_shared;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max(
            const common_params & global_params,
            bool advance_adaptive_probe = false,
            int dflash_cohort_n_max = -1) {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        if (global_params.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
            if (common_sampler_reasoning_is_forcing(smpl.get())) {
                return 0;
            }
        } else if (common_sampler_blocks_speculative(smpl.get())) {
            return 0;
        }

        const int n_draft_min = common_speculative_n_min(get_spec(), task->params.speculative);

        const int configured_base_n_max = common_speculative_n_max(get_spec(), task->params.speculative);
        const bool use_profit_controller = dm_adaptive && dm_controller == COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT;
        const int base_n_max = use_profit_controller && profit_has_key && profit_key.base_n_max > 0
            ? std::min(configured_base_n_max, (int) profit_key.base_n_max)
            : configured_base_n_max;
        const int probe_n_max = server_adaptive_dm_probe_n_max(base_n_max, dm_probe_fraction);
        int n_draft_max = (dm_adaptive && adaptive_n_max >= 0) ? std::min((int) adaptive_n_max, base_n_max) : base_n_max;
        const bool use_dflash_cohort_n_max =
            global_params.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
            global_params.speculative.branch_budget == 0 &&
            dflash_cohort_n_max > 0;

        if (use_profit_controller) {
            if (profit_baseline_probe_pending && use_dflash_cohort_n_max) {
                n_draft_max = dflash_cohort_n_max;
            } else if (profit_baseline_probe_pending) {
                return 0;
            } else if (profit_should_probe_baseline() && use_dflash_cohort_n_max) {
                n_draft_max = dflash_cohort_n_max;
            } else if (profit_should_probe_baseline()) {
                if (advance_adaptive_probe) {
                    profit_mark_baseline_probe();
                }
                return 0;
            } else if (adaptive_n_max < 0) {
                if (advance_adaptive_probe) {
                    adaptive_n_max = 0;
                }
                if (!use_dflash_cohort_n_max) {
                    return 0;
                }
                n_draft_max = dflash_cohort_n_max;
            } else if (adaptive_n_max == 0) {
                if (!profit_baseline_ready()) {
                    if (!use_dflash_cohort_n_max) {
                        return 0;
                    }
                    n_draft_max = dflash_cohort_n_max;
                } else {
                    const bool probe_now = adaptive_probe_counter + 1 >= profit_off_probe_interval();
                    if (!probe_now) {
                        if (advance_adaptive_probe) {
                            adaptive_probe_counter++;
                        }
                        if (!use_dflash_cohort_n_max) {
                            return 0;
                        }
                        n_draft_max = dflash_cohort_n_max;
                    } else {
                        const int profit_probe_n_max = [&]() {
                            return profit_next_off_probe_depth(base_n_max, probe_n_max);
                        }();
                        if (advance_adaptive_probe) {
                            adaptive_probe_counter = 0;
                            adaptive_n_max = profit_probe_n_max;
                        }
                        n_draft_max = profit_probe_n_max;
                    }
                }
            } else {
                n_draft_max = std::min((int) adaptive_n_max, base_n_max);
                if (advance_adaptive_probe) {
                    explore_counter++;
                    if (explore_counter % dm_explore_interval == 0) {
                        const int explore_n_max = profit_next_unready_explore_depth(
                                adaptive_n_max,
                                base_n_max,
                                explore_counter / dm_explore_interval);
                        if (explore_n_max > 0) {
                            n_draft_max = explore_n_max;
                        }
                    }
                }
            }
        } else if (dm_adaptive && adaptive_n_max < 0) {
            if (advance_adaptive_probe) {
                adaptive_n_max = probe_n_max;
            }
            n_draft_max = probe_n_max;
        } else if (dm_adaptive && adaptive_n_max == 0) {
            // off state: probe periodically to check if fringe has recovered
            const bool probe_now = adaptive_probe_counter + 1 >= dm_probe_interval;
            if (!probe_now) {
                if (advance_adaptive_probe) {
                    adaptive_probe_counter++;
                }
                return 0;
            }
            if (advance_adaptive_probe) {
                adaptive_probe_counter = 0;
                adaptive_n_max = probe_n_max;
            }
            n_draft_max = probe_n_max;
        } else if (dm_adaptive && adaptive_n_max > 0 && adaptive_n_max < base_n_max) {
            // active but not at full draft: periodically draft at base_n_max to collect
            // data beyond current n_max (exploration)
            if (advance_adaptive_probe) {
                explore_counter++;
                if (explore_counter % dm_explore_interval == 0) {
                    n_draft_max = base_n_max;
                }
            }
        }

        if (use_dflash_cohort_n_max && n_draft_max > 0) {
            n_draft_max = std::min(n_draft_max, dflash_cohort_n_max);
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        n_draft_max = std::min(n_draft_max, n_ctx - prompt.n_tokens() - 2);

        const int32_t remaining = remaining_generation_budget(global_params);
        if (remaining > 0) {
            n_draft_max = std::min(n_draft_max, remaining - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        if (n_draft_max < n_draft_min) {
            SLT_DBG(*this, "the max possible draft is too small: %d < %d - skipping speculative decoding\n", n_draft_max, n_draft_min);
            n_draft_max = 0;
        }

        return n_draft_max;
    }

    int32_t effective_n_predict(const common_params & global_params) const {
        GGML_ASSERT(task);

        if (task->params.n_predict != -1) {
            return task->params.n_predict;
        }
        return global_params.n_predict;
    }

    int32_t remaining_generation_budget(const common_params & global_params) const {
        const int32_t limit = effective_n_predict(global_params);
        if (limit == -1) {
            return -1;
        }
        return limit - n_decoded;
    }

    void update_batch(llama_batch & batch) {
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.n_tokens;

            common_batch_add(batch, sampled, prompt.tokens.pos_next(), { this->id }, true);

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.n_tokens);
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.n_tokens + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            common_batch_add(batch, sampled, pos0++, { this->id }, true);
            for (auto token : spec_draft) {
                common_batch_add(batch, token, pos0++, { this->id }, true);
            }
        }

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // clean up speculative backup sequence to avoid orphaned KV cells
            if (has_draft_backup && seq_id_backup >= 0) {
                llama_memory_t mem = llama_get_memory(ctx_tgt);
                if (has_recurrent_only_backup) {
                    llama_memory_seq_rm_recurrent(mem, seq_id_backup, -1, -1);
                } else {
                    llama_memory_seq_rm(mem, seq_id_backup, -1, -1);
                }
            }

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear(false);
            }

            reset();

            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    static int dflash_accept_bin(int n_accepted) {
        if (n_accepted <= 0) {
            return 0;
        }
        if (n_accepted <= 3) {
            return n_accepted;
        }
        return 4;
    }

    static int dflash_context_bucket(llama_pos pos) {
        if (pos < 1000) {
            return 0;
        }
        if (pos < 2000) {
            return 1;
        }
        if (pos < 4000) {
            return 2;
        }
        if (pos < 8000) {
            return 3;
        }
        return 4;
    }

    void note_dflash_cycle(int requested, int produced, int accepted, llama_pos pos) {
        dflash_cycle_count++;
        dflash_requested_total += std::max(0, requested);
        dflash_produced_total += std::max(0, produced);
        dflash_accepted_total += std::max(0, accepted);
        const int accept_bin = dflash_accept_bin(accepted);
        const int ctx_bucket = dflash_context_bucket(pos);
        dflash_accept_hist[(size_t) accept_bin]++;
        dflash_accept_hist_by_ctx[(size_t) ctx_bucket][(size_t) accept_bin]++;
    }

    void print_timings_tg() {
        if (n_decoded < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        t_print_last = t_now;

        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this, "n_decoded = %6d, tg = %6.2f t/s\n", n_decoded, n_gen_second);
    }

    void print_timings_pp() const {
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;
        const double f_progress = (float) prompt.n_tokens() / task->n_tokens();

        if (t_prompt_processing < 3000.0) {
            return;
        }

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                n_prompt_tokens_processed, f_progress, t_prompt_processing / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_token_generation, n_decoded, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        if (n_draft_total > 0) {
            const float draft_ratio = (float) n_draft_accepted / n_draft_total;
            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated)\n",
                    draft_ratio, n_draft_accepted, n_draft_total);
            if (adaptive_n_max >= 0) {
                SLT_INF(*this, "adaptive dm: fringe=%.2f n_max=%d\n", rolling_fringe, adaptive_n_max);
            }
            if (dflash_cycle_count > 0 && dflash_server_profile_enabled(DFLASH_PROFILE_SUMMARY)) {
                SLT_INF(*this,
                        "dflash acceptance histogram: cycles=%" PRId64
                        " requested=%" PRId64 " produced=%" PRId64 " accepted=%" PRId64
                        " bins[0,1,2,3,4+]=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                        dflash_cycle_count,
                        dflash_requested_total,
                        dflash_produced_total,
                        dflash_accepted_total,
                        dflash_accept_hist[0],
                        dflash_accept_hist[1],
                        dflash_accept_hist[2],
                        dflash_accept_hist[3],
                        dflash_accept_hist[4]);
                SLT_INF(*this,
                        "dflash acceptance by ctx: <1k=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] "
                        "1-2k=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] "
                        "2-4k=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] "
                        "4-8k=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] "
                        "8k+=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                        dflash_accept_hist_by_ctx[0][0], dflash_accept_hist_by_ctx[0][1],
                        dflash_accept_hist_by_ctx[0][2], dflash_accept_hist_by_ctx[0][3],
                        dflash_accept_hist_by_ctx[0][4],
                        dflash_accept_hist_by_ctx[1][0], dflash_accept_hist_by_ctx[1][1],
                        dflash_accept_hist_by_ctx[1][2], dflash_accept_hist_by_ctx[1][3],
                        dflash_accept_hist_by_ctx[1][4],
                        dflash_accept_hist_by_ctx[2][0], dflash_accept_hist_by_ctx[2][1],
                        dflash_accept_hist_by_ctx[2][2], dflash_accept_hist_by_ctx[2][3],
                        dflash_accept_hist_by_ctx[2][4],
                        dflash_accept_hist_by_ctx[3][0], dflash_accept_hist_by_ctx[3][1],
                        dflash_accept_hist_by_ctx[3][2], dflash_accept_hist_by_ctx[3][3],
                        dflash_accept_hist_by_ctx[3][4],
                        dflash_accept_hist_by_ctx[4][0], dflash_accept_hist_by_ctx[4][1],
                        dflash_accept_hist_by_ctx[4][2], dflash_accept_hist_by_ctx[4][3],
                        dflash_accept_hist_by_ctx[4][4]);
            }
            const int32_t n_reject_tested = n_reject_exact + n_reject_prob_accept + n_reject_reject + n_reject_no_prob;
            if (n_reject_tested > 0) {
                SLT_INF(*this, "rejection sampling: exact=%d prob_accept=%d reject=%d no_prob=%d\n",
                        n_reject_exact, n_reject_prob_accept, n_reject_reject, n_reject_no_prob);
            }
        }

        common_speculative_print_stats(get_spec());
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = n_prompt_tokens_processed;
            res["n_prompt_tokens_cache"]     = n_prompt_tokens_cache;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        common_context_seq_rm(ctx_tgt, other.id,     -1, -1);
        common_context_seq_cp(ctx_tgt, id, other.id, -1, -1);

        if (ctx_dft) {
            common_context_seq_rm(ctx_dft, other.id,     -1, -1);
            common_context_seq_cp(ctx_dft, id, other.id, -1, -1);
        }

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

static bool dflash_slot_in_view(const server_slot & slot, const llama_batch & view) {
    for (int32_t j = 0; j < view.n_tokens; ++j) {
        for (int32_t k = 0; k < view.n_seq_id[j]; ++k) {
            if (view.seq_id[j][k] == slot.id) {
                return true;
            }
        }
    }

    return false;
}

static bool dflash_view_has_unexpected_prompt_logits(
        const llama_batch              & view,
        const std::vector<server_slot> & slots) {
    if (!view.logits) {
        return false;
    }

    for (const auto & slot : slots) {
        if (!slot.task || !slot.task->need_sampling()) {
            continue;
        }
        if (slot.state != SLOT_STATE_PROCESSING_PROMPT && slot.state != SLOT_STATE_DONE_PROMPT) {
            continue;
        }
        if (!dflash_slot_in_view(slot, view)) {
            continue;
        }

        const int32_t final_prompt_pos = slot.task->n_tokens() - 1;
        for (int32_t j = 0; j < view.n_tokens; ++j) {
            if (!view.logits[j]) {
                continue;
            }

            bool belongs_to_slot = false;
            for (int32_t k = 0; k < view.n_seq_id[j]; ++k) {
                if (view.seq_id[j][k] == slot.id) {
                    belongs_to_slot = true;
                    break;
                }
            }
            if (belongs_to_slot && view.pos[j] != final_prompt_pos) {
                return true;
            }
        }
    }

    return false;
}

static void dflash_log_reduced_verify_decision(
        bool        graph_enabled,
        bool        view_enabled,
        int32_t     view_start,
        int32_t     n_tokens,
        int         top_k,
        const char * reason) {
    if (!dflash_server_profile_enabled(DFLASH_PROFILE_VERIFY)) {
        return;
    }

    SRV_INF("dflash reduced verifier decision: graph_enabled=%d view_enabled=%d view_start=%d n_tokens=%d top_k=%d reason=%s\n",
            graph_enabled ? 1 : 0, view_enabled ? 1 : 0, view_start, n_tokens, top_k, reason ? reason : "unknown");
}

static bool dflash_batch_view_is_reduced_verify(
        const std::vector<server_slot> & slots,
        const common_params_sampling   & fallback_sampling,
        const common_params_speculative & speculative,
        bool                             use_rejection,
        bool                             has_tree,
        int32_t                          view_start,
        int32_t                          view_n_tokens,
        int                              top_k,
        const char                    ** reason) {
    auto reject = [&](const char * why) {
        if (reason) {
            *reason = why;
        }
        return false;
    };

    if (view_n_tokens <= 0 || top_k <= 0) {
        return reject("empty-view");
    }

    std::vector<uint8_t> covered((size_t) view_n_tokens, 0);
    int covered_count = 0;
    int expected_rows_per_slot = -1;

    for (const auto & slot : slots) {
        if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.has_draft_tree || slot.spec_draft.empty()) {
            continue;
        }
        // Reasoning-budget state is token/control-flow state, not a full-vocab
        // logits transform by itself. The sampler capability check below is the
        // source of truth for whether reduced verifier candidates are safe.
        if (!common_sampler_supports_reduced(slot.smpl.get())) {
            return reject("sampler");
        }
        const common_params_sampling & slot_sampling = slot.task ? slot.task->params.sampling : fallback_sampling;
        const dflash_reduced_verify_plan slot_plan =
            dflash_select_reduced_verify_plan(slot_sampling, speculative, use_rejection, has_tree);
        if (!slot_plan.enabled) {
            return reject(slot_plan.reason);
        }
        if (slot_plan.top_k != top_k) {
            return reject("top-k-mismatch");
        }
        if (slot.spec_i_batch.size() != slot.spec_draft.size() + 1) {
            return reject("spec-index-count");
        }

        auto cover_index = [&](int idx) {
            if (idx < view_start || idx >= view_start + view_n_tokens) {
                return false;
            }
            const int rel = idx - view_start;
            if (covered[(size_t) rel] != 0) {
                return false;
            }
            covered[(size_t) rel] = 1;
            covered_count++;
            return true;
        };

        int slot_rows = 0;
        for (int idx : slot.spec_i_batch) {
            if (!cover_index(idx)) {
                return reject(idx < view_start || idx >= view_start + view_n_tokens
                        ? "spec-index-outside-view" : "duplicate-index");
            }
            slot_rows++;
        }
        for (int idx : slot.spec_pad_i_batch) {
            if (!cover_index(idx)) {
                return reject(idx < view_start || idx >= view_start + view_n_tokens
                        ? "pad-index-outside-view" : "duplicate-index");
            }
            slot_rows++;
        }

        if (slot_rows > 0) {
            if (expected_rows_per_slot < 0) {
                expected_rows_per_slot = slot_rows;
            } else if (slot_rows != expected_rows_per_slot) {
                return reject("reduced-row-count-mismatch");
            }
        }
    }

    if (covered_count != view_n_tokens ||
            !std::all_of(covered.begin(), covered.end(), [](uint8_t v) { return v != 0; })) {
        return reject("coverage");
    }

    if (reason) {
        *reason = "eligible";
    }
    return true;
}

static dflash_reduced_verify_plan dflash_select_batch_reduced_verify_plan(
        const std::vector<server_slot> & slots,
        const common_params_sampling   & fallback_sampling,
        const common_params_speculative & speculative,
        bool                             use_rejection,
        bool                             has_tree) {
    dflash_reduced_verify_plan selected;
    bool found = false;

    for (const auto & slot : slots) {
        if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.has_draft_tree || slot.spec_draft.empty()) {
            continue;
        }

        const common_params_sampling & slot_sampling = slot.task ? slot.task->params.sampling : fallback_sampling;
        const dflash_reduced_verify_plan slot_plan =
            dflash_select_reduced_verify_plan(slot_sampling, speculative, use_rejection, has_tree);

        if (!slot_plan.enabled) {
            continue;
        }

        if (!found) {
            selected = slot_plan;
            found = true;
            continue;
        }

        if (slot_plan.top_k != selected.top_k) {
            selected.enabled = false;
            selected.reason = "top-k-mismatch";
            return selected;
        }
    }

    if (!found) {
        selected.reason = "no-eligible-slot";
    }

    return selected;
}

static int dflash_flat_effective_draft_max(llama_context * ctx_dft, int n_draft_max) {
    if (!ctx_dft || n_draft_max <= 0) {
        return n_draft_max;
    }

    const llama_model * model_dft = llama_get_model(ctx_dft);
    const int block_size = model_dft ? llama_model_dflash_block_size(model_dft) : 0;
    if (block_size <= 1) {
        return n_draft_max;
    }

    // Flat DFlash batches reserve row 0 for the seed token, so a block with N
    // rows can produce at most N-1 draft tokens.
    return std::min(n_draft_max, block_size - 1);
}

static int dflash_effective_adaptive_base_n_max(
        llama_context * ctx_dft,
        const common_params_speculative & spec,
        int base_n_max) {
    if (spec.type() == COMMON_SPECULATIVE_TYPE_DFLASH && spec.branch_budget == 0) {
        return dflash_flat_effective_draft_max(ctx_dft, base_n_max);
    }
    return base_n_max;
}

static bool server_speculative_uses_fork_slot_impls(const common_params_speculative & speculative) {
    return speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH) ||
           speculative.has_type(COMMON_SPECULATIVE_TYPE_SUFFIX) ||
           speculative.has_type(COMMON_SPECULATIVE_TYPE_COPYSPEC) ||
           speculative.has_type(COMMON_SPECULATIVE_TYPE_RECYCLE);
}



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};


//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    // DFlash: one drafter context shared across all slots'
    // common_speculative states (non-owning refs). Must outlive all specs — the
    // destroy() order below (specs first, then this) enforces that; when destroy()
    // isn't called explicitly, member-destructor order (reverse-declaration) frees
    // specs via `slots` before this unique_ptr runs.
    llama_context_ptr ctx_dft_shared;

    llama_batch batch {};

    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    // hybrid/recurrent models need re-evaluation of accepted tokens after
    // rejecting draft tokens, because the recurrent state cannot be rolled back
    bool needs_reeval = false;
    int  n_parallel_user = 0;
    int  n_seq_max_full = 0;      // visible slots plus optional recurrent backup cells
    bool recurrent_expanded = true; // false = backup cells deferred, expand before first draft
    bool recurrent_backup_sequences = false; // explicit backup seqs for speculative recurrent rollback
    bool recurrent_backup_attention_streams = false; // backup seqs also have attention KV streams

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;
    int slots_debug = 0;
    int n_empty_consecutive = 0;
    int dflash_next_tg_slot = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    json json_ui_settings = json::object();    // Primary: new name
    json json_webui_settings = json::object();    // Deprecated: use json_ui_settings instead (kept for compat)

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    // MTP↔mmproj GPU swap state
    bool mmproj_gpu_swap = false;
    bool mmproj_is_on_gpu = false;

    void destroy() {
        spec.reset();
        ctx_dft.reset();
        model_dft.reset();

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;

        for (server_slot & slot : slots) {
            if (slot.can_speculate()) {
                slot.spec.reset();
            }
        }

        ctx_dft_shared.reset();


        llama_batch_free(batch);
    }

    mtmd_context_params make_mmproj_params(bool use_gpu) const {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu          = use_gpu;
        mparams.print_timings    = false;
        mparams.n_threads        = params_base.cpuparams.n_threads;
        mparams.flash_attn_type  = params_base.flash_attn_type;
        mparams.warmup           = use_gpu; // only warmup when on GPU
        mparams.image_min_tokens = params_base.image_min_tokens;
        mparams.image_max_tokens = params_base.image_max_tokens;
        mparams.decoder_n_ubatch = ctx_tgt ? llama_n_ubatch(ctx_tgt) : params_base.n_ubatch;
        mparams.media_marker     = get_media_marker();
        return mparams;
    }

    void adjust_mmproj_decoder_batch_for_non_causal(const mtmd_decode_requirements & mmproj_decode_req) {
        if (!mmproj_decode_req.needs_non_causal_full_batch || mmproj_decode_req.min_decoder_batch_tokens <= 0) {
            return;
        }

        const int32_t old_n_batch  = params_base.n_batch;
        const int32_t old_n_ubatch = params_base.n_ubatch;

        params_base.n_batch = std::max(params_base.n_batch, mmproj_decode_req.min_decoder_batch_tokens);
        params_base.n_ubatch = std::max(params_base.n_ubatch, mmproj_decode_req.min_decoder_batch_tokens);

        if (params_base.n_batch != old_n_batch || params_base.n_ubatch != old_n_ubatch) {
            SRV_WRN("mmproj non-causal image decode requires full chunks; raised batch/ubatch from %d/%d to %d/%d\n",
                    old_n_batch, old_n_ubatch, params_base.n_batch, params_base.n_ubatch);
        }
    }

    void reload_mmproj(bool use_gpu) {
        mtmd_free(mctx);
        mctx = nullptr;
        auto mparams = make_mmproj_params(use_gpu);
        mctx = mtmd_init_from_file(params_base.mmproj.path.c_str(), model_tgt, mparams);
        for (server_slot & slot : slots) {
            slot.mctx = mctx;
        }
    }

    llama_context * create_mtp_context() {
        auto cparams = common_context_params_to_llama(params_base);
        cparams.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
        cparams.type_k        = params_base.speculative.draft.cache_type_k;
        cparams.type_v        = params_base.speculative.draft.cache_type_v;
        cparams.n_rs_seq      = 0;
        cparams.n_outputs_max = params_base.n_parallel;
        return llama_init_from_model(model_tgt, cparams);
    }

    server_dflash_recurrent_rollback_plan speculative_recurrent_rollback_plan() const {
        server_dflash_recurrent_rollback_plan plan;

        if (params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH) {
            return plan;
        }

        const server_model_arch_probe arch = server_probe_model_arch(params_base.model.path);
        if (!arch.known) {
            SRV_WRN("%s", "could not determine target architecture before DFlash context allocation; reserving recurrent-only backup cells conservatively\n");
            plan.needs_backup_sequences = true;
            return plan;
        }

        plan = server_context_dflash_recurrent_rollback_plan(
                params_base.speculative,
                arch.recurrent_or_hybrid);

        if (!arch.recurrent_or_hybrid) {
            SRV_INF("DFlash target architecture %s does not need recurrent rollback; keeping n_parallel=%d\n",
                    arch.name.c_str(), params_base.n_parallel);
        } else if (plan.needs_attention_backup_streams) {
            SRV_WRN("DFlash DDTree target architecture %s needs attention backup streams for branch rollback\n",
                    arch.name.c_str());
        } else if (plan.needs_backup_sequences) {
            SRV_INF("flat DFlash target architecture %s will use recurrent-only backup cells; keeping attention streams at n_parallel=%d\n",
                    arch.name.c_str(), params_base.n_parallel);
        }

        return plan;
    }

    void swap_mtp_to_mmproj_gpu() {
        SRV_INF("%s", "swapping MTP out, loading mmproj to GPU...\n");
        int64_t t0 = ggml_time_us();

        for (server_slot & slot : slots) {
            slot.spec.reset();
            slot.spec_shared = nullptr;
        }
        spec.reset();

        ctx_dft.reset();
        params_base.speculative.draft.ctx_dft = nullptr;

        reload_mmproj(true);
        if (!mctx) {
            SRV_ERR("%s", "failed to load mmproj to GPU, falling back to CPU\n");
            reload_mmproj(false);
        } else {
            mmproj_is_on_gpu = true;
        }

        SRV_INF("MTP→mmproj swap done in %" PRId64 " ms\n", (ggml_time_us() - t0) / 1000);
    }

    void swap_mmproj_to_mtp() {
        SRV_INF("%s", "unloading mmproj from GPU, recreating MTP...\n");
        int64_t t0 = ggml_time_us();

        mmproj_is_on_gpu = false;
        reload_mmproj(false);

        ctx_dft.reset(create_mtp_context());
        if (!ctx_dft) {
            SRV_ERR("%s", "failed to recreate MTP context after mmproj swap\n");
            return;
        }

        ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());
        if (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("MTP draft context can_seq_rm returned NO after mmproj swap — overriding to FULL, type=%d\n", (int) ctx_dft_seq_rm_type);
            ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        }
        params_base.speculative.draft.ctx_tgt = ctx_tgt;
        params_base.speculative.draft.ctx_dft = ctx_dft.get();

        try {
            spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
        } catch (const std::exception & e) {
            SRV_ERR("failed to reinit speculative context: %s\n", e.what());
        }

        for (server_slot & slot : slots) {
            slot.ctx_dft = ctx_dft.get();
            if (spec) {
                slot.spec_shared = spec.get();
                common_speculative_set_seq_id(slot.get_spec(), slot.id);
            }
        }

        SRV_INF("mmproj→MTP swap done in %" PRId64 " ms\n", (ggml_time_us() - t0) / 1000);
    }

    void slot_save_and_clear(server_slot & slot) {
        if (slot.prompt.n_tokens() == 0) {
            return;
        }
        SLT_INF(slot, "%s", "saving idle slot to prompt cache\n");
        SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
        slot.prompt_save(*prompt_cache);
        slot.prompt_clear(false);
        prompt_cache->update();
    }

    bool recurrent_shrink_for_prompt_cache(const char * reason) {
        if (!recurrent_backup_sequences || !recurrent_expanded || !needs_reeval || n_seq_max_full <= n_parallel_user) {
            return true;
        }

        for (const server_slot & slot : slots) {
            if (slot.is_processing() || slot.has_draft_backup) {
                SRV_DBG("not shrinking recurrent state for prompt cache (%s): slot %d processing=%d has_backup=%d\n",
                        reason, slot.id, slot.is_processing(), slot.has_draft_backup);
                return true;
            }
        }

        auto * mem = llama_get_memory(ctx_tgt);
        for (const server_slot & slot : slots) {
            const llama_seq_id seq_backup = slot.id + n_parallel_user;
            if (recurrent_backup_attention_streams) {
                llama_memory_seq_rm(mem, seq_backup, -1, -1);
            } else {
                llama_memory_seq_rm_recurrent(mem, seq_backup, -1, -1);
            }
        }

        if (llama_context_recurrent_shrink(ctx_tgt, n_parallel_user)) {
            recurrent_expanded = false;
            SRV_INF("shrunk recurrent state to %d cells for prompt cache (%s, removed %d backup cells)\n",
                    n_parallel_user, reason, n_seq_max_full - n_parallel_user);
            return true;
        } else {
            SRV_ERR("failed to shrink recurrent state to %d cells for prompt cache (%s)\n",
                    n_parallel_user, reason);
            return false;
        }
    }

    void recurrent_expand_after_prompt_cache(const char * reason) {
        if (!recurrent_backup_sequences || recurrent_expanded || !needs_reeval || n_seq_max_full <= n_parallel_user) {
            return;
        }

        if (llama_context_recurrent_expand(ctx_tgt, n_seq_max_full)) {
            recurrent_expanded = true;
            SRV_INF("expanded recurrent state to %d cells after prompt cache (%s)\n", n_seq_max_full, reason);
            return;
        }

        SRV_ERR("failed to expand recurrent state to %d cells after prompt cache (%s)\n", n_seq_max_full, reason);
        GGML_ABORT("failed to expand recurrent state after prompt cache restore; continuing would make scheduler reservation inconsistent\n");
    }

    bool prompt_cache_boundary_checkpoints_enabled(const server_slot & slot) const {
        if (params_base.n_ctx_checkpoints > 0 || !prompt_cache) {
            return false;
        }

        if (!slot.task ||
                slot.task->type != SERVER_TASK_TYPE_COMPLETION ||
                !slot.task->params.cache_prompt) {
            return false;
        }

        return ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
               ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
               n_swa > 0 ||
               llama_model_is_hybrid(model_tgt);
    }

    void erase_oldest_checkpoint(server_slot & slot, const char * reason) {
        const auto & cur = slot.prompt.checkpoints.front();

        SLT_WRN(slot,
                "erasing old context checkpoint (%s, pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                reason && reason[0] ? reason : "limit", cur.pos_min, cur.pos_max, cur.n_tokens,
                (float) cur.size() / 1024 / 1024);

        slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
    }

    bool dflash_shared_drafter_batch_allowed(int n_drafting) {
        if (dflash_shared_drafter_batch_disabled()) {
            return false;
        }
        if (n_drafting < 2) {
            return false;
        }
        if (params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH ||
                params_base.speculative.branch_budget != 0) {
            return false;
        }
        return true;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        bool is_resume = sleeping;

        SRV_INF("loading model '%s'\n", params.model.path.c_str());

        params_base = params;
        params_base.n_outputs_max = server_n_outputs_max(params_base);
        const std::string & mmproj_path = params_base.mmproj.path;
        const bool has_mmproj = !mmproj_path.empty();

        ggml_backend_dev_t dflash_single_draft_dev = nullptr;
        if (server_dflash_single_explicit_draft_device(params_base, dflash_single_draft_dev)) {
            if (params_base.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
                SRV_WRN("%s", "DFlash single --spec-draft-device cannot pin the target output tensor while target split-mode tensor is active\n");
            } else {
                params_base.output_device = dflash_single_draft_dev;
                SRV_INF("DFlash target output tensor will use explicit draft device %s before loading the target model\n",
                        ggml_backend_dev_name(dflash_single_draft_dev));
            }
        }

        if (has_mmproj) {
            const mtmd_decode_requirements mmproj_decode_req = mtmd_get_decode_requirements_from_file(mmproj_path.c_str());
            adjust_mmproj_decoder_batch_for_non_causal(mmproj_decode_req);
        }

        // measure mmproj memory for auto-fit (upstream #21489)
        // skip when mmproj_gpu_swap: mmproj starts on CPU, swap path handles GPU margin separately
        if (has_mmproj && params_base.fit_params && params_base.mmproj_use_gpu && llama_supports_gpu_offload() && !params_base.mmproj_gpu_swap) {
            auto mparams_measure = make_mmproj_params(params_base.mmproj_use_gpu);
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams_measure);
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_INF("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB\n", total / (1024.0 * 1024.0));
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        n_parallel_user = params_base.n_parallel;
        n_seq_max_full = n_parallel_user;
        recurrent_expanded = true;
        recurrent_backup_sequences = false;
        recurrent_backup_attention_streams = false;

        const bool has_mtp = params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);

        // Optionally reserve VRAM for the draft / MTP context before fitting the target model.
        if (params_base.fit_params) {
            const bool has_draft = params_base.speculative.has_dft();

            if (has_draft || has_mtp) {
                common_params params_dft = params_base;
                bool measure_model_bytes = true;

                if (has_draft) {
                    const auto & params_spec = params_base.speculative.draft;
                    params_dft.devices               = params_spec.devices;
                    params_dft.model                 = params_spec.mparams;
                    params_dft.n_gpu_layers          = params_spec.n_gpu_layers;
                    params_dft.cache_type_k          = params_spec.cache_type_k;
                    params_dft.cache_type_v          = params_spec.cache_type_v;
                    params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;
                } else {
                    // MTP draft context lives on the target model, only context+compute are new.
                    measure_model_bytes = false;
                }

                params_dft.n_outputs_max = params_base.n_parallel;

                auto mparams_dft = common_model_params_to_llama(params_dft);
                auto cparams_dft = common_context_params_to_llama(params_dft);
                if (has_mtp) {
                    cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                    cparams_dft.type_k   = params_base.speculative.draft.cache_type_k;
                    cparams_dft.type_v   = params_base.speculative.draft.cache_type_v;
                }
                cparams_dft.n_rs_seq = 0;

                std::vector<ggml_backend_dev_t> devs;
                uint32_t hp_ngl = 0;
                uint32_t hp_nct = 0;
                uint32_t hp_nex = 0;
                try {
                    auto dmd = common_get_device_memory_data(
                        params_dft.model.path.c_str(), &mparams_dft, &cparams_dft,
                        devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);

                    GGML_ASSERT(!params_base.fit_params_target.empty());
                    size_t total = 0;

                    std::vector<ggml_backend_dev_t> tgt_devices = params.devices;
                    if (tgt_devices.empty()) {
                        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                            tgt_devices.push_back(ggml_backend_dev_get(i));
                        }
                    }

                    for (size_t j = 0; j < devs.size(); ++j) {
                        const size_t bytes =
                            (measure_model_bytes ? dmd[j].mb.model : 0) +
                            dmd[j].mb.context +
                            dmd[j].mb.compute;
                        total += bytes;
                        for (size_t i = 0; i < tgt_devices.size(); i++) {
                            if (tgt_devices[i] == devs[j]) {
                                SRV_DBG("[spec] adding %.2f MiB to fit_params_target for device %s\n",
                                        bytes / (1024.0 * 1024.0), ggml_backend_dev_name(devs[j]));
                                params_base.fit_params_target[i] += bytes;
                                break;
                            }
                        }
                    }
                    SRV_INF("[spec] estimated memory usage of %s is %.2f MiB\n",
                            has_draft ? "draft model" : "MTP context",
                            total / (1024.0 * 1024.0));
                } catch (const std::exception & e) {
                    SRV_ERR("[spec] failed to measure %s memory: %s\n",
                            has_draft ? "draft model" : "MTP context", e.what());
                }
            }
        }

        // When mmproj GPU swap is active, run fitter before adding any DDTree
        // attention-backup streams. Those internal streams inflate recurrent
        // state estimates, causing the fitter to think even minimum context
        // doesn't fit.
        if (params_base.mmproj_gpu_swap && has_mtp && has_mmproj
                && params_base.fit_params && params_base.n_ctx == 0) {
            auto mparams_gpu = make_mmproj_params(true);
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams_gpu);
            size_t mmproj_gpu_estimate = 0;
            for (auto & [dev, size] : mmproj_mem) {
                mmproj_gpu_estimate += size;
            }
            if (mmproj_gpu_estimate == 0) {
                std::error_code ec;
                const auto fsize = std::filesystem::file_size(mmproj_path, ec);
                if (!ec && fsize > 0) {
                    mmproj_gpu_estimate = fsize + 256ull * 1024 * 1024;
                }
            }
            for (auto & margin : params_base.fit_params_target) {
                if (margin < mmproj_gpu_estimate) {
                    margin = mmproj_gpu_estimate;
                }
            }

            auto mparams_fit = common_model_params_to_llama(params_base);
            auto cparams_fit = common_context_params_to_llama(params_base);

            SRV_INF("mmproj GPU swap: running auto-fit with n_parallel=%d, margin=%zu MiB\n",
                    params_base.n_parallel, params_base.fit_params_target[0] / (1024 * 1024));

            common_fit_params(params_base.model.path.c_str(), &mparams_fit, &cparams_fit,
                params_base.tensor_split,
                params_base.tensor_buft_overrides.data(),
                params_base.fit_params_target.data(),
                params_base.fit_params_min_ctx,
                params_base.verbosity >= 4 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);

            if (cparams_fit.n_ctx > 0) {
                params_base.n_ctx = cparams_fit.n_ctx;
                SRV_INF("mmproj GPU swap: auto-fit chose n_ctx = %u\n", cparams_fit.n_ctx);
            }
            params_base.fit_params = false;
        }

        const server_dflash_recurrent_rollback_plan recurrent_plan = speculative_recurrent_rollback_plan();
        if (recurrent_plan.needs_backup_sequences) {
            n_seq_max_full = n_parallel_user * 2;
            recurrent_expanded = false;
            recurrent_backup_sequences = true;
            recurrent_backup_attention_streams = recurrent_plan.needs_attention_backup_streams;

            if (recurrent_backup_attention_streams) {
                params_base.n_parallel = n_seq_max_full;

                if (!params_base.kv_unified && n_parallel_user == 1 &&
                        params_base.kvarn.type == LLAMA_KVARN_TYPE_DISABLED) {
                    params_base.kv_unified = true;
                    SRV_INF("%s", "auto-enabled kv-unified: DDTree backup uses internal attention streams\n");
                } else if (!params_base.kv_unified && n_parallel_user == 1) {
                    SRV_WRN("%s", "KVarN requires non-unified KV; DDTree backup streams will share the configured total KV context\n");
                }
            }
        }

        llama_init = common_init_from_params(params_base);
        params_base.n_parallel = n_parallel_user;

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        needs_reeval = llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
        if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
                needs_reeval &&
                !recurrent_backup_sequences &&
                llama_n_rs_seq(ctx_tgt) == 0) {
            SRV_ERR("%s", "DFlash target requires recurrent rollback, but neither recurrent snapshots nor backup cells are available\n");
            return false;
        }

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (recurrent_backup_sequences && recurrent_backup_attention_streams && !recurrent_expanded && needs_reeval) {
            if (llama_context_recurrent_shrink(ctx_tgt, n_parallel_user)) {
                SRV_INF("shrunk recurrent state to %d cells before draft load (deferred %d backup cells)\n",
                        n_parallel_user, n_seq_max_full - n_parallel_user);
            } else {
                SRV_ERR("failed to shrink recurrent state to %d cells before draft load\n", n_parallel_user);
                return false;
            }
        }

        if (params_base.speculative.has_dft()) {
            // TODO speculative: move to common/speculative.cpp?
            const auto & params_spec = params_base.speculative.draft;
            const bool draft_devices_explicit = !params_spec.devices.empty();

            SRV_INF("loading draft model '%s'\n", params_spec.mparams.path.c_str());

            auto params_dft = params_base;

            params_dft.n_parallel   = params_base.n_parallel;
            params_dft.n_ctx        = params_spec.n_ctx == 0 ? llama_n_ctx_seq(ctx_tgt) : params_spec.n_ctx;
            params_dft.n_batch      = params_dft.n_ctx;
            params_dft.devices      = params_spec.devices;
            params_dft.model        = params_spec.mparams;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            params_dft.cache_type_k = params_spec.cache_type_k;
            params_dft.cache_type_v = params_spec.cache_type_v;
            params_dft.kvarn        = llama_kvarn_default_params();

            const bool draft_type_is_dflash = params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH;
            ggml_backend_dev_t explicit_single_draft_dev = server_single_gpu_device_from_list(params_spec.devices);
            const bool dflash_explicit_single_draft_device = draft_type_is_dflash && explicit_single_draft_dev != nullptr;
            if (dflash_explicit_single_draft_device) {
                params_dft.split_mode = LLAMA_SPLIT_MODE_NONE;
                params_dft.main_gpu = 0;
            }
            ggml_backend_dev_t target_output_dev = llama_model_dev_output(model_tgt);
            const bool target_output_is_gpu  = server_backend_dev_is_gpu(target_output_dev);
            const bool target_output_is_meta = target_output_dev && ggml_backend_dev_type(target_output_dev) == GGML_BACKEND_DEVICE_TYPE_META;
            const bool target_output_needs_shared_placement =
                    server_backend_dev_is_dflash_shared_output_compatible(target_output_dev);

            if (draft_type_is_dflash && !draft_devices_explicit) {
                if (target_output_is_meta) {
                    params_dft.split_mode = LLAMA_SPLIT_MODE_TENSOR;
                    params_dft.devices = params_base.devices;
                    SRV_INF("%s",
                            "DFlash draft model will use tensor-split placement with target Meta output; pass matching --spec-draft-device values to override\n");
                } else {
                    params_dft.split_mode = LLAMA_SPLIT_MODE_NONE;
                    if (target_output_is_gpu) {
                        params_dft.devices = { target_output_dev, nullptr };
                        params_dft.main_gpu = 0;
                        SRV_INF("DFlash draft model will use target output device %s by default; pass --spec-draft-device to override\n",
                                ggml_backend_dev_name(target_output_dev));
                    } else {
                        SRV_INF("%s", "DFlash draft model will use a single device by default; pass --spec-draft-device to override\n");
                    }
                }
            }

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            // Auto-detect DFlash from complete drafter metadata.
            bool draft_is_dflash = server_model_is_dflash_drafter(model_dft.get());
            if (draft_is_dflash && draft_devices_explicit && target_output_needs_shared_placement &&
                    !server_model_supports_device_buffer(model_dft.get(), target_output_dev)) {
                if (target_output_is_meta) {
                    SRV_ERR("%s",
                            "DFlash draft model uses shared target output tensor on Meta placement, but --spec-draft-device did not create a compatible tensor-split backend; omit --spec-draft-device for automatic tensor-split placement or pass matching draft devices\n");
                } else {
                    SRV_ERR("DFlash draft model uses shared target output tensor on device %s, but --spec-draft-device did not create a compatible backend; for single-device DFlash, use one --spec-draft-device value so the target output tensor is pinned before target load, or omit --spec-draft-device for automatic placement\n",
                            ggml_backend_dev_name(target_output_dev));
                }
                return false;
            }
            const bool dflash_auto_device_mismatch =
                    draft_is_dflash && !draft_devices_explicit && target_output_needs_shared_placement &&
                    !server_model_supports_device_buffer(model_dft.get(), target_output_dev);
            const bool dflash_auto_single_gpu_reload =
                    target_output_is_gpu &&
                    (llama_model_n_devices(model_dft.get()) > 1 ||
                     dflash_auto_device_mismatch);
            if (draft_is_dflash && !draft_devices_explicit && dflash_auto_single_gpu_reload) {
                SRV_INF("%s", "reloading auto-detected DFlash draft model on a single device; pass --spec-draft-device to override\n");
                model_dft.reset();
                params_dft.split_mode = LLAMA_SPLIT_MODE_NONE;
                if (target_output_is_gpu) {
                    params_dft.devices = { target_output_dev, nullptr };
                    params_dft.main_gpu = 0;
                }
                mparams_dft = common_model_params_to_llama(params_dft);
                model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
                if (model_dft == nullptr) {
                    SRV_ERR("failed to reload DFlash draft model on a single device, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }
                draft_is_dflash = server_model_is_dflash_drafter(model_dft.get());
            }
            if (draft_is_dflash && !draft_devices_explicit && target_output_is_meta &&
                    !server_model_supports_device_buffer(model_dft.get(), target_output_dev)) {
                SRV_ERR("%s",
                        "DFlash draft model could not create a tensor-split backend compatible with shared target Meta output; omit --spec-draft-device for automatic tensor-split placement or pass matching draft devices\n");
                return false;
            }
            if (draft_is_dflash &&
                params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH) {
                params_base.speculative.set_type(COMMON_SPECULATIVE_TYPE_DFLASH);
                params_base.speculative.apply_dflash_effective_defaults();
                SRV_INF("auto-detected DFlash drafter (block_size=%d)\n",
                        llama_model_dflash_block_size(model_dft.get()));
            }

            if (params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH &&
                params_base.speculative.dflash_only_args_explicit) {
                SRV_WRN("%s", "DFlash-only speculative args were supplied but the draft model is not DFlash; ignoring them\n");
                params_base.speculative.reset_dflash_only_args();
            }

            if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                if (!draft_is_dflash) {
                    SRV_ERR("draft model '%s' is not a valid DFlash drafter: missing complete DFlash metadata\n",
                            params_dft.model.path.c_str());
                    return false;
                }
                if (params_base.speculative.draft.n_ctx == 0) {
                    params_base.speculative.draft.n_ctx = 256;
                    params_dft.n_ctx = params_base.speculative.draft.n_ctx;
                    params_dft.n_batch = params_dft.n_ctx;
                    SRV_INF("dflash: setting -cd to %d after draft-model auto-detect (pass -cd N to override)\n",
                            params_base.speculative.draft.n_ctx);
                }
                const int block_size = llama_model_dflash_block_size(model_dft.get());
                const int dflash_draft_slots = params_base.speculative.dflash_max_slots > 0
                    ? params_base.speculative.dflash_max_slots
                    : params_base.n_parallel;
                const int dflash_draft_slots_clamped = std::max(1,
                    std::min({ dflash_draft_slots, params_base.n_parallel, (int) LLAMA_DFLASH_MAX_SLOTS }));
                const int dflash_draft_ctx_per_slot = params_dft.n_ctx;
                params_dft.n_ctx = dflash_draft_ctx_per_slot * dflash_draft_slots_clamped;
                params_dft.n_ubatch = LLAMA_DFLASH_MAX_SLOTS * block_size;
                params_dft.n_parallel = dflash_draft_slots_clamped;
                params_dft.kv_unified = false;
            }

            params_base.speculative.model_dft = model_dft.get();
            params_base.speculative.cparams_dft = common_context_params_to_llama(params_dft);

            if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                llama_model_share_tensors(model_dft.get(), llama_get_model(ctx_tgt));
            }

            // Upstream MTP: create draft context from target model's MTP heads
            const bool spec_mtp = params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
            if (spec_mtp && params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH) {
                auto cparams = common_context_params_to_llama(params_dft);
                cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                cparams.type_k   = params_base.speculative.draft.cache_type_k;
                cparams.type_v   = params_base.speculative.draft.cache_type_v;
                cparams.n_rs_seq = 0;
                ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));
                if (ctx_dft == nullptr) {
                    SRV_ERR("failed to create MTP draft-model context, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());
                if (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
                    SRV_WRN("MTP draft-model context can_seq_rm returned NO — overriding to FULL, type=%d\n", (int) ctx_dft_seq_rm_type);
                    ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                }
                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft.get();
            } else if (params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH) {
                auto cparams = common_context_params_to_llama(params_dft);
                ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));
                if (ctx_dft == nullptr) {
                    SRV_ERR("failed to create draft context, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());
                if (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
                    SRV_WRN("draft-model context can_seq_rm returned NO — overriding to FULL, type=%d\n", (int) ctx_dft_seq_rm_type);
                    ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                }
                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft.get();
            }
        } else if (params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP)) {
            SRV_INF("creating MTP draft context against the target model '%s'\n",
                    params_base.model.path.c_str());

            ctx_dft.reset(create_mtp_context());
            if (ctx_dft == nullptr) {
                SRV_ERR("%s", "failed to create MTP context\n");
                return false;
            }

            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());

            if (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
                SRV_WRN("MTP draft context can_seq_rm returned NO — overriding to FULL, type=%d\n", (int) ctx_dft_seq_rm_type);
                ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
            }

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();
        }

        if (has_mmproj) {
            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mmproj_gpu_swap = params_base.mmproj_gpu_swap && ctx_dft && !model_dft;

            const bool use_gpu = mmproj_gpu_swap ? false : params_base.mmproj_use_gpu;
            auto mparams = make_mmproj_params(use_gpu);
            if (!mmproj_gpu_swap) {
                mparams.warmup = params_base.warmup;
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }

            if (mmproj_gpu_swap) {
                SRV_INF("loaded multimodal model on CPU (GPU swap enabled), '%s'\n", mmproj_path.c_str());
            } else {
                SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());
            }

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }

        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        if (recurrent_backup_sequences && !recurrent_expanded && needs_reeval) {
            if (llama_context_recurrent_expand(ctx_tgt, n_seq_max_full)) {
                SRV_INF("expanded recurrent state to %d cells before speculative GPU buffers\n", n_seq_max_full);
                recurrent_expanded = true;
            } else {
                SRV_ERR("failed to expand recurrent state to %d cells before speculative GPU buffers\n", n_seq_max_full);
                return false;
            }
        }

        // setup slots
        SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        int n_ctx_slot = llama_n_ctx_seq(ctx_tgt);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_WRN("%s", "speculative decoding will use checkpoints\n");
        }

        const bool can_spec = (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO);

        // DFlash multi-slot: --spec-dflash-max-slots caps how many server slots keep DFlash state;
        // slots above the cap fall back to non-speculative decode (slot.spec stays null). The
        // matching tape/hidden buffers are allocated after the per-slot init loop (set_dflash_capture
        // runs inside common_speculative_init for slot 0, so dflash_capture must exist first).
        int dflash_slots_cap = 0;
        if (can_spec && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
            const int dflash_requested_slots = params_base.speculative.dflash_max_slots > 0
                ? params_base.speculative.dflash_max_slots
                : params_base.n_parallel;
            dflash_slots_cap = std::max(1,
                std::min({ dflash_requested_slots, params_base.n_parallel, (int) LLAMA_DFLASH_MAX_SLOTS }));
            if (dflash_slots_cap < params_base.n_parallel) {
                SRV_INF("DFlash enabled for slots 0..%d; slots %d+ will use non-speculative decode\n",
                        dflash_slots_cap - 1, dflash_slots_cap);
            } else {
                SRV_INF("DFlash enabled for all %d slots\n", dflash_slots_cap);
            }

            // Create the shared DFlash drafter context once;
            // every slot's common_speculative gets a non-owning reference to it.
            // dflash_slots_cap is passed at init so the initial graph reserve sizes
            // the compute buffer for the requested width — single-slot servers stay
            // narrow (cheap), multi-slot servers get a compute buffer big enough for
            // the batched cross-attention. Runtime widening past this cap requires a
            // larger compute buffer than is available.
            ctx_dft_shared.reset(common_speculative_create_ctx_dft(params_base.speculative, dflash_slots_cap));
            if (!ctx_dft_shared) {
                SRV_ERR("%s", "failed to create shared DFlash drafter context\n");
                return false;
            }
        }

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding through upstream's shared state. Bee-only
        // fork speculators keep per-slot state and are initialized below.
        const bool uses_fork_slot_spec = server_speculative_uses_fork_slot_impls(params_base.speculative);
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO && !dflash_slots_cap && !uses_fork_slot_spec) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
            }
        }

        if (spec) {
            SRV_INF("%s", "speculative decoding context initialized\n");
        } else {
            ctx_dft.reset();
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft.get();
            slot.n_ctx   = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;
            slot.reject_rng.seed(i + 1);
            const bool slot_dflash = can_spec &&
                params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
                i < dflash_slots_cap;

            slot.dm_adaptive         = slot_dflash && params_base.speculative.dm_adaptive;
            slot.dm_fringe_min       = params_base.speculative.dm_fringe_min;
            slot.dm_fringe_max       = params_base.speculative.dm_fringe_max;
            slot.dm_off_dwell        = params_base.speculative.dm_off_dwell;
            slot.dm_explore_interval = params_base.speculative.dm_explore_interval;
            slot.dm_min_reach        = params_base.speculative.dm_min_reach;
            slot.dm_probe_interval   = params_base.speculative.dm_probe_interval;
            slot.dm_probe_fraction   = params_base.speculative.dm_probe_fraction;
            slot.dm_fringe_window    = params_base.speculative.dm_fringe_window;
            slot.dm_controller       = params_base.speculative.dm_controller;
            slot.dm_profit_min       = params_base.speculative.dm_profit_min;
            slot.dm_profit_raise_margin = params_base.speculative.dm_profit_raise_margin;
            slot.dm_profit_lower_margin = params_base.speculative.dm_profit_lower_margin;
            slot.dm_profit_ewma_alpha = params_base.speculative.dm_profit_ewma_alpha;
            slot.dm_profit_min_samples = params_base.speculative.dm_profit_min_samples;
            slot.dm_profit_warmup    = params_base.speculative.dm_profit_warmup;
            slot.dm_profit_baseline_interval = params_base.speculative.dm_profit_baseline_interval;

            const bool slot_can_spec = can_spec &&
                (params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH || slot_dflash);
            const bool slot_uses_fork_spec =
                slot_can_spec && server_speculative_uses_fork_slot_impls(params_base.speculative);

            if (slot_can_spec) {
                if (slot_uses_fork_spec) {
                    try {
                        slot.spec.reset(common_speculative_init(params_base.speculative, slot.ctx_tgt, ctx_dft_shared.get()));
                    } catch (const std::exception & e) {
                        SRV_ERR("failed to initialize slot speculative decoding context: %s\n", e.what());
                        return false;
                    }
                }
                if (!slot.spec && spec) {
                    slot.spec_shared = spec.get();
                }
                if (slot.can_speculate()) {
                    common_speculative_set_seq_id(slot.get_spec(), slot.id);
                    SLT_INF(slot, "%s", "speculative decoding context initialized\n");
                }
            }


            SLT_INF(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        // Allocate DFlash per-slot tape + hidden buffers now that common_speculative_init
        // (run for slot 0 above) has created dflash_capture on the target context.
        if (dflash_slots_cap > 0) {
            llama_dflash_allocate_slots(ctx_tgt, dflash_slots_cap);
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_INF("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_INF("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_INF("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_INF("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_INF("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_INF("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_INF("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.name.empty()) {
            model_name = params_base.model.name;
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (!params_base.kv_unified) {
                SRV_WRN("%s", "--cache-idle-slots requires --kv-unified, disabling\n");
                params_base.cache_idle_slots = false;
            } else if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                SRV_INF("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        // populate UI settings (from either new ui_config_json or deprecated webui_config_json)
        {
            const std::string & cfg = !params_base.ui_config_json.empty()
                ? params_base.ui_config_json
                : params_base.webui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                    json_webui_settings = json_settings; // deprecated: keep in sync
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                LOG_INF("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // thinking is enabled if:
            // 1. It's not explicitly disabled via --reasoning off
            // 2. The chat template supports it
            const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
            const bool enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
            SRV_INF("%s: chat template, thinking = %d\n", __func__, enable_thinking);

            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };
        }

        // Shrink recurrent state to free backup cells during prefill.
        // Must happen after init-time decodes (common_speculative_is_compat, warmup)
        // so the scheduler's CUDA graph state isn't stale.
        if (recurrent_backup_sequences && !recurrent_expanded && needs_reeval) {
            if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH && n_parallel_user <= 2) {
                recurrent_expanded = true;
            } else {
                auto * mem = llama_get_memory(ctx_tgt);
                if (llama_memory_recurrent_shrink(mem, n_parallel_user)) {
                    SRV_INF("shrunk recurrent state to %d cells for prefill (deferred %d backup cells)\n",
                            n_parallel_user, n_seq_max_full - n_parallel_user);
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f) {
            float sim_best = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const float sim_cur = float(tokens.get_common_prefix(task.tokens)) / task.tokens.size();

                // select the current slot if the criteria match
                if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                    sim_best = sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                SLT_INF(*ret, "selected slot by LCP similarity, sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        sim_best, slot_prompt_similarity, f_keep);

                if (ret->dm_adaptive) {
                    if (server_adaptive_dm_should_preserve_for_continuation(sim_best, f_keep)) {
                        SLT_INF(*ret, "adaptive dm: preserving state for continuation (sim=%.3f, keep=%.3f, n_max=%d)\n",
                                sim_best, f_keep, ret->adaptive_n_max);
                    } else {
                        const bool preserve_policy = sim_best > 0.98f;
                        ret->reset_request_state(preserve_policy);
                        SLT_INF(*ret, "adaptive dm: reset state for prompt change (sim=%.3f, keep=%.3f, preserve_policy=%d, n_max=%d)\n",
                                sim_best, f_keep, preserve_policy ? 1 : 0, ret->adaptive_n_max);
                    }
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        // prefer spec-capable (DFlash) slots so requests get speculative decoding
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // strongly prefer spec-capable slots: pick a spec slot over a non-spec
                // slot regardless of LRU, then use LRU within the same capability tier
                const bool curr_spec = ret && ret->can_speculate();
                const bool slot_spec = slot.can_speculate();
                if (!ret ||
                    (slot_spec && !curr_spec) ||
                    (slot_spec == curr_spec && slot.t_last_used < t_last)) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                if (ret->dm_adaptive) {
                    ret->reset_request_state();
                    SLT_INF(*ret, "%s", "adaptive dm: reset state for LRU slot selection\n");
                }

                update_cache = true;
            }
        }

        if (ret) {
            const auto & tokens = ret->prompt.tokens;

            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                recurrent_shrink_for_prompt_cache("before prompt cache save/load");
                SRV_INF("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                // don't save the slot's state if its context is empty
                if (tokens.size() > 0) {
                    ret->prompt_save(*prompt_cache);
                }

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear(false);
                }

                prompt_cache->update();
                recurrent_expand_after_prompt_cache("after prompt cache save/load");

                SRV_INF("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear(false);

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.tokens.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.can_speculate());

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());
        } else {
            slot.smpl.reset();
        }

        slot.loop_guard.configure(task.params.reasoning_loop_guard);

        if (slot.can_speculate() && task.need_sampling() && slot.dm_adaptive) {
            const int configured_base_n_max = common_speculative_n_max(slot.spec.get(), task.params.speculative);
            const int base_n_max = dflash_effective_adaptive_base_n_max(
                    ctx_dft_shared.get(), task.params.speculative, configured_base_n_max);
            slot.reset_profit_if_config_changed(task.params.speculative, base_n_max,
                                                (int32_t) slot.prompt.n_tokens(),
                                                &task.params.sampling);
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool loop_guard_accept_enabled(const server_slot & slot) const {
        return slot.task &&
               slot.smpl &&
               slot.task->params.reasoning_loop_guard.mode != COMMON_REASONING_LOOP_GUARD_OFF &&
               slot.task->params.sampling.reasoning_budget_tracking;
    }

    bool handle_loop_guard_accept(server_slot & slot, const common_sampler_accept_info & info) {
        if (!loop_guard_accept_enabled(slot) || !info.is_generated) {
            return true;
        }

        const bool is_reasoning = server_accept_info_is_reasoning(info);
        if (is_reasoning) {
            slot.reasoning_output_tokens++;
        } else {
            slot.visible_output_tokens++;
            return true;
        }

        const bool forcing_reasoning_end = info.reasoning_state_before == REASONING_BUDGET_FORCING ||
                                           info.reasoning_state_after  == REASONING_BUDGET_FORCING;
        if (forcing_reasoning_end) {
            return true;
        }

        slot.loop_guard.accept(info.token, SERVER_LOOP_REGION_REASONING);

        const bool token_is_eog = llama_vocab_is_eog(vocab, info.token);
        if (!slot.loop_guard.should_check(SERVER_LOOP_REGION_REASONING, token_is_eog, forcing_reasoning_end)) {
            return true;
        }

        const auto check = slot.loop_guard.check(SERVER_LOOP_REGION_REASONING);
        if (!check.triggered) {
            return true;
        }

        slot.loop_guard_triggered = true;
        slot.loop_guard_reason = server_loop_guard_reason_to_string(check);

        const auto & params = slot.task->params.reasoning_loop_guard;
        if (params.mode == COMMON_REASONING_LOOP_GUARD_FORCE_CLOSE &&
                slot.loop_guard_interventions < params.interventions_max &&
                common_sampler_force_reasoning_end(slot.smpl.get())) {
            slot.loop_guard_interventions++;
            slot.loop_guard_action = "force-close";
            SLT_WRN(slot, "reasoning loop guard force-closing hidden reasoning: %s\n", slot.loop_guard_reason.c_str());
            return false;
        }

        slot.loop_guard_action = "stop";
        slot.stop = STOP_TYPE_LIMIT;
        slot.stop_detail = "reasoning_loop_guard";
        slot.has_next_token = false;
        SLT_WRN(slot, "reasoning loop guard stopping generation: %s\n", slot.loop_guard_reason.c_str());
        return false;
    }

    common_sampler_accept_callback make_loop_guard_accept_callback(server_slot & slot) {
        if (!loop_guard_accept_enabled(slot)) {
            return {};
        }

        const int32_t remaining = slot.remaining_generation_budget(params_base);
        return [this, &slot, remaining, n_accepted = 0](const common_sampler_accept_info & info) mutable {
            n_accepted++;
            if (!handle_loop_guard_accept(slot, info)) {
                return false;
            }
            return remaining == -1 || n_accepted < remaining;
        };
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        const bool stopped_before_process = slot.stop != STOP_TYPE_NONE && !slot.has_next_token;

        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = !stopped_before_process;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());
            const size_t marker_scan_pos = pos > 64 ? pos - 64 : 0;

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            if (slot.has_next_token &&
                    !slot.reasoning_tool_marker_logged &&
                    slot.task->params.sampling.grammar_lazy &&
                    !slot.task->params.sampling.grammar_triggers.empty()) {
                const auto reasoning_state = common_sampler_get_reasoning_budget_state(slot.smpl.get());
                const bool in_reasoning =
                    reasoning_state == REASONING_BUDGET_COUNTING ||
                    reasoning_state == REASONING_BUDGET_WAITING_UTF8;

                if (server_tail_has_tool_call_marker(slot.generated_text, marker_scan_pos)) {
                    slot.reasoning_tool_marker_logged = true;
                    SLT_WRN(slot,
                            "raw tool marker observed while lazy grammar is enabled; keeping DFlash governed by active grammar boundary in_reasoning=%d n_decoded=%d reasoning_tokens=%d visible_tokens=%d\n",
                            in_reasoning ? 1 : 0,
                            slot.n_decoded,
                            slot.reasoning_output_tokens,
                            slot.visible_output_tokens);
                }
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete && !stopped_before_process) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.stop_detail    = "token_limit";
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            // TODO: optimize this with min-p optimization
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task) {
        if (mctx) {
            send_error(id_task, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->stop_detail           = slot.stop_detail;
        res->reasoning_output_tokens = slot.reasoning_output_tokens;
        res->visible_output_tokens   = slot.visible_output_tokens;
        res->loop_guard_triggered    = slot.loop_guard_triggered;
        res->loop_guard_action       = slot.loop_guard_action;
        res->loop_guard_reason       = slot.loop_guard_reason;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_INF("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(
            server_slot & slot,
            const int64_t n_tokens_cur,
            llama_pos pos_min,
            llama_pos pos_max,
            int max_checkpoints) {
        if (max_checkpoints <= 0) {
            return;
        }

        // Save DFlash ring buffer alongside the recurrent state checkpoint.
        const bool dflash_checkpoint_slot =
            slot.can_speculate() &&
            params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH;
        const size_t cur_size_tgt = ctx_tgt
            ? llama_state_seq_get_size_ext(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
            : 0;
        const size_t cur_size_dft = ctx_dft
            ? llama_state_seq_get_size_ext(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
            : 0;
        size_t ring_size = dflash_checkpoint_slot
            ? common_speculative_ring_state_size(slot.get_spec())
            : 0;
        const size_t new_size = cur_size_tgt + cur_size_dft + ring_size;

        while (slot.prompt.checkpoints.size() >= (size_t) max_checkpoints) {
            erase_oldest_checkpoint(slot, "count limit");
        }

        common_prompt_checkpoint cur;
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        try {
            cur.data_tgt.resize(cur_size_tgt);
            cur.data_dft.resize(cur_size_dft);
            cur.ring_data.resize(ring_size);
        } catch (const std::bad_alloc & e) {
            SLT_WRN(slot,
                    "skipping context checkpoint: failed to allocate %.3f MiB state (%s)\n",
                    new_size / (1024.0 * 1024.0), e.what());
            return;
        }

        if (cur_size_tgt > 0) {
            const size_t n = llama_state_seq_get_data_ext(
                    ctx_tgt, cur.data_tgt.data(), cur_size_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (n != cur_size_tgt) {
                GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", cur_size_tgt, n);
            }
        }

        if (ctx_dft && cur_size_dft > 0) {
            const size_t n = llama_state_seq_get_data_ext(
                    ctx_dft.get(), cur.data_dft.data(), cur_size_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (n != cur_size_dft) {
                GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", cur_size_dft, n);
            }
        }

        bool ring_saved = false;
        if (dflash_checkpoint_slot && ring_size > 0) {
            ring_saved = common_speculative_ring_state_save(
                    slot.get_spec(), cur.ring_data.data(), ring_size);
            if (!ring_saved) {
                cur.ring_data.clear();
                ring_size = 0;
            }
        }
        if (dflash_checkpoint_slot &&
                dflash_server_profile_enabled(DFLASH_PROFILE_SUMMARY | DFLASH_PROFILE_PREFILL)) {
            SLT_INF(slot,
                    "dflash checkpoint: create pos=[%d,%d] n_tokens=%" PRId64
                    " ring_state_bytes=%zu ring_saved=%d\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, ring_size, ring_saved ? 1 : 0);
        }

        slot.prompt.checkpoints.push_back(std::move(cur));
        const auto & saved = slot.prompt.checkpoints.back();

        SLT_INF(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), max_checkpoints, saved.pos_min,
                saved.pos_max, saved.n_tokens, (float) saved.size() / 1024 / 1024);
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_slot = task.id_slot;
                    const int id_task = task.id;

                    server_slot * slot = id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else {
                        // Unified KV: check if launching this task would overflow the shared cell pool.
                        // Use max(current, planned) since a just-launched slot hasn't filled yet.
                        if (params_base.kv_unified && task.n_tokens() > 0 && task.n_tokens() < slot->n_ctx) {
                            int64_t cells_committed = 0;
                            for (const auto & s : slots) {
                                if (s.is_processing() && s.task) {
                                    cells_committed += std::max((int64_t) s.prompt.n_tokens(), (int64_t) s.task->n_tokens());
                                }
                            }
                            const int64_t cells_available = (int64_t) slot->n_ctx - cells_committed;
                            if (cells_available < (int64_t) task.n_tokens()) {
                                SRV_DBG("defer task %d: needs %d tokens but only %" PRId64 " cells available (%" PRId64 " committed by active slots)\n",
                                        id_task, task.n_tokens(), cells_available, cells_committed);
                                queue_tasks.defer(std::move(task));
                                break;
                            }
                        }

                        if (!launch_slot_with_task(*slot, std::move(task))) {
                            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & s : slots) {
                            if (!s.is_processing()) {
                                slot_save_and_clear(s);
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            if (slot.dm_adaptive) {
                                slot.reset_request_state();
                                SLT_INF(slot, "%s", "adaptive dm: reset state for canceled task\n");
                            }
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }

                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const size_t token_count = slot->prompt.tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens & tokens = slot->prompt.tokens.get_tokens();
                    const size_t nwrite = llama_state_seq_save_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    if (!check_no_mtmd(task.id)) break;
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    llama_tokens tokens;
                    tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), tokens.size(), &token_count);
                    if (nread == 0) {
                        slot->prompt.tokens.clear(); // KV may already been invalidated?
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    tokens.resize(token_count);
                    slot->prompt.tokens.clear();
                    slot->prompt.tokens.insert(tokens);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear(false);

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_INF("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }
    }

    void update_slots() {
        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_INF("%s", "all slots are idle\n");

                return;
            }
        }

        {
            SRV_DBG("%s", "posting NEXT_RESPONSE\n");

            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(task));
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot & slot : slots) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                const int n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                common_context_seq_rm (ctx_tgt, slot.id, n_keep            , n_keep + n_discard);
                common_context_seq_add(ctx_tgt, slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                if (ctx_dft) {
                    common_context_seq_rm (ctx_dft.get(), slot.id, n_keep            , n_keep + n_discard);
                    common_context_seq_add(ctx_dft.get(), slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);
                }

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.tokens.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        }

        // start populating the batch for this iteration
        common_batch_clear(batch);

        // track if given slot can be batched with slots already in the batch
        server_slot * slot_batched = nullptr;

        std::vector<server_slot *> generating;

        const int64_t t_cycle_start = ggml_time_us();
        int64_t t_draft_total = 0;
        int64_t t_verify_total = 0;
        int64_t t_accept_total = 0;
        int64_t t_replay_sync_total = 0;
        int64_t t_recurrent_backup_total = 0;
        int64_t t_tape_record_total = 0;
        uint64_t recurrent_backup_layers = 0;
        uint64_t recurrent_backup_tensors = 0;
        uint64_t recurrent_backup_cuda_d2d = 0;
        uint64_t recurrent_backup_fallback = 0;
        uint64_t recurrent_backup_enqueue_us = 0;
        uint64_t recurrent_backup_sync_us = 0;
        const bool profile_dflash_cycle = dflash_server_profile_enabled(
                DFLASH_PROFILE_SUMMARY | DFLASH_PROFILE_REPLAY | DFLASH_PROFILE_COPY);
        auto dflash_profile_start = [&]() -> int64_t {
            return profile_dflash_cycle ? ggml_time_us() : 0;
        };
        auto dflash_profile_add = [&](int64_t & total, int64_t start) {
            if (start != 0) {
                total += ggml_time_us() - start;
            }
        };
        auto dflash_recurrent_profile_reset = [&](llama_memory_t mem) {
            if (profile_dflash_cycle && mem) {
                mem->recurrent_copy_profile_reset();
            }
        };
        auto dflash_recurrent_profile_collect = [&](llama_memory_t mem) {
            if (!profile_dflash_cycle || !mem) {
                return;
            }
            const llama_memory_recurrent_copy_profile profile = mem->recurrent_copy_profile();
            recurrent_backup_layers += profile.layers_scanned;
            recurrent_backup_tensors += profile.tensors_copied;
            recurrent_backup_cuda_d2d += profile.cuda_d2d_queued;
            recurrent_backup_fallback += profile.fallback_copies;
            recurrent_backup_enqueue_us += profile.enqueue_us;
            recurrent_backup_sync_us += profile.sync_us;
        };
        auto dflash_backup_recurrent_state = [&](llama_seq_id seq_id_src, llama_seq_id seq_id_dst) {
            auto * mem = llama_get_memory(ctx_tgt);
            dflash_recurrent_profile_reset(mem);
            const int64_t t_backup_start = dflash_profile_start();
            if (dflash_server_crash_trace_enabled()) {
                SRV_INF("dflash crash breadcrumb: recurrent backup enter src=%d dst=%d\n",
                        (int) seq_id_src, (int) seq_id_dst);
            }
            const bool ordered = llama_dflash_memory_seq_cp_recurrent_ordered(ctx_tgt, seq_id_src, seq_id_dst, -1, -1);
            if (!ordered) {
                if (dflash_server_crash_trace_enabled()) {
                    SRV_INF("dflash crash breadcrumb: recurrent backup fallback copy src=%d dst=%d\n",
                            (int) seq_id_src, (int) seq_id_dst);
                }
                llama_memory_seq_cp_recurrent(mem, seq_id_src, seq_id_dst, -1, -1);
            } else if (dflash_server_crash_trace_enabled()) {
                SRV_INF("dflash crash breadcrumb: recurrent backup ordered copy complete src=%d dst=%d\n",
                        (int) seq_id_src, (int) seq_id_dst);
            }
            dflash_profile_add(t_recurrent_backup_total, t_backup_start);
            dflash_recurrent_profile_collect(mem);
            if (dflash_server_crash_trace_enabled()) {
                SRV_INF("dflash crash breadcrumb: recurrent backup exit src=%d dst=%d\n",
                        (int) seq_id_src, (int) seq_id_dst);
            }
        };
        int n_slots_drafted = 0;
        int n_profit_baseline_slots = 0;
        std::vector<server_slot *> profit_baseline_slots;
        server_slot * dflash_cycle_slot = nullptr;
        int dflash_cycle_n_draft = 0;
        int dflash_cycle_n_accept = 0;
        int dflash_cycle_verify_rows = 0;
        int dflash_cycle_useful_rows = 0;
        int dflash_cycle_pad_rows = 0;
        bool dflash_cycle_reduced_verify = false;
        int dflash_cycle_reduced_top_k = 0;
        llama_pos dflash_cycle_pos = -1;

        std::vector<llama_tokens> batched_drafts(slots.size());
        std::vector<std::vector<float>> batched_log_probs(slots.size());
        const bool use_rejection_sampling = params_base.speculative.sample_temp > 0.0f && params_base.sampling.temp > 0.0f;
        int dflash_tg_batch_rows = -1;
        auto dflash_multi_seq_tg_rows_safe = [](int rows) {
            return rows == 1;
        };
        auto dflash_try_reserve_tg_rows = [&](const server_slot & slot, int rows) {
            if (params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH ||
                    params_base.speculative.branch_budget != 0 ||
                    rows <= 0) {
                return true;
            }

            if (batch.n_tokens == 0 || dflash_tg_batch_rows < 0) {
                dflash_tg_batch_rows = rows;
                return true;
            }

            if (dflash_tg_batch_rows == rows && dflash_multi_seq_tg_rows_safe(rows)) {
                return true;
            }

            if (dflash_server_crash_trace_enabled()) {
                SLT_INF(slot, "dflash uneven TG rows deferred: reason=%s batch_rows=%d slot_rows=%d batch_tokens=%d\n",
                        dflash_tg_batch_rows == rows ? "multi-row" : "uneven",
                        dflash_tg_batch_rows, rows, batch.n_tokens);
            }
            return false;
        };
        auto dflash_skip_tg_slot_for_next_cycle = [](server_slot & slot) {
            slot.spec_i_batch.clear();
            slot.spec_pad_i_batch.clear();
            slot.spec_draft.clear();
            slot.draft_log_probs.clear();
        };
        const bool dflash_has_pending_prompt =
            params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
            std::any_of(slots.begin(), slots.end(), [](const server_slot & slot) {
                return slot.can_speculate() &&
                    (slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT);
            });
        const bool dflash_has_profit_baseline_slot =
            params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
            params_base.speculative.branch_budget == 0 &&
            std::any_of(slots.begin(), slots.end(), [](const server_slot & slot) {
                return slot.state == SLOT_STATE_GENERATING &&
                    slot.can_speculate() &&
                    slot.dm_adaptive &&
                    server_adaptive_dm_uses_profit_controller(slot.dm_controller) &&
                    (slot.profit_baseline_probe_pending ||
                        !slot.profit_baseline_ready() ||
                        slot.adaptive_n_max < 0);
            });
        const bool dflash_force_profit_baseline_cycle =
            dflash_has_profit_baseline_slot && !dflash_has_pending_prompt;
        std::vector<int> dflash_cohort_n_draft_max(slots.size(), -1);
        if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
                params_base.speculative.branch_budget == 0 &&
                !dflash_has_pending_prompt &&
                !dflash_force_profit_baseline_cycle) {
            std::vector<int> normal_n_max(slots.size(), 0);
            std::vector<int> cohort_cap_n_max(slots.size(), 0);

            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING ||
                        !slot.can_speculate() ||
                        slot.id < 0 ||
                        slot.id >= (int) slots.size()) {
                    continue;
                }

                normal_n_max[slot.id] = slot.get_n_draft_max(params_base, false);
                cohort_cap_n_max[slot.id] = normal_n_max[slot.id] > 0
                    ? normal_n_max[slot.id]
                    : slot.get_n_draft_max(params_base, false, INT_MAX);
            }

            const int cohort_n_max = server_adaptive_dm_resolve_cohort_n_max(
                    normal_n_max.data(), cohort_cap_n_max.data(), (int) slots.size());
            if (cohort_n_max > 0) {
                for (auto & slot : slots) {
                    if (slot.id < 0 || slot.id >= (int) slots.size() ||
                            cohort_cap_n_max[slot.id] <= 0) {
                        continue;
                    }
                    dflash_cohort_n_draft_max[slot.id] = std::min(cohort_n_max, cohort_cap_n_max[slot.id]);
                }
            }
        }
        auto dflash_scheduler_n_draft_max = [&](server_slot & slot, bool advance_adaptive_probe) {
            if (slot.id >= 0 && slot.id < (int) dflash_cohort_n_draft_max.size() &&
                    dflash_cohort_n_draft_max[slot.id] > 0) {
                return slot.get_n_draft_max(
                        params_base, advance_adaptive_probe, dflash_cohort_n_draft_max[slot.id]);
            }
            return slot.get_n_draft_max(params_base, advance_adaptive_probe);
        };
        if (ctx_dft_shared) {
            int n_drafting = 0;
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_GENERATING &&
                        slot.can_speculate() &&
                        dflash_scheduler_n_draft_max(slot, false) > 0 &&
                        !dflash_has_pending_prompt &&
                        !dflash_force_profit_baseline_cycle) {
                    n_drafting++;
                }
            }
            const bool can_shared_drafter_batch = dflash_shared_drafter_batch_allowed(n_drafting);
            const int shared_drafter_slots = can_shared_drafter_batch ? std::max(1, n_drafting) : 1;
            llama_set_dflash_n_slots(ctx_dft_shared.get(), shared_drafter_slots);

            if (can_shared_drafter_batch) {
                std::vector<common_speculative *> batch_specs;
                std::vector<llama_token>          batch_id_lasts;
                std::vector<int>                  batch_slot_ids;

                for (auto & slot : slots) {
                    if (slot.state == SLOT_STATE_GENERATING &&
                            slot.can_speculate() &&
                            dflash_scheduler_n_draft_max(slot, false) > 0 &&
                            !dflash_has_pending_prompt &&
                            !dflash_force_profit_baseline_cycle) {
                        batch_specs.push_back(slot.get_spec());
                        batch_id_lasts.push_back(slot.sampled);
                        batch_slot_ids.push_back(slot.id);
                    }
                }

                std::vector<llama_tokens> batch_results;
                std::vector<std::vector<float>> batch_log_probs;
                common_params_speculative params_batch = params_base.speculative;
                int batch_cohort_n_max = INT_MAX;
                for (const int slot_id : batch_slot_ids) {
                    if (slot_id >= 0 && slot_id < (int) dflash_cohort_n_draft_max.size() &&
                            dflash_cohort_n_draft_max[slot_id] > 0) {
                        batch_cohort_n_max = std::min(batch_cohort_n_max, dflash_cohort_n_draft_max[slot_id]);
                    }
                }
                if (batch_cohort_n_max != INT_MAX) {
                    params_batch.n_max = std::min(params_batch.n_max, batch_cohort_n_max);
                }
                const int64_t t_batch_start = ggml_time_us();
                common_speculative_draft_batch(
                        batch_specs, ctx_dft_shared.get(),
                        params_batch, batch_id_lasts, batch_results,
                        use_rejection_sampling ? &batch_log_probs : nullptr);
                t_draft_total = ggml_time_us() - t_batch_start;
                llama_set_dflash_n_slots(ctx_dft_shared.get(), 1);

                for (size_t i = 0; i < batch_slot_ids.size(); i++) {
                    batched_drafts[batch_slot_ids[i]] = std::move(batch_results[i]);
                    if (use_rejection_sampling && i < batch_log_probs.size()) {
                        batched_log_probs[batch_slot_ids[i]] = std::move(batch_log_probs[i]);
                    }
                }
            }
        }

        bool ddtree_batch_active = false;
        std::vector<server_slot *> mtp_drafting;
        std::vector<server_slot *> mtp_batching; // slots with existing draft that need batch building but not re-drafting
        const bool dflash_rotate_tg_slots =
            params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
            params_base.speculative.branch_budget == 0 &&
            !slots.empty();
        const int dflash_tg_slot_start = dflash_rotate_tg_slots
            ? std::clamp(dflash_next_tg_slot, 0, (int) slots.size() - 1)
            : 0;

        // first, add sampled tokens from any ongoing sequences
        for (int slot_offset = 0; slot_offset < (int) slots.size(); ++slot_offset) {
            const int slot_index = dflash_rotate_tg_slots
                ? (dflash_tg_slot_start + slot_offset) % (int) slots.size()
                : slot_offset;
            auto & slot = slots[slot_index];

            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            if (dflash_has_pending_prompt && slot.can_speculate()) {
                continue;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                continue;
            }

            int n_draft_max = dflash_scheduler_n_draft_max(slot, true);
            if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
                    params_base.speculative.branch_budget == 0) {
                n_draft_max = dflash_flat_effective_draft_max(ctx_dft_shared.get(), n_draft_max);
            }
            if (dflash_force_profit_baseline_cycle && n_draft_max > 0) {
                n_draft_max = 0;
            }
            const bool dflash_active_grammar =
                params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
                slot.smpl && common_sampler_has_active_grammar(slot.smpl.get());
            const bool dflash_sampler_blocks_speculative =
                params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
                slot.smpl && common_sampler_reasoning_is_forcing(slot.smpl.get());
            const bool non_dflash_media_prompt =
                mctx &&
                params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH &&
                slot.prompt.tokens.has_media();
            if (non_dflash_media_prompt) {
                SLT_DBG(slot, "%s", "non-DFlash speculative decoding does not support media inputs; using regular decode\n");
                n_draft_max = 0;
            }

            if (n_draft_max > 0 && !dflash_sampler_blocks_speculative) {
                const int64_t t_draft_slot_start = ggml_time_us();

                auto params_spec = slot.task->params.speculative;
                const llama_tokens & cached_text_tokens = slot.prompt.tokens.get_text_tokens();
                const int original_n_max = params_spec.n_max;
                params_spec.n_max = std::min(params_spec.n_max, n_draft_max);
                if (slot.dm_adaptive && server_adaptive_dm_uses_profit_controller(slot.dm_controller)) {
                    slot.profit_pending_requested_n_max = n_draft_max;
                }
                slot.n_tokens_before_draft = slot.prompt.n_tokens();
                slot.n_pos_before_draft = slot.prompt.tokens.pos_next();
                const bool use_ddtree = params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
                    !dflash_active_grammar &&
                    params_spec.branch_budget > 0 && batch.n_tokens == 0 && batched_drafts[slot.id].empty();

                if (use_ddtree) {
                    common_params_speculative params_tree = params_spec;
                    params_tree.n_max_base = original_n_max;
                    params_tree.n_max = std::min(params_tree.n_max, n_draft_max);

                    int tree_budget = std::min<int>(params_tree.n_max_base + params_tree.branch_budget,
                            params_tree.n_max_base * std::max<int>(1, params_tree.draft_topk));
                    tree_budget = std::min(tree_budget, std::max(0, (int) llama_n_batch(ctx_tgt) - 1));
                    tree_budget = std::min(tree_budget, std::max(0, (int) llama_n_ubatch(ctx_tgt) - 1));

                    common_speculative_tree tree;
                    if (tree_budget > 0) {
                        tree = common_speculative_draft_tree(slot.get_spec(), params_tree, cached_text_tokens, slot.sampled, tree_budget);
                    }

                    if (tree.n_nodes > 0 && tree.main_path_len >= slot.task->params.speculative.n_min) {
                        slot.n_tokens_before_draft = slot.prompt.n_tokens();
                        slot.n_pos_before_draft = slot.prompt.tokens.pos_next();

                        slot.spec_i_batch.push_back(batch.n_tokens);
                        common_batch_add(batch, slot.sampled, slot.n_pos_before_draft, { slot.id }, true);
                        slot.prompt.tokens.push_back(slot.sampled);

                        for (int i = 0; i < tree.n_nodes; i++) {
                            slot.spec_i_batch.push_back(batch.n_tokens);
                            common_batch_add(batch, tree.tokens[i], slot.n_pos_before_draft + tree.depths[i], { slot.id }, true);
                            slot.prompt.tokens.push_back(tree.tokens[i]);
                        }

                        slot.n_draft_total += tree.n_nodes;

                        if (needs_reeval) {
                            GGML_ASSERT(recurrent_backup_sequences && "DDTree recurrent rollback requires backup sequences");
                            const int64_t t_replay_sync_start = dflash_profile_start();
                            llama_tape_replay_sync(ctx_tgt);
                            dflash_profile_add(t_replay_sync_total, t_replay_sync_start);

                            if (!recurrent_expanded) {
                                if (llama_context_recurrent_expand(ctx_tgt, n_seq_max_full)) {
                                    SRV_INF("expanded recurrent state to %d cells for speculative backup\n", n_seq_max_full);
                                } else {
                                    SRV_ERR("failed to expand recurrent state to %d cells\n", n_seq_max_full);
                                    GGML_ABORT("failed to expand recurrent state for speculative backup; continuing would corrupt recurrent replay\n");
                                }
                                recurrent_expanded = true;
                            }

                            const llama_seq_id seq_backup = slot.id + n_parallel_user;
                            auto * mem = llama_get_memory(ctx_tgt);
                            llama_memory_seq_rm(mem, seq_backup, -1, -1);
                            int n_branches = 0;
                            for (size_t i = 1; i < tree.parents.size(); ++i) {
                                if (tree.parents[i] != -1 && tree.parents[i] != (int32_t)(i - 1)) {
                                    n_branches++;
                                }
                            }
                            if (n_branches > 0) {
                                llama_memory_seq_cp(mem, slot.id, seq_backup, -1, -1);
                            } else {
                                dflash_backup_recurrent_state(slot.id, seq_backup);
                            }
                            slot.has_draft_backup = true;
                            slot.has_recurrent_only_backup = (n_branches == 0);
                            slot.seq_id_backup = seq_backup;
                        }

                        llama_set_tree_mask(ctx_tgt, tree.visibility.data(), tree.n_nodes + 1);
                        llama_set_tree_parent_ids(ctx_tgt, tree.parents.data(), tree.n_nodes + 1);

                        slot.drafted = tree.tokens;
                        slot.draft_log_probs = tree.log_probs;
                        slot.draft_tree = std::move(tree);
                        slot.has_draft_tree = true;
                        ddtree_batch_active = true;

                        t_draft_total += ggml_time_us() - t_draft_slot_start;
                        n_slots_drafted++;
                        break;
                    }

                    SLT_DBG(slot, "%s", "DDTree draft unavailable, falling back to flat speculative decode\n");
                }

                const bool use_mtp_spec = params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP) &&
                    params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH;
                const llama_pos draft_n_past = use_mtp_spec ? slot.prompt.n_tokens() : -1;
                if (use_mtp_spec) {
                    // MTP: collect draft params only; draft/checkpoint/batch handled in post-loop phases
                    // (matches upstream's collect-then-draft-then-checkpoint-then-batch pattern)
                    common_speculative_get_draft_params(slot.get_spec(), slot.id).drafting = false;
                    if (!slot.spec_draft.empty()) {
                        // Post-checkpoint-restore: reuse previous (partial) draft.
                        // Do NOT re-draft (assertion in common_speculative_draft requires
                        // dp.result->empty() when drafting=true). Just batch-build the
                        // already-accepted tokens for re-verification against the target.
                        const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                        mtp_batching.push_back(&slot);
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft.get(), slot.id,
                                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();
                        common_speculative_get_draft_params(slot.get_spec(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_min    = */ params_spec.n_min,
                            /* .n_past   = */ draft_n_past,
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };
                        slot.draft_log_probs.clear();
                        mtp_drafting.push_back(&slot);
                    }
                } else if (!batched_drafts[slot.id].empty()) {
                    slot.spec_draft = std::move(batched_drafts[slot.id]);
                    slot.draft_log_probs = std::move(batched_log_probs[slot.id]);
                } else if (use_rejection_sampling) {
                    slot.draft_log_probs.clear();
                    slot.spec_draft = common_speculative_draft(slot.get_spec(), params_spec, cached_text_tokens, slot.sampled, &slot.draft_log_probs, draft_n_past);
                } else {
                    slot.draft_log_probs.clear();
                    slot.spec_draft = common_speculative_draft(slot.get_spec(), params_spec, cached_text_tokens, slot.sampled, nullptr, draft_n_past);
                }
                slot.spec_pad_i_batch.clear();

                if (slot.spec_draft.size() > (size_t) n_draft_max) {
                    SLT_WRN(slot, "draft size %d exceeds max %d, truncating\n", (int) slot.spec_draft.size(), n_draft_max);
                    slot.spec_draft.resize(n_draft_max);
                    if (slot.draft_log_probs.size() > (size_t) n_draft_max) {
                        slot.draft_log_probs.resize(n_draft_max);
                    }
                }

                // MTP batch building is deferred to the post-loop phase below
                if (!use_mtp_spec) {
                    const bool small_draft =
                        slot.spec_draft.empty() ||
                        slot.task->params.speculative.n_min > (int) slot.spec_draft.size();
                    const int dflash_slot_tg_rows = small_draft ? 1 : 1 + (int) slot.spec_draft.size();
                    if (!dflash_try_reserve_tg_rows(slot, dflash_slot_tg_rows)) {
                        dflash_skip_tg_slot_for_next_cycle(slot);
                        continue;
                    }

                    slot.spec_i_batch.push_back(batch.n_tokens);
                    common_batch_add(batch, slot.sampled, slot.prompt.tokens.pos_next(), { slot.id }, true);
                    slot.prompt.tokens.push_back(slot.sampled);

                    if (small_draft) {
                        if (!slot.spec_draft.empty()) {
                            SLT_DBG(slot, "ignoring small draft: %d < %d\n", (int) slot.spec_draft.size(), slot.task->params.speculative.n_min);
                        }
                        slot.i_batch = slot.spec_i_batch[0];
                        slot.spec_draft.clear();
                        slot.draft_log_probs.clear();
                        slot.spec_i_batch.clear();
                        slot.spec_pad_i_batch.clear();
                    } else {
                        slot.n_draft_total += slot.spec_draft.size();

                        if (needs_reeval) {
                            if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                                const int64_t t_replay_sync_start = dflash_profile_start();
                                llama_tape_replay_sync(ctx_tgt);
                                dflash_profile_add(t_replay_sync_total, t_replay_sync_start);
                                if (params_base.speculative.branch_budget > 0) {
                                    const int n_batch_tokens = 1 + (int) slot.spec_draft.size();
                                    std::vector<int32_t> linear_parents(n_batch_tokens);
                                    linear_parents[0] = -1;
                                    for (int i = 1; i < n_batch_tokens; i++) {
                                        linear_parents[i] = i - 1;
                                    }
                                    llama_set_tree_parent_ids(ctx_tgt, linear_parents.data(), n_batch_tokens);
                                }
                            }

                            // backup sequences are only needed for DFlash on non-RS contexts
                            // RS contexts handle rollback internally via seq_rm snapshots
                            // MTP uses the checkpoint-based accept path (which cleans ctx_dft via seq_rm)
                            if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                                if (!recurrent_backup_sequences) {
                                    GGML_ABORT("speculative recurrent rollback requires backup sequences when bounded snapshots are unavailable\n");
                                }
                                if (!recurrent_expanded) {
                                    if (llama_context_recurrent_expand(ctx_tgt, n_seq_max_full)) {
                                        SRV_INF("expanded recurrent state to %d cells for speculative backup\n", n_seq_max_full);
                                    } else {
                                        SRV_ERR("failed to expand recurrent state to %d cells\n", n_seq_max_full);
                                        GGML_ABORT("failed to expand recurrent state for speculative backup; continuing would corrupt recurrent replay\n");
                                    }
                                    recurrent_expanded = true;
                                }
                                const llama_seq_id seq_backup = slot.id + n_parallel_user;
                                auto * mem = llama_get_memory(ctx_tgt);
                                if (recurrent_backup_attention_streams) {
                                    llama_memory_seq_rm(mem, seq_backup, -1, -1);
                                } else {
                                    llama_memory_seq_rm_recurrent(mem, seq_backup, -1, -1);
                                }
                                if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                                    dflash_backup_recurrent_state(slot.id, seq_backup);
                                    slot.has_recurrent_only_backup = true;
                                } else {
                                    llama_memory_seq_cp(mem, slot.id, seq_backup, -1, -1);
                                    slot.has_recurrent_only_backup = false;
                                }
                                slot.has_draft_backup = true;
                                slot.seq_id_backup = seq_backup;
                            }
                        }

                        for (size_t i = 0; i < slot.spec_draft.size(); i++) {
                            slot.spec_i_batch.push_back(batch.n_tokens);
                            common_batch_add(batch, slot.spec_draft[i], slot.prompt.tokens.pos_next(), { slot.id }, true);
                            slot.prompt.tokens.push_back(slot.spec_draft[i]);
                        }
                        const int active_verify_draft_max = n_draft_max;
                        const common_params_sampling & slot_sampling =
                            slot.task ? slot.task->params.sampling : params_base.sampling;
                        if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
                                params_base.speculative.branch_budget == 0 &&
                                dflash_verify_padding_enabled() &&
                                !use_rejection_sampling &&
                                slot.spec_draft.size() < (size_t) active_verify_draft_max &&
                                common_sampler_supports_reduced(slot.smpl.get()) &&
                                dflash_select_reduced_verify_plan(slot_sampling, params_base.speculative,
                                        use_rejection_sampling, false).enabled) {
                            const int max_pad_draft = std::min<int>(active_verify_draft_max,
                                    std::min<int>((int) llama_n_batch(ctx_tgt) - 1, (int) llama_n_ubatch(ctx_tgt) - 1));
                            const int max_rows = std::min<int>((int) llama_n_batch(ctx_tgt), (int) llama_n_ubatch(ctx_tgt));
                            const int rows_available = std::max<int>(0, max_rows - batch.n_tokens);
                            const int pad_count = std::min<int>(rows_available, std::max<int>(0, max_pad_draft - (int) slot.spec_draft.size()));
                            const llama_pos pad_pos0 = slot.prompt.tokens.pos_next();
                            for (int i = 0; i < pad_count; ++i) {
                                slot.spec_pad_i_batch.push_back(batch.n_tokens);
                                common_batch_add(batch, slot.sampled, pad_pos0 + i, { slot.id }, true);
                            }
                            if (pad_count > 0) {
                                SLT_DBG(slot, "padded DFlash verifier batch by %d tokens to active graph shape (GGML_DFLASH_VERIFY_PAD=1)\n", pad_count);
                            }
                        }
                    }
                }
                t_draft_total += ggml_time_us() - t_draft_slot_start;
                n_slots_drafted++;
            } else {
                if (!dflash_try_reserve_tg_rows(slot, 1)) {
                    dflash_skip_tg_slot_for_next_cycle(slot);
                    continue;
                }

                if (slot.can_speculate() && slot.dm_adaptive &&
                        server_adaptive_dm_uses_profit_controller(slot.dm_controller) &&
                        slot.profit_expects_baseline_sample()) {
                    n_profit_baseline_slots++;
                    profit_baseline_slots.push_back(&slot);
                }

                slot.i_batch = batch.n_tokens;

                common_batch_add(batch, slot.sampled, slot.prompt.tokens.pos_next(), { slot.id }, true);

                slot.prompt.tokens.push_back(slot.sampled);

                SLT_DBG(slot, "slot decode token, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                        slot.n_ctx, slot.prompt.n_tokens(), slot.truncated);
            }
        }

        // --- MTP post-loop phases (upstream-aligned) ---
        // Phase 2: single draft call for all collected MTP slots
        if (!mtp_drafting.empty()) {
            const int64_t t_mtp_draft_start = ggml_time_us();
            common_speculative_draft(spec.get());
            t_draft_total += ggml_time_us() - t_mtp_draft_start;
        }

        // Phase 3: checkpoint creation and batch building for MTP slots
        for (auto * slot_ptr : mtp_drafting) {
            auto & slot = *slot_ptr;

            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.n_draft_total += draft.size();

            // checkpoint: load dft state and seq_rm past checkpoint
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft.get(), slot.id,
                        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                }
                common_context_seq_rm(ctx_dft.get(), slot.id, ckpt.pos_max + 1, -1);
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft_rs =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft.get()));

                if (use_ckpt_tgt) {
                    ckpt.update_tgt(ctx_tgt, slot.id,
                        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                }

                if (use_ckpt_dft_rs) {
                    ckpt.update_dft(ctx_dft.get(), slot.id,
                        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                }
            }

            if (slot.spec_draft.empty() || slot.task->params.speculative.n_min > (int) slot.spec_draft.size()) {
                if (!slot.spec_draft.empty()) {
                    SLT_DBG(slot, "ignoring small draft: %d < %d\n", (int) slot.spec_draft.size(), slot.task->params.speculative.n_min);
                }
                slot.spec_draft.clear();
                slot.draft_log_probs.clear();
                slot.spec_pad_i_batch.clear();
            }

            slot.update_batch(batch);
        }

        // Phase 3b: batch building only for post-restore MTP slots (no re-drafting)
        for (auto * slot_ptr : mtp_batching) {
            auto & slot = *slot_ptr;

            slot.n_draft_total += slot.spec_draft.size();
            slot.update_batch(batch);
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        // track how many TG tokens are in the batch vs total, to detect
        // pure-verify batches where multi-seq batching is safe.
        const int32_t n_tg_tokens = batch.n_tokens;
        const bool dflash_batch_has_tg =
            params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
            n_tg_tokens > 0;

        float  alora_scale       = -1.0f;
        size_t alora_disabled_id = 0;

        // next, batch any pending prompts without exceeding n_batch
        if (!ddtree_batch_active && !dflash_batch_has_tg && (params_base.cont_batching || batch.n_tokens == 0)) {
            for (auto & slot : slots) {
                if (!slot.is_processing()) {
                    continue;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    continue;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    continue;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.n_tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            continue;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            common_context_seq_rm (ctx_tgt, slot.id, head_p, head_c);
                                            common_context_seq_add(ctx_tgt, slot.id, head_c, head_c + n_match, kv_shift);

                                            if (ctx_dft) {
                                                common_context_seq_rm (ctx_dft.get(), slot.id, head_p, head_c);
                                                common_context_seq_add(ctx_dft.get(), slot.id, head_c, head_c + n_match, kv_shift);
                                            }

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);
                            const bool has_new_tokens = n_past < slot.task->n_tokens();

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (n_swa > 0 && pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&, func_name = __func__](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            LOG_INF("slot %12.*s: id %2d | task %d | Checking checkpoint with [%d, %d] against %d...\n", 12,
                                                func_name, (slot).id, ((slot).task ? (slot).task->id : -1), cur.pos_min, cur.pos_max, pos_min_thold);
                                            // For hybrid/recurrent models (DeltaNet, Mamba), pos_min always equals
                                            // the full sequence length, so the SWA-based pos_min check always fails.
                                            // Use pos_max <= pos_next instead to find the most recent valid checkpoint.
                                            return server_prompt_checkpoint_matches_restore_window(
                                                    cur,
                                                    pos_min_thold,
                                                    pos_next,
                                                    llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt));
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        const size_t checkpoint_size = it->data_tgt.size();
                                        const size_t n = llama_state_seq_set_data_ext(ctx_tgt, it->data_tgt.data(), checkpoint_size, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                        if (n != checkpoint_size) {
                                            SLT_ERR(slot, "failed to restore context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, (float) checkpoint_size / 1024 / 1024);
                                            do_reset = true;
                                        } else {
                                            it->load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                            pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                            n_past = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);

                                            if (slot.can_speculate() && !it->ring_data.empty()) {
                                                common_speculative_ring_state_load(slot.get_spec(), it->ring_data.data(), it->ring_data.size());
                                            }

                                            SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                        }
                                    }

                                    if (do_reset) {
                                        SLT_WRN(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_WRN(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        {
                            const llama_pos prompt_trim_p0 = slot.prompt.tokens.pos_next(n_past);
                            const bool can_trim_tgt =
                                llama_memory_can_seq_rm(llama_get_memory(ctx_tgt), slot.id, prompt_trim_p0, -1);
                            const bool can_trim_dft =
                                ctx_dft == nullptr ||
                                llama_memory_can_seq_rm(llama_get_memory(ctx_dft.get()), slot.id, prompt_trim_p0, -1);

                            if (!can_trim_tgt || !can_trim_dft) {
                                SLT_WRN(slot,
                                        "forcing full prompt re-processing because memory cannot trim cached suffix from %d (target = %d, draft = %d)\n",
                                        prompt_trim_p0, can_trim_tgt ? 1 : 0, can_trim_dft ? 1 : 0);
                                n_past = 0;
                                slot.prompt.checkpoints.clear();
                            }
                        }

                        slot.n_prompt_tokens_cache = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        // A zero-prefix request cannot reuse the previous request's DFlash ring.
                        // Clear it before early prompt checkpoints can snapshot stale full-ring state.
                        if (n_past == 0 &&
                                slot.prompt.n_tokens() > 0 &&
                                slot.can_speculate() &&
                                params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                            common_speculative_discard_dflash_state(slot.get_spec(), nullptr);
                        }

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    }

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.task->n_tokens() > n_batch) {
                            continue;
                        }
                    }

                    const int64_t t_current = ggml_time_us();
                    slot.t_prompt_processing = (t_current - slot.t_start_process_prompt) / 1e3;
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    if (!llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, p0, -1)) {
                        SLT_WRN(slot, "partial seq_rm at p0=%d failed (recurrent backend cannot roll into cache); clearing slot and re-evaluating from scratch\n", p0);
                        slot.n_prompt_tokens_cache = 0;
                        slot.prompt.tokens.keep_first(0);
                        common_context_seq_rm(ctx_tgt, slot.id, 0, -1);
                    }

                    if (ctx_dft) {
                        if (!llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), slot.id, p0, -1)) {
                            common_context_seq_rm(ctx_dft.get(), slot.id, 0, -1);
                        }
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    const bool checkpoint_memory_needs_restore =
                        ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                        ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                        n_swa > 0 ||
                        llama_model_is_hybrid(model_tgt);

                    const bool do_prompt_cache_boundary_checkpoint =
                        prompt_cache_boundary_checkpoints_enabled(slot);
                    const bool do_regular_context_checkpoint =
                        params_base.n_ctx_checkpoints > 0;
                    const int effective_checkpoint_limit =
                        server_prompt_effective_checkpoint_limit(
                                params_base.n_ctx_checkpoints,
                                do_prompt_cache_boundary_checkpoint);

                    bool do_checkpoint = effective_checkpoint_limit > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // no-cache requests should not pay to create checkpoints for future prompt reuse
                    do_checkpoint = do_checkpoint && slot.task->params.cache_prompt;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    do_checkpoint = do_checkpoint && checkpoint_memory_needs_restore;

                    bool has_mtmd = false;

                    // swap MTP out of VRAM so mmproj can use GPU for image encoding
                    const bool needs_mmproj_swap = mmproj_gpu_swap && !mmproj_is_on_gpu
                        && slot.prompt.n_tokens() < slot.task->n_tokens()
                        && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL;

                    if (needs_mmproj_swap) {
                        swap_mtp_to_mmproj_gpu();
                    }

                    // check if we should process the image
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL) {
                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = input_tokens.process_chunk(ctx_tgt, mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (ctx_dft) {
                            // TODO: in the future, figure out how to infuse target embeddings to the images
                            //       for now, we skip this for simplicity
                            //       maybe we simply need to call `common_speculative_process()` on the mtmd batches in the `process_chunk` above?
                            res = input_tokens.process_chunk(ctx_dft.get(), mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                            if (res != 0) {
                                GGML_ABORT("failed to process multi-modal data on draft context\n");
                            }
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(slot.prompt.n_tokens());
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        has_mtmd = true;
                    }

                    if (needs_mmproj_swap && mmproj_is_on_gpu) {
                        swap_mmproj_to_mtp();
                    }

                    const int32_t n_before_user = slot.task->params.n_before_user;
                    const bool n_before_user_known = n_before_user > 0;

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.n_tokens < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requests require all tokens in the batch to be output;
                        // NextN/MTP hidden rows are captured separately and do not need
                        // prompt logits for every token.
                        common_batch_add(batch,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            { slot.id },
                            slot.need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // stop the prompt batch exactly before the latest user input, so a checkpoint
                        // can be created after the previous messages
                        if (n_before_user_known &&
                            slot.prompt.n_tokens() == n_before_user) {
                            break;
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_regular_context_checkpoint && do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.n_tokens - n_tokens_prev;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints
                        if (!n_before_user_known && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // checkpoints are created before the current batch is decoded, so
                    // their token position is the batch start rather than the prompt end
                    const int32_t n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    {
                        const bool is_on_user =
                            n_before_user_known &&
                            n_tokens_start == n_before_user;

                        const bool is_after_user =
                            n_before_user_known &&
                            n_tokens_start > n_before_user;

                        const bool is_allowed = server_prompt_checkpoint_creation_allowed(
                                do_prompt_cache_boundary_checkpoint,
                                n_before_user_known,
                                is_on_user,
                                is_after_user,
                                near_prompt_end);

                        if (do_checkpoint && !is_allowed) {
                            do_checkpoint = false;
                        }
                    }

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // no need for empty or small checkpoints
                    // for hybrid/recurrent models, lower the checkpoint threshold so short prompts also get checkpointed
                    const int checkpoint_min_tokens = (llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt)) ? 4 : 64;
                    do_checkpoint = do_checkpoint && slot.prompt.n_tokens() >= checkpoint_min_tokens;

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together
                    do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max, effective_checkpoint_limit);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        if (slot_batched) {
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        if (batch.n_tokens == 0) {
            SRV_WRN("%s", "no tokens to decode\n");
            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }
        } else {
            n_empty_consecutive = 0;
        }

        // DFlash: enable tape recording if any slot has draft backup (needs tape replay for rollback)
        const bool dflash_tape_active = needs_reeval
            && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH
            && std::any_of(slots.begin(), slots.end(), [](const server_slot & s) { return s.has_draft_backup; });
        if (dflash_tape_active) {
            const int64_t t_tape_record_start = dflash_profile_start();
            llama_set_tape_recording(ctx_tgt, true);
            dflash_profile_add(t_tape_record_total, t_tape_record_start);
        }

        int32_t i_next = 0;

        // Allow multi-seq batching only when the batch is pure TG (no prompt
        // tokens) and every participating sequence owns DFlash state. With
        // --spec-dflash-max-slots below -np, higher slots are intentionally
        // non-speculative; batching them with DFlash slots leaves the target
        // capture path with partial per-seq buffers.
        const bool dflash_pure_tg_batch =
            n_tg_tokens == batch.n_tokens &&
            n_tg_tokens > 0 &&
            params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH;
        const char * dflash_multiseq_block_reason = "not-pure-tg";
        llama_seq_id dflash_multiseq_block_seq = -1;
        int dflash_multiseq_n_unique = 0;
        int dflash_multiseq_rows[LLAMA_DFLASH_MAX_SLOTS] = {};
        const bool dflash_tg_batch_all_spec_slots = [&]() {
            if (!dflash_pure_tg_batch) {
                return false;
            }

            llama_seq_id unique_seqs[LLAMA_DFLASH_MAX_SLOTS] = {};
            for (int32_t t = 0; t < n_tg_tokens; ++t) {
                for (int32_t k = 0; k < batch.n_seq_id[t]; ++k) {
                    const llama_seq_id seq = batch.seq_id[t][k];
                    int seq_idx = -1;
                    for (int s = 0; s < dflash_multiseq_n_unique; ++s) {
                        if (unique_seqs[s] == seq) {
                            seq_idx = s;
                            break;
                        }
                    }
                    if (seq_idx < 0) {
                        if (dflash_multiseq_n_unique >= (int) LLAMA_DFLASH_MAX_SLOTS) {
                            dflash_multiseq_block_reason = "too-many-seqs";
                            dflash_multiseq_block_seq = seq;
                            return false;
                        }
                        seq_idx = dflash_multiseq_n_unique;
                        unique_seqs[dflash_multiseq_n_unique++] = seq;
                    }
                    dflash_multiseq_rows[seq_idx]++;
                }
            }

            if (needs_reeval && dflash_multiseq_n_unique > 1 && ddtree_batch_active) {
                dflash_multiseq_block_reason = "target-recurrent-tree";
                dflash_multiseq_block_seq = unique_seqs[1];
                return false;
            }

            // Graph-embedded GPU hidden capture stores one fixed row count per
            // ubatch. Uneven per-seq rows split into multiple ubatches and would
            // leave the longer slot with only the final ubatch's hidden rows.
            int expected_rows = -1;
            for (int s = 0; s < dflash_multiseq_n_unique; ++s) {
                const llama_seq_id seq = unique_seqs[s];
                auto it = std::find_if(slots.begin(), slots.end(), [seq](const server_slot & slot) {
                    return slot.id == seq;
                });
                if (it == slots.end() || !it->can_speculate() || !it->get_spec()) {
                    dflash_multiseq_block_reason = "non-dflash-slot";
                    dflash_multiseq_block_seq = seq;
                    return false;
                }

                const int rows = dflash_multiseq_rows[s];
                if (rows <= 0) {
                    dflash_multiseq_block_reason = "empty-seq";
                    dflash_multiseq_block_seq = seq;
                    return false;
                }
                if (expected_rows < 0) {
                    expected_rows = rows;
                } else if (rows != expected_rows) {
                    dflash_multiseq_block_reason = "uneven-rows";
                    dflash_multiseq_block_seq = seq;
                    return false;
                }
            }

            dflash_multiseq_block_reason = "ok";
            return true;
        }();
        if (dflash_pure_tg_batch && !dflash_tg_batch_all_spec_slots &&
                dflash_server_crash_trace_enabled()) {
            SRV_INF("dflash multiseq target batching disabled: reason=%s seq=%d unique=%d n_tg_tokens=%d batch_tokens=%d\n",
                    dflash_multiseq_block_reason,
                    (int) dflash_multiseq_block_seq,
                    dflash_multiseq_n_unique,
                    n_tg_tokens,
                    batch.n_tokens);
        }
        const bool can_batch_multiseq = dflash_pure_tg_batch && dflash_tg_batch_all_spec_slots;
        if (can_batch_multiseq) {
            llama_set_force_split_seq(ctx_tgt, false);
        }

        dflash_reduced_verify_plan dflash_verify_plan =
            dflash_select_batch_reduced_verify_plan(slots, params_base.sampling, params_base.speculative,
                    use_rejection_sampling, ddtree_batch_active);
        const char * dflash_target_reduced_reason = dflash_verify_plan.reason;
        if (dflash_verify_plan.enabled &&
                !dflash_target_reduced_verify_supported(ctx_tgt, &dflash_target_reduced_reason)) {
            dflash_verify_plan.enabled = false;
            dflash_verify_plan.reason = dflash_target_reduced_reason;
        }
        bool dflash_reduced_verify_ready = false;
        int  dflash_reduced_verify_top_k = dflash_verify_plan.top_k;
        int32_t dflash_reduced_verify_view_start = 0;
        bool dflash_verify_graph_enabled = false;

        if (dflash_verify_plan.enabled) {
            for (int32_t j = 0; j < batch.n_tokens; j += std::min(n_batch, batch.n_tokens - j)) {
                const int32_t n_tokens_probe = std::min(n_batch, batch.n_tokens - j);
                const char * dflash_reduce_reason_probe = dflash_verify_plan.reason;
                if (dflash_batch_view_is_reduced_verify(slots, params_base.sampling, params_base.speculative,
                            use_rejection_sampling, ddtree_batch_active, j, n_tokens_probe,
                            dflash_verify_plan.top_k, &dflash_reduce_reason_probe)) {
                    dflash_verify_graph_enabled = true;
                    break;
                }
            }
        }
        if (dflash_server_crash_trace_enabled() && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
            SRV_INF("dflash crash breadcrumb: before verify logits config graph=%d enabled=%d top_k=%d batch_tokens=%d\n",
                    dflash_verify_graph_enabled ? 1 : 0,
                    dflash_verify_plan.enabled ? 1 : 0,
                    dflash_verify_plan.top_k,
                    batch.n_tokens);
        }
        llama_set_dflash_verify_logits(ctx_tgt, dflash_verify_graph_enabled, dflash_verify_plan.top_k);
        if (dflash_server_crash_trace_enabled() && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
            SRV_INF("dflash crash breadcrumb: after verify logits config graph=%d enabled=%d top_k=%d batch_tokens=%d\n",
                    dflash_verify_graph_enabled ? 1 : 0,
                    dflash_verify_plan.enabled ? 1 : 0,
                    dflash_verify_plan.top_k,
                    batch.n_tokens);
        }
        if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
            dflash_log_reduced_verify_decision(
                    dflash_verify_graph_enabled,
                    dflash_verify_plan.enabled,
                    0,
                    batch.n_tokens,
                    dflash_verify_plan.top_k,
                    dflash_verify_plan.reason);
        }

        auto should_flush_dflash_prefill = [&](const server_slot & slot, const llama_batch & view, bool log_decision) -> common_dflash_prefill_span {
            common_dflash_prefill_span no_flush;

            if (!slot.can_speculate()) {
                return no_flush;
            }

            if (params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH) {
                no_flush.should_flush = true;
                return no_flush;
            }

            if (!slot.task) {
                return no_flush;
            }

            bool found_slot_token = false;
            llama_pos batch_pos_min = 0;
            llama_pos batch_pos_max = 0;

            for (int32_t j = 0; j < view.n_tokens; ++j) {
                bool belongs_to_slot = false;
                for (int32_t k = 0; k < view.n_seq_id[j]; ++k) {
                    if (view.seq_id[j][k] == slot.id) {
                        belongs_to_slot = true;
                        break;
                    }
                }

                if (!belongs_to_slot) {
                    continue;
                }

                const llama_pos pos = view.pos[j];
                if (!found_slot_token) {
                    batch_pos_min = pos;
                    batch_pos_max = pos;
                    found_slot_token = true;
                } else {
                    batch_pos_min = std::min(batch_pos_min, pos);
                    batch_pos_max = std::max(batch_pos_max, pos);
                }
            }

            if (!found_slot_token) {
                return no_flush;
            }

            const int32_t prompt_total = slot.task->n_tokens();
            const int32_t cross_ctx = std::max<int32_t>(1, params_base.speculative.dflash_cross_ctx);
            const int32_t capture_from = std::max<int32_t>(0, prompt_total - cross_ctx);
            const int32_t batch_end = (int32_t) batch_pos_max + 1;

            if (batch_end <= capture_from) {
                if (log_decision && dflash_server_profile_enabled(DFLASH_PROFILE_PREFILL)) {
                    SLT_INF(slot,
                            "dflash prefill: skip flush, batch_pos=[%d,%d], batch_end=%d, prompt_total=%d, cross_ctx=%d, capture_from=%d\n",
                            (int) batch_pos_min, (int) batch_pos_max, batch_end, prompt_total, cross_ctx, capture_from);
                }
                return no_flush;
            }

            const int32_t span_begin = std::max(batch_pos_min, capture_from);
            const int32_t span_end   = std::min(batch_end, prompt_total);
            const int     src_offset = span_begin - batch_pos_min;
            const int     n_tokens   = span_end - span_begin;

            if (n_tokens <= 0) {
                return no_flush;
            }

            common_dflash_prefill_span span;
            span.should_flush = true;
            span.capture_begin = span_begin;
            span.capture_end   = span_end;
            span.src_offset   = src_offset;
            span.n_tokens     = n_tokens;

            if (log_decision && dflash_server_profile_enabled(DFLASH_PROFILE_PREFILL)) {
                SLT_INF(slot,
                        "dflash prefill: suffix flush, batch_pos=[%d,%d], batch_end=%d, prompt_total=%d, cross_ctx=%d, capture_from=%d, capture_begin=%d, capture_end=%d, src_offset=%d, n_tokens=%d\n",
                        (int) batch_pos_min, (int) batch_pos_max, batch_end, prompt_total, cross_ctx, capture_from, span.capture_begin, span.capture_end, span.src_offset, span.n_tokens);
            }

            return span;
        };

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            dflash_reduced_verify_ready = false;
            const char * dflash_reduce_reason = dflash_verify_plan.reason;
            bool dflash_reduce_this_view = false;
            if (dflash_verify_plan.enabled) {
                dflash_reduce_this_view =
                    dflash_batch_view_is_reduced_verify(slots, params_base.sampling, params_base.speculative,
                            use_rejection_sampling, ddtree_batch_active, i, n_tokens,
                            dflash_verify_plan.top_k, &dflash_reduce_reason);
            }
            llama_set_dflash_consume_reduced(ctx_tgt, dflash_reduce_this_view);

            bool dflash_capture_needed_for_view = false;

            struct pending_dflash_prefill_flush {
                common_speculative * spec;
                int slot_id;
                common_dflash_prefill_span span;
            };
            std::vector<pending_dflash_prefill_flush> pending_prefill_flushes;
            std::vector<int> dflash_skip_begin_slots;
            auto dflash_mark_skip_begin = [&](int slot_id) {
                if (std::find(dflash_skip_begin_slots.begin(), dflash_skip_begin_slots.end(), slot_id) ==
                        dflash_skip_begin_slots.end()) {
                    dflash_skip_begin_slots.push_back(slot_id);
                }
            };
            auto dflash_should_skip_begin = [&](int slot_id) {
                return std::find(dflash_skip_begin_slots.begin(), dflash_skip_begin_slots.end(), slot_id) !=
                    dflash_skip_begin_slots.end();
            };

            if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                bool dflash_view_has_generating = false;
                std::vector<server_slot *> dflash_slots_in_view;
                for (auto & slot : slots) {
                    if (!slot.can_speculate() || !slot.get_spec()) {
                        continue;
                    }

                    if (!dflash_slot_in_view(slot, batch_view)) {
                        continue;
                    }
                    dflash_slots_in_view.push_back(&slot);

                    if (slot.state == SLOT_STATE_GENERATING) {
                        dflash_capture_needed_for_view = true;
                        dflash_view_has_generating = true;
                    }
                }

                for (auto * slot_ptr : dflash_slots_in_view) {
                    auto & slot = *slot_ptr;
                    if (slot.state == SLOT_STATE_GENERATING) {
                        continue;
                    }

                    if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                        auto span = should_flush_dflash_prefill(slot, batch_view, true);
                        if (span.should_flush) {
                            if (dflash_view_has_generating) {
                                common_speculative_discard_dflash_state(slot.get_spec(), "mixed prompt/generation prefill view");
                                dflash_mark_skip_begin(slot.id);
                                if (dflash_server_profile_enabled(DFLASH_PROFILE_PREFILL)) {
                                    SLT_WRN(slot, "%s", "dflash prefill: skipping suffix flush in mixed prompt/generation view; discarded DFlash state");
                                }
                                continue;
                            }

                            dflash_capture_needed_for_view = true;
                            pending_prefill_flushes.push_back({ slot.get_spec(), slot.id, span });
                            common_speculative_note_prefill_suffix_scheduled(slot.get_spec());
                            llama_dflash_prefill_capture_begin(ctx_tgt, slot.id, span.capture_begin, span.capture_end);
                        }
                    }
                }

                for (auto & slot : slots) {
                    if (slot.can_speculate() && slot.get_spec()) {
                        common_speculative_set_prefill_capture_enabled(slot.get_spec(), dflash_capture_needed_for_view);
                    }
                }

                for (auto & pf : pending_prefill_flushes) {
                    if (dflash_server_profile_enabled(DFLASH_PROFILE_PREFILL)) {
                        SRV_INF("dflash prefill schedule: slot=%d capture_begin=%d capture_end=%d src_offset=%d n_tokens=%d\n",
                                pf.slot_id, pf.span.capture_begin, pf.span.capture_end, pf.span.src_offset, pf.span.n_tokens);
                    }
                }

                if (dflash_server_profile_enabled(DFLASH_PROFILE_PREFILL)) {
                    SRV_INF("dflash prefill capture: view_start=%d n_tokens=%d enabled=%d\n",
                            i, n_tokens, dflash_capture_needed_for_view ? 1 : 0);
                }
            }

            if (dflash_server_profile_enabled(DFLASH_PROFILE_PREFILL | DFLASH_PROFILE_VERIFY) &&
                    params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
                    dflash_view_has_unexpected_prompt_logits(batch_view, slots)) {
                SRV_INF("dflash prompt logits diagnostic: view_start=%d n_tokens=%d has unexpected prompt raw logits outside final sampling row\n",
                        i, n_tokens);
            }

            if (needs_reeval && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                // DFlash: a previous async rollback replay can remain pending when the next
                // cycle falls back to a plain target decode. Sync here so recurrent memory
                // positions and state are current before any target ubatch is scheduled.
                const int64_t t_replay_sync_start = dflash_profile_start();
                llama_tape_replay_sync(ctx_tgt);
                dflash_profile_add(t_replay_sync_total, t_replay_sync_start);
            }

            const int64_t t_verify_start = ggml_time_us();
            const int ret = llama_decode(ctx_tgt, batch_view);
            if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
                    dflash_server_crash_trace_enabled() &&
                    !pending_prefill_flushes.empty()) {
                for (auto & pf : pending_prefill_flushes) {
                    int32_t planned = -1;
                    int32_t written = -1;
                    const bool has_plan = llama_dflash_prefill_capture_info(ctx_tgt, pf.slot_id, &planned, &written);
                    const int64_t prefill_gpu_n = llama_dflash_prefill_gpu_n_tokens(ctx_tgt, pf.slot_id);
                    llama_dflash_set_active_slot(ctx_tgt, pf.slot_id);
                    const int64_t cpu_hidden_n = llama_get_layer_hidden_n_tokens(ctx_tgt, 0);
                    SRV_INF("dflash prefill capture result: slot=%d requested=%d src_offset=%d ret=%d plan=%d planned=%d written=%d prefill_gpu_active=%d prefill_gpu_n=%lld cpu_hidden_n=%lld\n",
                            pf.slot_id,
                            pf.span.n_tokens,
                            pf.span.src_offset,
                            ret,
                            has_plan ? 1 : 0,
                            planned,
                            written,
                            llama_dflash_prefill_gpu_active(ctx_tgt) ? 1 : 0,
                            (long long) prefill_gpu_n,
                            (long long) cpu_hidden_n);
                }
            }
            if (ret == 0 && dflash_reduce_this_view) {
                int32_t * compact_argmax = llama_get_logits_argmax(ctx_tgt);
                const int32_t compact_n = llama_get_logits_argmax_n(ctx_tgt);
                const int compact_k = llama_get_logits_argmax_k(ctx_tgt);
                if (compact_argmax != nullptr &&
                        compact_n >= n_tokens &&
                        compact_k == dflash_verify_plan.top_k) {
                    dflash_reduced_verify_ready = true;
                    dflash_reduced_verify_top_k = dflash_verify_plan.top_k;
                    dflash_reduced_verify_view_start = i;
                } else if (dflash_server_profile_enabled(DFLASH_PROFILE_VERIFY)) {
                    SRV_INF("dflash compact-output mismatch: view_start=%d n_tokens=%d expected_top_k=%d got_ptr=%d got_n=%d got_k=%d\n",
                            i, n_tokens, dflash_verify_plan.top_k,
                            compact_argmax != nullptr ? 1 : 0, compact_n, compact_k);
                }
            }
            const int64_t t_verify_elapsed = ggml_time_us() - t_verify_start;
            t_verify_total += t_verify_elapsed;
            if (dflash_server_profile_enabled(DFLASH_PROFILE_VERIFY)) {
                SRV_INF("  verify ubatch: %d tok, %.1fms (%.2fms/tok)\n",
                        n_tokens, t_verify_elapsed / 1e3, t_verify_elapsed / 1e3 / std::max(1, n_tokens));
            }

            metrics.on_decoded(slots);

            if (ret != 0) {
                if (!pending_prefill_flushes.empty()) {
                    llama_dflash_prefill_capture_end(ctx_tgt);
                }

                {
                    std::string err;

                    if (n_batch == 1 && ret == 1) {
                        // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                        //       need to remove the tokens from the current batch too
                        err = "Context size has been exceeded.";
                    }

                    if (ret == -1) {
                        err = "Invalid input batch.";
                    }

                    if (ret < -1) {
                        // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                        err = "Compute error.";
                    }

                    // TODO: handle ret == 2 (abort) when we start aborting

                    if (!err.empty()) {
                        SRV_ERR("%s i = %d, n_batch = %d, ret = %d\n", err.c_str(), i, n_batch, ret);

                        for (auto & slot : slots) {
                            if (slot.is_processing()) {
                                send_error(slot, err);
                                slot.release();

                                // note: it's complicated to keep track of how much of the current batch has been
                                //       processed before the error occurred, so we simply clear the entire context
                                slot.prompt_clear(false);
                            }
                        }

                        break;
                    }
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                if (!try_clear_idle_slots()) {
                    n_batch /= 2;
                }

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            //       for now, always re-evaluate for simplicity
            //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
            //
            // | spec type   | need re-eval |
            // | ---         | ---          |
            // | draft model | no           | because the draft model does not use embeddings from the target
            // | MTP (std)   | yes          |
            // | MTP Gemma4  | no           | because the KV cache is shared
            // | Eagle3      | yes          |
            // | DFlash      | yes          | https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4405406982
            //
            // note: this logic is now moved in `common_speculative_process()`
            //       keeping the sketch here until for a bit, until the logic is finalized
            //
            //if (ctx_dft) {
            //    // TODO: update as needed for MTP, Eagle3, etc.
            //    const bool need_tgt_embd = false;

            //    if (need_tgt_embd) {
            //        llama_synchronize(ctx_tgt);
            //    }

            //    // the logic here varies depending on the speculative decoding method
            //    //  - some draft contexts require embeddings from the target context, others don't
            //    //  - some draft contexts involve an encoder step to transform the target embeddings to draft embeddings
            //    // TODO: extract this in a function ?
            //    {
            //        // TODO: hook the embeddings from the last target batch here
            //        if (llama_model_has_encoder(model_dft.get())) {
            //            //llama_encode(ctx_dft, ...);

            //            GGML_ABORT("not implemented yet\n");
            //        }

            //        const int ret = llama_decode(ctx_dft.get(), batch_view);

            //        if (ret != 0) {
            //            SRV_ERR("failed to decode draft batch, ret = %d\n", ret);

            //            // TODO: handle error
            //            break;
            //        }
            //    }
            //}
            if (!common_speculative_process(spec.get(), batch_view)) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                break;
            }

            // move the head of the batch forward with the number of tokens we just processed
            i_next = i + n_tokens;

            // on successful decode, restore the original batch size
            n_batch = llama_n_batch(ctx_tgt);

            // DFlash: flush captured hidden states into the ring buffer before
            // the next llama_decode resets the capture buffer.
            for (auto & pf : pending_prefill_flushes) {
                const int written = common_speculative_flush_prefill(
                        pf.spec, pf.span.src_offset, pf.span.n_tokens, pf.span.capture_end);
                if (written != pf.span.n_tokens) {
                    SRV_ERR("dflash prefill flush mismatch: slot=%d requested=%d written=%d src_offset=%d; disabling DFlash drafting until fresh hiddens are available\n",
                            pf.slot_id, pf.span.n_tokens, written, pf.span.src_offset);

                    common_speculative_discard_dflash_state(pf.spec, "prefill flush mismatch");
                    dflash_mark_skip_begin(pf.slot_id);
                    common_speculative_set_prefill_capture_enabled(pf.spec, false);
                }
            }

            if (!pending_prefill_flushes.empty()) {
                llama_dflash_prefill_capture_end(ctx_tgt);
            }

            // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                    std::vector<server_slot *> children;
                    for (auto & other : slots) {
                        if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                            children.push_back(&other);
                        }
                    }

                    // all children slots should already launched by launch_slots_with_parent_task()
                    // copy state to the child slots
                    for (auto & child : children) {
                        SLT_INF(slot, " - copying state to child %d\n", child->id);

                        GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                        slot.copy_state_to(*child);
                        child->state = SLOT_STATE_DONE_PROMPT;
                    }
                }
            }

            for (auto & slot : slots) {
                // optionally send prompt processing progress
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->params.stream && slot.task->params.return_progress) {
                        send_partial_response(slot, {}, true);
                    }
                }

                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                        send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    GGML_ASSERT(slot.task->need_sampling());

                    // prompt evaluated for next-token prediction
                    slot.state = SLOT_STATE_GENERATING;

                    if (slot.can_speculate()) {
                        bool begin_speculative_state = true;
                        if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                            llama_dflash_set_active_slot(ctx_tgt, slot.id);
                            if (dflash_should_skip_begin(slot.id)) {
                                common_speculative_set_prefill_capture_enabled(slot.get_spec(), false);
                                begin_speculative_state = false;
                                if (dflash_server_profile_enabled(DFLASH_PROFILE_PREFILL)) {
                                    SLT_WRN(slot, "%s", "dflash prefill: skipped begin after unsafe prefill capture");
                                }
                            } else {
                                common_speculative_set_prefill_capture_enabled(slot.get_spec(), true);
                            }
                        }
                        if (begin_speculative_state) {
                            common_speculative_begin(slot.get_spec(), slot.id, slot.prompt.tokens.get_text_tokens());
                        }
                    }
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                if (slot.can_speculate() && (!slot.spec_draft.empty() || slot.has_draft_tree)) {
                    continue; // sample using speculative decoding
                }

                const int tok_idx = slot.i_batch - i;

                llama_token id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);

                slot.i_batch = -1;

                if (loop_guard_accept_enabled(slot)) {
                    const auto accept_info = common_sampler_accept_with_info(slot.smpl.get(), id, true);
                    (void) handle_loop_guard_accept(slot, accept_info);
                } else {
                    common_sampler_accept(slot.smpl.get(), id, true);
                }

                // update DFlash hidden state ring buffer with the decoded token's hidden states.
                // Skip on the first sample after prompt: common_speculative_begin() above already
                // populated the ring with all prefill hiddens. The capture buffer at this point
                // still holds prefill hiddens (no new decode happened), so ring_write(1) here would
                // append a stale duplicate at the position that should later hold `id`'s hidden —
                // silently corrupting the drafter's cross-attention context on every subsequent
                // verify. Fires correctly on the fallback non-spec path during generation
                // (draft too small → single-token decode), where slot.sampled was just decoded.
                if (slot.can_speculate() && slot.n_decoded > 0) {
                    if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                        llama_dflash_set_active_slot(ctx_tgt, slot.id);
                    }
                    llama_tokens batch_tokens = { id };
                    const bool dflash_defer_kv_cache_update =
                        params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
                        slot.dm_adaptive && slot.adaptive_n_max == 0;
                    if (dflash_defer_kv_cache_update) {
                        common_speculative_update_logits_deferred_dflash_kv(slot.get_spec(), ctx_tgt, batch_tokens, 1);
                    } else {
                        common_speculative_update_logits(slot.get_spec(), ctx_tgt, batch_tokens, 1);
                    }
                }

                // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
                const int64_t t_current = ggml_time_us();

                slot.n_decoded += 1;

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = t_current;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

                if (slot.task->params.sampling.n_probs > 0) {
                    populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
                }

                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    continue;
                }

                slot.print_timings_tg();
            }

            if (ddtree_batch_active) {
                llama_clear_tree_mask(ctx_tgt);
                const bool has_pending_tree_slot = std::any_of(slots.begin(), slots.end(), [](const server_slot & s) {
                    return s.state == SLOT_STATE_GENERATING && s.has_draft_tree && !s.spec_i_batch.empty();
                });
                if (!has_pending_tree_slot) {
                    llama_clear_tree_parent_ids(ctx_tgt);
                }
            }

            // speculative decoding - main model sample and accept
            const int64_t t_accept_start = ggml_time_us();
            struct dflash_flat_accept_prefetch {
                bool ready = false;
                llama_tokens ids;
                bool speculative_has_bonus = true;
                int n_accepted_draft = 0;
                int n_hidden_keep = 0;
                int64_t sample_us = 0;
                int64_t update_us = 0;
            };
            const bool dflash_multislot_flat_accept_barrier =
                params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
                n_slots_drafted > 1 &&
                !ddtree_batch_active &&
                std::none_of(slots.begin(), slots.end(), [](const server_slot & slot) {
                    return slot.state == SLOT_STATE_GENERATING &&
                        slot.can_speculate() &&
                        slot.has_draft_tree;
                });
            std::vector<dflash_flat_accept_prefetch> dflash_flat_accept_prefetches(slots.size());

            if (dflash_multislot_flat_accept_barrier) {
                for (auto & slot : slots) {
                    if (slot.id < 0 || slot.id >= (int) dflash_flat_accept_prefetches.size()) {
                        continue;
                    }
                    if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.spec_draft.empty()) {
                        continue;
                    }

                    auto & prefetched = dflash_flat_accept_prefetches[slot.id];
                    const int64_t t_sample_start = ggml_time_us();
                    const bool grammar_active_for_accept = common_sampler_has_active_grammar(slot.smpl.get());
                    const bool use_rejection = params_base.speculative.sample_temp > 0.0f
                                            && params_base.sampling.temp > 0.0f
                                            && !slot.draft_log_probs.empty()
                                            && !grammar_active_for_accept;
                    const auto on_accept = make_loop_guard_accept_callback(slot);
                    if (use_rejection) {
                        prefetched.ids = speculative_reject_sample(slot.smpl.get(), ctx_tgt, slot.spec_draft,
                            slot.spec_i_batch, slot.draft_log_probs, params_base.sampling.temp, slot.reject_rng,
                            slot.n_reject_exact, slot.n_reject_prob_accept, slot.n_reject_reject, slot.n_reject_no_prob,
                            on_accept);
                    } else {
                        if (dflash_reduced_verify_ready && dflash_verify_plan.enabled) {
                            std::vector<int> reduced_idxs = slot.spec_i_batch;
                            for (int & idx : reduced_idxs) {
                                idx -= dflash_reduced_verify_view_start;
                            }
                            prefetched.ids = dflash_sample_reduced_verify(slot.smpl.get(), ctx_tgt, reduced_idxs,
                                    slot.spec_draft, dflash_reduced_verify_top_k, on_accept);
                            if (prefetched.ids.empty()) {
                                GGML_ABORT("DFlash reduced verifier output missing; falling back is unsafe because raw logits were not copied\n");
                            }
                        } else {
                            prefetched.ids = common_sampler_sample_and_accept_n(slot.smpl.get(), ctx_tgt, slot.spec_i_batch, slot.spec_draft, false, on_accept);
                        }
                    }
                    prefetched.sample_us = ggml_time_us() - t_sample_start;

                    const int32_t remaining = slot.remaining_generation_budget(params_base);
                    if (remaining != -1 && (int32_t) prefetched.ids.size() > remaining) {
                        SLT_WRN(slot, "accepted draft tokens exceed remaining budget (%d > %d), truncating\n",
                                (int) prefetched.ids.size(), remaining);
                        prefetched.ids.resize((size_t) std::max<int32_t>(0, remaining));
                    }
                    GGML_ASSERT(slot.remaining_generation_budget(params_base) == -1 ||
                            (int32_t) prefetched.ids.size() <= slot.remaining_generation_budget(params_base));

                    prefetched.speculative_has_bonus =
                        speculative_flat_result_has_bonus(prefetched.ids, slot.spec_draft, slot.smpl.get());
                    prefetched.n_accepted_draft =
                        std::max(0, (int) prefetched.ids.size() - (prefetched.speculative_has_bonus ? 1 : 0));
                    prefetched.n_hidden_keep = prefetched.ids.empty() ? 0 : prefetched.n_accepted_draft + 1;
                    prefetched.ready = true;
                }

                for (auto & slot : slots) {
                    if (slot.id < 0 || slot.id >= (int) dflash_flat_accept_prefetches.size()) {
                        continue;
                    }
                    auto & prefetched = dflash_flat_accept_prefetches[slot.id];
                    if (!prefetched.ready || prefetched.ids.empty()) {
                        continue;
                    }

                    const int64_t t_update_start = ggml_time_us();
                    llama_dflash_set_active_slot(ctx_tgt, slot.id);
                    llama_tokens batch_tokens;
                    batch_tokens.push_back(slot.sampled);
                    batch_tokens.insert(batch_tokens.end(), slot.spec_draft.begin(), slot.spec_draft.end());
                    common_speculative_update_logits_deferred_dflash_kv(slot.get_spec(), ctx_tgt, batch_tokens, prefetched.n_hidden_keep);
                    prefetched.update_us = ggml_time_us() - t_update_start;
                }
            }

            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || (slot.spec_draft.empty() && !slot.has_draft_tree)) {
                    continue;
                }

                const bool is_draft_tree = slot.has_draft_tree;
                const size_t n_draft = is_draft_tree ? (size_t) slot.draft_tree.n_nodes : slot.spec_draft.size();
                const bool had_dflash_padding = !slot.spec_pad_i_batch.empty();
                const int dflash_verify_rows = (int) slot.spec_i_batch.size() + (int) slot.spec_pad_i_batch.size();
                const int dflash_useful_rows = (int) slot.spec_i_batch.size();
                const int dflash_pad_rows = (int) slot.spec_pad_i_batch.size();
                const bool profile_dflash_accept = dflash_server_profile_enabled(DFLASH_PROFILE_SUMMARY | DFLASH_PROFILE_COPY) &&
                        params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH;
                const int64_t profile_accept_start = profile_dflash_accept ? ggml_time_us() : 0;
                int64_t profile_accept_phase_start = profile_accept_start;
                int64_t profile_accept_sample_us = 0;
                int64_t profile_accept_update_us = 0;
                int64_t profile_accept_book_us = 0;
                int64_t profile_accept_rollback_us = 0;
                int64_t profile_accept_emit_us = 0;
                auto profile_accept_lap = [&](int64_t & dst) {
                    if (!profile_dflash_accept) {
                        return;
                    }
                    const int64_t now = ggml_time_us();
                    dst += now - profile_accept_phase_start;
                    profile_accept_phase_start = now;
                };

                // MTP-only speculative accept — hard-isolated upstream lifecycle
                // Upstream contract: sample_and_accept_n() → n_rollback → on rollback
                // ckpt-restore+continue (skip accept), on commit accept+insert[0..N-2]+sampled=N-1.
                // This block replaces the DFlash-era generic accept path for MTP because
                // that path has speculative_has_bonus/n_prompt_insert bugs and calls
                // common_speculative_accept() before the rollback decision.
                const bool use_mtp_spec_accept =
                    params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP) &&
                    params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH &&
                    !is_draft_tree;

                if (use_mtp_spec_accept) {
                    GGML_ASSERT(n_draft > 0);
                    GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);

                    common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));
                    const server_loop_guard loop_guard_save = slot.loop_guard;
                    const int32_t loop_guard_interventions_save = slot.loop_guard_interventions;
                    const bool loop_guard_triggered_save = slot.loop_guard_triggered;
                    const std::string loop_guard_action_save = slot.loop_guard_action;
                    const std::string loop_guard_reason_save = slot.loop_guard_reason;
                    const int32_t reasoning_output_tokens_save = slot.reasoning_output_tokens;
                    const int32_t visible_output_tokens_save = slot.visible_output_tokens;
                    const stop_type stop_save = slot.stop;
                    const std::string stop_detail_save = slot.stop_detail;
                    const bool has_next_token_save = slot.has_next_token;

                    const auto on_accept = make_loop_guard_accept_callback(slot);
                    auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft, false, on_accept);
                    slot.spec_i_batch.clear();

                    GGML_ASSERT(accepted.size() >= 1);

                    const uint16_t n_accepted_draft = (uint16_t) (accepted.size() - 1);
                    const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                    const bool use_ckpt_tgt =
                        ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                       (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                    // check for partial draft acceptance
                    if (n_rollback > 0) {
                        if (use_ckpt_tgt) {
                            SLT_INF(slot, "accepted %2d/%2zu draft tokens (restore checkpoint)\n", n_accepted_draft, slot.spec_draft.size());

                            // partial acceptance is not supported by the context -> truncate the draft and restore the state
                            slot.spec_draft = std::move(accepted);

                            const auto & ckpt = slot.spec_ckpt;

                            SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                            {
                                ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

                                common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
                            }

                            if (slot.ctx_dft) {
                                ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

                                common_context_seq_rm(slot.ctx_dft, slot.id, ckpt.pos_max + 1, -1);
                            }

                            slot.prompt.tokens.keep_first(ckpt.n_tokens);
                            slot.smpl = std::move(smpl_save);
                            slot.loop_guard = loop_guard_save;
                            slot.loop_guard_interventions = loop_guard_interventions_save;
                            slot.loop_guard_triggered = loop_guard_triggered_save;
                            slot.loop_guard_action = loop_guard_action_save;
                            slot.loop_guard_reason = loop_guard_reason_save;
                            slot.reasoning_output_tokens = reasoning_output_tokens_save;
                            slot.visible_output_tokens = visible_output_tokens_save;
                            slot.stop = stop_save;
                            slot.stop_detail = stop_detail_save;
                            slot.has_next_token = has_next_token_save;

                            continue;
                        }
                    }

                    SLT_INF(slot, "accepted %2d/%2zu draft tokens\n", n_accepted_draft, n_draft);

                    common_speculative_accept(slot.get_spec(), slot.id, n_accepted_draft);

                    slot.spec_draft = std::move(accepted);

                    const int64_t t_current = ggml_time_us();

                    const auto ids = std::move(slot.spec_draft);

                    slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                    // update how many tokens out of those tested were accepted
                    slot.n_draft_accepted += ids.size() - 1;

                    if (slot.dm_adaptive) {
                        slot.observe_profit_acceptance((int) n_draft, (int) ids.size() - 1);
                        slot.profit_pending = true;
                        if (slot.profit_pending_requested_n_max <= 0) {
                            slot.profit_pending_requested_n_max = (int32_t) n_draft;
                        }
                        slot.profit_pending_n_draft = (int32_t) n_draft;
                        slot.profit_pending_n_accepted = (int32_t) ids.size() - 1;
                        slot.profit_pending_tree = false;
                    }

                    // add accepted tokens to the prompt
                    slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
                    slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

                    slot.sampled = ids.back(); // last accepted token
                    SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

                    common_context_seq_rm(slot.ctx_tgt, slot.id, slot.prompt.tokens.pos_next(), -1);
                    if (slot.ctx_dft) {
                        common_context_seq_rm(slot.ctx_dft, slot.id, slot.prompt.tokens.pos_next(), -1);
                    }

                    for (size_t i = 0; i < ids.size(); ++i) {
                        completion_token_output result;

                        result.tok          = ids[i];
                        result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                        result.prob         = 1.0f; // set later

                        // TODO: set result.probs

                        slot.n_decoded += 1;

                        if (!process_token(result, slot)) {
                            slot.print_timings();
                            send_final_response(slot);
                            metrics.on_prediction(slot);
                            slot.release();

                            break;
                        }
                    }

                    slot.print_timings_tg();

                    SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft, slot.prompt.n_tokens());
                    continue;
                }

                llama_tokens ids;
                int  tree_commit_n = 0;
                bool tree_accepted_on_main_path = true;
                bool speculative_has_bonus = true;
                int  n_accepted_draft = 0;
                int  n_hidden_keep = 0;
                std::vector<int> capture_indices;

                if (dflash_multislot_flat_accept_barrier && !is_draft_tree) {
                    const auto & prefetched = dflash_flat_accept_prefetches[slot.id];
                    GGML_ASSERT(prefetched.ready);
                    ids = prefetched.ids;
                    speculative_has_bonus = prefetched.speculative_has_bonus;
                    n_accepted_draft = prefetched.n_accepted_draft;
                    n_hidden_keep = prefetched.n_hidden_keep;
                    profile_accept_sample_us = prefetched.sample_us;
                    profile_accept_update_us = prefetched.update_us;
                    profile_accept_phase_start = ggml_time_us();
                } else if (is_draft_tree) {
                    const bool use_tree_rejection = params_base.speculative.sample_temp > 0.0f
                                                   && params_base.sampling.temp > 0.0f
                                                   && !slot.draft_log_probs.empty();
                    const float inv_temp = (params_base.sampling.temp > 0.0f) ? (1.0f / params_base.sampling.temp) : 1.0f;
                    const int32_t max_batch_idx = slot.spec_i_batch.empty() ? -1 :
                        *std::max_element(slot.spec_i_batch.begin(), slot.spec_i_batch.end());
                    log_norm_cache ln_cache(ctx_tgt, params_base.sampling.temp, max_batch_idx);
                    const auto on_accept = make_loop_guard_accept_callback(slot);
                    int current = 0;
                    while (true) {
                        const llama_token id = common_sampler_sample(slot.smpl.get(), ctx_tgt, slot.spec_i_batch[current]);
                        ids.push_back(id);

                        auto it = slot.draft_tree.child_maps[current].find(id);
                        if (it != slot.draft_tree.child_maps[current].end()) {
                            const int accepted_child = it->second;
                            bool keep_accepting = true;
                            if (on_accept) {
                                const auto info = common_sampler_accept_with_info(slot.smpl.get(), id, true);
                                keep_accepting = on_accept(info);
                            } else {
                                common_sampler_accept(slot.smpl.get(), id, true);
                            }
                            current = accepted_child;
                            tree_commit_n = current;
                            if (current > slot.draft_tree.main_path_len) {
                                tree_accepted_on_main_path = false;
                            }
                            slot.n_reject_exact++;
                            if (!keep_accepting || common_sampler_blocks_speculative(slot.smpl.get())) {
                                speculative_has_bonus = false;
                                break;
                            }
                            continue;
                        }

                        if (use_tree_rejection) {
                            const auto & child_map = slot.draft_tree.child_maps[current];
                            if (!child_map.empty()) {
                                const float log_norm = ln_cache.get(slot.spec_i_batch[current]);
                                const float * logits_ptr = (log_norm > -FLT_MAX) ? llama_get_logits_ith(ctx_tgt, slot.spec_i_batch[current]) : nullptr;
                                int accepted_child = -1;
                                for (const auto & kv : child_map) {
                                    const llama_token tok = kv.first;
                                    const int child_idx = kv.second;
                                    if (child_idx < 1 || (size_t)child_idx > slot.draft_log_probs.size()) {
                                        continue;
                                    }
                                    const float q_log = slot.draft_log_probs[child_idx - 1];
                                    if (!std::isfinite(q_log) || q_log > 0.0f) {
                                        continue;
                                    }
                                    const float log_prob = logits_ptr ? logits_ptr[tok] * inv_temp - log_norm : -FLT_MAX;
                                    const float accept_prob = (log_prob > -FLT_MAX) ? std::min(1.0f, expf(log_prob - q_log)) : 0.0f;
                                    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                                    if (uniform(slot.reject_rng) < accept_prob) {
                                        accepted_child = child_idx;
                                        break;
                                    }
                                }
                                if (accepted_child > 0) {
                                    const llama_token accepted_token = slot.draft_tree.tokens[accepted_child - 1];
                                    bool keep_accepting = true;
                                    if (on_accept) {
                                        const auto info = common_sampler_accept_with_info(slot.smpl.get(), accepted_token, true);
                                        keep_accepting = on_accept(info);
                                    } else {
                                        common_sampler_accept(slot.smpl.get(), accepted_token, true);
                                    }
                                    ids.back() = accepted_token;
                                    current = accepted_child;
                                    tree_commit_n = accepted_child;
                                    if (accepted_child > slot.draft_tree.main_path_len) {
                                        tree_accepted_on_main_path = false;
                                    }
                                    slot.n_reject_prob_accept++;
                                    if (!keep_accepting || common_sampler_blocks_speculative(slot.smpl.get())) {
                                        speculative_has_bonus = false;
                                        break;
                                    }
                                    continue;
                                }
                            }
                            slot.n_reject_reject++;
                        }
                        if (on_accept) {
                            const auto info = common_sampler_accept_with_info(slot.smpl.get(), id, true);
                            (void) on_accept(info);
                        } else {
                            common_sampler_accept(slot.smpl.get(), id, true);
                        }
                        break;
                    }

                    if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                        llama_dflash_set_active_slot(ctx_tgt, slot.id);
                    }

                    std::vector<int> accepted_path;
                    for (int node = tree_commit_n; node > 0; node = slot.draft_tree.parents[node]) {
                        accepted_path.push_back(node);
                    }
                    std::reverse(accepted_path.begin(), accepted_path.end());

                    capture_indices.clear();
                    capture_indices.push_back(0);
                    for (int node : accepted_path) {
                        capture_indices.push_back(node);
                    }
                } else {
                    const bool grammar_active_for_accept = common_sampler_has_active_grammar(slot.smpl.get());
                    const bool use_rejection = params_base.speculative.sample_temp > 0.0f
                                            && params_base.sampling.temp > 0.0f
                                            && !slot.draft_log_probs.empty()
                                            && !grammar_active_for_accept;
                    const auto on_accept = make_loop_guard_accept_callback(slot);
                    if (use_rejection) {
                        ids = speculative_reject_sample(slot.smpl.get(), ctx_tgt, slot.spec_draft,
                            slot.spec_i_batch, slot.draft_log_probs, params_base.sampling.temp, slot.reject_rng,
                            slot.n_reject_exact, slot.n_reject_prob_accept, slot.n_reject_reject, slot.n_reject_no_prob,
                            on_accept);
                    } else {
                        if (dflash_reduced_verify_ready && dflash_verify_plan.enabled) {
                            std::vector<int> reduced_idxs = slot.spec_i_batch;
                            for (int & idx : reduced_idxs) {
                                idx -= dflash_reduced_verify_view_start;
                            }
                            ids = dflash_sample_reduced_verify(slot.smpl.get(), ctx_tgt, reduced_idxs,
                                    slot.spec_draft, dflash_reduced_verify_top_k, on_accept);
                            if (ids.empty()) {
                                GGML_ABORT("DFlash reduced verifier output missing; falling back is unsafe because raw logits were not copied\n");
                            }
                        } else {
                            ids = common_sampler_sample_and_accept_n(slot.smpl.get(), ctx_tgt, slot.spec_i_batch, slot.spec_draft, false, on_accept);
                        }
                    }
                    profile_accept_lap(profile_accept_sample_us);

                    const int32_t remaining = slot.remaining_generation_budget(params_base);
                    if (remaining != -1 && (int32_t) ids.size() > remaining) {
                        SLT_WRN(slot, "accepted draft tokens exceed remaining budget (%d > %d), truncating\n",
                                (int) ids.size(), remaining);
                        ids.resize((size_t) std::max<int32_t>(0, remaining));
                    }
                    GGML_ASSERT(slot.remaining_generation_budget(params_base) == -1 ||
                            (int32_t) ids.size() <= slot.remaining_generation_budget(params_base));

                    speculative_has_bonus = speculative_flat_result_has_bonus(ids, slot.spec_draft, slot.smpl.get());
                    n_accepted_draft = std::max(0, (int) ids.size() - (speculative_has_bonus ? 1 : 0));
                    n_hidden_keep = ids.empty() ? 0 : n_accepted_draft + 1;

                    if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                        llama_dflash_set_active_slot(ctx_tgt, slot.id);
                    }
                    llama_tokens batch_tokens;
                    batch_tokens.push_back(slot.sampled);
                    batch_tokens.insert(batch_tokens.end(), slot.spec_draft.begin(), slot.spec_draft.end());
                    common_speculative_update_logits_deferred_dflash_kv(slot.get_spec(), ctx_tgt, batch_tokens, n_hidden_keep);
                    profile_accept_lap(profile_accept_update_us);
                }

                GGML_ASSERT(slot.remaining_generation_budget(params_base) == -1 ||
                        (int32_t) ids.size() <= slot.remaining_generation_budget(params_base));

                const int64_t t_current = ggml_time_us();

                if (ids.empty()) {
                    slot.spec_i_batch.clear();
                    slot.spec_pad_i_batch.clear();
                    slot.spec_draft.clear();
                    slot.draft_log_probs.clear();
                    slot.stop = STOP_TYPE_LIMIT;
                    slot.stop_detail = "token_limit";
                    slot.has_next_token = false;
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();
                    continue;
                }

                if (is_draft_tree) {
                    n_accepted_draft = std::max(0, (int) ids.size() - (speculative_has_bonus ? 1 : 0));
                    n_hidden_keep = n_accepted_draft + 1;
                }

                slot.n_draft_accepted += n_accepted_draft;
                if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                    slot.note_dflash_cycle((int) n_draft, (int) n_draft, n_accepted_draft, slot.n_pos_before_draft);
                    dflash_cycle_slot = &slot;
                    dflash_cycle_n_draft = (int) n_draft;
                    dflash_cycle_n_accept = n_accepted_draft;
                    dflash_cycle_verify_rows = dflash_verify_rows;
                    dflash_cycle_useful_rows = dflash_useful_rows;
                    dflash_cycle_pad_rows = dflash_pad_rows;
                    dflash_cycle_reduced_verify = dflash_reduced_verify_ready && dflash_verify_plan.enabled;
                    dflash_cycle_reduced_top_k = dflash_reduced_verify_top_k;
                    dflash_cycle_pos = slot.n_pos_before_draft;
                }

                if (slot.dm_adaptive && !is_draft_tree) {
                    const int n_accepted = n_accepted_draft;
                    slot.observe_profit_acceptance((int) n_draft, n_accepted);
                    slot.profit_pending = true;
                    if (slot.profit_pending_requested_n_max <= 0) {
                        slot.profit_pending_requested_n_max = (int32_t) n_draft;
                    }
                    slot.profit_pending_n_draft = (int32_t) n_draft;
                    slot.profit_pending_n_accepted = n_accepted;
                    slot.profit_pending_tree = false;

                    if (server_adaptive_dm_uses_fringe_controller(slot.dm_controller)) {
                        const int32_t prev_adaptive_n_max = slot.adaptive_n_max;

                        {
                            server_slot::fringe_entry & e = slot.fringe_ring[slot.fringe_ring_idx];
                            e.n_accepted = (int16_t) n_accepted;
                            e.n_draft    = (int16_t) n_draft;
                            e.epoch      = slot.fringe_epoch;
                            slot.fringe_ring_idx = (slot.fringe_ring_idx + 1) % server_slot::FRINGE_WINDOW;
                            if (slot.fringe_ring_count < server_slot::FRINGE_WINDOW) {
                                slot.fringe_ring_count++;
                            }
                        }
                        const int reached_positions = std::min<int>(n_draft, server_slot::FRINGE_REACH_POSITIONS);
                        for (int pos = 0; pos < reached_positions; pos++) {
                            slot.fringe_epoch_reached[pos]++;
                        }
                        const int accepted_positions = std::min<int>(n_accepted, server_slot::FRINGE_REACH_POSITIONS);
                        for (int pos = 0; pos < accepted_positions; pos++) {
                            slot.fringe_epoch_accepted[pos]++;
                        }

                        const int base_n_max = common_speculative_n_max(slot.get_spec(), slot.task->params.speculative);
                        const int fringe_n_max = std::max(0, slot.adaptive_n_max > 0 ? slot.adaptive_n_max : base_n_max);

                        const auto decision = slot.fringe_decision_at_window(fringe_n_max, slot.dm_fringe_window, slot.dm_min_reach);
                        const float decision_fringe = decision.fringe;
                        const int decision_reached = decision.reached;
                        const int decision_accepted = decision.accepted;
                        slot.rolling_fringe = decision_fringe;

                        int target_n_max;
                        if (decision_fringe < slot.dm_fringe_min) {
                            target_n_max = 0;
                        } else if (decision_fringe >= slot.dm_fringe_max) {
                            target_n_max = base_n_max;
                        } else {
                            float t = (decision_fringe - slot.dm_fringe_min)
                                    / (slot.dm_fringe_max - slot.dm_fringe_min);
                            target_n_max = std::max(2, (int)(base_n_max * t + 0.5f));
                            target_n_max = std::min(target_n_max, base_n_max);
                        }

                        if (target_n_max == 0 && prev_adaptive_n_max > 0) {
                            slot.off_dwell++;
                            if (slot.off_dwell < slot.dm_off_dwell) {
                                target_n_max = prev_adaptive_n_max;
                            } else {
                                slot.off_dwell = 0;
                            }
                        } else {
                            slot.off_dwell = 0;
                        }

                        if (target_n_max > 0 && prev_adaptive_n_max > 0) {
                            int delta = target_n_max - prev_adaptive_n_max;
                            delta = std::clamp(delta, -2, 2);
                            target_n_max = prev_adaptive_n_max + delta;
                        }

                        if (target_n_max > prev_adaptive_n_max && target_n_max > 0 && prev_adaptive_n_max > 0) {
                            const int win_start = std::max(0, target_n_max - slot.dm_fringe_window);
                            int guard_reached = 0;
                            int guard_accepted = 0;
                            for (int pos = win_start; pos < target_n_max; pos++) {
                                if (pos < server_slot::FRINGE_REACH_POSITIONS) {
                                    guard_reached += slot.fringe_epoch_reached[pos];
                                    guard_accepted += slot.fringe_epoch_accepted[pos];
                                }
                            }
                            const float guard_fringe = guard_reached > 0 ? (float)guard_accepted / guard_reached : 0.0f;
                            const float required_fringe = server_adaptive_dm_required_fringe_for_n_max(
                                    target_n_max, base_n_max, slot.dm_fringe_min, slot.dm_fringe_max);
                            if (guard_reached < slot.dm_min_reach || guard_fringe < required_fringe) {
                                target_n_max = prev_adaptive_n_max;
                            }
                        }

                        const bool demoting = prev_adaptive_n_max > 0 && target_n_max < prev_adaptive_n_max;
                        const bool going_off = prev_adaptive_n_max > 0 && target_n_max == 0;
                        if (demoting) {
                            slot.fringe_epoch++;
                            slot.reset_fringe_epoch_reached();
                        }
                        if (going_off) {
                            slot.off_dwell = 0;
                        }

                        slot.adaptive_n_max = target_n_max;
                        slot.adaptive_probe_counter = 0;
                        if (slot.adaptive_n_max != prev_adaptive_n_max) {
                            SLT_INF(slot, "adaptive dm: fringe=%.2f (win=%d, rch %d, acc %d) n_max=%d->%d\n",
                                    slot.rolling_fringe, fringe_n_max, decision_reached, decision_accepted,
                                    prev_adaptive_n_max, slot.adaptive_n_max);
                        }
                    }
                }

                // Plain MTP is handled by the dedicated block above (use_mtp_spec_accept).
                // This path is only reached for DFlash, tree, or other speculative types.
                {
                    common_speculative_accept(slot.get_spec(), n_accepted_draft);
                }

                if (is_draft_tree) {
                    slot.prompt.tokens.keep_first(slot.n_tokens_before_draft + 1);
                } else {
                    slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
                }

                const size_t n_prompt_insert = std::min(ids.size(), (size_t) n_accepted_draft);
                slot.prompt.tokens.insert(llama_tokens(ids.begin(), ids.begin() + n_prompt_insert));
                profile_accept_lap(profile_accept_book_us);

                if (slot.has_draft_backup) {
                    const llama_seq_id seq_backup = slot.seq_id_backup;
                    GGML_ASSERT(seq_backup >= 0);
                    const bool all_accepted_flat = (n_accepted_draft == (int) n_draft) && !had_dflash_padding;

                    if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH && is_draft_tree) {
                        llama_dflash_set_active_slot(ctx_tgt, slot.id);

                        const int commit_depth = tree_commit_n > 0 ? slot.draft_tree.depths[tree_commit_n - 1] : 0;
                        llama_tree_rollback_seq(ctx_tgt, slot.id, seq_backup, tree_commit_n, slot.draft_tree.parents.data(),
                                slot.n_pos_before_draft + commit_depth);
                        llama_clear_tree_parent_ids(ctx_tgt);

                        auto * mem = llama_get_memory(ctx_tgt);
                        const int n_branches = slot.draft_tree.n_nodes - slot.draft_tree.main_path_len;
                        const bool all_accepted_tree = commit_depth == slot.draft_tree.main_path_len && tree_accepted_on_main_path;

                        if (n_branches == 0 && tree_accepted_on_main_path) {
                            if (!all_accepted_tree) {
                                llama_memory_seq_rm(mem, slot.id, slot.n_pos_before_draft + commit_depth + 1, -1);
                            }
                            llama_memory_seq_rm(mem, seq_backup, -1, -1);
                        } else if (tree_accepted_on_main_path) {
                            const llama_pos n_before = slot.n_pos_before_draft;
                            std::set<int> branch_depths;
                            for (int i = slot.draft_tree.main_path_len; i < slot.draft_tree.n_nodes; ++i) {
                                int d = slot.draft_tree.depths[i];
                                if (d >= 1 && d <= commit_depth) {
                                    branch_depths.insert(d);
                                }
                            }

                            for (int d : branch_depths) {
                                llama_pos pos = n_before + d;
                                uint32_t cell_indices[64];
                                int n_cells = llama_memory_cells_at_pos(mem, slot.id, pos, cell_indices, 64);
                                if (n_cells > 1) {
                                    std::sort(cell_indices, cell_indices + std::min(n_cells, 64));
                                    for (int c = 1; c < std::min(n_cells, 64); ++c) {
                                        llama_memory_seq_rm_cell(mem, slot.id, cell_indices[c]);
                                    }
                                }
                            }

                            int max_depth = 0;
                            for (int i = 0; i < slot.draft_tree.n_nodes; ++i) {
                                max_depth = std::max(max_depth, slot.draft_tree.depths[i]);
                            }
                            if (max_depth > commit_depth) {
                                llama_memory_seq_rm(mem, slot.id, n_before + commit_depth + 1, n_before + max_depth + 1);
                            }

                            llama_memory_seq_rm(mem, seq_backup, -1, -1);
                        } else {
                            llama_memory_seq_rm(mem, slot.id, slot.n_pos_before_draft, -1);
                            llama_memory_seq_cp(mem, seq_backup, slot.id, -1, -1);
                            llama_memory_seq_rm(mem, seq_backup, -1, -1);

                            const int n_reeval = slot.prompt.n_tokens() - slot.n_tokens_before_draft;
                            if (n_reeval > 0) {
                                llama_batch batch_reeval = llama_batch_init(n_reeval, 0, 1);
                                for (int j = slot.n_tokens_before_draft; j < slot.prompt.n_tokens(); ++j) {
                                    common_batch_add(batch_reeval, slot.prompt.tokens[j], slot.prompt.tokens.pos_next(j), { slot.id }, false);
                                }
                                llama_decode(ctx_tgt, batch_reeval);
                                llama_batch_free(batch_reeval);

                                capture_indices.clear();
                                for (int j = 0; j < n_reeval; ++j) {
                                    capture_indices.push_back(j);
                                }
                            }
                        }

                        if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                            llama_dflash_set_active_slot(ctx_tgt, slot.id);
                            common_speculative_update_logits_by_indices(slot.get_spec(), ctx_tgt, capture_indices);
                        }
                    } else if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                        llama_dflash_set_active_slot(ctx_tgt, slot.id);
                        if (all_accepted_flat) {
                            llama_clear_tree_parent_ids(ctx_tgt);
                            auto * mem = llama_get_memory(ctx_tgt);
                            if (slot.has_recurrent_only_backup) {
                                llama_memory_seq_rm_recurrent(mem, seq_backup, -1, -1);
                            } else {
                                llama_memory_seq_rm(mem, seq_backup, -1, -1);
                            }
                            llama_memory_seq_rm(mem, slot.id, slot.prompt.tokens.pos_next(), -1);
                        } else {
                            llama_clear_tree_parent_ids(ctx_tgt);
                            llama_dflash_rollback(ctx_tgt, slot.id, seq_backup, slot.n_pos_before_draft, n_hidden_keep);
                            if (n_slots_drafted > 1) {
                                // Multi-slot accept can immediately roll back another seq; make this
                                // seq's async recurrent replay visible before the next mutation.
                                llama_tape_replay_sync(ctx_tgt);
                            }
                        }
                    } else {
                        auto * mem = llama_get_memory(ctx_tgt);

                        if (all_accepted_flat) {
                            llama_memory_seq_rm(mem, seq_backup, -1, -1);
                            llama_memory_seq_rm(mem, slot.id, slot.prompt.tokens.pos_next(), -1);
                        } else {
                            const llama_pos pos_before = slot.n_pos_before_draft;

                            llama_memory_seq_rm(mem, slot.id, pos_before, -1);
                            llama_memory_seq_cp(mem, seq_backup, slot.id, -1, -1);
                            llama_memory_seq_rm(mem, seq_backup, -1, -1);

                            const int n_reeval = slot.prompt.n_tokens() - slot.n_tokens_before_draft;
                            if (n_reeval > 0) {
                                llama_batch batch_reeval = llama_batch_init(n_reeval, 0, 1);
                                for (int j = slot.n_tokens_before_draft; j < slot.prompt.n_tokens(); ++j) {
                                    common_batch_add(batch_reeval, slot.prompt.tokens[j], slot.prompt.tokens.pos_next(j), { slot.id }, false);
                                }
                                llama_decode(ctx_tgt, batch_reeval);
                                llama_batch_free(batch_reeval);
                            }
                        }
                    }

                    slot.has_draft_backup = false;
                    slot.has_recurrent_only_backup = false;
                    slot.seq_id_backup = -1;
                } else {
                    const llama_pos pos_next_d = slot.prompt.tokens.pos_next();
                    common_context_seq_rm(ctx_tgt, slot.id, pos_next_d, -1);
                    if (ctx_dft) {
                        common_context_seq_rm(ctx_dft.get(), slot.id, pos_next_d, -1);
                    }
                    if (is_draft_tree) {
                        llama_clear_tree_parent_ids(ctx_tgt);
                    }
                }
                profile_accept_lap(profile_accept_rollback_us);

                slot.spec_i_batch.clear();
                slot.spec_pad_i_batch.clear();
                slot.drafted.clear();
                slot.draft_tree.tokens.clear();
                slot.draft_tree.parents.clear();
                slot.draft_tree.depths.clear();
                slot.draft_tree.child_maps.clear();
                slot.draft_tree.visibility.clear();
                slot.draft_tree.log_probs.clear();
                slot.draft_tree.n_nodes = 0;
                slot.draft_tree.main_path_len = 0;
                slot.has_draft_tree = false;
                slot.spec_draft.clear();
                slot.draft_log_probs.clear();

                for (size_t i = 0; i < ids.size(); ++i) {
                    slot.n_decoded += 1;
                    slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                    completion_token_output result;

                    result.tok          = ids[i];
                    result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f; // set later

                    // TODO: set result.probs

                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();

                        break;
                    }
                }
                profile_accept_lap(profile_accept_emit_us);

                if (profile_dflash_accept) {
                    SLT_INF(slot,
                            "dflash profile: accept n_draft=%zu ids=%zu sample=%.3f ms update=%.3f ms "
                            "book=%.3f ms rollback=%.3f ms emit=%.3f ms total=%.3f ms\n",
                            n_draft, ids.size(),
                            profile_accept_sample_us / 1e3,
                            profile_accept_update_us / 1e3,
                            profile_accept_book_us / 1e3,
                            profile_accept_rollback_us / 1e3,
                            profile_accept_emit_us / 1e3,
                            (ggml_time_us() - profile_accept_start) / 1e3);
                }

                slot.print_timings_tg();

                SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", n_accepted_draft, (int) n_draft, slot.prompt.n_tokens());
            }
            t_accept_total += ggml_time_us() - t_accept_start;
        }

        const bool pure_tg_cycle = n_tg_tokens > 0 && batch.n_tokens == n_tg_tokens;
        auto apply_profit_decision = [&](server_slot & slot) {
            if (!slot.task || !server_adaptive_dm_uses_profit_controller(slot.dm_controller)) {
                return;
            }

            const int configured_base_n_max = common_speculative_n_max(slot.get_spec(), slot.task->params.speculative);
            const int base_n_max = dflash_effective_adaptive_base_n_max(
                    ctx_dft_shared.get(), slot.task->params.speculative, configured_base_n_max);
            const int prev_n_max = slot.adaptive_n_max;
            int recommended = slot.decide_profit_n_max(base_n_max);

            slot.apply_profit_recommendation(recommended);
            if (slot.adaptive_n_max != prev_n_max) {
                slot.adaptive_probe_counter = 0;
                SLT_INF(slot, "adaptive dm profit: cur=%d recommended=%d score=%.1f action=apply\n",
                        prev_n_max, slot.adaptive_n_max, (double) slot.profit_current_score);
            }
        };

        if (pure_tg_cycle && n_slots_drafted > 0) {
            const int64_t t_cycle_total_now = ggml_time_us() - t_cycle_start;
            const float dflash_profit_shared_scale = n_slots_drafted > 1 ? 1.0f / (float) n_slots_drafted : 1.0f;
            for (auto & slot : slots) {
                if (!slot.profit_pending || slot.profit_pending_tree || !slot.task) {
                    continue;
                }

                const float draft_ms  = (t_draft_total / 1000.0f) * dflash_profit_shared_scale;
                const float verify_ms = (t_verify_total / 1000.0f) * dflash_profit_shared_scale;
                const float accept_ms = (t_accept_total / 1000.0f) * dflash_profit_shared_scale;
                const float cycle_ms  = (t_cycle_total_now / 1000.0f) * dflash_profit_shared_scale;

                slot.observe_profit_timing(
                        slot.profit_pending_requested_n_max > 0
                            ? slot.profit_pending_requested_n_max
                            : slot.profit_pending_n_draft,
                        slot.profit_pending_n_draft,
                        slot.profit_pending_n_accepted,
                        draft_ms,
                        verify_ms,
                        accept_ms,
                        cycle_ms);

                LOG_DBG("profit telemetry: n_draft=%d n_accepted=%d "
                        "draft_ms=%.1f verify_ms=%.1f accept_ms=%.1f cycle_ms=%.1f "
                        "shared_scale=%.3f n_encoded=%d\n",
                        slot.profit_pending_n_draft,
                        slot.profit_pending_n_accepted,
                        (double) draft_ms,
                        (double) verify_ms,
                        (double) accept_ms,
                        (double) cycle_ms,
                        (double) dflash_profit_shared_scale,
                        slot.n_decoded);

                apply_profit_decision(slot);
                slot.profit_pending = false;
                slot.profit_pending_requested_n_max = 0;
            }
        } else {
            for (auto & slot : slots) {
                slot.profit_pending = false;
                slot.profit_pending_requested_n_max = 0;
            }
        }
        llama_set_dflash_consume_reduced(ctx_tgt, false);

        if (pure_tg_cycle && n_slots_drafted == 0 && n_profit_baseline_slots > 0) {
            const int64_t t_cycle_total_now = ggml_time_us() - t_cycle_start;
            const float baseline_shared_scale = n_tg_tokens > 1 ? 1.0f / (float) n_tg_tokens : 1.0f;
            for (server_slot * slot_ptr : profit_baseline_slots) {
                if (!slot_ptr || !slot_ptr->task || slot_ptr->state != SLOT_STATE_GENERATING) {
                    continue;
                }

                slot_ptr->observe_profit_timing(
                        0,
                        0,
                        0,
                        (t_draft_total / 1000.0f) * baseline_shared_scale,
                        (t_verify_total / 1000.0f) * baseline_shared_scale,
                        (t_accept_total / 1000.0f) * baseline_shared_scale,
                        (t_cycle_total_now / 1000.0f) * baseline_shared_scale);
                apply_profit_decision(*slot_ptr);
            }
        }

        // --- profiling: log per-cycle breakdown ---
        if (n_slots_drafted > 0) {
            const int64_t t_cycle_total = ggml_time_us() - t_cycle_start;
            const int64_t t_other = t_cycle_total - t_draft_total - t_verify_total - t_accept_total;
            if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
                    params_base.speculative.branch_budget == 0 &&
                    dflash_cycle_slot &&
                    dflash_cycle_slot->id >= 0 &&
                    !slots.empty()) {
                dflash_next_tg_slot = (dflash_cycle_slot->id + 1) % (int) slots.size();
            }
            if (profile_dflash_cycle && n_slots_drafted == 1 && dflash_cycle_slot &&
                    params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                const common_dflash_ring_stats ring_stats =
                    common_speculative_dflash_ring_stats(dflash_cycle_slot->get_spec());
                const int base_n_max = dflash_cycle_slot->task
                    ? common_speculative_n_max(dflash_cycle_slot->get_spec(), dflash_cycle_slot->task->params.speculative)
                    : 0;
                const int active_n_max = dflash_cycle_slot->dm_adaptive && dflash_cycle_slot->adaptive_n_max >= 0
                    ? dflash_cycle_slot->adaptive_n_max
                    : base_n_max;
                const int task_id = dflash_cycle_slot->task ? dflash_cycle_slot->task->id : -1;
                SLT_INF(*dflash_cycle_slot,
                        "dflash cycle: task=%d pos=%d adaptive_n_max=%d n_draft=%d n_accept=%d "
                        "verify_rows=%d useful_rows=%d pad_rows=%d "
                        "cross_len=%d ring_filled=%d committed=%d "
                        "draft_ms=%.1f verify_ms=%.1f accept_ms=%.1f total_ms=%.1f "
                        "reduced_verify=%d top_k=%d accept_bin=%d ctx_bucket=%d\n",
                        task_id,
                        (int) dflash_cycle_pos,
                        active_n_max,
                        dflash_cycle_n_draft,
                        dflash_cycle_n_accept,
                        dflash_cycle_verify_rows,
                        dflash_cycle_useful_rows,
                        dflash_cycle_pad_rows,
                        ring_stats.cross_len,
                        ring_stats.ring_filled,
                        ring_stats.committed_len,
                        t_draft_total / 1e3,
                        t_verify_total / 1e3,
                        t_accept_total / 1e3,
                        t_cycle_total / 1e3,
                        dflash_cycle_reduced_verify ? 1 : 0,
                        dflash_cycle_reduced_top_k,
                        server_slot::dflash_accept_bin(dflash_cycle_n_accept),
                        server_slot::dflash_context_bucket(dflash_cycle_pos));
            }
            if (profile_dflash_cycle) {
                SRV_INF("spec cycle (%d slots): draft=%.1fms verify=%.1fms accept=%.1fms "
                        "other=%.1fms replay_sync=%.1fms recurrent_backup=%.1fms "
                        "backup_enqueue=%.1fms backup_sync=%.1fms backup_layers=%" PRIu64
                        " backup_tensors=%" PRIu64 " backup_cuda_d2d=%" PRIu64
                        " backup_fallback=%" PRIu64 " tape_record=%.1fms total=%.1fms\n",
                        n_slots_drafted,
                        t_draft_total / 1e3, t_verify_total / 1e3, t_accept_total / 1e3,
                        t_other / 1e3, t_replay_sync_total / 1e3, t_recurrent_backup_total / 1e3,
                        recurrent_backup_enqueue_us / 1e3, recurrent_backup_sync_us / 1e3,
                        recurrent_backup_layers, recurrent_backup_tensors, recurrent_backup_cuda_d2d,
                        recurrent_backup_fallback, t_tape_record_total / 1e3, t_cycle_total / 1e3);
            }
        }

        // turn off DFlash tape recording after all sub-batches — was turned on
        // before the sub-batch for loop. Placing it outside the loop (vs inside,
        // after the first decode) keeps recording active across all sub-batches,
        // which matters when multiple slots share one pass and the combined
        // verify batch spans more than one ubatch.
        if (dflash_tape_active) {
            const int64_t t_tape_record_start = dflash_profile_start();
            llama_set_tape_recording(ctx_tgt, false);
            dflash_profile_add(t_tape_record_total, t_tape_record_start);
        }

        // restore force_split_seq for the next cycle (prompt batches need it)
        if (can_batch_multiseq) {
            llama_set_force_split_seq(ctx_tgt, true);
        }

        SRV_DBG("%s", "run slots completed\n");
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* json_webui_settings    */ impl->json_webui_settings,  // Deprecated
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
    };
}



// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_http_res {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::on_sleeping_changed(std::function<void(bool)> callback) {
    impl->queue_tasks.on_sleeping_state(std::move(callback));
}

// compute the number of tokens before the last user message in the prompt
static int32_t prompt_get_n_before_user(
        const json & message_spans,
        const std::string & prompt,
        const std::vector<raw_buffer> & files,
        const llama_vocab * vocab,
        mtmd_context * mctx) {
    int32_t result = -1;
    int32_t byte_pos = -1;

    for (const auto & span : message_spans) {
        const std::string role = json_value(span, "role", std::string());

        if (role == "user") {
            byte_pos = json_value(span, "pos", -1);
        }
    }

    if (byte_pos >= 0) {
        GGML_ASSERT((size_t) byte_pos <= prompt.size());

        const std::string prefix = prompt.substr(0, (size_t) byte_pos);

        const std::string marker = get_media_marker();
        size_t n_prefix_media = 0;
        for (size_t pos = 0; (pos = prefix.find(marker, pos)) != std::string::npos; pos += marker.size()) {
            n_prefix_media++;
        }

        GGML_ASSERT(n_prefix_media <= files.size());

        if (mctx != nullptr && n_prefix_media > 0) {
            // TODO: this makes a copy - avoid it
            std::vector<raw_buffer> prefix_files(files.begin(), files.begin() + n_prefix_media);

            result = (int32_t) process_mtmd_prompt(mctx, prefix, prefix_files).size();
        } else {
            result = (int32_t) tokenize_input_prompts(vocab, nullptr, prefix, true, true)[0].size();
        }

        SRV_TRC("message_spans: last user message: byte_pos=%d, media=%zu, n_before_user=%d\n",
                byte_pos, n_prefix_media, result);
    }

    return result;
}


//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    meta->logit_bias_eog,
                    data);

            const auto message_spans = json_value(data, "message_spans", json::array());
            if (prompt.is_string() && message_spans.is_array()) {
                task.params.n_before_user =
                    prompt_get_n_before_user(
                        message_spans,
                        prompt.get<std::string>(),
                        files,
                        ctx_server.vocab,
                        ctx_server.mctx);
            }

            task.id_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->next = [res_this = res.get(), res_type, &req, &params](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            try {
                if (req.should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &req, &start_time, &params]() {
                    if (req.should_stop()) {
                        return true; // should_stop condition met
                    } else if (params.sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)params.sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(req.should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        };
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            },{
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"audio",  meta->has_inp_audio},
            } },
            { "media_marker",                get_media_marker() },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            // New keys
            { "ui",                           params.ui },
            { "ui_settings",                  meta->json_ui_settings },
            // Deprecated: use ui/ui_settings instead (kept for backward compat)
            { "webui",                        params.webui },
            { "webui_settings",               meta->json_webui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
            { "cors_proxy_enabled",          params.ui_mcp_proxy || params.webui_mcp_proxy },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);

        json prompt = body_parsed.at("prompt");
        llama_tokens tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true);
        res->ok({{"input_tokens", static_cast<int>(tokens.size())}});
        return res;
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                get_model_info(),
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

json server_routes::get_model_info() const {
    return json {
        {"id",       meta->model_name},
        {"aliases",  meta->model_aliases},
        {"tags",     meta->model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta->model_vocab_type},
            {"n_vocab",     meta->model_vocab_n_tokens},
            {"n_ctx",       meta->slot_n_ctx},
            {"n_ctx_train", meta->model_n_ctx_train},
            {"n_embd",      meta->model_n_embd_inp},
            {"n_params",    meta->model_n_params},
            {"size",        meta->model_size},
        }},
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}
