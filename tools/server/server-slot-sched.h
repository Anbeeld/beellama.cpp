#pragma once

#include "server-common.h"
#include "server-task.h"

#include <list>
#include <memory>
#include <vector>

// Some memory layouts need a durable checkpoint even when ordinary attention
// retains the complete prefix (pos_min == 0). In that case the live suffix
// threshold alone would not enter checkpoint selection.
static inline bool server_prompt_reuse_requires_checkpoint_search(
        bool      state_required,
        llama_pos pos_min,
        llama_pos pos_min_threshold) {
    return state_required || pos_min >= pos_min_threshold;
}

static inline bool server_prompt_checkpoint_must_prune(
        bool mtp, llama_pos pos_max, int64_t n_tokens,
        llama_pos pos_next, int64_t n_prefix, int64_t n_past) {
    // MTP's pending hidden row must stay within the restored continuation,
    // even when later tokens still match the incoming request.
    return pos_max > pos_next || n_tokens > n_prefix || (mtp && n_tokens > n_past);
}

// Durable KVarN checkpoints follow the physical descriptor boundary. Ordinary
// standard and recurrent caches do not impose a Bee-specific prompt cadence.
static inline int32_t server_prompt_reuse_alignment(int32_t kvarn_group) {
    return kvarn_group > 0 ? kvarn_group : 1;
}

static inline int64_t server_prompt_checkpoint_boundary(
        int64_t n_prompt_tokens,
        int64_t n_tokens_remaining,
        int32_t alignment) {
    GGML_ASSERT(alignment > 0);
    const int64_t boundary = n_prompt_tokens - n_tokens_remaining;
    return boundary > 0 ? boundary - boundary % alignment : 0;
}

// Keep durable checkpoints distributed across the context even when callers do
// not provide chat message spans. This bounds historical reprocessing instead
// of concentrating every checkpoint near the latest prompt tail.
static inline int64_t server_prompt_checkpoint_interval(
        int64_t n_ctx,
        int32_t n_checkpoints,
        int32_t min_step,
        int32_t alignment) {
    GGML_ASSERT(alignment > 0);
    if (n_ctx <= 0 || n_checkpoints <= 0) {
        return 0;
    }
    const int64_t budget_step = (n_ctx + n_checkpoints - 1) / n_checkpoints;
    const int64_t requested = std::max<int64_t>(std::max(0, min_step), budget_step);
    return ((requested + alignment - 1) / alignment) * alignment;
}

// which prompt goes to which slot: slot prompt state, prompt cache and slot selection

// what a cached prompt can give to a new prompt
struct server_prompt_reuse {
    int32_t   n_past   = 0; // tokens to keep
    llama_pos pos_next = 0;

    // restore this checkpoint first, nullptr when the live state is used as it is
    std::shared_ptr<const common_prompt_checkpoint> ckpt;
};

struct server_prompt {
    server_tokens tokens;

    // a checkpoint does not change after it is created, so the prompt cache shares it with the slot instead of a copy
    std::list<std::shared_ptr<const common_prompt_checkpoint>> checkpoints;

    // the lora config that the tokens were processed with, it follows the state to the prompt cache and back
    // TODO: slot files do not store it
    std::vector<common_adapter_lora_info> lora;

    void clear() {
        tokens.clear();
        checkpoints.clear();
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
            lora,
        };
    }

    // n_prefix: tokens shared with the new prompt, must be > 0
    // has_new_tokens: the new prompt is longer than n_prefix
    // n_swa: 0 if a state must end right before the next token (no SWA, or recurrent)
    // pos_min: min position of the live state
    // n_past == 0 without a checkpoint means that nothing can be reused
    server_prompt_reuse get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min,
            int32_t reuse_alignment = 1, bool force_checkpoint = false, bool limit_checkpoint_tokens = false) const;
};

enum server_prompt_state_kind {
    SERVER_PROMPT_STATE_MAIN,
    SERVER_PROMPT_STATE_DRAFT,
    SERVER_PROMPT_STATE_SPECULATIVE,
};

enum server_prompt_reuse_reason {
    SERVER_PROMPT_REUSE_NONE,
    SERVER_PROMPT_REUSE_NATIVE,
    SERVER_PROMPT_REUSE_CHECKPOINT,
    SERVER_PROMPT_REUSE_SELF_CONTAINED,
};

struct server_prompt_reuse_plan {
    size_t lexical_tokens = 0;
    size_t restorable_tokens = 0;
    server_prompt_reuse_reason reason = SERVER_PROMPT_REUSE_NONE;
};

static inline server_prompt_reuse_plan server_prompt_plan_reuse(
        const server_prompt & prompt,
        const server_tokens & requested,
        int32_t reuse_alignment,
        size_t native_restorable_tokens,
        bool self_contained) {
    const int32_t alignment = std::max(1, reuse_alignment);
    server_prompt_reuse_plan result;
    result.lexical_tokens = prompt.tokens.get_common_prefix(requested);
    result.restorable_tokens = std::min(result.lexical_tokens, native_restorable_tokens);
    if (result.restorable_tokens > 0) {
        result.reason = SERVER_PROMPT_REUSE_NATIVE;
    }

    if (self_contained && result.lexical_tokens == prompt.tokens.size()) {
        result.restorable_tokens = result.lexical_tokens;
        result.reason = SERVER_PROMPT_REUSE_SELF_CONTAINED;
        return result;
    }

    const llama_pos requested_p0 = requested.pos_next(result.lexical_tokens);
    for (const auto & ckpt : prompt.checkpoints) {
        const auto & checkpoint = *ckpt;
        if (checkpoint.n_tokens > 0 &&
                checkpoint.n_tokens <= int64_t(result.lexical_tokens) &&
                checkpoint.n_tokens%alignment == 0 &&
                checkpoint.pos_max <= requested_p0 &&
                size_t(checkpoint.n_tokens) > result.restorable_tokens) {
            result.restorable_tokens = size_t(checkpoint.n_tokens);
            result.reason = SERVER_PROMPT_REUSE_CHECKPOINT;
        }
    }
    return result;
}

inline bool server_draft_context_owns_state(bool has_draft_context, bool draft_memory_is_shared) {
    return has_draft_context && !draft_memory_is_shared;
}

struct server_prompt_cache_state_io {
    bool has_draft;
    bool has_speculative;
    std::function<bool(
            const uint8_t *, size_t,
            const uint8_t *, size_t,
            const uint8_t *, size_t)> restore_transaction;
};

struct server_prompt_state_view {
    const uint8_t * data = nullptr;
    size_t size = 0;
};

struct server_prompt_restore_transaction_io {
    bool restore_target;
    bool restore_draft;
    bool restore_speculative;
    std::function<bool(server_prompt_state_kind, server_prompt_state_view)> prepare;
    std::function<void(server_prompt_state_kind)> commit;
};

enum server_prompt_restore_reason {
    SERVER_PROMPT_RESTORE_NONE,
    SERVER_PROMPT_RESTORE_INVALID_IO,
    SERVER_PROMPT_RESTORE_MISSING_REQUIRED_STATE,
    SERVER_PROMPT_RESTORE_PREPARE_REJECTED,
};

struct server_prompt_restore_result {
    bool success = false;
    bool has_component = false;
    server_prompt_state_kind component = SERVER_PROMPT_STATE_MAIN;
    server_prompt_restore_reason reason = SERVER_PROMPT_RESTORE_NONE;
};

struct server_prompt_restore_timings {
    int64_t prepare_us = 0;
    int64_t commit_us = 0;
};

server_prompt_restore_result server_prompt_restore_transaction_diagnostic(
        server_prompt_state_view target,
        server_prompt_state_view draft,
        server_prompt_state_view speculative,
        const server_prompt_restore_transaction_io & io,
        server_prompt_restore_timings * timings = nullptr);

bool server_prompt_restore_transaction(
        server_prompt_state_view target,
        server_prompt_state_view draft,
        server_prompt_state_view speculative,
        const server_prompt_restore_transaction_io & io);

bool server_prompt_restore_transaction(
        llama_context * target,
        llama_context * draft,
        common_speculative * speculative,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        server_prompt_state_view target_state,
        server_prompt_state_view draft_state,
        server_prompt_state_view speculative_state,
        bool restore_target,
        bool restore_draft,
        bool restore_speculative);

server_prompt_restore_result server_prompt_restore_transaction_diagnostic(
        llama_context * target,
        llama_context * draft,
        common_speculative * speculative,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        server_prompt_state_view target_state,
        server_prompt_state_view draft_state,
        server_prompt_state_view speculative_state,
        bool restore_target,
        bool restore_draft,
        bool restore_speculative,
        server_prompt_restore_timings * timings = nullptr);

struct server_prompt_data {
    common_state_data main;
    common_state_data drft;
    std::vector<uint8_t> spec;

    size_t size() const {
        return main.size() + drft.size() + spec.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    size_t accounted_size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt->size();
        }

        return res;
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    uint64_t admission_attempts = 0;
    uint64_t admission_successes = 0;
    uint64_t admission_failures = 0;
    uint64_t restore_attempts = 0;
    uint64_t restore_successes = 0;
    uint64_t restore_failures = 0;

    // Payload-only accounting: serialized target/draft/speculative bytes and
    // deduplicated shared checkpoint buffers. Container/token capacity and
    // allocator overhead are intentionally excluded.
    size_t accounted_size() const;

    size_t n_tokens() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    server_prompt_cache_state * insert(const server_prompt & prompt, server_prompt_data && data);

    bool erase(const server_prompt_cache_state * entry);

    bool load(
            server_prompt & prompt,
            const server_tokens & tokens_new,
            size_t live_native_restorable_tokens,
            int32_t reuse_alignment,
            const server_prompt_cache_state_io & io,
            const server_prompt_cache_state * excluded = nullptr,
            uint32_t native_rollback_limit = UINT32_MAX,
            const std::vector<common_adapter_lora_info> * lora_new = nullptr);

    bool load(
            server_prompt & prompt,
            const server_tokens & tokens_new,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            common_speculative * spec,
            int32_t id_slot,
            size_t live_native_restorable_tokens,
            int32_t reuse_alignment,
            const server_prompt_cache_state * excluded = nullptr,
            server_prompt_restore_timings * timings = nullptr,
            uint32_t native_rollback_limit = UINT32_MAX,
            const std::vector<common_adapter_lora_info> * lora_new = nullptr);

    void update();

private:
    server_prompt_cache_state * admit(server_prompt_cache_state && candidate);
};


// Query each owned memory's non-mutating suffix-removal plan, using the common safe boundary.
size_t server_prompt_live_native_restorable(const server_prompt & prompt, const server_tokens & requested,
        llama_context * target, llama_context * draft, llama_seq_id id);

// what the scheduler needs to know about a slot
struct server_slot_info {
    int id = -1;
    const server_task * task = nullptr; // only for logging

    bool is_processing = false;
    int64_t t_last_used = -1;

    const server_prompt * prompt = nullptr;

    // Actual cache contexts: selection must rank a restorable boundary, not only lexical LCP.
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr; // owned draft only; never a shared MTP cache
    int32_t reuse_alignment = 1;
    bool has_cached_prompts = false;
};

struct server_slot_plan {
    // -1 if no slot is available; a busy slot if the task asks for it by id, the caller defers the task
    int id_slot = -1;

    // save the slot state to the prompt cache, then load the best match for the task
    bool update_cache = false;
};

// decides only, the caller applies the plan
struct server_slot_sched {
    float slot_prompt_similarity = 0.0f;

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<int> get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const;

    // lora: the lora config of the task
    server_slot_plan select(const std::vector<server_slot_info> & slots, const server_task & task, const std::vector<common_adapter_lora_info> & lora) const;
};
