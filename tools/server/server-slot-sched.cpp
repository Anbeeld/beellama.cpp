#include "server-slot-sched.h"

#include "common.h"
#include "llama.h"
#include "log.h"
#include "speculative.h"

#include <algorithm>
#include <chrono>
#include <unordered_set>

//
// server_prompt
//

server_prompt_reuse server_prompt::get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min,
        int32_t reuse_alignment, bool force_checkpoint, bool limit_checkpoint_tokens) const {
    server_prompt_reuse res;
    res.n_past   = n_prefix;
    res.pos_next = tokens.pos_next(n_prefix);

    // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
    // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
    const llama_pos pos_eval = res.pos_next - (has_new_tokens ? 0 : 1);

    // a state or a checkpoint is useful when its pos_min is below this threshold
    const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);

    if (!force_checkpoint && pos_min < pos_min_thold) {
        return res;
    }

    // search for a context checkpoint
    const auto it = std::find_if(
        checkpoints.rbegin(),
        checkpoints.rend(),
        [&](const auto & ckpt) {
            const auto & cur = *ckpt;
            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
            SRV_TRC("checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
            if (cur.pos_max > res.pos_next ||
                    (limit_checkpoint_tokens && cur.n_tokens > n_prefix) ||
                    cur.n_tokens < 0 || cur.n_tokens % std::max(1, reuse_alignment) != 0) {
                return false;
            }
            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
        }
    );

    if (it == checkpoints.rend()) {
        res.n_past   = 0;
        res.pos_next = 0;
        return res;
    }

    const auto & cur = **it;

    res.pos_next = std::min(res.pos_next, std::max(cur.pos_min + 1, cur.pos_max));
    res.n_past   = std::min(tokens.size_up_to_pos(res.pos_next), (size_t) cur.n_tokens);
    res.ckpt     = *it;

    return res;
}


size_t server_prompt_cache::accounted_size() const {
    size_t res = 0;
    std::unordered_set<const void *> checkpoint_storage;

    for (const auto & state : states) {
        res += state.data.size();
        for (const auto & ckpt : state.prompt.checkpoints) {
            const auto & checkpoint = *ckpt;
            if (!checkpoint.data_tgt.empty() &&
                    checkpoint_storage.insert(checkpoint.data_tgt.storage_id()).second) {
                res += checkpoint.data_tgt.size();
            }
            if (!checkpoint.data_dft.empty() &&
                    checkpoint_storage.insert(checkpoint.data_dft.storage_id()).second) {
                res += checkpoint.data_dft.size();
            }
            if (!checkpoint.data_spec.empty() &&
                    checkpoint_storage.insert(checkpoint.data_spec.data()).second) {
                res += checkpoint.data_spec.size();
            }
        }
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::admit(server_prompt_cache_state && candidate) {
    ++admission_attempts;
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(candidate.prompt.tokens);

        if (cur_lcp_len == (int) candidate.prompt.tokens.size() && are_lora_equal(it->prompt.lora, candidate.prompt.lora)) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            ++admission_failures;
            return nullptr;
        }
    }

    // Count retained immutable checkpoint buffers once across cache entries.
    // Copying a server_prompt shares these buffers and detaches on mutation.
    std::unordered_set<const void *> retained_checkpoint_storage;
    for (const auto & state : states) {
        for (const auto & ckpt : state.prompt.checkpoints) {
            const auto & checkpoint = *ckpt;
            retained_checkpoint_storage.insert(checkpoint.data_tgt.storage_id());
            retained_checkpoint_storage.insert(checkpoint.data_dft.storage_id());
            if (!checkpoint.data_spec.empty()) {
                retained_checkpoint_storage.insert(checkpoint.data_spec.data());
            }
        }
    }
    size_t checkpoints_size = 0;
    for (const auto & checkpoint : candidate.prompt.checkpoints) {
        const auto & ckpt = *checkpoint;
        if (!ckpt.data_tgt.empty() &&
                retained_checkpoint_storage.insert(ckpt.data_tgt.storage_id()).second) {
            checkpoints_size += ckpt.data_tgt.size();
        }
        if (!ckpt.data_dft.empty() &&
                retained_checkpoint_storage.insert(ckpt.data_dft.storage_id()).second) {
            checkpoints_size += ckpt.data_dft.size();
        }
        if (!ckpt.data_spec.empty() &&
                retained_checkpoint_storage.insert(ckpt.data_spec.data()).second) {
            checkpoints_size += ckpt.data_spec.size();
        }
    }

    const size_t state_size_new = candidate.data.size() + checkpoints_size;
    const size_t standalone_size = candidate.accounted_size();

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && (state_size_new > limit_size || standalone_size > limit_size)) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                standalone_size / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        ++admission_failures;
        return nullptr;
    }

    // Publish the already-materialized candidate node before touching existing
    // entries.  list node allocation is the last fallible admission step; if it
    // fails, the cache is unchanged.
    try {
        states.push_back(std::move(candidate));
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate prompt cache entry: %s\n", e.what());
        ++admission_failures;
        return nullptr;
    }
    auto admitted = std::prev(states.end());

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        if (it == admitted) {
            ++it;
            continue;
        }
        const int len = it->prompt.tokens.get_common_prefix(admitted->prompt.tokens);

        if (len == (int) it->prompt.tokens.size() && are_lora_equal(it->prompt.lora, admitted->prompt.lora)) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // The candidate is already present and fully materialized. Evict only
        // older entries until unique payload accounting is within budget.
        while (!states.empty() && accounted_size() > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    states.front().accounted_size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    ++admission_successes;
    return &*admitted;
}

server_prompt_cache_state * server_prompt_cache::insert(
        const server_prompt & prompt,
        server_prompt_data && data) {
    server_prompt_cache_state candidate;
    candidate.prompt = prompt.clone();
    candidate.data = std::move(data);
    return admit(std::move(candidate));
}

server_prompt_cache_state * server_prompt_cache::alloc(
        const server_prompt & prompt,
        size_t state_size_tgt,
        size_t state_size_dft) {
    server_prompt_data data;
    try {
        data.main.resize(state_size_tgt);
        data.drft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        ++admission_attempts;
        ++admission_failures;
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());
        return nullptr;
    }
    return insert(prompt, std::move(data));
}

bool server_prompt_cache::erase(const server_prompt_cache_state * entry) {
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (&*it == entry) {
            states.erase(it);
            return true;
        }
    }
    return false;
}

server_prompt_restore_result server_prompt_restore_transaction_diagnostic(
        server_prompt_state_view target,
        server_prompt_state_view draft,
        server_prompt_state_view speculative,
        const server_prompt_restore_transaction_io & io,
        server_prompt_restore_timings * timings) {
    if (!io.prepare || !io.commit) {
        return { false, false, SERVER_PROMPT_STATE_MAIN, SERVER_PROMPT_RESTORE_INVALID_IO };
    }
    if (io.restore_target && target.size == 0) {
        return { false, true, SERVER_PROMPT_STATE_MAIN, SERVER_PROMPT_RESTORE_MISSING_REQUIRED_STATE };
    }
    if (io.restore_draft && draft.size == 0) {
        return { false, true, SERVER_PROMPT_STATE_DRAFT, SERVER_PROMPT_RESTORE_MISSING_REQUIRED_STATE };
    }

    const auto prepare = [&](bool enabled, server_prompt_state_kind kind, server_prompt_state_view state) {
        if (enabled) {
            const auto t_start = std::chrono::steady_clock::now();
            const bool prepared = io.prepare(kind, state);
            if (timings) {
                timings->prepare_us += std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t_start).count();
            }
            if (!prepared) {
                return server_prompt_restore_result {
                    false, true, kind, SERVER_PROMPT_RESTORE_PREPARE_REJECTED
                };
            }
        }
        return server_prompt_restore_result {
            true, false, SERVER_PROMPT_STATE_MAIN, SERVER_PROMPT_RESTORE_NONE
        };
    };
    for (const auto & step : {
            std::pair { io.restore_target, SERVER_PROMPT_STATE_MAIN },
            std::pair { io.restore_draft, SERVER_PROMPT_STATE_DRAFT },
            std::pair { io.restore_speculative, SERVER_PROMPT_STATE_SPECULATIVE } }) {
        const server_prompt_state_view state = step.second == SERVER_PROMPT_STATE_MAIN ? target :
                step.second == SERVER_PROMPT_STATE_DRAFT ? draft : speculative;
        const auto result = prepare(step.first, step.second, state);
        if (!result.success) {
            return result;
        }
    }

    // Speculative apply is prepared and no-fail. Memory commits likewise only
    // publish already-validated backend writes and metadata.
    const auto commit = [&](bool enabled, server_prompt_state_kind kind) {
        if (!enabled) {
            return;
        }
        const auto t_start = std::chrono::steady_clock::now();
        io.commit(kind);
        if (timings) {
            timings->commit_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_start).count();
        }
    };
    commit(io.restore_speculative, SERVER_PROMPT_STATE_SPECULATIVE);
    commit(io.restore_target, SERVER_PROMPT_STATE_MAIN);
    commit(io.restore_draft, SERVER_PROMPT_STATE_DRAFT);
    return { true, false, SERVER_PROMPT_STATE_MAIN, SERVER_PROMPT_RESTORE_NONE };
}

bool server_prompt_restore_transaction(
        server_prompt_state_view target,
        server_prompt_state_view draft,
        server_prompt_state_view speculative,
        const server_prompt_restore_transaction_io & io) {
    return server_prompt_restore_transaction_diagnostic(target, draft, speculative, io).success;
}

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
        server_prompt_restore_timings * timings) {
    using memory_plan_ptr = std::unique_ptr<
            llama_state_seq_restore_plan,
            decltype(&llama_state_seq_restore_plan_free)>;
    using speculative_plan_ptr = std::unique_ptr<
            common_speculative_state_restore_plan,
            decltype(&common_speculative_state_restore_plan_free)>;

    memory_plan_ptr target_plan(nullptr, llama_state_seq_restore_plan_free);
    memory_plan_ptr draft_plan(nullptr, llama_state_seq_restore_plan_free);
    speculative_plan_ptr speculative_plan(nullptr, common_speculative_state_restore_plan_free);

    server_prompt_restore_transaction_io io {
        /*.restore_target =*/ restore_target,
        /*.restore_draft =*/ restore_draft,
        /*.restore_speculative =*/ restore_speculative,
        /*.prepare =*/ [&](server_prompt_state_kind kind, server_prompt_state_view state) {
            if (kind == SERVER_PROMPT_STATE_SPECULATIVE) {
                speculative_plan.reset(common_speculative_prepare_state(
                        speculative, seq_id, state.data, state.size));
                return speculative_plan != nullptr;
            }

            llama_context * ctx = kind == SERVER_PROMPT_STATE_MAIN ? target : draft;
            memory_plan_ptr & plan = kind == SERVER_PROMPT_STATE_MAIN ? target_plan : draft_plan;
            if (ctx == nullptr) {
                return false;
            }
            plan.reset(llama_state_seq_prepare_data_ext(
                    ctx, state.data, state.size, seq_id, flags));
            return plan != nullptr;
        },
        /*.commit =*/ [&](server_prompt_state_kind kind) {
            if (kind == SERVER_PROMPT_STATE_SPECULATIVE) {
                common_speculative_state_restore_plan_commit(speculative_plan.get());
                return;
            }
            memory_plan_ptr & plan = kind == SERVER_PROMPT_STATE_MAIN ? target_plan : draft_plan;
            const size_t expected = kind == SERVER_PROMPT_STATE_MAIN ? target_state.size : draft_state.size;
            GGML_ASSERT(llama_state_seq_restore_plan_commit(plan.get()) == expected);
        },
    };
    return server_prompt_restore_transaction_diagnostic(
            target_state, draft_state, speculative_state, io, timings);
}

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
        bool restore_speculative) {
    return server_prompt_restore_transaction_diagnostic(
            target, draft, speculative, seq_id, flags,
            target_state, draft_state, speculative_state,
            restore_target, restore_draft, restore_speculative).success;
}

bool server_prompt_cache::load(
        server_prompt & prompt,
        const server_tokens & tokens_new,
        size_t live_native_restorable_tokens,
        int32_t reuse_alignment,
        const server_prompt_cache_state_io & io,
        const server_prompt_cache_state * excluded,
        uint32_t native_rollback_limit,
        const std::vector<common_adapter_lora_info> * lora_new) {
    const auto live_plan = server_prompt_plan_reuse(
            prompt, tokens_new, reuse_alignment, live_native_restorable_tokens, false);

    const bool is_empty = prompt.tokens.empty() ||
            (lora_new && lora_should_clear_cache(prompt.lora, *lora_new));
    size_t restorable_best = is_empty ? 0 : live_plan.restorable_tokens;
    float f_keep_best = !is_empty ?
            float(restorable_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(restorable_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (&*it == excluded || (lora_new && lora_should_clear_cache(it->prompt.lora, *lora_new))) {
            continue;
        }
        const auto plan_cur = server_prompt_plan_reuse(
                it->prompt, tokens_new, reuse_alignment, 0, true);
        const size_t lcp_cur = plan_cur.lexical_tokens;
        const size_t restorable_cur = plan_cur.restorable_tokens;

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7zu, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // A lexical tie is not useful if neither a checkpoint nor the saved
        // native state can reach the requested boundary. Do not publish that
        // snapshot only to discard it and reprocess the entire prompt.
        const llama_pos rollback = it->prompt.tokens.pos_next() -
                it->prompt.tokens.pos_next(lcp_cur);
        if (restorable_cur == 0 && native_rollback_limit != UINT32_MAX &&
                rollback > int64_t(native_rollback_limit)) {
            continue;
        }

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (restorable_cur > restorable_best ||
                (restorable_cur == restorable_best && f_keep_cur > f_keep_best)) {
            restorable_best = restorable_cur;
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        ++restore_attempts;
        SRV_TRC(" - found better restorable prompt with n = %zu, f_keep = %.3f, f_sim = %.3f\n",
                restorable_best, f_keep_best, f_sim_best);

        auto & data = it_best->data;
        if (data.main.empty() ||
                io.has_draft != !data.drft.empty() ||
                (!io.has_speculative && !data.spec.empty()) ||
                !io.restore_transaction) {
            ++restore_failures;
            return false;
        }

        if (!io.restore_transaction(
                    data.main.data(), data.main.size(),
                    data.drft.data(), data.drft.size(),
                    data.spec.data(), data.spec.size())) {
            ++restore_failures;
            return false;
        }

        prompt = it_best->prompt.clone();
        ++restore_successes;
    }

    return true;
}

bool server_prompt_cache::load(
        server_prompt & prompt,
        const server_tokens & tokens_new,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        common_speculative * spec,
        int32_t id_slot,
        size_t live_native_restorable_tokens,
        int32_t reuse_alignment,
        const server_prompt_cache_state * excluded,
        server_prompt_restore_timings * timings,
        uint32_t native_rollback_limit,
        const std::vector<common_adapter_lora_info> * lora_new) {
    const auto recurrent_rollback = [](llama_context * ctx) -> uint32_t {
        const auto * model = llama_get_model(ctx);
        if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
            // Body-backed attention can plan a deep historical truncation even
            // when its exact-tail rollback reserve is deliberately bounded.
            return UINT32_MAX;
        }
        return common_context_seq_rm_max_rollback(ctx);
    };
    uint32_t rollback = recurrent_rollback(ctx_tgt);
    if (ctx_dft) {
        rollback = std::min(rollback, recurrent_rollback(ctx_dft));
    }
    // The prompt tracks the final sampled token, which may not yet be decoded.
    if (rollback != UINT32_MAX) {
        ++rollback;
    }
    native_rollback_limit = std::min(native_rollback_limit, rollback);
    constexpr llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED;
    server_prompt_cache_state_io io {
        /*.has_draft =*/ ctx_dft != nullptr,
        /*.has_speculative =*/ spec != nullptr,
        /*.restore_transaction =*/ [&](const uint8_t * main, size_t main_size,
                                       const uint8_t * drft, size_t drft_size,
                                       const uint8_t * speculative_state, size_t speculative_size) {
            return server_prompt_restore_transaction_diagnostic(
                    ctx_tgt, ctx_dft, spec, id_slot, flags,
                    { main, main_size }, { drft, drft_size },
                    { speculative_state, speculative_size },
                    true, ctx_dft != nullptr, spec != nullptr, timings).success;
        },
    };

    return load(prompt, tokens_new, live_native_restorable_tokens, reuse_alignment, io, excluded,
            native_rollback_limit, lora_new);
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && accounted_size() > limit_size) {
            SRV_WRN(" - cache accounted-payload limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().accounted_size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(accounted_size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().accounted_size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), accounted_size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.accounted_size() / (1024.0 * 1024.0));
    }
}

size_t server_prompt_live_native_restorable(const server_prompt & prompt, const server_tokens & requested,
        llama_context * target, llama_context * draft, llama_seq_id id) {
    if (!target) {
        return 0;
    }
    const size_t lcp_len = prompt.tokens.get_common_prefix(requested);
    if (lcp_len == 0) {
        return 0;
    }
    if (lcp_len == prompt.tokens.size()) {
        return lcp_len;
    }
    const llama_pos requested_p0 = prompt.tokens.pos_next(lcp_len);
    llama_pos target_p0 = requested_p0;
    llama_pos target_p1 = -1;
    llama_pos draft_p0 = requested_p0;
    llama_pos draft_p1 = -1;
    const bool target_planned = llama_memory_seq_rm_plan(
            llama_get_memory(target), id, requested_p0, -1,
            &target_p0, &target_p1);
    const bool draft_planned = draft == nullptr || llama_memory_seq_rm_plan(
            llama_get_memory(draft), id, requested_p0, -1,
            &draft_p0, &draft_p1);
    if (!target_planned || !draft_planned || target_p1 >= 0 ||
            (draft != nullptr && draft_p1 >= 0)) {
        return 0;
    }
    const llama_pos common_p0 = draft != nullptr ? std::min(target_p0, draft_p0) : target_p0;
    return common_p0 > 0 && common_p0 <= requested_p0 ?
            prompt.tokens.size_up_to_pos(common_p0) : 0;
}

//
// server_slot_sched
//

static std::vector<const server_slot_info *> get_free_slot_infos(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) {
    std::vector<const server_slot_info *> free_slots;
    for (const auto & slot : slots) {
        if (!slot.is_processing && slot.id != exclude_id_slot) {
            free_slots.push_back(&slot);
        }
    }

    std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot_info * a, const server_slot_info * b) {
        if ((a->prompt->n_tokens() == 0) != (b->prompt->n_tokens() == 0)) {
            return a->prompt->n_tokens() == 0;
        }
        return a->t_last_used < b->t_last_used;
    });

    free_slots.resize(std::min(free_slots.size(), n_slots_needed));

    return free_slots;
}

std::vector<int> server_slot_sched::get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const {
    std::vector<int> res;
    for (const auto * slot : get_free_slot_infos(slots, n_slots_needed, exclude_id_slot)) {
        res.push_back(slot->id);
    }

    return res;
}

server_slot_plan server_slot_sched::select(const std::vector<server_slot_info> & slots, const server_task & task, const std::vector<common_adapter_lora_info> & lora) const {
    const server_slot_info * ret = nullptr;

    bool update_cache = false;

    // if a specific slot is requested, use it (still goes through cache update logic below)
    if (task.id_slot != -1 && !slots.empty()) {
        // note: allow id_slot to be out of bounds (wrap around)
        const int id_slot = task.id_slot % slots.size();

        for (const auto & slot : slots) {
            if (slot.id == id_slot) {
                ret = &slot;
                break;
            }
        }

        if (ret) {
            // a busy slot is returned untouched, the caller defers the task
            if (ret->is_processing) {
                return { ret->id, false };
            }

            SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            // Parked unified slots need a RAM lookup even when explicitly selected.
            update_cache = ret->has_cached_prompts && ret->prompt->tokens.empty();
        }
    }

    // find the slot that has at least n% prompt similarity
    if (slot_prompt_similarity != 0.0f) {
        float f_sim_best = 0;
        size_t restorable_best = 0;

        for (const auto & slot : slots) {
            if (task.id_slot != -1 && ret && slot.id != ret->id) {
                continue;
            }

            // skip the slot if it is not available
            if (slot.is_processing) {
                SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing);
                continue;
            }

            const auto & tokens = slot.prompt->tokens;

            // skip the slot if it does not contains cached tokens
            if (tokens.empty()) {
                SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                continue;
            }

            // the task would clear this slot, its state cannot be reused with another lora config
            if (lora_should_clear_cache(slot.prompt->lora, lora)) {
                SLT_TRC(slot, "%s", " - skipping, other lora config\n");
                continue;
            }

            const auto reuse = server_prompt_plan_reuse(*slot.prompt, task.tokens, slot.reuse_alignment,
                    server_prompt_live_native_restorable(*slot.prompt, task.tokens,
                            slot.ctx_tgt, slot.ctx_dft, slot.id), false);
            const size_t restorable = reuse.restorable_tokens;
            const float f_sim_cur = float(restorable) / task.tokens.size();

            SLT_TRC(slot, " - checking restorable sim = %.3f (%zu restorable, %zu lexical/%zu) > %.3f\n",
                    f_sim_cur, restorable, reuse.lexical_tokens, task.tokens.size(), slot_prompt_similarity);

            if ((restorable > restorable_best ||
                    (restorable == restorable_best && f_sim_cur > f_sim_best)) && f_sim_cur > slot_prompt_similarity) {
                f_sim_best = f_sim_cur;
                restorable_best = restorable;

                ret = &slot;
            }
        }

        if (ret != nullptr) {
            const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt->tokens.size();

            if (task.id_slot == -1) {
                SLT_INF(*ret, "selected slot by restorable prefix, n_restorable = %zu, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        restorable_best, f_sim_best, slot_prompt_similarity, f_keep);
            }

            // if we are about to lose a large portion of the existing context - save it in the prompt cache
            if (f_keep < 0.5f) {
                update_cache = true;
            }
        }
    }

    // find an empty slot, or the one that has been least recently used
    if (ret == nullptr) {
        const auto free_slots = get_free_slot_infos(slots, 1, -1);

        if (!free_slots.empty()) {
            ret = free_slots[0];

            SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt->n_tokens(), ret->t_last_used);

            update_cache = true;
        }
    }

    if (ret == nullptr) {
        return {};
    }

    // cache prompts only for the tasks that reuse them
    update_cache = update_cache && !task.is_stateless() && task.params.cache_prompt;

    return { ret->id, update_cache };
}
