#include "../tools/server/server-task.h"

#undef NDEBUG
#include <cassert>

static constexpr size_t KIB = 1024;

static server_prompt make_prompt(const llama_tokens & tokens) {
    server_prompt prompt;
    prompt.tokens = server_tokens(tokens, false);
    return prompt;
}

static void prompt_cache_load_target_success_draft_failure_is_atomic() {
    server_prompt_cache cache(1, 0);
    server_prompt saved = make_prompt({1, 2, 3});
    saved.data.main.resize(16);
    saved.data.drft.resize(8);
    cache.states.push_back(std::move(saved));

    server_prompt current = make_prompt({9});
    current.data.main.resize(4);
    current.checkpoints.emplace_back().data_tgt.resize(4);
    server_tokens requested(llama_tokens {1, 2, 4}, false);

    bool restored_main = false;
    bool restored_draft = false;
    bool cleared_main = false;
    bool cleared_draft = false;
    server_prompt_cache_state_io io {
        /*.has_draft =*/ true,
        /*.can_restore =*/ [](server_prompt_state_kind) { return true; },
        /*.restore =*/ [&](server_prompt_state_kind kind, const std::vector<uint8_t> &) {
            if (kind == SERVER_PROMPT_STATE_MAIN) {
                restored_main = true;
                return true;
            }
            restored_draft = true;
            return false;
        },
        /*.clear =*/ [&](server_prompt_state_kind kind) {
            (kind == SERVER_PROMPT_STATE_MAIN ? cleared_main : cleared_draft) = true;
            return true;
        },
    };

    assert(!cache.load(current, requested, io));
    assert(restored_main && restored_draft);
    assert(cleared_main && cleared_draft);
    assert(cache.states.empty());
    assert(current.tokens.empty());
    assert(current.data.size() == 0);
    assert(current.checkpoints.empty());
}

static void server_unsupported_removal_falls_back_to_full_reprocess() {
    int partial_removals = 0;
    int full_clears = 0;
    server_seq_rm_io io {
        /*.has_draft =*/ true,
        /*.plan =*/ [](server_prompt_state_kind kind, llama_seq_id, llama_pos, llama_pos,
                       llama_pos & planned_p0, llama_pos & planned_p1) {
            if (kind == SERVER_PROMPT_STATE_DRAFT) {
                return false;
            }
            planned_p0 = 5504;
            planned_p1 = -1;
            return true;
        },
        /*.can_remove =*/ [](server_prompt_state_kind, llama_seq_id, llama_pos, llama_pos) { return true; },
        /*.remove =*/ [&](server_prompt_state_kind, llama_seq_id, llama_pos p0, llama_pos p1) {
            if (p0 == -1 && p1 == -1) {
                ++full_clears;
            } else {
                ++partial_removals;
            }
            return true;
        },
    };
    llama_pos planned_p0 = -1;
    const auto result = server_plan_and_remove_suffix(0, 5626, io, planned_p0);
    assert(result == SERVER_SEQ_RM_FULL_REPROCESS);
    assert(planned_p0 == 0);
    assert(partial_removals == 0);
    assert(full_clears == 2);
}

static void server_post_preflight_mutation_failure_clears_both_contexts() {
    bool main_partial = false;
    bool draft_partial = false;
    bool main_cleared = false;
    bool draft_cleared = false;
    server_seq_rm_io io {
        /*.has_draft =*/ true,
        /*.plan =*/ [](server_prompt_state_kind, llama_seq_id, llama_pos p0, llama_pos p1,
                       llama_pos & planned_p0, llama_pos & planned_p1) {
            planned_p0 = p0;
            planned_p1 = p1;
            return true;
        },
        /*.can_remove =*/ [](server_prompt_state_kind, llama_seq_id, llama_pos, llama_pos) { return true; },
        /*.remove =*/ [&](server_prompt_state_kind kind, llama_seq_id, llama_pos p0, llama_pos p1) {
            if (p0 == -1 && p1 == -1) {
                (kind == SERVER_PROMPT_STATE_MAIN ? main_cleared : draft_cleared) = true;
                return true;
            }
            if (kind == SERVER_PROMPT_STATE_MAIN) {
                main_partial = true;
                return true;
            }
            draft_partial = true;
            return false;
        },
    };
    llama_pos planned_p0 = -1;
    const auto result = server_plan_and_remove_suffix(0, 5626, io, planned_p0);
    assert(result == SERVER_SEQ_RM_MUTATION_FAILED);
    assert(main_partial && draft_partial);
    assert(main_cleared && draft_cleared);
}

int main() {
    prompt_cache_load_target_success_draft_failure_is_atomic();
    server_unsupported_removal_falls_back_to_full_reprocess();
    server_post_preflight_mutation_failure_clears_both_contexts();
    {
        common_prompt_checkpoint ckpt;
        ckpt.n_tokens = 3;
        ckpt.pos_min = 1;
        ckpt.pos_max = 2;
        ckpt.data_tgt.resize(128);
        ckpt.data_dft.resize(64);
        ckpt.data_spec.resize(32);
        assert(ckpt.size() == 224);

        ckpt.clear();
        assert(ckpt.n_tokens == 0);
        assert(ckpt.pos_min == 0);
        assert(ckpt.pos_max == 0);
        assert(ckpt.empty());
        assert(ckpt.size() == 0);
    }

    {
        server_prompt prompt = make_prompt({1, 2, 3});
        prompt.data.main.resize(64);
        prompt.data.drft.resize(32);
        auto & ckpt = prompt.checkpoints.emplace_back();
        ckpt.n_tokens = 3;
        ckpt.data_tgt.resize(16);
        ckpt.data_dft.resize(8);
        assert(prompt.size() == 120);

        const server_prompt clone = prompt.clone();
        assert(clone.n_tokens() == 3);
        assert(clone.data.size() == 96);
        assert(clone.checkpoints.size() == 1);
        assert(clone.size() == prompt.size());
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt existing = make_prompt({1, 2});
        existing.data.main.resize(700*KIB);
        cache.states.push_back(std::move(existing));

        server_prompt current = make_prompt({3, 4});
        auto * saved = cache.alloc(current, 600*KIB, 0);

        assert(saved != nullptr);
        assert(cache.size() <= cache.limit_size);
        assert(cache.states.size() == 1);
        assert(cache.states.back().data.main.size() == 600*KIB);
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt current = make_prompt({3, 4});
        auto & ckpt = current.checkpoints.emplace_back();
        ckpt.n_tokens = 4;
        ckpt.data_tgt.resize(200*KIB);

        auto * saved = cache.alloc(current, 800*KIB, 0);
        assert(saved != nullptr);
        assert(saved->checkpoints.size() == 1);
        assert(saved->size() == 1000*KIB);
        assert(cache.size() == saved->size());
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt existing = make_prompt({1, 2});
        existing.data.main.resize(100*KIB);
        cache.states.push_back(std::move(existing));

        server_prompt current = make_prompt({3, 4});
        auto & ckpt = current.checkpoints.emplace_back();
        ckpt.n_tokens = 4;
        ckpt.data_tgt.resize(200*KIB);

        assert(cache.alloc(current, 900*KIB, 0) == nullptr);
        assert(cache.states.size() == 1);
        assert(cache.size() == 100*KIB);
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt current = make_prompt({3, 4});
        assert(cache.alloc(current, 100*KIB, 0) != nullptr);
        assert(cache.alloc(current, 100*KIB, 0) == nullptr);
        assert(cache.states.size() == 1);
    }

    return 0;
}
