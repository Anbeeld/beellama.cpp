#include "../tools/server/server-task.h"

#undef NDEBUG
#include <cassert>

static constexpr size_t KIB = 1024;

static void speculative_rollback_checkpoint_boundary() {
    constexpr uint32_t reserve = 8;

    assert(!server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_NO, reserve, reserve + 1));
    assert(!server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_PART, reserve, reserve + 1));
    assert( server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_FULL, reserve, 1));
    assert(!server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_RS, reserve, 1));
    assert(!server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_RS, reserve, reserve));
    assert( server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_RS, reserve, reserve + 1));
    assert( server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_RS, 0, 1));

    assert(!server_prompt_reuse_requires_checkpoint_search(false, 0, 528));
    assert( server_prompt_reuse_requires_checkpoint_search(false, 512, 512));
    assert( server_prompt_reuse_requires_checkpoint_search(true, 0, 528));

    assert(server_prompt_reuse_alignment(0) == 1);
    assert(server_prompt_reuse_alignment(128) == 128);
    assert(server_prompt_reuse_alignment(192) == 192);

    assert(server_prompt_checkpoint_boundary(795, 132,   1) == 663);
    assert(server_prompt_checkpoint_boundary(795, 132, 128) == 640);
    assert(server_prompt_checkpoint_boundary(795,   4, 128) == 768);
    assert(server_prompt_checkpoint_boundary(3,     4, 128) == 0);
}

static server_prompt make_prompt(const llama_tokens & tokens) {
    server_prompt prompt;
    prompt.tokens = server_tokens(tokens, false);
    return prompt;
}

static common_memory_seq_rm_result test_seq_rm_suffix(
        llama_seq_id seq_id,
        llama_pos requested_p0,
        const server_tokens & prompt_tokens,
        const common_memory_seq_rm_io & io,
        llama_pos & planned_p0) {
    const auto normalize_p0 = [&](llama_pos value) {
        return value > 0 ? prompt_tokens.pos_next(prompt_tokens.size_up_to_pos(value)) : value;
    };
    return common_memory_seq_rm_suffix(seq_id, requested_p0, io, normalize_p0, planned_p0);
}

static void prompt_cache_load_target_success_draft_failure_is_atomic() {
    server_prompt_cache cache(1, 0);
    server_prompt_cache_state saved;
    saved.prompt = make_prompt({1, 2, 3});
    saved.data.main.resize(16);
    saved.data.drft.resize(8);
    cache.states.push_back(std::move(saved));

    server_prompt current = make_prompt({9});
    current.checkpoints.emplace_back().data_tgt.resize(4);
    server_tokens requested(llama_tokens {1, 2, 4}, false);

    bool restored_main = false;
    bool restored_draft = false;
    bool cleared_main = false;
    bool cleared_draft = false;
    int main_state = 90;
    int draft_state = 80;
    server_prompt_cache_state_io io {
        /*.has_draft =*/ true,
        /*.has_speculative =*/ false,
        /*.restore_transaction =*/ [&](const uint8_t * main, size_t main_size,
                                       const uint8_t * drft, size_t drft_size,
                                       const uint8_t *, size_t) {
            server_prompt_restore_transaction_io tx {
                /*.restore_target =*/ true,
                /*.restore_draft =*/ true,
                /*.restore_speculative =*/ false,
                /*.prepare =*/ [&](server_prompt_state_kind kind, server_prompt_state_view state) {
                    if (kind == SERVER_PROMPT_STATE_MAIN) {
                        return state.data == main && state.size == main_size;
                    }
                    return state.data != drft || state.size != drft_size;
                },
                /*.commit =*/ [&](server_prompt_state_kind kind) {
                    if (kind == SERVER_PROMPT_STATE_MAIN) {
                        restored_main = true;
                        main_state = int(main_size);
                    } else {
                        restored_draft = true;
                        draft_state = int(drft_size);
                    }
                },
            };
            return server_prompt_restore_transaction(
                    { main, main_size }, { drft, drft_size }, {}, tx);
        },
    };

    assert(!cache.load(current, requested, 0, 1, io));
    assert(!restored_main && !restored_draft);
    assert(!cleared_main && !cleared_draft);
    assert(main_state == 90 && draft_state == 80);
    assert(cache.states.size() == 1);
    assert(current.tokens.size() == 1);
    assert(current.tokens[0] == 9);
    assert(current.checkpoints.size() == 1);
}

static void restore_transaction_validation_failures_are_atomic() {
    const server_prompt_state_view states[] = {
        { reinterpret_cast<const uint8_t *>("target"), 6 },
        { reinterpret_cast<const uint8_t *>("draft"), 5 },
        { reinterpret_cast<const uint8_t *>("spec"), 4 },
    };
    const server_prompt_state_kind kinds[] = {
        SERVER_PROMPT_STATE_MAIN,
        SERVER_PROMPT_STATE_DRAFT,
        SERVER_PROMPT_STATE_SPECULATIVE,
    };

    for (const auto failed_kind : kinds) {
        int prepared = 0;
        int committed = 0;
        server_prompt_restore_transaction_io io {
            /*.restore_target =*/ true,
            /*.restore_draft =*/ true,
            /*.restore_speculative =*/ true,
            /*.prepare =*/ [&](server_prompt_state_kind kind, server_prompt_state_view) {
                ++prepared;
                return kind != failed_kind;
            },
            /*.commit =*/ [&](server_prompt_state_kind) { ++committed; },
        };
        assert(!server_prompt_restore_transaction(states[0], states[1], states[2], io));
        assert(prepared >= 1 && prepared <= 3);
        assert(committed == 0);
    }
}

static void prompt_cache_ranks_safe_restorable_prefix_before_lexical_lcp() {
    server_prompt_cache cache(1, 0);
    server_prompt_cache_state saved;
    saved.prompt = make_prompt({1, 2});
    saved.data.main = {0x2a};
    cache.states.push_back(std::move(saved));

    // The live slot has the larger lexical prefix, but no durable checkpoint
    // at the divergent boundary.  The self-contained RAM prompt is fully
    // restorable and must therefore win despite its shorter lexical LCP.
    server_prompt current = make_prompt({1, 2, 3, 4});
    server_tokens requested(llama_tokens {1, 2, 3, 9}, false);
    bool restored = false;
    server_prompt_cache_state_io io {
        /*.has_draft =*/ false,
        /*.has_speculative =*/ false,
        /*.restore_transaction =*/ [&](const uint8_t * main, size_t main_size,
                                       const uint8_t *, size_t,
                                       const uint8_t *, size_t) {
            restored = main_size == 1 && main[0] == 0x2a;
            return restored;
        },
    };

    assert(cache.load(current, requested, 0, 128, io));
    assert(restored);
    assert(current.tokens.size() == 2);
}

static void checkpoint_failed_target_save_cannot_reuse_stale_bytes() {
    common_prompt_checkpoint checkpoint;
    checkpoint.data_tgt.resize(32, 0x5a);

    checkpoint.update_tgt(nullptr, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    assert(checkpoint.data_tgt.empty());
    assert(checkpoint.empty());
}

static void server_unsupported_removal_falls_back_to_full_reprocess() {
    server_tokens prompt_tokens(llama_tokens(5626, 1), false);
    int partial_removals = 0;
    int full_clears = 0;
    common_memory_seq_rm_io io {
        /*.has_draft =*/ true,
        /*.plan =*/ [](common_memory_context_kind kind, llama_seq_id, llama_pos, llama_pos,
                       llama_pos & planned_p0, llama_pos & planned_p1) {
            if (kind == COMMON_MEMORY_CONTEXT_DRAFT) {
                return false;
            }
            planned_p0 = 5504;
            planned_p1 = -1;
            return true;
        },
        /*.can_remove =*/ [](common_memory_context_kind, llama_seq_id, llama_pos, llama_pos) { return true; },
        /*.remove =*/ [&](common_memory_context_kind, llama_seq_id, llama_pos p0, llama_pos p1) {
            if (p0 == -1 && p1 == -1) {
                ++full_clears;
            } else {
                ++partial_removals;
            }
            return true;
        },
    };
    llama_pos planned_p0 = -1;
    const auto result = test_seq_rm_suffix(0, 5626, prompt_tokens, io, planned_p0);
    assert(result == COMMON_MEMORY_SEQ_RM_FULL_REPROCESS);
    assert(planned_p0 == 0);
    assert(partial_removals == 0);
    assert(full_clears == 2);
}

static void server_post_preflight_mutation_failure_clears_both_contexts() {
    server_tokens prompt_tokens(llama_tokens(5626, 1), false);
    bool main_partial = false;
    bool draft_partial = false;
    bool main_cleared = false;
    bool draft_cleared = false;
    common_memory_seq_rm_io io {
        /*.has_draft =*/ true,
        /*.plan =*/ [](common_memory_context_kind, llama_seq_id, llama_pos p0, llama_pos p1,
                       llama_pos & planned_p0, llama_pos & planned_p1) {
            planned_p0 = p0;
            planned_p1 = p1;
            return true;
        },
        /*.can_remove =*/ [](common_memory_context_kind, llama_seq_id, llama_pos, llama_pos) { return true; },
        /*.remove =*/ [&](common_memory_context_kind kind, llama_seq_id, llama_pos p0, llama_pos p1) {
            if (p0 == -1 && p1 == -1) {
                (kind == COMMON_MEMORY_CONTEXT_TARGET ? main_cleared : draft_cleared) = true;
                return true;
            }
            if (kind == COMMON_MEMORY_CONTEXT_TARGET) {
                main_partial = true;
                return true;
            }
            draft_partial = true;
            return false;
        },
    };
    llama_pos planned_p0 = -1;
    const auto result = test_seq_rm_suffix(0, 5626, prompt_tokens, io, planned_p0);
    assert(result == COMMON_MEMORY_SEQ_RM_MUTATION_FAILED);
    assert(main_partial && draft_partial);
    assert(main_cleared && draft_cleared);
}

static void server_planned_removal_preserves_atomic_media_chunks() {
    mtmd::input_chunks chunks(mtmd_test_create_input_chunks());
    server_tokens prompt_tokens(chunks, true);

    const llama_pos requested_p0 = prompt_tokens.pos_next();
    const llama_pos inside_media = 6;
    const llama_pos media_end = prompt_tokens.pos_next(prompt_tokens.size_up_to_pos(inside_media));
    assert(media_end > inside_media);

    llama_pos removed_p0 = -1;
    common_memory_seq_rm_io io {
        /*.has_draft =*/ false,
        /*.plan =*/ [&](common_memory_context_kind, llama_seq_id, llama_pos, llama_pos,
                       llama_pos & planned_p0, llama_pos & planned_p1) {
            planned_p0 = inside_media;
            planned_p1 = -1;
            return true;
        },
        /*.can_remove =*/ [](common_memory_context_kind, llama_seq_id, llama_pos, llama_pos) { return true; },
        /*.remove =*/ [&](common_memory_context_kind, llama_seq_id, llama_pos p0, llama_pos) {
            removed_p0 = p0;
            return true;
        },
    };

    llama_pos planned_p0 = -1;
    const auto result = test_seq_rm_suffix(0, requested_p0, prompt_tokens, io, planned_p0);
    assert(result == COMMON_MEMORY_SEQ_RM_APPLIED);
    assert(planned_p0 == media_end);
    assert(removed_p0 == media_end);
}

static void prompt_cache_snapshot_restore_evict_stress() {
    server_prompt_cache cache(0, 0);

    int logical_state = 0;
    server_prompt_cache_state_io io {
        /*.has_draft =*/ false,
        /*.has_speculative =*/ false,
        /*.restore_transaction =*/ [&](const uint8_t * main, size_t main_size,
                                       const uint8_t *, size_t,
                                       const uint8_t *, size_t) {
            assert(main_size > 0);
            logical_state = main[0];
            return true;
        },
    };

    for (int i = 0; i < 10000; ++i) {
        server_prompt source = make_prompt({i + 1, i + 20001});
        auto & checkpoint = source.checkpoints.emplace_back();
        checkpoint.n_tokens = 2;
        checkpoint.data_tgt.resize(4*KIB);

        auto * admitted = cache.alloc(source, 32*KIB, 0);
        assert(admitted != nullptr);
        admitted->data.main.front() = uint8_t(i);

        server_prompt destination;
        server_tokens requested(llama_tokens {i + 1, i + 20001, i + 40001}, false);
        assert(cache.load(destination, requested, 0, 1, io));
        assert(destination.tokens.size() == source.tokens.size());
        assert(logical_state == uint8_t(i));
        assert(cache.states.size() == 1);
        assert(cache.erase(admitted));
        assert(cache.states.empty());
        assert(cache.accounted_size() == 0);
    }

    assert(cache.admission_successes == 10000);
    assert(cache.restore_successes == 10000);
    assert(cache.admission_failures == 0);
    assert(cache.restore_failures == 0);
}

int main() {
    prompt_cache_ranks_safe_restorable_prefix_before_lexical_lcp();
    prompt_cache_load_target_success_draft_failure_is_atomic();
    restore_transaction_validation_failures_are_atomic();
    speculative_rollback_checkpoint_boundary();
    checkpoint_failed_target_save_cannot_reuse_stale_bytes();
    server_unsupported_removal_falls_back_to_full_reprocess();
    server_post_preflight_mutation_failure_clears_both_contexts();
    server_planned_removal_preserves_atomic_media_chunks();
    prompt_cache_snapshot_restore_evict_stress();
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
        auto & ckpt = prompt.checkpoints.emplace_back();
        ckpt.n_tokens = 3;
        ckpt.data_tgt.resize(16);
        ckpt.data_dft.resize(8);

        const server_prompt clone = prompt.clone();
        assert(clone.n_tokens() == 3);
        assert(clone.checkpoints.size() == 1);
        assert(clone.checkpoints.front().data_tgt.storage_id() ==
                prompt.checkpoints.front().data_tgt.storage_id());

        const void * shared_storage = clone.checkpoints.front().data_tgt.storage_id();
        prompt.checkpoints.front().data_tgt.resize(24);
        assert(prompt.checkpoints.front().data_tgt.storage_id() != shared_storage);
        assert(clone.checkpoints.front().data_tgt.size() == 16);

        server_prompt_cache_state state {
            /*.prompt =*/ std::move(prompt),
            /*.data =*/ {
                /*.main =*/ std::vector<uint8_t>(64),
                /*.drft =*/ std::vector<uint8_t>(32),
            },
        };
        assert(state.accounted_size() == 128);
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt_cache_state existing;
        existing.prompt = make_prompt({1, 2});
        existing.data.main.resize(700*KIB);
        cache.states.push_back(std::move(existing));

        server_prompt current = make_prompt({3, 4});
        auto * saved = cache.alloc(current, 600*KIB, 0);

        assert(saved != nullptr);
        assert(cache.accounted_size() <= cache.limit_size);
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
        assert(saved->prompt.checkpoints.size() == 1);
        assert(saved->accounted_size() == 1000*KIB);
        assert(cache.accounted_size() == saved->accounted_size());
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt_cache_state existing;
        existing.prompt = make_prompt({1, 2});
        existing.data.main.resize(100*KIB);
        cache.states.push_back(std::move(existing));

        server_prompt current = make_prompt({3, 4});
        auto & ckpt = current.checkpoints.emplace_back();
        ckpt.n_tokens = 4;
        ckpt.data_tgt.resize(200*KIB);

        assert(cache.alloc(current, 900*KIB, 0) == nullptr);
        assert(cache.states.size() == 1);
        assert(cache.accounted_size() == 100*KIB);
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt current = make_prompt({3, 4});
        assert(cache.alloc(current, 100*KIB, 0) != nullptr);
        assert(cache.alloc(current, 100*KIB, 0) == nullptr);
        assert(cache.states.size() == 1);
    }

    {
        server_prompt source = make_prompt({7, 8, 9});
        source.checkpoints.emplace_back().data_tgt.resize(200*KIB);
        server_prompt_cache cache(0, 0);
        server_prompt_cache_state first;
        first.prompt = source.clone();
        first.data.main.resize(10*KIB);
        server_prompt_cache_state second;
        second.prompt = source.clone();
        second.data.main.resize(20*KIB);
        cache.states.push_back(std::move(first));
        cache.states.push_back(std::move(second));
        assert(cache.accounted_size() == 230*KIB);
    }

    return 0;
}
