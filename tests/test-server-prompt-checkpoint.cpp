#include "../tools/server/server-task.h"

#include <algorithm>

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

    assert(server_prompt_checkpoint_interval(131072, 32,    0, 128) == 4096);
    assert(server_prompt_checkpoint_interval(131072, 32, 8192, 128) == 8192);
    assert(server_prompt_checkpoint_interval(120000, 32,    0, 128) == 3840);
    assert(server_prompt_checkpoint_interval(0,      32, 8192, 128) == 0);
    assert(server_prompt_checkpoint_interval(131072,  0, 8192, 128) == 0);

    assert(!server_draft_context_owns_state(false, false));
    assert( server_draft_context_owns_state(true,  false));
    assert(!server_draft_context_owns_state(true,  true));
}

static server_prompt make_prompt(const llama_tokens & tokens) {
    server_prompt prompt;
    prompt.tokens = server_tokens(tokens, false);
    return prompt;
}

static server_prompt_cache_state * insert_fake_state(
        server_prompt_cache & cache,
        const llama_tokens & tokens,
        size_t state_bytes,
        bool protect = false,
        int64_t now_us = 0,
        uint8_t marker = 0x5a) {
    server_prompt prompt = make_prompt(tokens);
    server_prompt_data data;
    data.main.resize(state_bytes, marker);
    return cache.insert(prompt, std::move(data), protect, now_us);
}

static bool observe_divergent_boundary(
        server_prompt_cache & cache,
        const llama_tokens & prefix,
        llama_token divergent_token,
        int64_t now_us) {
    llama_tokens requested_ids = prefix;
    requested_ids.push_back(divergent_token);
    server_tokens requested(requested_ids, false);
    return cache.observe_reuse(requested, prefix.size(), true, now_us);
}

static bool cache_contains_state(
        const server_prompt_cache & cache,
        const server_prompt_cache_state * expected) {
    return std::any_of(cache.states.begin(), cache.states.end(),
            [expected](const auto & state) { return &state == expected; });
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

static void restore_transaction_validation_failure_identifies_prepare_leg() {
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
        int committed = 0;
        server_prompt_restore_transaction_io io {
            /*.restore_target =*/ true,
            /*.restore_draft =*/ true,
            /*.restore_speculative =*/ true,
            /*.prepare =*/ [&](server_prompt_state_kind kind, server_prompt_state_view) {
                return kind != failed_kind;
            },
            /*.commit =*/ [&](server_prompt_state_kind) { ++committed; },
        };
        const auto result = server_prompt_restore_transaction_diagnostic(
                states[0], states[1], states[2], io);
        assert(!result.success);
        assert(result.component == failed_kind);
        assert(result.reason == SERVER_PROMPT_RESTORE_PREPARE_REJECTED);
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

static void prompt_cache_does_not_restore_just_admitted_live_state() {
    server_prompt_cache cache(1, 0);
    server_prompt current = make_prompt({1, 2, 3, 4});
    server_prompt_data data;
    data.main = {0x2a};
    const auto * admitted = cache.insert(current, std::move(data));
    assert(admitted != nullptr);

    bool restored = false;
    server_prompt_cache_state_io io {
        /*.has_draft =*/ false,
        /*.has_speculative =*/ false,
        /*.restore_transaction =*/ [&](const uint8_t *, size_t,
                                       const uint8_t *, size_t,
                                       const uint8_t *, size_t) {
            restored = true;
            return true;
        },
    };
    server_tokens requested(llama_tokens {1, 2, 9}, false);
    assert(cache.load(current, requested, 0, 1, io, admitted));
    assert(!restored);
    assert(cache.restore_attempts == 0);
    assert(current.tokens.size() == 4);
}

static void checkpoint_failed_target_save_cannot_reuse_stale_bytes() {
    common_prompt_checkpoint checkpoint;
    checkpoint.data_tgt.resize(32, 0x5a);

    checkpoint.update_tgt(nullptr, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    assert(checkpoint.data_tgt.empty());
    assert(checkpoint.empty());
}

static void speculative_draft_rollback_uses_draft_axis_and_recovers() {
    assert(server_speculative_draft_rollback_p0(4096, 63) == 4096);
    assert(server_speculative_draft_rollback_p0(0, 63) == 64);

    std::vector<std::pair<llama_pos, llama_pos>> removals;
    server_speculative_draft_rollback_io exact_io {
        /*.plan =*/ {},
        /*.remove =*/ [&](llama_pos p0, llama_pos p1) {
            removals.emplace_back(p0, p1);
            return true;
        },
    };
    llama_pos applied_p0 = -1;
    assert(server_speculative_draft_rollback(4096, exact_io, applied_p0) ==
            SERVER_SPECULATIVE_DRAFT_ROLLBACK_EXACT);
    const std::vector<std::pair<llama_pos, llama_pos>> exact_expected = {{4096, -1}};
    assert(applied_p0 == 4096 && removals == exact_expected);

    removals.clear();
    server_speculative_draft_rollback_io widened_io {
        /*.plan =*/ [](llama_pos p0, llama_pos p1, llama_pos & planned_p0, llama_pos & planned_p1) {
            assert(p0 == 4096 && p1 == -1);
            planned_p0 = 3968;
            planned_p1 = -1;
            return true;
        },
        /*.remove =*/ [&](llama_pos p0, llama_pos p1) {
            removals.emplace_back(p0, p1);
            return p0 == 3968 && p1 == -1;
        },
    };
    assert(server_speculative_draft_rollback(4096, widened_io, applied_p0) ==
            SERVER_SPECULATIVE_DRAFT_ROLLBACK_WIDENED);
    assert(applied_p0 == 3968);
    const std::vector<std::pair<llama_pos, llama_pos>> widened_expected = {{4096, -1}, {3968, -1}};
    assert(removals == widened_expected);

    removals.clear();
    server_speculative_draft_rollback_io clear_io {
        /*.plan =*/ [](llama_pos, llama_pos, llama_pos & planned_p0, llama_pos & planned_p1) {
            planned_p0 = -1;
            planned_p1 = -1;
            return true;
        },
        /*.remove =*/ [&](llama_pos p0, llama_pos p1) {
            removals.emplace_back(p0, p1);
            return p0 == -1 && p1 == -1;
        },
    };
    assert(server_speculative_draft_rollback(4096, clear_io, applied_p0) ==
            SERVER_SPECULATIVE_DRAFT_ROLLBACK_CLEARED);
    assert(applied_p0 == -1);
    const std::vector<std::pair<llama_pos, llama_pos>> clear_expected = {{4096, -1}, {-1, -1}};
    assert(removals == clear_expected);

    server_speculative_draft_rollback_io failed_io {
        /*.plan =*/ [](llama_pos, llama_pos, llama_pos & planned_p0, llama_pos & planned_p1) {
            planned_p0 = 3968;
            planned_p1 = -1;
            return true;
        },
        /*.remove =*/ [](llama_pos, llama_pos) { return false; },
    };
    assert(server_speculative_draft_rollback(4096, failed_io, applied_p0) ==
            SERVER_SPECULATIVE_DRAFT_ROLLBACK_FAILED);
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

static void cache_phase_timings_are_public() {
    server_slot_stats stats;
    stats.cache_slot_ms = 12.5;
    stats.cache_ram_save_ms = 3.0;
    stats.cache_ram_load_ms = 7.0;
    stats.cache_ram_restore_prepare_ms = 5.0;
    stats.cache_ram_restore_commit_ms = 2.0;
    stats.cache_ram_update_ms = 0.5;
    stats.cache_checkpoint_restore_ms = 2.0;
    stats.cache_checkpoint_prepare_ms = 1.5;
    stats.cache_checkpoint_commit_ms = 0.5;

    const auto value = stats.to_json();
    assert(value.at("cache_slot_ms") == 12.5);
    assert(value.at("cache_ram_save_ms") == 3.0);
    assert(value.at("cache_ram_load_ms") == 7.0);
    assert(value.at("cache_ram_restore_prepare_ms") == 5.0);
    assert(value.at("cache_ram_restore_commit_ms") == 2.0);
    assert(value.at("cache_ram_update_ms") == 0.5);
    assert(value.at("cache_checkpoint_restore_ms") == 2.0);
    assert(value.at("cache_checkpoint_prepare_ms") == 1.5);
    assert(value.at("cache_checkpoint_commit_ms") == 0.5);
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

static void prompt_cache_protection_disabled_and_missing_evidence_rejects_pin() {
    server_prompt_cache cache(0, 0);
    llama_tokens existing_tokens {7, 8};
    auto * existing = insert_fake_state(cache, existing_tokens, 3);
    assert(existing != nullptr);

    llama_tokens requested_ids {1, 2, 3, 9};
    server_tokens requested(requested_ids, false);
    assert(!cache.observe_reuse(requested, 3, true, 10));
    assert(cache.protection_candidates.empty());

    llama_tokens pin_tokens {1, 2, 3};
    assert(insert_fake_state(cache, pin_tokens, 4, true, 11) == nullptr);
    assert(cache.protected_count() == 0);
    assert(cache.states.size() == 1);
    assert(&cache.states.front() == existing);
    assert(existing->data.main.size() == 3);
}

static void prompt_cache_protection_learns_only_qualified_divergent_reuse() {
    server_prompt_cache cache(0, 0);
    cache.protection.max_bytes = 128;
    cache.protection.min_tokens = 3;
    cache.protection.min_hits = 3;

    llama_tokens prefix {10, 11, 12};
    llama_tokens request_ids = prefix;
    request_ids.push_back(99);
    server_tokens requested(request_ids, false);

    assert(!cache.observe_reuse(requested, 0, true, 1));
    assert(!cache.observe_reuse(requested, 2, true, 2));
    assert(!cache.observe_reuse(requested, 3, false, 3));
    assert(cache.protection_candidates.empty());

    assert(!cache.observe_reuse(requested, 3, true, 4));
    assert(!cache.observe_reuse(requested, 3, true, 5));
    assert(cache.observe_reuse(requested, 3, true, 6));
    assert(cache.protection_candidates.size() == 1);
    assert(cache.protection_candidates.front().hits == 3);
    assert(cache.protection_candidates.front().prefix.get_tokens() == prefix);
}

static void prompt_cache_protection_learning_survives_ordinary_subsumption() {
    server_prompt_cache cache(0, 0);
    cache.protection.max_bytes = 128;
    cache.protection.min_tokens = 3;
    cache.protection.min_hits = 2;

    llama_tokens shorter {1, 2};
    assert(insert_fake_state(cache, shorter, 1) != nullptr);
    llama_tokens boundary {1, 2, 3};
    assert(!observe_divergent_boundary(cache, boundary, 9, 10));
    assert(cache.protection_candidates.size() == 1);
    assert(cache.protection_candidates.front().hits == 1);

    llama_tokens longer {1, 2, 3, 4};
    assert(insert_fake_state(cache, longer, 1) != nullptr);
    assert(cache.states.size() == 1);
    assert(cache.states.front().prompt.tokens.size() == longer.size());
    assert(cache.protection_candidates.size() == 1);
    assert(cache.protection_candidates.front().hits == 1);

    assert(observe_divergent_boundary(cache, boundary, 10, 11));
    assert(cache.protection_candidates.front().hits == 2);
}

static void prompt_cache_protection_continuation_refreshes_residency() {
    server_prompt_cache cache(0, 0);
    cache.protection.max_bytes = 64;
    cache.protection.min_tokens = 2;
    cache.protection.min_hits = 1;

    llama_tokens prefix {20, 21};
    assert(observe_divergent_boundary(cache, prefix, 90, 100));
    auto * protected_state = insert_fake_state(cache, prefix, 2, true, 100, 0xa1);
    assert(protected_state != nullptr);
    assert(protected_state->protected_entry);
    assert(protected_state->last_used_us == 100);

    llama_tokens continuation_ids = prefix;
    continuation_ids.push_back(91);
    server_tokens continuation(continuation_ids, false);
    assert(!cache.observe_reuse(continuation, prefix.size(), false, 250));
    assert(protected_state->last_used_us == 250);
    assert(cache.protection_candidates.front().hits == 1);
}

static void prompt_cache_protection_requires_exact_boundary_and_loads_matching_prefix() {
    server_prompt_cache cache(0, 0);
    cache.protection.max_bytes = 64;
    cache.protection.min_tokens = 3;
    cache.protection.min_hits = 1;

    llama_tokens prefix {31, 32, 33};
    assert(observe_divergent_boundary(cache, prefix, 90, 10));

    llama_tokens too_long = prefix;
    too_long.push_back(34);
    assert(insert_fake_state(cache, too_long, 1, true, 11) == nullptr);
    assert(cache.protected_count() == 0);

    auto * protected_state = insert_fake_state(cache, prefix, 1, true, 12, 0x7a);
    assert(protected_state != nullptr);
    assert(protected_state->protected_entry);
    assert(protected_state->prompt.tokens.size() == prefix.size());
    assert(protected_state->data.main.front() == 0x7a);

    bool restored = false;
    server_prompt_cache_state_io io {
        /*.has_draft =*/ false,
        /*.has_speculative =*/ false,
        /*.restore_transaction =*/ [&](const uint8_t * main, size_t main_size,
                                       const uint8_t *, size_t,
                                       const uint8_t *, size_t) {
            restored = main_size == 1 && main[0] == 0x7a;
            return restored;
        },
    };
    server_prompt current = make_prompt({99});
    llama_tokens requested_ids = prefix;
    requested_ids.push_back(100);
    server_tokens requested(requested_ids, false);
    assert(cache.load(current, requested, 0, 1, io));
    assert(restored);
    assert(current.tokens.size() == prefix.size());
    assert(current.tokens[0] == prefix[0]);
    assert(current.tokens[1] == prefix[1]);
    assert(current.tokens[2] == prefix[2]);
}

static void prompt_cache_protected_entries_survive_subsumption_and_eviction() {
    {
        server_prompt_cache cache(0, 0);
        cache.protection.max_bytes = 64;
        cache.protection.min_tokens = 2;
        cache.protection.min_hits = 1;

        llama_tokens short_prompt {1};
        assert(insert_fake_state(cache, short_prompt, 1) != nullptr);
        llama_tokens prefix {1, 2};
        assert(observe_divergent_boundary(cache, prefix, 9, 10));
        auto * pinned = insert_fake_state(cache, prefix, 2, true, 10, 0xb1);
        assert(pinned != nullptr);
        assert(pinned->protected_entry);
        assert(cache.states.size() == 1);

        llama_tokens extending_prompt {1, 2, 3};
        assert(insert_fake_state(cache, extending_prompt, 1) != nullptr);
        assert(cache.states.size() == 2);
        assert(cache_contains_state(cache, pinned));
        assert(pinned->protected_entry);
        assert(pinned->prompt.tokens.size() == prefix.size());
        assert(pinned->data.main.front() == 0xb1);
    }

    {
        server_prompt_cache cache(1, 0);
        cache.protection.max_bytes = 900 * KIB;
        cache.protection.min_tokens = 2;
        cache.protection.min_hits = 1;

        llama_tokens old_tokens {9, 8};
        assert(insert_fake_state(cache, old_tokens, 500 * KIB) != nullptr);
        llama_tokens prefix {1, 2};
        assert(observe_divergent_boundary(cache, prefix, 90, 10));
        auto * pinned = insert_fake_state(cache, prefix, 200 * KIB, true, 10, 0xc1);
        assert(pinned != nullptr);

        llama_tokens new_tokens {5, 6};
        assert(insert_fake_state(cache, new_tokens, 600 * KIB) != nullptr);
        assert(cache.accounted_size() <= cache.limit_size);
        assert(cache.states.size() == 2);
        assert(cache_contains_state(cache, pinned));
        assert(pinned->protected_entry);
        assert(pinned->data.main.size() == 200 * KIB);
        assert(pinned->data.main.front() == 0xc1);
    }

    {
        server_prompt_cache cache(0, 3);
        cache.protection.max_bytes = 64;
        cache.protection.min_tokens = 2;
        cache.protection.min_hits = 1;

        llama_tokens prefix {11, 12};
        assert(observe_divergent_boundary(cache, prefix, 90, 10));
        auto * pinned = insert_fake_state(cache, prefix, 2, true, 10, 0xd1);
        assert(pinned != nullptr);
        llama_tokens ordinary {21, 22, 23};
        assert(insert_fake_state(cache, ordinary, 1) != nullptr);
        assert(cache.n_tokens() == 5);

        cache.update();
        assert(cache.states.size() == 1);
        assert(cache_contains_state(cache, pinned));
        assert(pinned->protected_entry);
        assert(cache.n_tokens() == prefix.size());
    }
}

static void prompt_cache_failed_ordinary_admission_preserves_existing_state() {
    server_prompt_cache cache(1, 0);
    llama_tokens existing_tokens {1, 2};
    auto * existing = insert_fake_state(cache, existing_tokens, 16, false, 0, 0xe1);
    assert(existing != nullptr);
    const size_t initial_size = cache.accounted_size();

    llama_tokens oversized_tokens {3, 4};
    assert(insert_fake_state(cache, oversized_tokens, 1024 * KIB + 1) == nullptr);
    assert(cache.states.size() == 1);
    assert(cache.accounted_size() == initial_size);
    assert(cache_contains_state(cache, existing));
    assert(existing->data.main.size() == 16);
    assert(existing->data.main.front() == 0xe1);
}

static void prompt_cache_ordinary_admission_reserves_protected_budget_atomically() {
    server_prompt_cache cache(1, 0);
    cache.protection.max_bytes = 800 * KIB;
    cache.protection.min_tokens = 2;
    cache.protection.min_hits = 1;

    llama_tokens protected_prefix {1, 2};
    assert(observe_divergent_boundary(cache, protected_prefix, 90, 10));
    auto * pinned = insert_fake_state(cache, protected_prefix, 700 * KIB, true, 10, 0xe2);
    assert(pinned != nullptr);
    llama_tokens existing_ordinary_tokens {8, 9};
    auto * existing_ordinary = insert_fake_state(cache, existing_ordinary_tokens, 100 * KIB, false, 11, 0xe3);
    assert(existing_ordinary != nullptr);
    const size_t original_size = cache.accounted_size();

    llama_tokens candidate_tokens {5, 6};
    assert(insert_fake_state(cache, candidate_tokens, 400 * KIB, false, 12, 0xe4) == nullptr);
    assert(cache.states.size() == 2);
    assert(cache.accounted_size() == original_size);
    assert(cache_contains_state(cache, pinned));
    assert(cache_contains_state(cache, existing_ordinary));
    assert(pinned->protected_entry);
    assert(pinned->data.main.size() == 700 * KIB);
    assert(pinned->data.main.front() == 0xe2);
    assert(existing_ordinary->data.main.size() == 100 * KIB);
    assert(existing_ordinary->data.main.front() == 0xe3);
}

static void prompt_cache_protection_caps_idle_replacement_and_demotes_old_state() {
    server_prompt_cache cache(1, 0);
    cache.protection.max_bytes = 64;
    cache.protection.min_tokens = 2;
    cache.protection.min_hits = 1;
    cache.protection.max_entries = 1;
    cache.protection.replace_after_us = 100;

    llama_tokens old_prefix {1, 2};
    assert(observe_divergent_boundary(cache, old_prefix, 90, 100));
    auto * old_state = insert_fake_state(cache, old_prefix, 4, true, 100, 0xf1);
    assert(old_state != nullptr);

    llama_tokens new_prefix {3, 4};
    assert(observe_divergent_boundary(cache, new_prefix, 91, 150));
    assert(insert_fake_state(cache, new_prefix, 4, true, 150, 0xf2) == nullptr);
    assert(cache.protected_count() == 1);
    assert(cache_contains_state(cache, old_state));
    assert(old_state->protected_entry);
    assert(old_state->data.main.front() == 0xf1);

    auto * replacement = insert_fake_state(cache, new_prefix, 4, true, 201, 0xf2);
    assert(replacement != nullptr);
    assert(cache.protected_count() == 1);
    assert(cache.states.size() == 2);
    assert(cache_contains_state(cache, old_state));
    assert(!old_state->protected_entry);
    assert(old_state->data.main.front() == 0xf1);
    assert(replacement->protected_entry);
    assert(replacement->data.main.front() == 0xf2);
}

static void prompt_cache_protection_byte_budget_rejects_recent_snapshot() {
    server_prompt_cache cache(1, 0);
    cache.protection.max_bytes = 6;
    cache.protection.min_tokens = 2;
    cache.protection.min_hits = 1;
    cache.protection.max_entries = 2;
    cache.protection.replace_after_us = 1000;

    llama_tokens old_prefix {8, 9};
    assert(observe_divergent_boundary(cache, old_prefix, 90, 100));
    auto * old_state = insert_fake_state(cache, old_prefix, 4, true, 100, 0x61);
    assert(old_state != nullptr);

    llama_tokens new_prefix {10, 11};
    assert(observe_divergent_boundary(cache, new_prefix, 91, 150));
    assert(insert_fake_state(cache, new_prefix, 4, true, 150, 0x62) == nullptr);
    assert(cache.protected_count() == 1);
    assert(cache.protected_size() == old_state->accounted_size());
    assert(cache_contains_state(cache, old_state));
    assert(old_state->protected_entry);
    assert(old_state->data.main.front() == 0x61);
}

static void prompt_cache_protection_candidate_hits_expire_after_idle_period() {
    server_prompt_cache cache(0, 0);
    cache.protection.max_bytes = 64;
    cache.protection.min_tokens = 2;
    cache.protection.min_hits = 2;
    cache.protection.replace_after_us = 100;

    llama_tokens prefix {71, 72};
    assert(!observe_divergent_boundary(cache, prefix, 90, 100));
    assert(cache.protection_candidates.front().hits == 1);
    assert(cache.protection_candidates.front().last_used_us == 100);

    assert(!observe_divergent_boundary(cache, prefix, 91, 201));
    assert(cache.protection_candidates.front().hits == 1);
    assert(cache.protection_candidates.front().last_used_us == 201);
    assert(observe_divergent_boundary(cache, prefix, 92, 250));
    assert(cache.protection_candidates.front().hits == 2);
}

static void prompt_cache_protected_token_cap_rejects_promotion() {
    server_prompt_cache cache(0, 5);
    cache.protection.max_bytes = 64;
    cache.protection.min_tokens = 3;
    cache.protection.min_hits = 1;
    cache.protection.max_entries = 2;
    cache.protection.replace_after_us = 1000;

    llama_tokens old_prefix {81, 82, 83};
    assert(observe_divergent_boundary(cache, old_prefix, 90, 10));
    auto * old_state = insert_fake_state(cache, old_prefix, 2, true, 10, 0x65);
    assert(old_state != nullptr);

    llama_tokens new_prefix {91, 92, 93};
    assert(observe_divergent_boundary(cache, new_prefix, 91, 20));
    assert(insert_fake_state(cache, new_prefix, 2, true, 20, 0x66) == nullptr);
    assert(cache.protected_count() == 1);
    assert(cache.protected_size() == old_state->accounted_size());
    assert(cache.n_tokens() == old_prefix.size());
    assert(cache_contains_state(cache, old_state));
    assert(old_state->protected_entry);
    assert(old_state->data.main.front() == 0x65);
}

static void prompt_cache_protection_candidates_remain_bounded() {
    {
        server_prompt_cache cache(0, 0);
        cache.protection.max_bytes = 1;
        cache.protection.min_tokens = 1;
        cache.protection.min_hits = 2;

        for (llama_token token = 1; token <= 65; ++token) {
            llama_tokens prefix {token};
            assert(!observe_divergent_boundary(cache, prefix, 1000 + token, token));
        }
        assert(cache.protection_candidates.size() == 64);
        assert(cache.protection_candidates.front().prefix.get_tokens().front() == 2);
        assert(cache.protection_candidates.back().prefix.get_tokens().front() == 65);
    }

    {
        server_prompt_cache cache(0, 0);
        cache.protection.max_bytes = 1;
        cache.protection.min_tokens = 1;
        cache.protection.min_hits = 2;

        constexpr size_t max_candidate_tokens = 1024 * 1024;
        llama_tokens max_prefix(max_candidate_tokens, 42);
        assert(!observe_divergent_boundary(cache, max_prefix, 43, 1));
        assert(cache.protection_candidates.size() == 1);
        assert(cache.protection_candidates.front().prefix.size() == max_candidate_tokens);

        llama_tokens small_prefix {99};
        assert(!observe_divergent_boundary(cache, small_prefix, 100, 2));
        assert(cache.protection_candidates.size() == 1);
        assert(cache.protection_candidates.front().prefix.get_tokens().front() == 99);

        llama_tokens oversized_prefix(max_candidate_tokens + 1, 42);
        assert(!observe_divergent_boundary(cache, oversized_prefix, 43, 3));
        assert(cache.protection_candidates.size() == 1);
        assert(cache.protection_candidates.front().prefix.get_tokens().front() == 99);
    }
}

int main() {
    prompt_cache_protection_disabled_and_missing_evidence_rejects_pin();
    prompt_cache_protection_learns_only_qualified_divergent_reuse();
    prompt_cache_protection_learning_survives_ordinary_subsumption();
    prompt_cache_protection_continuation_refreshes_residency();
    prompt_cache_protection_requires_exact_boundary_and_loads_matching_prefix();
    prompt_cache_protected_entries_survive_subsumption_and_eviction();
    prompt_cache_failed_ordinary_admission_preserves_existing_state();
    prompt_cache_ordinary_admission_reserves_protected_budget_atomically();
    prompt_cache_protection_caps_idle_replacement_and_demotes_old_state();
    prompt_cache_protection_byte_budget_rejects_recent_snapshot();
    prompt_cache_protection_candidate_hits_expire_after_idle_period();
    prompt_cache_protected_token_cap_rejects_promotion();
    prompt_cache_protection_candidates_remain_bounded();
    prompt_cache_ranks_safe_restorable_prefix_before_lexical_lcp();
    prompt_cache_load_target_success_draft_failure_is_atomic();
    prompt_cache_does_not_restore_just_admitted_live_state();
    restore_transaction_validation_failures_are_atomic();
    restore_transaction_validation_failure_identifies_prepare_leg();
    speculative_rollback_checkpoint_boundary();
    checkpoint_failed_target_save_cannot_reuse_stale_bytes();
    speculative_draft_rollback_uses_draft_axis_and_recovers();
    server_unsupported_removal_falls_back_to_full_reprocess();
    server_post_preflight_mutation_failure_clears_both_contexts();
    server_planned_removal_preserves_atomic_media_chunks();
    cache_phase_timings_are_public();
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
                /*.spec =*/ { },
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
