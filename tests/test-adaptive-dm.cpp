#include "../tools/server/server-adaptive-dm.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <vector>

static void assert_close(float actual, float expected) {
    assert(std::fabs(actual - expected) < 1e-6f);
}

static std::vector<int> build_candidates(int base_n_max) {
    int out[160];
    const int n = server_adaptive_dm_build_candidates(base_n_max, out, 160);
    return std::vector<int>(out, out + n);
}

static void assert_candidates(int base_n_max, const std::vector<int> & expected) {
    assert(build_candidates(base_n_max) == expected);
}

static void observe_profit_cycle(
        server_adaptive_dm_state & state,
        int requested_n_max,
        int actual_n_draft,
        int n_accepted,
        float cycle_ms) {
    state.observe_profit_acceptance(actual_n_draft, n_accepted);
    state.observe_profit_timing(requested_n_max, actual_n_draft, n_accepted,
            0.0f, cycle_ms, 0.0f, cycle_ms);
}

static common_params_speculative make_dflash_spec(int n_max) {
    common_params_speculative spec;
    spec.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
    spec.draft.n_max = n_max;
    spec.draft.n_min = 0;
    spec.draft.p_split = 0.1f;
    spec.draft.p_min = 0.0f;
    spec.draft.backend_sampling = true;
    return spec;
}

int main() {
    assert(server_adaptive_dm_probe_n_max(8, 0.25f) == 2);
    assert(server_adaptive_dm_probe_n_max(16, 0.25f) == 4);
    assert(server_adaptive_dm_probe_n_max(8, 0.50f) == 4);
    assert(server_adaptive_dm_probe_n_max(1, 0.25f) == 1);
    assert(server_adaptive_dm_probe_n_max(0, 0.25f) == 0);

    assert_candidates(0,  {0});
    assert_candidates(1,  {0, 1});
    assert_candidates(2,  {0, 1, 2});
    assert_candidates(8,  {0, 1, 2, 3, 4, 5, 6, 7, 8});
    assert_candidates(11, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    assert_candidates(16, {0, 1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16});

    assert(server_adaptive_dm_next_explore_depth(0, 8, 0.25f) == 2);
    assert(server_adaptive_dm_next_explore_depth(2, 8, 0.25f) == 3);
    assert(server_adaptive_dm_next_explore_depth(4, 8, 0.25f) == 5);
    assert(server_adaptive_dm_next_explore_depth(6, 8, 0.25f) == 7);
    assert(server_adaptive_dm_next_explore_depth(8, 8, 0.25f) == 8);

    {
        server_adaptive_dm_state explore;
        assert(explore.profit_explore_depth_for_step(15, 15, 1) == 14);
        assert(explore.profit_explore_depth_for_step(15, 15, 2) == 13);
        assert(explore.profit_explore_depth_for_step(15, 15, 3) == 12);
        assert(explore.profit_explore_depth_for_step(15, 15, 4) == 11);
        assert(explore.profit_explore_depth_for_step(15, 15, 5) == 10);
        assert(explore.profit_explore_depth_for_step(15, 15, 8) == 14);
        assert(explore.profit_explore_depth_for_step(8, 15, 1) == 7);
        assert(explore.profit_explore_depth_for_step(8, 15, 2) == 9);
        assert(explore.profit_explore_depth_for_step(8, 15, 7) == 4);

        int active_order[SERVER_ADAPTIVE_DM_PROFIT_CANDIDATES];
        const int n_active = explore.profit_active_explore_depths(
                15, 15, active_order, SERVER_ADAPTIVE_DM_PROFIT_CANDIDATES);
        assert(n_active > 0);
        assert(n_active <= 7);
        for (int i = 0; i < n_active; ++i) {
            assert(active_order[i] >= 4);
        }
    }

    {
        server_adaptive_dm_state explore_ready;
        explore_ready.dm_profit_min_samples = 1;
        assert(explore_ready.profit_next_unready_explore_depth(10, 15, 1) == 9);
        const int ready_depths[] = {1, 2, 3, 4, 5, 6, 7, 8,
                9, 10, 11, 12, 13, 14, 15};
        for (const int depth : ready_depths) {
            for (int i = 0; i < 6; ++i) {
                observe_profit_cycle(explore_ready, depth, depth, depth >= 8 ? 1 : 0, 80.0f);
            }
        }
        assert(explore_ready.profit_next_unready_explore_depth(10, 15, 1) == 0);
    }

    {
        server_adaptive_dm_state explore_confirm;
        explore_confirm.dm_profit_min_samples = 3;
        explore_confirm.adaptive_n_max = 15;
        explore_confirm.profit_last_recommended_n = 15;

        for (int i = 0; i < 3; ++i) {
            observe_profit_cycle(explore_confirm, 14, 14, 2, 80.0f);
        }
        assert(explore_confirm.profit_next_unready_explore_depth(15, 15, 1) == 14);
        assert(explore_confirm.profit_next_unready_explore_depth(15, 15, 2) == 14);

        for (int i = 0; i < 3; ++i) {
            observe_profit_cycle(explore_confirm, 14, 14, 2, 80.0f);
        }
        assert(explore_confirm.profit_next_unready_explore_depth(15, 15, 1) == 13);
    }

    {
        server_adaptive_dm_state strong_max;
        strong_max.dm_profit_min_samples = 1;
        strong_max.adaptive_n_max = 15;
        strong_max.profit_last_recommended_n = 15;

        strong_max.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(strong_max, 15, 15, 10, 60.0f);
        }
        assert(strong_max.profit_next_unready_explore_depth(15, 15, 1) == 0);
    }

    {
        server_adaptive_dm_state weak_max;
        weak_max.dm_profit_min_samples = 1;
        weak_max.adaptive_n_max = 15;
        weak_max.profit_last_recommended_n = 15;

        weak_max.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(weak_max, 15, 15, 1, 95.0f);
        }
        assert(weak_max.profit_next_unready_explore_depth(15, 15, 1) == 14);
    }

    {
        server_adaptive_dm_state strong_mid;
        strong_mid.dm_profit_min_samples = 1;
        strong_mid.adaptive_n_max = 8;
        strong_mid.profit_last_recommended_n = 8;

        strong_mid.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(strong_mid, 8, 8, 7, 50.0f);
        }
        assert(strong_mid.profit_next_unready_explore_depth(8, 15, 1) == 9);
    }

    {
        server_adaptive_dm_state initial_large;
        initial_large.dm_profit_min_samples = 1;
        assert(initial_large.profit_next_unready_probe_depth(15) == 4);
        observe_profit_cycle(initial_large, 4, 4, 0, 80.0f);
        assert(initial_large.profit_next_unready_probe_depth(15) == 8);
        observe_profit_cycle(initial_large, 8, 8, 1, 80.0f);
        assert(initial_large.profit_next_unready_probe_depth(15) == 15);
        observe_profit_cycle(initial_large, 15, 15, 1, 80.0f);
        assert(initial_large.profit_next_unready_probe_depth(15) < 0);
    }

    {
        const float p[] = {0.80f, 0.70f, 0.50f, 0.25f};
        const int32_t samples[] = {6, 6, 6, 6};
        bool ready = false;
        const float expected_accept = server_adaptive_dm_survival_expected_accept(p, samples, 4, 4, 6, &ready);
        assert(ready);
        assert_close(expected_accept, 2.25f);
    }

    {
        const float p[] = {0.80f, 0.70f, 0.50f, 0.25f};
        const int32_t samples[] = {6, 6, 5, 6};
        bool ready = true;
        (void) server_adaptive_dm_survival_expected_accept(p, samples, 4, 4, 6, &ready);
        assert(!ready);
    }

    assert_close(server_adaptive_dm_score(2.1f, 80.0f), 26.25f);
    assert_close(server_adaptive_dm_score(2.6f, 120.0f), 21.666666f);

    {
        const int candidates[] = {2, 4};
        const bool ready[] = {true, true};
        const bool estimated[] = {false, false};
        const float expected_accept[] = {1.1f, 1.6f};
        const float cycle_ms[] = {80.0f, 120.0f};
        const auto best = server_adaptive_dm_best_profit_candidate(candidates, 2, ready, estimated, expected_accept, cycle_ms);
        assert(best.n == 2);
        assert_close(best.score, 26.25f);
        assert(!best.estimated);
    }

    {
        const int candidates[] = {512, 2048};
        const bool ready[] = {true, true};
        const bool estimated[] = {false, false};
        const float expected_accept[] = {1.108f, 1.495f};
        const float cycle_ms[] = {90.2f, 116.6f};
        const auto best = server_adaptive_dm_best_profit_candidate(candidates, 2, ready, estimated, expected_accept, cycle_ms);
        assert(best.n == 512);
        assert(!best.estimated);
    }

    {
        int cand[160];
        const int nc8 = server_adaptive_dm_build_candidates(8, cand, 160);
        assert(server_adaptive_dm_apply_profit_hysteresis(2, 4, 30.0f, 30.9f, 0.06f, 0.02f, cand, nc8) == 2);
        assert(server_adaptive_dm_apply_profit_hysteresis(2, 8, 30.0f, 33.0f, 0.06f, 0.02f, cand, nc8) == 4);
        assert(server_adaptive_dm_apply_profit_hysteresis(6, 2, 30.0f, 31.5f, 0.06f, 0.02f, cand, nc8) == 4);

        int cand16[160];
        const int nc16 = server_adaptive_dm_build_candidates(16, cand16, 160);
        assert(server_adaptive_dm_apply_profit_hysteresis(16, 8, 30.0f, 36.0f, 0.06f, 0.02f, cand16, nc16) == 14);
    }

    assert(server_adaptive_dm_should_preserve_for_continuation(0.995f, 1.000f));
    assert(!server_adaptive_dm_should_preserve_for_continuation(0.995f, 0.500f));
    assert(!server_adaptive_dm_should_preserve_for_continuation(0.500f, 1.000f));
    assert(!server_adaptive_dm_should_preserve_for_continuation(0.0f, 0.0f));

    server_adaptive_dm_state state;
    state.adaptive_n_max = 6;
    state.profit_pos_accept_ewma[2] = 0.75f;
    state.profit_pos_samples[2] = 9;
    state.profit_depth[4].samples = 3;
    state.profit_baseline.samples = 2;
    state.profit_has_key = true;
    state.profit_key = {};
    state.profit_key.base_n_max = 8;
    state.profit_key.draft_n_min = 0;
    state.profit_key.draft_p_split = 0.1f;
    state.profit_key.draft_p_min = 0.0f;
    state.profit_key.draft_backend_sampling = 1;
    state.profit_key.draft_cache_type_k = GGML_TYPE_F16;
    state.profit_key.draft_cache_type_v = GGML_TYPE_F16;
    state.profit_pending = true;
    state.profit_last_recommended_n = 4;
    state.profit_consecutive_below_profit = 2;
    state.profit_cycles_since_baseline = 9;
    state.profit_baseline_probe_pending = true;
    state.profit_baseline_probe_resume_n = 6;
    state.reset_request_state();

    assert(state.adaptive_n_max == -1);
    // request resets are prompt-change boundaries. Profit telemetry is keyed
    // by configuration and survives, but the last request's recommendation
    // must not carry over to a different prompt class.
    assert(state.profit_pos_accept_ewma[2] == 0.75f);
    assert(state.profit_pos_samples[2] == 9);
    assert(state.profit_depth[4].samples == 3);
    assert(state.profit_baseline.samples == 2);
    assert(state.profit_has_key == true);
    assert(!state.profit_pending);
    assert(state.profit_last_recommended_n == -1);
    assert(state.profit_consecutive_below_profit == 0);
    assert(state.profit_cycles_since_baseline == 0);
    assert(!state.profit_baseline_probe_pending);
    assert(state.profit_baseline_probe_resume_n == -1);
    assert(state.profit_off_probe_failures == 0);

    // A non-preserved prompt reset keeps measured telemetry but must not seed
    // production from best-history samples when the previous request was off.
    {
        server_adaptive_dm_state learned;
        learned.dm_profit_min_samples = 1;
        learned.profit_has_key = true;
        learned.profit_key = {};
        learned.profit_key.base_n_max = 8;
        learned.adaptive_n_max = 0;
        learned.profit_last_recommended_n = 0;
        learned.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
        observe_profit_cycle(learned, 2, 2, 0, 70.0f);
        observe_profit_cycle(learned, 4, 4, 1, 58.0f);
        observe_profit_cycle(learned, 8, 8, 0, 90.0f);

        learned.reset_request_state();
        assert(learned.adaptive_n_max == -1);
        assert(learned.profit_last_recommended_n == -1);
        assert(!learned.profit_pending);
        assert(learned.profit_baseline.samples == 1);
        assert(learned.profit_depth[4].samples == 1);
    }

    {
        server_adaptive_dm_state baseline_winner;
        baseline_winner.dm_profit_min_samples = 1;
        baseline_winner.profit_has_key = true;
        baseline_winner.profit_key = {};
        baseline_winner.profit_key.base_n_max = 8;
        baseline_winner.adaptive_n_max = 4;
        baseline_winner.profit_last_recommended_n = 4;
        baseline_winner.observe_profit_timing(0, 0, 0, 0.0f, 25.0f, 0.0f, 25.0f);
        observe_profit_cycle(baseline_winner, 4, 4, 0, 70.0f);

        baseline_winner.reset_request_state();
        assert(baseline_winner.adaptive_n_max == -1);
        assert(baseline_winner.profit_last_recommended_n == -1);
        assert(baseline_winner.profit_baseline.samples == 1);
        assert(baseline_winner.profit_depth[4].samples == 1);
    }

    {
        server_adaptive_dm_state repeated_prompt;
        repeated_prompt.dm_profit_min_samples = 1;
        repeated_prompt.profit_has_key = true;
        repeated_prompt.profit_key = {};
        repeated_prompt.profit_key.base_n_max = 15;
        repeated_prompt.adaptive_n_max = 8;
        repeated_prompt.profit_last_recommended_n = 8;
        repeated_prompt.observe_profit_timing(0, 0, 0, 0.0f, 24.0f, 0.0f, 24.0f);
        observe_profit_cycle(repeated_prompt, 8, 8, 1, 70.0f);

        repeated_prompt.reset_request_state(true);
        assert(repeated_prompt.adaptive_n_max == 8);
        assert(repeated_prompt.profit_last_recommended_n == 8);
        assert(repeated_prompt.profit_active_episode_n == 8);
        assert(!repeated_prompt.profit_pending);
        assert(repeated_prompt.profit_baseline.samples == 1);
        assert(repeated_prompt.profit_depth[8].samples == 1);
    }

    {
        server_adaptive_dm_state warm_start;
        warm_start.dm_profit_min_samples = 1;
        warm_start.profit_has_key = true;
        warm_start.profit_key = {};
        warm_start.profit_key.base_n_max = 15;
        warm_start.adaptive_n_max = 12;
        warm_start.profit_last_recommended_n = 12;
        warm_start.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
        observe_profit_cycle(warm_start, 12, 12, 4, 70.0f);
        observe_profit_cycle(warm_start, 15, 15, 5, 60.0f);

        warm_start.reset_request_state(false);
        assert(warm_start.adaptive_n_max == 12);
        assert(warm_start.profit_last_recommended_n == 12);
        assert(warm_start.profit_active_episode_n == 12);
        assert(warm_start.profit_request_requires_fresh_switch_sample);
        assert(warm_start.profit_depth[15].samples == 1);
    }

    assert(server_adaptive_dm_uses_profit_controller(COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT));
    assert(!server_adaptive_dm_uses_profit_controller(COMMON_SPECULATIVE_DM_CONTROLLER_OFF));

    state.adaptive_n_max = 4;
    state.apply_profit_recommendation(2);
    assert(state.adaptive_n_max == 2);
    assert(state.profit_last_recommended_n == 2);

    // reset_profit_state zeros everything including profit data (config change)
    state.reset_profit_state();
    assert(state.profit_pos_accept_ewma[2] == 0.0f);
    assert(state.profit_pos_samples[2] == 0);
    assert(state.profit_depth[4].samples == 0);
    assert(state.profit_baseline.samples == 0);
    assert(!state.profit_has_key);

    state.dm_profit_min_samples = 1;
    state.dm_off_dwell = 1;
    state.adaptive_n_max = 2;
    observe_profit_cycle(state, 2, 2, 0, 80.0f);
    {
        const int recommended = state.decide_profit_n_max(8);
        assert(recommended == 0);
        state.apply_profit_recommendation(recommended);
    }
    state.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
    // With a baseline available the cold controller bootstraps at the maximum
    // draft depth rather than walking up from a shallow probe.
    assert(state.decide_profit_n_max(8) == 8);

    // test cold start: fresh profit controllers seed the no-spec baseline before
    // any positive-depth DFlash cycle, then bootstrap production at the maximum
    // draft depth (the lower spread is characterized via transient excursions).
    state.reset_profit_state();
    state.dm_profit_min_samples = 2;
    state.dm_off_dwell = 1;
    assert(state.decide_profit_n_max(8) == 0);
    assert(state.profit_expects_baseline_sample());
    state.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
    assert(state.decide_profit_n_max(8) == 0);
    state.observe_profit_timing(0, 0, 0, 0.0f, 32.0f, 0.0f, 32.0f);
    assert(state.profit_baseline_ready());
    assert(state.decide_profit_n_max(8) == 8);

    // Baseline scoring uses the same current EWMA policy as positive depths;
    // stale best no-spec spikes must not make baseline unbeatable.
    state.reset_profit_state();
    state.dm_profit_min_samples = 2;
    state.observe_profit_timing(0, 0, 0, 0.0f, 100.0f, 0.0f, 100.0f);
    state.observe_profit_timing(0, 0, 0, 0.0f, 25.0f, 0.0f, 25.0f);
    assert(state.profit_baseline_ready());
    assert(state.profit_baseline.best_score >= 39.9f);
    assert_close(state.profit_score_for_depth(0), 16.0f);

    // test explicit warmup requires extra measured samples per spread depth before
    // the initial probe set is treated as characterized; production holds at the
    // maximum draft depth throughout, and only argmax (post-characterization) moves.
    state.reset_profit_state();
    state.dm_profit_min_samples = 1;
    state.dm_profit_warmup = 2;
    state.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
    assert(state.profit_baseline_ready());
    assert(state.decide_profit_n_max(8) == 8);
    // one sample per spread depth is not enough under warmup=2
    observe_profit_cycle(state, 2, 2, 1, 60.0f);
    observe_profit_cycle(state, 4, 4, 1, 60.0f);
    observe_profit_cycle(state, 8, 8, 1, 60.0f);
    assert(!state.profit_initial_probe_set_ready(8));
    assert(state.decide_profit_n_max(8) == 8);
    // a second sample per spread depth completes characterization
    observe_profit_cycle(state, 2, 2, 1, 60.0f);
    observe_profit_cycle(state, 4, 4, 1, 60.0f);
    observe_profit_cycle(state, 8, 8, 1, 60.0f);
    assert(state.profit_initial_probe_set_ready(8));
    state.dm_profit_warmup = 0;

    // End-to-end convergence: a cold profit controller bootstraps at the maximum
    // draft depth and must then settle on the genuinely fastest depth, regardless
    // of whether the optimum is high, mid, or low. This is the property the earlier
    // "start high, walk down, rest at the walk terminal" regression violated: it
    // collapsed to the floor and could not climb back. Throughput here is modeled
    // unimodally (peak at `optimum`): score(d) = (1 + min(d, optimum)) / (20 + 2d),
    // with a no-spec baseline every positive depth beats.
    auto converged_depth = [](int base_n_max, int optimum) -> int {
        server_adaptive_dm_state s;
        s.dm_profit_min_samples = 2;
        s.dm_profit_baseline_interval = 0; // disable reprobe noise for a deterministic check
        auto feed = [&](int d) {
            if (d <= 0) {
                s.observe_profit_timing(0, 0, 0, 0.0f, 35.0f, 0.0f, 35.0f);
                return;
            }
            const int acc = d < optimum ? d : optimum;
            const float ms = 20.0f + 2.0f * (float) d;
            s.observe_profit_acceptance(d, acc);
            s.observe_profit_timing(d, d, acc, 0.0f, ms, 0.0f, ms);
        };
        feed(0);
        feed(0);
        assert(s.profit_baseline_ready());
        int rec = s.decide_profit_n_max(base_n_max);
        assert(rec == base_n_max); // cold start bootstraps at max, not a low probe
        s.apply_profit_recommendation(rec);
        // Each iteration samples production plus the full depth range, standing in
        // for transient characterization/exploration excursions accumulating over
        // time. The controller must converge to (and hold) the throughput optimum.
        for (int iter = 0; iter < 200; ++iter) {
            feed(s.adaptive_n_max > 0 ? s.adaptive_n_max : base_n_max);
            for (int d = 1; d <= base_n_max; ++d) {
                feed(d);
            }
            feed(0);
            rec = s.decide_profit_n_max(base_n_max);
            s.apply_profit_recommendation(rec);
        }
        return s.adaptive_n_max;
    };
    {
        const int high = converged_depth(15, 15);
        assert(high >= 12); // high optimum: stay high (the regressed case wanted this)
        const int mid8 = converged_depth(15, 8);
        assert(mid8 >= 6 && mid8 <= 10); // mid optimum identified, not stuck high or low
        const int mid12 = converged_depth(15, 12);
        assert(mid12 >= 10 && mid12 <= 14); // mid-high optimum identified
        const int low = converged_depth(15, 4);
        assert(low >= 2 && low <= 6); // low optimum identified, not stuck high
    }

    // test reset_request_state preserves learned profit data while resetting
    // request-local counters
    state.reset_profit_state();
    state.dm_profit_min_samples = 1;
    state.dm_off_dwell = 1;
    state.adaptive_n_max = 4;
    observe_profit_cycle(state, 4, 4, 2, 73.0f);
    state.profit_pending = true;
    state.profit_consecutive_below_profit = 3;

    state.reset_request_state();
    // request counters reset
    assert(state.adaptive_n_max == -1);
    assert(state.profit_pending == false);
    assert(state.profit_consecutive_below_profit == 0);
    assert(state.profit_last_recommended_n == -1);
    assert(state.profit_off_probe_failures == 0);
    // learned profit data preserved
    assert(state.profit_pos_samples[0] == 1);
    assert(state.profit_depth[4].samples == 1);
    assert(state.profit_pos_accept_ewma[0] > 0.0f);

    state.dm_profit_min_samples = 1;
    common_params_speculative empty_spec;
    state.reset_profit_if_config_changed(empty_spec, 1, 0);
    // actually need a proper spec struct; let's use the simpler check
    common_params_speculative spec = make_dflash_spec(8);
    common_params_sampling sampling;
    sampling.temp = 0.6f;
    sampling.top_k = 20;
    sampling.top_p = 1.0f;
    sampling.min_p = 0.0f;
    state.observe_profit_timing(2, 2, 1, 10.0f, 20.0f, 5.0f, 35.0f);
    state.reset_profit_if_config_changed(spec, 8, 0, &sampling);
    assert(state.profit_has_key);
    assert(state.profit_depth[2].samples == 0);
    assert(state.adaptive_n_max == -1);
    state.observe_profit_timing(2, 2, 1, 10.0f, 20.0f, 5.0f, 35.0f);
    state.reset_profit_if_config_changed(spec, 8, 0, &sampling);
    assert(state.profit_depth[2].samples == 1);
    sampling.temp = 0.2f;
    state.reset_profit_if_config_changed(spec, 8, 0, &sampling);
    assert(state.profit_depth[2].samples == 0);
    state.observe_profit_timing(2, 2, 1, 10.0f, 20.0f, 5.0f, 35.0f);
    state.adaptive_n_max = 8;
    spec.draft.p_split = 0.2f;
    state.reset_profit_if_config_changed(spec, 8, 0, &sampling);
    assert(state.profit_depth[2].samples == 0);
    assert(state.adaptive_n_max == -1);

    // Context position alone is not a configuration boundary. The profit
    // controller must not use fixed context buckets that force relearning at
    // arbitrary positions in a long-context model.
    {
        server_adaptive_dm_state position;
        common_params_speculative position_spec = make_dflash_spec(16);

        position.reset_profit_if_config_changed(position_spec, 16, 900);
        position.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
        position.observe_profit_timing(0, 0, 0, 0.0f, 31.0f, 0.0f, 31.0f);
        position.observe_profit_timing(0, 0, 0, 0.0f, 32.0f, 0.0f, 32.0f);
        position.apply_profit_recommendation(14);
        observe_profit_cycle(position, 14, 14, 10, 72.0f);

        position.reset_profit_if_config_changed(position_spec, 16, 1200);
        assert(position.adaptive_n_max == 14);
        assert(position.profit_baseline_ready());
        assert(position.profit_depth[14].samples == 1);
        assert(!position.profit_baseline_probe_pending);
        assert(!position.profit_expects_baseline_sample());

        position.reset_profit_if_config_changed(position_spec, 16, 64000);
        assert(position.adaptive_n_max == 14);
        assert(position.profit_baseline_ready());
        assert(position.profit_depth[14].samples == 1);
        assert(!position.profit_baseline_probe_pending);
        assert(!position.profit_expects_baseline_sample());
    }

    // test cross-depth estimation
    {
        server_adaptive_dm_state est;
        est.dm_profit_min_samples = 3;
        est.dm_profit_raise_margin = 0.01f;
        est.dm_profit_lower_margin = 0.05f;
        est.adaptive_n_max = 8;

        // simulate 3 cycles at depth 8 with high acceptance
        for (int i = 0; i < 3; i++) {
            observe_profit_cycle(est, 8, 8, 6, 67.0f);
        }
        // baseline
        for (int i = 0; i < 3; i++) {
            est.observe_profit_timing(0, 0, 0, 0.0f, 22.0f, 0.0f, 22.0f);
        }

        // depth 8 should be ready, depth 4 should be estimated
        const int recommended = est.decide_profit_n_max(8);
        assert(recommended > 0);
    }

    // Direct measured lower-depth candidates can demote an active full-depth
    // DFlash horizon, but the change must still pass through hysteresis so a
    // single request cannot collapse from full depth to shallow depth.
    {
        server_adaptive_dm_state est;
        est.dm_profit_min_samples = 3;
        est.dm_profit_raise_margin = 0.01f;
        est.dm_profit_lower_margin = 0.01f;
        est.adaptive_n_max = 15;
        est.profit_last_recommended_n = 15;

        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(est, 15, 15, 1, 90.0f);
            observe_profit_cycle(est, 4, 4, 2, 45.0f);
            observe_profit_cycle(est, 8, 8, 1, 80.0f);
            observe_profit_cycle(est, 10, 10, 1, 85.0f);
            observe_profit_cycle(est, 12, 12, 1, 88.0f);
        }
        for (int i = 0; i < 3; ++i) {
            est.observe_profit_timing(0, 0, 0, 0.0f, 35.0f, 0.0f, 35.0f);
        }

        assert(est.decide_profit_n_max(15) == 13);
    }

    // Sustained low acceptance without a baseline must collect baseline data,
    // not keep serving an unmeasured speculative depth.
    {
        server_adaptive_dm_state weak;
        weak.dm_profit_min_samples = 3;
        weak.adaptive_n_max = 15;

        for (int i = 0; i < 12; ++i) {
            observe_profit_cycle(weak, 15, 15, 0, 135.0f);
        }

        const int rec = weak.decide_profit_n_max(15);
        assert(rec == 0);

        weak.reset_request_state();
        assert(weak.profit_last_recommended_n == -1);
    }

    // test baseline-best still observes dwell after the initial probe set is measured
    {
        server_adaptive_dm_state weak;
        weak.dm_profit_min_samples = 1;
        weak.dm_off_dwell = 2;
        weak.adaptive_n_max = 8;
        weak.profit_last_recommended_n = 8;
        weak.observe_profit_timing(0, 0, 0, 0.0f, 25.0f, 0.0f, 25.0f);
        observe_profit_cycle(weak, 2, 2, 0, 80.0f);
        observe_profit_cycle(weak, 4, 4, 0, 90.0f);
        observe_profit_cycle(weak, 8, 8, 0, 105.0f);
        assert(weak.decide_profit_n_max(8) == 8);
        observe_profit_cycle(weak, 8, 8, 0, 105.0f);
        assert(weak.decide_profit_n_max(8) == 8);
        observe_profit_cycle(weak, 8, 8, 0, 105.0f);
        assert(weak.decide_profit_n_max(8) == 8);
        observe_profit_cycle(weak, 8, 8, 0, 105.0f);
        assert(weak.decide_profit_n_max(8) == 0);
    }

    // test active profit cycles trigger periodic no-spec baseline reprobes
    {
        server_adaptive_dm_state reprobe;
        reprobe.dm_profit_min_samples = 1;
        reprobe.dm_profit_baseline_interval = 3;
        common_params_speculative reprobe_spec = make_dflash_spec(8);
        reprobe.reset_profit_if_config_changed(reprobe_spec, 8, 40000);
        reprobe.adaptive_n_max = 8;
        reprobe.observe_profit_timing(0, 0, 0, 0.0f, 40.0f, 0.0f, 40.0f);
        observe_profit_cycle(reprobe, 8, 8, 7, 40.0f);
        assert(!reprobe.profit_should_probe_baseline());
        observe_profit_cycle(reprobe, 8, 8, 7, 40.0f);
        observe_profit_cycle(reprobe, 8, 8, 7, 40.0f);
        assert(reprobe.profit_should_probe_baseline());
        reprobe.profit_mark_baseline_probe();
        assert(reprobe.adaptive_n_max == 0);
        assert(reprobe.profit_baseline_probe_pending);
        reprobe.observe_profit_timing(0, 0, 0, 0.0f, 42.0f, 0.0f, 42.0f);
        assert(!reprobe.profit_should_probe_baseline());
        assert(reprobe.decide_profit_n_max(8) == 8);
    }

    // test periodic baseline reprobes are interval-based, not context-bucket gated
    {
        server_adaptive_dm_state early;
        early.dm_profit_min_samples = 1;
        early.dm_profit_baseline_interval = 1;
        common_params_speculative early_spec = make_dflash_spec(8);
        early.reset_profit_if_config_changed(early_spec, 8, 4096);
        early.adaptive_n_max = 8;
        early.observe_profit_timing(0, 0, 0, 0.0f, 40.0f, 0.0f, 40.0f);
        observe_profit_cycle(early, 8, 8, 7, 40.0f);
        assert(early.profit_should_probe_baseline());
    }

    // A positive-depth demotion while running on an old baseline should first
    // remeasure no-spec. If a directly measured DFlash depth still beats the
    // fresh baseline, stay speculative and demote gradually.
    {
        server_adaptive_dm_state stale;
        stale.dm_profit_min_samples = 1;
        stale.dm_off_dwell = 4;
        stale.dm_profit_baseline_interval = 4;
        stale.dm_profit_raise_margin = 0.0f;
        stale.dm_profit_lower_margin = 0.0f;
        stale.dm_profit_ewma_alpha = 1.0f;
        stale.adaptive_n_max = 15;
        stale.profit_last_recommended_n = 15;

        stale.observe_profit_timing(0, 0, 0, 0.0f, 50.0f, 0.0f, 50.0f);
        observe_profit_cycle(stale, 4, 4, 0, 80.0f);
        observe_profit_cycle(stale, 8, 8, 0, 100.0f);
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(stale, 10, 10, 2, 75.0f);
            observe_profit_cycle(stale, 12, 12, 2, 70.0f);
            observe_profit_cycle(stale, 14, 14, 2, 85.0f);
        }
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(stale, 15, 15, 2, 90.0f);
        }

        const int reprobe = stale.decide_profit_n_max(15);
        assert(reprobe == 0);
        assert(stale.profit_baseline_probe_pending);
        assert(stale.profit_baseline_probe_resume_n == 15);

        stale.observe_profit_timing(0, 0, 0, 0.0f, 24.0f, 0.0f, 24.0f);
        const int after = stale.decide_profit_n_max(15);
        assert(after == 13);
    }

    // Positive-depth changes also need a residence window. A directly measured
    // lower depth can win after the active episode is measured, but not after
    // one noisy cycle at the current production depth.
    {
        server_adaptive_dm_state positive_residence;
        positive_residence.dm_profit_min_samples = 1;
        positive_residence.dm_profit_raise_margin = 0.0f;
        positive_residence.dm_profit_lower_margin = 0.0f;
        positive_residence.adaptive_n_max = 12;
        positive_residence.profit_last_recommended_n = 12;

        positive_residence.observe_profit_timing(0, 0, 0, 0.0f, 40.0f, 0.0f, 40.0f);
        observe_profit_cycle(positive_residence, 4, 4, 0, 80.0f);
        observe_profit_cycle(positive_residence, 8, 8, 0, 80.0f);
        observe_profit_cycle(positive_residence, 14, 14, 0, 100.0f);
        observe_profit_cycle(positive_residence, 15, 15, 0, 100.0f);
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(positive_residence, 10, 10, 3, 70.0f);
        }
        observe_profit_cycle(positive_residence, 12, 12, 1, 90.0f);
        assert(positive_residence.decide_profit_n_max(15) == 12);

        for (int i = 0; i < 5; ++i) {
            observe_profit_cycle(positive_residence, 12, 12, 1, 90.0f);
        }
        assert(positive_residence.decide_profit_n_max(15) == 10);
    }

    // Directly measured raises must still clear the raise margin. Without
    // this, a noisy 14/15 probe can pull a stable 12-depth prose run upward
    // even when the throughput gain is inside normal cycle noise.
    {
        server_adaptive_dm_state measured_raise_margin;
        measured_raise_margin.dm_profit_min_samples = 1;
        measured_raise_margin.dm_profit_raise_margin = 0.05f;
        measured_raise_margin.dm_profit_lower_margin = 0.0f;
        measured_raise_margin.adaptive_n_max = 12;
        measured_raise_margin.profit_last_recommended_n = 12;

        measured_raise_margin.observe_profit_timing(0, 0, 0, 0.0f, 40.0f, 0.0f, 40.0f);
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(measured_raise_margin, 4, 4, 0, 90.0f);
            observe_profit_cycle(measured_raise_margin, 8, 8, 0, 90.0f);
            observe_profit_cycle(measured_raise_margin, 10, 10, 0, 90.0f);
            observe_profit_cycle(measured_raise_margin, 12, 12, 2, 75.0f); // 40.0 t/s
            observe_profit_cycle(measured_raise_margin, 14, 14, 2, 73.0f); // 41.1 t/s, below 5% raise margin
            observe_profit_cycle(measured_raise_margin, 15, 15, 1, 95.0f);
        }

        assert(measured_raise_margin.decide_profit_n_max(15) == 12);
    }

    // Positive demotion should not force a no-spec baseline probe on the short
    // off-dwell counter; normal baseline reprobe cadence handles stale baseline
    // data. Reprobing on every demotion causes 12/14/15 -> 0 churn.
    {
        server_adaptive_dm_state demote_without_reprobe;
        demote_without_reprobe.dm_profit_min_samples = 1;
        demote_without_reprobe.dm_off_dwell = 1;
        demote_without_reprobe.dm_profit_baseline_interval = 1024;
        demote_without_reprobe.dm_profit_raise_margin = 0.0f;
        demote_without_reprobe.dm_profit_lower_margin = 0.0f;
        demote_without_reprobe.adaptive_n_max = 15;
        demote_without_reprobe.profit_last_recommended_n = 15;

        demote_without_reprobe.observe_profit_timing(0, 0, 0, 0.0f, 50.0f, 0.0f, 50.0f);
        observe_profit_cycle(demote_without_reprobe, 4, 4, 0, 80.0f);
        observe_profit_cycle(demote_without_reprobe, 8, 8, 0, 80.0f);
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(demote_without_reprobe, 10, 10, 3, 70.0f);
            observe_profit_cycle(demote_without_reprobe, 12, 12, 1, 90.0f);
            observe_profit_cycle(demote_without_reprobe, 14, 14, 1, 95.0f);
        }
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(demote_without_reprobe, 15, 15, 1, 95.0f);
        }

        assert(demote_without_reprobe.decide_profit_n_max(15) == 13);
        assert(!demote_without_reprobe.profit_baseline_probe_pending);
    }

    // Preserving an active depth across a similar prompt should not let stale
    // challenger stats immediately demote the run. A challenger can replace the
    // preserved production depth after it has fresh samples in the new request.
    {
        server_adaptive_dm_state preserved_challenger;
        preserved_challenger.dm_profit_min_samples = 1;
        preserved_challenger.dm_profit_raise_margin = 0.0f;
        preserved_challenger.dm_profit_lower_margin = 0.0f;
        preserved_challenger.dm_profit_baseline_interval = 1024;
        preserved_challenger.adaptive_n_max = 15;
        preserved_challenger.profit_last_recommended_n = 15;
        preserved_challenger.profit_has_key = true;
        preserved_challenger.profit_key = {};
        preserved_challenger.profit_key.base_n_max = 15;

        preserved_challenger.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(preserved_challenger, 12, 12, 10, 70.0f);
            observe_profit_cycle(preserved_challenger, 15, 15, 10, 90.0f);
        }

        preserved_challenger.reset_request_state(true);
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(preserved_challenger, 15, 15, 10, 90.0f);
        }

        assert(preserved_challenger.decide_profit_n_max(15) == 15);
        observe_profit_cycle(preserved_challenger, 12, 12, 10, 70.0f);
        assert(preserved_challenger.decide_profit_n_max(15) == 15);
        for (int i = 0; i < 5; ++i) {
            observe_profit_cycle(preserved_challenger, 12, 12, 10, 70.0f);
        }
        assert(preserved_challenger.decide_profit_n_max(15) == 13);
    }

    // A preserved max-depth run must actively collect fresh local challenger
    // evidence. Stale global stats can rank a lower challenger, but they cannot
    // replace a same-request probe, and active mode should not sweep the low end.
    {
        server_adaptive_dm_state preserved_probe;
        preserved_probe.dm_profit_min_samples = 1;
        preserved_probe.adaptive_n_max = 15;
        preserved_probe.profit_last_recommended_n = 15;
        preserved_probe.profit_has_key = true;
        preserved_probe.profit_key = {};
        preserved_probe.profit_key.base_n_max = 15;

        preserved_probe.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
        observe_profit_cycle(preserved_probe, 12, 12, 10, 70.0f);
        observe_profit_cycle(preserved_probe, 15, 15, 10, 90.0f);

        preserved_probe.reset_request_state(true);
        assert(preserved_probe.profit_next_unready_explore_depth(15, 15, 1) == 14);
        observe_profit_cycle(preserved_probe, 14, 14, 10, 70.0f);
        assert(preserved_probe.profit_next_unready_explore_depth(15, 15, 1) == 14);
        for (int i = 0; i < 5; ++i) {
            observe_profit_cycle(preserved_probe, 14, 14, 10, 70.0f);
        }
        assert(preserved_probe.profit_next_unready_explore_depth(15, 15, 1) == 13);
    }

    // test lower acceptance can still be faster; controller must optimize TPS,
    // not raw acceptance rate.
    {
        server_adaptive_dm_state tps;
        tps.dm_profit_min_samples = 2;
        tps.dm_profit_raise_margin = 0.0f;
        tps.dm_profit_lower_margin = 0.0f;
        tps.adaptive_n_max = 2;
        tps.profit_last_recommended_n = 2;
        for (int i = 0; i < 3; ++i) {
            tps.observe_profit_timing(0, 0, 0, 0.0f, 35.0f, 0.0f, 35.0f);
            observe_profit_cycle(tps, 2, 2, 2, 120.0f);
            observe_profit_cycle(tps, 4, 4, 2, 100.0f);
            observe_profit_cycle(tps, 8, 8, 1, 45.0f);
        }
        assert(tps.decide_profit_n_max(8) == 4);
    }

    // Off state stays off on baseline-only samples even when stale positive
    // history looks profitable; get_n_draft_max must schedule an explicit
    // probe before speculation can wake back up.
    {
        server_adaptive_dm_state off;
        off.dm_profit_min_samples = 1;
        off.dm_profit_raise_margin = 0.0f;
        off.dm_profit_lower_margin = 0.0f;
        off.adaptive_n_max = 0;
        off.observe_profit_timing(0, 0, 0, 0.0f, 35.0f, 0.0f, 35.0f);
        observe_profit_cycle(off, 2, 2, 2, 40.0f);
        observe_profit_cycle(off, 4, 4, 1, 70.0f);
        observe_profit_cycle(off, 8, 8, 5, 70.0f);
        observe_profit_cycle(off, 8, 8, 5, 70.0f);
        assert(off.decide_profit_n_max(8) == 0);
        off.adaptive_n_max = 2;
        observe_profit_cycle(off, 2, 2, 2, 40.0f);
        int wake = off.decide_profit_n_max(8);
        assert(wake == 2);
        off.apply_profit_recommendation(wake);
        assert(off.profit_off_probe_failures == 0);
        assert(off.decide_profit_n_max(8) == 2);
    }

    // A directly measured off-state probe must still honor dm_profit_min.
    // Users can set the margin to zero when any measured win should wake.
    {
        server_adaptive_dm_state small_win;
        small_win.dm_profit_min_samples = 1;
        small_win.dm_profit_min = 0.05f;
        small_win.adaptive_n_max = 4;
        small_win.profit_last_recommended_n = 0;
        small_win.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
        observe_profit_cycle(small_win, 2, 2, 0, 80.0f);
        observe_profit_cycle(small_win, 4, 4, 1, 58.0f);
        observe_profit_cycle(small_win, 8, 8, 0, 100.0f);
        assert(small_win.decide_profit_n_max(8) == 0);

        server_adaptive_dm_state zero_margin = small_win;
        zero_margin.dm_profit_min = 0.0f;
        assert(zero_margin.decide_profit_n_max(8) == 4);
    }

    // A one-off fast baseline sample must not make no-spec unbeatable when
    // the current baseline EWMA is slower than a directly measured depth.
    {
        server_adaptive_dm_state spike;
        spike.dm_profit_min_samples = 1;
        spike.dm_off_dwell = 1;
        spike.adaptive_n_max = 8;
        spike.profit_last_recommended_n = 8;
        spike.profit_baseline.samples = 2;
        spike.profit_baseline.cycle_ms = 30.0f; // current baseline score: 33.3 t/s
        spike.profit_baseline.output_tokens = 1.0f;
        spike.profit_baseline.best_score = 100.0f; // stale spike must not dominate
        spike.profit_baseline.total_output_tokens = 2.0;
        spike.profit_baseline.total_cycle_ms = 60.0;
        observe_profit_cycle(spike, 2, 2, 0, 90.0f);
        observe_profit_cycle(spike, 4, 4, 0, 90.0f);
        observe_profit_cycle(spike, 8, 8, 1, 50.0f); // 40.0 t/s
        assert(spike.decide_profit_n_max(8) == 8);
    }

    // Active depths need a residence window before a single weak cycle can
    // disable speculation. Fixed-depth runs show prose throughput stabilizes
    // over longer windows; per-cycle shutdown causes churn.
    {
        server_adaptive_dm_state residence;
        residence.dm_profit_min_samples = 1;
        residence.dm_off_dwell = 1;
        residence.adaptive_n_max = 8;
        residence.profit_last_recommended_n = 8;
        residence.observe_profit_timing(0, 0, 0, 0.0f, 25.0f, 0.0f, 25.0f);
        observe_profit_cycle(residence, 2, 2, 0, 90.0f);
        observe_profit_cycle(residence, 4, 4, 0, 90.0f);
        observe_profit_cycle(residence, 8, 8, 0, 90.0f);
        assert(residence.decide_profit_n_max(8) == 8);
    }

    // Once the active depth has had a residence window, an unmeasured deeper
    // ladder rung must not keep a clearly bad depth alive. The off-state sweep
    // is responsible for rediscovery after demotion.
    {
        server_adaptive_dm_state bad_current;
        bad_current.dm_profit_min_samples = 1;
        bad_current.dm_off_dwell = 1;
        bad_current.adaptive_n_max = 12;
        bad_current.profit_last_recommended_n = 12;
        bad_current.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
        observe_profit_cycle(bad_current, 4, 4, 0, 90.0f);
        observe_profit_cycle(bad_current, 8, 8, 0, 90.0f);
        observe_profit_cycle(bad_current, 10, 10, 0, 90.0f);
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(bad_current, 12, 12, 0, 90.0f);
        }
        assert(bad_current.decide_profit_n_max(15) == 0);
    }

    // Depth scores must represent the measured episode, not just the latest
    // noisy cycle. One bad cycle after a profitable run should not immediately
    // make baseline win.
    {
        server_adaptive_dm_state episode;
        episode.dm_profit_min_samples = 1;
        episode.dm_off_dwell = 1;
        episode.dm_profit_min = 0.0f;
        episode.dm_profit_ewma_alpha = 1.0f;
        episode.adaptive_n_max = 8;
        episode.profit_last_recommended_n = 8;
        episode.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
        observe_profit_cycle(episode, 2, 2, 0, 80.0f);
        observe_profit_cycle(episode, 4, 4, 0, 90.0f);
        for (int i = 0; i < 5; ++i) {
            observe_profit_cycle(episode, 8, 8, 2, 70.0f);
        }
        observe_profit_cycle(episode, 8, 8, 0, 120.0f);
        assert_close(episode.profit_score_for_depth(0), 33.333332f);
        assert_close(episode.profit_score_for_depth(8), 34.042553f);
        assert(episode.decide_profit_n_max(8) == 8);
    }

    // Exploration winners need confirmation before replacing the active
    // production depth, otherwise one fast sample can cause ping-pong.
    {
        server_adaptive_dm_state confirm;
        confirm.dm_profit_min_samples = 1;
        confirm.dm_profit_raise_margin = 0.0f;
        confirm.dm_profit_lower_margin = 0.0f;
        confirm.adaptive_n_max = 8;
        confirm.profit_last_recommended_n = 8;
        confirm.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
        observe_profit_cycle(confirm, 4, 4, 0, 90.0f);
        observe_profit_cycle(confirm, 10, 10, 0, 90.0f);
        observe_profit_cycle(confirm, 8, 8, 2, 75.0f);
        observe_profit_cycle(confirm, 8, 8, 2, 75.0f);
        observe_profit_cycle(confirm, 8, 8, 2, 75.0f);
        observe_profit_cycle(confirm, 8, 8, 2, 75.0f);
        observe_profit_cycle(confirm, 8, 8, 2, 75.0f);
        observe_profit_cycle(confirm, 8, 8, 2, 75.0f);
        observe_profit_cycle(confirm, 12, 12, 3, 85.0f);
        assert(confirm.decide_profit_n_max(12) == 8);
        observe_profit_cycle(confirm, 12, 12, 3, 85.0f);
        observe_profit_cycle(confirm, 12, 12, 3, 85.0f);
        observe_profit_cycle(confirm, 12, 12, 3, 85.0f);
        observe_profit_cycle(confirm, 12, 12, 3, 85.0f);
        observe_profit_cycle(confirm, 12, 12, 3, 85.0f);
        assert(confirm.decide_profit_n_max(12) == 12);
    }

    // Cold-start measurement should ramp quickly to the configured base depth.
    // If that active depth is clearly bad after its residence window, the
    // controller should try a lower rescue depth before disabling speculation.
    {
        server_adaptive_dm_state cold_ladder;
        cold_ladder.dm_profit_min_samples = 1;
        cold_ladder.dm_off_dwell = 1;
        cold_ladder.adaptive_n_max = 15;
        cold_ladder.profit_last_recommended_n = 15;

        cold_ladder.observe_profit_timing(0, 0, 0, 0.0f, 25.0f, 0.0f, 25.0f);
        observe_profit_cycle(cold_ladder, 4, 4, 0, 80.0f);
        observe_profit_cycle(cold_ladder, 4, 4, 0, 80.0f);
        observe_profit_cycle(cold_ladder, 8, 8, 0, 90.0f);
        observe_profit_cycle(cold_ladder, 8, 8, 0, 90.0f);
        cold_ladder.profit_begin_active_episode(15);
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(cold_ladder, 15, 15, 0, 110.0f);
        }

        assert(cold_ladder.profit_next_unready_probe_depth(15) < 0);
        assert(cold_ladder.decide_profit_n_max(15) == 12);
    }

    // A shorter horizon can become best when recent per-position acceptance is
    // low, even if its own old direct acceptance samples were weak. Candidate
    // scoring should combine measured timing cost with current survival data.
    {
        server_adaptive_dm_state recent_acceptance;
        recent_acceptance.dm_profit_min_samples = 2;
        recent_acceptance.dm_profit_raise_margin = 0.0f;
        recent_acceptance.dm_profit_lower_margin = 0.0f;
        recent_acceptance.dm_off_dwell = 99;
        recent_acceptance.dm_profit_ewma_alpha = 1.0f;
        recent_acceptance.adaptive_n_max = 12;
        recent_acceptance.profit_last_recommended_n = 12;

        recent_acceptance.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
        recent_acceptance.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);
        observe_profit_cycle(recent_acceptance, 4, 4, 0, 80.0f);
        observe_profit_cycle(recent_acceptance, 4, 4, 0, 80.0f);
        observe_profit_cycle(recent_acceptance, 8, 8, 0, 70.0f);
        observe_profit_cycle(recent_acceptance, 8, 8, 0, 70.0f);
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(recent_acceptance, 10, 10, 0, 50.0f);
        }
        observe_profit_cycle(recent_acceptance, 14, 14, 0, 90.0f);
        observe_profit_cycle(recent_acceptance, 14, 14, 0, 90.0f);
        observe_profit_cycle(recent_acceptance, 15, 15, 0, 95.0f);
        observe_profit_cycle(recent_acceptance, 15, 15, 0, 95.0f);
        for (int i = 0; i < 6; ++i) {
            observe_profit_cycle(recent_acceptance, 12, 12, 2, 80.0f);
        }

        assert(recent_acceptance.decide_profit_n_max(15) == 10);
    }

    // Off-state probes should try useful anchors first, then fill nearby low
    // depths. Dense candidate scoring is separate; recovery probes should not
    // spend an entire sweep walking 1,2,3,... before testing the known productive
    // mid/high range.
    {
        server_adaptive_dm_state sweep;
        sweep.dm_profit_min_samples = 1;
        sweep.dm_probe_interval = 4;
        sweep.adaptive_n_max = 0;
        sweep.observe_profit_timing(0, 0, 0, 0.0f, 30.0f, 0.0f, 30.0f);

        const int expected_order[] = {4, 8, 12, 15, 2, 3, 1, 5, 7, 9, 11, 13, 14, 6, 10};
        for (int i = 0; i < 14; ++i) {
            const int depth = expected_order[i];
            assert(sweep.profit_next_off_probe_depth(15, 3) == depth);
            observe_profit_cycle(sweep, depth, depth, 0, 60.0f + (float) depth);
            sweep.profit_note_off_probe_result(depth, 15, false);
            assert(sweep.profit_next_off_probe_depth(15, 3) == expected_order[i + 1]);
            assert(sweep.profit_off_probe_failures == 0);
            assert(sweep.profit_off_probe_interval() == 4);
        }

        observe_profit_cycle(sweep, expected_order[14], expected_order[14], 0, 80.0f);
        sweep.profit_note_off_probe_result(expected_order[14], 15, false);
        assert(sweep.profit_next_off_probe_depth(15, 3) == expected_order[0]);
        assert(sweep.profit_off_probe_failures == 1);
        assert(sweep.profit_off_probe_interval() == 8);

        sweep.profit_note_off_probe_result(expected_order[0], 15, true);
        assert(sweep.profit_off_probe_failures == 0);
        assert(sweep.profit_next_off_probe_depth(15, 3) == expected_order[0]);
    }

    // Multi-slot flat DFlash verification needs equal row counts. If any
    // eligible slot already has a positive adaptive depth, cold/off slots join a
    // shared cohort depth instead of forcing a mixed speculative/baseline batch.
    {
        const int one_hot_normal[] = {15, 0};
        const int one_hot_cap[] = {15, 15};
        assert(server_adaptive_dm_resolve_cohort_n_max(one_hot_normal, one_hot_cap, 2) == 15);

        const int mixed_positive_normal[] = {15, 8};
        const int mixed_positive_cap[] = {15, 8};
        assert(server_adaptive_dm_resolve_cohort_n_max(mixed_positive_normal, mixed_positive_cap, 2) == 8);

        const int all_cold_normal[] = {0, 0};
        const int all_cold_cap[] = {15, 15};
        assert(server_adaptive_dm_resolve_cohort_n_max(all_cold_normal, all_cold_cap, 2) == 0);

        const int blocked_normal[] = {15, 0};
        const int blocked_cap[] = {15, 0};
        assert(server_adaptive_dm_resolve_cohort_n_max(blocked_normal, blocked_cap, 2) == 0);
    }

    // Failed wake probes advance the off-state sweep instead of retrying the
    // same shallow depth forever.
    {
        server_adaptive_dm_state failed;
        failed.dm_profit_min_samples = 1;
        failed.dm_probe_interval = 4;
        failed.adaptive_n_max = 4;
        failed.profit_last_recommended_n = 0;
        failed.observe_profit_timing(0, 0, 0, 0.0f, 20.0f, 0.0f, 20.0f);
        observe_profit_cycle(failed, 2, 2, 0, 50.0f);
        observe_profit_cycle(failed, 4, 4, 0, 60.0f);
        observe_profit_cycle(failed, 4, 4, 0, 70.0f);
        observe_profit_cycle(failed, 8, 8, 0, 80.0f);
        assert(failed.profit_off_probe_interval() == 4);
        assert(failed.decide_profit_n_max(8) == 0);
        assert(failed.profit_off_probe_failures == 0);
        assert(failed.profit_next_off_probe_depth(8, 2) == 8);
        assert(failed.profit_off_probe_interval() == 4);
    }

    return 0;
}
