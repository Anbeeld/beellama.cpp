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

int main() {
    const float min = 0.30f;
    const float max = 0.65f;

    assert_close(server_adaptive_dm_required_fringe_for_n_max(2, 8, min, max), 0.30f);
    assert_close(server_adaptive_dm_required_fringe_for_n_max(3, 8, min, max), 0.35833335f);
    assert_close(server_adaptive_dm_required_fringe_for_n_max(4, 8, min, max), 0.41666669f);
    assert_close(server_adaptive_dm_required_fringe_for_n_max(6, 8, min, max), 0.53333336f);
    assert_close(server_adaptive_dm_required_fringe_for_n_max(8, 8, min, max), 0.65f);

    assert_close(server_adaptive_dm_required_fringe_for_n_max(99, 8, min, max), 0.65f);
    assert_close(server_adaptive_dm_required_fringe_for_n_max(4, 1, min, max), 0.30f);

    assert(server_adaptive_dm_probe_n_max(8, 0.25f) == 2);
    assert(server_adaptive_dm_probe_n_max(16, 0.25f) == 4);
    assert(server_adaptive_dm_probe_n_max(8, 0.50f) == 4);
    assert(server_adaptive_dm_probe_n_max(1, 0.25f) == 1);
    assert(server_adaptive_dm_probe_n_max(0, 0.25f) == 0);

    assert_candidates(0,  {0});
    assert_candidates(1,  {0, 1});
    assert_candidates(2,  {0, 1, 2});
    assert_candidates(8,  {0, 1, 2, 3, 4, 5, 6, 7, 8});
    assert_candidates(16, {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16});

    assert(server_adaptive_dm_next_explore_depth(0, 8, 0.25f) == 2);
    assert(server_adaptive_dm_next_explore_depth(2, 8, 0.25f) == 3);
    assert(server_adaptive_dm_next_explore_depth(4, 8, 0.25f) == 5);
    assert(server_adaptive_dm_next_explore_depth(6, 8, 0.25f) == 7);
    assert(server_adaptive_dm_next_explore_depth(8, 8, 0.25f) == 8);

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
        assert(server_adaptive_dm_apply_profit_hysteresis(6, 2, 30.0f, 31.5f, 0.06f, 0.02f, cand, nc8) == 5);

        int cand16[160];
        const int nc16 = server_adaptive_dm_build_candidates(16, cand16, 160);
        assert(server_adaptive_dm_apply_profit_hysteresis(16, 8, 30.0f, 36.0f, 0.06f, 0.02f, cand16, nc16) == 14);
    }

    assert(server_adaptive_dm_should_preserve_for_continuation(0.995f, 1.000f));
    assert(!server_adaptive_dm_should_preserve_for_continuation(0.995f, 0.500f));
    assert(!server_adaptive_dm_should_preserve_for_continuation(0.500f, 1.000f));
    assert(!server_adaptive_dm_should_preserve_for_continuation(0.0f, 0.0f));

    server_adaptive_dm_state state;
    state.fringe_ring_idx = 7;
    state.fringe_ring_count = 13;
    state.rolling_fringe = 0.75f;
    state.adaptive_n_max = 6;
    state.adaptive_probe_counter = 11;
    state.off_dwell = 3;
    state.explore_counter = 19;
    state.fringe_epoch = 4;
    state.fringe_epoch_reached[2] = 9;
    state.fringe_epoch_accepted[2] = 5;
    state.profit_pos_accept_ewma[2] = 0.75f;
    state.profit_pos_samples[2] = 9;
    state.profit_depth[4].samples = 3;
    state.profit_baseline.samples = 2;
    state.profit_has_key = true;
    state.profit_key = {};
    state.profit_key.base_n_max = 8;
    state.profit_key.branch_budget = 0;
    state.profit_key.draft_topk = 1;
    state.profit_key.dflash_cross_ctx = 1024;
    state.profit_key.context_bucket = 0;
    state.profit_key.draft_temp = 0.0f;
    state.profit_key.p_min = 0.0f;
    state.profit_pending = true;
    state.profit_last_recommended_n = 4;
    state.profit_consecutive_below_profit = 2;
    state.profit_cycles_since_baseline = 9;
    state.profit_baseline_probe_pending = true;
    state.profit_baseline_probe_resume_n = 6;
    state.reset_request_state();

    assert(state.fringe_ring_idx == 0);
    assert(state.fringe_ring_count == 0);
    assert(state.rolling_fringe == 0.0f);
    assert(state.adaptive_n_max == -1);
    assert(state.adaptive_probe_counter == 0);
    assert(state.off_dwell == 0);
    assert(state.explore_counter == 0);
    assert(state.fringe_epoch == 5);
    assert(state.fringe_epoch_reached[2] == 0);
    assert(state.fringe_epoch_accepted[2] == 0);
    // profit data should be preserved across request resets
    assert(state.profit_pos_accept_ewma[2] == 0.75f);
    assert(state.profit_pos_samples[2] == 9);
    assert(state.profit_depth[4].samples == 3);
    assert(state.profit_baseline.samples == 2);
    assert(state.profit_has_key == true);
    assert(!state.profit_pending);
    assert(state.profit_last_recommended_n == -1);
    assert(state.profit_consecutive_below_profit == 0);
    assert(state.profit_warmup_cycles == 0);
    assert(state.profit_cycles_since_baseline == 0);
    assert(!state.profit_baseline_probe_pending);
    assert(state.profit_baseline_probe_resume_n == -1);

    state.fringe_epoch = 2;
    state.fringe_ring_idx = 0;
    state.fringe_ring_count = 0;
    state.fringe_ring[state.fringe_ring_idx++] = { 8, 8, 1 };
    state.fringe_ring_count++;
    state.fringe_ring[state.fringe_ring_idx++] = { 0, 8, 2 };
    state.fringe_ring_count++;
    state.fringe_ring[state.fringe_ring_idx++] = { 0, 8, 2 };
    state.fringe_ring_count++;

    auto decision = state.fringe_decision_at_window(8, 1, 1);
    assert(decision.reached == 2);
    assert(decision.accepted == 0);
    assert_close(decision.fringe, 0.0f);

    state.reset_request_state();
    state.fringe_epoch = 3;
    state.fringe_ring[0] = { 0, 5, 3 };
    state.fringe_ring_idx = 1;
    state.fringe_ring_count = 1;
    state.fringe_epoch_reached[4] = 6;
    state.fringe_epoch_accepted[4] = 3;

    decision = state.fringe_decision_at_window(5, 1, 6);
    assert(decision.reached == 6);
    assert(decision.accepted == 3);
    assert_close(decision.fringe, 0.5f);

    assert(server_adaptive_dm_uses_fringe_controller(COMMON_SPECULATIVE_DM_CONTROLLER_FRINGE));
    assert(!server_adaptive_dm_uses_fringe_controller(COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT));
    assert(!server_adaptive_dm_uses_profit_controller(COMMON_SPECULATIVE_DM_CONTROLLER_FRINGE));
    assert(server_adaptive_dm_uses_profit_controller(COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT));

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
    state.observe_profit_acceptance(2, 0);
    state.observe_profit_timing(2, 15.0f, 60.0f, 5.0f, 80.0f);
    {
        const int recommended = state.decide_profit_n_max(8);
        assert(recommended == 0);
        state.apply_profit_recommendation(recommended);
    }
    state.observe_profit_timing(0, 0.0f, 30.0f, 0.0f, 30.0f);
    assert(state.decide_profit_n_max(8) == 0);

    // test warmup: seed no-spec baseline before positive-depth warmup/probing
    state.reset_profit_state();
    state.dm_profit_min_samples = 2;
    state.dm_off_dwell = 1;
    state.profit_warmup_cycles = 3;
    state.adaptive_n_max = 8;
    assert(state.decide_profit_n_max(8) == 0);
    state.observe_profit_timing(0, 0.0f, 30.0f, 0.0f, 30.0f);
    assert(state.decide_profit_n_max(8) == 0);
    state.observe_profit_timing(0, 0.0f, 28.0f, 0.0f, 28.0f);
    assert(state.decide_profit_n_max(8) == 8);

    // test reset_request_state preserves profit data but resets request counters
    state.reset_profit_state();
    state.dm_profit_min_samples = 1;
    state.dm_off_dwell = 1;
    state.adaptive_n_max = 4;
    state.profit_warmup_cycles = 2;
    state.observe_profit_acceptance(4, 2);
    state.observe_profit_timing(4, 20.0f, 50.0f, 3.0f, 73.0f);
    state.profit_pending = true;
    state.profit_consecutive_below_profit = 3;

    state.reset_request_state();
    // request counters reset
    assert(state.adaptive_n_max == -1);
    assert(state.profit_pending == false);
    assert(state.profit_consecutive_below_profit == 0);
    assert(state.profit_warmup_cycles == 0);
    assert(state.profit_last_recommended_n == -1);
    // profit data preserved
    assert(state.profit_pos_samples[0] == 1);
    assert(state.profit_depth[4].samples == 1);
    assert(state.profit_pos_accept_ewma[0] > 0.0f);

    state.dm_profit_min_samples = 1;
    common_params_speculative empty_spec;
    state.reset_profit_if_config_changed(empty_spec, 1, 0);
    // actually need a proper spec struct; let's use the simpler check
    common_params_speculative spec;
    spec.n_max = 8;
    spec.branch_budget = 0;
    spec.draft_topk = 1;
    spec.dflash_cross_ctx = 1024;
    spec.sample_temp = 0.0f;
    spec.p_min = 0.0f;
    state.observe_profit_timing(2, 10.0f, 20.0f, 5.0f, 35.0f);
    state.reset_profit_if_config_changed(spec, 8, 0);
    assert(state.profit_has_key);
    assert(state.profit_depth[2].samples == 0);
    assert(state.adaptive_n_max == -1);
    state.observe_profit_timing(2, 10.0f, 20.0f, 5.0f, 35.0f);
    state.reset_profit_if_config_changed(spec, 8, 0);
    assert(state.profit_depth[2].samples == 1);
    state.adaptive_n_max = 8;
    spec.dflash_cross_ctx = 2048;
    state.reset_profit_if_config_changed(spec, 8, 0);
    assert(state.profit_depth[2].samples == 0);
    assert(state.adaptive_n_max == -1);

    // test a bucket transition cannot leave baseline collection invisible
    {
        server_adaptive_dm_state bucket;
        common_params_speculative bucket_spec;
        bucket_spec.n_max = 16;
        bucket_spec.branch_budget = 0;
        bucket_spec.draft_topk = 1;
        bucket_spec.dflash_cross_ctx = 1024;
        bucket_spec.sample_temp = 0.0f;
        bucket_spec.p_min = 0.0f;

        bucket.reset_profit_if_config_changed(bucket_spec, 16, 1000);
        bucket.observe_profit_timing(0, 0.0f, 30.0f, 0.0f, 30.0f);
        bucket.observe_profit_timing(0, 0.0f, 31.0f, 0.0f, 31.0f);
        bucket.observe_profit_timing(0, 0.0f, 32.0f, 0.0f, 32.0f);
        bucket.apply_profit_recommendation(14);
        assert(bucket.profit_baseline_ready());
        assert(!bucket.profit_expects_baseline_sample());

        bucket.reset_profit_if_config_changed(bucket_spec, 16, 9000);
        assert(bucket.adaptive_n_max == -1);
        assert(!bucket.profit_baseline_ready());
        assert(bucket.profit_expects_baseline_sample());
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
            est.observe_profit_acceptance(8, 6);
            est.observe_profit_timing(8, 10.0f, 55.0f, 2.0f, 67.0f);
        }
        // baseline
        for (int i = 0; i < 3; i++) {
            est.observe_profit_timing(0, 0.0f, 22.0f, 0.0f, 22.0f);
        }

        // depth 8 should be ready, depth 4 should be estimated
        const int recommended = est.decide_profit_n_max(8);
        assert(recommended > 0);
    }

    // test baseline-best shuts speculation fully off after dwell instead of only downshifting
    {
        server_adaptive_dm_state weak;
        weak.dm_profit_min_samples = 1;
        weak.dm_off_dwell = 2;
        weak.adaptive_n_max = 8;
        weak.observe_profit_timing(0, 0.0f, 25.0f, 0.0f, 25.0f);
        weak.observe_profit_acceptance(8, 0);
        weak.observe_profit_timing(8, 10.0f, 90.0f, 5.0f, 105.0f);
        assert(weak.decide_profit_n_max(8) == 8);
        assert(weak.decide_profit_n_max(8) == 0);
    }

    // test active profit cycles trigger periodic no-spec baseline reprobes
    {
        server_adaptive_dm_state reprobe;
        reprobe.dm_profit_min_samples = 1;
        reprobe.dm_profit_baseline_interval = 3;
        common_params_speculative reprobe_spec;
        reprobe_spec.n_max = 8;
        reprobe_spec.branch_budget = 0;
        reprobe_spec.draft_topk = 1;
        reprobe_spec.dflash_cross_ctx = 1024;
        reprobe_spec.sample_temp = 0.0f;
        reprobe_spec.p_min = 0.0f;
        reprobe.reset_profit_if_config_changed(reprobe_spec, 8, 40000);
        reprobe.adaptive_n_max = 8;
        reprobe.observe_profit_timing(0, 0.0f, 40.0f, 0.0f, 40.0f);
        reprobe.observe_profit_acceptance(8, 7);
        reprobe.observe_profit_timing(8, 8.0f, 30.0f, 2.0f, 40.0f);
        assert(!reprobe.profit_should_probe_baseline());
        reprobe.observe_profit_timing(8, 8.0f, 30.0f, 2.0f, 40.0f);
        reprobe.observe_profit_timing(8, 8.0f, 30.0f, 2.0f, 40.0f);
        assert(reprobe.profit_should_probe_baseline());
        reprobe.profit_mark_baseline_probe();
        assert(reprobe.adaptive_n_max == 0);
        assert(reprobe.profit_baseline_probe_pending);
        reprobe.observe_profit_timing(0, 0.0f, 42.0f, 0.0f, 42.0f);
        assert(!reprobe.profit_should_probe_baseline());
        assert(reprobe.decide_profit_n_max(8) == 8);
    }

    // test periodic baseline reprobes wait until longer context buckets
    {
        server_adaptive_dm_state early;
        early.dm_profit_min_samples = 1;
        early.dm_profit_baseline_interval = 1;
        common_params_speculative early_spec;
        early_spec.n_max = 8;
        early_spec.branch_budget = 0;
        early_spec.draft_topk = 1;
        early_spec.dflash_cross_ctx = 1024;
        early_spec.sample_temp = 0.0f;
        early_spec.p_min = 0.0f;
        early.reset_profit_if_config_changed(early_spec, 8, 4096);
        early.adaptive_n_max = 8;
        early.observe_profit_timing(0, 0.0f, 40.0f, 0.0f, 40.0f);
        early.observe_profit_acceptance(8, 7);
        early.observe_profit_timing(8, 8.0f, 30.0f, 2.0f, 40.0f);
        assert(!early.profit_should_probe_baseline());
    }

    return 0;
}
