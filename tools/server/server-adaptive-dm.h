#pragma once

#include "common.h"
#include "log.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

static constexpr int SERVER_ADAPTIVE_DM_PROFIT_POSITIONS  = 128;
static constexpr int SERVER_ADAPTIVE_DM_PROFIT_DEPTHS     = SERVER_ADAPTIVE_DM_PROFIT_POSITIONS + 1;
static constexpr int SERVER_ADAPTIVE_DM_PROFIT_CANDIDATES = SERVER_ADAPTIVE_DM_PROFIT_DEPTHS + 1;

static inline int server_adaptive_dm_probe_n_max(int base_n_max, float probe_fraction) {
    if (base_n_max <= 0) {
        return 0;
    }

    const int probe_n_max = std::max(2, (int) (base_n_max * probe_fraction));
    return std::min(probe_n_max, base_n_max);
}

static inline bool server_adaptive_dm_should_preserve_for_continuation(float prompt_similarity, float keep_fraction) {
    return prompt_similarity > 0.90f && keep_fraction > 0.90f;
}

static inline bool server_adaptive_dm_uses_profit_controller(common_speculative_dm_controller controller) {
    return controller == COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT;
}

static inline const char * server_adaptive_dm_controller_name(common_speculative_dm_controller controller) {
    switch (controller) {
        case COMMON_SPECULATIVE_DM_CONTROLLER_OFF:    return "off";
        case COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT: return "profit";
    }
    return "unknown";
}

// Initialize EWMA with first sample, then run normal update.
// Call this instead of checking dst == 0.0f (which is a valid EWMA value).
static inline void server_adaptive_dm_ewma_init_or_update(float & dst, float sample, float alpha, int32_t samples_seen) {
    if (!std::isfinite(sample)) {
        return;
    }
    alpha = std::clamp(alpha, 0.01f, 1.0f);
    if (samples_seen == 0) {
        dst = sample;
    } else {
        dst += alpha * (sample - dst);
    }
}

static inline int server_adaptive_dm_build_candidates(int base_n_max, int * out, int out_cap) {
    if (out_cap <= 0) {
        return 0;
    }

    int n = 0;
    auto add_unique = [&](int candidate) {
        if (candidate < 0 || candidate > base_n_max || n >= out_cap) {
            return;
        }
        bool exists = false;
        for (int i = 0; i < n; ++i) {
            if (out[i] == candidate) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            out[n++] = candidate;
        }
    };

    add_unique(0);
    const int tracked_n_max = std::min(base_n_max, SERVER_ADAPTIVE_DM_PROFIT_POSITIONS);
    for (int candidate = 1; candidate <= tracked_n_max; ++candidate) {
        add_unique(candidate);
    }
    std::sort(out, out + n);
    return n;
}

static inline int server_adaptive_dm_resolve_cohort_n_max(
        const int * normal_n_max,
        const int * cohort_cap_n_max,
        int n_slots) {
    if (!normal_n_max || !cohort_cap_n_max || n_slots < 2) {
        return 0;
    }

    int n_eligible = 0;
    int n_positive = 0;
    int cohort_n_max = INT_MAX;
    for (int i = 0; i < n_slots; ++i) {
        const int slot_cap = cohort_cap_n_max[i];
        if (slot_cap <= 0) {
            continue;
        }

        n_eligible++;
        if (normal_n_max[i] > 0) {
            n_positive++;
        }

        const int slot_limit = normal_n_max[i] > 0
            ? std::min(normal_n_max[i], slot_cap)
            : slot_cap;
        cohort_n_max = std::min(cohort_n_max, slot_limit);
    }

    if (n_eligible < 2 || n_positive == 0 || cohort_n_max == INT_MAX) {
        return 0;
    }

    return std::max(0, cohort_n_max);
}

static inline int server_adaptive_dm_next_explore_depth(int current_n, int base_n_max, float probe_fraction) {
    if (base_n_max <= 0) {
        return 0;
    }
    if (current_n <= 0) {
        return server_adaptive_dm_probe_n_max(base_n_max, probe_fraction);
    }

    int candidates[SERVER_ADAPTIVE_DM_PROFIT_CANDIDATES];
    const int n_candidates = server_adaptive_dm_build_candidates(
            base_n_max, candidates, SERVER_ADAPTIVE_DM_PROFIT_CANDIDATES);
    for (int i = 0; i < n_candidates; ++i) {
        if (candidates[i] > current_n) {
            return candidates[i];
        }
    }
    return std::min(current_n, base_n_max);
}

static inline float server_adaptive_dm_survival_expected_accept(
        const float * pos_accept_ewma,
        const int32_t * pos_samples,
        int n_positions,
        int depth,
        int min_samples,
        bool * ready) {
    bool is_ready = true;
    float expected = 0.0f;

    if (depth <= 0) {
        if (ready) {
            *ready = true;
        }
        return 0.0f;
    }

    // pos_accept_ewma[pos] records cumulative survival P(accepted at least pos+1).
    // Expected accepted tokens = sum of survival probabilities over positions.
    for (int pos = 0; pos < depth; ++pos) {
        if (pos >= n_positions || pos_samples[pos] < min_samples) {
            is_ready = false;
            break;
        }
        if (!std::isfinite(pos_accept_ewma[pos])) {
            is_ready = false;
            break;
        }
        expected += std::clamp(pos_accept_ewma[pos], 0.0f, 1.0f);
    }

    if (ready) {
        *ready = is_ready;
    }
    return is_ready ? expected : 0.0f;
}

static inline float server_adaptive_dm_score(float expected_output_tokens, float cycle_ms) {
    if (cycle_ms <= 0.0f || expected_output_tokens <= 0.0f
            || !std::isfinite(cycle_ms) || !std::isfinite(expected_output_tokens)) {
        return 0.0f;
    }
    return expected_output_tokens * 1000.0f / cycle_ms;
}

struct server_adaptive_dm_profit_candidate {
    int n = 0;
    float score = 0.0f;
    bool ready = false;
    bool estimated = false;
};

static inline server_adaptive_dm_profit_candidate server_adaptive_dm_best_profit_candidate(
        const int * candidates,
        int n_candidates,
        const bool * ready,
        const bool * estimated,
        const float * expected_accept,
        const float * cycle_ms) {
    server_adaptive_dm_profit_candidate best;
    for (int i = 0; i < n_candidates; ++i) {
        if (!ready[i]) {
            continue;
        }
        const float score = server_adaptive_dm_score(1.0f + expected_accept[i], cycle_ms[i]);
        if (!best.ready || score > best.score) {
            best.ready = true;
            best.n = candidates[i];
            best.score = score;
            best.estimated = estimated[i];
        }
    }
    return best;
}

static inline int server_adaptive_dm_apply_profit_hysteresis(
        int current_n,
        int best_n,
        float current_score,
        float best_score,
        float raise_margin,
        float lower_margin,
        const int * candidates,
        int n_candidates) {
    if (best_n > current_n) {
        if (current_score > 0.0f && best_score < current_score * (1.0f + raise_margin)) {
            return current_n;
        }
        const int max_raise_delta = current_n >= 8 ? 4 : 2;
        const int raise_limit = std::min(best_n, current_n + max_raise_delta);
        int next = current_n;
        for (int i = 0; i < n_candidates; ++i) {
            if (candidates[i] > current_n && candidates[i] <= raise_limit) {
                next = candidates[i];
            }
        }
        return next;
    }
    if (best_n < current_n) {
        if (current_score > 0.0f && best_score < current_score * (1.0f + lower_margin)) {
            return current_n;
        }
        // Gradual downshift: move down a couple of candidate steps, not all the way to best_n.
        // Dense candidates should not make demotion materially slower than the old coarse ladder.
        int next = current_n;
        for (int step = 0; step < 2 && next > best_n; ++step) {
            int found = -1;
            for (int i = n_candidates - 1; i >= 0; --i) {
                if (candidates[i] < next) {
                    found = candidates[i];
                    break;
                }
            }
            if (found < 0) break;
            next = found;
        }
        return next;
    }
    return current_n;
}

struct server_adaptive_dm_state {
    static constexpr int PROFIT_POSITIONS = SERVER_ADAPTIVE_DM_PROFIT_POSITIONS;
    static constexpr int PROFIT_DEPTHS = SERVER_ADAPTIVE_DM_PROFIT_DEPTHS;
    static constexpr int PROFIT_CANDIDATES = SERVER_ADAPTIVE_DM_PROFIT_CANDIDATES;

    int32_t adaptive_n_max  = -1;

    bool    dm_adaptive       = true;
    int32_t dm_off_dwell      = 8;
    int32_t dm_probe_interval = 16;
    float   dm_probe_fraction = 0.25f;
    float   dm_profit_high_acceptance_threshold = 0.50f;
    common_speculative_dm_controller dm_controller = COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT;
    float   dm_profit_min          = 0.05f;
    float   dm_profit_raise_margin = 0.05f;
    float   dm_profit_lower_margin = 0.05f;
    float   dm_profit_ewma_alpha   = 0.15f;
    int32_t dm_profit_min_samples  = 3;
    int32_t dm_profit_warmup       = 0;
    int32_t dm_profit_baseline_interval = 1024;

    struct profit_depth_stats {
        int32_t samples = 0;
        float draft_ms  = 0.0f;
        float verify_ms = 0.0f;
        float accept_ms = 0.0f;
        float cycle_ms  = 0.0f;
        float output_tokens = 0.0f;
        float actual_draft_tokens = 0.0f;
        float best_score = 0.0f;
        double total_output_tokens = 0.0;
        double total_cycle_ms = 0.0;
    };

    struct profit_config_key {
        int32_t base_n_max            = -1;
        int32_t draft_n_min           = -1;
        float   draft_p_split          = -1.0f;
        float   draft_p_min            = -1.0f;
        int32_t draft_backend_sampling = -1;
        int32_t draft_cache_type_k     = -1;
        int32_t draft_cache_type_v     = -1;
        int32_t target_top_k     = -1;
        int32_t target_has_grammar = -1;
        int32_t target_grammar_lazy = -1;
        float   target_temp      = -1.0f;
        float   target_top_p     = -1.0f;
        float   target_min_p     = -1.0f;
    };

    float   profit_pos_accept_ewma[PROFIT_POSITIONS] = {};
    int32_t profit_pos_samples[PROFIT_POSITIONS] = {};
    profit_depth_stats profit_depth[PROFIT_DEPTHS] = {};
    profit_depth_stats profit_baseline;
    profit_config_key profit_key;
    bool    profit_has_key = false;
    bool    profit_pending = false;
    int32_t profit_pending_requested_n_max = 0;
    int32_t profit_pending_n_draft = 0;
    int32_t profit_pending_n_accepted = 0;
    uint32_t profit_epoch = 0;
    float   profit_current_score = 0.0f;
    int32_t profit_last_recommended_n = -1;
    int32_t profit_consecutive_below_profit = 0;
    int32_t profit_cycles_since_baseline = 0;
    bool    profit_baseline_probe_pending = false;
    int32_t profit_baseline_probe_resume_n = -1;
    int32_t profit_off_probe_failures = 0;
    int32_t profit_off_probe_next_n = -1;
    int32_t profit_active_episode_n = -1;
    int32_t profit_active_episode_start_samples = 0;
    double  profit_active_episode_start_output_tokens = 0.0;
    double  profit_active_episode_start_cycle_ms = 0.0;
    bool    profit_request_measure_depths = false;
    bool    profit_request_requires_fresh_switch_sample = false;
    int32_t profit_request_baseline_start_samples = 0;
    double  profit_request_baseline_start_output_tokens = 0.0;
    double  profit_request_baseline_start_cycle_ms = 0.0;
    int32_t profit_request_depth_start_samples[PROFIT_DEPTHS] = {};
    double  profit_request_depth_start_output_tokens[PROFIT_DEPTHS] = {};
    double  profit_request_depth_start_cycle_ms[PROFIT_DEPTHS] = {};

    void configure(const common_params_speculative & spec) {
        dm_adaptive = server_adaptive_dm_uses_profit_controller(spec.dm_controller);
        dm_profit_min = spec.dm_profit_min;
        dm_profit_raise_margin = spec.dm_profit_raise_margin;
        dm_profit_lower_margin = spec.dm_profit_lower_margin;
        dm_profit_ewma_alpha = spec.dm_profit_ewma_alpha;
        dm_profit_min_samples = spec.dm_profit_min_samples;
        dm_profit_warmup = spec.dm_profit_warmup;
        dm_profit_baseline_interval = spec.dm_profit_baseline_interval;
    }

    void profit_mark_request_measurement_boundary(bool require_depth_measurements) {
        profit_request_measure_depths = require_depth_measurements;
        profit_request_requires_fresh_switch_sample = false;
        profit_request_baseline_start_samples = profit_baseline.samples;
        profit_request_baseline_start_output_tokens = profit_baseline.total_output_tokens;
        profit_request_baseline_start_cycle_ms = profit_baseline.total_cycle_ms;
        for (int d = 0; d < PROFIT_DEPTHS; ++d) {
            profit_request_depth_start_samples[d] = profit_depth[d].samples;
            profit_request_depth_start_output_tokens[d] = profit_depth[d].total_output_tokens;
            profit_request_depth_start_cycle_ms[d] = profit_depth[d].total_cycle_ms;
        }
    }

    void reset_request_state(bool preserve_active_recommendation = false) {
        const int32_t previous_recommended_n = profit_last_recommended_n;
        const bool previous_recommendation_profitable =
            !preserve_active_recommendation &&
            previous_recommended_n > 0 &&
            profit_baseline_ready() &&
            profit_score_for_depth(previous_recommended_n) >=
                profit_score_for_depth(0) * (1.0f + dm_profit_min);
        const int32_t resume_n_max =
            (preserve_active_recommendation || previous_recommendation_profitable) &&
            previous_recommended_n > 0
                ? previous_recommended_n
                : -1;
        const bool can_resume =
            resume_n_max > 0 &&
            profit_has_key &&
            (profit_key.base_n_max <= 0 || resume_n_max <= profit_key.base_n_max);

        adaptive_n_max = -1;
        reset_request_profit_state(false);
        profit_mark_request_measurement_boundary(!can_resume);
        if (can_resume) {
            profit_request_requires_fresh_switch_sample = true;
            adaptive_n_max = resume_n_max;
            profit_last_recommended_n = resume_n_max;
            profit_current_score = profit_score_for_depth(resume_n_max);
            profit_begin_active_episode(resume_n_max);
        }
    }

    void reset_request_profit_state(bool preserve_recommendation = false) {
        profit_pending = false;
        profit_pending_requested_n_max = 0;
        profit_pending_n_draft = 0;
        profit_pending_n_accepted = 0;
        profit_consecutive_below_profit = 0;
        profit_current_score = 0.0f;
        if (!preserve_recommendation) {
            profit_last_recommended_n = -1;
        }
        profit_cycles_since_baseline = 0;
        profit_baseline_probe_pending = false;
        profit_baseline_probe_resume_n = -1;
        profit_off_probe_failures = 0;
        profit_off_probe_next_n = -1;
        profit_active_episode_n = -1;
        profit_active_episode_start_samples = 0;
        profit_active_episode_start_output_tokens = 0.0;
        profit_active_episode_start_cycle_ms = 0.0;
    }

    void reset_profit_state() {
        std::fill_n(profit_pos_accept_ewma, PROFIT_POSITIONS, 0.0f);
        std::fill_n(profit_pos_samples, PROFIT_POSITIONS, 0);
        std::fill_n(profit_depth, PROFIT_DEPTHS, profit_depth_stats{});
        profit_baseline = profit_depth_stats{};
        profit_key = profit_config_key{};
        profit_has_key = false;
        profit_epoch++;
        reset_request_profit_state();
        profit_mark_request_measurement_boundary(false);
        adaptive_n_max = -1;
    }

    void reset_profit_if_config_changed(
            const common_params_speculative & spec,
            int base_n_max,
            int32_t n_past,
            const common_params_sampling * sampling = nullptr) {
        (void) n_past;
        const profit_config_key next {
            base_n_max,
            spec.draft.n_min,
            spec.draft.p_split,
            spec.draft.p_min,
            (int32_t) spec.draft.backend_sampling,
            (int32_t) spec.draft.cache_type_k,
            (int32_t) spec.draft.cache_type_v,
            sampling ? sampling->top_k : -1,
            sampling ? (common_grammar_value(sampling->grammar).empty() ? 0 : 1) : -1,
            sampling ? (sampling->grammar_lazy ? 1 : 0) : -1,
            sampling ? sampling->temp : -1.0f,
            sampling ? sampling->top_p : -1.0f,
            sampling ? sampling->min_p : -1.0f,
        };
        const bool changed =
            !profit_has_key ||
            profit_key.base_n_max            != next.base_n_max ||
            profit_key.draft_n_min           != next.draft_n_min ||
            profit_key.draft_p_split          != next.draft_p_split ||
            profit_key.draft_p_min            != next.draft_p_min ||
            profit_key.draft_backend_sampling != next.draft_backend_sampling ||
            profit_key.draft_cache_type_k     != next.draft_cache_type_k ||
            profit_key.draft_cache_type_v     != next.draft_cache_type_v ||
            profit_key.target_top_k     != next.target_top_k ||
            profit_key.target_has_grammar != next.target_has_grammar ||
            profit_key.target_grammar_lazy != next.target_grammar_lazy ||
            profit_key.target_temp      != next.target_temp ||
            profit_key.target_top_p     != next.target_top_p ||
            profit_key.target_min_p     != next.target_min_p;
        if (!changed) {
            return;
        }

        reset_profit_state();
        profit_key = next;
        profit_has_key = true;
    }

    bool profit_baseline_ready() const {
        return profit_baseline.samples >= dm_profit_min_samples && profit_baseline.cycle_ms > 0.0f;
    }

    bool profit_depth_ready(int depth) const {
        return depth > 0 &&
            depth < PROFIT_DEPTHS &&
            profit_depth[depth].samples >= dm_profit_min_samples &&
            profit_depth[depth].cycle_ms > 0.0f &&
            profit_depth[depth].output_tokens > 0.0f;
    }

    int profit_depth_samples_since_request(int depth) const {
        if (depth <= 0 || depth >= PROFIT_DEPTHS) {
            return 0;
        }
        return std::max<int>(0, profit_depth[depth].samples - profit_request_depth_start_samples[depth]);
    }

    bool profit_depth_ready_since_request(int depth, int min_samples) const {
        return depth > 0 &&
            depth < PROFIT_DEPTHS &&
            profit_depth[depth].samples >= profit_request_depth_start_samples[depth] + min_samples &&
            profit_depth[depth].total_cycle_ms > profit_request_depth_start_cycle_ms[depth] &&
            profit_depth[depth].total_output_tokens > profit_request_depth_start_output_tokens[depth];
    }

    float profit_score_for_depth_since_request(int depth, int min_samples) const {
        if (!profit_depth_ready_since_request(depth, min_samples)) {
            return 0.0f;
        }
        const double output_tokens =
            profit_depth[depth].total_output_tokens - profit_request_depth_start_output_tokens[depth];
        const double cycle_ms =
            profit_depth[depth].total_cycle_ms - profit_request_depth_start_cycle_ms[depth];
        if (cycle_ms <= 0.0 || output_tokens <= 0.0 ||
                !std::isfinite(cycle_ms) || !std::isfinite(output_tokens)) {
            return 0.0f;
        }
        return server_adaptive_dm_score((float) output_tokens, (float) cycle_ms);
    }

    bool profit_average_for_depth_since_request(
            int depth,
            int min_samples,
            float * output_tokens,
            float * cycle_ms) const {
        if (!profit_depth_ready_since_request(depth, min_samples)) {
            return false;
        }

        const int samples =
            profit_depth[depth].samples - profit_request_depth_start_samples[depth];
        const double total_output_tokens =
            profit_depth[depth].total_output_tokens - profit_request_depth_start_output_tokens[depth];
        const double total_cycle_ms =
            profit_depth[depth].total_cycle_ms - profit_request_depth_start_cycle_ms[depth];
        if (samples <= 0 ||
                total_cycle_ms <= 0.0 ||
                total_output_tokens <= 0.0 ||
                !std::isfinite(total_cycle_ms) ||
                !std::isfinite(total_output_tokens)) {
            return false;
        }

        if (output_tokens) {
            *output_tokens = (float) (total_output_tokens / (double) samples);
        }
        if (cycle_ms) {
            *cycle_ms = (float) (total_cycle_ms / (double) samples);
        }
        return true;
    }

    bool profit_depth_ready_for_decision(int depth) const {
        if (!profit_depth_ready(depth)) {
            return false;
        }
        if (!profit_request_measure_depths) {
            return true;
        }
        return profit_depth_ready_since_request(depth, dm_profit_min_samples);
    }

    bool profit_positive_switch_has_fresh_evidence(int depth, int current_n, int base_n_max) const {
        if (!profit_request_requires_fresh_switch_sample ||
                depth <= 0 ||
                depth == current_n) {
            return true;
        }
        return profit_depth_ready_since_request(depth, profit_candidate_min_samples(base_n_max));
    }

    int profit_initial_probe_min_samples() const {
        return dm_profit_warmup > 0 ?
            std::max<int>(dm_profit_min_samples, dm_profit_warmup) :
            dm_profit_min_samples;
    }

    bool profit_initial_probe_depth_ready(int depth) const {
        if (!profit_request_measure_depths) {
            return depth > 0 &&
                depth < PROFIT_DEPTHS &&
                profit_depth[depth].samples >= profit_initial_probe_min_samples() &&
                profit_depth[depth].cycle_ms > 0.0f &&
                profit_depth[depth].output_tokens > 0.0f;
        }
        return profit_depth_ready_since_request(depth, profit_initial_probe_min_samples());
    }

    bool profit_has_ready_positive_depth() const {
        for (int d = 1; d < PROFIT_DEPTHS; ++d) {
            if (profit_depth_ready(d)) {
                return true;
            }
        }
        return false;
    }

    float profit_score_for_stats(const profit_depth_stats & stats) const {
        if (stats.samples < dm_profit_min_samples ||
                stats.total_cycle_ms <= 0.0 ||
                stats.total_output_tokens <= 0.0 ||
                !std::isfinite(stats.total_cycle_ms) ||
                !std::isfinite(stats.total_output_tokens)) {
            return 0.0f;
        }
        return server_adaptive_dm_score(
                (float) stats.total_output_tokens,
                (float) stats.total_cycle_ms);
    }

    float profit_score_for_depth(int depth) const {
        if (depth == 0) {
            if (!profit_baseline_ready()) {
                return 0.0f;
            }
            return profit_score_for_stats(profit_baseline);
        }
        if (!profit_depth_ready_for_decision(depth)) {
            return 0.0f;
        }
        if (profit_request_measure_depths) {
            const float request_score = profit_score_for_depth_since_request(depth, dm_profit_min_samples);
            if (request_score > 0.0f) {
                return request_score;
            }
            return 0.0f;
        }
        return profit_score_for_stats(profit_depth[depth]);
    }

    int profit_initial_probe_depths(int base_n_max, int * out, int out_cap) const {
        if (out_cap <= 0 || base_n_max <= 0) {
            return 0;
        }

        int n = 0;
        auto add_unique = [&](int value) {
            if (value <= 0 || value > base_n_max || n >= out_cap) {
                return;
            }
            for (int i = 0; i < n; ++i) {
                if (out[i] == value) {
                    return;
                }
            }
            out[n++] = value;
        };

        if (base_n_max >= 12) {
            add_unique(std::max(4, server_adaptive_dm_probe_n_max(base_n_max, dm_probe_fraction)));
            add_unique(std::min(8, base_n_max));
            add_unique(base_n_max);
        } else {
            const int shallow = server_adaptive_dm_probe_n_max(base_n_max, dm_probe_fraction);
            const int mid = std::max(1, base_n_max / 2);
            add_unique(shallow);
            add_unique(mid);
            add_unique(base_n_max);
        }
        return n;
    }

    int profit_next_unready_probe_depth(int base_n_max) const {
        int probes[8];
        const int n_probes = profit_initial_probe_depths(base_n_max, probes, 8);
        for (int i = 0; i < n_probes; ++i) {
            if (!profit_initial_probe_depth_ready(probes[i])) {
                return probes[i];
            }
        }
        return -1;
    }

    int profit_next_lower_rescue_probe_depth(int current_n, int base_n_max) const {
        if (current_n <= 1 || base_n_max < 12) {
            return -1;
        }

        auto usable_unready = [&](int depth) {
            return depth > 0 &&
                depth < current_n &&
                depth <= base_n_max &&
                !profit_ladder_probe_depth_ready(depth);
        };

        const int rescue_ladder[] = {12, 10, 8, 4};
        for (const int depth : rescue_ladder) {
            if (usable_unready(depth)) {
                return depth;
            }
        }
        return -1;
    }

    int profit_first_initial_probe_depth(int base_n_max) const {
        int probes[8];
        const int n_probes = profit_initial_probe_depths(base_n_max, probes, 8);
        return n_probes > 0 ? probes[0] : 0;
    }

    int profit_next_initial_probe_depth_after(int current_n, int base_n_max, bool * wrapped = nullptr) const {
        int probes[8];
        const int n_probes = profit_initial_probe_depths(base_n_max, probes, 8);
        if (n_probes <= 0) {
            if (wrapped) {
                *wrapped = false;
            }
            return 0;
        }
        for (int i = 0; i < n_probes; ++i) {
            if (probes[i] > current_n) {
                if (wrapped) {
                    *wrapped = false;
                }
                return probes[i];
            }
        }
        if (wrapped) {
            *wrapped = true;
        }
        return probes[0];
    }

    int profit_off_probe_depths(int base_n_max, int * out, int out_cap) const {
        if (out_cap <= 0 || base_n_max <= 0) {
            return 0;
        }

        int n = 0;
        auto add_unique = [&](int value) {
            if (value <= 0 || value > base_n_max || n >= out_cap) {
                return;
            }
            for (int i = 0; i < n; ++i) {
                if (out[i] == value) {
                    return;
                }
            }
            out[n++] = value;
        };

        if (base_n_max >= 12) {
            const int raw_probe = server_adaptive_dm_probe_n_max(base_n_max, dm_probe_fraction);
            const int shallow = std::max(4, raw_probe);
            const int mid_low = std::min(8, base_n_max);
            const int mid_high = std::min(12, base_n_max);
            const int anchors[] = {shallow, mid_low, mid_high, base_n_max};

            add_unique(shallow);
            add_unique(mid_low);
            add_unique(mid_high);
            add_unique(base_n_max);
            add_unique(raw_probe - 1);
            add_unique(raw_probe);
            add_unique(1);

            for (int radius = 1; radius <= base_n_max && n < out_cap; ++radius) {
                for (const int anchor : anchors) {
                    add_unique(anchor - radius);
                    add_unique(anchor + radius);
                }
            }
        } else {
            const int shallow = server_adaptive_dm_probe_n_max(base_n_max, dm_probe_fraction);
            const int mid = std::max(1, base_n_max / 2);
            const int anchors[] = {shallow, mid, base_n_max};

            add_unique(shallow);
            add_unique(mid);
            add_unique(base_n_max);
            add_unique(1);

            for (int radius = 1; radius <= base_n_max && n < out_cap; ++radius) {
                for (const int anchor : anchors) {
                    add_unique(anchor - radius);
                    add_unique(anchor + radius);
                }
            }
        }

        return n;
    }

    int profit_first_off_probe_depth(int base_n_max) const {
        int probes[PROFIT_CANDIDATES];
        const int n_probes = profit_off_probe_depths(base_n_max, probes, PROFIT_CANDIDATES);
        return n_probes > 0 ? probes[0] : 0;
    }

    int profit_next_off_probe_depth_after(int current_n, int base_n_max, bool * wrapped = nullptr) const {
        int probes[PROFIT_CANDIDATES];
        const int n_probes = profit_off_probe_depths(base_n_max, probes, PROFIT_CANDIDATES);
        if (n_probes <= 0) {
            if (wrapped) {
                *wrapped = false;
            }
            return 0;
        }
        for (int i = 0; i < n_probes; ++i) {
            if (probes[i] == current_n) {
                if (wrapped) {
                    *wrapped = i + 1 >= n_probes;
                }
                return probes[(i + 1) % n_probes];
            }
        }
        if (wrapped) {
            *wrapped = true;
        }
        return probes[0];
    }

    int profit_next_off_probe_depth(int base_n_max, int fallback_probe_n_max) const {
        if (base_n_max <= 0) {
            return 0;
        }

        if (profit_off_probe_next_n > 0) {
            return std::clamp<int>(profit_off_probe_next_n, 1, base_n_max);
        }

        const int first = profit_first_off_probe_depth(base_n_max);
        if (first > 0) {
            return first;
        }

        return std::clamp<int>(fallback_probe_n_max, 0, base_n_max);
    }

    void profit_prepare_off_probe_sweep(int base_n_max) {
        const int first = profit_first_off_probe_depth(base_n_max);
        profit_off_probe_next_n = first > 0 ? first : -1;
    }

    void profit_note_off_probe_result(int probed_n, int base_n_max, bool profitable) {
        if (profitable) {
            profit_off_probe_failures = 0;
            profit_prepare_off_probe_sweep(base_n_max);
            return;
        }

        bool wrapped = false;
        const int next = profit_next_off_probe_depth_after(probed_n, base_n_max, &wrapped);
        profit_off_probe_next_n = next > 0 ? next : -1;
        if (wrapped) {
            profit_off_probe_failures++;
        }
    }

    int profit_active_explore_budget(int base_n_max) const {
        if (base_n_max <= 1) {
            return 0;
        }
        const int budget = base_n_max <= 8 ? 6 : 7;
        return std::min(base_n_max - 1, budget);
    }

    bool profit_survival_expected_accept(int depth, float * expected_accept) const {
        bool ready = false;
        const float expected = server_adaptive_dm_survival_expected_accept(
                profit_pos_accept_ewma,
                profit_pos_samples,
                PROFIT_POSITIONS,
                depth,
                dm_profit_min_samples,
                &ready);
        if (ready && expected_accept) {
            *expected_accept = expected;
        }
        return ready;
    }

    bool profit_active_explore_prefers_higher(int current_n, int base_n_max) const {
        if (current_n <= 0 || current_n >= base_n_max) {
            return false;
        }

        float expected_accept = 0.0f;
        if (!profit_survival_expected_accept(current_n, &expected_accept)) {
            return false;
        }

        return expected_accept >= (float) current_n * dm_profit_high_acceptance_threshold;
    }

    bool profit_should_skip_active_explore(int current_n, int base_n_max) const {
        if (current_n <= 0 || current_n < base_n_max) {
            return false;
        }
        if (!profit_baseline_ready() ||
                !profit_active_episode_ready(current_n, base_n_max)) {
            return false;
        }

        const float baseline_score = profit_score_for_depth(0);
        const float active_score = profit_active_episode_score(current_n);
        if (baseline_score <= 0.0f ||
                active_score < baseline_score * (1.0f + dm_profit_min)) {
            return false;
        }

        float expected_accept = 0.0f;
        if (!profit_survival_expected_accept(current_n, &expected_accept)) {
            return false;
        }

        return expected_accept >= (float) current_n * dm_profit_high_acceptance_threshold;
    }

    int profit_active_explore_depths(int current_n, int base_n_max, int * out, int out_cap) const {
        if (out_cap <= 0 || base_n_max <= 0) {
            return 0;
        }

        const int current = std::clamp<int>(current_n, 0, base_n_max);
        const int budget = std::min(out_cap, profit_active_explore_budget(base_n_max));
        if (budget <= 0) {
            return 0;
        }

        int n = 0;
        auto add_unique = [&](int value) {
            if (value <= 0 || value > base_n_max || value == current || n >= budget) {
                return;
            }
            for (int i = 0; i < n; ++i) {
                if (out[i] == value) {
                    return;
                }
            }
            out[n++] = value;
        };

        if (current <= 0) {
            add_unique(server_adaptive_dm_probe_n_max(base_n_max, dm_probe_fraction));
            add_unique(std::max(1, base_n_max / 2));
            add_unique(base_n_max);
            return n;
        }

        const bool prefer_higher = profit_active_explore_prefers_higher(current, base_n_max);
        const int local_span = std::min(base_n_max, std::max(3, std::min(5, base_n_max / 3 + 1)));
        for (int delta = 1; delta <= local_span && n < budget; ++delta) {
            const int lower = current - delta;
            const int higher = current + delta;
            if (prefer_higher) {
                add_unique(higher);
                add_unique(lower);
            } else {
                add_unique(lower);
                add_unique(higher);
            }
        }

        if (n < budget) {
            if (prefer_higher) {
                add_unique(current + std::max(1, (base_n_max - current) / 2));
                add_unique(base_n_max);
                add_unique(current - std::max(1, current / 4));
                add_unique(std::max(1, current / 2));
            } else {
                add_unique(current - std::max(1, current / 4));
                add_unique((current + 1) / 2);
                add_unique(std::min(base_n_max,
                            std::max(4, server_adaptive_dm_probe_n_max(base_n_max, dm_probe_fraction))));
                add_unique(base_n_max);
            }
        }

        return n;
    }

    int profit_explore_depth_for_step(int current_n, int base_n_max, int explore_step) const {
        if (base_n_max <= 0) {
            return 0;
        }

        int ordered[PROFIT_CANDIDATES];
        const int n_ordered = profit_active_explore_depths(
                current_n, base_n_max, ordered, PROFIT_CANDIDATES);
        if (n_ordered <= 0) {
            return std::clamp<int>(current_n, 0, base_n_max);
        }

        const int index = (std::max(1, explore_step) - 1) % n_ordered;
        return ordered[index];
    }

    int profit_next_unready_explore_depth(int current_n, int base_n_max, int explore_step) const {
        (void) explore_step;

        if (profit_should_skip_active_explore(current_n, base_n_max)) {
            return 0;
        }

        int ordered[PROFIT_CANDIDATES];
        const int n_ordered = profit_active_explore_depths(
                current_n, base_n_max, ordered, PROFIT_CANDIDATES);
        if (n_ordered <= 0) {
            return 0;
        }

        for (int i = 0; i < n_ordered; ++i) {
            const int candidate = ordered[i];
            if (candidate > 0 && !profit_explore_probe_depth_ready(candidate, current_n, base_n_max)) {
                return candidate;
            }
        }
        return 0;
    }

    bool profit_explore_probe_depth_ready(int depth, int current_n, int base_n_max) const {
        if (depth <= 0 || depth >= PROFIT_DEPTHS) {
            return false;
        }

        const int min_samples = profit_candidate_min_samples(base_n_max);
        if (profit_request_measure_depths ||
                (profit_request_requires_fresh_switch_sample && depth != current_n)) {
            return profit_depth_ready_since_request(depth, min_samples);
        }
        return profit_depth[depth].samples >= min_samples &&
            profit_depth[depth].total_cycle_ms > 0.0 &&
            profit_depth[depth].total_output_tokens > 0.0;
    }

    bool profit_ladder_probe_depth_ready(int depth) const {
        if (!profit_request_measure_depths) {
            return depth > 0 &&
                depth < PROFIT_DEPTHS &&
                profit_depth[depth].samples >= profit_initial_probe_min_samples() &&
                profit_depth[depth].cycle_ms > 0.0f &&
                profit_depth[depth].output_tokens > 0.0f;
        }
        return profit_depth_ready_since_request(depth, profit_initial_probe_min_samples());
    }

    int profit_candidate_min_samples() const {
        return std::max<int>(2, dm_profit_min_samples);
    }

    int profit_active_min_samples(int base_n_max) const {
        const int capped_window = base_n_max <= 8 ? 3 : 6;
        return std::max<int>(dm_profit_min_samples, capped_window);
    }

    int profit_candidate_min_samples(int base_n_max) const {
        return std::max<int>(profit_candidate_min_samples(), profit_active_min_samples(base_n_max));
    }

    bool profit_candidate_ready_for_switch(int depth) const {
        return depth > 0 &&
            depth < PROFIT_DEPTHS &&
            profit_depth[depth].samples >= profit_candidate_min_samples() &&
            profit_depth[depth].total_cycle_ms > 0.0 &&
            profit_depth[depth].total_output_tokens > 0.0;
    }

    bool profit_candidate_ready_for_switch(int depth, int base_n_max) const {
        if (depth <= 0 || depth >= PROFIT_DEPTHS) {
            return false;
        }
        const int min_samples = profit_candidate_min_samples(base_n_max);
        if (profit_request_measure_depths) {
            return profit_depth_ready_since_request(depth, min_samples);
        }
        return profit_depth[depth].samples >= min_samples &&
            profit_depth[depth].total_cycle_ms > 0.0 &&
            profit_depth[depth].total_output_tokens > 0.0;
    }

    void profit_begin_active_episode_from_stats(int depth, const profit_depth_stats & stats) {
        if (depth <= 0 || depth >= PROFIT_DEPTHS) {
            profit_active_episode_n = -1;
            profit_active_episode_start_samples = 0;
            profit_active_episode_start_output_tokens = 0.0;
            profit_active_episode_start_cycle_ms = 0.0;
            return;
        }

        profit_active_episode_n = depth;
        profit_active_episode_start_samples = stats.samples;
        profit_active_episode_start_output_tokens = stats.total_output_tokens;
        profit_active_episode_start_cycle_ms = stats.total_cycle_ms;
    }

    void profit_begin_active_episode(int depth) {
        if (depth <= 0 || depth >= PROFIT_DEPTHS) {
            profit_begin_active_episode_from_stats(-1, profit_depth_stats{});
            return;
        }
        profit_begin_active_episode_from_stats(depth, profit_depth[depth]);
    }

    bool profit_active_episode_ready(int depth, int base_n_max) const {
        if (depth <= 0 || depth >= PROFIT_DEPTHS || profit_active_episode_n != depth) {
            return false;
        }
        const profit_depth_stats & stats = profit_depth[depth];
        return stats.samples - profit_active_episode_start_samples >= profit_active_min_samples(base_n_max) &&
            stats.total_cycle_ms > profit_active_episode_start_cycle_ms &&
            stats.total_output_tokens > profit_active_episode_start_output_tokens;
    }

    float profit_active_episode_score(int depth) const {
        if (depth <= 0 || depth >= PROFIT_DEPTHS || profit_active_episode_n != depth) {
            return 0.0f;
        }
        const profit_depth_stats & stats = profit_depth[depth];
        const double output_tokens = stats.total_output_tokens - profit_active_episode_start_output_tokens;
        const double cycle_ms = stats.total_cycle_ms - profit_active_episode_start_cycle_ms;
        if (cycle_ms <= 0.0 || output_tokens <= 0.0 ||
                !std::isfinite(cycle_ms) || !std::isfinite(output_tokens)) {
            return 0.0f;
        }
        return server_adaptive_dm_score((float) output_tokens, (float) cycle_ms);
    }

    bool profit_initial_probe_set_ready(int base_n_max) const {
        return profit_baseline_ready() && profit_next_unready_probe_depth(base_n_max) < 0;
    }

    int profit_off_probe_interval() const {
        const int shift = std::clamp<int>(profit_off_probe_failures, 0, 6);
        const int multiplier = 1 << shift;
        const int base = std::max(1, (int) dm_probe_interval);
        if (base > INT_MAX / multiplier) {
            return INT_MAX;
        }
        return base * multiplier;
    }

    bool profit_should_probe_baseline() const {
        return dm_profit_baseline_interval > 0 &&
            profit_baseline_ready() &&
            !profit_baseline_probe_pending &&
            adaptive_n_max > 0 &&
            profit_cycles_since_baseline >= dm_profit_baseline_interval;
    }

    bool profit_expects_baseline_sample() const {
        return profit_baseline_probe_pending ||
            !profit_baseline_ready() ||
            adaptive_n_max == 0;
    }

    void profit_mark_baseline_probe(int resume_n = -1) {
        profit_baseline_probe_pending = true;
        profit_baseline_probe_resume_n = resume_n >= 0 ? resume_n : adaptive_n_max;
        adaptive_n_max = 0;
    }

    void apply_profit_recommendation(int recommended_n) {
        const int previous_n = adaptive_n_max;
        profit_last_recommended_n = recommended_n;
        adaptive_n_max = recommended_n;
        if (recommended_n > 0 && recommended_n != previous_n) {
            profit_begin_active_episode(recommended_n);
        } else if (recommended_n <= 0) {
            profit_begin_active_episode(-1);
        }
    }

    void observe_profit_acceptance(int n_draft, int n_accepted) {
        const int reached_positions = std::min<int>(n_draft, PROFIT_POSITIONS);
        for (int pos = 0; pos < reached_positions; ++pos) {
            const bool accepted = n_accepted > pos;
            const float sample = accepted ? 1.0f : 0.0f;
            server_adaptive_dm_ewma_init_or_update(profit_pos_accept_ewma[pos], sample, dm_profit_ewma_alpha,
                    profit_pos_samples[pos]);
            profit_pos_samples[pos]++;
        }
    }

    void observe_profit_timing(
            int requested_n_max,
            int actual_n_draft,
            int n_accepted,
            float draft_ms,
            float verify_ms,
            float accept_ms,
            float cycle_ms) {
        profit_depth_stats * dst = nullptr;
        if (requested_n_max <= 0) {
            dst = &profit_baseline;
        } else if (requested_n_max < PROFIT_DEPTHS) {
            dst = &profit_depth[requested_n_max];
        }
        if (!dst || cycle_ms <= 0.0f) {
            if (!dst && requested_n_max >= PROFIT_DEPTHS) {
                LOG_DBG("observe_profit_timing: requested_n_max=%d >= PROFIT_DEPTHS=%d, "
                        "timing data dropped\n", requested_n_max, PROFIT_DEPTHS);
            }
            return;
        }

        if (!std::isfinite(draft_ms) || !std::isfinite(verify_ms) ||
                !std::isfinite(accept_ms) || !std::isfinite(cycle_ms)) {
            return;
        }

        const float output_tokens = requested_n_max <= 0 ? 1.0f : 1.0f + (float) std::max(0, n_accepted);
        const float actual_draft_tokens = (float) std::max(0, actual_n_draft);

        if (requested_n_max > 0 &&
                requested_n_max == adaptive_n_max &&
                profit_last_recommended_n > 0 &&
                profit_active_episode_n != requested_n_max) {
            profit_begin_active_episode_from_stats(requested_n_max, *dst);
        }

        server_adaptive_dm_ewma_init_or_update(dst->draft_ms, draft_ms, dm_profit_ewma_alpha, dst->samples);
        server_adaptive_dm_ewma_init_or_update(dst->verify_ms, verify_ms, dm_profit_ewma_alpha, dst->samples);
        server_adaptive_dm_ewma_init_or_update(dst->accept_ms, accept_ms, dm_profit_ewma_alpha, dst->samples);
        server_adaptive_dm_ewma_init_or_update(dst->cycle_ms, cycle_ms, dm_profit_ewma_alpha, dst->samples);
        server_adaptive_dm_ewma_init_or_update(dst->output_tokens, output_tokens, dm_profit_ewma_alpha, dst->samples);
        server_adaptive_dm_ewma_init_or_update(dst->actual_draft_tokens, actual_draft_tokens, dm_profit_ewma_alpha, dst->samples);
        dst->best_score = std::max(dst->best_score, server_adaptive_dm_score(output_tokens, cycle_ms));
        dst->total_output_tokens += (double) output_tokens;
        dst->total_cycle_ms += (double) cycle_ms;
        dst->samples++;

        if (requested_n_max <= 0) {
            profit_cycles_since_baseline = 0;
            profit_baseline_probe_pending = false;
        } else if (profit_baseline_ready()) {
            profit_cycles_since_baseline++;
        }
    }

    void observe_profit_timing(int n_draft, float draft_ms, float verify_ms, float accept_ms, float cycle_ms) {
        observe_profit_timing(n_draft, n_draft, 0, draft_ms, verify_ms, accept_ms, cycle_ms);
    }

    int decide_profit_n_max(int base_n_max) {
        if (base_n_max <= 0) {
            profit_last_recommended_n = 0;
            return 0;
        }

        const float baseline_score = profit_score_for_depth(0);
        if (!profit_baseline_ready()) {
            profit_current_score = 0.0f;
            profit_last_recommended_n = 0;
            return 0;
        }

        const int current_n = profit_baseline_probe_resume_n > 0
            ? std::clamp<int>(profit_baseline_probe_resume_n, 0, base_n_max)
            : (adaptive_n_max < 0
                ? base_n_max
                : std::clamp<int>(adaptive_n_max, 0, base_n_max));
        const bool returning_from_baseline_probe = profit_baseline_probe_resume_n > 0;
        const bool collecting_initial_probe_set =
            profit_baseline_probe_resume_n <= 0 &&
            !profit_request_requires_fresh_switch_sample &&
            !profit_initial_probe_set_ready(base_n_max);
        // Cold-start bootstrap: once the no-spec baseline exists, run production at
        // the maximum draft depth and let the scheduler characterize the lower probe
        // spread through transient exploration excursions (see get_dflash_n_draft_max).
        // Holding max keeps short requests fast instead of stranding them at a low
        // probe depth, and the argmax candidate scorer below demotes only when a
        // measured lower depth is genuinely faster. If the held positive depth is
        // measured to be clearly worse than no-spec, fall through so a better
        // characterized depth (or baseline) can take over without waiting for the
        // full spread.
        if (collecting_initial_probe_set) {
            const int hold_n = current_n > 0 ? current_n : base_n_max;
            const bool hold_episode_clearly_bad =
                current_n > 0 &&
                profit_active_episode_ready(current_n, base_n_max) &&
                profit_active_episode_score(current_n) < baseline_score * (1.0f + dm_profit_min);
            if (!hold_episode_clearly_bad) {
                profit_current_score = baseline_score;
                profit_last_recommended_n = hold_n;
                return hold_n;
            }
        }

        if (current_n == 0 && profit_baseline_probe_resume_n <= 0) {
            profit_current_score = baseline_score;
            profit_last_recommended_n = 0;
            return 0;
        }

        int candidates[PROFIT_CANDIDATES];
        int n_candidates = server_adaptive_dm_build_candidates(base_n_max, candidates, PROFIT_CANDIDATES);
        {
            bool found = false;
            for (int i = 0; i < n_candidates; i++) {
                if (candidates[i] == current_n) { found = true; break; }
            }
            if (!found && n_candidates < PROFIT_CANDIDATES && current_n > 0) {
                candidates[n_candidates++] = current_n;
                std::sort(candidates, candidates + n_candidates);
            }
        }

        bool ready[PROFIT_CANDIDATES] = {};
        bool estimated[PROFIT_CANDIDATES] = {};
        float expected_accept[PROFIT_CANDIDATES] = {};
        float cycle_ms[PROFIT_CANDIDATES] = {};
        float direct_score[PROFIT_CANDIDATES] = {};

        float current_score = 0.0f;
        bool current_ready = false;
        const bool baseline_ready = profit_baseline_ready();
        for (int i = 0; i < n_candidates; ++i) {
            const int n = candidates[i];
            if (n == 0) {
                ready[i] = baseline_ready;
                estimated[i] = false;
                expected_accept[i] = 0.0f;
                cycle_ms[i] = profit_baseline.cycle_ms;
                direct_score[i] = baseline_score;
            } else if (profit_depth_ready_for_decision(n)) {
                const bool fresh_challenger_required =
                    profit_request_requires_fresh_switch_sample &&
                    n != current_n;
                if (fresh_challenger_required &&
                        !profit_depth_ready_since_request(n, profit_candidate_min_samples(base_n_max))) {
                    continue;
                }

                bool positions_ready = false;
                const float survival_expected = server_adaptive_dm_survival_expected_accept(
                        profit_pos_accept_ewma, profit_pos_samples, PROFIT_POSITIONS,
                        n, dm_profit_min_samples, &positions_ready);
                const bool lower_from_current_survival = n < current_n && positions_ready;
                float request_output_tokens = 0.0f;
                float request_cycle_ms = 0.0f;
                const bool use_request_average =
                    fresh_challenger_required &&
                    profit_average_for_depth_since_request(
                            n,
                            profit_candidate_min_samples(base_n_max),
                            &request_output_tokens,
                            &request_cycle_ms);
                expected_accept[i] = use_request_average ?
                    std::max(0.0f, request_output_tokens - 1.0f) :
                    (lower_from_current_survival ?
                        survival_expected :
                        std::max(0.0f, profit_depth[n].output_tokens - 1.0f));
                ready[i] = true;
                estimated[i] = false;
                cycle_ms[i] = use_request_average ? request_cycle_ms : profit_depth[n].cycle_ms;
                direct_score[i] = use_request_average ?
                    server_adaptive_dm_score(request_output_tokens, request_cycle_ms) :
                    (lower_from_current_survival ?
                    server_adaptive_dm_score(1.0f + expected_accept[i], cycle_ms[i]) :
                    profit_score_for_depth(n));
            } else if (n > 0) {
                // Find the ready depth closest to THIS candidate for extrapolation.
                int ref_depth = -1;
                int ref_dist  = INT_MAX;
                for (int d = 1; d < PROFIT_DEPTHS; ++d) {
                    if (profit_depth_ready_for_decision(d)) {
                        const int dist = std::abs(d - n);
                        if (dist < ref_dist) { ref_dist = dist; ref_depth = d; }
                    }
                }
                if (ref_depth > 0) {
                    bool positions_ready = false;
                    expected_accept[i] = server_adaptive_dm_survival_expected_accept(
                            profit_pos_accept_ewma, profit_pos_samples, PROFIT_POSITIONS,
                            n, dm_profit_min_samples, &positions_ready);
                    if (positions_ready) {
                        const profit_depth_stats & ref = profit_depth[ref_depth];
                        const float draft_per_token = ref.draft_ms / (float) ref_depth;
                        const float est_cycle_ms = (float) n * draft_per_token + ref.verify_ms + ref.accept_ms;
                        if (est_cycle_ms > 0.0f) {
                            ready[i]     = true;
                            estimated[i] = true;
                            cycle_ms[i]  = est_cycle_ms;
                        }
                    }
                }
            }

            if (n == current_n && ready[i]) {
                current_ready = true;
                current_score = direct_score[i] > 0.0f ?
                    direct_score[i] :
                    server_adaptive_dm_score(1.0f + expected_accept[i], cycle_ms[i]);
            }

            if (ready[i]) {
                const float score = direct_score[i] > 0.0f ?
                    direct_score[i] :
                    server_adaptive_dm_score(1.0f + expected_accept[i], cycle_ms[i]);
                LOG_DBG("profit cand: n=%d ready=1 est=%d exp=%.2f cycle=%.1f score=%.2f%s\n",
                        n, estimated[i] ? 1 : 0,
                        expected_accept[i], cycle_ms[i], score,
                        n == current_n ? " (current)" : "");
            }
        }

        if (!current_ready && current_n > 0) {
            // estimate current_score so hysteresis is never bypassed
            int ref_depth = -1;
            int ref_dist  = INT_MAX;
            for (int d = 1; d < PROFIT_DEPTHS; ++d) {
                if (profit_depth_ready_for_decision(d)) {
                    const int dist = std::abs(d - current_n);
                    if (dist < ref_dist) { ref_dist = dist; ref_depth = d; }
                }
            }
            if (ref_depth > 0) {
                bool pos_ready = false;
                const float ea = server_adaptive_dm_survival_expected_accept(
                        profit_pos_accept_ewma, profit_pos_samples, PROFIT_POSITIONS,
                        current_n, dm_profit_min_samples, &pos_ready);
                if (pos_ready) {
                    const profit_depth_stats & ref = profit_depth[ref_depth];
                    const float draft_per_token = ref.draft_ms / (float) ref_depth;
                    const float est_cycle = (float) current_n * draft_per_token + ref.verify_ms + ref.accept_ms;
                    if (est_cycle > 0.0f) {
                        current_ready = true;
                        current_score = server_adaptive_dm_score(1.0f + ea, est_cycle);
                    }
                }
            }
        }

        const bool evaluating_off_probe =
            profit_last_recommended_n == 0 &&
            current_n > 0 &&
            profit_baseline_probe_resume_n <= 0;
        if (evaluating_off_probe) {
            const float off_probe_score = profit_score_for_depth(current_n);
            const bool current_profitable =
                off_probe_score >= baseline_score * (1.0f + dm_profit_min);
            if (!current_profitable) {
                profit_note_off_probe_result(current_n, base_n_max, false);
                profit_consecutive_below_profit = 0;
                profit_current_score = off_probe_score > 0.0f ? off_probe_score : baseline_score;
                profit_last_recommended_n = 0;
                return 0;
            }
            profit_note_off_probe_result(current_n, base_n_max, true);
            profit_consecutive_below_profit = 0;
            profit_current_score = off_probe_score;
            profit_last_recommended_n = current_n;
            return current_n;
        }

        auto best = server_adaptive_dm_best_profit_candidate(
                candidates, n_candidates, ready, estimated, expected_accept, cycle_ms);
        server_adaptive_dm_profit_candidate best_direct;
        for (int i = 0; i < n_candidates; ++i) {
            if (!ready[i] || estimated[i]) {
                continue;
            }
            const float score = direct_score[i] > 0.0f ?
                direct_score[i] :
                server_adaptive_dm_score(1.0f + expected_accept[i], cycle_ms[i]);
            if (!best_direct.ready || score > best_direct.score) {
                best_direct.ready = true;
                best_direct.n = candidates[i];
                best_direct.score = score;
                best_direct.estimated = false;
            }
        }
        if (best_direct.ready) {
            best = best_direct;
        }
        if (!best.ready) {
            profit_current_score = current_score;
            profit_last_recommended_n = current_n;
            return current_n;
        }

        int recommended = best.n;
        const bool estimated_demote =
            best.estimated &&
            best.n > 0 &&
            best.n < current_n &&
            current_ready;
        const bool active_episode_ready =
            current_n > 0 &&
            profit_active_episode_ready(current_n, base_n_max);
        const float active_episode_score =
            active_episode_ready ? profit_active_episode_score(current_n) : 0.0f;
        const bool baseline_wins =
            best.n == 0 ||
            (best.n > 0 && best.score < baseline_score * (1.0f + dm_profit_min));
        if (estimated_demote && (!baseline_ready ||
                    current_score >= baseline_score * (1.0f + dm_profit_min))) {
            recommended = current_n;
        } else if (baseline_ready && baseline_wins) {
            if (current_n > 0 &&
                    !returning_from_baseline_probe &&
                    !active_episode_ready) {
                recommended = current_n;
                profit_consecutive_below_profit = 0;
                profit_current_score = current_ready ? current_score : best.score;
                profit_last_recommended_n = recommended;
                profit_baseline_probe_resume_n = -1;
                return recommended;
            }
            if (current_n > 0 &&
                    active_episode_ready &&
                    active_episode_score >= baseline_score * (1.0f + dm_profit_min)) {
                recommended = current_n;
                profit_consecutive_below_profit = 0;
                profit_current_score = active_episode_score;
                profit_last_recommended_n = recommended;
                profit_baseline_probe_resume_n = -1;
                return recommended;
            }
            const int lower_rescue_probe = profit_next_lower_rescue_probe_depth(current_n, base_n_max);
            if (current_n > 0 &&
                    !returning_from_baseline_probe &&
                    active_episode_ready &&
                    lower_rescue_probe > 0) {
                recommended = lower_rescue_probe;
                profit_consecutive_below_profit = 0;
                profit_current_score = active_episode_score;
                profit_last_recommended_n = recommended;
                profit_baseline_probe_resume_n = -1;
                return recommended;
            }
            profit_consecutive_below_profit++;
            const bool disable_now =
                returning_from_baseline_probe ||
                profit_consecutive_below_profit >= dm_off_dwell;
            recommended = disable_now ? 0 : current_n;
            if (disable_now && current_n > 0) {
                profit_prepare_off_probe_sweep(base_n_max);
                LOG_INF("adaptive-dm: disabling speculative depth "
                        "(best_score=%.3f baseline=%.3f profit_min=%.3f below_count=%d)\n",
                        best.score, baseline_score,
                        dm_profit_min, profit_consecutive_below_profit);
            }
        } else {
            profit_consecutive_below_profit = 0;
            profit_off_probe_failures = 0;
            const bool positive_switch =
                current_n > 0 &&
                best.n > 0 &&
                best.n != current_n;
            const bool unconfirmed_positive_switch =
                positive_switch &&
                !best.estimated &&
                !profit_candidate_ready_for_switch(best.n, base_n_max);
            const bool stale_preserved_positive_switch =
                positive_switch &&
                !profit_positive_switch_has_fresh_evidence(best.n, current_n, base_n_max);
            if (positive_switch && !active_episode_ready) {
                recommended = current_n;
            } else if (unconfirmed_positive_switch) {
                recommended = current_n;
            } else if (stale_preserved_positive_switch) {
                recommended = current_n;
            } else {
                recommended = server_adaptive_dm_apply_profit_hysteresis(
                        current_n,
                        best.n,
                        current_ready ? current_score : 0.0f,
                        best.score,
                        dm_profit_raise_margin,
                        dm_profit_lower_margin,
                        candidates,
                        n_candidates);
            }
        }

        const bool positive_demotion =
            current_n > 0 &&
            recommended > 0 &&
            recommended < current_n;
        if (positive_demotion &&
                baseline_ready &&
                !returning_from_baseline_probe &&
                !profit_baseline_probe_pending &&
                dm_profit_baseline_interval > 0 &&
                profit_cycles_since_baseline >= dm_profit_baseline_interval) {
            profit_mark_baseline_probe(current_n);
            profit_current_score = current_ready ? current_score : best.score;
            profit_last_recommended_n = 0;
            return 0;
        }

        recommended = std::clamp(recommended, 0, base_n_max);
        profit_current_score = best.score;
        profit_last_recommended_n = recommended;
        profit_baseline_probe_resume_n = -1;

        LOG_DBG("profit decide: current_n=%d best_n=%d rec=%d "
                "best_ready=%d best_est=%d best_score=%.3f "
                "curr_ready=%d curr_score=%.3f base_n=%d\n",
                current_n, best.n, recommended,
                best.ready ? 1 : 0, best.estimated ? 1 : 0,
                best.score,
                current_ready ? 1 : 0, current_score, base_n_max);

        return recommended;
    }

};
