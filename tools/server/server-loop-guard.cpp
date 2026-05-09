#include "server-loop-guard.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>

server_loop_guard::server_loop_guard(common_reasoning_loop_guard_params params)
    : params(params) {
}

void server_loop_guard::reset() {
    reasoning_tail.clear();
    visible_tail.clear();
    reasoning_seen = 0;
    visible_seen = 0;
}

std::vector<llama_token> & server_loop_guard::tail(server_loop_guard_region region) {
    return region == SERVER_LOOP_REGION_REASONING ? reasoning_tail : visible_tail;
}

const std::vector<llama_token> & server_loop_guard::tail(server_loop_guard_region region) const {
    return region == SERVER_LOOP_REGION_REASONING ? reasoning_tail : visible_tail;
}

void server_loop_guard::accept(llama_token token, server_loop_guard_region region) {
    auto & tokens = tail(region);
    tokens.push_back(token);
    if ((int32_t) tokens.size() > params.window_tokens) {
        tokens.erase(tokens.begin(), tokens.begin() + ((int32_t) tokens.size() - params.window_tokens));
    }

    if (region == SERVER_LOOP_REGION_REASONING) {
        reasoning_seen++;
    } else {
        visible_seen++;
    }
}

int32_t server_loop_guard::seen(server_loop_guard_region region) const {
    return region == SERVER_LOOP_REGION_REASONING ? reasoning_seen : visible_seen;
}

bool server_loop_guard::should_check(server_loop_guard_region region, bool token_is_eog, bool forcing_reasoning_end) const {
    if (params.mode == COMMON_REASONING_LOOP_GUARD_OFF || token_is_eog || forcing_reasoning_end) {
        return false;
    }
    const int32_t n_seen = seen(region);
    if (region == SERVER_LOOP_REGION_REASONING && n_seen < params.min_reasoning_tokens) {
        return false;
    }
    return n_seen > 0 && n_seen % params.check_interval == 0;
}

server_loop_guard_result server_loop_guard::check(server_loop_guard_region region) const {
    if (params.mode == COMMON_REASONING_LOOP_GUARD_OFF) {
        return {};
    }
    if (region == SERVER_LOOP_REGION_REASONING && reasoning_seen < params.min_reasoning_tokens) {
        return {};
    }

    const auto & tokens = tail(region);
    auto periodic = check_periodic_tail(tokens);
    if (periodic.triggered) {
        return periodic;
    }

    if (region == SERVER_LOOP_REGION_REASONING) {
        auto ngram = check_ngram_dominance(tokens);
        if (ngram.triggered) {
            return ngram;
        }
    }

    return check_low_entropy(tokens);
}

server_loop_guard_result server_loop_guard::check_periodic_tail(const std::vector<llama_token> & tokens) const {
    server_loop_guard_result best;
    const int32_t n = (int32_t) tokens.size();
    const int32_t max_period = std::min(params.max_period, n / 3);

    for (int32_t p = 1; p <= max_period; ++p) {
        const int32_t coverage = n - p;
        if (coverage < params.min_repeated_coverage) {
            continue;
        }

        const float threshold = p <= 8 ? 0.995f : (p <= 64 ? 0.990f : 0.980f);
        const int32_t max_mismatches = (int32_t) std::floor((1.0f - threshold) * coverage);
        int32_t mismatches = 0;
        int32_t matches = 0;

        for (int32_t i = p; i < n; ++i) {
            if (tokens[i] == tokens[i - p]) {
                matches++;
            } else if (++mismatches > max_mismatches) {
                break;
            }
        }

        if (mismatches <= max_mismatches) {
            const float score = (float) matches / (float) coverage;
            return server_loop_guard_result {
                true,
                "periodic_tail",
                p,
                coverage,
                score,
            };
        }

        const float score = (float) matches / (float) coverage;
        if (score > best.score) {
            best = server_loop_guard_result {
                false,
                "periodic_tail",
                p,
                coverage,
                score,
            };
        }
    }

    return {};
}

static uint64_t server_loop_guard_hash_ngram(const std::vector<llama_token> & tokens, int start, int ngram) {
    uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < ngram; ++i) {
        hash ^= (uint64_t) (uint32_t) tokens[start + i] + 0x9e3779b97f4a7c15ULL;
        hash *= 1099511628211ULL;
    }
    return hash;
}

static bool server_loop_guard_equal_ngram(const std::vector<llama_token> & tokens, int a, int b, int ngram) {
    for (int i = 0; i < ngram; ++i) {
        if (tokens[a + i] != tokens[b + i]) {
            return false;
        }
    }
    return true;
}

server_loop_guard_result server_loop_guard::check_ngram_dominance(const std::vector<llama_token> & tokens) const {
    const int32_t n = (int32_t) tokens.size();
    if (n < params.min_repeated_coverage) {
        return {};
    }

    const int ngrams[] = {8, 16, 32};
    for (int ngram : ngrams) {
        if (n < ngram * 2) {
            continue;
        }

        std::unordered_map<uint64_t, std::vector<int>> starts_by_hash;
        std::vector<std::pair<int, int>> intervals;

        for (int start = 0; start + ngram <= n; ++start) {
            const uint64_t hash = server_loop_guard_hash_ngram(tokens, start, ngram);
            auto & starts = starts_by_hash[hash];
            for (int previous : starts) {
                if (server_loop_guard_equal_ngram(tokens, previous, start, ngram)) {
                    intervals.emplace_back(previous, previous + ngram);
                    intervals.emplace_back(start, start + ngram);
                    break;
                }
            }
            starts.push_back(start);
        }

        if (intervals.empty()) {
            continue;
        }

        std::sort(intervals.begin(), intervals.end());
        int coverage = 0;
        int cur_begin = intervals[0].first;
        int cur_end = intervals[0].second;
        for (size_t i = 1; i < intervals.size(); ++i) {
            if (intervals[i].first <= cur_end) {
                cur_end = std::max(cur_end, intervals[i].second);
            } else {
                coverage += cur_end - cur_begin;
                cur_begin = intervals[i].first;
                cur_end = intervals[i].second;
            }
        }
        coverage += cur_end - cur_begin;

        if (coverage >= params.min_repeated_coverage && coverage * 2 >= n) {
            return server_loop_guard_result {
                true,
                "ngram_dominance",
                ngram,
                coverage,
                (float) coverage / (float) n,
            };
        }
    }

    return {};
}

server_loop_guard_result server_loop_guard::check_low_entropy(const std::vector<llama_token> & tokens) const {
    const int32_t n_tail = std::min<int32_t>({(int32_t) tokens.size(), params.window_tokens, 1024});
    if (n_tail < 1024 || n_tail < params.min_repeated_coverage) {
        return {};
    }

    std::unordered_map<llama_token, int> counts;
    counts.reserve((size_t) n_tail);
    const int32_t start = (int32_t) tokens.size() - n_tail;
    for (int32_t i = start; i < (int32_t) tokens.size(); ++i) {
        counts[tokens[i]]++;
    }

    std::vector<int> sorted_counts;
    sorted_counts.reserve(counts.size());
    for (const auto & kv : counts) {
        sorted_counts.push_back(kv.second);
    }
    std::sort(sorted_counts.begin(), sorted_counts.end(), std::greater<int>());

    int top8 = 0;
    for (int i = 0; i < std::min<int>(8, sorted_counts.size()); ++i) {
        top8 += sorted_counts[i];
    }

    const float coverage = (float) top8 / (float) n_tail;
    if (counts.size() <= 32 && coverage >= 0.95f) {
        return server_loop_guard_result {
            true,
            "low_entropy",
            0,
            top8,
            coverage,
        };
    }

    return {};
}
