#include "fit-kvarn-tail.h"

#include <algorithm>
#include <limits>

common_bee_fit_retry_action common_bee_fit_retry_state::observe(
        const std::string & candidate_identity,
        const std::vector<int64_t> & shortfalls,
        std::vector<size_t> & margins) {
    if (shortfalls.size() != margins.size()) {
        return COMMON_BEE_FIT_INVALID_SHORTFALL;
    }
    if (std::none_of(shortfalls.begin(), shortfalls.end(), [](int64_t value) {
            return value > 0;
        })) {
        return COMMON_BEE_FIT_ACCEPT;
    }
    if (!rejected_candidates.insert(candidate_identity).second) {
        return COMMON_BEE_FIT_REPEATED_CANDIDATE;
    }
    for (size_t i = 0; i < margins.size(); ++i) {
        if (shortfalls[i] <= 0) {
            continue;
        }
        const uint64_t shortfall = uint64_t(shortfalls[i]);
        if (shortfall > std::numeric_limits<size_t>::max() - margins[i]) {
            return COMMON_BEE_FIT_INVALID_SHORTFALL;
        }
        margins[i] += size_t(shortfall);
    }
    return COMMON_BEE_FIT_RETRY;
}
