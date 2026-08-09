#pragma once

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

enum common_bee_fit_retry_action {
    COMMON_BEE_FIT_ACCEPT,
    COMMON_BEE_FIT_RETRY,
    COMMON_BEE_FIT_REPEATED_CANDIDATE,
    COMMON_BEE_FIT_INVALID_SHORTFALL,
};
// State for the Bee-only exact validation loop. Candidate identities are
// retained until a fitting candidate is accepted so a non-improving upstream
// result terminates deterministically rather than relying on a retry limit.
class common_bee_fit_retry_state {
public:
    common_bee_fit_retry_action observe(
            const std::string & candidate_identity,
            const std::vector<int64_t> & shortfalls,
            std::vector<size_t> & margins);

private:
    std::set<std::string> rejected_candidates;
};
