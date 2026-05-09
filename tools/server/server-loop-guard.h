#pragma once

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

enum server_loop_guard_region {
    SERVER_LOOP_REGION_REASONING,
    SERVER_LOOP_REGION_VISIBLE,
};

struct server_loop_guard_result {
    bool triggered = false;
    std::string kind;
    int32_t period = 0;
    int32_t coverage = 0;
    float score = 0.0f;
};

class server_loop_guard {
public:
    explicit server_loop_guard(common_reasoning_loop_guard_params params = {});

    void reset();
    void accept(llama_token token, server_loop_guard_region region);
    bool should_check(server_loop_guard_region region, bool token_is_eog, bool forcing_reasoning_end) const;
    server_loop_guard_result check(server_loop_guard_region region) const;
    int32_t seen(server_loop_guard_region region) const;

private:
    common_reasoning_loop_guard_params params;
    std::vector<llama_token> reasoning_tail;
    std::vector<llama_token> visible_tail;
    int32_t reasoning_seen = 0;
    int32_t visible_seen = 0;

    std::vector<llama_token> & tail(server_loop_guard_region region);
    const std::vector<llama_token> & tail(server_loop_guard_region region) const;

    server_loop_guard_result check_periodic_tail(const std::vector<llama_token> & tokens) const;
    server_loop_guard_result check_ngram_dominance(const std::vector<llama_token> & tokens) const;
    server_loop_guard_result check_low_entropy(const std::vector<llama_token> & tokens) const;
};
