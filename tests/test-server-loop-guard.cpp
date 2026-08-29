#include "../tools/server/server-loop-guard.h"

#undef NDEBUG
#include <cassert>
#include <random>
#include <vector>

static common_reasoning_loop_guard_params test_params() {
    common_reasoning_loop_guard_params params;
    params.min_reasoning_tokens = 1024;
    params.window_tokens = 2048;
    params.max_period = 512;
    params.min_repeated_coverage = 768;
    params.check_interval = 32;
    return params;
}

static void accept_many(server_loop_guard & guard, const std::vector<llama_token> & tokens, server_loop_guard_region region) {
    for (llama_token token : tokens) {
        guard.accept(token, region);
    }
}

static std::vector<llama_token> repeat_block(const std::vector<llama_token> & block, int repeats) {
    std::vector<llama_token> tokens;
    tokens.reserve(block.size() * (size_t) repeats);
    for (int i = 0; i < repeats; ++i) {
        tokens.insert(tokens.end(), block.begin(), block.end());
    }
    return tokens;
}

static void assert_triggered(const server_loop_guard_result & result, const char * kind) {
    assert(result.triggered);
    assert(result.kind == kind);
    assert(result.coverage >= 768);
}

static void assert_not_triggered(const server_loop_guard_result & result) {
    assert(!result.triggered);
}

int main() {
    {
        server_loop_guard guard(test_params());
        accept_many(guard, std::vector<llama_token>(1200, 42), SERVER_LOOP_REGION_REASONING);
        const auto result = guard.check(SERVER_LOOP_REGION_REASONING);
        assert_triggered(result, "periodic_tail");
        assert(result.period == 1);
    }

    {
        server_loop_guard guard(test_params());
        accept_many(guard, repeat_block({1, 2}, 600), SERVER_LOOP_REGION_REASONING);
        const auto result = guard.check(SERVER_LOOP_REGION_REASONING);
        assert_triggered(result, "periodic_tail");
        assert(result.period == 2);
    }

    {
        server_loop_guard guard(test_params());
        std::vector<llama_token> block;
        for (int i = 0; i < 200; ++i) {
            block.push_back(1000 + i);
        }
        accept_many(guard, repeat_block(block, 6), SERVER_LOOP_REGION_REASONING);
        const auto result = guard.check(SERVER_LOOP_REGION_REASONING);
        assert_triggered(result, "periodic_tail");
        assert(result.period == 200);
    }

    {
        server_loop_guard guard(test_params());
        std::vector<llama_token> tokens;
        tokens.reserve(1200);
        std::mt19937 rng(1234);
        std::uniform_int_distribution<int> dist(0, 5);
        for (int i = 0; i < 1200; ++i) {
            tokens.push_back(200 + dist(rng));
        }
        accept_many(guard, tokens, SERVER_LOOP_REGION_REASONING);
        assert_triggered(guard.check(SERVER_LOOP_REGION_REASONING), "low_entropy");
    }

    {
        server_loop_guard guard(test_params());
        for (int i = 0; i < 20000; ++i) {
            const llama_token token = 1000 + (llama_token) ((i * 48271LL + (i / 37) * 7919LL + (i % 11) * 104729LL) % 50000);
            guard.accept(token, SERVER_LOOP_REGION_REASONING);
            if (guard.should_check(SERVER_LOOP_REGION_REASONING, false, false)) {
                assert_not_triggered(guard.check(SERVER_LOOP_REGION_REASONING));
            }
        }
    }

    {
        server_loop_guard guard(test_params());
        for (int row = 0; row < 180; ++row) {
            const std::vector<llama_token> table_row = {
                500, 600 + row, 501, 700 + (row * 7) % 997, 501, 800 + (row * 13) % 997, 500, 502
            };
            accept_many(guard, table_row, SERVER_LOOP_REGION_VISIBLE);
        }
        assert_not_triggered(guard.check(SERVER_LOOP_REGION_VISIBLE));
    }

    {
        common_reasoning_loop_guard_params params = test_params();
        params.min_reasoning_tokens = 32;
        params.min_repeated_coverage = 24;
        params.window_tokens = 64;
        params.max_period = 8;
        params.check_interval = 8;
        server_loop_guard guard(params);

        accept_many(guard, std::vector<llama_token>(32, 42), SERVER_LOOP_REGION_REASONING);
        const auto reasoning = guard.check(SERVER_LOOP_REGION_REASONING);
        assert(reasoning.triggered);
        assert(reasoning.kind == "periodic_tail");
        assert(reasoning.period == 1);

        // A reasoning intervention must not disarm the independent visible tail.
        accept_many(guard, repeat_block({7, 8}, 16), SERVER_LOOP_REGION_VISIBLE);
        const auto visible = guard.check(SERVER_LOOP_REGION_VISIBLE);
        assert(visible.triggered);
        assert(visible.kind == "periodic_tail");
        assert(visible.period == 2);
    }

    {
        common_reasoning_loop_guard_params params = test_params();
        params.min_repeated_coverage = 24;
        params.window_tokens = 64;
        params.max_period = 8;
        params.check_interval = 8;
        server_loop_guard guard(params);

        accept_many(guard, std::vector<llama_token>(16, 9), SERVER_LOOP_REGION_VISIBLE);
        assert_not_triggered(guard.check(SERVER_LOOP_REGION_VISIBLE));
    }

    {
        common_reasoning_loop_guard_params params = test_params();
        params.min_reasoning_tokens = 0;
        params.min_repeated_coverage = 1;
        params.window_tokens = 64;
        params.max_period = 8;
        params.check_interval = 32;
        server_loop_guard guard(params);
        for (int i = 0; i < 31; ++i) {
            guard.accept(1, SERVER_LOOP_REGION_REASONING);
            assert(!guard.should_check(SERVER_LOOP_REGION_REASONING, false, false));
        }
        guard.accept(1, SERVER_LOOP_REGION_REASONING);
        assert(guard.should_check(SERVER_LOOP_REGION_REASONING, false, false));
        assert(!guard.should_check(SERVER_LOOP_REGION_REASONING, true, false));
        assert(!guard.should_check(SERVER_LOOP_REGION_REASONING, false, true));
    }

    {
        server_loop_guard guard(test_params());
        accept_many(guard, std::vector<llama_token>(1200, 42), SERVER_LOOP_REGION_REASONING);
        assert(guard.check(SERVER_LOOP_REGION_REASONING).triggered);
        guard.reset();
        for (int i = 0; i < 2000; ++i) {
            guard.accept((llama_token) (1000 + (i * 37) % 50000), SERVER_LOOP_REGION_REASONING);
        }
        assert_not_triggered(guard.check(SERVER_LOOP_REGION_REASONING));
    }

    {
        server_loop_guard guard(test_params());
        accept_many(guard, std::vector<llama_token>(1200, 42), SERVER_LOOP_REGION_REASONING);
        assert(guard.check(SERVER_LOOP_REGION_REASONING).triggered);

        common_reasoning_loop_guard_params params = test_params();
        params.mode = COMMON_REASONING_LOOP_GUARD_OFF;
        guard.configure(params);
        accept_many(guard, std::vector<llama_token>(1200, 42), SERVER_LOOP_REGION_REASONING);
        assert(!guard.should_check(SERVER_LOOP_REGION_REASONING, false, false));
        assert_not_triggered(guard.check(SERVER_LOOP_REGION_REASONING));
        assert(guard.seen(SERVER_LOOP_REGION_REASONING) == 1200);
    }

    return 0;
}
