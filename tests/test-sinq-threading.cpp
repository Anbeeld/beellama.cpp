#include "sinq/sinq-parallel.h"
#include "sinq/sinq-local-score.h"
#include "sinq/sinq-research-config.h"
#include <mutex>
#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <thread>

static void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main() try {
    std::vector<int> visits(10000, 0);
    std::mutex mutex;
    std::set<std::thread::id> workers;
    sinq::parallel_for(int64_t(visits.size()), 4, [&](int64_t begin, int64_t end) {
        {
            std::lock_guard<std::mutex> guard(mutex);
            workers.insert(std::this_thread::get_id());
        }
        for (int64_t i = begin; i < end; ++i) ++visits[size_t(i)];
    });
    check(workers.size() > 1 && workers.size() <= 4, "parallel range used multiple bounded workers");
    for (int value : visits) check(value == 1, "parallel ranges cover every element exactly once");

    bool propagated = false;
    try {
        sinq::parallel_for(100, 4, [](int64_t begin, int64_t) {
            if (begin != 0) throw std::runtime_error("worker failure");
        });
    } catch (const std::runtime_error &) { propagated = true; }
    check(propagated, "worker exception propagated after join");

    const int64_t rows = 23, columns = 64;
    std::vector<float> weights(size_t(rows * columns)), decoded(size_t(rows * columns));
    std::vector<float> activation(5 * columns);
    std::vector<ggml_fp16_t> scales(static_cast<size_t>(columns));
    for (int64_t i = 0; i < rows * columns; ++i) {
        weights[size_t(i)] = std::sin(float(i) * .013f);
        decoded[size_t(i)] = std::cos(float(i) * .019f);
    }
    for (int64_t i = 0; i < 5 * columns; ++i) activation[size_t(i)] = std::sin(float(i) * .071f);
    for (auto & scale : scales) scale = ggml_fp32_to_fp16(1.0f);
    for (auto objective : {sinq::local_objective::output_l1, sinq::local_objective::output_mse}) {
        std::vector<double> one(static_cast<size_t>(rows));
        std::vector<double> many(static_cast<size_t>(rows));
        sinq::output_scores(weights.data(), decoded.data(), scales.data(), rows, columns,
            activation.data(), 5, objective, one.data(), 1);
        sinq::output_scores(weights.data(), decoded.data(), scales.data(), rows, columns,
            activation.data(), 5, objective, many.data(), 4);
        check(one == many, "parallel per-row scoring preserves exact values");
    }

    const auto default_threads = std::max(1u, std::thread::hardware_concurrency());
    check(sinq::search_options{}.threads == int(default_threads), "search defaults to all logical CPU threads");
    check(sinq::bridge_request{}.threads == int(default_threads), "bridge defaults to all logical CPU threads");
    std::puts("test-sinq-threading: PASS");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "test-sinq-threading: %s\n", error.what());
    return 1;
}
