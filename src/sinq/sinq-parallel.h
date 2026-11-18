#pragma once
#include "sinq-numeric.h"
#include <algorithm>
#include <exception>
#include <cstdint>
#include <thread>
#include <vector>

namespace sinq {
// Independent ranges only: callers reduce results in original index order.
// Join on every exit, and propagate worker exceptions on the calling thread.
template <typename Function>
void parallel_for(int64_t count, int threads, const Function & function) {
    require(count >= 0 && threads > 0, "SINQ invalid parallel range/thread count");
    if (count == 0) return;
    const int use = int(std::min<int64_t>(count, threads));
    if (use == 1) { function(0, count); return; }
    std::vector<std::exception_ptr> failures(size_t(use), nullptr);
    const auto work = [&](int worker) {
        try {
            const int64_t chunk = count / use, extra = count % use;
            const int64_t begin = chunk * worker + std::min<int64_t>(worker, extra);
            function(begin, begin + chunk + (worker < extra));
        } catch (...) { failures[size_t(worker)] = std::current_exception(); }
    };
    std::vector<std::thread> pool;
    try {
        for (int worker = 1; worker < use; ++worker) pool.emplace_back(work, worker);
        work(0);
    } catch (...) {
        for (auto & worker : pool) worker.join();
        throw;
    }
    for (auto & worker : pool) worker.join();
    for (const auto & failure : failures) if (failure) std::rethrow_exception(failure);
}
}
