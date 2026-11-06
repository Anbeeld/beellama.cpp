#include "sinq-native-codec.h"
#include "sinq-numeric.h"
#include "sinq-parallel.h"
#include <cmath>
#include <algorithm>

void sinq::precondition_weights(const float * weights, float * output, int64_t rows,
        int64_t columns, const ggml_fp16_t * scales, int threads) {
    require(weights && output && scales && rows > 0 && columns > 0 && threads > 0, "invalid preconditioning buffers/shape");
    for (int64_t j = 0; j < columns; ++j) {
        const float u = ggml_fp16_to_fp32(scales[j]);
        require(std::isfinite(u) && u > 0, "invalid serialized compensation");
    }
    parallel_for(rows, threads, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin*columns; i < end*columns; ++i) {
            output[i] = weights[i]/ggml_fp16_to_fp32(scales[i%columns]);
        }
    });
}

void sinq::transform_importance(const float * importance, float * output,
        int64_t columns, const ggml_fp16_t * scales) {
    require(importance && output && scales && columns > 0, "invalid importance buffers/shape");
    for (int64_t j = 0; j < columns; ++j) {
        const float u = ggml_fp16_to_fp32(scales[j]);
        require(std::isfinite(u) && u > 0 && std::isfinite(importance[j]) && importance[j] >= 0, "invalid importance/compensation");
        output[j] = importance[j]*u*u;
        require(std::isfinite(output[j]) && output[j] >= 0, "transformed importance overflow");
    }
}

#include <mutex>
#include <stdexcept>
#include <string>
size_t sinq::native_encode(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t first_row, int64_t nrows, int64_t nrows_per_expert, int64_t n_per_row, const float * imatrix, std::vector<std::thread> & workers, const int nthread, const native_encode_hooks * hooks) {
    require(f32_data && new_data && nrows > 0 && nrows_per_expert > 0 &&
        n_per_row > 0 && first_row >= 0 && chunk_size >= n_per_row && nthread > 0 &&
        workers.empty(), "invalid native encoder buffers/shape/workers");
    const auto quantize = hooks ? hooks->quantize : ggml_quantize_chunk;
    const size_t row_size = ggml_row_size(new_type, n_per_row);

    auto imatrix_for_row = [=](int64_t row_global) {
        return imatrix ? imatrix + (row_global / nrows_per_expert) * n_per_row : nullptr;
    };

    const auto encode_rows = [&](int64_t row, int64_t rows) {
        const auto context = [&]() {
            return std::string("SINQ native ") + ggml_type_name(new_type) + " rows [" +
                std::to_string(first_row + row) + "," + std::to_string(first_row + row + rows) + ")";
        };
        try {
            void * data = (char *) new_data + row * row_size;
            const auto bytes = quantize(new_type, f32_data + row * n_per_row,
                data, 0, rows, n_per_row, imatrix_for_row(first_row + row));
            if (!ggml_validate_row_data(new_type, data, bytes)) {
                throw std::runtime_error("quantized data validation failed");
            }
            return bytes;
        } catch (const std::exception & error) {
            std::throw_with_nested(std::runtime_error(context() + ": " + error.what()));
        } catch (...) {
            std::throw_with_nested(std::runtime_error(context() + ": unknown codec failure"));
        }
    };

    if (nthread < 2) {
        // single-thread
        size_t new_size = 0;
        for (int64_t row = 0; row < nrows;) {
            const int64_t row_global = first_row + row;
            const int64_t this_nrow  = std::min(nrows - row, nrows_per_expert - row_global % nrows_per_expert);
            new_size += encode_rows(row, this_nrow);
            row += this_nrow;
        }
        return new_size;
    }

    // Count actual tasks, including partial chunks at expert boundaries.
    const int64_t rows_per_chunk = std::max<int64_t>(1, chunk_size / n_per_row);
    int64_t tasks = 0;
    for (int64_t row = 0; row < nrows;) {
        row += std::min({nrows - row, rows_per_chunk,
            nrows_per_expert - (first_row + row) % nrows_per_expert});
        ++tasks;
    }
    const int use = int(std::min<int64_t>(nthread, tasks));
    // Allocate before launching: allocation failure must not destroy a temporary
    // joinable thread during insertion into the caller-owned vector.
    workers.reserve(size_t(std::max(0, use - 1)));
    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    std::exception_ptr failure;
    bool cancelled = false;
    auto compute = [&]() {
        try {
            while (true) {
                int64_t row, this_nrow;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (cancelled || counter >= nrows) return;
                    row = counter;
                    this_nrow = std::min({nrows - row, rows_per_chunk,
                        nrows_per_expert - (first_row + row) % nrows_per_expert});
                    counter += this_nrow;
                }
                const size_t bytes = encode_rows(row, this_nrow);
                std::lock_guard<std::mutex> lock(mutex);
                new_size += bytes;
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!failure) failure = std::current_exception();
            cancelled = true;
        }
    };
    const auto join = [&]() {
        for (auto & worker : workers) if (worker.joinable()) worker.join();
        workers.clear();
    };
    try {
        for (int it = 1; it < use; ++it) {
            workers.push_back(hooks ? hooks->launch(compute) : std::thread(compute));
        }
        compute();
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            cancelled = true;
        }
        join();
        throw;
    }
    join();
    if (failure) std::rethrow_exception(failure);
    return new_size;
}
