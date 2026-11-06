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
size_t sinq::native_encode(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t first_row, int64_t nrows, int64_t nrows_per_expert, int64_t n_per_row, const float * imatrix, std::vector<std::thread> & workers, const int nthread) {
    const size_t row_size = ggml_row_size(new_type, n_per_row);

    auto imatrix_for_row = [=](int64_t row_global) {
        return imatrix ? imatrix + (row_global / nrows_per_expert) * n_per_row : nullptr;
    };

    if (nthread < 2) {
        // single-thread
        size_t new_size = 0;
        for (int64_t row = 0; row < nrows;) {
            const int64_t row_global = first_row + row;
            const int64_t this_nrow  = std::min(nrows - row, nrows_per_expert - row_global % nrows_per_expert);
            void * this_data = (char *) new_data + row * row_size;
            size_t this_size = ggml_quantize_chunk(new_type, f32_data + row * n_per_row, this_data, 0, this_nrow, n_per_row, imatrix_for_row(row_global));
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                throw std::runtime_error("quantized data validation failed");
            }
            new_size += this_size;
            row += this_nrow;
        }
        return new_size;
    }

    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    bool valid = true;
    auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, chunk_size,
            first_row, nrows, nrows_per_expert, n_per_row, row_size, imatrix_for_row]() {
        const int64_t nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            if (counter >= nrows) {
                if (local_size > 0) {
                    new_size += local_size;
                }
                break;
            }
            const int64_t row        = counter;
            const int64_t row_global = first_row + row;
            // stop at the expert boundary
            const int64_t this_nrow  = std::min(std::min(nrows - row, nrows_per_chunk), nrows_per_expert - row_global % nrows_per_expert);
            counter += this_nrow;
            lock.unlock();

            void * this_data = (char *) new_data + row * row_size;
            size_t this_size = ggml_quantize_chunk(new_type, f32_data + row * n_per_row, this_data, 0, this_nrow, n_per_row, imatrix_for_row(row_global));
            local_size += this_size;

            // validate the quantized data
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                std::unique_lock<std::mutex> lock(mutex);
                valid = false;
                break;
            }
        }
    };
    for (int it = 0; it < nthread - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) { w.join(); }
    workers.clear();
    if (!valid) {
        throw std::runtime_error("quantized data validation failed");
    }
    return new_size;
}
