#include "sinq-native-codec.h"
#include "sinq-numeric.h"
#include "sinq-parallel.h"
#include <cmath>
#include <algorithm>
#include <limits>

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
        int64_t columns, const ggml_fp16_t * scales, double exponent) {
    require(importance && output && scales && columns > 0, "invalid importance buffers/shape");
    require(std::isfinite(exponent), "invalid importance exponent");
    for (int64_t j = 0; j < columns; ++j) {
        const float u = ggml_fp16_to_fp32(scales[j]);
        require(std::isfinite(u) && u > 0 && std::isfinite(importance[j]) && importance[j] >= 0, "invalid importance/compensation");
        // Keep the historical FP32 multiplication order exactly at p=2.
        output[j] = exponent == 2.0 ? importance[j]*u*u :
            float(double(importance[j])*std::pow(double(u), exponent));
        require(std::isfinite(output[j]) && output[j] >= 0, "transformed importance overflow");
    }
}

sinq::scale_refinement::scale_refinement(int64_t columns) : columns_(columns) {
    require(columns > 0 && uint64_t(columns) <= std::numeric_limits<size_t>::max()/sizeof(double) &&
        uint64_t(columns) <= numerators_.max_size() &&
        uint64_t(columns) <= denominators_.max_size(), "invalid refinement columns/allocation size");
    numerators_.assign(size_t(columns), 0.0);
    denominators_.assign(size_t(columns), 0.0);
}

void sinq::scale_refinement::accumulate(const float * original_weights,
        const void * native_payload, ggml_type codec, int64_t rows) {
    require(original_weights && native_payload && rows > 0, "invalid refinement buffers/rows");
    require(codec >= 0 && codec < GGML_TYPE_COUNT, "invalid refinement codec");
    const auto * traits = ggml_get_type_traits(codec);
    require(traits && traits->to_float && traits->blck_size > 0 &&
        columns_%traits->blck_size == 0, "invalid refinement codec/columns");
    const size_t max_size = std::numeric_limits<size_t>::max();
    require(uint64_t(rows) <= max_size/sizeof(float)/uint64_t(columns_) &&
        uint64_t(rows) <= uint64_t(std::numeric_limits<int64_t>::max())/uint64_t(columns_),
        "refinement weight shape overflow");
    require(traits->type_size > 0 && uint64_t(columns_) <= max_size/traits->type_size,
        "refinement row size overflow");
    const size_t row_bytes = ggml_row_size(codec, columns_);
    require(row_bytes > 0 && uint64_t(rows) <= max_size/row_bytes,
        "refinement payload shape overflow");
    std::vector<float> decoded(size_t(columns_), 0.f);
    for (int64_t r = 0; r < rows; ++r) {
        traits->to_float(static_cast<const char *>(native_payload) + size_t(r)*row_bytes,
            decoded.data(), columns_);
        for (int64_t j = 0; j < columns_; ++j) {
            const float w = original_weights[r*columns_ + j], b = decoded[j];
            require(std::isfinite(w) && std::isfinite(b), "nonfinite refinement weight/decoded value");
            numerators_[j] += double(w)*double(b);
            denominators_[j] += double(b)*double(b);
        }
    }
}

sinq::refinement_statistics sinq::scale_refinement::apply(ggml_fp16_t * scales, double lambda) const {
    require(scales && std::isfinite(lambda) && lambda >= 0 && lambda <= 1,
        "invalid refinement scales/lambda");
    refinement_statistics stats;
    for (int64_t j = 0; j < columns_; ++j) {
        const float u0 = ggml_fp16_to_fp32(scales[j]);
        require(std::isfinite(u0) && u0 > 0, "invalid serialized compensation");
    }
    if (lambda == 0) { return stats; }
    for (int64_t j = 0; j < columns_; ++j) {
        const float u0 = ggml_fp16_to_fp32(scales[j]);
        if (!std::isfinite(numerators_[j]) || !std::isfinite(denominators_[j]) || denominators_[j] <= 0) {
            ++stats.invalid_ls_count;
            continue;
        }
        const double ls = numerators_[j]/denominators_[j];
        // Test F16 representability without rounding the value used in blending.
        const auto representable = [](double value) {
            if (!std::isfinite(value) || value <= 0 || value > 65504.0) { return false; }
            const float serialized = ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(value)));
            return std::isfinite(serialized) && serialized > 0;
        };
        if (!representable(ls)) {
            ++stats.invalid_ls_count;
            continue;
        }
        const double blended = (1.0 - lambda)*double(u0) + lambda*ls;
        if (!representable(blended)) {
            ++stats.invalid_blend_count;
            continue;
        }
        scales[j] = ggml_fp32_to_fp16(float(blended));
    }
    return stats;
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
