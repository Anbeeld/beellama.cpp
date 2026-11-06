#pragma once
#include "ggml.h"
#include <vector>
#include <thread>
#include <functional>
namespace sinq {
// The serialized F16 diagonal is authoritative in both direct and offline encoding.
void precondition_weights(const float *, float *, int64_t rows, int64_t columns, const ggml_fp16_t *, int threads);
void transform_importance(const float *, float *, int64_t columns, const ggml_fp16_t *);
// Optional fault-injection boundary; normal producers use the native defaults.
struct native_encode_hooks {
    decltype(&ggml_quantize_chunk) quantize = ggml_quantize_chunk;
    std::function<std::thread(std::function<void()>)> launch =
        [](std::function<void()> work) { return std::thread(std::move(work)); };
};
// Canonical producer encoder. Rows are [output,input]; imatrix is per expert.
size_t native_encode(ggml_type, const float *, void *, int64_t chunk_size,
    int64_t first_row, int64_t nrows, int64_t rows_per_expert, int64_t columns,
    const float * importance, std::vector<std::thread> &, int threads,
    const native_encode_hooks * hooks = nullptr);
}
