#pragma once
#include "ggml.h"
#include <vector>
#include <thread>
namespace sinq {
// The serialized F16 diagonal is authoritative in both direct and offline encoding.
void precondition_weights(const float *, float *, int64_t rows, int64_t columns, const ggml_fp16_t *, int threads);
void transform_importance(const float *, float *, int64_t columns, const ggml_fp16_t *);
// Canonical producer encoder. Rows are [output,input]; imatrix is per expert.
size_t native_encode(ggml_type, const float *, void *, int64_t chunk_size,
    int64_t first_row, int64_t nrows, int64_t rows_per_expert, int64_t columns,
    const float * importance, std::vector<std::thread> &, int threads);
}
