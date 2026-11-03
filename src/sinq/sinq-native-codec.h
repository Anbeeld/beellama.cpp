#pragma once
#include "ggml.h"
#include <vector>
#include <thread>
namespace sinq {
// Canonical producer encoder. Rows are [output,input]; imatrix is per expert.
size_t native_encode(ggml_type, const float *, void *, int64_t chunk_size,
    int64_t first_row, int64_t nrows, int64_t rows_per_expert, int64_t columns,
    const float * importance, std::vector<std::thread> &, int threads);
}
