#pragma once
#include "ggml.h"
#include <vector>
#include <thread>
#include <functional>
namespace sinq {
// The serialized F16 diagonal is authoritative in both direct and offline encoding.
void precondition_weights(const float *, float *, int64_t rows, int64_t columns, const ggml_fp16_t *, int threads);
void transform_importance(const float *, float *, int64_t columns, const ggml_fp16_t *, double exponent = 2.0);
struct refinement_statistics {
    size_t invalid_ls_count = 0;
    size_t invalid_blend_count = 0;
};
// Accumulate slabs in global row order; the encoded payload remains fixed.
class scale_refinement {
public:
    explicit scale_refinement(int64_t columns);
    void accumulate(const float * original_weights, const void * native_payload, ggml_type codec, int64_t rows);
    refinement_statistics apply(ggml_fp16_t * scales, double lambda) const;
private:
    int64_t columns_;
    std::vector<double> numerators_, denominators_;
};
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
