#pragma once
#include "sinq-types.h"
namespace sinq {
double output_score(const float *, const float *, const ggml_fp16_t *, int64_t, int64_t, const float *, int, local_objective);
double row_loss(const float *, const float *, const ggml_fp16_t *, int64_t, const float *, int, std::vector<double> &);
double output_l1(const float *, const float *, const ggml_fp16_t *, int64_t, int64_t, const float *, int);
}
