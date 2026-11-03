#pragma once
#include "sinq-types.h"
namespace sinq {
// raw contains strength-powered, unrounded Sinkhorn column factors.
std::vector<ggml_fp16_t> compensation(const std::vector<float> & raw,
    const std::vector<double> & activation_mean, const std::vector<double> & weight_mean,
    const candidate_spec & spec);
std::vector<double> legacy_awq_factors(const std::vector<double> &, const std::vector<double> &, int);
}
