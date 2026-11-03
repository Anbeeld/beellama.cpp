#pragma once

#include "llama-sinq.h"
#include <functional>

// Readers expose original row-major [output,input] weights in bounded slabs.
// The encoder callback is the producer's ordinary native quantization path.
using llama_sinq_weight_reader = std::function<void(int64_t, int64_t, float *)>;
using llama_sinq_native_encoder = std::function<size_t(const float *, void *, int64_t, int64_t, const float *)>;

struct llama_sinq_calibration_result {
    std::vector<ggml_fp16_t> scales;
    double baseline_loss = 0;
    double selected_loss = 0;
    int alpha_index = -1; // -1 denotes the separately scored uncalibrated candidate.
    size_t workspace_bytes = 0;
    std::vector<double> candidate_losses;
};

std::vector<double> llama_sinq_awq_factors(const std::vector<double> & activation_mean,
                                         const std::vector<double> & weight_mean, int alpha_index);

double llama_sinq_output_l1(const float * original, const float * decoded, const ggml_fp16_t * scale,
                           int64_t rows, int64_t columns, const float * activations, int samples);

llama_sinq_calibration_result llama_sinq_calibrate_native(
    ggml_type codec, int64_t rows, int64_t columns,
    const std::vector<float> & raw_sinq, const std::vector<ggml_fp16_t> & ordinary,
    const std::vector<float> & activations, int samples, const float * importance,
    size_t workspace_limit, const llama_sinq_weight_reader & read,
    const llama_sinq_native_encoder & encode, int nthread = 1);
