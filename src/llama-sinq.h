#pragma once

#include "ggml.h"

#include <cstdint>
#include <vector>

// Native codecs audited for column-conditioned matmul and channel importance.
// This changes neither their block layouts nor their encoder algorithms.
inline bool llama_sinq_supported_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q6_K: case GGML_TYPE_Q8_0:
            return true;
        default: return false;
    }
}

// Offline statistics only. Weight storage and matmul remain ordinary GGML types.
struct llama_sinq_options {
    int strip_width = 64;
    int iterations = 16;
    bool gate = true;
    bool identity = false;
    float strength = 1.f;
};

struct llama_sinq_tile_statistics {
    int chosen_iteration = 0;
    float initial_imbalance = 0;
    float best_imbalance = 0;
    float terminal_imbalance = 0;
};

struct llama_sinq_statistics {
    double kurtosis = 0;
    bool identity = false;
    bool gated = false;
    bool degenerate = false;
    float min_scale = 1;
    float max_scale = 1;
    float median_scale = 1;
    size_t selected_identity_count = 0;
    std::vector<llama_sinq_tile_statistics> strips;
};

// Input to this primitive is already FP16-rounded FP32, row-major [N,G].
// Returns the best-iterate column and row factors, before any Q_K encoding.
bool llama_sinq_balance_tile(const float * weights, int64_t rows, int width,
                            int iterations, float * columns, float * row_factors,
                            llama_sinq_tile_statistics * statistics = nullptr);

// Full-height strips, bounded per-worker workspace. F16-round t BEFORE W/t:
// serialized t and runtime compensation must be the identical diagonal matrix.
bool llama_sinq_compute_scales(const void * source, ggml_type source_type,
                              int64_t rows, int64_t columns,
                              const llama_sinq_options & options, int threads,
                              std::vector<ggml_fp16_t> & scales,
                              llama_sinq_statistics & statistics);
