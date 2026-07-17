#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace kld_logits {

inline constexpr char magic_v1[8] = { '_', 'l', 'o', 'g', 'i', 't', 's', '_' };
inline constexpr char magic_v2[8] = { '_', 'l', 'o', 'g', 'i', 't', 's', '2' };

static inline size_t encoded_row_size(size_t n_vocab) {
    return 2 * ((n_vocab + 1) / 2) + 4;
}

static inline double encode_row(
        int n_vocab, const float * logits, uint16_t * encoded, int token) {
    float max_logit = -INFINITY;
    float min_logit = INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        if (std::isfinite(logits[i])) {
            max_logit = std::max(max_logit, logits[i]);
            min_logit = std::min(min_logit, logits[i]);
        }
    }
    if (!std::isfinite(max_logit)) {
        const float scale = 0.0f;
        const float min_log_probability = -INFINITY;
        std::memcpy(encoded + 0, &scale, sizeof(scale));
        std::memcpy(encoded + 2, &min_log_probability, sizeof(min_log_probability));
        std::memset(encoded + 4, 0, size_t(n_vocab) * sizeof(uint16_t));
        return INFINITY;
    }

    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        if (std::isfinite(logits[i])) {
            sum_exp += std::exp(double(logits[i] - max_logit));
        }
    }
    const float log_sum_exp = float(std::log(sum_exp));
    const float min_log_probability = min_logit - max_logit - log_sum_exp;
    // Code zero is reserved for an exact zero probability (-infinity logit).
    const float scale = (max_logit - min_logit) / 65534.0f;
    std::memcpy(encoded + 0, &scale, sizeof(scale));
    std::memcpy(encoded + 2, &min_log_probability, sizeof(min_log_probability));

    uint16_t * values = encoded + 4;
    if (scale > 0.0f) {
        const float inv_scale = 1.0f / scale;
        for (int i = 0; i < n_vocab; ++i) {
            if (!std::isfinite(logits[i])) {
                values[i] = 0;
                continue;
            }
            const long q = 1 + std::lround((logits[i] - min_logit) * inv_scale);
            values[i] = uint16_t(std::max(1L, std::min(65535L, q)));
        }
    } else {
        for (int i = 0; i < n_vocab; ++i) {
            values[i] = std::isfinite(logits[i]) ? 1 : 0;
        }
    }

    return max_logit + log_sum_exp - logits[token];
}

struct decoded_row {
    const uint16_t * values;
    float scale;
    float min_log_probability;
    float log_normalizer;
    bool zero_is_negative_infinity;
};

static inline decoded_row decode_row(
        int n_vocab, const uint16_t * encoded, bool normalize) {
    decoded_row row = { encoded + 4, 0.0f, 0.0f, 0.0f, normalize };
    std::memcpy(&row.scale, encoded + 0, sizeof(row.scale));
    std::memcpy(&row.min_log_probability, encoded + 2, sizeof(row.min_log_probability));
    if (!normalize) {
        return row;
    }

    double sum = 0.0;
    float max_log_probability = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        if (row.values[i] == 0) {
            continue;
        }
        max_log_probability = std::max(
                max_log_probability,
                row.scale * (row.values[i] - 1) + row.min_log_probability);
    }
    if (!std::isfinite(max_log_probability)) {
        return row;
    }
    for (int i = 0; i < n_vocab; ++i) {
        if (row.values[i] == 0) {
            continue;
        }
        const float log_probability =
                row.scale * (row.values[i] - 1) + row.min_log_probability;
        sum += std::exp(double(log_probability - max_log_probability));
    }
    row.log_normalizer = max_log_probability + float(std::log(sum));
    return row;
}

static inline float log_probability(const decoded_row & row, size_t token) {
    if (row.zero_is_negative_infinity && row.values[token] == 0) {
        return -INFINITY;
    }
    const uint16_t q = row.zero_is_negative_infinity ?
        uint16_t(row.values[token] - 1) : row.values[token];
    return row.scale * q +
        row.min_log_probability - row.log_normalizer;
}

} // namespace kld_logits
