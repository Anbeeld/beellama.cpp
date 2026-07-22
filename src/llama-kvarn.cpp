#include "llama-kvarn.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#define LLAMA_KVARN_DESC(KB, VB) { LLAMA_KVARN_K##KB##V##VB##_G128, "kvarn_k" #KB "v" #VB "_g128", KB, VB, 128 }

static constexpr std::array<llama_kvarn_type_desc, LLAMA_KVARN_TYPE_COUNT> KVAR_N_TYPES = {{
    { LLAMA_KVARN_TYPE_DISABLED, "off", 0, 0, 128 },

    LLAMA_KVARN_DESC(2, 2),
    LLAMA_KVARN_DESC(2, 3),
    LLAMA_KVARN_DESC(2, 4),

    LLAMA_KVARN_DESC(3, 2),
    LLAMA_KVARN_DESC(3, 3),
    LLAMA_KVARN_DESC(3, 4),

    LLAMA_KVARN_DESC(4, 2),
    LLAMA_KVARN_DESC(4, 3),
    LLAMA_KVARN_DESC(4, 4),

    LLAMA_KVARN_DESC(2, 5),
    LLAMA_KVARN_DESC(2, 6),
    LLAMA_KVARN_DESC(2, 8),

    LLAMA_KVARN_DESC(3, 5),
    LLAMA_KVARN_DESC(3, 6),
    LLAMA_KVARN_DESC(3, 8),

    LLAMA_KVARN_DESC(4, 5),
    LLAMA_KVARN_DESC(4, 6),
    LLAMA_KVARN_DESC(4, 8),

    LLAMA_KVARN_DESC(5, 2),
    LLAMA_KVARN_DESC(5, 3),
    LLAMA_KVARN_DESC(5, 4),
    LLAMA_KVARN_DESC(5, 5),
    LLAMA_KVARN_DESC(5, 6),
    LLAMA_KVARN_DESC(5, 8),

    LLAMA_KVARN_DESC(6, 2),
    LLAMA_KVARN_DESC(6, 3),
    LLAMA_KVARN_DESC(6, 4),
    LLAMA_KVARN_DESC(6, 5),
    LLAMA_KVARN_DESC(6, 6),
    LLAMA_KVARN_DESC(6, 8),

    LLAMA_KVARN_DESC(8, 2),
    LLAMA_KVARN_DESC(8, 3),
    LLAMA_KVARN_DESC(8, 4),
    LLAMA_KVARN_DESC(8, 5),
    LLAMA_KVARN_DESC(8, 6),
    LLAMA_KVARN_DESC(8, 8),
}};

llama_kvarn_attention_plan llama_kvarn_plan_attention(
        bool native_attention,
        bool native_original_v,
        uint32_t native_rotated_max_query_tokens,
        uint32_t n_query_tokens) {
    if (!native_attention) {
        return { false, GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED };
    }

    // A backend that predates the extended capability still supports the
    // established one-row rotated decode contract.
    const uint32_t rotated_limit = std::max(1u, native_rotated_max_query_tokens);
    if (n_query_tokens <= rotated_limit) {
        return { true, GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED };
    }
    if (native_original_v) {
        return { true, GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V };
    }
    return { false, GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED };
}

enum ggml_flash_attn_ext_kvarn_domain llama_kvarn_attention_domain(
        bool native_attention,
        bool native_original_v,
        uint32_t native_rotated_max_query_tokens,
        uint32_t n_query_tokens) {
    return llama_kvarn_plan_attention(
        native_attention,
        native_original_v,
        native_rotated_max_query_tokens,
        n_query_tokens).domain;
}

#undef LLAMA_KVARN_DESC

static bool llama_kvarn_valid_bits(int bits) {
    return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8;
}

static bool llama_kvarn_valid_bit_pair(int key_bits, int value_bits) {
    for (const auto & desc : KVAR_N_TYPES) {
        if (desc.key_bits == key_bits && desc.value_bits == value_bits && desc.group == 128) {
            return desc.type != LLAMA_KVARN_TYPE_DISABLED && desc.type != LLAMA_KVARN_TYPE_INVALID;
        }
    }
    return false;
}

static size_t llama_kvarn_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

const char * llama_kvarn_type_name(llama_kvarn_type type) {
    const auto * desc = llama_kvarn_type_desc_from_type(type);
    return desc ? desc->name : "invalid";
}

llama_kvarn_type llama_kvarn_type_from_name(const char * name) {
    const auto * desc = llama_kvarn_type_desc_from_name(name);
    return desc ? desc->type : LLAMA_KVARN_TYPE_INVALID;
}

llama_kvarn_params llama_kvarn_default_params() {
    return {
        /*.type                =*/ LLAMA_KVARN_TYPE_DISABLED,
        /*.key_bits            =*/ 0,
        /*.value_bits          =*/ 0,
        /*.swa_key_bits        =*/ 0,
        /*.swa_value_bits      =*/ 0,
        /*.group               =*/ 128,
        /*.sinkhorn_iters      =*/ 16,
        /*.sink_tokens         =*/ 128,
        /*.fail_if_unsupported =*/ true,
    };
}

llama_kvarn_params llama_kvarn_params_for_type(llama_kvarn_type type) {
    llama_kvarn_params result = llama_kvarn_default_params();
    result.type = type;

    const auto * desc = llama_kvarn_type_desc_from_type(type);
    if (desc != nullptr) {
        result.key_bits   = desc->key_bits;
        result.value_bits = desc->value_bits;
        result.group      = desc->group;
    }

    return result;
}

size_t llama_kvarn_type_count() {
    return KVAR_N_TYPES.size();
}

const llama_kvarn_type_desc * llama_kvarn_type_desc_from_name(const char * name) {
    if (name == nullptr) {
        return nullptr;
    }

    for (const auto & desc : KVAR_N_TYPES) {
        if (std::strcmp(desc.name, name) == 0) {
            return &desc;
        }
    }

    return nullptr;
}

const llama_kvarn_type_desc * llama_kvarn_type_desc_from_type(llama_kvarn_type type) {
    for (const auto & desc : KVAR_N_TYPES) {
        if (desc.type == type) {
            return &desc;
        }
    }

    return nullptr;
}

const char * llama_kvarn_validate_runtime(
        const llama_kvarn_params & params,
        const llama_kvarn_runtime_requirements & requirements) {
    if (params.type == LLAMA_KVARN_TYPE_DISABLED) {
        return nullptr;
    }

    const auto * desc = llama_kvarn_type_desc_from_type(params.type);
    if (desc == nullptr || desc->type == LLAMA_KVARN_TYPE_DISABLED) {
        return "invalid KVarN cache type";
    }
    if (params.key_bits != desc->key_bits || params.value_bits != desc->value_bits || params.group != desc->group) {
        return "KVarN cache parameters do not match the selected preset";
    }
    if (!llama_kvarn_valid_bits(params.key_bits) || !llama_kvarn_valid_bits(params.value_bits)) {
        return "KVarN supports only 2-, 3-, 4-, 5-, 6-, and 8-bit cache payloads";
    }
    if ((params.swa_key_bits != 0 && !llama_kvarn_valid_bits(params.swa_key_bits)) ||
            (params.swa_value_bits != 0 && !llama_kvarn_valid_bits(params.swa_value_bits))) {
        return "KVarN SWA overrides support only 2-, 3-, 4-, 5-, 6-, and 8-bit cache payloads";
    }
    if ((params.swa_key_bits == 0) != (params.swa_value_bits == 0)) {
        return "KVarN SWA override must specify both K and V bits";
    }
    if (params.swa_key_bits != 0 && !llama_kvarn_valid_bit_pair(params.swa_key_bits, params.swa_value_bits)) {
        return "invalid KVarN SWA override bit combination";
    }
    if (params.group != 128) {
        return "KVarN currently requires a group size of 128 tokens";
    }
    if (params.sinkhorn_iters <= 0) {
        return "KVarN requires at least one Sinkhorn iteration";
    }
    if (params.sink_tokens != 128) {
        return "KVarN currently requires exactly 128 unquantized sink tokens";
    }
    if (!requirements.attention_supported) {
        return "KVarN is not supported by this attention/cache path";
    }
    if (!requirements.head_dims_supported) {
        return "KVarN requires 128-, 256-, or 512-dimensional key/value heads";
    }
    if (!requirements.backend_ops_supported) {
        return "KVarN requires a backend with KVarN store and materialization support";
    }
    return nullptr;
}

llama_kvarn_iswa_policy llama_kvarn_iswa_policy_for(
        bool enabled,
        bool has_swa,
        uint32_t n_seq_max,
        bool unified,
        bool fail_if_unsupported) {
    if (!enabled) {
        return LLAMA_KVARN_ISWA_DISABLED;
    }
    if (!has_swa || n_seq_max <= 1 || unified) {
        return LLAMA_KVARN_ISWA_ALL_LAYERS;
    }
    return fail_if_unsupported ?
        LLAMA_KVARN_ISWA_UNSUPPORTED : LLAMA_KVARN_ISWA_STANDARD_SWA_FALLBACK;
}

bool llama_kvarn_can_remove_range(llama_pos pos_max, llama_pos p0, llama_pos p1, uint32_t group) {
    assert(group > 0);

    if (pos_max < 0) {
        return true;
    }

    const llama_pos begin = std::max<llama_pos>(p0, 0);
    const llama_pos end = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;

    if (begin == 0 && end > pos_max) {
        return true;
    }

    if (end <= pos_max) {
        return false;
    }

    const llama_pos live_group = pos_max / group;
    const llama_pos earliest_exact = std::max<llama_pos>(0, live_group - 1) * group;
    return begin >= earliest_exact;
}

bool llama_kvarn_plan_remove_range(
        llama_pos pos_max,
        llama_pos p0,
        llama_pos p1,
        uint32_t group,
        bool stream_owned,
        llama_pos & planned_p0,
        llama_pos & planned_p1) {
    if (llama_kvarn_can_remove_range(pos_max, p0, p1, group)) {
        planned_p0 = p0;
        planned_p1 = p1;
        return true;
    }

    if (!stream_owned || p0 <= 0 || p1 >= 0) {
        return false;
    }

    planned_p0 = (p0 / llama_pos(group)) * llama_pos(group);
    planned_p1 = -1;
    return true;
}

llama_kvarn_tile_layout llama_kvarn_make_layout(int head_dim, int group, int key_bits, int value_bits) {
    assert(head_dim > 0);
    assert(group > 0);
    assert(llama_kvarn_valid_bits(key_bits));
    assert(llama_kvarn_valid_bits(value_bits));

    llama_kvarn_tile_layout layout = {};
    size_t off = 0;

    layout.k_payload_off = off;
    layout.k_payload_bytes = llama_kvarn_packed_bytes(head_dim * group, key_bits);
    off += layout.k_payload_bytes;

    layout.k_s_col_off = off;
    off += size_t(head_dim) * sizeof(uint16_t);

    layout.k_zp_off = off;
    off += size_t(head_dim) * sizeof(uint16_t);

    layout.k_s_row_off = off;
    off += size_t(group) * sizeof(uint16_t);

    layout.v_payload_off = off;
    layout.v_payload_bytes = llama_kvarn_packed_bytes(group * head_dim, value_bits);
    off += layout.v_payload_bytes;

    layout.v_s_col_off = off;
    off += size_t(head_dim) * sizeof(uint16_t);

    layout.v_s_row_off = off;
    off += size_t(group) * sizeof(uint16_t);

    layout.v_zp_off = off;
    off += size_t(group) * sizeof(uint16_t);

    layout.tile_bytes = llama_kvarn_align_up(off, 8);
    return layout;
}

int llama_kvarn_head_slices(int head_dim) {
    if (head_dim != 128 && head_dim != 256 && head_dim != 512) {
        return 0;
    }

    return head_dim / 128;
}

bool llama_kvarn_head_dim_supported(int head_dim) {
    return llama_kvarn_head_slices(head_dim) > 0;
}

size_t llama_kvarn_packed_bytes(int n_values, int bits) {
    assert(n_values >= 0);
    assert(llama_kvarn_valid_bits(bits));
    return (size_t(n_values) * size_t(bits) + 7) / 8;
}

void llama_kvarn_pack_bits(const uint8_t * values, int n_values, int bits, uint8_t * dst) {
    assert(values != nullptr);
    assert(n_values >= 0);
    assert(llama_kvarn_valid_bits(bits));
    assert(dst != nullptr);

    std::memset(dst, 0, llama_kvarn_packed_bytes(n_values, bits));

    const uint8_t mask = uint8_t((1u << bits) - 1u);
    for (int i = 0; i < n_values; ++i) {
        const uint8_t value = values[i] & mask;
        const size_t bit_offset = size_t(i) * size_t(bits);

        for (int bit = 0; bit < bits; ++bit) {
            const size_t dst_bit = bit_offset + size_t(bit);
            dst[dst_bit / 8] |= uint8_t(((value >> bit) & 1u) << (dst_bit % 8));
        }
    }
}

uint8_t llama_kvarn_unpack_bits_value(const uint8_t * src, int index, int bits) {
    assert(src != nullptr);
    assert(index >= 0);
    assert(llama_kvarn_valid_bits(bits));

    uint8_t value = 0;
    const size_t bit_offset = size_t(index) * size_t(bits);
    for (int bit = 0; bit < bits; ++bit) {
        const size_t src_bit = bit_offset + size_t(bit);
        value |= uint8_t(((src[src_bit / 8] >> (src_bit % 8)) & 1u) << bit);
    }

    return value;
}

void llama_kvarn_hadamard_128(float * values) {
    assert(values != nullptr);

    for (int stride = 1; stride < 128; stride *= 2) {
        for (int base = 0; base < 128; base += 2 * stride) {
            for (int i = 0; i < stride; ++i) {
                const float a = values[base + i];
                const float b = values[base + stride + i];
                values[base + i] = a + b;
                values[base + stride + i] = a - b;
            }
        }
    }

    constexpr float INV_SQRT_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; ++i) {
        values[i] *= INV_SQRT_128;
    }
}

static float llama_kvarn_sample_std(const float * values, int n, int stride) {
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const double value = values[i * stride];
        sum += value;
        sum_sq += value * value;
    }

    const double mean = sum / n;
    const double variance = std::max(0.0, (sum_sq - n * mean * mean) / (n - 1));
    return float(std::sqrt(variance));
}

static float llama_kvarn_imbalance(const std::vector<float> & tile) {
    float col_min = std::numeric_limits<float>::infinity();
    float col_max = 0.0f;
    float row_min = std::numeric_limits<float>::infinity();
    float row_max = 0.0f;

    for (int c = 0; c < 128; ++c) {
        const float value = llama_kvarn_sample_std(tile.data() + c, 128, 128);
        col_min = std::min(col_min, value);
        col_max = std::max(col_max, value);
    }
    for (int r = 0; r < 128; ++r) {
        const float value = llama_kvarn_sample_std(tile.data() + r * 128, 128, 1);
        row_min = std::min(row_min, value);
        row_max = std::max(row_max, value);
    }

    return col_max / std::max(col_min, 1e-8f) + row_max / std::max(row_min, 1e-8f);
}

static void llama_kvarn_variance_normalize(
        const float * tile,
        int sinkhorn_iters,
        std::vector<float> & balanced,
        std::array<float, 128> & s_col_best,
        std::array<float, 128> & s_row_best) {
    assert(tile != nullptr);
    assert(sinkhorn_iters > 0);

    std::array<float, 128> log_s_col = {};
    std::array<float, 128> log_s_row = {};
    std::vector<float> cur(tile, tile + 128 * 128);

    s_col_best.fill(1.0f);
    s_row_best.fill(1.0f);
    float imbalance_best = llama_kvarn_imbalance(cur);

    auto rebuild_cur = [&]() {
        for (int r = 0; r < 128; ++r) {
            const float s_row = std::exp(log_s_row[r]);
            for (int c = 0; c < 128; ++c) {
                cur[r * 128 + c] = tile[r * 128 + c] / (std::exp(log_s_col[c]) * s_row);
            }
        }
    };

    for (int iter = 0; iter < sinkhorn_iters; ++iter) {
        for (int c = 0; c < 128; ++c) {
            const float std = std::clamp(llama_kvarn_sample_std(cur.data() + c, 128, 128), 1e-3f, 1e3f);
            log_s_col[c] = std::clamp(log_s_col[c] + std::log(std), -0.3f, 10.0f);
        }
        rebuild_cur();

        for (int r = 0; r < 128; ++r) {
            const float std = std::clamp(llama_kvarn_sample_std(cur.data() + r * 128, 128, 1), 1e-3f, 1e3f);
            log_s_row[r] = std::clamp(log_s_row[r] + std::log(std), -0.3f, 10.0f);
        }
        rebuild_cur();

        const float imbalance = llama_kvarn_imbalance(cur);
        if (imbalance <= imbalance_best) {
            imbalance_best = imbalance;
            for (int i = 0; i < 128; ++i) {
                s_col_best[i] = std::exp(log_s_col[i]);
                s_row_best[i] = std::exp(log_s_row[i]);
            }
        }
    }

    balanced.resize(128 * 128);
    for (int r = 0; r < 128; ++r) {
        for (int c = 0; c < 128; ++c) {
            balanced[r * 128 + c] = tile[r * 128 + c] / (s_col_best[c] * s_row_best[r]);
        }
    }
}

static void llama_kvarn_store_fp16(uint8_t * record, size_t offset, int index, float value) {
    const ggml_fp16_t fp16 = ggml_fp32_to_fp16(value);
    std::memcpy(record + offset + size_t(index) * sizeof(fp16), &fp16, sizeof(fp16));
}

static float llama_kvarn_load_fp16(const uint8_t * record, size_t offset, int index) {
    ggml_fp16_t fp16;
    std::memcpy(&fp16, record + offset + size_t(index) * sizeof(fp16), sizeof(fp16));
    return ggml_fp16_to_fp32(fp16);
}

static void llama_kvarn_quantize_tile(
        const float * tile,
        int sinkhorn_iters,
        int bits,
        uint8_t * payload,
        size_t payload_bytes,
        const std::array<float, 128> * fixed_col,
        size_t scale_axis_offset,
        size_t zp_axis_offset,
        size_t other_axis_offset,
        uint8_t * record) {
    assert(tile != nullptr);
    assert(llama_kvarn_valid_bits(bits));
    assert(payload != nullptr);
    assert(record != nullptr);

    std::vector<float> balanced;
    std::array<float, 128> s_col;
    std::array<float, 128> s_row;
    llama_kvarn_variance_normalize(tile, sinkhorn_iters, balanced, s_col, s_row);

    std::vector<uint8_t> q(128 * 128);
    const int qmax = (1 << bits) - 1;
    for (int r = 0; r < 128; ++r) {
        const auto begin = balanced.begin() + r * 128;
        const auto end = begin + 128;
        const float lo = *std::min_element(begin, end);
        const float hi = *std::max_element(begin, end);
        const float scale = std::max((hi - lo) / qmax, 1e-10f);

        for (int c = 0; c < 128; ++c) {
            const float value = std::round((balanced[r * 128 + c] - lo) / scale);
            q[r * 128 + c] = uint8_t(std::clamp(value, 0.0f, float(qmax)));
        }

        const float absorb = fixed_col ? (*fixed_col)[r] : s_row[r];
        llama_kvarn_store_fp16(record, scale_axis_offset, r, absorb * scale);
        llama_kvarn_store_fp16(record, zp_axis_offset, r, absorb * lo);
    }

    for (int i = 0; i < 128; ++i) {
        llama_kvarn_store_fp16(record, other_axis_offset, i, s_col[i]);
    }

    GGML_ASSERT(payload_bytes == llama_kvarn_packed_bytes(128 * 128, bits));
    llama_kvarn_pack_bits(q.data(), 128 * 128, bits, payload);
}

void llama_kvarn_quantize_k_tile(
        const float * tile,
        int sinkhorn_iters,
        int bits,
        const llama_kvarn_tile_layout & layout,
        uint8_t * record) {
    llama_kvarn_quantize_tile(
            tile,
            sinkhorn_iters,
            bits,
            record + layout.k_payload_off,
            layout.k_payload_bytes,
            nullptr,
            layout.k_s_col_off,
            layout.k_zp_off,
            layout.k_s_row_off,
            record);
}

void llama_kvarn_quantize_v_tile(
        const float * tile,
        int sinkhorn_iters,
        int bits,
        const llama_kvarn_tile_layout & layout,
        uint8_t * record) {
    assert(tile != nullptr);
    assert(llama_kvarn_valid_bits(bits));
    assert(record != nullptr);

    std::vector<float> balanced;
    std::array<float, 128> s_col;
    std::array<float, 128> s_row;
    llama_kvarn_variance_normalize(tile, sinkhorn_iters, balanced, s_col, s_row);

    std::vector<uint8_t> q(128 * 128);
    const int qmax = (1 << bits) - 1;
    for (int r = 0; r < 128; ++r) {
        const auto begin = balanced.begin() + r * 128;
        const auto end = begin + 128;
        const float lo = *std::min_element(begin, end);
        const float hi = *std::max_element(begin, end);
        const float scale = std::max((hi - lo) / qmax, 1e-10f);

        for (int c = 0; c < 128; ++c) {
            const float value = std::round((balanced[r * 128 + c] - lo) / scale);
            q[r * 128 + c] = uint8_t(std::clamp(value, 0.0f, float(qmax)));
        }

        llama_kvarn_store_fp16(record, layout.v_s_row_off, r, s_row[r] * scale);
        llama_kvarn_store_fp16(record, layout.v_zp_off, r, s_row[r] * lo);
    }

    for (int c = 0; c < 128; ++c) {
        llama_kvarn_store_fp16(record, layout.v_s_col_off, c, s_col[c]);
    }

    llama_kvarn_pack_bits(q.data(), 128 * 128, bits, record + layout.v_payload_off);
}

void llama_kvarn_dequantize_k_tile(
        const uint8_t * record,
        int bits,
        const llama_kvarn_tile_layout & layout,
        float * tile) {
    assert(record != nullptr);
    assert(llama_kvarn_valid_bits(bits));
    assert(tile != nullptr);

    for (int r = 0; r < 128; ++r) {
        const float scale = llama_kvarn_load_fp16(record, layout.k_s_col_off, r);
        const float zp = llama_kvarn_load_fp16(record, layout.k_zp_off, r);
        for (int c = 0; c < 128; ++c) {
            const float other = llama_kvarn_load_fp16(record, layout.k_s_row_off, c);
            const uint8_t q = llama_kvarn_unpack_bits_value(record + layout.k_payload_off, r * 128 + c, bits);
            tile[r * 128 + c] = (float(q) * scale + zp) * other;
        }
    }
}

void llama_kvarn_dequantize_v_tile(
        const uint8_t * record,
        int bits,
        const llama_kvarn_tile_layout & layout,
        float * tile) {
    assert(record != nullptr);
    assert(llama_kvarn_valid_bits(bits));
    assert(tile != nullptr);

    for (int r = 0; r < 128; ++r) {
        const float scale = llama_kvarn_load_fp16(record, layout.v_s_row_off, r);
        const float zp = llama_kvarn_load_fp16(record, layout.v_zp_off, r);
        for (int c = 0; c < 128; ++c) {
            const float other = llama_kvarn_load_fp16(record, layout.v_s_col_off, c);
            const uint8_t q = llama_kvarn_unpack_bits_value(record + layout.v_payload_off, r * 128 + c, bits);
            tile[r * 128 + c] = (float(q) * scale + zp) * other;
        }
    }
}
