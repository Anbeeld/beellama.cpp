#include "llama-kvarn.h"
#include "llama-kv-cache-kvarn.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void require(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "test-kvarn: %s\n", msg);
        std::abort();
    }
}

static void test_type_table() {
    const int supported_bits[] = { 2, 3, 4, 5, 6, 8 };

    require(llama_kvarn_type_count() == 37, "unexpected KVarN type count");

    const llama_kvarn_type_desc * disabled = llama_kvarn_type_desc_from_name("off");
    require(disabled != nullptr, "disabled type name did not parse");
    require(disabled->type == LLAMA_KVARN_TYPE_DISABLED, "disabled type enum mismatch");
    require(disabled->key_bits == 0 && disabled->value_bits == 0, "disabled bits mismatch");
    require(disabled->group == 128, "disabled group mismatch");

    for (int key_bits : supported_bits) {
        for (int value_bits : supported_bits) {
            const std::string name = "kvarn_k" + std::to_string(key_bits) + "v" + std::to_string(value_bits) + "_g128";
            const llama_kvarn_type_desc * desc = llama_kvarn_type_desc_from_name(name.c_str());
            require(desc != nullptr, "expected type name did not parse");
            require(desc->type != LLAMA_KVARN_TYPE_DISABLED && desc->type != LLAMA_KVARN_TYPE_INVALID, "parsed type enum mismatch");
            require(desc->key_bits == key_bits, "parsed key bits mismatch");
            require(desc->value_bits == value_bits, "parsed value bits mismatch");
            require(desc->group == 128, "parsed group mismatch");

            const llama_kvarn_type_desc * by_type = llama_kvarn_type_desc_from_type(desc->type);
            require(by_type != nullptr, "expected enum did not map to descriptor");
            require(std::string(by_type->name) == name, "enum descriptor name mismatch");
        }
    }

    require(llama_kvarn_type_desc_from_name("kvarn_k7v2_g128") == nullptr, "invalid type parsed");
}

static void test_stage_policy() {
    require(llama_kvarn_non_swa_tail_groups(32768, 256, 256) == 6,
            "short-context KVarN tail policy changed unexpectedly");
    require(llama_kvarn_non_swa_tail_groups(65536, 256, 256) == 8,
            "long-context KVarN precision floor changed unexpectedly");
    require(llama_kvarn_non_swa_tail_groups(65536, 2048, 256) == 18,
            "b2048/ub256 KVarN tail must cover the scheduler window");
    require(llama_kvarn_non_swa_tail_groups(65536, 2048, 512) == 20,
            "b2048/ub512 KVarN tail must cover the scheduler window");
    require(llama_kvarn_non_swa_tail_groups(65536, 2048, 1024) == 24,
            "b2048/ub1024 KVarN tail must cover the scheduler window");
}

static void test_tile_layout() {
    for (size_t i = 0; i < llama_kvarn_type_count(); ++i) {
        const llama_kvarn_type type = (llama_kvarn_type) i;
        if (type == LLAMA_KVARN_TYPE_DISABLED) {
            continue;
        }

        const llama_kvarn_type_desc * desc = llama_kvarn_type_desc_from_type(type);
        require(desc != nullptr, "layout type descriptor missing");

        const llama_kvarn_tile_layout layout = llama_kvarn_make_layout(128, 128, desc->key_bits, desc->value_bits);
        require(layout.k_payload_bytes == size_t(2048 * desc->key_bits), "K payload bytes mismatch");
        require(layout.v_payload_bytes == size_t(2048 * desc->value_bits), "V payload bytes mismatch");
        require(layout.tile_bytes == size_t(2048 * (desc->key_bits + desc->value_bits) + 1536), "tile bytes mismatch");
        require(layout.k_s_col_off == layout.k_payload_off + layout.k_payload_bytes, "K scale offset mismatch");
        require(layout.v_payload_off == layout.k_s_row_off + 128 * sizeof(uint16_t), "V payload offset mismatch");
        require(layout.v_s_col_off == layout.v_payload_off + layout.v_payload_bytes, "V scale offset mismatch");
        require(layout.tile_bytes % 8 == 0, "tile bytes not 8-byte aligned");
    }
}

static void test_head_dimension_slicing() {
    require(llama_kvarn_head_slices(128) == 1, "128-dim head should use one KVarN slice");
    require(llama_kvarn_head_slices(256) == 2, "256-dim head should use two KVarN slices");
    require(llama_kvarn_head_slices(512) == 4, "512-dim head should use four KVarN slices");
    require(llama_kvarn_head_slices(384) == 0, "384-dim head has no native KVarN FA route");
    require(llama_kvarn_head_slices(64)  == 0, "64-dim head is not KVarN slice-compatible");
    require(llama_kvarn_head_slices(513) == 0, "non-128-multiple head is not KVarN slice-compatible");
}

static void test_runtime_validation() {
    llama_kvarn_runtime_requirements supported = {};
    supported.attention_supported = true;
    supported.head_dims_supported = true;
    supported.kv_offload = true;
    supported.native_backend_supported = true;
    supported.n_seq_max = 1;
    supported.kv_unified = false;

    for (size_t i = 0; i < llama_kvarn_type_count(); ++i) {
        const llama_kvarn_type type = (llama_kvarn_type) i;
        if (type == LLAMA_KVARN_TYPE_DISABLED) {
            continue;
        }

        const auto params = llama_kvarn_params_for_type(type);
        require(llama_kvarn_validate_runtime(params, supported) == nullptr, "valid runtime rejected");
    }

    auto invalid = llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128);
    invalid.key_bits = 3;
    require(llama_kvarn_validate_runtime(invalid, supported) != nullptr, "mismatched preset bits accepted");

    invalid = llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128);
    invalid.sink_tokens = 0;
    require(llama_kvarn_validate_runtime(invalid, supported) != nullptr, "unsupported sink tokens accepted");

    auto requirements = supported;
    requirements.attention_supported = false;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) != nullptr,
            "unsupported attention accepted");

    requirements = supported;
    requirements.head_dims_supported = false;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) != nullptr,
            "unsupported head dimension accepted");

    requirements = supported;
    requirements.kv_offload = false;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) != nullptr,
            "CPU KV placement accepted");

    requirements = supported;
    requirements.native_backend_supported = false;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) != nullptr,
            "backend without native KVarN FA accepted");

    requirements = supported;
    requirements.kv_unified = true;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) == nullptr,
            "unified single-sequence runtime rejected");

    requirements = supported;
    requirements.n_seq_max = 2;
    requirements.kv_unified = false;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) == nullptr,
            "non-unified multi-sequence runtime rejected");

    requirements.kv_unified = true;
    require(llama_kvarn_validate_runtime(llama_kvarn_params_for_type(LLAMA_KVARN_K4V2_G128), requirements) == nullptr,
            "unified multi-sequence runtime rejected");
}

static void test_remove_policy() {
    require(llama_kvarn_can_remove_range(-1, 0, -1, 128), "empty sequence removal rejected");
    require(llama_kvarn_can_remove_range(783, -1, -1, 128), "full sequence removal with negative range rejected");
    require(llama_kvarn_can_remove_range(783, 0, -1, 128), "full sequence removal from zero rejected");
    require(llama_kvarn_can_remove_range(783, 0, 784, 128), "explicit full sequence range removal rejected");
    require(!llama_kvarn_can_remove_range(783, 0, 640, 128), "old compressed partial removal accepted");
    require(llama_kvarn_can_remove_range(783, 640, -1, 128), "current/previous tail removal rejected");
}

static void test_pack_roundtrip(int bits) {
    const int n = 257;
    std::vector<uint8_t> values(n);
    for (int i = 0; i < n; ++i) {
        values[i] = uint8_t((i * 7 + 3) & ((1 << bits) - 1));
    }

    std::vector<uint8_t> packed(llama_kvarn_packed_bytes(n, bits), 0);
    llama_kvarn_pack_bits(values.data(), n, bits, packed.data());

    for (int i = 0; i < n; ++i) {
        const uint8_t got = llama_kvarn_unpack_bits_value(packed.data(), i, bits);
        if (got != values[i]) {
            std::fprintf(stderr, "test-kvarn: %d-bit roundtrip mismatch at %d: got %u expected %u\n",
                    bits, i, unsigned(got), unsigned(values[i]));
            std::abort();
        }
    }
}

static void test_hadamard_roundtrip() {
    std::vector<float> values(128);
    std::vector<float> expected(128);
    for (int i = 0; i < 128; ++i) {
        values[i] = std::sin(float(i) * 0.19f) + float(i - 64) * 0.002f;
    }
    expected = values;

    llama_kvarn_hadamard_128(values.data());
    llama_kvarn_hadamard_128(values.data());

    for (int i = 0; i < 128; ++i) {
        require(std::fabs(values[i] - expected[i]) < 1e-5f, "Hadamard roundtrip mismatch");
    }
}

static float tile_rmse(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size(), "RMSE shape mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = double(a[i]) - double(b[i]);
        sum += diff * diff;
    }
    return float(std::sqrt(sum / a.size()));
}

static void test_tile_quantization(llama_kvarn_type type) {
    const auto * desc = llama_kvarn_type_desc_from_type(type);
    require(desc != nullptr, "quantization type descriptor missing");

    const auto layout = llama_kvarn_make_layout(128, 128, desc->key_bits, desc->value_bits);
    std::vector<float> k(128 * 128);
    std::vector<float> v(128 * 128);
    for (int r = 0; r < 128; ++r) {
        for (int c = 0; c < 128; ++c) {
            k[r * 128 + c] =
                std::sin(float(r) * 0.071f) +
                std::cos(float(c) * 0.113f) +
                float((r * 17 + c * 13) % 29 - 14) * 0.015f;
            v[r * 128 + c] =
                std::cos(float(r) * 0.057f) -
                std::sin(float(c) * 0.091f) +
                float((r * 11 + c * 19) % 31 - 15) * 0.012f;
        }
    }

    std::vector<uint8_t> record(layout.tile_bytes, 0);
    llama_kvarn_quantize_k_tile(k.data(), 16, desc->key_bits, layout, record.data());
    llama_kvarn_quantize_v_tile(v.data(), 16, desc->value_bits, layout, record.data());

    std::vector<float> k_dequant(k.size());
    std::vector<float> v_dequant(v.size());
    llama_kvarn_dequantize_k_tile(record.data(), desc->key_bits, layout, k_dequant.data());
    llama_kvarn_dequantize_v_tile(record.data(), desc->value_bits, layout, v_dequant.data());

    for (size_t i = 0; i < k.size(); ++i) {
        require(std::isfinite(k_dequant[i]), "K dequant produced non-finite value");
        require(std::isfinite(v_dequant[i]), "V dequant produced non-finite value");
    }

    const float max_rmse[] = { 0.0f, 0.0f, 0.40f, 0.22f, 0.12f, 0.08f, 0.05f, 0.0f, 0.025f };
    require(tile_rmse(k, k_dequant) < max_rmse[desc->key_bits], "K tile RMSE too high");
    require(tile_rmse(v, v_dequant) < max_rmse[desc->value_bits], "V tile RMSE too high");
}

// Proof that KVarN rotated-domain attention is algebraically equivalent to the
// original-domain decode path, using only the CPU reference quant/dequant. Let R = the
// normalized WHT-128 (symmetric involution: R^2 = I, verified separately by
// test_hadamard_roundtrip). KVarN compressed records store K_rot = R*K and
// V_rot = R*V. Live fp16 K/V stage rows are rotated-domain so decode can read
// stage and records through the same hot path. Reference decode reconstructs
// X_orig = R*dequant(record_or_stage). Rotated-domain attention
// skips that inverse-WHT and instead rotates the query / inverse-rotates the output:
//   K:  Q . K_orig[:,c]        == (R Q) . K_rot[:,c]
//   V:  sum_t w[t] V_orig[t,:] == R ( sum_t w[t] V_rot[t,:] )
static void test_rotated_domain_equivalence() {
    const int bits = 4; // kvarn4
    const auto layout = llama_kvarn_make_layout(128, 128, bits, bits);

    std::vector<float> k(128 * 128);
    std::vector<float> v(128 * 128);
    for (int r = 0; r < 128; ++r) {
        for (int c = 0; c < 128; ++c) {
            k[r * 128 + c] = std::sin(float(r) * 0.071f) + std::cos(float(c) * 0.113f) +
                             float((r * 17 + c * 13) % 29 - 14) * 0.015f;
            v[r * 128 + c] = std::cos(float(r) * 0.057f) - std::sin(float(c) * 0.091f) +
                             float((r * 11 + c * 19) % 31 - 15) * 0.012f;
        }
    }

    std::vector<uint8_t> k_record(layout.tile_bytes, 0);
    std::vector<uint8_t> v_record(layout.tile_bytes, 0);
    llama_kvarn_quantize_k_tile(k.data(), 16, bits, layout, k_record.data());
    llama_kvarn_quantize_v_tile(v.data(), 16, bits, layout, v_record.data());

    std::vector<float> k_rot(128 * 128); // tile[dim*128 + token]
    std::vector<float> v_rot(128 * 128); // tile[token*128 + dim]
    llama_kvarn_dequantize_k_tile(k_record.data(), bits, layout, k_rot.data());
    llama_kvarn_dequantize_v_tile(v_record.data(), bits, layout, v_rot.data());

    // ---- K side: scores ----
    std::vector<float> q(128);
    for (int d = 0; d < 128; ++d) {
        q[d] = std::sin(float(d) * 0.037f) + 0.25f * std::cos(float(d) * 0.0131f);
    }
    std::vector<float> rq = q;
    llama_kvarn_hadamard_128(rq.data()); // R q

    float k_max_abs = 0.0f, k_max_diff = 0.0f;
    for (int c = 0; c < 128; ++c) {
        std::array<float, 128> kcol;                         // K_rot[:,c]
        for (int d = 0; d < 128; ++d) kcol[d] = k_rot[d * 128 + c];
        std::array<float, 128> korig = kcol;                 // K_orig[:,c] = R * K_rot[:,c]
        llama_kvarn_hadamard_128(korig.data());

        double ref = 0.0, rot = 0.0;
        for (int d = 0; d < 128; ++d) {
            ref += double(q[d]) * double(korig[d]);          // Q . K_orig
            rot += double(rq[d]) * double(kcol[d]);          // (R Q) . K_rot
        }
        k_max_abs  = std::max(k_max_abs, std::fabs(float(ref)));
        k_max_diff = std::max(k_max_diff, std::fabs(float(ref - rot)));
    }
    require(k_max_diff < 1e-3f * (1.0f + k_max_abs), "K rotated-domain score mismatch");

    // ---- V side: weighted output ----
    std::vector<float> w(128);
    double wsum = 0.0;
    for (int t = 0; t < 128; ++t) { w[t] = 0.5f + 0.5f * std::sin(float(t) * 0.083f) + 0.01f * float(t); wsum += w[t]; }
    for (int t = 0; t < 128; ++t) w[t] = float(w[t] / wsum);

    std::array<float, 128> ref_o = {}; // sum_t w[t] * R(V_rot[t,:])
    for (int t = 0; t < 128; ++t) {
        std::array<float, 128> vorig;
        for (int d = 0; d < 128; ++d) vorig[d] = v_rot[t * 128 + d];
        llama_kvarn_hadamard_128(vorig.data());
        for (int d = 0; d < 128; ++d) ref_o[d] += w[t] * vorig[d];
    }
    std::array<float, 128> o_rot = {}; // R( sum_t w[t] * V_rot[t,:] )
    for (int t = 0; t < 128; ++t) {
        for (int d = 0; d < 128; ++d) o_rot[d] += w[t] * v_rot[t * 128 + d];
    }
    llama_kvarn_hadamard_128(o_rot.data());

    float v_max_abs = 0.0f, v_max_diff = 0.0f;
    for (int d = 0; d < 128; ++d) {
        v_max_abs  = std::max(v_max_abs, std::fabs(ref_o[d]));
        v_max_diff = std::max(v_max_diff, std::fabs(ref_o[d] - o_rot[d]));
    }
    require(v_max_diff < 1e-3f * (1.0f + v_max_abs), "V rotated-domain output mismatch");
}

static ggml_backend_t init_test_backend(enum ggml_backend_dev_type device_type, bool required) {
    const char * backend_name = std::getenv("GGML_KVARN_TEST_BACKEND");
    const bool use_named_gpu = backend_name != nullptr && backend_name[0] != '\0' && device_type == GGML_BACKEND_DEVICE_TYPE_GPU;

    ggml_backend_t backend = use_named_gpu ?
        ggml_backend_init_by_name(backend_name, nullptr) :
        ggml_backend_init_by_type(device_type, nullptr);
    if (backend == nullptr && !required) {
        return nullptr;
    }
    require(backend != nullptr, use_named_gpu ? "failed to initialize GGML_KVARN_TEST_BACKEND" : "failed to initialize requested backend");
    return backend;
}

static void apply_reference_kvarn_wht_head(float * values, int head_width) {
    require(head_width == 128 || head_width == 256 || head_width == 512, "reference KVarN WHT invalid head width");
    const int slices = head_width / 128;

    for (int slice = 0; slice < slices; ++slice) {
        llama_kvarn_hadamard_128(values + slice * 128);
    }
    if (slices == 1) {
        return;
    }

    const float scale = slices == 2 ? 0.7071067811865475f : 0.5f;
    for (int d = 0; d < 128; ++d) {
        float x[4] = {};
        for (int slice = 0; slice < slices; ++slice) {
            x[slice] = values[slice * 128 + d];
        }
        for (int stride = 1; stride < slices; stride <<= 1) {
            for (int base = 0; base < slices; base += 2 * stride) {
                for (int i = 0; i < stride; ++i) {
                    const float a = x[base + i];
                    const float b = x[base + stride + i];
                    x[base + i] = a + b;
                    x[base + stride + i] = a - b;
                }
            }
        }
        for (int slice = 0; slice < slices; ++slice) {
            values[slice * 128 + d] = x[slice] * scale;
        }
    }
}

static void test_kvarn_wht_op(enum ggml_backend_dev_type device_type, bool required, int head_width) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "KVarN WHT: failed to initialize ggml context");

    constexpr int n_tokens = 7;
    constexpr int n_heads = 3;
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_width, n_heads, n_tokens);
    ggml_tensor * output = ggml_kvarn_wht(ctx, input, head_width);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "KVarN WHT: failed to allocate tensors");

    std::vector<float> src(head_width * n_heads * n_tokens);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = std::sin(float(i) * 0.013f) + std::cos(float(i) * 0.021f) + float(int(i % 17) - 8) * 0.007f;
    }
    std::vector<float> ref = src;
    for (int group = 0; group < n_heads * n_tokens; ++group) {
        apply_reference_kvarn_wht_head(ref.data() + size_t(group) * head_width, head_width);
    }

    ggml_backend_tensor_set(input, src.data(), 0, ggml_nbytes(input));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "KVarN WHT: graph compute failed");

    std::vector<float> got(src.size());
    ggml_backend_tensor_get(output, got.data(), 0, ggml_nbytes(output));

    double mse = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = double(got[i]) - double(ref[i]);
        mse += diff * diff;
        max_diff = std::max(max_diff, std::fabs(diff));
    }
    const double rmse = std::sqrt(mse / double(got.size()));
    if (!std::isfinite(rmse) || rmse > 1e-6 || max_diff > 2e-5) {
        std::fprintf(stderr, "KVarN WHT: head_width=%d rmse=%g max_diff=%g\n", head_width, rmse, max_diff);
        require(false, "KVarN WHT output mismatch");
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static float test_kvarn_record_value(const uint8_t * record, int bits, bool value, int token, int dim) {
    const size_t payload_bytes = llama_kvarn_packed_bytes(128 * 128, bits);
    const size_t scale_axis_off = payload_bytes;
    const size_t zp_axis_off = scale_axis_off + 128 * sizeof(ggml_fp16_t);
    const size_t other_axis_off = zp_axis_off + 128 * sizeof(ggml_fp16_t);
    const int row = value ? token : dim;
    const int col = value ? dim : token;
    ggml_fp16_t scale_fp16;
    ggml_fp16_t zp_fp16;
    ggml_fp16_t other_fp16;
    std::memcpy(&scale_fp16, record + scale_axis_off + row * sizeof(scale_fp16), sizeof(scale_fp16));
    std::memcpy(&zp_fp16, record + zp_axis_off + row * sizeof(zp_fp16), sizeof(zp_fp16));
    std::memcpy(&other_fp16, record + other_axis_off + col * sizeof(other_fp16), sizeof(other_fp16));
    const float scale = ggml_fp16_to_fp32(scale_fp16);
    const float zp = ggml_fp16_to_fp32(zp_fp16);
    const float other = ggml_fp16_to_fp32(other_fp16);
    const uint8_t q = llama_kvarn_unpack_bits_value(record, row * 128 + col, bits);
    return (float(q) * scale + zp) * other;
}

static std::vector<ggml_fp16_t> test_kvarn_reference_decode(
        const ggml_tensor * records,
        const ggml_tensor * stage,
        const std::vector<int64_t> & indices,
        int n_kv,
        int stream_start,
        int n_stream,
        int bits,
        bool value,
        int stage_groups,
        bool emit_rotated = false,
        bool swa = false,
        int head_slices = 1) {
    require(records->type == GGML_TYPE_I8, "reference decode records type mismatch");
    require(stage->type == GGML_TYPE_F16, "reference decode stage type mismatch");
    require(stage->ne[0] == 128, "reference decode stage width mismatch");
    require(stage_groups >= 2, "reference decode invalid stage_groups");
    require(stage->ne[2] % (128 * stage_groups) == 0, "reference decode stage shape mismatch");
    const int n_heads = (int) stage->ne[1];
    require(head_slices == 1 || head_slices == 2 || head_slices == 4,
            "reference decode invalid head_slices");
    require(n_heads % head_slices == 0, "reference decode head_slices mismatch");
    const int total_streams = (int) (stage->ne[2] / (128 * stage_groups));
    require(total_streams > 0, "reference decode total stream mismatch");
    require(records->ne[1] == n_heads, "reference decode head count mismatch");
    require(records->ne[2] % total_streams == 0, "reference decode record shape mismatch");
    require(stream_start >= 0 && n_stream > 0 && stream_start + n_stream <= total_streams,
            "reference decode stream range mismatch");
    if (swa) {
        require((int) indices.size() >= n_kv, "reference decode SWA indices too short");
    }

    const int groups_per_stream = (int) (records->ne[2] / total_streams);
    int tail_groups = stage_groups - 1;
    const int explicit_tail_groups = stage != nullptr ? stage->op_params[8] : 0;
    if (explicit_tail_groups > 0) {
        tail_groups = explicit_tail_groups;
    }
    std::vector<ggml_fp16_t> stage_data(ggml_nelements(stage));
    std::vector<uint8_t> record_data(ggml_nbytes(records));
    ggml_backend_tensor_get(stage, stage_data.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());

    std::vector<int64_t> live_groups(n_stream, 0);
    for (int64_t idx : indices) {
        if (idx < 0) {
            require(swa, "reference decode negative non-SWA index");
            continue;
        }
        const int64_t group_global = idx / 128;
        if (swa) {
            live_groups[0] = std::max(live_groups[0], group_global);
        } else {
            const int64_t stream = group_global / groups_per_stream;
            if (stream >= stream_start && stream < stream_start + n_stream) {
                const int64_t group = group_global - stream * groups_per_stream;
                live_groups[stream - stream_start] = std::max(live_groups[stream - stream_start], group);
            }
        }
    }

    std::vector<ggml_fp16_t> output((size_t) 128 * n_heads * n_kv * n_stream, ggml_fp32_to_fp16(0.0f));
    for (int out_stream = 0; out_stream < n_stream; ++out_stream) {
        const int stream = stream_start + out_stream;
        const int64_t live_group = live_groups[out_stream];
        const int64_t stage_base = (int64_t) stream * 128 * stage_groups;
        const int64_t stage_begin = swa
            ? (live_group >= (tail_groups - 1) ? live_group - (tail_groups - 1) : 0)
            : 0;
        for (int cell = 0; cell < n_kv; ++cell) {
            const int64_t abs_pos = swa ? indices[cell] : cell;
            if (abs_pos < 0) {
                continue;
            }
            const int64_t group = abs_pos / 128;
            const int64_t pos = abs_pos % 128;
            for (int h = 0; h < n_heads; ++h) {
                std::array<float, 128> values = {};
                bool values_original = false;
                bool from_stage;
                bool from_record;
                int64_t stage_pos = 0;
                int64_t record_group = 0;
                if (swa) {
                    from_stage  = group >= stage_begin && group <= live_group;
                    from_record = !from_stage && group >= 0 && group < stage_begin &&
                                  (live_group - group) < groups_per_stream + tail_groups;
                    stage_pos    = stage_base + (group % stage_groups) * 128 + pos;
                    record_group = (int64_t) stream * groups_per_stream + (group % groups_per_stream);
                } else {
                    from_stage  = group == 0 ||
                                  (group > 0 && group <= live_group &&
                                   group + (tail_groups - 1) >= live_group);
                    from_record = !from_stage && group < live_group;
                    stage_pos    = stage_base + (group == 0 ? pos : 128 + ((group - 1) % tail_groups) * 128 + pos);
                    record_group = (int64_t) stream * groups_per_stream + group;
                }
                if (from_stage) {
                    require(stage_pos >= 0 && stage_pos < stage->ne[2], "reference decode stage offset out of range");
                    for (int d = 0; d < 128; ++d) {
                        const size_t off = (size_t) d + (size_t) h * 128 + (size_t) stage_pos * 128 * n_heads;
                        values[d] = ggml_fp16_to_fp32(stage_data[off]);
                    }
                    values_original = false;
                } else if (from_record) {
                    require(record_group >= 0 && record_group < records->ne[2], "reference decode record offset out of range");
                    const size_t record_off = ((size_t) record_group * n_heads + h) * (size_t) records->ne[0];
                    const uint8_t * record = record_data.data() + record_off;
                    for (int d = 0; d < 128; ++d) {
                        values[d] = test_kvarn_record_value(record, bits, value, (int) pos, d);
                    }
                }
                if (emit_rotated == values_original && head_slices == 1) {
                    llama_kvarn_hadamard_128(values.data());
                }
                for (int d = 0; d < 128; ++d) {
                    const size_t out_off = (size_t) d + (size_t) h * 128 +
                        (size_t) cell * 128 * n_heads + (size_t) out_stream * 128 * n_heads * n_kv;
                    output[out_off] = ggml_fp32_to_fp16(values[d]);
                }
            }
        }
    }
    if (!emit_rotated && head_slices > 1) {
        const int head_width = 128 * head_slices;
        std::vector<float> head_values(head_width);
        for (int out_stream = 0; out_stream < n_stream; ++out_stream) {
            for (int cell = 0; cell < n_kv; ++cell) {
                for (int logical_head = 0; logical_head < n_heads / head_slices; ++logical_head) {
                    for (int slice = 0; slice < head_slices; ++slice) {
                        const int h = logical_head * head_slices + slice;
                        for (int d = 0; d < 128; ++d) {
                            const size_t off = (size_t) d + (size_t) h * 128 +
                                (size_t) cell * 128 * n_heads + (size_t) out_stream * 128 * n_heads * n_kv;
                            head_values[slice * 128 + d] = ggml_fp16_to_fp32(output[off]);
                        }
                    }

                    apply_reference_kvarn_wht_head(head_values.data(), head_width);

                    for (int slice = 0; slice < head_slices; ++slice) {
                        const int h = logical_head * head_slices + slice;
                        for (int d = 0; d < 128; ++d) {
                            const size_t off = (size_t) d + (size_t) h * 128 +
                                (size_t) cell * 128 * n_heads + (size_t) out_stream * 128 * n_heads * n_kv;
                            output[off] = ggml_fp32_to_fp16(head_values[slice * 128 + d]);
                        }
                    }
                }
            }
        }
    }
    return output;
}

static std::vector<float> test_kvarn_reference_decode_f32(
        const ggml_tensor * records,
        const ggml_tensor * stage,
        const std::vector<int64_t> & indices,
        int n_kv,
        int stream_start,
        int n_stream,
        int bits,
        bool value,
        int stage_groups,
        bool emit_rotated = false,
        bool swa = false,
        int head_slices = 1) {
    std::vector<ggml_fp16_t> output_f16 = test_kvarn_reference_decode(
            records, stage, indices, n_kv, stream_start, n_stream, bits, value, stage_groups, emit_rotated, swa, head_slices);
    std::vector<float> output(output_f16.size());
    ggml_fp16_to_fp32_row(output_f16.data(), output.data(), output.size());
    return output;
}

static void test_cache_ops(enum ggml_backend_dev_type device_type, bool required, int bits) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize ggml context");

    constexpr int n_tokens = 385;
    constexpr int n_heads = 1;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 384);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, 4);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false, 3);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate KVarN tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            input[t * 128 + d] =
                std::sin(float(d) * 0.071f) +
                std::cos(float(t) * 0.037f) +
                float((d * 13 + t * 17) % 31 - 15) * 0.01f;
        }
    }
    std::vector<int64_t> idx(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        idx[i] = i;
    }
    std::vector<uint8_t> zeros(ggml_nbytes(stage) + ggml_nbytes(records), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, zeros.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, zeros.data(), 0, ggml_nbytes(records));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "KVarN graph compute failed");

    const std::vector<float> output = test_kvarn_reference_decode_f32(records, stored, idx, n_tokens, 0, 1, bits, false, 3);

    double sink_error = 0.0;
    double compressed_error = 0.0;
    double previous_tail_error = 0.0;
    double live_tail_error = 0.0;
    for (int t = 0; t < n_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            const double diff = double(input[t * 128 + d]) - double(output[t * 128 + d]);
            if (t < 128) {
                sink_error += diff * diff;
            } else if (t < 256) {
                compressed_error += diff * diff;
            } else if (t < 384) {
                previous_tail_error += diff * diff;
            } else {
                live_tail_error += diff * diff;
            }
        }
    }
    sink_error = std::sqrt(sink_error / (128 * 128));
    compressed_error = std::sqrt(compressed_error / (128 * 128));
    previous_tail_error = std::sqrt(previous_tail_error / (128 * 128));
    live_tail_error = std::sqrt(live_tail_error / 128);
    require(sink_error < 0.01, "sink reconstruction error too high");
    require(compressed_error < 0.25, "compressed reconstruction error too high");
    require(previous_tail_error < 0.01, "previous tail reconstruction error too high");
    require(live_tail_error < 0.01, "live tail reconstruction error too high");

    std::vector<uint8_t> record_data(ggml_nbytes(records));
    ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());
    require(std::any_of(record_data.begin(), record_data.end(), [](uint8_t v) { return v != 0; }),
            "completed group was not flushed");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void test_cache_ops_multi_stream(enum ggml_backend_dev_type device_type, bool required, int bits) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize ggml context");

    constexpr int n_stream = 2;
    constexpr int kv_size = 512;
    constexpr int n_groups_per_stream = kv_size / 128;
    constexpr int n_tokens_per_stream = 385;
    constexpr int n_tokens = n_tokens_per_stream * n_stream;
    constexpr int n_heads = 1;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 384 * n_stream);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream * n_stream);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false, 3);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate multi-stream KVarN tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            for (int d = 0; d < 128; ++d) {
                input[(s * n_tokens_per_stream + t) * 128 + d] =
                    std::sin(float(d) * 0.071f + float(s) * 0.31f) +
                    std::cos(float(t) * 0.037f + float(s) * 0.23f) +
                    float((d * 13 + t * 17 + s * 19) % 31 - 15) * 0.01f;
            }
        }
    }
    std::vector<int64_t> idx(n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            idx[s * n_tokens_per_stream + t] = int64_t(s * kv_size + t);
        }
    }
    std::vector<uint8_t> zeros(std::max(ggml_nbytes(stage), ggml_nbytes(records)), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, zeros.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, zeros.data(), 0, ggml_nbytes(records));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "multi-stream KVarN graph compute failed");

    const std::vector<float> output = test_kvarn_reference_decode_f32(
            records, stored, idx, n_tokens_per_stream, 0, n_stream, bits, false, 3);

    for (int s = 0; s < n_stream; ++s) {
        double sink_error = 0.0;
        double compressed_error = 0.0;
        double previous_tail_error = 0.0;
        double live_tail_error = 0.0;
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            for (int d = 0; d < 128; ++d) {
                const size_t input_off = size_t(s * n_tokens_per_stream + t) * 128 + d;
                const size_t output_off = size_t(s * n_tokens_per_stream + t) * 128 + d;
                const double diff = double(input[input_off]) - double(output[output_off]);
                if (t < 128) {
                    sink_error += diff * diff;
                } else if (t < 256) {
                    compressed_error += diff * diff;
                } else if (t < 384) {
                    previous_tail_error += diff * diff;
                } else {
                    live_tail_error += diff * diff;
                }
            }
        }
        sink_error = std::sqrt(sink_error / (128 * 128));
        compressed_error = std::sqrt(compressed_error / (128 * 128));
        previous_tail_error = std::sqrt(previous_tail_error / (128 * 128));
        live_tail_error = std::sqrt(live_tail_error / 128);
        require(sink_error < 0.01, "multi-stream sink reconstruction error too high");
        require(compressed_error < 0.25, "multi-stream compressed reconstruction error too high");
        require(previous_tail_error < 0.01, "multi-stream previous tail reconstruction error too high");
        require(live_tail_error < 0.01, "multi-stream live tail reconstruction error too high");
    }

    std::vector<uint8_t> record_data(ggml_nbytes(records));
    ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());
    const size_t stream_record_bytes = size_t(record_bytes) * n_groups_per_stream * n_heads;
    for (int s = 0; s < n_stream; ++s) {
        const auto begin = record_data.begin() + ptrdiff_t(s * stream_record_bytes);
        const auto end = begin + ptrdiff_t(stream_record_bytes);
        require(std::any_of(begin, end, [](uint8_t v) { return v != 0; }),
                "multi-stream completed group was not flushed");
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// SWA sliding-window ring: write more tiles than the record ring holds so old
// slots are reused, then decode the live window. SWA has no permanent sink slot,
// so op_params[8] makes every stage group part of the local F16 tail. The older
// in-window tile comes from records
// whose ring slots were reused — a ring/seal bug would surface stale tiles and
// blow up the error.
static void test_cache_ops_swa(enum ggml_backend_dev_type device_type, bool required) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    const int bits = 4;
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "swa: failed to init ctx");

    constexpr int n_heads = 1;
    constexpr int stage_groups = 4;
    constexpr int tail_groups = 4;         // explicit SWA no-sink tail: tail == stage
    constexpr int gps = 1;                 // deduplicated record ring for the one sealed window tile
    constexpr int n_tiles = 10;            // tiles 0..9 -> ring wraps (tile 6 reuses tile 0's slot)
    constexpr int n_tokens = n_tiles * 128;
    constexpr int window_base = 5 * 128;   // window covers tiles 5..9
    constexpr int n_kv = 5 * 128;          // tile 5 sealed; tiles 6..9 live in staging
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, gps);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false, stage_groups);
    stored->op_params[4] = 1; // SWA ring store
    stored->op_params[8] = tail_groups;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "swa: failed to allocate tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            input[t * 128 + d] =
                std::sin(float(d) * 0.071f) +
                std::cos(float(t) * 0.0037f) +
                float((d * 13 + t * 17) % 31 - 15) * 0.01f;
        }
    }
    std::vector<int64_t> idx(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        idx[i] = i; // absolute token position
    }
    std::vector<int64_t> mat_idx(n_kv);
    for (int cell = 0; cell < n_kv; ++cell) {
        mat_idx[cell] = window_base + cell; // window covers tiles 6..9
    }
    std::vector<uint8_t> zeros(ggml_nbytes(stage) + ggml_nbytes(records), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, zeros.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, zeros.data(), 0, ggml_nbytes(records));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "swa: graph compute failed");

    const std::vector<float> output = test_kvarn_reference_decode_f32(
            records, stored, mat_idx, n_kv, 0, 1, bits, false, stage_groups, false, true);

    double sealed_error = 0.0; // tile 5 -> reused ring slot 0
    double live_error = 0.0;   // tiles 6..9 -> fp16 staging
    for (int cell = 0; cell < n_kv; ++cell) {
        const int abs_pos = window_base + cell;
        for (int d = 0; d < 128; ++d) {
            const double diff = double(input[abs_pos * 128 + d]) - double(output[cell * 128 + d]);
            if (cell < 128) {
                sealed_error += diff * diff;
            } else {
                live_error += diff * diff;
            }
        }
    }
    sealed_error = std::sqrt(sealed_error / (128 * 128));
    live_error = std::sqrt(live_error / (4 * 128 * 128));
    require(sealed_error < 0.25, "swa: sealed (wrapped) tile reconstruction error too high");
    require(live_error < 0.02, "swa: live tail reconstruction error too high");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static std::vector<ggml_fp16_t> test_store_reference_output(
        ggml_backend_t backend,
        int            bits,
        bool           value,
        int            n_stream,
        int            n_heads,
        int            n_tokens_per_stream,
        int            start_idx,
        bool           discontinuous_indices = false,
        bool           seed_stage = true,
        int            stage_groups = 3,
        int            head_slices = 1) {
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize KVarN store parity context");

    const int n_kv = start_idx + n_tokens_per_stream + (discontinuous_indices ? 1 : 0);
    const int n_groups_per_stream = std::max(4, (n_kv + 127) / 128);
    const int n_tokens = n_tokens_per_stream * n_stream;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups * n_stream);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream * n_stream);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, value, stage_groups);
    stored->op_params[3] = n_tokens_per_stream;
    stored->op_params[5] = head_slices;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate KVarN store parity tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            for (int h = 0; h < n_heads; ++h) {
                for (int d = 0; d < 128; ++d) {
                    input[((s * n_tokens_per_stream + t) * n_heads + h) * 128 + d] =
                        std::sin(float(d) * 0.071f + float(h) * 0.13f + float(s) * 0.31f) +
                        std::cos(float(t) * 0.037f + float(h) * 0.11f + float(s) * 0.23f) +
                        float((d * 13 + h * 7 + t * 17 + s * 19) % 31 - 15) * 0.01f;
                }
            }
        }
    }

    std::vector<int64_t> idx(n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens_per_stream; ++t) {
            const int local_idx = start_idx + t + (discontinuous_indices && t >= n_tokens_per_stream / 2 ? 1 : 0);
            idx[s * n_tokens_per_stream + t] = int64_t(s * n_groups_per_stream * 128 + local_idx);
        }
    }

    std::vector<uint8_t> record_zeros(ggml_nbytes(records), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    if (seed_stage) {
        std::vector<ggml_fp16_t> stage_data(ggml_nelements(stage));
        for (size_t i = 0; i < stage_data.size(); ++i) {
            const float f = std::sin(float(i) * 0.017f) + std::cos(float(i) * 0.011f);
            stage_data[i] = ggml_fp32_to_fp16(f);
        }
        ggml_backend_tensor_set(stage, stage_data.data(), 0, ggml_nbytes(stage));
    } else {
        std::vector<uint8_t> stage_zeros(ggml_nbytes(stage), 0);
        ggml_backend_tensor_set(stage, stage_zeros.data(), 0, ggml_nbytes(stage));
    }
    ggml_backend_tensor_set(records, record_zeros.data(), 0, record_zeros.size());

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "KVarN store path graph compute failed");

    std::vector<ggml_fp16_t> output = test_kvarn_reference_decode(
            records, stored, idx, n_kv, 0, n_stream, bits, value, stage_groups, false, false, head_slices);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return output;
}

static std::vector<ggml_fp16_t> test_store_segmented_output(
        ggml_backend_t backend,
        int            bits,
        bool           value,
        int            n_heads,
        int            head_slices,
        int            total_tokens,
        int            segment_tokens,
        int            stage_groups) {
    require(total_tokens > 0 && segment_tokens > 0 && total_tokens % segment_tokens == 0,
            "segmented store route requires even segments");
    require(n_heads > 0 && n_heads % head_slices == 0,
            "segmented store route invalid head slicing");

    ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "segmented store route: failed to initialize context");

    constexpr int n_stream = 1;
    const int n_segments = total_tokens / segment_tokens;
    const int groups_per_stream = std::max(64, (total_tokens + 127) / 128 + stage_groups + 2);
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups * n_stream);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, groups_per_stream * n_stream);

    std::vector<ggml_tensor *> current_segments;
    std::vector<ggml_tensor *> index_segments;
    current_segments.reserve(n_segments);
    index_segments.reserve(n_segments);

    ggml_tensor * stored = stage;
    for (int segment = 0; segment < n_segments; ++segment) {
        ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, segment_tokens);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, segment_tokens);
        current_segments.push_back(current);
        index_segments.push_back(indices);

        stored = ggml_kvarn_store(ctx, current, indices, stored, records, bits, 16, value, stage_groups);
        stored->op_params[3] = segment_tokens;
        stored->op_params[5] = head_slices;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "segmented store route: failed to allocate tensors");

    std::vector<uint8_t> stage_zeros(ggml_nbytes(stage), 0);
    std::vector<uint8_t> record_zeros(ggml_nbytes(records), 0);
    ggml_backend_tensor_set(stage, stage_zeros.data(), 0, stage_zeros.size());
    ggml_backend_tensor_set(records, record_zeros.data(), 0, record_zeros.size());

    for (int segment = 0; segment < n_segments; ++segment) {
        const int base_token = segment * segment_tokens;
        std::vector<float> input((size_t) 128 * n_heads * segment_tokens);
        std::vector<int64_t> indices(segment_tokens);
        for (int t = 0; t < segment_tokens; ++t) {
            const int abs_t = base_token + t;
            indices[t] = abs_t;
            for (int h = 0; h < n_heads; ++h) {
                const int logical_head = h / head_slices;
                const int slice = h % head_slices;
                for (int d = 0; d < 128; ++d) {
                    const int full_d = slice * 128 + d;
                    input[((size_t) t * n_heads + h) * 128 + d] =
                        0.74f * std::sin(float(full_d) * 0.011f + float(abs_t) * 0.019f + float(logical_head) * 0.037f) +
                        0.19f * std::cos(float(full_d) * 0.017f - float(abs_t) * 0.013f + float(logical_head) * 0.029f) +
                        float((full_d * 5 + h * 11 + abs_t * 7) % 41 - 20) * 0.006f;
                }
            }
        }
        ggml_backend_tensor_set(current_segments[segment], input.data(), 0, ggml_nbytes(current_segments[segment]));
        ggml_backend_tensor_set(index_segments[segment], indices.data(), 0, ggml_nbytes(index_segments[segment]));
    }

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "segmented store route: graph compute failed");

    std::vector<int64_t> full_indices(total_tokens);
    for (int t = 0; t < total_tokens; ++t) {
        full_indices[t] = t;
    }
    std::vector<ggml_fp16_t> output = test_kvarn_reference_decode(
            records, stored, full_indices, total_tokens, 0, n_stream, bits, value, stage_groups, false, false, head_slices);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return output;
}

static void require_close_f16_rmse(
        const std::vector<ggml_fp16_t> & actual,
        const std::vector<ggml_fp16_t> & expected,
        float                            rmse_limit,
        const char *                     message) {
    require(actual.size() == expected.size(), "f16 RMSE parity size mismatch");
    double mse = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = double(ggml_fp16_to_fp32(actual[i])) - double(ggml_fp16_to_fp32(expected[i]));
        mse += diff * diff;
        max_diff = std::max(max_diff, std::fabs(diff));
    }

    const double rmse = std::sqrt(mse / double(actual.size()));
    if (!std::isfinite(rmse) || rmse > rmse_limit) {
        std::fprintf(stderr,
                "KVarN CPU/CUDA parity mismatch: rmse=%g max_diff=%g limit=%g\n",
                rmse, max_diff, double(rmse_limit));
        require(false, message);
    }
}

static void fill_hadamard_matrix_128(std::vector<float> & data) {
    constexpr int n = 128;
    data.assign(n * n, 0.0f);
    data[0] = 1.0f / std::sqrt(float(n));

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                const float val = data[i * n + j];

                data[(i + s) * n + j]       =  val;
                data[i * n + (j + s)]       =  val;
                data[(i + s) * n + (j + s)] = -val;
            }
        }
    }
}

static ggml_tensor * apply_hadamard_128(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * rot) {
    const int64_t n = rot->ne[0];
    ggml_tensor * res = nullptr;

    if (!ggml_is_contiguous(cur)) {
        res = ggml_cont_2d(ctx, cur, n, ggml_nelements(cur) / n);
    } else {
        res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur) / n);
    }
    res = ggml_mul_mat(ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    return ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
}

static ggml_tensor * apply_kvarn_wht_head(ggml_context * ctx, ggml_tensor * cur, int head_dim) {
    return ggml_kvarn_wht(ctx, cur, head_dim);
}

static std::vector<float> test_native_flash_attention_output(
        ggml_backend_t backend,
        bool           native_view,
        bool           rotate_graph,
        int            head_dim,
        int            bits_k,
        int            bits_v,
        int            n_q,
        int            n_q_heads = 1,
        int            n_kv_heads = 1,
        int            n_kv = 512,
        int            stage_groups = 3) {
    ggml_init_params params = {
        /*.mem_size   =*/ 32 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "native FA: failed to initialize ggml context");

    constexpr int n_stream   = 1;
    const int slices = head_dim / 128;
    const int record_heads = n_kv_heads * slices;
    const int groups_per_stream = std::max(4, (n_kv + 127) / 128);
    const int k_record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits_k) + 3 * 128 * sizeof(ggml_fp16_t));
    const int v_record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits_v) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * q_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_q, n_q_heads, n_stream);
    const bool use_q_rot = native_view && rotate_graph;
    const bool use_output_rot = use_q_rot;
    ggml_tensor * kvarn_rot = use_q_rot && head_dim == 128 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128) : nullptr;
    ggml_tensor * q = use_q_rot ?
        (head_dim == 128 ? apply_hadamard_128(ctx, q_in, kvarn_rot) : apply_kvarn_wht_head(ctx, q_in, head_dim)) :
        q_in;
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv * n_stream);
    ggml_tensor * current_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, record_heads, n_kv * n_stream);
    ggml_tensor * current_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, record_heads, n_kv * n_stream);
    ggml_tensor * k_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, record_heads, 128 * stage_groups * n_stream);
    ggml_tensor * v_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, record_heads, 128 * stage_groups * n_stream);
    ggml_tensor * k_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, k_record_bytes, record_heads, groups_per_stream * n_stream);
    ggml_tensor * v_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, v_record_bytes, record_heads, groups_per_stream * n_stream);

    ggml_tensor * stored_k = ggml_kvarn_store(ctx, current_k, indices, k_stage, k_records, bits_k, 16, false, stage_groups);
    ggml_tensor * stored_v = ggml_kvarn_store(ctx, current_v, indices, v_stage, v_records, bits_v, 16, true,  stage_groups);
    stored_k->op_params[3] = n_kv;
    stored_v->op_params[3] = n_kv;
    stored_k->op_params[5] = slices;
    stored_v->op_params[5] = slices;

    ggml_tensor * k_ref = native_view ? nullptr : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, n_kv_heads, n_kv, n_stream);
    ggml_tensor * v_ref = native_view ? nullptr : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, n_kv_heads, n_kv, n_stream);
    ggml_tensor * k = native_view ?
        ggml_kvarn_view(ctx, k_records, stored_k, indices, n_kv, 0, n_stream, bits_k, false, stage_groups) : k_ref;
    ggml_tensor * v = native_view ?
        ggml_kvarn_view(ctx, v_records, stored_v, indices, n_kv, 0, n_stream, bits_v, true,  stage_groups) : v_ref;

    if (native_view && slices > 1) {
        k = ggml_reshape_4d(ctx, k, head_dim, n_kv_heads, n_kv, n_stream);
        v = ggml_reshape_4d(ctx, v, head_dim, n_kv_heads, n_kv, n_stream);
    }
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, 1, n_stream);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / std::sqrt(float(head_dim)), 0.0f, 0.0f);
    if (native_view) {
        out->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN] = rotate_graph ?
            GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED :
            GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL;
    }
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    if (use_output_rot) {
        out = ggml_cont(ctx, out);
        out = head_dim == 128 ? apply_hadamard_128(ctx, out, kvarn_rot) : apply_kvarn_wht_head(ctx, out, head_dim);
    }

    ggml_cgraph * store_graph = nullptr;
    if (!native_view) {
        store_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(store_graph, stored_k);
        ggml_build_forward_expand(store_graph, stored_v);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "native FA: failed to allocate tensors");

    std::vector<float> q_data((size_t) head_dim * n_q * n_q_heads * n_stream);
    for (int qh = 0; qh < n_q_heads; ++qh) {
        for (int iq = 0; iq < n_q; ++iq) {
            for (int d = 0; d < head_dim; ++d) {
                q_data[((size_t) qh * n_q + iq) * head_dim + d] =
                    0.09f * std::sin(float(d) * 0.017f + float(iq) * 0.13f + float(qh) * 0.021f) +
                    0.07f * std::cos(float(d) * 0.031f - float(iq) * 0.05f + float(qh) * 0.033f);
            }
        }
    }

    std::vector<float> k_data((size_t) 128 * record_heads * n_kv * n_stream);
    std::vector<float> v_data(k_data.size());
    for (int t = 0; t < n_kv; ++t) {
        for (int h = 0; h < n_kv_heads; ++h) {
            for (int slice = 0; slice < slices; ++slice) {
                const int record_head = h * slices + slice;
                for (int d = 0; d < 128; ++d) {
                    const int full_d = slice * 128 + d;
                    const size_t off = ((size_t) t * record_heads + record_head) * 128 + d;
                    k_data[off] =
                        0.80f * std::sin(float(full_d) * 0.011f + float(t) * 0.021f) +
                        0.10f * std::cos(float(t) * 0.009f + float(h) * 0.17f);
                    v_data[off] =
                        0.75f * std::cos(float(full_d) * 0.013f - float(t) * 0.019f) +
                        0.08f * std::sin(float(t) * 0.015f + float(h) * 0.23f);
                }
            }
        }
    }

    std::vector<int64_t> idx(n_kv * n_stream);
    for (int i = 0; i < n_kv * n_stream; ++i) {
        idx[i] = i;
    }

    std::vector<ggml_fp16_t> mask_data((size_t) n_kv * n_q * n_stream);
    for (int iq = 0; iq < n_q; ++iq) {
        for (int ikv = 0; ikv < n_kv; ++ikv) {
            mask_data[(size_t) iq * n_kv + ikv] = ggml_fp32_to_fp16(ikv <= iq + n_kv - n_q ? 0.0f : -INFINITY);
        }
    }

    std::vector<uint8_t> k_stage_zeros(ggml_nbytes(k_stage), 0);
    std::vector<uint8_t> v_stage_zeros(ggml_nbytes(v_stage), 0);
    std::vector<uint8_t> k_record_zeros(ggml_nbytes(k_records), 0);
    std::vector<uint8_t> v_record_zeros(ggml_nbytes(v_records), 0);

    ggml_backend_tensor_set(q_in, q_data.data(), 0, ggml_nbytes(q_in));
    if (kvarn_rot != nullptr) {
        std::vector<float> rot_data;
        fill_hadamard_matrix_128(rot_data);
        ggml_backend_tensor_set(kvarn_rot, rot_data.data(), 0, ggml_nbytes(kvarn_rot));
    }
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(current_k, k_data.data(), 0, ggml_nbytes(current_k));
    ggml_backend_tensor_set(current_v, v_data.data(), 0, ggml_nbytes(current_v));
    ggml_backend_tensor_set(k_stage, k_stage_zeros.data(), 0, k_stage_zeros.size());
    ggml_backend_tensor_set(v_stage, v_stage_zeros.data(), 0, v_stage_zeros.size());
    ggml_backend_tensor_set(k_records, k_record_zeros.data(), 0, k_record_zeros.size());
    ggml_backend_tensor_set(v_records, v_record_zeros.data(), 0, v_record_zeros.size());
    ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask));

    if (!native_view) {
        require(ggml_backend_graph_compute(backend, store_graph) == GGML_STATUS_SUCCESS,
                "native FA: reference store graph compute failed");
        const std::vector<ggml_fp16_t> k_ref_data = test_kvarn_reference_decode(
                k_records, stored_k, idx, n_kv, 0, n_stream, bits_k, false, stage_groups, false, false, slices);
        const std::vector<ggml_fp16_t> v_ref_data = test_kvarn_reference_decode(
                v_records, stored_v, idx, n_kv, 0, n_stream, bits_v, true, stage_groups, false, false, slices);
        ggml_backend_tensor_set(k_ref, k_ref_data.data(), 0, ggml_nbytes(k_ref));
        ggml_backend_tensor_set(v_ref, v_ref_data.data(), 0, ggml_nbytes(v_ref));
    }

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            native_view ? "native FA: native-view graph compute failed" : "native FA: reference graph compute failed");

    std::vector<float> output(ggml_nelements(out));
    ggml_backend_tensor_get(out, output.data(), 0, ggml_nbytes(out));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return output;
}

static std::vector<float> test_native_flash_attention_segmented_output(
        ggml_backend_t backend,
        bool           native_view,
        int            head_dim,
        int            bits_k,
        int            bits_v,
        int            n_q,
        int            n_q_heads,
        int            n_kv_heads,
        int            n_segments,
        int            segment_tokens,
        int            stage_groups,
        bool           swa = false,
        int            n_kv_override = 0,
        bool           ring_view = false,
        int            sliding_window = 0,
        float          attn_scale = 0.0f,
        bool           scramble_view = false,
        int            tail_groups_override = 0,
        bool           mixed_domain = false) {
    ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "segmented native FA: failed to initialize ggml context");

    constexpr int n_stream = 1;
    const int n_total_tokens = n_segments * segment_tokens;
    const int n_kv = n_kv_override > 0 ? n_kv_override : n_total_tokens;
    require(n_kv <= n_total_tokens, "segmented native FA: view cannot exceed stored tokens");
    require(n_q <= segment_tokens, "segmented native FA: q width must fit in final segment");
    require(segment_tokens > 0 && (segment_tokens % 128) == 0, "segmented native FA: segment size must be tile aligned");
    const int slices = head_dim / 128;
    const int record_heads = n_kv_heads * slices;
    const int tail_groups = tail_groups_override > 0 ? tail_groups_override : stage_groups - 1;
    require(tail_groups > 0 && tail_groups <= stage_groups,
            "segmented native FA: invalid tail_groups override");
    const int groups_per_stream = std::max(4, (n_kv + 127) / 128 + (swa ? 2 : 0));
    const int k_record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits_k) + 3 * 128 * sizeof(ggml_fp16_t));
    const int v_record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits_v) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * q_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_q, n_q_heads, n_stream);
    ggml_tensor * q = mixed_domain ? apply_kvarn_wht_head(ctx, q_in, head_dim) : q_in;
    ggml_tensor * k_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, record_heads, 128 * stage_groups * n_stream);
    ggml_tensor * v_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, record_heads, 128 * stage_groups * n_stream);
    ggml_tensor * k_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, k_record_bytes, record_heads, groups_per_stream * n_stream);
    ggml_tensor * v_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, v_record_bytes, record_heads, groups_per_stream * n_stream);
    ggml_tensor * view_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv * n_stream);

    std::vector<ggml_tensor *> current_k_segments;
    std::vector<ggml_tensor *> current_v_segments;
    std::vector<ggml_tensor *> indices_segments;
    current_k_segments.reserve(n_segments);
    current_v_segments.reserve(n_segments);
    indices_segments.reserve(n_segments);

    ggml_tensor * stored_k = k_stage;
    ggml_tensor * stored_v = v_stage;
    for (int segment = 0; segment < n_segments; ++segment) {
        ggml_tensor * current_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, record_heads, segment_tokens * n_stream);
        ggml_tensor * current_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, record_heads, segment_tokens * n_stream);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, segment_tokens * n_stream);
        current_k_segments.push_back(current_k);
        current_v_segments.push_back(current_v);
        indices_segments.push_back(indices);

        stored_k = ggml_kvarn_store(ctx, current_k, indices, stored_k, k_records, bits_k, 16, false, stage_groups);
        stored_v = ggml_kvarn_store(ctx, current_v, indices, stored_v, v_records, bits_v, 16, true,  stage_groups);
        stored_k->op_params[3] = segment_tokens;
        stored_v->op_params[3] = segment_tokens;
        stored_k->op_params[5] = slices;
        stored_v->op_params[5] = slices;
        if (swa) {
            stored_k->op_params[4] = 1;
            stored_v->op_params[4] = 1;
        }
        if (tail_groups_override > 0) {
            stored_k->op_params[8] = tail_groups;
            stored_v->op_params[8] = tail_groups;
        }
    }

    ggml_tensor * k_ref = native_view ? nullptr : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, n_kv_heads, n_kv, n_stream);
    ggml_tensor * v_ref = native_view ? nullptr : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, n_kv_heads, n_kv, n_stream);
    ggml_tensor * k = native_view ?
        ggml_kvarn_view(ctx, k_records, stored_k, view_indices, n_kv, 0, n_stream, bits_k, false, stage_groups) : k_ref;
    ggml_tensor * v = native_view ?
        ggml_kvarn_view(ctx, v_records, stored_v, view_indices, n_kv, 0, n_stream, bits_v, true,  stage_groups) : v_ref;
    if (native_view && swa) {
        k->op_params[6] = 1;
        v->op_params[6] = 1;
    }
    if (native_view && tail_groups_override > 0) {
        k->op_params[8] = tail_groups;
        v->op_params[8] = tail_groups;
    }

    if (native_view && slices > 1) {
        k = ggml_reshape_4d(ctx, k, head_dim, n_kv_heads, n_kv, n_stream);
        v = ggml_reshape_4d(ctx, v, head_dim, n_kv_heads, n_kv, n_stream);
    }
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, 1, n_stream);
    const float scale = attn_scale > 0.0f ? attn_scale : 1.0f / std::sqrt(float(head_dim));
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    if (native_view) {
        out->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN] = mixed_domain ?
            GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V :
            GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ORIGINAL;
    }
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    ggml_cgraph * store_graph = nullptr;
    if (!native_view) {
        store_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(store_graph, stored_k);
        ggml_build_forward_expand(store_graph, stored_v);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "segmented native FA: failed to allocate tensors");

    std::vector<float> q_data((size_t) head_dim * n_q * n_q_heads * n_stream);
    for (int qh = 0; qh < n_q_heads; ++qh) {
        for (int iq = 0; iq < n_q; ++iq) {
            const int abs_q = n_total_tokens - n_q + iq;
            for (int d = 0; d < head_dim; ++d) {
                q_data[((size_t) qh * n_q + iq) * head_dim + d] =
                    0.08f * std::sin(float(d) * 0.019f + float(abs_q) * 0.037f + float(qh) * 0.023f) +
                    0.06f * std::cos(float(d) * 0.041f - float(abs_q) * 0.029f + float(qh) * 0.017f);
            }
        }
    }

    std::vector<std::vector<float>> k_segment_data;
    std::vector<std::vector<float>> v_segment_data;
    std::vector<std::vector<int64_t>> idx_segment_data;
    k_segment_data.reserve(n_segments);
    v_segment_data.reserve(n_segments);
    idx_segment_data.reserve(n_segments);

    for (int segment = 0; segment < n_segments; ++segment) {
        const int base_token = segment * segment_tokens;
        std::vector<float> k_data((size_t) 128 * record_heads * segment_tokens * n_stream);
        std::vector<float> v_data(k_data.size());
        std::vector<int64_t> idx(segment_tokens * n_stream);

        for (int t = 0; t < segment_tokens; ++t) {
            const int abs_t = base_token + t;
            idx[t] = int64_t(abs_t);
            for (int h = 0; h < n_kv_heads; ++h) {
                for (int slice = 0; slice < slices; ++slice) {
                    const int record_head = h * slices + slice;
                    for (int d = 0; d < 128; ++d) {
                        const int full_d = slice * 128 + d;
                        const size_t off = ((size_t) t * record_heads + record_head) * 128 + d;
                        k_data[off] =
                            0.82f * std::sin(float(full_d) * 0.010f + float(abs_t) * 0.023f) +
                            0.09f * std::cos(float(abs_t) * 0.007f + float(h) * 0.19f);
                        v_data[off] =
                            0.73f * std::cos(float(full_d) * 0.012f - float(abs_t) * 0.017f) +
                            0.10f * std::sin(float(abs_t) * 0.013f + float(h) * 0.29f);
                    }
                }
            }
        }

        k_segment_data.push_back(std::move(k_data));
        v_segment_data.push_back(std::move(v_data));
        idx_segment_data.push_back(std::move(idx));
    }

    std::vector<int64_t> view_idx(n_kv * n_stream);
    const int view_base = n_total_tokens - n_kv;
    if (ring_view) {
        const int last = n_total_tokens - 1;
        for (int i = 0; i < n_kv * n_stream; ++i) {
            int abs_pos = last - ((last - i) % n_kv + n_kv) % n_kv;
            if (abs_pos < view_base) {
                abs_pos -= n_kv;
            }
            view_idx[i] = abs_pos >= view_base ? int64_t(abs_pos) : int64_t(-1);
        }
    } else {
        for (int i = 0; i < n_kv * n_stream; ++i) {
            view_idx[i] = int64_t(view_base + i);
        }
    }
    if (scramble_view && n_kv >= 128) {
        std::swap(view_idx[1], view_idx[2]);
        std::swap(view_idx[65], view_idx[66]);
    }

    std::vector<ggml_fp16_t> mask_data((size_t) n_kv * n_q * n_stream);
    for (int iq = 0; iq < n_q; ++iq) {
        const int abs_q = n_total_tokens - n_q + iq;
        for (int ikv = 0; ikv < n_kv; ++ikv) {
            const bool visible = view_idx[ikv] <= abs_q &&
                (sliding_window <= 0 || abs_q - view_idx[ikv] < sliding_window);
            mask_data[(size_t) iq * n_kv + ikv] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
        }
    }

    std::vector<uint8_t> k_stage_zeros(ggml_nbytes(k_stage), 0);
    std::vector<uint8_t> v_stage_zeros(ggml_nbytes(v_stage), 0);
    std::vector<uint8_t> k_record_zeros(ggml_nbytes(k_records), 0);
    std::vector<uint8_t> v_record_zeros(ggml_nbytes(v_records), 0);

    ggml_backend_tensor_set(q_in, q_data.data(), 0, ggml_nbytes(q_in));
    ggml_backend_tensor_set(k_stage, k_stage_zeros.data(), 0, k_stage_zeros.size());
    ggml_backend_tensor_set(v_stage, v_stage_zeros.data(), 0, v_stage_zeros.size());
    ggml_backend_tensor_set(k_records, k_record_zeros.data(), 0, k_record_zeros.size());
    ggml_backend_tensor_set(v_records, v_record_zeros.data(), 0, v_record_zeros.size());
    ggml_backend_tensor_set(view_indices, view_idx.data(), 0, ggml_nbytes(view_indices));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask));

    for (int segment = 0; segment < n_segments; ++segment) {
        ggml_backend_tensor_set(current_k_segments[segment], k_segment_data[segment].data(), 0, ggml_nbytes(current_k_segments[segment]));
        ggml_backend_tensor_set(current_v_segments[segment], v_segment_data[segment].data(), 0, ggml_nbytes(current_v_segments[segment]));
        ggml_backend_tensor_set(indices_segments[segment], idx_segment_data[segment].data(), 0, ggml_nbytes(indices_segments[segment]));
    }

    if (!native_view) {
        require(ggml_backend_graph_compute(backend, store_graph) == GGML_STATUS_SUCCESS,
                "segmented native FA: reference store graph compute failed");
        const std::vector<ggml_fp16_t> k_ref_data = test_kvarn_reference_decode(
                k_records, stored_k, view_idx, n_kv, 0, n_stream, bits_k, false, stage_groups, mixed_domain, swa, slices);
        const std::vector<ggml_fp16_t> v_ref_data = test_kvarn_reference_decode(
                v_records, stored_v, view_idx, n_kv, 0, n_stream, bits_v, true, stage_groups, false, swa, slices);
        ggml_backend_tensor_set(k_ref, k_ref_data.data(), 0, ggml_nbytes(k_ref));
        ggml_backend_tensor_set(v_ref, v_ref_data.data(), 0, ggml_nbytes(v_ref));
    }

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            native_view ? "segmented native FA: native-view graph compute failed" : "segmented native FA: reference graph compute failed");

    std::vector<float> output(ggml_nelements(out));
    ggml_backend_tensor_get(out, output.data(), 0, ggml_nbytes(out));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return output;
}

static void require_close_f32_rmse(
        const std::vector<float> & actual,
        const std::vector<float> & expected,
        float                      rmse_limit,
        const char *               message) {
    require(actual.size() == expected.size(), "f32 RMSE parity size mismatch");
    double mse = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "f32 parity output contained non-finite value");
        const double diff = double(actual[i]) - double(expected[i]);
        mse += diff * diff;
        max_diff = std::max(max_diff, std::fabs(diff));
    }

    const double rmse = std::sqrt(mse / double(actual.size()));
    if (!std::isfinite(rmse) || rmse > rmse_limit) {
        std::fprintf(stderr,
                "KVarN native FA parity mismatch: rmse=%g max_diff=%g limit=%g\n",
                rmse, max_diff, double(rmse_limit));
        require(false, message);
    }
}

static void require_segmented_raw_roundtrip(
        ggml_backend_t backend,
        int            head_dim,
        int            bits,
        bool           value,
        int            n_heads,
        int            n_segments,
        int            segment_tokens,
        int            stage_groups,
        float          rmse_limit,
        const char *   message,
        bool           swa = false,
        int            n_kv_override = 0,
        bool           ring_view = false,
        int            tail_groups_override = 0) {
    ggml_init_params params = {
        /*.mem_size   =*/ 32 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "segmented raw roundtrip: failed to initialize ggml context");

    const int n_tokens = n_segments * segment_tokens;
    const int n_kv = n_kv_override > 0 ? n_kv_override : n_tokens;
    require(n_kv <= n_tokens, "segmented raw roundtrip: view cannot exceed stored tokens");
    const int slices = head_dim / 128;
    const int record_heads = n_heads * slices;
    const int tail_groups = tail_groups_override > 0 ? tail_groups_override : stage_groups - 1;
    require(tail_groups > 0 && tail_groups <= stage_groups,
            "segmented raw roundtrip: invalid tail_groups override");
    const int groups_per_stream = std::max(4, (swa ? (n_kv + 127) / 128 + 2 : (n_tokens + 127) / 128));
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, record_heads, 128 * stage_groups);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, record_heads, groups_per_stream);

    std::vector<ggml_tensor *> currents;
    std::vector<ggml_tensor *> indices_tensors;
    currents.reserve(n_segments);
    indices_tensors.reserve(n_segments);

    ggml_tensor * stored = stage;
    for (int segment = 0; segment < n_segments; ++segment) {
        ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, record_heads, segment_tokens);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, segment_tokens);
        currents.push_back(current);
        indices_tensors.push_back(indices);

        stored = ggml_kvarn_store(ctx, current, indices, stored, records, bits, 16, value, stage_groups);
        stored->op_params[3] = segment_tokens;
        stored->op_params[5] = slices;
        if (swa) {
            stored->op_params[4] = 1;
        }
        if (tail_groups_override > 0) {
            stored->op_params[8] = tail_groups;
        }
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "segmented raw roundtrip: failed to allocate tensors");

    std::vector<float> expected((size_t) n_tokens * n_heads * head_dim);
    std::vector<std::vector<float>> segment_data;
    std::vector<std::vector<int64_t>> idx_data;
    segment_data.reserve(n_segments);
    idx_data.reserve(n_segments);

    for (int segment = 0; segment < n_segments; ++segment) {
        const int base_token = segment * segment_tokens;
        std::vector<float> data((size_t) 128 * record_heads * segment_tokens);
        std::vector<int64_t> idx(segment_tokens);
        for (int t = 0; t < segment_tokens; ++t) {
            const int abs_t = base_token + t;
            idx[t] = int64_t(abs_t);
            for (int h = 0; h < n_heads; ++h) {
                for (int slice = 0; slice < slices; ++slice) {
                    const int record_head = h * slices + slice;
                    for (int d = 0; d < 128; ++d) {
                        const int full_d = slice * 128 + d;
                        const float x =
                            0.67f * std::sin(float(full_d) * 0.013f + float(abs_t) * 0.017f + float(h) * 0.07f) +
                            0.21f * std::cos(float(full_d) * 0.037f - float(abs_t) * 0.011f + float(h) * 0.13f);
                        data[((size_t) t * record_heads + record_head) * 128 + d] = x;
                        expected[((size_t) abs_t * n_heads + h) * head_dim + full_d] = x;
                    }
                }
            }
        }
        segment_data.push_back(std::move(data));
        idx_data.push_back(std::move(idx));
    }

    std::vector<uint8_t> stage_zeros(ggml_nbytes(stage), 0);
    std::vector<uint8_t> record_zeros(ggml_nbytes(records), 0);
    ggml_backend_tensor_set(stage, stage_zeros.data(), 0, stage_zeros.size());
    ggml_backend_tensor_set(records, record_zeros.data(), 0, record_zeros.size());
    for (int segment = 0; segment < n_segments; ++segment) {
        ggml_backend_tensor_set(currents[segment], segment_data[segment].data(), 0, ggml_nbytes(currents[segment]));
        ggml_backend_tensor_set(indices_tensors[segment], idx_data[segment].data(), 0, ggml_nbytes(indices_tensors[segment]));
    }

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "segmented raw roundtrip: store graph compute failed");

    std::vector<int64_t> view_idx;
    if (swa) {
        view_idx.resize(n_kv);
        const int view_base = n_tokens - n_kv;
        if (ring_view) {
            const int last = n_tokens - 1;
            for (int i = 0; i < n_kv; ++i) {
                int abs_pos = last - ((last - i) % n_kv + n_kv) % n_kv;
                if (abs_pos < view_base) {
                    abs_pos -= n_kv;
                }
                view_idx[i] = abs_pos >= view_base ? int64_t(abs_pos) : int64_t(-1);
            }
        } else {
            for (int i = 0; i < n_kv; ++i) {
                view_idx[i] = int64_t(view_base + i);
            }
        }
    }

    const std::vector<ggml_fp16_t> decoded = test_kvarn_reference_decode(
            records, stored, swa ? view_idx : idx_data.back(), n_kv, 0, 1, bits, value, stage_groups, false, swa, slices);
    double mse = 0.0;
    double max_diff = 0.0;
    size_t n_checked = 0;
    for (int t = 0; t < n_kv; ++t) {
        const int expected_token = swa ? int(view_idx[t]) : t;
        if (expected_token < 0) {
            continue;
        }
        for (int h = 0; h < n_heads; ++h) {
            for (int slice = 0; slice < slices; ++slice) {
                const int record_head = h * slices + slice;
                for (int d = 0; d < 128; ++d) {
                    const int full_d = slice * 128 + d;
                    const size_t decoded_off = (size_t) d + (size_t) record_head * 128 +
                        (size_t) t * 128 * record_heads;
                    const size_t expected_off = ((size_t) expected_token * n_heads + h) * head_dim + full_d;
                    const double diff = double(ggml_fp16_to_fp32(decoded[decoded_off])) - double(expected[expected_off]);
                    mse += diff * diff;
                    ++n_checked;
                    max_diff = std::max(max_diff, std::fabs(diff));
                }
            }
        }
    }
    require(n_checked > 0, "segmented raw roundtrip: no cells checked");

    const double rmse = std::sqrt(mse / double(n_checked));
    if (!std::isfinite(rmse) || rmse > rmse_limit) {
        std::fprintf(stderr,
                "segmented raw KVarN roundtrip mismatch: rmse=%g max_diff=%g limit=%g\n",
                rmse, max_diff, double(rmse_limit));
        require(false, message);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static bool backend_supports_kvarn_flash_attention_shape(ggml_backend_t backend, int head_dim) {
    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "support gate: failed to initialize ggml context");

    constexpr int n_q          = 4;
    constexpr int n_kv         = 128;
    constexpr int n_stream     = 1;
    constexpr int stage_groups = 3;
    const int slices = head_dim / 128;
    const int record_heads = slices;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, 4) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_q, 1, n_stream);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv);
    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, record_heads, n_kv);
    ggml_tensor * k_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, record_heads, 128 * stage_groups);
    ggml_tensor * v_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, record_heads, 128 * stage_groups);
    ggml_tensor * k_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, record_heads, 1);
    ggml_tensor * v_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, record_heads, 1);
    ggml_tensor * stored_k = ggml_kvarn_store(ctx, current, indices, k_stage, k_records, 4, 16, false, stage_groups);
    ggml_tensor * stored_v = ggml_kvarn_store(ctx, current, indices, v_stage, v_records, 4, 16, true,  stage_groups);
    ggml_tensor * k = ggml_kvarn_view(ctx, k_records, stored_k, indices, n_kv, 0, n_stream, 4, false, stage_groups);
    ggml_tensor * v = ggml_kvarn_view(ctx, v_records, stored_v, indices, n_kv, 0, n_stream, 4, true,  stage_groups);
    if (slices > 1) {
        k = ggml_reshape_4d(ctx, k, head_dim, 1, n_kv, n_stream);
        v = ggml_reshape_4d(ctx, v, head_dim, 1, n_kv, n_stream);
    }
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, 1, n_stream);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / std::sqrt(float(head_dim)), 0.0f, 0.0f);

    const bool supported = ggml_backend_supports_op(backend, out);
    ggml_free(ctx);
    return supported;
}

static void test_native_flash_attention_support_gates() {
    ggml_backend_t cpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    require(!backend_supports_kvarn_flash_attention_shape(cpu_backend, 128),
            "CPU backend accepted KVarN view FlashAttention as ordinary F16");
    ggml_backend_free(cpu_backend);

    ggml_backend_t gpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_GPU, false);
    if (gpu_backend == nullptr) {
        return;
    }

    if (backend_supports_kvarn_flash_attention_shape(gpu_backend, 128)) {
        require( backend_supports_kvarn_flash_attention_shape(gpu_backend, 256),
                "native KVarN FlashAttention rejected supported 256-dim heads");
        require( backend_supports_kvarn_flash_attention_shape(gpu_backend, 512),
                "native KVarN FlashAttention rejected supported 512-dim heads");
        require(!backend_supports_kvarn_flash_attention_shape(gpu_backend, 384),
                "GPU backend accepted unsupported 384-dim KVarN view FlashAttention as ordinary F16");
    }

    ggml_backend_free(gpu_backend);
}

static void test_native_flash_attention_gpu() {
    ggml_backend_t gpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_GPU, false);
    if (gpu_backend == nullptr) {
        return;
    }
    ggml_backend_t cpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_CPU, true);

    for (int head_dim : { 128, 256, 512 }) {
        constexpr int n_q_native = 4;
        const std::vector<float> expected = test_native_flash_attention_output(cpu_backend, false, false, head_dim, 4, 3, n_q_native);
        const std::vector<float> actual   = test_native_flash_attention_output(gpu_backend, true,  true,  head_dim, 4, 3, n_q_native);
        require_close_f32_rmse(actual, expected, 1e-2f,
                "native KVarN FlashAttention output differs from CPU reference decode");

        constexpr int n_q_prefill = 64;
        const std::vector<float> expected_prefill = test_native_flash_attention_output(cpu_backend, false, false, head_dim, 4, 3, n_q_prefill);
        const std::vector<float> actual_prefill   = test_native_flash_attention_output(gpu_backend, true,  false, head_dim, 4, 3, n_q_prefill);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "large-prefill original-domain KVarN FlashAttention differs from CPU reference");
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_q_heads = 6;
        constexpr int n_kv_heads = 1;
        constexpr int n_kv = 4096;
        constexpr int stage_groups = 25;
        for (int n_q_prefill : { 512, 1024 }) {
            const std::vector<float> expected_prefill = test_native_flash_attention_output(
                    cpu_backend, false, false, head_dim, 4, 4, n_q_prefill, n_q_heads, n_kv_heads, n_kv, stage_groups);
            const std::vector<float> actual_prefill = test_native_flash_attention_output(
                    gpu_backend, true, false, head_dim, 4, 4, n_q_prefill, n_q_heads, n_kv_heads, n_kv, stage_groups);
            require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                    "large-prefill multi-head original-domain KVarN FlashAttention differs from CPU reference");
        }
    }

    {
        constexpr int head_dim = 512;
        constexpr int n_q_prefill = 256;
        constexpr int n_q_heads = 32;
        constexpr int n_kv_heads = 4;
        constexpr int n_segments = 16;
        constexpr int segment_tokens = 256;
        constexpr int stage_groups = 19;
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, false, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "segmented D512 KVarN8 K roundtrip differs from original-domain input");
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, true, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "segmented D512 KVarN8 V roundtrip differs from original-domain input");
        const std::vector<float> expected_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, false, head_dim, 4, 4, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups);
        const std::vector<float> actual_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, true, head_dim, 4, 4, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "segmented Gemma-style original-domain KVarN FlashAttention differs from materialized reference");
    }

    {
        constexpr int head_dim = 512;
        constexpr int n_q_prefill = 256;
        constexpr int n_q_heads = 8;
        constexpr int n_kv_heads = 1;
        constexpr int n_segments = 64;
        constexpr int segment_tokens = 256;
        constexpr int stage_groups = 19;
        constexpr float gemma_attn_scale = 1.0f;
        const std::vector<float> expected_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, false, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, false, 0, false, 0, gemma_attn_scale);
        const std::vector<float> actual_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, true, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, false, 0, false, 0, gemma_attn_scale);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "deep D512 original-domain KVarN8 FlashAttention differs from materialized reference");
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_q_prefill = 256;
        constexpr int n_q_heads = 32;
        constexpr int n_kv_heads = 16;
        constexpr int n_segments = 8;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 1280;
        constexpr int stage_groups = 3;
        const std::vector<float> expected_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, false, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window);
        const std::vector<float> actual_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, true, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "segmented Gemma-style SWA original-domain KVarN FlashAttention differs from materialized reference");
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_q_prefill = 256;
        constexpr int n_q_heads = 32;
        constexpr int n_kv_heads = 16;
        constexpr int n_segments = 64;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 1280;
        constexpr int stage_groups = 2;
        constexpr int tail_groups = 2;
        constexpr float gemma_attn_scale = 1.0f;
        const std::vector<float> expected_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, false, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window, false, 0,
                gemma_attn_scale, false, tail_groups, true);
        const std::vector<float> actual_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, true, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window, false, 0,
                gemma_attn_scale, false, tail_groups, true);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "tail-equals-stage SWA original-domain KVarN FlashAttention differs from materialized reference");
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_kv_heads = 16;
        constexpr int n_segments = 64;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 1280;
        constexpr int stage_groups = 3;
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, false, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "wrapped SWA D256 KVarN8 K roundtrip differs from original-domain input",
                true, n_swa_window, true);
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, true, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "wrapped SWA D256 KVarN8 V roundtrip differs from original-domain input",
                true, n_swa_window, true);
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_kv_heads = 16;
        constexpr int n_segments = 64;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 1280;
        constexpr int stage_groups = 2;
        constexpr int tail_groups = 2;
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, false, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "tail-equals-stage SWA D256 KVarN8 K roundtrip differs from original-domain input",
                true, n_swa_window, true, tail_groups);
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, true, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "tail-equals-stage SWA D256 KVarN8 V roundtrip differs from original-domain input",
                true, n_swa_window, true, tail_groups);
    }

    {
        // stage_groups=4 gives tail_groups=3, which is the direct SWA safety guard.
        constexpr int head_dim = 256;
        constexpr int n_kv_heads = 4;
        constexpr int n_segments = 8;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 768;
        constexpr int stage_groups = 4;
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 4, false, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-1f,
                "direct SWA D256 KVarN4 K roundtrip differs from original-domain input",
                true, n_swa_window, true);
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 4, true, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-1f,
                "direct SWA D256 KVarN4 V roundtrip differs from original-domain input",
                true, n_swa_window, true);
    }

    {
        constexpr int head_dim = 512;
        constexpr int n_kv_heads = 1;
        constexpr int n_segments = 64;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 1280;
        constexpr int stage_groups = 3;
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, false, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "wrapped SWA D512 KVarN8 K roundtrip differs from original-domain input",
                true, n_swa_window, true);
        require_segmented_raw_roundtrip(gpu_backend, head_dim, 8, true, n_kv_heads,
                n_segments, segment_tokens, stage_groups, 3e-2f,
                "wrapped SWA D512 KVarN8 V roundtrip differs from original-domain input",
                true, n_swa_window, true);
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_q_prefill = 256;
        constexpr int n_q_heads = 32;
        constexpr int n_kv_heads = 16;
        constexpr int n_segments = 64;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 1280;
        constexpr int n_swa_visible = 1024;
        constexpr int stage_groups = 3;
        constexpr float gemma_attn_scale = 1.0f;
        const std::vector<float> expected_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, false, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window, true, n_swa_visible, gemma_attn_scale);
        const std::vector<float> actual_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, true, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window, true, n_swa_visible, gemma_attn_scale);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "wrapped Gemma-style SWA original-domain KVarN FlashAttention differs from materialized reference");
    }

    {
        constexpr int head_dim = 512;
        constexpr int n_q_prefill = 128;
        constexpr int n_q_heads = 4;
        constexpr int n_kv_heads = 1;
        constexpr int n_segments = 8;
        constexpr int segment_tokens = 256;
        constexpr int n_swa_window = 768;
        constexpr int stage_groups = 3;
        constexpr float gemma_attn_scale = 1.0f;
        const std::vector<float> expected_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, false, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window, false, 0, gemma_attn_scale, true);
        const std::vector<float> actual_prefill = test_native_flash_attention_segmented_output(
                gpu_backend, true, head_dim, 8, 8, n_q_prefill, n_q_heads, n_kv_heads,
                n_segments, segment_tokens, stage_groups, true, n_swa_window, false, 0, gemma_attn_scale, true);
        require_close_f32_rmse(actual_prefill, expected_prefill, 1e-2f,
                "scrambled SWA D512 original-domain KVarN FlashAttention differs from materialized reference");
    }

    {
        constexpr int head_dim = 256;
        constexpr int n_q_decode = 1;
        constexpr int n_q_heads = 6;
        constexpr int n_kv_heads = 1;
        constexpr int stage_groups = 5;
        for (int bits_k : { 2, 3, 4, 5, 6, 8 }) {
            for (int bits_v : { 2, 3, 4, 5, 6, 8 }) {
                constexpr int n_kv_matrix = 1024;
                const std::vector<float> expected_decode = test_native_flash_attention_output(
                        cpu_backend, false, false, head_dim, bits_k, bits_v, n_q_decode, n_q_heads, n_kv_heads, n_kv_matrix, stage_groups);
                const std::vector<float> actual_decode = test_native_flash_attention_output(
                        gpu_backend, true, true, head_dim, bits_k, bits_v, n_q_decode, n_q_heads, n_kv_heads, n_kv_matrix, stage_groups);
                require_close_f32_rmse(actual_decode, expected_decode, 1e-2f,
                        "single-token GQA mixed-bit native KVarN FlashAttention output differs from CPU reference decode");
            }
        }

        constexpr int n_kv_deep = 4096;
        const std::vector<float> expected_deep = test_native_flash_attention_output(
                cpu_backend, false, false, head_dim, 4, 4, n_q_decode, n_q_heads, n_kv_heads, n_kv_deep, stage_groups);
        const std::vector<float> actual_deep = test_native_flash_attention_output(
                gpu_backend, true, true, head_dim, 4, 4, n_q_decode, n_q_heads, n_kv_heads, n_kv_deep, stage_groups);
        require_close_f32_rmse(actual_deep, expected_deep, 1e-2f,
                "single-token GQA deep K4/V4 native KVarN FlashAttention output differs from CPU reference decode");
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);
}

static void test_store_paths_gpu() {
    ggml_backend_t gpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_GPU, false);
    if (gpu_backend == nullptr) {
        return;
    }
    ggml_backend_t cpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_CPU, true);

    for (int bits : { 2, 3, 4, 5, 6, 8 }) {
        for (bool value : { false, true }) {
            const std::vector<ggml_fp16_t> cuda_output = test_store_reference_output(
                    gpu_backend, bits, value, 2, 2, 385, 64);
            const std::vector<ggml_fp16_t> cpu_output = test_store_reference_output(
                    cpu_backend, bits, value, 2, 2, 385, 64);
            require_close_f16_rmse(cuda_output, cpu_output, 1e-1f, "KVarN CUDA store output differs from CPU reference");
        }
    }

    for (int bits : { 2, 3, 4, 5, 6, 8 }) {
        for (bool value : { false, true }) {
            const std::vector<ggml_fp16_t> cuda_output = test_store_reference_output(
                    gpu_backend, bits, value, 1, 2, 512, 200);
            const std::vector<ggml_fp16_t> cpu_output = test_store_reference_output(
                    cpu_backend, bits, value, 1, 2, 512, 200);
            require_close_f16_rmse(cuda_output, cpu_output, 1e-1f, "KVarN CUDA split workspace store output differs from CPU reference");
        }
    }

    for (bool value : { false, true }) {
        const std::vector<ggml_fp16_t> workspace_output = test_store_segmented_output(
                gpu_backend, 4, value, 16, 2, 4096, 512, 7);
        const std::vector<ggml_fp16_t> fallback_output = test_store_segmented_output(
                gpu_backend, 4, value, 16, 2, 4096, 256, 7);
        require(workspace_output == fallback_output,
                "KVarN CUDA D256 workspace ubatch store output is not segmentation-independent");
    }

    for (int bits : { 2, 3, 4, 5, 6, 8 }) {
        for (bool value : { false, true }) {
            const std::vector<ggml_fp16_t> cuda_output = test_store_reference_output(
                    gpu_backend, bits, value, 1, 2, 16, 504);
            const std::vector<ggml_fp16_t> cpu_output = test_store_reference_output(
                    cpu_backend, bits, value, 1, 2, 16, 504);
            require_close_f16_rmse(cuda_output, cpu_output, 1e-1f, "KVarN CUDA direct-flush store output differs from CPU reference");
        }
    }

    for (bool value : { false, true }) {
        const std::vector<ggml_fp16_t> cuda_output = test_store_reference_output(
                gpu_backend, 4, value, 1, 2, 385, 64, true, false);
        const std::vector<ggml_fp16_t> cpu_output = test_store_reference_output(
                cpu_backend, 4, value, 1, 2, 385, 64, true, false);
        require_close_f16_rmse(cuda_output, cpu_output, 1e-1f, "KVarN CUDA stale workspace hint fallback output differs from CPU reference");
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);
}

// Validates the GPU/CPU rotated reference decode: emitting K_rot (skip the
// inverse-WHT) and then applying R on the host must reproduce normal original-
// domain decode (X_orig = R*K_rot). The same store feeds both, so this
// holds for sink/record/stage groups alike.
static void test_rotated_decode_transform_consistency(enum ggml_backend_dev_type device_type, bool required) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    const int bits = 4;
    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "rotated parity: failed to init ctx");

    constexpr int n_stream = 1;
    constexpr int kv_size = 512;
    constexpr int n_groups_per_stream = kv_size / 128;
    constexpr int n_tokens = 385;
    constexpr int n_heads = 2;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 384 * n_stream);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream * n_stream);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false, 3);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "rotated parity: failed to allocate tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            for (int d = 0; d < 128; ++d) {
                input[(size_t(t) * n_heads + h) * 128 + d] =
                    std::sin(float(d) * 0.07f + float(h) * 0.13f) + std::cos(float(t) * 0.037f) +
                    float((d * 13 + h * 7 + t * 17) % 31 - 15) * 0.01f;
            }
        }
    }
    std::vector<int64_t> idx(n_tokens);
    for (int t = 0; t < n_tokens; ++t) idx[t] = t;
    std::vector<uint8_t> zeros(std::max(ggml_nbytes(stage), ggml_nbytes(records)), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, zeros.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, zeros.data(), 0, ggml_nbytes(records));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "rotated parity: graph compute failed");

    std::vector<ggml_fp16_t> orig_h = test_kvarn_reference_decode(
            records, stored, idx, n_tokens, 0, n_stream, bits, false, 3);
    std::vector<ggml_fp16_t> rot_h = test_kvarn_reference_decode(
            records, stored, idx, n_tokens, 0, n_stream, bits, false, 3, true);

    double sum_sq = 0.0;
    double max_diff = 0.0;
    size_t count = 0;
    std::array<float, 128> buf;
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            const size_t base = (size_t(t) * n_heads + h) * 128;
            for (int d = 0; d < 128; ++d) buf[d] = ggml_fp16_to_fp32(rot_h[base + d]);
            llama_kvarn_hadamard_128(buf.data());
            for (int d = 0; d < 128; ++d) {
                const float ref = ggml_fp16_to_fp32(orig_h[base + d]);
                require(std::isfinite(ref) && std::isfinite(buf[d]), "rotated decode transform produced non-finite value");
                const double diff = double(ref) - double(buf[d]);
                sum_sq += diff * diff;
                max_diff = std::max(max_diff, std::fabs(diff));
                ++count;
            }
        }
    }
    const double rmse = std::sqrt(sum_sq / std::max<size_t>(count, 1));
    require(rmse <= 5e-4, "rotated decode inverse-WHT RMSE too high");
    require(max_diff <= 2e-3, "rotated decode inverse-WHT max error too high");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// W2 dynamic-stage-depth test: verifies the store path honors
// stage_groups carried in op_params[7] instead of the legacy three-slot stride.
// Writes 768 tokens (6 groups) through a 5-deep stage (tail_groups=4)
// and checks reconstruction of sink, compressed, previous-tail, and live-tail
// groups. Writing 6 groups with stage_groups=5 forces group 5 to reuse transient
// slot 1, flushing the completed group 1 to records — exercising the dynamic
// tail_groups flush predicate (group > tail_groups instead of group > 2) and
// the slot reuse modulo (1 + ((group - 1) % tail_groups)).
static void test_cache_ops_dynamic_stage(enum ggml_backend_dev_type device_type, bool required, int bits) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "dynamic-stage: failed to initialize ggml context");

    constexpr int n_tokens = 768;     // 6 complete groups (0..5)
    constexpr int n_heads   = 1;
    constexpr int stage_groups = 5;   // tail_groups = 4
    constexpr int tail_groups  = stage_groups - 1;
    // 8 record groups per stream (kv_size = 1024) — enough to hold 6 groups with
    // room for the flush ring to grow without collision.
    constexpr int n_groups_per_stream = 8;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage    = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups);
    ggml_tensor * records  = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false, stage_groups);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "dynamic-stage: failed to allocate tensors");

    std::vector<float> input(128 * n_heads * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            input[t * 128 + d] =
                std::sin(float(d) * 0.071f) +
                std::cos(float(t) * 0.037f) +
                float((d * 13 + t * 17) % 31 - 15) * 0.01f;
        }
    }
    std::vector<int64_t> idx(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        idx[i] = i;
    }
    std::vector<uint8_t> zeros(ggml_nbytes(stage) + ggml_nbytes(records), 0);

    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, zeros.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, zeros.data(), 0, ggml_nbytes(records));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "dynamic-stage: graph compute failed");

    const std::vector<float> output = test_kvarn_reference_decode_f32(
            records, stored, idx, n_tokens, 0, 1, bits, false, stage_groups);

    // Group decomposition for n_tokens=768, stage_groups=5, tail_groups=4:
    //   group 0: tokens   0..127  (permanent sink, slot 0)
    //   group 1: tokens 128..255  (transient slot 1, flushed to records when group 5 begins)
    //   group 2: tokens 256..383  (transient slot 2)
    //   group 3: tokens 384..511  (transient slot 3)
    //   group 4: tokens 512..639  (transient slot 4)
    //   group 5: tokens 640..767  (reuses transient slot 1, group 1 must be in records)
    // live_group after processing = 5. Stage holds groups 2..5 (tail_groups=4 groups);
    // group 1 comes from records; group 0 is the sink.
    double sink_error = 0.0;          // group 0
    double compressed_error = 0.0;    // group 1 (flushed to records)
    double stage_transit_error = 0.0; // groups 2, 3, 4, 5 (in stage transient slots)
    for (int t = 0; t < n_tokens; ++t) {
        const int group = t / 128;
        for (int d = 0; d < 128; ++d) {
            const double diff = double(input[t * 128 + d]) - double(output[t * 128 + d]);
            if (group == 0) {
                sink_error += diff * diff;
            } else if (group == 1) {
                compressed_error += diff * diff;
            } else {
                stage_transit_error += diff * diff;
            }
        }
    }
    sink_error          = std::sqrt(sink_error          / (128 * 128));
    compressed_error   = std::sqrt(compressed_error    / (128 * 128));
    stage_transit_error = std::sqrt(stage_transit_error / (128 * 512));
    require(sink_error < 0.01,          "dynamic-stage: sink reconstruction error too high");
    require(compressed_error < 0.25,   "dynamic-stage: compressed reconstruction error too high");
    require(stage_transit_error < 0.01, "dynamic-stage: in-stage transient reconstruction error too high");

    // Group 1 must have been flushed to records when group 5 began (reusing slot 1).
    std::vector<uint8_t> record_data(ggml_nbytes(records));
    ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());
    const size_t group1_off = size_t(1) * record_bytes;
    require(std::any_of(record_data.begin() + ptrdiff_t(group1_off),
                        record_data.begin() + ptrdiff_t(group1_off + record_bytes),
                        [](uint8_t v) { return v != 0; }),
            "dynamic-stage: completed group 1 was not flushed to records");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// W2 unaligned-start persistence test: first store a prefix into stage/records,
// then run a second store whose first absolute index is inside an already-open
// non-sink group. Reference decode covers the whole logical range, while the
// live-group input is only the second store's indices, matching the production
// non-SWA contract.
static void test_unaligned_start(enum ggml_backend_dev_type device_type, bool required,
                                  int start_offset, int n_tokens, int stage_groups) {
    ggml_backend_t backend = init_test_backend(device_type, required);
    if (backend == nullptr) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "unaligned-start: failed to initialize ggml context");

    // Start the second store inside group 1 so the unaligned group is non-sink.
    // For stage_groups 3 and 5, these cases also cross enough boundaries to
    // reuse a transient slot and flush group 1 from the second store.
    const int second_start = 128 + start_offset;
    const int total_tokens = second_start + n_tokens;
    constexpr int n_heads = 1;
    const int tail_groups = stage_groups - 1;
    const int bits = 4;
    const int last_group = (total_tokens - 1) / 128;
    const int n_groups_per_stream = std::max(8, last_group + 1);
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current_prefix = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, second_start);
    ggml_tensor * indices_prefix = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, second_start);
    ggml_tensor * current        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices        = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage    = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups);
    ggml_tensor * records  = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream);

    ggml_tensor * stored_prefix = ggml_kvarn_store(ctx, current_prefix, indices_prefix, stage, records, bits, 16, false, stage_groups);
    stored_prefix->op_params[3] = second_start;
    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stored_prefix, records, bits, 16, false, stage_groups);
    stored->op_params[3] = n_tokens;
    // Non-SWA reference decode uses output cell t as absolute position t. The indices
    // tensor is still the second store's index range, which supplies the current
    // live group exactly as production graph construction does.

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "unaligned-start: failed to allocate tensors");

    auto sample = [](int abs_pos, int d) {
        return std::sin(float(d) * 0.071f) +
            std::cos(float(abs_pos) * 0.037f) +
            float((d * 13 + abs_pos * 17) % 31 - 15) * 0.01f;
    };
    std::vector<float> prefix_input(128 * n_heads * second_start);
    std::vector<float> input(128 * n_heads * n_tokens);
    std::vector<float> expected(128 * n_heads * total_tokens);
    for (int t = 0; t < total_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            const float value = sample(t, d);
            expected[t * 128 + d] = value;
            if (t < second_start) {
                prefix_input[t * 128 + d] = value;
            } else {
                input[(t - second_start) * 128 + d] = value;
            }
        }
    }
    std::vector<int64_t> idx_prefix(second_start);
    for (int i = 0; i < second_start; ++i) {
        idx_prefix[i] = int64_t(i);
    }
    std::vector<int64_t> idx(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        idx[i] = int64_t(second_start + i);
    }
    require(idx.front() == second_start && (idx.front() % 128) == start_offset,
            "unaligned-start: second store did not start at the requested unaligned absolute position");
    std::vector<uint8_t> stage_zeros(ggml_nbytes(stage), 0);
    std::vector<uint8_t> record_zeros(ggml_nbytes(records), 0);

    ggml_backend_tensor_set(current_prefix, prefix_input.data(), 0, ggml_nbytes(current_prefix));
    ggml_backend_tensor_set(indices_prefix, idx_prefix.data(), 0, ggml_nbytes(indices_prefix));
    ggml_backend_tensor_set(current, input.data(), 0, ggml_nbytes(current));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    ggml_backend_tensor_set(stage, stage_zeros.data(), 0, stage_zeros.size());
    ggml_backend_tensor_set(records, record_zeros.data(), 0, record_zeros.size());

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "unaligned-start: graph compute failed");

    const std::vector<float> output = test_kvarn_reference_decode_f32(
            records, stored, idx, total_tokens, 0, 1, bits, false, stage_groups);

    // Verify reconstruction across prefix and second store, including the group
    // partially filled by the prefix and completed by the second store.
    double mse = 0.0;
    double max_diff = 0.0;
    for (int t = 0; t < total_tokens; ++t) {
        for (int d = 0; d < 128; ++d) {
            const double diff = double(expected[t * 128 + d]) - double(output[t * 128 + d]);
            mse += diff * diff;
            max_diff = std::max(max_diff, std::fabs(diff));
        }
    }
    const double rmse = std::sqrt(mse / double(total_tokens * 128));
    if (!std::isfinite(rmse) || rmse >= 0.30) {
        std::fprintf(stderr, "unaligned-start: reconstruction RMSE too high (start=%d abs=%d n=%d sg=%d rmse=%g max=%g)\n",
                start_offset, second_start, n_tokens, stage_groups, rmse, max_diff);
        require(false, "unaligned-start: reconstruction RMSE too high");
    }
    if (max_diff >= 2.0) {
        std::fprintf(stderr, "unaligned-start: max reconstruction error too high (start=%d abs=%d n=%d sg=%d max=%g)\n",
                start_offset, second_start, n_tokens, stage_groups, max_diff);
        require(false, "unaligned-start: max reconstruction error too high");
    }

    if (last_group > tail_groups) {
        std::vector<uint8_t> record_data(ggml_nbytes(records));
        ggml_backend_tensor_get(records, record_data.data(), 0, record_data.size());
        const int flushed_group = last_group - tail_groups;
        const size_t off = size_t(flushed_group) * size_t(record_bytes);
        require(std::any_of(record_data.begin() + ptrdiff_t(off),
                            record_data.begin() + ptrdiff_t(off + record_bytes),
                            [](uint8_t v) { return v != 0; }),
                "unaligned-start: second store did not flush the reused transient slot");
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

int main() {
    ggml_backend_load_all();

    test_type_table();
    test_stage_policy();
    test_tile_layout();
    test_head_dimension_slicing();
    test_runtime_validation();
    test_remove_policy();
    test_pack_roundtrip(2);
    test_pack_roundtrip(3);
    test_pack_roundtrip(4);
    test_pack_roundtrip(5);
    test_pack_roundtrip(6);
    test_pack_roundtrip(8);
    test_hadamard_roundtrip();
    test_rotated_domain_equivalence();
    for (int head_width : { 128, 256, 512 }) {
        test_kvarn_wht_op(GGML_BACKEND_DEVICE_TYPE_CPU, true,  head_width);
        test_kvarn_wht_op(GGML_BACKEND_DEVICE_TYPE_GPU, false, head_width);
    }

    for (size_t i = 0; i < llama_kvarn_type_count(); ++i) {
        const llama_kvarn_type type = (llama_kvarn_type) i;
        if (type != LLAMA_KVARN_TYPE_DISABLED) {
            test_tile_quantization(type);
        }
    }

    for (int bits : { 3, 5, 6, 8 }) {
        test_cache_ops(GGML_BACKEND_DEVICE_TYPE_CPU, true, bits);
        test_cache_ops(GGML_BACKEND_DEVICE_TYPE_GPU, false, bits);
    }
    // W2 dynamic stage-depth coverage: stage_groups=5 (tail_groups=4).
    test_cache_ops_dynamic_stage(GGML_BACKEND_DEVICE_TYPE_CPU, true, 4);
    test_cache_ops_dynamic_stage(GGML_BACKEND_DEVICE_TYPE_GPU, false, 4);
    // W2 unaligned-start coverage: first persist a prefix, then run a second
    // store whose first index is 128 + {1, 64, 127}. The 256/512 cases force a
    // transient slot reuse/flush from the second store; all cases decode
    // the full prefix + second-store range from the same stage/records.
    for (int start : { 1, 64, 127 }) {
        test_unaligned_start(GGML_BACKEND_DEVICE_TYPE_CPU, true,  start, 256, 3);
        test_unaligned_start(GGML_BACKEND_DEVICE_TYPE_GPU, false, start, 256, 3);
        test_unaligned_start(GGML_BACKEND_DEVICE_TYPE_CPU, true,  start, 512, 5);
        test_unaligned_start(GGML_BACKEND_DEVICE_TYPE_GPU, false, start, 512, 5);
        test_unaligned_start(GGML_BACKEND_DEVICE_TYPE_CPU, true,  start, 513, 6);
        test_unaligned_start(GGML_BACKEND_DEVICE_TYPE_GPU, false, start, 513, 6);
    }
    test_cache_ops_multi_stream(GGML_BACKEND_DEVICE_TYPE_CPU, true, 6);
    test_cache_ops_multi_stream(GGML_BACKEND_DEVICE_TYPE_GPU, false, 6);
    test_cache_ops_swa(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    test_cache_ops_swa(GGML_BACKEND_DEVICE_TYPE_GPU, false); // CUDA SWA ring parity
    test_store_paths_gpu();
    test_native_flash_attention_support_gates();
    test_native_flash_attention_gpu();
    test_rotated_decode_transform_consistency(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    test_rotated_decode_transform_consistency(GGML_BACKEND_DEVICE_TYPE_GPU, false);

    std::printf("test-kvarn: all tests OK\n");
    return 0;
}
