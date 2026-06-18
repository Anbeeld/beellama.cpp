#include "llama-kvarn.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    require(llama_kvarn_head_slices(384) == 3, "384-dim head should use three KVarN slices");
    require(llama_kvarn_head_slices(64)  == 0, "64-dim head is not KVarN slice-compatible");
    require(llama_kvarn_head_slices(513) == 0, "non-128-multiple head is not KVarN slice-compatible");
}

static void test_runtime_validation() {
    llama_kvarn_runtime_requirements supported = {};
    supported.attention_supported = true;
    supported.head_dims_supported = true;
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
// materialize path, using only the CPU reference quant/dequant. Let R = the
// normalized WHT-128 (symmetric involution: R^2 = I, verified separately by
// test_hadamard_roundtrip). KVarN stores K_rot = R*K, V_rot = R*V; materialize
// reconstructs X_orig = R*dequant(record). Rotated-domain attention skips that
// inverse-WHT and instead rotates the query / inverse-rotates the output:
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

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false);
    ggml_tensor * materialized = ggml_kvarn_materialize(ctx, records, stored, indices, n_tokens, 0, 1, bits, false);
    if (device_type == GGML_BACKEND_DEVICE_TYPE_GPU) {
        // Regression guard: materialize live groups are per-execution state and
        // must be derived from the current indices tensor, not frozen op params.
        materialized->op_params[4] = 0;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);
    ggml_build_forward_expand(graph, materialized);

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

    std::vector<ggml_fp16_t> output_f16(ggml_nelements(materialized));
    std::vector<float> output(output_f16.size());
    ggml_backend_tensor_get(materialized, output_f16.data(), 0, ggml_nbytes(materialized));
    ggml_fp16_to_fp32_row(output_f16.data(), output.data(), output.size());

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

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false);
    ggml_tensor * materialized = ggml_kvarn_materialize(
            ctx, records, stored, indices, n_tokens_per_stream, 0, n_stream, bits, false);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);
    ggml_build_forward_expand(graph, materialized);

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

    std::vector<ggml_fp16_t> output_f16(ggml_nelements(materialized));
    std::vector<float> output(output_f16.size());
    ggml_backend_tensor_get(materialized, output_f16.data(), 0, ggml_nbytes(materialized));
    ggml_fp16_to_fp32_row(output_f16.data(), output.data(), output.size());

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
// slots are reused, then materialize the live window. The two newest tiles come
// from fp16 staging (near-lossless); the older in-window tiles come from records
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
    constexpr int gps = 6;                 // 6 ring tiles (768 record capacity)
    constexpr int n_tiles = 10;            // tiles 0..9 -> ring wraps (tile 6 reuses tile 0's slot)
    constexpr int n_tokens = n_tiles * 128;
    constexpr int window_base = 6 * 128;   // window covers tiles 6..9
    constexpr int n_kv = 4 * 128;          // tiles 6,7 sealed (slots 0,1); tiles 8,9 live in staging
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * mat_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv); // per-cell absolute positions
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 384);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, gps);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false);
    stored->op_params[4] = 1; // SWA ring store
    ggml_tensor * materialized = ggml_kvarn_materialize(ctx, records, stored, mat_indices, n_kv, 0, 1, bits, false);
    materialized->op_params[6] = 1; // SWA ring materialize (per-cell positions)

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);
    ggml_build_forward_expand(graph, materialized);

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
    ggml_backend_tensor_set(mat_indices, mat_idx.data(), 0, ggml_nbytes(mat_indices));
    ggml_backend_tensor_set(stage, zeros.data(), 0, ggml_nbytes(stage));
    ggml_backend_tensor_set(records, zeros.data(), 0, ggml_nbytes(records));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "swa: graph compute failed");

    std::vector<ggml_fp16_t> output_f16(ggml_nelements(materialized));
    std::vector<float> output(output_f16.size());
    ggml_backend_tensor_get(materialized, output_f16.data(), 0, ggml_nbytes(materialized));
    ggml_fp16_to_fp32_row(output_f16.data(), output.data(), output.size());

    double sealed_error = 0.0; // tiles 6,7 -> reused ring slots 0,1
    double live_error = 0.0;   // tiles 8,9 -> fp16 staging
    for (int cell = 0; cell < n_kv; ++cell) {
        const int abs_pos = window_base + cell;
        for (int d = 0; d < 128; ++d) {
            const double diff = double(input[abs_pos * 128 + d]) - double(output[cell * 128 + d]);
            if (cell < 256) {
                sealed_error += diff * diff;
            } else {
                live_error += diff * diff;
            }
        }
    }
    sealed_error = std::sqrt(sealed_error / (256 * 128));
    live_error = std::sqrt(live_error / (256 * 128));
    require(sealed_error < 0.25, "swa: sealed (wrapped) tile reconstruction error too high");
    require(live_error < 0.02, "swa: live tail reconstruction error too high");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static std::vector<ggml_fp16_t> test_materialize_output(
        ggml_backend_t backend,
        int            bits,
        bool           value,
        int            n_stream,
        int            n_heads,
        int            n_tokens_per_stream,
        int            start_idx,
        bool           discontinuous_indices = false,
        bool           seed_stage = true) {
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize materialize parity context");

    const int n_kv = start_idx + n_tokens_per_stream + (discontinuous_indices ? 1 : 0);
    const int n_groups_per_stream = std::max(4, (n_kv + 127) / 128);
    const int n_tokens = n_tokens_per_stream * n_stream;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 384 * n_stream);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream * n_stream);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, value);
    stored->op_params[3] = n_tokens_per_stream;
    ggml_tensor * materialized = ggml_kvarn_materialize(ctx, records, stored, indices, n_kv, 0, n_stream, bits, value);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);
    ggml_build_forward_expand(graph, materialized);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate materialize parity tensors");

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

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "materialize path graph compute failed");

    std::vector<ggml_fp16_t> output(ggml_nelements(materialized));
    ggml_backend_tensor_get(materialized, output.data(), 0, ggml_nbytes(materialized));

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

static void test_store_paths_gpu() {
    ggml_backend_t gpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_GPU, false);
    if (gpu_backend == nullptr) {
        return;
    }
    ggml_backend_t cpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_CPU, true);

    for (int bits : { 2, 3, 4, 5, 6, 8 }) {
        for (bool value : { false, true }) {
            const std::vector<ggml_fp16_t> cuda_output = test_materialize_output(
                    gpu_backend, bits, value, 2, 2, 385, 64);
            const std::vector<ggml_fp16_t> cpu_output = test_materialize_output(
                    cpu_backend, bits, value, 2, 2, 385, 64);
            require_close_f16_rmse(cuda_output, cpu_output, 1e-1f, "KVarN CUDA store output differs from CPU reference");
        }
    }

    for (int bits : { 2, 3, 4, 5, 6, 8 }) {
        for (bool value : { false, true }) {
            const std::vector<ggml_fp16_t> cuda_output = test_materialize_output(
                    gpu_backend, bits, value, 1, 2, 512, 200);
            const std::vector<ggml_fp16_t> cpu_output = test_materialize_output(
                    cpu_backend, bits, value, 1, 2, 512, 200);
            require_close_f16_rmse(cuda_output, cpu_output, 1e-1f, "KVarN CUDA split workspace store output differs from CPU reference");
        }
    }

    for (int bits : { 2, 3, 4, 5, 6, 8 }) {
        for (bool value : { false, true }) {
            const std::vector<ggml_fp16_t> cuda_output = test_materialize_output(
                    gpu_backend, bits, value, 1, 2, 16, 504);
            const std::vector<ggml_fp16_t> cpu_output = test_materialize_output(
                    cpu_backend, bits, value, 1, 2, 16, 504);
            require_close_f16_rmse(cuda_output, cpu_output, 1e-1f, "KVarN CUDA direct-flush store output differs from CPU reference");
        }
    }

    for (bool value : { false, true }) {
        const std::vector<ggml_fp16_t> cuda_output = test_materialize_output(
                gpu_backend, 4, value, 1, 2, 385, 64, true, false);
        const std::vector<ggml_fp16_t> cpu_output = test_materialize_output(
                cpu_backend, 4, value, 1, 2, 385, 64, true, false);
        require_close_f16_rmse(cuda_output, cpu_output, 1e-1f, "KVarN CUDA stale workspace hint fallback output differs from CPU reference");
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);
}

static void test_materialize_paths_gpu() {
    ggml_backend_t gpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_GPU, false);
    if (gpu_backend == nullptr) {
        return;
    }
    ggml_backend_t cpu_backend = init_test_backend(GGML_BACKEND_DEVICE_TYPE_CPU, true);

    for (int bits : { 2, 3, 4, 5, 6, 8 }) {
        for (bool value : { false, true }) {
            const std::vector<ggml_fp16_t> cuda_output = test_materialize_output(
                    gpu_backend, bits, value, 2, 2, 385, 64);
            const std::vector<ggml_fp16_t> cpu_output = test_materialize_output(
                    cpu_backend, bits, value, 2, 2, 385, 64);
            require_close_f16_rmse(cuda_output, cpu_output, 1e-1f, "KVarN CUDA materialize output differs from CPU reference");
        }
    }

    for (bool value : { false, true }) {
        const std::vector<ggml_fp16_t> cuda_output = test_materialize_output(
                gpu_backend, 4, value, 1, 2, 17, 504);
        const std::vector<ggml_fp16_t> cpu_output = test_materialize_output(
                cpu_backend, 4, value, 1, 2, 17, 504);
        require_close_f16_rmse(cuda_output, cpu_output, 1e-1f, "KVarN CUDA materialize tail output differs from CPU reference");
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);
}

// Validates the GPU/CPU rotated materialize kernel: emitting K_rot (skip the
// inverse-WHT) and then applying R on the host must reproduce the normal
// materialize output (X_orig = R*K_rot). The same store feeds both, so this
// holds for sink/record/stage groups alike.
static void test_materialize_rotated_parity(enum ggml_backend_dev_type device_type, bool required) {
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

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false);
    ggml_tensor * m_orig = ggml_kvarn_materialize(ctx, records, stored, indices, n_tokens, 0, n_stream, bits, false);
    ggml_tensor * m_rot  = ggml_kvarn_materialize(ctx, records, stored, indices, n_tokens, 0, n_stream, bits, false);
    m_rot->op_params[5] = 1; // emit rotated-domain values (skip inverse-WHT)

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, m_orig);
    ggml_build_forward_expand(graph, m_rot);

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

    std::vector<ggml_fp16_t> orig_h(ggml_nelements(m_orig));
    std::vector<ggml_fp16_t> rot_h(ggml_nelements(m_rot));
    ggml_backend_tensor_get(m_orig, orig_h.data(), 0, ggml_nbytes(m_orig));
    ggml_backend_tensor_get(m_rot, rot_h.data(), 0, ggml_nbytes(m_rot));

    double max_abs = 0.0, max_diff = 0.0;
    std::array<float, 128> buf;
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            const size_t base = (size_t(t) * n_heads + h) * 128;
            for (int d = 0; d < 128; ++d) buf[d] = ggml_fp16_to_fp32(rot_h[base + d]);
            llama_kvarn_hadamard_128(buf.data());
            for (int d = 0; d < 128; ++d) {
                const float ref = ggml_fp16_to_fp32(orig_h[base + d]);
                max_abs  = std::max(max_abs, double(std::fabs(ref)));
                max_diff = std::max(max_diff, double(std::fabs(ref - buf[d])));
            }
        }
    }
    require(max_diff < 5e-2 * (1.0 + max_abs), "rotated materialize != inverse-WHT of normal materialize");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// W2 dynamic-stage-depth test: verifies the store/materialize path honors
// stage_groups carried in op_params[7] instead of the legacy three-slot stride.
// Writes 768 tokens (6 groups) through a 5-deep stage (tail_groups=4, n_ubatch=512)
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
    constexpr int stage_groups = 5;   // tail_groups = 4, n_ubatch = 512
    constexpr int tail_groups  = stage_groups - 1;
    // 8 record groups per stream (kv_size = 1024) — enough to hold 6 groups with
    // room for the flush ring to grow without collision.
    constexpr int n_groups_per_stream = 8;
    const int record_bytes = int(llama_kvarn_packed_bytes(128 * 128, bits) + 3 * 128 * sizeof(ggml_fp16_t));

    ggml_tensor * current  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage    = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_heads, 128 * stage_groups);
    ggml_tensor * records  = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, n_groups_per_stream);

    ggml_tensor * stored = ggml_kvarn_store(ctx, current, indices, stage, records, bits, 16, false);
    stored->op_params[7] = stage_groups; // dynamic stage depth
    ggml_tensor * materialized = ggml_kvarn_materialize(ctx, records, stored, indices, n_tokens, 0, 1, bits, false);
    materialized->op_params[7] = stage_groups;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stored);
    ggml_build_forward_expand(graph, materialized);

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

    std::vector<ggml_fp16_t> output_f16(ggml_nelements(materialized));
    std::vector<float> output(output_f16.size());
    ggml_backend_tensor_get(materialized, output_f16.data(), 0, ggml_nbytes(materialized));
    ggml_fp16_to_fp32_row(output_f16.data(), output.data(), output.size());

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

int main() {
    ggml_backend_load_all();

    test_type_table();
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
    // W2 dynamic stage-depth coverage: stage_groups=5 (n_ubatch=512).
    test_cache_ops_dynamic_stage(GGML_BACKEND_DEVICE_TYPE_CPU, true, 4);
    test_cache_ops_dynamic_stage(GGML_BACKEND_DEVICE_TYPE_GPU, false, 4);
    test_cache_ops_multi_stream(GGML_BACKEND_DEVICE_TYPE_CPU, true, 6);
    test_cache_ops_multi_stream(GGML_BACKEND_DEVICE_TYPE_GPU, false, 6);
    test_cache_ops_swa(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    test_cache_ops_swa(GGML_BACKEND_DEVICE_TYPE_GPU, false); // CUDA SWA ring parity
    test_store_paths_gpu();
    test_materialize_paths_gpu();
    test_materialize_rotated_parity(GGML_BACKEND_DEVICE_TYPE_CPU, true);
    test_materialize_rotated_parity(GGML_BACKEND_DEVICE_TYPE_GPU, false);

    std::printf("test-kvarn: all tests OK\n");
    return 0;
}
