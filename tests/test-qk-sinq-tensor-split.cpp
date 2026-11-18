#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

// Exercise the production model callback, rather than supplying a test-only
// split policy that could agree with itself while the model loader is wrong.
static void test_split(ggml_backend_dev_t first, ggml_backend_dev_t second, ggml_backend_dev_t cpu,
                       float a, float b, int layer, bool runtime, ggml_type type) {
    constexpr int64_t width = 1024;
    constexpr int64_t tokens = 3;
    std::vector<float> proportions(llama_max_devices(), 0.0f);
    proportions[0] = a;
    proportions[1] = b;
    auto params = llama_model_default_params();
    params.tensor_split = proportions.data();
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, params));
    CHECK(model);
    model->hparams.n_layer_all = 2;
    model->hparams.n_ff_arr.fill(width);
    model->hparams.n_head_arr.fill(8);
    model->hparams.n_head_kv_arr.fill(8);
    model->hparams.n_embd_head_k_full = 128;
    model->hparams.n_embd_head_v_full = 128;
    llama_meta_device_get_split_state_userdata userdata = {2, model.get()};
    ggml_backend_dev_t devices[] = {first, second};
    auto device = ggml_backend_meta_device(devices, 2, llama_meta_device_get_split_state, &userdata);
    auto backend = ggml_backend_dev_init(device, nullptr);
    auto host = ggml_backend_dev_init(cpu, nullptr);
    auto control = ggml_backend_dev_init(first, nullptr);
    CHECK(backend && host && control);

    ggml_init_params init = {4*1024*1024, nullptr, true};
    auto storage = ggml_init(init);
    auto ctx = ggml_init(init);
    CHECK(storage && ctx);
    auto up = ggml_new_tensor_2d(storage, type, width, width);
    auto down = ggml_new_tensor_2d(storage, type, width, width);
    const auto prefix = "blk." + std::to_string(layer) + ".";
    ggml_set_name(up, (prefix + "ffn_up.weight").c_str());
    ggml_set_name(down, (prefix + "ffn_down.weight").c_str());
    model->tensors_by_name.emplace_back(up->name, up);
    model->tensors_by_name.emplace_back(down->name, down);
    const auto scale_type = runtime ? GGML_TYPE_F32 : GGML_TYPE_F16;
    auto up_scale = ggml_new_tensor_1d(storage, scale_type, width);
    auto down_scale = ggml_new_tensor_1d(storage, scale_type, width);
    const auto suffix = runtime ? ".sinq_scale2.runtime_f32" : ".sinq_scale2";
    ggml_set_name(up_scale, (std::string(up->name) + suffix).c_str());
    ggml_set_name(down_scale, (std::string(down->name) + suffix).c_str());
    // Identical scale values deliberately occur in incompatible layouts.
    model->sinq_scales.emplace(up, up_scale);
    model->sinq_scales.emplace(down, down_scale);
    auto input = ggml_new_tensor_2d(storage, GGML_TYPE_F32, width, tokens);
    ggml_set_name(input, "sinq_test_input");

    auto up_state = llama_meta_device_get_split_state(up, &userdata);
    auto down_state = llama_meta_device_get_split_state(down, &userdata);
    auto us = llama_meta_device_get_split_state(up_scale, &userdata);
    auto ds = llama_meta_device_get_split_state(down_scale, &userdata);
    CHECK(up_state.axis == GGML_BACKEND_SPLIT_AXIS_1);
    CHECK(down_state.axis == GGML_BACKEND_SPLIT_AXIS_0);
    CHECK(us.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    CHECK(ds.axis == down_state.axis && ds.n_segments == down_state.n_segments);
    for (size_t s = 0; s < ds.n_segments; ++s) {
        CHECK(ds.nr[s] == down_state.nr[s]);
        for (size_t rank = 0; rank < 2; ++rank) {
            CHECK(ds.ne[2*s + rank] == down_state.ne[2*s + rank]);
        }
    }
    if (a == 0 || b == 0) { CHECK(ds.ne[0] == 0 || ds.ne[1] == 0); }
    else { CHECK(ds.ne[0] > 0 && ds.ne[1] > 0 && ds.ne[0] != ds.ne[1]); }

    // Loaded models materialize F32 scales before graph execution. Check the
    // serialized F16 names above, but do not exercise an unrelated meta CAST
    // path: numerical coverage below uses the production runtime cache.
    if (!runtime) {
        ggml_free(ctx);
        ggml_free(storage);
        ggml_backend_free(backend);
        ggml_backend_free(host);
        ggml_backend_free(control);
        return;
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(storage, backend);
    CHECK(buffer);
    std::vector<float> weights(width*width);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = float(int((i*7 + i/width*13) % 37) - 18)/1024.0f;
    }
    std::vector<uint8_t> packed(ggml_nbytes(up));
    CHECK(ggml_quantize_chunk(type, weights.data(), packed.data(), 0, width, width, nullptr) == packed.size());
    ggml_backend_tensor_set(up, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(down, packed.data(), 0, packed.size());
    std::vector<float> scales(width);
    std::vector<ggml_fp16_t> half(width);
    for (int64_t i = 0; i < width; ++i) {
        half[i] = ggml_fp32_to_fp16(0.25f + float((i*17)%31)/8.0f);
        scales[i] = ggml_fp16_to_fp32(half[i]);
    }
    for (auto scale : {up_scale, down_scale}) {
        ggml_backend_tensor_set(scale, runtime ? static_cast<const void *>(scales.data()) : half.data(), 0, ggml_nbytes(scale));
    }
    auto graph = ggml_new_graph(ctx);
    auto scale_up = runtime ? up_scale : ggml_cast(ctx, up_scale, GGML_TYPE_F32);
    auto scale_down = runtime ? down_scale : ggml_cast(ctx, down_scale, GGML_TYPE_F32);
    auto hidden = ggml_mul_mat(ctx, up, ggml_mul(ctx, input, scale_up));
    ggml_mul_mat_set_prec(hidden, GGML_PREC_F32);
    ggml_set_output(hidden);
    // The up projection produces the feature-sharded activation consumed by
    // down. Mirroring a synthetic down input would not model this contract.
    auto result = ggml_mul_mat(ctx, down, ggml_mul(ctx, hidden, scale_down));
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    // A consumer observes the synchronized result, not the partial matmul
    // representation retained in each device's reduction buffer.
    result = ggml_scale(ctx, result, 1.0f);
    ggml_build_forward_expand(graph, result);
    // Control graph executes the identical quantized projections on one device;
    // it never invokes the meta callback or any sharding code.
    auto control_ctx = ggml_init(init);
    CHECK(control_ctx);
    auto control_up = ggml_new_tensor_2d(control_ctx, type, width, width);
    auto control_down = ggml_new_tensor_2d(control_ctx, type, width, width);
    auto control_scale = ggml_new_tensor_1d(control_ctx, GGML_TYPE_F32, width);
    auto control_input = ggml_new_tensor_2d(control_ctx, GGML_TYPE_F32, width, tokens);
    auto control_hidden = ggml_mul_mat(control_ctx, control_up, ggml_mul(control_ctx, control_input, control_scale));
    ggml_mul_mat_set_prec(control_hidden, GGML_PREC_F32);
    ggml_set_output(control_hidden);
    auto control_result = ggml_mul_mat(control_ctx, control_down, ggml_mul(control_ctx, control_hidden, control_scale));
    ggml_mul_mat_set_prec(control_result, GGML_PREC_F32);
    auto control_graph = ggml_new_graph(control_ctx);
    ggml_build_forward_expand(control_graph, control_result);
    control_graph->uid = ggml_graph_next_uid();
    auto control_buffer = ggml_backend_alloc_ctx_tensors(control_ctx, control);
    CHECK(control_buffer);
    ggml_backend_tensor_set(control_up, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(control_down, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(control_scale, scales.data(), 0, ggml_nbytes(control_scale));
    ggml_backend_t backends[] = {backend, host};
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, 2048, false, true);
    CHECK(ggml_backend_sched_alloc_graph(sched, graph));
    std::vector<float> values(width*tokens), actual(values.size()), expected(values.size());
    // Reuse the same graph with unchanged and changed contents; a stale
    // projected activation or scale layout must not survive graph reuse.
    for (int run = 0; run < 4; ++run) {
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = float(int((i*11 + (run/2)*19)%43) - 21)/16.0f;
        }
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        ggml_backend_tensor_set(control_input, values.data(), 0, ggml_nbytes(control_input));
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_graph_compute(control, control_graph) == GGML_STATUS_SUCCESS);
        for (auto pair : {std::make_pair(hidden, control_hidden), std::make_pair(result, control_result)}) {
            ggml_backend_tensor_get(pair.first, actual.data(), 0, ggml_nbytes(pair.first));
            ggml_backend_tensor_get(pair.second, expected.data(), 0, ggml_nbytes(pair.second));
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > 2e-4f + 2e-4f*std::fabs(expected[i])) {
                    std::fprintf(stderr, "%s run=%d index=%zu expected=%.9g actual=%.9g\n",
                                 pair.first->src[0]->name, run, i, expected[i], actual[i]);
                    CHECK(false);
                }
            }
        }
    }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(control_buffer);
    ggml_free(control_ctx);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_free(storage);
    ggml_backend_free(backend);
    ggml_backend_free(host);
    ggml_backend_free(control);
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    CHECK(cpu);
    const bool cuda = argc == 2 && std::strcmp(argv[1], "cuda") == 0;
    auto first = cuda ? ggml_backend_dev_by_name("CUDA0") : cpu;
    auto second = cuda ? ggml_backend_dev_by_name("CUDA1") : cpu;
    if (!first || !second) {
        std::puts("two CUDA devices unavailable; skipping");
        return 77;
    }
    for (bool runtime : {true, false}) {
        for (int layer : {0, 1}) {
            for (auto weights : {std::make_pair(3.f, 1.f), std::make_pair(1.f, 0.f), std::make_pair(0.f, 1.f)}) {
                for (auto type : {GGML_TYPE_Q3_K,GGML_TYPE_Q4_K,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,GGML_TYPE_IQ4_XS}) {
                    test_split(first, second, cpu, weights.first, weights.second, layer, runtime,type);
                }
            }
        }
    }
    std::puts("SINQ tensor-split numerical checks passed");
    return 0;
}
