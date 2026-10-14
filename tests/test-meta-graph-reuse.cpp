#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

static decltype(ggml_backend_device_i::init_backend) init_original;
static std::map<ggml_backend_t, decltype(ggml_backend_i::graph_compute)> compute_original;
static std::vector<uint64_t> observed;
static std::vector<const ggml_tensor *> projected;

static ggml_status observe_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    observed.push_back(graph->uid);
    projected.push_back(graph->nodes[graph->n_nodes - 1]);
    return compute_original.at(backend)(backend, graph);
}

static ggml_backend_t observe_init(ggml_backend_dev_t dev, const char * params) {
    auto backend = init_original(dev, params);
    compute_original[backend] = backend->iface.graph_compute;
    backend->iface.graph_compute = observe_compute;
    return backend;
}

static ggml_backend_meta_split_state mirrored(const ggml_tensor * tensor, void *) {
    if (std::strncmp(tensor->name, "kvarn_", 6) == 0) {
        return {GGML_BACKEND_SPLIT_AXIS_1, {1, 1}, {1}, 1};
    }
    return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1 };
}

static void test_kvarn_reuse(ggml_backend_t backend, ggml_backend_t host) {
    ggml_init_params params = {4*1024*1024, nullptr, true};
    auto storage = ggml_init(params);
    auto ctx = ggml_init(params);
    // Two 128-wide K4 heads, one per device, and three live staging groups.
    auto records = ggml_new_tensor_3d(storage, GGML_TYPE_I8, 8960, 2, 1);
    auto stage = ggml_new_tensor_3d(storage, GGML_TYPE_F16, 128, 2, 384);
    auto indices = ggml_new_tensor_1d(storage, GGML_TYPE_I64, 1);
    auto a = ggml_new_tensor_3d(storage, GGML_TYPE_F32, 128, 2, 1);
    auto b = ggml_new_tensor_3d(storage, GGML_TYPE_F32, 128, 2, 1);
    ggml_set_name(records, "kvarn_records");
    ggml_set_name(stage, "kvarn_stage");
    ggml_set_name(a, "kvarn_a");
    ggml_set_name(b, "kvarn_b");
    auto buffer = ggml_backend_alloc_ctx_tensors(storage, backend);
    CHECK(buffer);
    ggml_backend_buffer_clear(buffer, 0);
    std::vector<float> values(256, 1.0f);
    ggml_backend_tensor_set(a, values.data(), 0, ggml_nbytes(a));
    values.assign(256, 2.0f);
    ggml_backend_tensor_set(b, values.data(), 0, ggml_nbytes(b));
    int64_t index = 0;
    ggml_backend_tensor_set(indices, &index, 0, sizeof(index));
    auto store = ggml_kvarn_store(ctx, a, indices, stage, records, 4, 16, false, 3);
    auto materialized = ggml_kvarn_materialize(ctx, records, store, indices, 1, 0, 1, 4, false, 3);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, materialized);
    ggml_backend_t backends[] = {backend, host};
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, 2048, false, true);
    CHECK(ggml_backend_sched_alloc_graph(sched, graph));
    auto run = [&](float expected) {
        observed.clear();
        projected.clear();
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        std::vector<ggml_fp16_t> row(256);
        ggml_backend_tensor_get(materialized, row.data(), 0, row.size()*sizeof(row[0]));
        // Staging and reconstructed output both round through F16.
        for (auto v : row) { CHECK(std::fabs(ggml_fp16_to_fp32(v) - expected) < 0.002f); }
        CHECK(observed.size() == 2 && observed[0] != 0 && observed[1] != 0);
        return projected;
    };
    auto first = run(1.0f);
    CHECK(run(1.0f) == first);
    store->src[0] = b;
    auto changed = run(2.0f);
    CHECK(changed != first);
    CHECK(run(2.0f) == changed);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_free(storage);
}

// Gemma4 full-attention MQA layers can assign the only KV head to one
// device. A singleton-head reshape must retain a head-axis split, including
// single-token decode where ggml_n_dims squeezes all trailing dimensions.
static ggml_backend_meta_split_state single_head_split(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_0, {512, 0}, {1}, 1};
}

static ggml_backend_meta_split_state single_head_rotation_split(const ggml_tensor * tensor, void *) {
    if (std::strcmp(tensor->name, "rotation_matrix") == 0 || std::strcmp(tensor->name, "rotation_indices") == 0) {
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    }
    return {GGML_BACKEND_SPLIT_AXIS_0, {tensor->ne[0], 0}, {1}, 1};
}

// The rotation is mirrored, while one rank owns every feature of the
// activation and the other rank has an empty shard.
static void test_single_head_rotation(ggml_backend_dev_t simple, ggml_backend_dev_t cpu) {
    ggml_backend_dev_t devices[] = {simple, simple};
    auto device = ggml_backend_meta_device(devices, 2, single_head_rotation_split, nullptr);
    auto backend = ggml_backend_dev_init(device, nullptr);
    auto host = ggml_backend_dev_init(cpu, nullptr);
    CHECK(backend && host);
    const int64_t features = 64;
    for (bool projection : {false, true}) {
    for (int64_t rotation : {int64_t(32), features}) {
    for (int64_t tokens : {int64_t(1), int64_t(3)}) {
        ggml_init_params params = {4*1024*1024, nullptr, true};
        auto storage = ggml_init(params);
        auto ctx = ggml_init(params);
        auto matrix = ggml_new_tensor_2d(storage, GGML_TYPE_F32, rotation, rotation);
        auto activation = ggml_new_tensor_2d(storage, GGML_TYPE_F32, features, tokens);
        auto cache = ggml_new_tensor_2d(storage, GGML_TYPE_F32, features, tokens);
        auto indices = ggml_new_tensor_1d(storage, GGML_TYPE_I64, tokens);
        ggml_set_name(matrix, projection ? "projection_matrix" : "rotation_matrix");
        ggml_set_name(activation, "rotation_activation");
        ggml_set_name(cache, "rotation_cache");
        ggml_set_name(indices, "rotation_indices");
        auto buffer = ggml_backend_alloc_ctx_tensors(storage, backend);
        CHECK(buffer);
        std::vector<float> weights(rotation*rotation);
        std::vector<float> values(features*tokens);
        for (int64_t row = 0; row < rotation; ++row) {
            for (int64_t i = 0; i < rotation; ++i) {
                weights[row*rotation + i] = float((row*11 + i*7) % 31 - 15)/16.0f;
            }
        }
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t i = 0; i < features; ++i) {
                values[t*features + i] = float((t*13 + i*5) % 23 - 11)/8.0f;
            }
        }
        ggml_backend_tensor_set(matrix, weights.data(), 0, ggml_nbytes(matrix));
        ggml_backend_tensor_set(activation, values.data(), 0, ggml_nbytes(activation));
        std::vector<int64_t> rows(tokens);
        for (int64_t t = 0; t < tokens; ++t) { rows[t] = t; }
        ggml_backend_tensor_set(indices, rows.data(), 0, ggml_nbytes(indices));
        auto chunks = ggml_reshape_2d(ctx, activation, rotation, features*tokens/rotation);
        auto rotated = ggml_mul_mat(ctx, matrix, chunks);
        auto heads = ggml_reshape_3d(ctx, rotated, features, 1, tokens);
        auto flat = ggml_view_2d(ctx, heads, features, tokens, heads->nb[2], 0);
        auto out = projection ? ggml_reshape_2d(ctx, heads, features, tokens) : ggml_set_rows(ctx, cache, flat, indices);
        auto graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_t backends[] = {backend, host};
        auto sched = ggml_backend_sched_new(backends, nullptr, 2, 2048, false, true);
        CHECK(ggml_backend_sched_alloc_graph(sched, graph));
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        std::vector<float> result(features*tokens);
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t row = 0; row < features; ++row) {
                double expected = 0;
                for (int64_t i = 0; i < rotation; ++i) {
                    expected += double(weights[(row % rotation)*rotation + i])*
                                double(values[t*features + (row/rotation)*rotation + i]);
                }
                CHECK(std::fabs(double(result[t*features + row]) - expected) <=
                      1e-4 + 1e-4*std::fabs(expected));
            }
        }
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_free(storage);
    }
    }
    }
    ggml_backend_free(backend);
    ggml_backend_free(host);
    std::puts("single-head rotation/cache-store/output-projection checks passed");
}

static void test_single_head_norm(ggml_backend_dev_t simple, ggml_backend_dev_t cpu) {
    ggml_backend_dev_t devices[] = {simple, simple};
    auto device = ggml_backend_meta_device(devices, 2, single_head_split, nullptr);
    auto backend = ggml_backend_dev_init(device, nullptr);
    auto host = ggml_backend_dev_init(cpu, nullptr);
    CHECK(backend && host);
    for (int64_t tokens : {int64_t(1), int64_t(3)}) {
        ggml_init_params params = {4*1024*1024, nullptr, true};
        auto storage = ggml_init(params);
        auto ctx = ggml_init(params);
        auto input = ggml_new_tensor_2d(storage, GGML_TYPE_F32, 512, tokens);
        auto buffer = ggml_backend_alloc_ctx_tensors(storage, backend);
        CHECK(buffer);
        std::vector<float> values(512*tokens);
        for (size_t i = 0; i < values.size(); ++i) { values[i] = float(i % 13 + 1); }
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        auto reshaped = ggml_reshape_3d(ctx, input, 512, 1, tokens);
        auto norm = ggml_rms_norm(ctx, reshaped, 1e-6f);
        // Cache stores flatten the normalized head before writing rows.
        auto out = ggml_reshape_2d(ctx, norm, 512, tokens);
        auto graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_t backends[] = {backend, host};
        auto sched = ggml_backend_sched_new(backends, nullptr, 2, 2048, false, true);
        CHECK(ggml_backend_sched_alloc_graph(sched, graph));
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        std::vector<float> result(values.size());
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        for (int64_t t = 0; t < tokens; ++t) {
            double sum = 0;
            for (int64_t i = 0; i < 512; ++i) { sum += double(values[t*512+i])*values[t*512+i]; }
            const float scale = 1.0f / std::sqrt(float(sum/512) + 1e-6f);
            for (int64_t i = 0; i < 512; ++i) {
                CHECK(std::fabs(result[t*512+i] - values[t*512+i]*scale) < 1e-5f);
            }
        }
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_free(storage);
    }
    ggml_backend_free(backend);
    ggml_backend_free(host);
    std::puts("single-head tensor-split RMS norm checks passed");
}

// A partial MQA cache view keeps separate stream strides while moving the
// whole-row split to the singleton head axis; the second device stays empty.
static void test_single_head_cache_view_norm(ggml_backend_dev_t simple, ggml_backend_dev_t cpu) {
    ggml_backend_dev_t devices[] = {simple, simple};
    auto device = ggml_backend_meta_device(devices, 2, single_head_split, nullptr);
    auto backend = ggml_backend_dev_init(device, nullptr);
    auto host = ggml_backend_dev_init(cpu, nullptr);
    CHECK(backend && host);
    ggml_init_params params = {4*1024*1024, nullptr, true};
    auto storage = ggml_init(params);
    auto ctx = ggml_init(params);
    auto input = ggml_new_tensor_3d(storage, GGML_TYPE_F32, 512, 512, 2);
    auto buffer = ggml_backend_alloc_ctx_tensors(storage, backend);
    CHECK(buffer);
    std::vector<float> values(512*512*2);
    for (int64_t stream = 0; stream < 2; ++stream) {
        for (int64_t row = 0; row < 512; ++row) {
            for (int64_t i = 0; i < 512; ++i) {
                values[(stream*512 + row)*512 + i] =
                    float((i*7 + row*3 + stream*11) % 29 + 1) + float(row)/512 + float(stream);
            }
        }
    }
    ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    auto view = ggml_view_4d(ctx, input, 512, 1, 256, 2,
                             input->nb[1], input->nb[1], input->nb[2], 0);
    CHECK(!ggml_is_contiguous(view));
    auto norm = ggml_rms_norm(ctx, view, 1e-6f);
    auto out = ggml_reshape_2d(ctx, norm, 512, 512);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backends[] = {backend, host};
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, 2048, false, true);
    CHECK(ggml_backend_sched_alloc_graph(sched, graph));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(512*512);
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    for (int64_t stream = 0; stream < 2; ++stream) {
        for (int64_t row = 0; row < 256; ++row) {
            const size_t source_offset = (stream*input->nb[2] + row*input->nb[1])/sizeof(float);
            const size_t result_offset = (stream*256 + row)*512;
            double sum = 0;
            for (int64_t i = 0; i < 512; ++i) {
                sum += double(values[source_offset + i])*values[source_offset + i];
            }
            const float scale = 1.0f / std::sqrt(float(sum/512) + 1e-6f);
            for (int64_t i = 0; i < 512; ++i) {
                CHECK(std::fabs(result[result_offset + i] - values[source_offset + i]*scale) < 1e-5f);
            }
        }
    }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_free(storage);
    ggml_backend_free(backend);
    ggml_backend_free(host);
    std::puts("single-head noncontiguous cache-view RMS norm checks passed");
}

// Scheduler graphs can end in an already-materialized host cache view.
// The last host boundary must still close the preceding meta subgraph.
static void test_trailing_host_view(ggml_backend_dev_t simple, ggml_backend_dev_t cpu) {
    ggml_backend_dev_t devices[] = {simple, simple};
    auto device = ggml_backend_meta_device(devices, 2, mirrored, nullptr);
    auto backend = ggml_backend_dev_init(device, nullptr);
    auto host = ggml_backend_dev_init(cpu, nullptr);
    CHECK(backend && host);
    ggml_init_params params = {4*1024*1024, nullptr, true};
    auto storage = ggml_init(params);
    auto host_storage = ggml_init(params);
    auto ctx = ggml_init(params);
    auto input = ggml_new_tensor_1d(storage, GGML_TYPE_F32, 16);
    auto external = ggml_new_tensor_1d(host_storage, GGML_TYPE_F32, 16);
    auto buffer = ggml_backend_alloc_ctx_tensors(storage, backend);
    auto host_buffer = ggml_backend_alloc_ctx_tensors(host_storage, host);
    CHECK(buffer && host_buffer);
    std::vector<float> values(16, 3.0f);
    ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    values.assign(16, 7.0f);
    ggml_backend_tensor_set(external, values.data(), 0, ggml_nbytes(external));
    auto out = ggml_scale(ctx, input, 2.0f);
    auto boundary = ggml_reshape_2d(ctx, external, 4, 4);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_build_forward_expand(graph, boundary);
    ggml_backend_t backends[] = {backend, host};
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, 2048, false, true);
    CHECK(ggml_backend_sched_alloc_graph(sched, graph));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(out, values.data(), 0, ggml_nbytes(out));
    for (float value : values) { CHECK(value == 6.0f); }
    ggml_backend_tensor_get(boundary, values.data(), 0, ggml_nbytes(boundary));
    for (float value : values) { CHECK(value == 7.0f); }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_backend_buffer_free(host_buffer);
    ggml_free(ctx);
    ggml_free(storage);
    ggml_free(host_storage);
    ggml_backend_free(backend);
    ggml_backend_free(host);
    std::puts("trailing external host-view boundary checks passed");
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    CHECK(cpu);
    const bool use_cuda = argc == 2 && std::strstr(argv[1], "cuda") != nullptr;
    auto simple = use_cuda ? ggml_backend_dev_by_name("CUDA0") : cpu;
    if (!simple) {
        std::puts("CUDA0 unavailable; skipping meta CUDA graph reuse check");
        return 0;
    }
    test_single_head_norm(simple, cpu);
    test_single_head_cache_view_norm(simple, cpu);
    test_single_head_rotation(simple, cpu);
    test_trailing_host_view(simple, cpu);
    init_original = simple->iface.init_backend;
    simple->iface.init_backend = observe_init;
    ggml_backend_dev_t devices[] = {simple, simple};
    auto device = ggml_backend_meta_device(devices, 2, mirrored, nullptr);
    auto backend = ggml_backend_dev_init(device, nullptr);
    simple->iface.init_backend = init_original;
    CHECK(backend);
    auto host = ggml_backend_dev_init(cpu, nullptr);
    if (argc == 2 && std::strncmp(argv[1], "kvarn", 5) == 0) {
        test_kvarn_reuse(backend, host);
        ggml_backend_free(backend);
        ggml_backend_free(host);
        std::puts("KVarN projection reuse and source mutation checks passed");
        return 0;
    }

    ggml_init_params params = {4*1024*1024, nullptr, true};
    auto storage = ggml_init(params);
    auto ctx = ggml_init(params);
    auto a = ggml_new_tensor_1d(storage, GGML_TYPE_F32, 1024);
    auto b = ggml_new_tensor_1d(storage, GGML_TYPE_F32, 1024);
    auto buffer = ggml_backend_alloc_ctx_tensors(storage, backend);
    CHECK(buffer);
    std::vector<float> data(1024, 1.0f);
    ggml_backend_tensor_set(a, data.data(), 0, ggml_nbytes(a));
    data.assign(1024, 10.0f);
    ggml_backend_tensor_set(b, data.data(), 0, ggml_nbytes(b));
    auto view = ggml_view_1d(ctx, a, 1024, 0);
    auto out = ggml_scale(ctx, view, 2.0f);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backends[] = {backend, host};
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, 2048, false, true);
    CHECK(ggml_backend_sched_alloc_graph(sched, graph));

    auto run = [&](float expected) {
        observed.clear();
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, data.data(), 0, ggml_nbytes(out));
        for (float v : data) { CHECK(std::fabs(v - expected) < 1e-6f); }
        CHECK(observed.size() == 2);
        CHECK(observed[0] != 0 && observed[1] != 0);
        return observed;
    };
    auto first = run(2.0f);
    CHECK(run(2.0f) == first);

    // Stable parent UID must not conceal source or kernel-parameter changes.
    view->src[0] = b;
    view->view_src = b;
    auto changed = run(20.0f);
    CHECK(changed != first);
    CHECK(run(20.0f) == changed);
    float scale = 3.0f;
    std::memcpy(out->op_params, &scale, sizeof(scale));
    auto scaled = run(30.0f);
    CHECK(scaled != changed);
    CHECK(run(30.0f) == scaled);

    // Input contents are not executable identity: retain the projection.
    data.assign(1024, 7.0f);
    ggml_backend_tensor_set(b, data.data(), 0, ggml_nbytes(b));
    CHECK(run(21.0f) == scaled);

    // A new scheduler generation may rewrite properties before any meta-buffer
    // init callback. It must rotate projected storage rather than reuse stale
    // slices from the previous generation.
    view->src[0] = a;
    view->view_src = a;
    ++graph->uid;
    auto generation = run(3.0f);
    CHECK(generation != scaled);
    CHECK(run(3.0f) == generation);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_free(storage);
    ggml_backend_free(backend);
    ggml_backend_free(host);
    std::puts("meta graph reuse and mutation checks passed");
}
