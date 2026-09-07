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
