#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void fail(const char * message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

static void set_internal_provider() {
#if defined(_WIN32)
    if (_putenv_s("GGML_CUDA_ALLREDUCE", "internal") != 0) {
        fail("failed to select internal CUDA AllReduce");
    }
#else
    if (setenv("GGML_CUDA_ALLREDUCE", "internal", 1) != 0) {
        fail("failed to select internal CUDA AllReduce");
    }
#endif
}

static int read_positive_env(const char * name, int default_value, int maximum) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > maximum) {
        fail("invalid CUDA AllReduce graph multi-context test environment value");
    }
    return static_cast<int>(parsed);
}

// This intentionally mirrors tensor-parallel projection: static matrices are
// split along their contraction dimension, so MUL_MAT produces a per-device
// partial sum and the following node consumes its all-reduced mirrored result.
static ggml_backend_meta_split_state split_contraction_axis(
        const ggml_tensor * tensor, void *) {
    ggml_backend_meta_split_state result = {
        GGML_BACKEND_SPLIT_AXIS_0, { 0 }, { 1 }, 1
    };
    result.ne[0] = tensor->ne[0] / 2;
    result.ne[1] = tensor->ne[0] - result.ne[0];
    return result;
}

struct graph_case {
    ggml_context *            static_ctx = nullptr;
    ggml_context *            compute_ctx = nullptr;
    std::vector<ggml_tensor *> matrices_a;
    std::vector<ggml_tensor *> vectors_b;
    std::vector<ggml_tensor *> outputs;
    ggml_backend_buffer_t     static_buffer = nullptr;
    ggml_backend_t            cpu_backend = nullptr;
    ggml_backend_sched_t      sched = nullptr;
    ggml_cgraph *             graph = nullptr;
    int64_t                   contraction = 0;
    int64_t                   output_width = 0;
    std::vector<float>        vector_data;
    std::vector<float>        result;
};

static void free_graph_case(graph_case & c) {
    if (c.sched != nullptr) {
        ggml_backend_sched_free(c.sched);
    }
    if (c.cpu_backend != nullptr) {
        ggml_backend_free(c.cpu_backend);
    }
    if (c.static_buffer != nullptr) {
        ggml_backend_buffer_free(c.static_buffer);
    }
    if (c.compute_ctx != nullptr) {
        ggml_free(c.compute_ctx);
    }
    if (c.static_ctx != nullptr) {
        ggml_free(c.static_ctx);
    }
}

static void init_graph_case(
        graph_case &    c,
        ggml_backend_t meta_backend,
        int64_t        contraction,
        int64_t        output_width,
        int            n_pairs) {
    const ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    c.static_ctx = ggml_init(params);
    c.compute_ctx = ggml_init(params);
    if (c.static_ctx == nullptr || c.compute_ctx == nullptr) {
        fail("failed to initialize paired graph tensor contexts");
    }

    c.contraction = contraction;
    c.output_width = output_width;
    c.matrices_a.reserve(n_pairs);
    c.vectors_b.reserve(n_pairs);
    c.outputs.reserve(n_pairs);
    for (int pair = 0; pair < n_pairs; ++pair) {
        ggml_tensor * a = ggml_new_tensor_2d(
                c.static_ctx, GGML_TYPE_F32, contraction, output_width);
        ggml_tensor * b = ggml_new_tensor_2d(
                c.static_ctx, GGML_TYPE_F32, contraction, 1);
        if (a == nullptr || b == nullptr) {
            fail("failed to allocate paired graph inputs");
        }
        ggml_set_name(a, "cuda_ar_graph_multi_a");
        ggml_set_name(b, "cuda_ar_graph_multi_b");

        ggml_tensor * partial = ggml_mul_mat(c.compute_ctx, a, b);
        ggml_tensor * output = ggml_dup(c.compute_ctx, partial);
        if (partial == nullptr || output == nullptr) {
            fail("failed to build paired tensor-split graph nodes");
        }
        ggml_set_name(output, "cuda_ar_graph_multi_output");
        c.matrices_a.push_back(a);
        c.vectors_b.push_back(b);
        c.outputs.push_back(output);
    }

    c.graph = ggml_new_graph(c.compute_ctx);
    if (c.graph == nullptr) {
        fail("failed to allocate paired tensor-split graph");
    }
    for (ggml_tensor * output : c.outputs) {
        ggml_build_forward_expand(c.graph, output);
    }

    c.static_buffer = ggml_backend_alloc_ctx_tensors(c.static_ctx, meta_backend);
    if (c.static_buffer == nullptr) {
        fail("failed to allocate paired tensor-split graph inputs");
    }
    c.cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (c.cpu_backend == nullptr) {
        fail("failed to initialize paired graph CPU fallback");
    }
    ggml_backend_t backends[] = { meta_backend, c.cpu_backend };
    c.sched = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (c.sched == nullptr || !ggml_backend_sched_alloc_graph(c.sched, c.graph)) {
        fail("failed to allocate paired scheduled tensor-split graph");
    }

    std::vector<float> matrix_data(
            static_cast<size_t>(contraction * output_width), 1.0f);
    for (ggml_tensor * a : c.matrices_a) {
        ggml_backend_tensor_set(a, matrix_data.data(), 0, ggml_nbytes(a));
    }
    c.vector_data.resize(static_cast<size_t>(contraction));
    c.result.resize(static_cast<size_t>(output_width));
}

static void prepare_graph_case(graph_case & c, int iteration, int phase) {
    for (size_t pair = 0; pair < c.vectors_b.size(); ++pair) {
        const float sign = ((iteration + phase + static_cast<int>(pair)) & 1) == 0 ? 1.0f : -1.0f;
        std::fill(c.vector_data.begin(), c.vector_data.end(), sign);
        ggml_backend_tensor_set(
                c.vectors_b[pair], c.vector_data.data(), 0, ggml_nbytes(c.vectors_b[pair]));
    }
}

static void compute_graph_case(graph_case & c) {
    if (ggml_backend_sched_graph_compute(c.sched, c.graph) != GGML_STATUS_SUCCESS) {
        fail("paired scheduled tensor-split graph compute failed");
    }
}

static void synchronize_and_verify_graph_case(graph_case & c, int iteration, int phase) {
    ggml_backend_sched_synchronize(c.sched);
    for (size_t pair = 0; pair < c.outputs.size(); ++pair) {
        const float expected = ((iteration + phase + static_cast<int>(pair)) & 1) == 0 ?
                static_cast<float>(c.contraction) : -static_cast<float>(c.contraction);
        ggml_backend_tensor_get(c.outputs[pair], c.result.data(), 0, ggml_nbytes(c.outputs[pair]));
        for (float value : c.result) {
            if (value != expected) {
                fail("paired CUDA-graph tensor-split AllReduce returned incorrect data");
            }
        }
    }
}

int main() {
    set_internal_provider();
    ggml_backend_load_all();

    ggml_backend_dev_t dev_0 = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_dev_t dev_1 = ggml_backend_dev_by_name("CUDA1");
    if (dev_0 == nullptr || dev_1 == nullptr) {
        std::fprintf(stderr, "two CUDA devices are unavailable, skipping\n");
        return 0;
    }

    ggml_backend_dev_t devices[] = { dev_0, dev_1 };
    ggml_backend_dev_t target_meta_device = ggml_backend_meta_device(
            devices, 2, split_contraction_axis, nullptr);
    ggml_backend_dev_t draft_meta_device = ggml_backend_meta_device(
            devices, 2, split_contraction_axis, nullptr);
    if (target_meta_device == nullptr || draft_meta_device == nullptr) {
        fail("failed to initialize paired tensor-split meta devices");
    }
    ggml_backend_t target_meta_backend = ggml_backend_dev_init(target_meta_device, nullptr);
    ggml_backend_t draft_meta_backend = ggml_backend_dev_init(draft_meta_device, nullptr);
    if (target_meta_backend == nullptr || draft_meta_backend == nullptr) {
        fail("failed to initialize paired tensor-split meta backends");
    }

    const int iterations = read_positive_env(
            "GGML_CUDA_AR_TEST_GRAPH_MULTI_ITERATIONS", 64, 100 * 1000);

    auto run_paired = [&](int64_t output_width, int n_pairs, const char * size_name) {
        graph_case target;
        graph_case draft;
        init_graph_case(target, target_meta_backend, 2, output_width, n_pairs);
        init_graph_case(draft, draft_meta_backend, 2, output_width, n_pairs);

        std::fprintf(stderr,
                     "internal CUDA AllReduce paired graph test: %d replays x %d %s reductions per context\n",
                     iterations, n_pairs, size_name);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            prepare_graph_case(target, iteration, 0);
            prepare_graph_case(draft, iteration, 1);
            compute_graph_case(target);
            compute_graph_case(draft);
            synchronize_and_verify_graph_case(target, iteration, 0);
            synchronize_and_verify_graph_case(draft, iteration, 1);
        }

        free_graph_case(draft);
        free_graph_case(target);
    };

    // Match the target model's four-token MTP verification shape and layer
    // cadence before transitioning both persistent communication contexts to
    // large prefill traffic.
    run_paired(5120 * 4, 64, "four-token");

    // A 5 Mi-element F32 output is sent as a 10 MiB BF16 wire payload. Two
    // independent meta backends are submitted before either is synchronized,
    // matching target-plus-MTP-draft graph submission while retaining exact
    // output checks after every replay.
    run_paired(5 * 1024 * 1024, 4, "large");
    ggml_backend_free(draft_meta_backend);
    ggml_backend_free(target_meta_backend);
    return 0;
}
