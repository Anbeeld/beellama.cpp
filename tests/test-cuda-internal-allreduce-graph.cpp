#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// This intentionally mirrors tensor-parallel projection: static matrices are
// split along their contraction dimension, so MUL_MAT produces a per-device
// partial sum and the following node consumes its all-reduced mirrored result.
static ggml_backend_meta_split_state split_contraction_axis(
        const ggml_tensor * tensor, void *) {
    if (std::strstr(tensor->name, "mirrored") != nullptr) {
        return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, { 0 }, { 1 }, 1 };
    }
    ggml_backend_meta_split_state result = {
        GGML_BACKEND_SPLIT_AXIS_0, { 0 }, { 1 }, 1
    };
    result.ne[0] = tensor->ne[0] / 2;
    result.ne[1] = tensor->ne[0] - result.ne[0];
    return result;
}

static void run_graph_lifecycle(
        ggml_backend_t meta_backend,
        int64_t        contraction,
        int64_t        output_width,
        int            n_pairs,
        int            iterations,
        int            in_flight_replays = 1) {
    ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * static_ctx = ggml_init(params);
    ggml_context * compute_ctx = ggml_init(params);
    if (static_ctx == nullptr || compute_ctx == nullptr) {
        fail("failed to initialize graph lifecycle tensor contexts");
    }

    std::vector<ggml_tensor *> matrices_a;
    std::vector<ggml_tensor *> vectors_b;
    std::vector<ggml_tensor *> vectors_b_replacement;
    std::vector<ggml_tensor *> partials;
    std::vector<ggml_tensor *> outputs;
    matrices_a.reserve(n_pairs);
    vectors_b.reserve(n_pairs);
    vectors_b_replacement.reserve(n_pairs);
    partials.reserve(n_pairs);
    outputs.reserve(n_pairs);

    for (int pair = 0; pair < n_pairs; ++pair) {
        ggml_tensor * a = ggml_new_tensor_2d(
                static_ctx, GGML_TYPE_F32, contraction, output_width);
        ggml_tensor * b = ggml_new_tensor_2d(
                static_ctx, GGML_TYPE_F32, contraction, 1);
        ggml_tensor * b_replacement = ggml_new_tensor_2d(
                static_ctx, GGML_TYPE_F32, contraction, 1);
        if (a == nullptr || b == nullptr || b_replacement == nullptr) {
            fail("failed to allocate static tensor-split graph inputs");
        }
        ggml_set_name(a, "cuda_ar_graph_a");
        ggml_set_name(b, "cuda_ar_graph_b");

        ggml_tensor * partial = ggml_mul_mat(compute_ctx, a, b);
        ggml_tensor * output = ggml_dup(compute_ctx, partial);
        if (partial == nullptr || output == nullptr) {
            fail("failed to build tensor-split graph nodes");
        }
        ggml_set_name(output, "cuda_ar_graph_output");
        matrices_a.push_back(a);
        vectors_b.push_back(b);
        vectors_b_replacement.push_back(b_replacement);
        partials.push_back(partial);
        outputs.push_back(output);
    }

    ggml_cgraph * graph = ggml_new_graph(compute_ctx);
    if (graph == nullptr) {
        fail("failed to allocate tensor-split graph");
    }
    for (ggml_tensor * output : outputs) {
        ggml_build_forward_expand(graph, output);
    }

    ggml_backend_buffer_t static_buffer = ggml_backend_alloc_ctx_tensors(
            static_ctx, meta_backend);
    if (static_buffer == nullptr) {
        fail("failed to allocate static tensor-split graph inputs");
    }

    ggml_backend_t cpu_backend = ggml_backend_init_by_type(
            GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu_backend == nullptr) {
        fail("failed to initialize CPU graph fallback");
    }
    ggml_backend_t backends[] = { meta_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (sched == nullptr || !ggml_backend_sched_alloc_graph(sched, graph)) {
        fail("failed to allocate scheduled tensor-split graph");
    }

    std::vector<float> matrix_data(
            static_cast<size_t>(contraction * output_width), 1.0f);
    for (ggml_tensor * a : matrices_a) {
        ggml_backend_tensor_set(a, matrix_data.data(), 0, ggml_nbytes(a));
    }

    std::vector<float> vector_data(static_cast<size_t>(contraction));
    std::vector<float> result(static_cast<size_t>(output_width));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        // Replace a projected node source without changing the scheduler graph
        // identity.  The meta backend must rebuild its per-device projection;
        // otherwise CUDA sees the old cloned source forever even though its own
        // captured-graph property validation is correct.
        const bool use_replacement = iteration >= 2;
        if (iteration == 2) {
            for (int pair = 0; pair < n_pairs; ++pair) {
                partials[pair]->src[1] = vectors_b_replacement[pair];
            }
        }
        // Meta graphs are submitted asynchronously.  Queue multiple projected
        // CUDA graph executions before synchronizing to cover a cache
        // transition's re-entry pattern: the next projection must not update
        // or destroy a graph executable still owned by an earlier launch.
        for (int replay = 0; replay < in_flight_replays; ++replay) {
            for (int pair = 0; pair < n_pairs; ++pair) {
                const float sign = ((iteration + pair) & 1) == 0 ? 1.0f : -1.0f;
                std::fill(vector_data.begin(), vector_data.end(), sign);
                ggml_backend_tensor_set(
                        vectors_b[pair], vector_data.data(), 0, ggml_nbytes(vectors_b[pair]));
                std::fill(vector_data.begin(), vector_data.end(), 2.0f * sign);
                ggml_backend_tensor_set(
                        vectors_b_replacement[pair], vector_data.data(), 0,
                        ggml_nbytes(vectors_b_replacement[pair]));
            }

            if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
                fail("scheduled tensor-split graph compute failed");
            }
        }
        ggml_backend_sched_synchronize(sched);

        for (int pair = 0; pair < n_pairs; ++pair) {
            const float scale = use_replacement ? 2.0f : 1.0f;
            const float expected = ((iteration + pair) & 1) == 0 ?
                    scale * static_cast<float>(contraction) : -scale * static_cast<float>(contraction);
            ggml_backend_tensor_get(outputs[pair], result.data(), 0, ggml_nbytes(outputs[pair]));
            for (float value : result) {
                if (value != expected) {
                    std::fprintf(stderr, "tensor-split graph mismatch: iteration=%d pair=%d value=%g expected=%g\n",
                            iteration, pair, value, expected);
                    fail("CUDA-graph tensor-split AllReduce returned incorrect data");
                }
            }
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);
    ggml_backend_buffer_free(static_buffer);
    ggml_free(compute_ctx);
    ggml_free(static_ctx);
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
    ggml_backend_dev_t meta_device = ggml_backend_meta_device(
            devices, 2, split_contraction_axis, nullptr);
    if (meta_device == nullptr) {
        fail("failed to initialize tensor-split meta device");
    }
    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_device, nullptr);
    if (meta_backend == nullptr) {
        fail("failed to initialize tensor-split meta backend");
    }

    // Device-side communication keeps projected CUDA subgraphs graph-eligible
    // while exercising scheduler rebuild and source-rewiring lifecycle with
    // the communication pipeline alive.
    run_graph_lifecycle(meta_backend, 512, 512, 4, 128);
    run_graph_lifecycle(meta_backend, 1024, 256, 3, 96);

    // F32 output of this width produces a 1 MiB BF16 wire payload, which
    // selects the staged copy-engine path instead of the mapped small-payload
    // path.  Keep the graph static across repeated evaluations so event reuse
    // and scheduler-owned graph-buffer lifetimes are covered together.
    run_graph_lifecycle(meta_backend, 2, 512 * 1024, 1, 64, 2);

    ggml_backend_free(meta_backend);
    return 0;
}
