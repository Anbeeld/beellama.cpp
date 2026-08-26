#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-impl.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void fail(const char * message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

static std::vector<float> get_tensor(ggml_tensor * tensor) {
    std::vector<float> result(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, ggml_nbytes(tensor));
    return result;
}

static void require_all(const std::vector<float> & values, float expected, const char * message) {
    for (float value : values) {
        if (std::fabs(value - expected) > 1e-6f) {
            fail(message);
        }
    }
}

int main() {
    if (std::getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr) {
        std::fprintf(stderr, "CUDA graphs explicitly disabled, skipping\n");
        return 0;
    }

    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("CUDA0");
    if (dev == nullptr) {
        std::fprintf(stderr, "CUDA0 is unavailable, skipping\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        fail("failed to initialize CUDA0");
    }

    ggml_init_params params = {
        /* .mem_size   = */ 1024*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        fail("failed to initialize ggml context");
    }

    constexpr int64_t n = 1024;
    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * c   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * out = ggml_add(ctx, a, b);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    // The CUDA graph cache uses this identity to retain a captured executable.
    // Replacing a source below must still invalidate the executable because the
    // source address is part of its kernel parameters.
    graph->uid = 1;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        fail("failed to allocate CUDA graph tensors");
    }

    const std::vector<float> a_data(n, 1.0f);
    const std::vector<float> b_data(n, 10.0f);
    const std::vector<float> c_data(n, 100.0f);
    ggml_backend_tensor_set(a, a_data.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_data.data(), 0, ggml_nbytes(b));
    ggml_backend_tensor_set(c, c_data.data(), 0, ggml_nbytes(c));

    // First evaluation establishes CUDA's property snapshot; the second can
    // capture and replay it.  The third changes only the source identity while
    // preserving the scheduler UID and all tensor shapes.
    for (int i = 0; i < 2; ++i) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            fail("failed to warm up CUDA graph reuse test");
        }
        require_all(get_tensor(out), 11.0f, "CUDA graph warmup returned incorrect output");
    }

    out->src[0] = c;
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fail("CUDA graph source replacement compute failed");
    }
    require_all(get_tensor(out), 110.0f,
            "CUDA graph replay used a stale source after its stable UID was reused");

    // The refreshed graph must remain correct when it is replayed again.
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fail("CUDA graph refreshed replay failed");
    }
    require_all(get_tensor(out), 110.0f,
            "CUDA graph refreshed replay returned incorrect output");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
