#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

[[noreturn]] static void fail(const char * msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static void test_out_prod_f16(ggml_backend_t backend, ggml_type type_b) {
    ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fail("failed to initialize ggml context");
    }

    ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 2, 2, 1, 1);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, type_b, 3, 2, 1, 1);
    ggml_tensor * out = ggml_out_prod(ctx, a, b);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        fail("failed to allocate CPU tensors");
    }

    const float a_f32[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<ggml_fp16_t> a_f16(4);
    ggml_fp32_to_fp16_row(a_f32, a_f16.data(), (int64_t) a_f16.size());

    const float b_f32[] = { 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
    ggml_backend_tensor_set(a, a_f16.data(), 0, a_f16.size() * sizeof(ggml_fp16_t));
    std::vector<ggml_fp16_t> b_f16(6);
    if (type_b == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(b_f32, b_f16.data(), (int64_t) b_f16.size());
        ggml_backend_tensor_set(b, b_f16.data(), 0, b_f16.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(b, b_f32, 0, sizeof(b_f32));
    }

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "OUT_PROD F16 CPU compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        exit(1);
    }

    float got[6] = {};
    ggml_backend_tensor_get(out, got, 0, sizeof(got));

    const float expected[] = { 29.0f, 42.0f, 33.0f, 48.0f, 37.0f, 54.0f };
    for (int i = 0; i < 6; ++i) {
        if (std::fabs(got[i] - expected[i]) > 1e-5f) {
            fprintf(stderr, "OUT_PROD F16 mismatch at %d: got %.8f expected %.8f\n", i, got[i], expected[i]);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            exit(1);
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        fail("failed to initialize CPU backend");
    }

    test_out_prod_f16(backend, GGML_TYPE_F32);
    test_out_prod_f16(backend, GGML_TYPE_F16);

    ggml_backend_free(backend);
    return 0;
}
