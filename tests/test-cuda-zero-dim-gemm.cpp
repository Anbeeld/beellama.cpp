#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>
#include <vector>

static void test_cublas_zero_dim_guard_present(const char * source_file) {
    std::ifstream input(source_file);
    if (!input) {
        fprintf(stderr, "failed to open '%s'\n", source_file);
        exit(1);
    }

    const std::string source((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

    const bool has_guard =
        source.find("row_diff == 0")   != std::string::npos &&
        source.find("src1_ncols == 0") != std::string::npos &&
        source.find("ne10 == 0")       != std::string::npos &&
        source.find("ne00 == 0")       != std::string::npos &&
        source.find("ldc == 0")        != std::string::npos;

    if (!has_guard) {
        fprintf(stderr, "missing zero-dimension cuBLAS guard in %s\n", source_file);
        exit(1);
    }
}

static void test_zero_column_mul_mat_is_noop(ggml_backend_t backend) {
    ggml_init_params params = {
        /* .mem_size   = */ 1024*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "failed to initialize ggml context\n");
        exit(1);
    }

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 0);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_free(ctx);
        exit(1);
    }

    const std::vector<float> a_data(16, 1.0f);
    ggml_backend_tensor_set(a, a_data.data(), 0, ggml_nbytes(a));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "zero-column mul_mat failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        exit(1);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <ggml-cuda.cu>\n", argv[0]);
        return 1;
    }

    test_cublas_zero_dim_guard_present(argv[1]);

    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        fprintf(stderr, "no GPU backend available, skipping\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        fprintf(stderr, "failed to initialize GPU backend\n");
        return 1;
    }

    test_zero_column_mul_mat_is_noop(backend);

    ggml_backend_free(backend);
    return 0;
}
