#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void fail(const char * message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev_0 = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_dev_t dev_1 = ggml_backend_dev_by_name("CUDA1");
    if (dev_0 == nullptr || dev_1 == nullptr) {
        std::fprintf(stderr, "two CUDA devices are unavailable, skipping\n");
        return 0;
    }

    ggml_backend_t backend_0 = ggml_backend_dev_init(dev_0, nullptr);
    ggml_backend_t backend_1 = ggml_backend_dev_init(dev_1, nullptr);
    if (backend_0 == nullptr || backend_1 == nullptr) {
        fail("failed to initialize two CUDA backends");
    }

    // Match the 20 MiB cross-device transfers observed in the real MTP
    // prefill reproducer. On a topology without direct peer access, the CUDA
    // backend declines the async copy and the generic backend stages safely
    // through host memory.
    constexpr int64_t n_elements = 5 * 1024 * 1024;
    ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx_0 = ggml_init(params);
    ggml_context * ctx_1 = ggml_init(params);
    if (ctx_0 == nullptr || ctx_1 == nullptr) {
        fail("failed to initialize tensor contexts");
    }

    ggml_tensor * src_0 = ggml_new_tensor_1d(ctx_0, GGML_TYPE_F32, n_elements);
    ggml_tensor * dst_0 = ggml_new_tensor_1d(ctx_0, GGML_TYPE_F32, n_elements);
    ggml_tensor * src_1 = ggml_new_tensor_1d(ctx_1, GGML_TYPE_F32, n_elements);
    ggml_tensor * dst_1 = ggml_new_tensor_1d(ctx_1, GGML_TYPE_F32, n_elements);

    ggml_backend_buffer_t buf_0 = ggml_backend_alloc_ctx_tensors(ctx_0, backend_0);
    ggml_backend_buffer_t buf_1 = ggml_backend_alloc_ctx_tensors(ctx_1, backend_1);
    if (buf_0 == nullptr || buf_1 == nullptr) {
        fail("failed to allocate CUDA copy tensors");
    }

    std::vector<float> data_0(n_elements);
    std::vector<float> data_1(n_elements);
    for (int64_t i = 0; i < n_elements; ++i) {
        data_0[i] = float(i % 251) + 0.25f;
        data_1[i] = -float(i % 241) - 0.5f;
    }
    ggml_backend_tensor_set(src_0, data_0.data(), 0, ggml_nbytes(src_0));
    ggml_backend_tensor_set(src_1, data_1.data(), 0, ggml_nbytes(src_1));

    // Exercise both directions repeatedly. The regression timeout catches the
    // no-progress failure that originally appeared in opposite-direction
    // tensor-reduction traffic.
    for (int i = 0; i < 64; ++i) {
        ggml_backend_tensor_copy_async(backend_0, backend_1, src_0, dst_1);
        ggml_backend_tensor_copy_async(backend_1, backend_0, src_1, dst_0);
    }

    ggml_backend_synchronize(backend_0);
    ggml_backend_synchronize(backend_1);

    std::vector<float> result_0(n_elements);
    std::vector<float> result_1(n_elements);
    ggml_backend_tensor_get(dst_0, result_0.data(), 0, ggml_nbytes(dst_0));
    ggml_backend_tensor_get(dst_1, result_1.data(), 0, ggml_nbytes(dst_1));
    if (result_0 != data_1 || result_1 != data_0) {
        fail("cross-device CUDA copy returned incorrect data");
    }

    ggml_backend_buffer_free(buf_0);
    ggml_backend_buffer_free(buf_1);
    ggml_free(ctx_0);
    ggml_free(ctx_1);
    ggml_backend_free(backend_0);
    ggml_backend_free(backend_1);
    return 0;
}
