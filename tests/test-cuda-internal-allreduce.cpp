#include "ggml.h"
#include "ggml-backend.h"

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

int main() {
    using comm_init_t = void * (*)(ggml_backend_t *, size_t);
    using comm_free_t = void (*)(void *);
    using comm_allreduce_t = bool (*)(void *, ggml_tensor **);

    set_internal_provider();
    ggml_backend_load_all();

    ggml_backend_dev_t dev_0 = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_dev_t dev_1 = ggml_backend_dev_by_name("CUDA1");
    if (dev_0 == nullptr || dev_1 == nullptr) {
        std::fprintf(stderr, "two CUDA devices are unavailable, skipping\n");
        return 0;
    }

    ggml_backend_t backends[] = {
        ggml_backend_dev_init(dev_0, nullptr),
        ggml_backend_dev_init(dev_1, nullptr),
    };
    if (backends[0] == nullptr || backends[1] == nullptr) {
        fail("failed to initialize two CUDA backends");
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev_0);
    auto comm_init = (comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    auto comm_free = (comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    auto allreduce = (comm_allreduce_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
    if (comm_init == nullptr || comm_free == nullptr || allreduce == nullptr) {
        fail("CUDA backend does not provide internal communication procedures");
    }

    void * comm = comm_init(backends, 2);
    if (comm == nullptr) {
        fail("failed to initialize internal CUDA communication");
    }

    constexpr int64_t n_elements = 64 * 1024;
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

    ggml_tensor * tensors[] = {
        ggml_new_tensor_1d(ctx_0, GGML_TYPE_F32, n_elements),
        ggml_new_tensor_1d(ctx_1, GGML_TYPE_F32, n_elements),
    };
    tensors[0]->flags |= GGML_TENSOR_FLAG_COMPUTE;
    tensors[1]->flags |= GGML_TENSOR_FLAG_COMPUTE;

    ggml_backend_buffer_t buffers[] = {
        ggml_backend_alloc_ctx_tensors(ctx_0, backends[0]),
        ggml_backend_alloc_ctx_tensors(ctx_1, backends[1]),
    };
    if (buffers[0] == nullptr || buffers[1] == nullptr) {
        fail("failed to allocate CUDA AllReduce tensors");
    }

    std::vector<float> positive(n_elements, 1.0f);
    std::vector<float> negative(n_elements, -1.0f);
    ggml_backend_tensor_set(tensors[0], positive.data(), 0, ggml_nbytes(tensors[0]));
    ggml_backend_tensor_set(tensors[1], negative.data(), 0, ggml_nbytes(tensors[1]));

    // WDDM must decline before it accepts any work.  On supported topologies,
    // this still covers enough repeated reductions to catch a lost-rank
    // forward-progress failure without making normal CTest excessively long.
    constexpr int n_reductions = 4096;
    bool handled = true;
    for (int i = 0; i < n_reductions; ++i) {
        if (!allreduce(comm, tensors)) {
            if (i != 0) {
                fail("internal CUDA AllReduce stopped after accepting work");
            }
            handled = false;
            break;
        }
    }

    if (handled) {
        ggml_backend_synchronize(backends[0]);
        ggml_backend_synchronize(backends[1]);

        std::vector<float> result(n_elements);
        ggml_backend_tensor_get(tensors[0], result.data(), 0, ggml_nbytes(tensors[0]));
        for (float value : result) {
            if (value != 0.0f) {
                fail("internal CUDA AllReduce returned incorrect data");
            }
        }
    } else {
        std::fprintf(stderr, "internal CUDA AllReduce safely declined this topology\n");
    }

    comm_free(comm);
    ggml_backend_buffer_free(buffers[0]);
    ggml_backend_buffer_free(buffers[1]);
    ggml_free(ctx_0);
    ggml_free(ctx_1);
    ggml_backend_free(backends[0]);
    ggml_backend_free(backends[1]);
    return 0;
}
