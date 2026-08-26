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

static int64_t read_positive_env(const char * name, int64_t default_value, int64_t maximum) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > maximum) {
        fail("invalid multi-context CUDA AllReduce test environment value");
    }
    return parsed;
}

using comm_init_t = void * (*)(ggml_backend_t *, size_t);
using comm_free_t = void (*)(void *);
using comm_allreduce_t = bool (*)(void *, ggml_tensor **, int64_t);

struct pipeline {
    ggml_backend_t backends[2] = {};
    void *         comm = nullptr;
    ggml_context * contexts[2] = {};
    ggml_tensor *  tensors[2] = {};
    ggml_backend_buffer_t buffers[2] = {};
};

static void free_pipeline(pipeline & p, comm_free_t comm_free) {
    if (p.comm != nullptr) {
        comm_free(p.comm);
    }
    for (int rank = 0; rank < 2; ++rank) {
        if (p.buffers[rank] != nullptr) {
            ggml_backend_buffer_free(p.buffers[rank]);
        }
        if (p.contexts[rank] != nullptr) {
            ggml_free(p.contexts[rank]);
        }
        if (p.backends[rank] != nullptr) {
            ggml_backend_free(p.backends[rank]);
        }
    }
}

static pipeline make_pipeline(
        ggml_backend_dev_t dev_0,
        ggml_backend_dev_t dev_1,
        comm_init_t        comm_init,
        int64_t            n_elements) {
    pipeline result = {};
    result.backends[0] = ggml_backend_dev_init(dev_0, nullptr);
    result.backends[1] = ggml_backend_dev_init(dev_1, nullptr);
    if (result.backends[0] == nullptr || result.backends[1] == nullptr) {
        fail("failed to initialize multi-context CUDA backends");
    }

    result.comm = comm_init(result.backends, 2);
    if (result.comm == nullptr) {
        fail("failed to initialize a multi-context CUDA communication pipeline");
    }

    const ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    for (int rank = 0; rank < 2; ++rank) {
        result.contexts[rank] = ggml_init(params);
        if (result.contexts[rank] == nullptr) {
            fail("failed to allocate multi-context tensor context");
        }
        result.tensors[rank] = ggml_new_tensor_1d(
                result.contexts[rank], GGML_TYPE_F32, n_elements);
        if (result.tensors[rank] == nullptr) {
            fail("failed to allocate multi-context AllReduce tensor");
        }
        result.tensors[rank]->flags |= GGML_TENSOR_FLAG_COMPUTE;
        result.buffers[rank] = ggml_backend_alloc_ctx_tensors(
                result.contexts[rank], result.backends[rank]);
        if (result.buffers[rank] == nullptr) {
            fail("failed to allocate multi-context CUDA AllReduce buffer");
        }
    }

    std::vector<float> positive(n_elements, 1.0f);
    std::vector<float> negative(n_elements, -1.0f);
    ggml_backend_tensor_set(result.tensors[0], positive.data(), 0, ggml_nbytes(result.tensors[0]));
    ggml_backend_tensor_set(result.tensors[1], negative.data(), 0, ggml_nbytes(result.tensors[1]));
    return result;
}

static void verify_zero(const pipeline & p, int64_t n_elements) {
    std::vector<float> result(n_elements);
    for (int rank = 0; rank < 2; ++rank) {
        ggml_backend_tensor_get(p.tensors[rank], result.data(), 0, ggml_nbytes(p.tensors[rank]));
        for (float value : result) {
            if (value != 0.0f) {
                fail("multi-context internal CUDA AllReduce returned incorrect data");
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

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev_0);
    const auto comm_init = (comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    const auto comm_free = (comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    const auto allreduce = (comm_allreduce_t) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_comm_allreduce_tensor");
    if (comm_init == nullptr || comm_free == nullptr || allreduce == nullptr) {
        fail("CUDA backend does not provide internal communication procedures");
    }

    const int64_t n_elements = read_positive_env(
            "GGML_CUDA_AR_TEST_MULTI_ELEMENTS", 5 * 1024 * 1024, 16 * 1024 * 1024);
    const int n_reductions = (int) read_positive_env(
            "GGML_CUDA_AR_TEST_MULTI_REDUCTIONS", 128, 1000 * 1000);

    pipeline target = make_pipeline(dev_0, dev_1, comm_init, n_elements);
    pipeline draft  = make_pipeline(dev_0, dev_1, comm_init, n_elements);

    std::fprintf(stderr, "internal CUDA AllReduce multi-context test: %d alternating reductions x %lld elements\n",
                 n_reductions, (long long) n_elements);
    for (int i = 0; i < n_reductions; ++i) {
        if (!allreduce(target.comm, target.tensors, 1) || !allreduce(draft.comm, draft.tensors, 1)) {
            fail("internal CUDA AllReduce was not selected by both communication pipelines");
        }
    }

    for (int rank = 0; rank < 2; ++rank) {
        ggml_backend_synchronize(target.backends[rank]);
        ggml_backend_synchronize(draft.backends[rank]);
    }
    verify_zero(target, n_elements);
    verify_zero(draft, n_elements);

    free_pipeline(draft, comm_free);
    free_pipeline(target, comm_free);
    return 0;
}
