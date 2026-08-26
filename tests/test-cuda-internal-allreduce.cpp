#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
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
        fail("invalid CUDA AllReduce test environment value");
    }
    return parsed;
}

static void upload_opposite_values(ggml_tensor ** tensors, int64_t n_elements) {
    const ggml_type type = tensors[0]->type;
    if (tensors[1]->type != type) {
        fail("CUDA AllReduce test tensor types differ");
    }

    switch (type) {
        case GGML_TYPE_F32: {
            std::vector<float> positive(n_elements, 1.0f);
            std::vector<float> negative(n_elements, -1.0f);
            ggml_backend_tensor_set(tensors[0], positive.data(), 0, ggml_nbytes(tensors[0]));
            ggml_backend_tensor_set(tensors[1], negative.data(), 0, ggml_nbytes(tensors[1]));
            return;
        }
        case GGML_TYPE_F16: {
            std::vector<ggml_fp16_t> positive(n_elements, ggml_fp32_to_fp16(1.0f));
            std::vector<ggml_fp16_t> negative(n_elements, ggml_fp32_to_fp16(-1.0f));
            ggml_backend_tensor_set(tensors[0], positive.data(), 0, ggml_nbytes(tensors[0]));
            ggml_backend_tensor_set(tensors[1], negative.data(), 0, ggml_nbytes(tensors[1]));
            return;
        }
        case GGML_TYPE_BF16: {
            std::vector<ggml_bf16_t> positive(n_elements, ggml_fp32_to_bf16(1.0f));
            std::vector<ggml_bf16_t> negative(n_elements, ggml_fp32_to_bf16(-1.0f));
            ggml_backend_tensor_set(tensors[0], positive.data(), 0, ggml_nbytes(tensors[0]));
            ggml_backend_tensor_set(tensors[1], negative.data(), 0, ggml_nbytes(tensors[1]));
            return;
        }
        default:
            fail("unsupported CUDA AllReduce test tensor type");
    }
}

static void verify_zero(ggml_tensor ** tensors, int64_t n_elements) {
    const ggml_type type = tensors[0]->type;
    for (int rank = 0; rank < 2; ++rank) {
        if (tensors[rank]->type != type) {
            fail("CUDA AllReduce result tensor types differ");
        }

        switch (type) {
            case GGML_TYPE_F32: {
                std::vector<float> result(n_elements);
                ggml_backend_tensor_get(tensors[rank], result.data(), 0, ggml_nbytes(tensors[rank]));
                for (float value : result) {
                    if (value != 0.0f) {
                        fail("internal CUDA AllReduce returned incorrect F32 data");
                    }
                }
                break;
            }
            case GGML_TYPE_F16: {
                std::vector<ggml_fp16_t> result(n_elements);
                ggml_backend_tensor_get(tensors[rank], result.data(), 0, ggml_nbytes(tensors[rank]));
                for (ggml_fp16_t value : result) {
                    if (ggml_fp16_to_fp32(value) != 0.0f) {
                        fail("internal CUDA AllReduce returned incorrect F16 data");
                    }
                }
                break;
            }
            case GGML_TYPE_BF16: {
                std::vector<ggml_bf16_t> result(n_elements);
                ggml_backend_tensor_get(tensors[rank], result.data(), 0, ggml_nbytes(tensors[rank]));
                for (ggml_bf16_t value : result) {
                    if (ggml_bf16_to_fp32(value) != 0.0f) {
                        fail("internal CUDA AllReduce returned incorrect BF16 data");
                    }
                }
                break;
            }
            default:
                fail("unsupported CUDA AllReduce result tensor type");
        }
    }
}

int main() {
    using comm_init_t = void * (*)(ggml_backend_t *, size_t);
    using comm_free_t = void (*)(void *);
    using comm_allreduce_t = bool (*)(void *, ggml_tensor **, int64_t);

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

    const int64_t n_small_elements = read_positive_env(
        "GGML_CUDA_AR_TEST_ELEMENTS", 64 * 1024, 8 * 1024 * 1024);
    const int64_t n_large_elements = read_positive_env(
        "GGML_CUDA_AR_TEST_LARGE_ELEMENTS", 1024 * 1024, 8 * 1024 * 1024);
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

    ggml_tensor * small_f32[] = {
        ggml_new_tensor_1d(ctx_0, GGML_TYPE_F32, n_small_elements),
        ggml_new_tensor_1d(ctx_1, GGML_TYPE_F32, n_small_elements),
    };
    ggml_tensor * large_f32[] = {
        ggml_new_tensor_1d(ctx_0, GGML_TYPE_F32, n_large_elements),
        ggml_new_tensor_1d(ctx_1, GGML_TYPE_F32, n_large_elements),
    };
    ggml_tensor * small_f16[] = {
        ggml_new_tensor_1d(ctx_0, GGML_TYPE_F16, n_small_elements),
        ggml_new_tensor_1d(ctx_1, GGML_TYPE_F16, n_small_elements),
    };
    ggml_tensor * large_f16[] = {
        ggml_new_tensor_1d(ctx_0, GGML_TYPE_F16, n_large_elements),
        ggml_new_tensor_1d(ctx_1, GGML_TYPE_F16, n_large_elements),
    };
    ggml_tensor * small_bf16[] = {
        ggml_new_tensor_1d(ctx_0, GGML_TYPE_BF16, n_small_elements),
        ggml_new_tensor_1d(ctx_1, GGML_TYPE_BF16, n_small_elements),
    };
    ggml_tensor * large_bf16[] = {
        ggml_new_tensor_1d(ctx_0, GGML_TYPE_BF16, n_large_elements),
        ggml_new_tensor_1d(ctx_1, GGML_TYPE_BF16, n_large_elements),
    };
    ggml_tensor * const all_tensors[] = {
        small_f32[0], small_f32[1], large_f32[0], large_f32[1],
        small_f16[0], small_f16[1], large_f16[0], large_f16[1],
        small_bf16[0], small_bf16[1], large_bf16[0], large_bf16[1],
    };
    for (ggml_tensor * tensor : all_tensors) {
        tensor->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    ggml_backend_buffer_t buffers[] = {
        ggml_backend_alloc_ctx_tensors(ctx_0, backends[0]),
        ggml_backend_alloc_ctx_tensors(ctx_1, backends[1]),
    };
    if (buffers[0] == nullptr || buffers[1] == nullptr) {
        fail("failed to allocate CUDA AllReduce tensors");
    }

    // The CUDA tensor-split provider is required on the supported two-device
    // topology.  A decline here would route production tensor split through
    // the slower meta-backend butterfly instead of exercising this regression.
    const int n_small_reductions = (int) read_positive_env(
        "GGML_CUDA_AR_TEST_REDUCTIONS", 50000, 1000 * 1000);
    const int n_large_reductions = (int) read_positive_env(
        "GGML_CUDA_AR_TEST_LARGE_REDUCTIONS", 8, 1000 * 1000);
    const int n_type_reductions = (int) read_positive_env(
        "GGML_CUDA_AR_TEST_TYPE_REDUCTIONS", 1024, 1000 * 1000);

    auto run_case = [&](const char * name, ggml_tensor ** tensors, int64_t n_elements, int n_reductions) {
        upload_opposite_values(tensors, n_elements);

        std::fprintf(stderr, "internal CUDA AllReduce test: %s: %d reductions x %lld elements\n",
                     name, n_reductions, (long long) n_elements);
        for (int i = 0; i < n_reductions; ++i) {
            if (!allreduce(comm, tensors, 1024)) {
                if (i == 0) {
                    fail("internal CUDA AllReduce was not selected for this two-device topology");
                }
                fail("internal CUDA AllReduce stopped after accepting work");
            }
        }

        ggml_backend_synchronize(backends[0]);
        ggml_backend_synchronize(backends[1]);

        verify_zero(tensors, n_elements);
    };

    auto run_f32_precision_case = [&](ggml_tensor ** tensors, int64_t n_elements, int64_t graph_batch_size) {
        constexpr float rank_0 = 1.001f;
        constexpr float rank_1 = -0.333f;
        const float expected =
            ggml_bf16_to_fp32(ggml_fp32_to_bf16(rank_0)) +
            ggml_bf16_to_fp32(ggml_fp32_to_bf16(rank_1));
        std::vector<float> first(n_elements, rank_0);
        std::vector<float> second(n_elements, rank_1);
        ggml_backend_tensor_set(tensors[0], first.data(), 0, ggml_nbytes(tensors[0]));
        ggml_backend_tensor_set(tensors[1], second.data(), 0, ggml_nbytes(tensors[1]));
        if (!allreduce(comm, tensors, graph_batch_size)) {
            fail("internal CUDA AllReduce declined the F32 precision case");
        }
        ggml_backend_synchronize(backends[0]);
        ggml_backend_synchronize(backends[1]);
        for (int rank = 0; rank < 2; ++rank) {
            std::vector<float> result(n_elements);
            ggml_backend_tensor_get(tensors[rank], result.data(), 0, ggml_nbytes(tensors[rank]));
            for (float value : result) {
                if (value != expected) {
                    fail("internal CUDA AllReduce did not use the upstream BF16 wire round trip");
                }
            }
        }
    };

    auto run_f32_chained_case = [&](ggml_tensor ** tensors, int64_t n_elements) {
        constexpr int reductions = 12;
        std::vector<float> first(n_elements, 1.0f);
        std::vector<float> second(n_elements, 3.0f);
        ggml_backend_tensor_set(tensors[0], first.data(), 0, ggml_nbytes(tensors[0]));
        ggml_backend_tensor_set(tensors[1], second.data(), 0, ggml_nbytes(tensors[1]));
        for (int i = 0; i < reductions; ++i) {
            if (!allreduce(comm, tensors, 1024)) {
                fail("internal CUDA AllReduce declined the chained F32 case");
            }
        }
        ggml_backend_synchronize(backends[0]);
        ggml_backend_synchronize(backends[1]);

        const float expected = 4.0f * (float) (1 << (reductions - 1));
        for (int rank = 0; rank < 2; ++rank) {
            std::vector<float> result(n_elements);
            ggml_backend_tensor_get(tensors[rank], result.data(), 0, ggml_nbytes(tensors[rank]));
            for (float value : result) {
                if (value != expected) {
                    fail("internal CUDA AllReduce lost ordering across chained F32 reductions");
                }
            }
        }
    };

    // Exercise small and large payload paths repeatedly through one native
    // communication context. The small passes lap the staging ring many times;
    // the transitions also cover persistent slot reuse across tensor types.
    run_case("f32-small-first", small_f32, n_small_elements, n_small_reductions);
    run_case("f16-small", small_f16, n_small_elements, n_type_reductions);
    run_case("bf16-small", small_bf16, n_small_elements, n_type_reductions);
    run_case("f32-large", large_f32, n_large_elements, n_large_reductions);
    run_case("f16-large", large_f16, n_large_elements, n_type_reductions);
    run_case("bf16-large", large_bf16, n_large_elements, n_type_reductions);
    run_case("f32-small-second", small_f32, n_small_elements, n_small_reductions);
    run_f32_precision_case(small_f32, n_small_elements, 1);
    run_f32_precision_case(large_f32, n_large_elements, 1024);
    run_f32_chained_case(small_f32, n_small_elements);

    comm_free(comm);
    ggml_backend_buffer_free(buffers[0]);
    ggml_backend_buffer_free(buffers[1]);
    ggml_free(ctx_0);
    ggml_free(ctx_1);
    ggml_backend_free(backends[0]);
    ggml_backend_free(backends[1]);
    return 0;
}
