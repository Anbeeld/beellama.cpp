#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static ggml_tensor * make_fattn(ggml_context * ctx, ggml_type type_k, ggml_type type_v) {
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 1, 1, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, type_k,        128, 1, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, type_v,        128, 1, 1, 1);

    return ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f, 0.0f, 0.0f);
}

static ggml_tensor * make_scale(ggml_context * ctx, ggml_type type) {
    return ggml_scale(ctx, ggml_new_tensor_1d(ctx, type, 128), 2.0f);
}

int main() {
    ggml_backend_load_all();

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(backend != nullptr, "failed to initialize CPU backend");

    ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize ggml context");

    require(
        ggml_backend_supports_op(backend, make_fattn(ctx, GGML_TYPE_F16, GGML_TYPE_F16)),
        "CPU backend should support F16 FlashAttention");
    require(
        ggml_backend_supports_op(backend, make_scale(ctx, GGML_TYPE_F32)),
        "CPU backend should support F32 scale");
    require(
        !ggml_backend_supports_op(backend, make_scale(ctx, GGML_TYPE_BF16)),
        "CPU backend must reject BF16 scale because its scale kernel is F32-only");

    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
