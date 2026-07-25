#include "kvarn-wht.cuh"

#include <cstdlib>

static __device__ __forceinline__ float kvarn_wht_to_float(float value) {
    return value;
}

static __device__ __forceinline__ float kvarn_wht_to_float(half value) {
    return __half2float(value);
}

static __device__ __forceinline__ float kvarn_wht_to_float(nv_bfloat16 value) {
    return __bfloat162float(value);
}

template<typename T>
static __device__ __forceinline__ T kvarn_wht_from_float(float value);

template<>
__device__ __forceinline__ float kvarn_wht_from_float<float>(float value) {
    return value;
}

template<>
__device__ __forceinline__ half kvarn_wht_from_float<half>(float value) {
    return __float2half_rn(value);
}

template<>
__device__ __forceinline__ nv_bfloat16 kvarn_wht_from_float<nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template<typename T, int SLICES>
static __global__ void kvarn_wht_shared_kernel(
        const T * __restrict__ src,
        T * __restrict__ dst,
        int64_t n_groups) {
    const int64_t group = blockIdx.x;
    if (group >= n_groups) {
        return;
    }

    const int tid = threadIdx.x;
    const int64_t offset = group * (SLICES * 128);
    __shared__ float buf[SLICES][128];

#pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        buf[slice][tid] = kvarn_wht_to_float(src[offset + slice * 128 + tid]);
    }
    __syncthreads();

    for (int h = 1; h < 128; h *= 2) {
        if (tid < 64) {
            const int j = (tid / h) * (2 * h) + (tid % h);
#pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                const float a = buf[slice][j];
                const float b = buf[slice][j + h];
                buf[slice][j] = a + b;
                buf[slice][j + h] = a - b;
            }
        }
        __syncthreads();
    }

    float x[SLICES];
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
#pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        x[slice] = buf[slice][tid] * inv_sqrt_128;
    }

    if constexpr (SLICES == 2) {
        const float a = x[0];
        const float b = x[1];
        x[0] = (a + b) * 0.7071067811865475f;
        x[1] = (a - b) * 0.7071067811865475f;
    } else if constexpr (SLICES == 4) {
        const float a0 = x[0];
        const float a1 = x[1];
        const float a2 = x[2];
        const float a3 = x[3];
        const float b0 = a0 + a1;
        const float b1 = a0 - a1;
        const float b2 = a2 + a3;
        const float b3 = a2 - a3;
        x[0] = (b0 + b2) * 0.5f;
        x[1] = (b1 + b3) * 0.5f;
        x[2] = (b0 - b2) * 0.5f;
        x[3] = (b1 - b3) * 0.5f;
    }

#pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        dst[offset + slice * 128 + tid] = kvarn_wht_from_float<T>(x[slice]);
    }
}

template<typename T, int SLICES>
static __global__ void kvarn_wht_wave_kernel(
        const T * __restrict__ src,
        T * __restrict__ dst,
        int64_t n_groups) {
    const int64_t group = blockIdx.x;
    if (group >= n_groups) {
        return;
    }

    constexpr int PHYSICAL_WAVE_SIZE = ggml_cuda_get_physical_warp_size();
    static_assert(PHYSICAL_WAVE_SIZE == 32 || PHYSICAL_WAVE_SIZE == 64,
        "KVarN WHT requires physical wave32 or wave64");
    const int tid = threadIdx.x;
    const int lane = tid % PHYSICAL_WAVE_SIZE;
    const int64_t offset = group * (SLICES * 128);
    float x[SLICES];
    __shared__ float cross_wave[SLICES][128];

#pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        x[slice] = kvarn_wht_to_float(src[offset + slice * 128 + tid]);
    }

    for (int h = 1; h < PHYSICAL_WAVE_SIZE; h *= 2) {
#pragma unroll
        for (int slice = 0; slice < SLICES; ++slice) {
            const float partner = __shfl_xor_sync(
                0xFFFFFFFFULL, x[slice], h, PHYSICAL_WAVE_SIZE);
            x[slice] = (lane & h) != 0 ? partner - x[slice] : x[slice] + partner;
        }
    }

    for (int h = PHYSICAL_WAVE_SIZE; h < 128; h *= 2) {
        float partner[SLICES];
#pragma unroll
        for (int slice = 0; slice < SLICES; ++slice) {
            cross_wave[slice][tid] = x[slice];
        }
        __syncthreads();
#pragma unroll
        for (int slice = 0; slice < SLICES; ++slice) {
            partner[slice] = cross_wave[slice][tid ^ h];
        }
        __syncthreads();
#pragma unroll
        for (int slice = 0; slice < SLICES; ++slice) {
            x[slice] = (tid & h) != 0 ?
                partner[slice] - x[slice] : x[slice] + partner[slice];
        }
    }

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
#pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        x[slice] *= inv_sqrt_128;
    }

    if constexpr (SLICES == 2) {
        const float a = x[0];
        const float b = x[1];
        x[0] = (a + b) * 0.7071067811865475f;
        x[1] = (a - b) * 0.7071067811865475f;
    } else if constexpr (SLICES == 4) {
        const float a0 = x[0];
        const float a1 = x[1];
        const float a2 = x[2];
        const float a3 = x[3];
        const float b0 = a0 + a1;
        const float b1 = a0 - a1;
        const float b2 = a2 + a3;
        const float b3 = a2 - a3;
        x[0] = (b0 + b2) * 0.5f;
        x[1] = (b1 + b3) * 0.5f;
        x[2] = (b0 - b2) * 0.5f;
        x[3] = (b1 - b3) * 0.5f;
    }

#pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        dst[offset + slice * 128 + tid] = kvarn_wht_from_float<T>(x[slice]);
    }
}

void ggml_cuda_op_kvarn_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == src0->type);

    int head_width;
    memcpy(&head_width, dst->op_params, sizeof(head_width));
    GGML_ASSERT(head_width == 128 || head_width == 256 || head_width == 512);

    const int64_t n_elements = ggml_nelements(src0);
    GGML_ASSERT(n_elements % head_width == 0);
    const int64_t n_groups = n_elements / head_width;
    if (n_groups == 0) {
        return;
    }

    cudaStream_t stream = ctx.stream();
    const char * force_shared = std::getenv("GGML_KVARN_TEST_FORCE_SHARED_WHT");
    const bool use_shared_oracle = force_shared != nullptr && std::atoi(force_shared) != 0;

#define GGML_CUDA_KVARN_WHT_LAUNCH(T, SLICES) \
    do { \
        if (use_shared_oracle) { \
            kvarn_wht_shared_kernel<T, SLICES><<<(int) n_groups, 128, 0, stream>>>( \
                (const T *) src0->data, (T *) dst->data, n_groups); \
        } else { \
            kvarn_wht_wave_kernel<T, SLICES><<<(int) n_groups, 128, 0, stream>>>( \
                (const T *) src0->data, (T *) dst->data, n_groups); \
        } \
    } while (0)
#define GGML_CUDA_KVARN_WHT_TYPE(T) \
    do { \
        switch (head_width) { \
            case 128: GGML_CUDA_KVARN_WHT_LAUNCH(T, 1); break; \
            case 256: GGML_CUDA_KVARN_WHT_LAUNCH(T, 2); break; \
            case 512: GGML_CUDA_KVARN_WHT_LAUNCH(T, 4); break; \
            default: GGML_ABORT("unsupported KVarN WHT head width"); \
        } \
    } while (0)
    switch (src0->type) {
        case GGML_TYPE_F32:  GGML_CUDA_KVARN_WHT_TYPE(float); break;
        case GGML_TYPE_F16:  GGML_CUDA_KVARN_WHT_TYPE(half); break;
        case GGML_TYPE_BF16: GGML_CUDA_KVARN_WHT_TYPE(nv_bfloat16); break;
        default: GGML_ABORT("unsupported KVarN WHT input type");
    }
#undef GGML_CUDA_KVARN_WHT_TYPE
#undef GGML_CUDA_KVARN_WHT_LAUNCH
}
