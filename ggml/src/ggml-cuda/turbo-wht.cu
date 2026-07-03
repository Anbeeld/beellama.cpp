#include "turbo-wht.cuh"

// Sign arrays for FWHT rotation (from turbo-wht.h, seed=42)
static __constant__ float d_turbo_wht_s1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1};
static __constant__ float d_turbo_wht_s2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1};

// One block per 128-element group. 128 threads per block.
static __global__ void k_turbo_wht(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t n_elements, const int direction) {

    const int64_t group = blockIdx.x;
    const int64_t offset = group * 128;
    if (offset >= n_elements) return;

    const float * s_first  = (direction == 0) ? d_turbo_wht_s1 : d_turbo_wht_s2;
    const float * s_second = (direction == 0) ? d_turbo_wht_s2 : d_turbo_wht_s1;

    __shared__ float buf[128];

    // Load and apply first signs
    if (threadIdx.x < 128) {
        buf[threadIdx.x] = src[offset + threadIdx.x] * s_first[threadIdx.x];
    }
    __syncthreads();

    // Parallel FWHT butterfly: 64 threads, 7 passes
    for (int h = 1; h < 128; h *= 2) {
        if (threadIdx.x < 64) {
            int j = (threadIdx.x / h) * (2 * h) + (threadIdx.x % h);
            float a = buf[j], b = buf[j + h];
            buf[j] = a + b; buf[j + h] = a - b;
        }
        __syncthreads();
    }

    // Normalize and apply second signs, write output
    constexpr float inv_sqrt_128 = 0.08838834764831845f; // 1/sqrt(128)
    if (threadIdx.x < 128) {
        dst[offset + threadIdx.x] = buf[threadIdx.x] * inv_sqrt_128 * s_second[threadIdx.x];
    }
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const float * src_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t n_elements = ggml_nelements(src0);
    const int64_t n_groups = n_elements / 128;

    k_turbo_wht<<<(int)n_groups, 128, 0, stream>>>(src_d, dst_d, n_elements, direction);
}

template<int SLICES>
static __global__ void k_kvarn_wht(
        const float * __restrict__ src,
        float * __restrict__ dst,
        const int64_t n_groups) {
    const int64_t group = blockIdx.x;
    if (group >= n_groups) {
        return;
    }

    const int tid = threadIdx.x;
    const int64_t offset = group * (SLICES * 128);
    __shared__ float buf[SLICES][128];

#pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        buf[slice][tid] = src[offset + slice * 128 + tid];
    }
    __syncthreads();

    for (int h = 1; h < 128; h *= 2) {
        if (tid < 64) {
            const int j = (tid / h) * (2 * h) + (tid % h);
#pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                const float a = buf[slice][j];
                const float b = buf[slice][j + h];
                buf[slice][j]     = a + b;
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
        dst[offset + slice * 128 + tid] = x[slice];
    }
}

void ggml_cuda_op_kvarn_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    int head_width;
    memcpy(&head_width, dst->op_params, sizeof(int));
    GGML_ASSERT(head_width == 128 || head_width == 256 || head_width == 512);

    const int64_t n_elements = ggml_nelements(src0);
    GGML_ASSERT(n_elements % head_width == 0);
    const int64_t n_groups = n_elements / head_width;
    if (n_groups == 0) {
        return;
    }

    const float * src_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    switch (head_width) {
        case 128:
            k_kvarn_wht<1><<<(int)n_groups, 128, 0, stream>>>(src_d, dst_d, n_groups);
            break;
        case 256:
            k_kvarn_wht<2><<<(int)n_groups, 128, 0, stream>>>(src_d, dst_d, n_groups);
            break;
        case 512:
            k_kvarn_wht<4><<<(int)n_groups, 128, 0, stream>>>(src_d, dst_d, n_groups);
            break;
        default:
            GGML_ABORT("unsupported KVarN WHT head width");
    }
}
