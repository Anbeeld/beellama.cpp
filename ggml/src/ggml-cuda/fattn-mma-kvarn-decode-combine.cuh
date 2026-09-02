#pragma once

#include "fattn-mma-kvarn-decode-decl.cuh"

// This bit-independent reduction is instantiated once per head dimension in a
// dedicated translation unit rather than once per compiled K/V bit pair.
static constexpr int GGML_CUDA_FATTN_KVARN_DECODE_COMBINE_THREADS = 256;

template<int D>
static __global__ void ggml_cuda_fattn_kvarn_decode_combine_kernel(
        const float * partial,
        const float2 * partial_meta,
        float * dst,
        float2 * dst_meta,
        int n_splits,
        int n_q,
        int n_q_heads) {
    const int q_head = blockIdx.x;
    const int q_index = blockIdx.y;
    const int stream = blockIdx.z;
    const int tid = threadIdx.x;

    __shared__ float reduce_sh[GGML_CUDA_FATTN_KVARN_DECODE_COMBINE_THREADS];
    extern __shared__ float split_weights[];

    float local_max = -FLT_MAX / 2.0f;
    for (int split = tid; split < n_splits; split += blockDim.x) {
        const float2 meta = partial_meta[(((size_t) stream * n_q + q_index) * n_q_heads + q_head) * n_splits + split];
        if (meta.y > 0.0f) {
            local_max = fmaxf(local_max, meta.x);
        }
    }
    reduce_sh[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_sh[tid] = fmaxf(reduce_sh[tid], reduce_sh[tid + stride]);
        }
        __syncthreads();
    }
    const float m = reduce_sh[0];

    float local_denom = 0.0f;
    for (int split = tid; split < n_splits; split += blockDim.x) {
        const float2 meta = partial_meta[(((size_t) stream * n_q + q_index) * n_q_heads + q_head) * n_splits + split];
        float weight = 0.0f;
        if (meta.y > 0.0f) {
            weight = __expf(meta.x - m);
            local_denom += weight * meta.y;
        }
        split_weights[split] = weight;
    }
    reduce_sh[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_sh[tid] += reduce_sh[tid + stride];
        }
        __syncthreads();
    }
    const float denom = reduce_sh[0];

    const size_t output_row = ((size_t) stream * n_q + q_index) * n_q_heads + q_head;
    if (tid == 0 && dst_meta != nullptr) {
        dst_meta[output_row] = make_float2(m, denom);
    }

    for (int dim = tid; dim < D; dim += blockDim.x) {
        float out = 0.0f;
        if (denom > 0.0f) {
            for (int split = 0; split < n_splits; ++split) {
                // Skipped splits did not write partials; their zero weight also
                // avoids reading those unwritten values during the reduction.
                const float weight = split_weights[split];
                if (weight == 0.0f) {
                    continue;
                }
                const size_t base = (((size_t) stream * n_q + q_index) * n_q_heads + q_head) * n_splits + split;
                out += weight * partial[base * D + dim];
            }
            out /= denom;
        }
        dst[output_row * D + dim] = out;
    }
}

template<int D>
ggml_cuda_fattn_kvarn_decode_combine_kernel_t ggml_cuda_fattn_kvarn_decode_combine_get_kernel() {
    return ggml_cuda_fattn_kvarn_decode_combine_kernel<D>;
}
