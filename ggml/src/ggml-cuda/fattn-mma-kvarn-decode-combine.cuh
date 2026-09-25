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
    // Every warp must consume the shared maximum before thread 0 can reuse
    // reduce_sh[0] for its partial denominator below.
    __syncthreads();

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

    if constexpr (D == 64) {
        // A one-thread-per-dimension reduction leaves three quarters of this
        // 256-thread block idle while reading the large split buffer. Give each
        // D64 dimension four contiguous split ranges, then reduce those four
        // numerators in shared memory. Each 64-thread group still issues
        // contiguous loads for a split, while all 256 threads remain useful.
        constexpr int DIM_GROUPS = GGML_CUDA_FATTN_KVARN_DECODE_COMBINE_THREADS / D;
        const int dim = tid % D;
        const int group = tid / D;
        const int splits_per_group = (n_splits + DIM_GROUPS - 1) / DIM_GROUPS;
        const int split_begin = group * splits_per_group;
        const int split_end = min(n_splits, split_begin + splits_per_group);
        float out = 0.0f;
        if (denom > 0.0f) {
            for (int split = split_begin; split < split_end; ++split) {
                const float weight = split_weights[split];
                if (weight == 0.0f) {
                    continue;
                }
                const size_t base = (((size_t) stream * n_q + q_index) * n_q_heads + q_head) *
                    n_splits + split;
                out += weight * partial[base * D + dim];
            }
        }
        reduce_sh[tid] = out;
        __syncthreads();
        if (group == 0) {
            for (int g = 1; g < DIM_GROUPS; ++g) {
                out += reduce_sh[g * D + dim];
            }
            dst[output_row * D + dim] = denom > 0.0f ? out / denom : 0.0f;
        }
    } else {
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
}

template<int D>
ggml_cuda_fattn_kvarn_decode_combine_kernel_t ggml_cuda_fattn_kvarn_decode_combine_get_kernel() {
    return ggml_cuda_fattn_kvarn_decode_combine_kernel<D>;
}
