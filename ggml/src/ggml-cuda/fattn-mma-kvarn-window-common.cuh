#pragma once

#include "fattn-mma-f16.cuh"
#include "fattn-mma-kvarn-case-decl.cuh"
#include "fattn-mma-kvarn-impl.cuh"

// These D-only kernels are instantiated once per head dimension instead of
// once per KVarN MMA column geometry.
template <int D>
static __global__ void ggml_cuda_fattn_kvarn_window_dequant_kernel(
        const ggml_cuda_fattn_kvarn_desc * k_descs,
        const ggml_cuda_fattn_kvarn_desc * v_descs,
        half * k_f16,
        half * v_f16,
        int chunk_begin,
        int chunk_len,
        int n_kv_heads) {
    static_assert(D == 128 || D == 256 || D == 512, "windowed KVarN prefill supports 128-wide slices through D512");
    constexpr int slices = D / GGML_CUDA_FATTN_KVARN_DIM;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int seq   = blockIdx.z;
    const int warp  = threadIdx.x / warp_size;
    const int side  = warp / slices;
    const int slice = warp - side * slices;
    const int lane  = threadIdx.x - warp * warp_size;

    if (side >= 2 || token >= chunk_len) {
        return;
    }

    const ggml_cuda_fattn_kvarn_desc & desc = side == 0 ?
        k_descs[(size_t) seq * n_kv_heads + head] :
        v_descs[(size_t) seq * n_kv_heads + head];

    __shared__ float row_scratch[2][slices][2][GGML_CUDA_FATTN_KVARN_DIM];
    float * row0 = row_scratch[side][slice][0];
    float * row1 = row_scratch[side][slice][1];

    ggml_cuda_fattn_kvarn_load_rotated_slice_warp(
            desc, chunk_begin + token, slice, true, row0, lane);
    const bool needs_original = desc.original_domain != 0;
    __syncthreads();
    float * out = row0;
    const bool combine_slices = needs_original && desc.head_slices > 1;
    if (combine_slices) {
        constexpr float inv_sqrt_slices = slices == 1 ? 1.0f : (slices == 2 ? 0.7071067811865475f : 0.5f);
        for (int d = lane; d < GGML_CUDA_FATTN_KVARN_DIM; d += warp_size) {
            float x = 0.0f;
#pragma unroll
            for (int src_slice = 0; src_slice < slices; ++src_slice) {
                x += ggml_cuda_fattn_kvarn_hslice_sign(slice, src_slice) *
                    row_scratch[side][src_slice][0][d];
            }
            row1[d] = x * inv_sqrt_slices;
        }
    }
    // Every combining warp reads row0 from every peer slice above. Keep this
    // barrier unconditional because K and V may use different domains.
    __syncthreads();
    if (needs_original) {
        if (combine_slices) {
            out = ggml_cuda_fattn_kvarn_inverse_wht_128_warp(row1, row0, lane);
        } else {
            out = ggml_cuda_fattn_kvarn_inverse_wht_128_warp(row0, row1, lane);
        }
    }

    half * dst = (side == 0 ? k_f16 : v_f16) +
        (((size_t) seq * n_kv_heads + head) * chunk_len + token) * D;
    for (int d = lane; d < GGML_CUDA_FATTN_KVARN_DIM; d += warp_size) {
        dst[slice * GGML_CUDA_FATTN_KVARN_DIM + d] = __float2half(out[d]);
    }
}

template<int D>
__launch_bounds__(D, 1)
static __global__ void ggml_cuda_fattn_kvarn_window_finalize_kernel(
        float * acc_ptr,
        const float2 * acc_meta_ptr,
        float2 * dst_meta_ptr,
        const int n_rows) {
    const int row = blockIdx.x;
    const int d = threadIdx.x;
    if (row >= n_rows) {
        return;
    }

    const float rowsum = acc_meta_ptr[row].y;
    float & v = acc_ptr[(size_t) row * D + d];
    v = rowsum > 0.0f ? v / rowsum : 0.0f;
    if (d == 0 && dst_meta_ptr != nullptr) {
        dst_meta_ptr[row] = acc_meta_ptr[row];
    }
}

template <int D>
ggml_cuda_fattn_kvarn_window_dequant_kernel_t ggml_cuda_fattn_kvarn_window_dequant_get_kernel() {
    return ggml_cuda_fattn_kvarn_window_dequant_kernel<D>;
}

template<int D>
ggml_cuda_fattn_kvarn_window_finalize_kernel_t ggml_cuda_fattn_kvarn_window_finalize_get_kernel() {
    return ggml_cuda_fattn_kvarn_window_finalize_kernel<D>;
}
