#pragma once

#include "fattn-mma-f16.cuh"
#include "fattn-mma-kvarn-case-decl.cuh"
#include "fattn-mma-kvarn-impl.cuh"

#include <atomic>

#if defined(GGML_USE_HIP)
using ggml_cuda_fattn_kernel_attr_ptr_t = const void *;
#else
using ggml_cuda_fattn_kernel_attr_ptr_t = fattn_kernel_t;
#endif

// STOPGAP: keep windowed prefill single-chunk by default until the chunked
// merge path emits reference-faithful partials. Smaller chunks remain useful
// for profiling via GGML_KVARN_WINDOW_CHUNK.
static constexpr int GGML_CUDA_FATTN_KVARN_WINDOW_CHUNK = 65536;

static inline bool ggml_cuda_fattn_kvarn_window_enabled() {
    const char * env = getenv("GGML_KVARN_WINDOW");
    return env == nullptr || atoi(env) != 0;
}

static inline int ggml_cuda_fattn_kvarn_window_chunk(const int n_kv) {
    const char * env = getenv("GGML_KVARN_WINDOW_CHUNK");
    if (env == nullptr) {
        return std::min(n_kv, GGML_CUDA_FATTN_KVARN_WINDOW_CHUNK);
    }
    const int override_chunk = atoi(env);
    return override_chunk > 0 ? std::min(n_kv, override_chunk) :
        std::min(n_kv, GGML_CUDA_FATTN_KVARN_WINDOW_CHUNK);
}

template <int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
static inline fattn_kernel_t ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel(
        bool k_original_domain,
        bool v_original_domain) {
    constexpr bool V_is_K_view = false;

    if (k_original_domain) {
        GGML_ASSERT(v_original_domain);
        return flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view,
            GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE, GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE>;
    }

    if (v_original_domain) {
        return flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view,
            GGML_CUDA_FATTN_KVARN_TYPE, GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE>;
    }

    return flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view,
        GGML_CUDA_FATTN_KVARN_TYPE, GGML_CUDA_FATTN_KVARN_TYPE>;
}

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

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void ggml_cuda_fattn_kvarn_window_f16_partial_kernel(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        float2 * partial_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne33, const int32_t nb31, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
    constexpr int ncols = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads (DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / ggml_cuda_get_physical_warp_size();

    const int gqa_ratio = ne02 / ne12;
    const int iter_j = (ne01.z + ncols1 - 1) / ncols1;
    const int iter_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;

    const int tile = blockIdx.x;
    const int sequence = tile / (iter_j * iter_z_gqa * ne12);
    const int rem0 = tile - sequence * iter_j * iter_z_gqa * ne12;
    const int z_KV = rem0 / (iter_j * iter_z_gqa);
    const int rem1 = rem0 - z_KV * iter_j * iter_z_gqa;
    const int zt_gqa = rem1 / iter_j;
    const int jt = rem1 - zt_gqa * iter_j;
    const int zt_Q = z_KV * gqa_ratio + zt_gqa * ncols2;

    const float2 * Q_f2 = (const float2 *) (Q_ptr + nb03 * sequence + nb02 * zt_Q);
    const half2  * K_h2 = (const half2  *) (K_ptr + nb13 * sequence + nb12 * z_KV);
    const half2  * V_h2 = (const half2  *) (V_ptr + nb23 * sequence + nb22 * z_KV);
    const half   * mask_h = ncols2 == 1 && !mask_ptr ? nullptr :
        (const half *) (mask_ptr + nb33 * (sequence % ne33));
    const float  * sinks_f = sinks_ptr ? (const float *) sinks_ptr + zt_Q : nullptr;

    float2 * dstk = nullptr;
    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;
    const int iter_k = (ne11 + nbatch_fa - 1) / nbatch_fa;
    constexpr bool V_is_K_view = false;
    constexpr bool needs_fixup = false;
    constexpr bool is_fixup = true;
    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, partial_ptr, nullptr, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11, nb01 / (int32_t) sizeof(float2), nb02 / (int32_t) sizeof(float2),
         nb11 / (int32_t) sizeof(half2), nb21 / (int32_t) sizeof(half2), nb31 / (int32_t) sizeof(half),
         jt, zt_gqa, 0, iter_k);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, partial_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23, ne33, nb31, nb33);
    NO_DEVICE_CODE;
#endif
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void ggml_cuda_fattn_kvarn_window_f16_direct_kernel(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        float * dst_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne33, const int32_t nb31, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
    constexpr int ncols = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads (DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / ggml_cuda_get_physical_warp_size();

    const int gqa_ratio = ne02 / ne12;
    const int iter_j = (ne01.z + ncols1 - 1) / ncols1;
    const int iter_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;

    const int tile = blockIdx.x;
    const int sequence = tile / (iter_j * iter_z_gqa * ne12);
    const int rem0 = tile - sequence * iter_j * iter_z_gqa * ne12;
    const int z_KV = rem0 / (iter_j * iter_z_gqa);
    const int rem1 = rem0 - z_KV * iter_j * iter_z_gqa;
    const int zt_gqa = rem1 / iter_j;
    const int jt = rem1 - zt_gqa * iter_j;
    const int zt_Q = z_KV * gqa_ratio + zt_gqa * ncols2;

    const float2 * Q_f2 = (const float2 *) (Q_ptr + nb03 * sequence + nb02 * zt_Q);
    const half2  * K_h2 = (const half2  *) (K_ptr + nb13 * sequence + nb12 * z_KV);
    const half2  * V_h2 = (const half2  *) (V_ptr + nb23 * sequence + nb22 * z_KV);
    const half   * mask_h = ncols2 == 1 && !mask_ptr ? nullptr :
        (const half *) (mask_ptr + nb33 * (sequence % ne33));
    const float  * sinks_f = sinks_ptr ? (const float *) sinks_ptr + zt_Q : nullptr;
    float2       * dstk = ((float2 *) dst_ptr) + ((size_t) sequence * ne01.z * ne02 + zt_Q) * (DV / 2);

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;
    const int iter_k = (ne11 + nbatch_fa - 1) / nbatch_fa;
    constexpr bool V_is_K_view = false;
    constexpr bool needs_fixup = false;
    constexpr bool is_fixup = false;
    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, nullptr, nullptr, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11, nb01 / (int32_t) sizeof(float2), nb02 / (int32_t) sizeof(float2),
         nb11 / (int32_t) sizeof(half2), nb21 / (int32_t) sizeof(half2), nb31 / (int32_t) sizeof(half),
         jt, zt_gqa, 0, iter_k);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, dst_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23, ne33, nb31, nb33);
    NO_DEVICE_CODE;
#endif
}

template<int D, int ncols1, int ncols2>
__launch_bounds__(D, 1)
static __global__ void ggml_cuda_fattn_kvarn_window_merge_kernel(
        const float2 * partial_ptr,
        float * acc_ptr,
        float2 * acc_meta_ptr,
        bool init,
        const uint3 ne01,
        const int ne02,
        const int ne12,
        const int gqa_ratio,
        const int ntiles_dst) {
    constexpr int ncols = ncols1 * ncols2;
    const int tile = blockIdx.x;
    const int jc = blockIdx.y;
    const int d = threadIdx.x;
    const int j = jc / ncols2;
    const int c = jc - j * ncols2;

    const int iter_j = (ne01.z + ncols1 - 1) / ncols1;
    const int iter_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
    const int sequence = tile / (iter_j * iter_z_gqa * ne12);
    const int rem0 = tile - sequence * iter_j * iter_z_gqa * ne12;
    const int z_KV = rem0 / (iter_j * iter_z_gqa);
    const int rem1 = rem0 - z_KV * iter_j * iter_z_gqa;
    const int zt_gqa = rem1 / iter_j;
    const int jt = rem1 - zt_gqa * iter_j;

    const int q = jt * ncols1 + j;
    if (q >= (int) ne01.z || zt_gqa * ncols2 + c >= gqa_ratio) {
        return;
    }

    const int q_head = z_KV * gqa_ratio + zt_gqa * ncols2 + c;
    const size_t out_off = ((size_t) sequence * ne01.z * ne02 + (size_t) q * ne02 + q_head) * D + d;
    const size_t row_off = ((size_t) sequence * ne01.z + q) * ne02 + q_head;

    const float2 part_meta = partial_ptr[((size_t) ntiles_dst + tile) * ncols + jc];
    const float2 * partial_data = partial_ptr + (size_t) ntiles_dst * (2 * ncols) +
        ((size_t) tile * ncols + jc) * (D / 2);
    const float part = ((const float *) partial_data)[d];

    if (init) {
        const bool has_data = part_meta.y > 0.0f;
        acc_ptr[out_off] = has_data ? part : 0.0f;
        if (d == 0) {
            acc_meta_ptr[row_off] = has_data ? part_meta : make_float2(0.0f, 0.0f);
        }
        return;
    }

    const float2 acc_meta = acc_meta_ptr[row_off];
    __syncthreads();
    if (part_meta.y <= 0.0f) {
        return;
    }
    if (acc_meta.y <= 0.0f) {
        acc_ptr[out_off] = part;
        if (d == 0) {
            acc_meta_ptr[row_off] = part_meta;
        }
        return;
    }

    const float max_new = fmaxf(acc_meta.x, part_meta.x);
    const float acc_diff = acc_meta.x - max_new;
    const float part_diff = part_meta.x - max_new;
    const float acc_scale = acc_diff >= SOFTMAX_FTZ_THRESHOLD ? expf(acc_diff) : 0.0f;
    const float part_scale = part_diff >= SOFTMAX_FTZ_THRESHOLD ? expf(part_diff) : 0.0f;
    acc_ptr[out_off] = acc_scale * acc_ptr[out_off] + part_scale * part;
    if (d == 0) {
        acc_meta_ptr[row_off] = make_float2(max_new, acc_scale * acc_meta.y + part_scale * part_meta.y);
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

template<int D, int ncols1, int ncols2>
__launch_bounds__(D, 1)
static __global__ void ggml_cuda_fattn_kvarn_window_single_finalize_kernel(
        const float2 * partial_ptr,
        float * dst_ptr,
        float2 * dst_meta_ptr,
        const uint3 ne01,
        const int ne02,
        const int ne12,
        const int gqa_ratio,
        const int ntiles_dst) {
    constexpr int ncols = ncols1 * ncols2;
    const int tile = blockIdx.x;
    const int jc = blockIdx.y;
    const int d = threadIdx.x;
    const int j = jc / ncols2;
    const int c = jc - j * ncols2;

    const int iter_j = (ne01.z + ncols1 - 1) / ncols1;
    const int iter_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
    const int sequence = tile / (iter_j * iter_z_gqa * ne12);
    const int rem0 = tile - sequence * iter_j * iter_z_gqa * ne12;
    const int z_KV = rem0 / (iter_j * iter_z_gqa);
    const int rem1 = rem0 - z_KV * iter_j * iter_z_gqa;
    const int zt_gqa = rem1 / iter_j;
    const int jt = rem1 - zt_gqa * iter_j;

    const int q = jt * ncols1 + j;
    if (q >= (int) ne01.z || zt_gqa * ncols2 + c >= gqa_ratio) {
        return;
    }

    const int q_head = z_KV * gqa_ratio + zt_gqa * ncols2 + c;
    const size_t out_off = ((size_t) sequence * ne01.z * ne02 + (size_t) q * ne02 + q_head) * D + d;
    const size_t row_off = ((size_t) sequence * ne01.z + q) * ne02 + q_head;

    const float2 part_meta = partial_ptr[((size_t) ntiles_dst + tile) * ncols + jc];
    const float2 * partial_data = partial_ptr + (size_t) ntiles_dst * (2 * ncols) +
        ((size_t) tile * ncols + jc) * (D / 2);
    const float part = ((const float *) partial_data)[d];
    dst_ptr[out_off] = part_meta.y > 0.0f ? part / part_meta.y : 0.0f;
    if (d == 0 && dst_meta_ptr != nullptr) {
        dst_meta_ptr[row_off] = part_meta;
    }
}

template <int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
static bool ggml_cuda_flash_attn_ext_mma_kvarn_windowed_case_impl(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_cuda_fattn_kvarn_plan & plan,
        size_t nbytes_shared_total) {
    if constexpr (DKQ != DV || (DKQ != 128 && DKQ != 256 && DKQ != 512)) {
        GGML_UNUSED_VARS(ctx, dst, plan, nbytes_shared_total);
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    float2 * const dst_meta = dst->src[8] != nullptr ? (float2 *) dst->src[8]->data : nullptr;
    const enum ggml_flash_attn_ext_kvarn_domain domain = ggml_cuda_fattn_kvarn_domain(dst);
    if (!ggml_cuda_fattn_kvarn_window_enabled() ||
            Q->ne[1] <= 1 || sinks != nullptr ||
            domain != GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V) {
        return false;
    }

#if !defined(GGML_USE_MUSA)
    // The partial kernel does not share the FlashAttention signature, so it is
    // passed as an opaque entry pointer instead of being cast to fattn_kernel_t.
    CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<const void *>(
            ggml_cuda_fattn_kvarn_window_f16_partial_kernel<DKQ, DV, ncols1, ncols2, use_logit_softcap>),
        cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
#endif

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    constexpr int ncols = ncols1 * ncols2;
    const int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols, cc);
    const int nthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols, cc);
    const int nwarps = nthreads / warp_size_host;
    const int window_chunk = ggml_cuda_fattn_kvarn_window_chunk(plan.n_kv);

    if (getenv("GGML_CUDA_FA_ROUTE_DEBUG") != nullptr) {
        fprintf(stderr,
            "CUDA_FA_ROUTE_EXEC_DISPATCH kernel=KVARN_WINDOWED "
            "Q=[%lld,%lld,%lld,%lld] n_kv=%d n_kv_heads=%d n_stream=%d q_stream=%lld "
            "chunk=%d ncols=[%d,%d] domain=%s bits=[%d,%d]\n",
            (long long) Q->ne[0], (long long) Q->ne[1],
            (long long) Q->ne[2], (long long) Q->ne[3],
            plan.n_kv, plan.n_kv_heads, plan.n_stream, (long long) Q->ne[3],
            window_chunk, ncols1, ncols2, ggml_cuda_fattn_kvarn_domain_name(dst), plan.k.bits, plan.v.bits);
        fflush(stderr);
    }

    const size_t n_desc = (size_t) plan.n_stream * plan.n_kv_heads;
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> k_desc(pool, n_desc);
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> v_desc(pool, n_desc);
    const bool k_original_domain = ggml_cuda_fattn_kvarn_k_original_domain(dst);
    const bool v_original_domain = ggml_cuda_fattn_kvarn_v_original_domain(dst);
    ggml_cuda_fattn_kvarn_init_descs(plan, k_desc.get(), v_desc.get(),
            k_original_domain ? 1 : 0, v_original_domain ? 1 : 0, stream);

    const int chunk_cap = window_chunk;
    ggml_cuda_pool_alloc<half> k_f16(pool, (size_t) chunk_cap * plan.n_kv_heads * plan.n_stream * DKQ);
    ggml_cuda_pool_alloc<half> v_f16(pool, (size_t) chunk_cap * plan.n_kv_heads * plan.n_stream * DV);

    const int gqa_ratio = Q->ne[2] / plan.n_kv_heads;
    const int ntiles_x = (Q->ne[1] + ncols1 - 1) / ncols1;
    const int ntiles_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
    const int ntiles_dst = ntiles_x * ntiles_z_gqa * plan.n_kv_heads * Q->ne[3];
    const int n_rows = (int) ((size_t) Q->ne[1] * Q->ne[2] * Q->ne[3]);
    const size_t partial_fixup_f2 = (size_t) ntiles_dst * (2 * ncols + ncols * (DV / 2));

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    const dim3 dequant_block((uint32_t) (2 * plan.slices * warp_size_host), 1, 1);
    const dim3 partial_block((uint32_t) warp_size_host,
            (uint32_t) nwarps, 1);
    const dim3 partial_grid((uint32_t) ntiles_dst, 1, 1);
    const dim3 merge_block(DV, 1, 1);
    const dim3 merge_grid((uint32_t) ntiles_dst, ncols, 1);
    const dim3 finalize_grid((uint32_t) n_rows, 1, 1);

    // The single-window path below may dequantize the full active window, but it
    // is bounded by window_chunk and kept as a transient scratch allocation.
    // It is not a graph-level KVarN materialize fallback.
    if (window_chunk >= plan.n_kv) {
        const int chunk_len = plan.n_kv;
        const dim3 dequant_grid((uint32_t) chunk_len, (uint32_t) plan.n_kv_heads, (uint32_t) plan.n_stream);
        ggml_cuda_kernel_launch_params dequant_params(dequant_grid, dequant_block, 0, stream);
        ggml_cuda_kernel_launch(ggml_cuda_fattn_kvarn_window_dequant_kernel<DKQ>, dequant_params,
            k_desc.get(), v_desc.get(), k_f16.get(), v_f16.get(), 0, chunk_len, plan.n_kv_heads);

        const char * mask_data = mask ? (const char *) mask->data : nullptr;
        if (Q->ne[1] >= 512) {
            ggml_tensor k_win = *dst->src[1];
            ggml_tensor v_win = *dst->src[2];

            k_win.type = GGML_TYPE_F16;
            k_win.data = k_f16.get();
            k_win.view_src = nullptr;
            k_win.view_offs = 0;
            k_win.ne[0] = DKQ;
            k_win.ne[1] = chunk_len;
            k_win.ne[2] = plan.n_kv_heads;
            k_win.ne[3] = plan.n_stream;
            k_win.nb[0] = sizeof(half);
            k_win.nb[1] = DKQ * (int64_t) sizeof(half);
            k_win.nb[2] = chunk_len * DKQ * (int64_t) sizeof(half);
            k_win.nb[3] = (int64_t) plan.n_kv_heads * chunk_len * DKQ * (int64_t) sizeof(half);

            v_win.type = GGML_TYPE_F16;
            v_win.data = v_f16.get();
            v_win.view_src = nullptr;
            v_win.view_offs = 0;
            v_win.ne[0] = DV;
            v_win.ne[1] = chunk_len;
            v_win.ne[2] = plan.n_kv_heads;
            v_win.ne[3] = plan.n_stream;
            v_win.nb[0] = sizeof(half);
            v_win.nb[1] = DV * (int64_t) sizeof(half);
            v_win.nb[2] = chunk_len * DV * (int64_t) sizeof(half);
            v_win.nb[3] = (int64_t) plan.n_kv_heads * chunk_len * DV * (int64_t) sizeof(half);

            fattn_kernel_t f16_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, false>;
#if !defined(GGML_USE_MUSA)
            CUDA_CHECK(cudaFuncSetAttribute(
                reinterpret_cast<ggml_cuda_fattn_kernel_attr_ptr_t>(f16_kernel),
                cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
#endif

            ggml_tensor * orig_k = dst->src[1];
            ggml_tensor * orig_v = dst->src[2];
            dst->src[1] = &k_win;
            dst->src[2] = &v_win;
            launch_fattn<DV, ncols1, ncols2>(
                ctx, dst, f16_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true, warp_size_host);
            dst->src[1] = orig_k;
            dst->src[2] = orig_v;
            return true;
        }

        ggml_cuda_pool_alloc<float2> partial(pool, partial_fixup_f2);
        ggml_cuda_kernel_launch_params partial_params(partial_grid, partial_block, nbytes_shared_total, stream);
        ggml_cuda_kernel_launch(ggml_cuda_fattn_kvarn_window_f16_partial_kernel<DKQ, DV, ncols1, ncols2, use_logit_softcap>,
            partial_params,
            (const char *) Q->data,
            (const char *) k_f16.get(),
            (const char *) v_f16.get(),
            mask_data,
            nullptr,
            partial.get(),
            scale, max_bias, m0, m1, n_head_log2, logit_softcap,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            DKQ, chunk_len, plan.n_kv_heads, plan.n_stream,
            DKQ * (int32_t) sizeof(half),
            chunk_len * DKQ * (int32_t) sizeof(half),
            (int64_t) plan.n_kv_heads * chunk_len * DKQ * (int64_t) sizeof(half),
            DV * (int32_t) sizeof(half),
            chunk_len * DV * (int32_t) sizeof(half),
            (int64_t) plan.n_kv_heads * chunk_len * DV * (int64_t) sizeof(half),
            mask ? (int32_t) mask->ne[3] : 1,
            mask ? (int32_t) mask->nb[1] : 0,
            mask ? (int64_t) mask->nb[3] : 0);
        ggml_cuda_kernel_launch_params single_finalize_params(merge_grid, merge_block, 0, stream);
        ggml_cuda_kernel_launch(ggml_cuda_fattn_kvarn_window_single_finalize_kernel<DV, ncols1, ncols2>, single_finalize_params,
            partial.get(), (float *) dst->data, dst_meta, ne01, Q->ne[2], plan.n_kv_heads, gqa_ratio, ntiles_dst);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    ggml_cuda_pool_alloc<float2> partial(pool, partial_fixup_f2);
    ggml_cuda_pool_alloc<float2> acc_meta(pool, n_rows);

    bool init = true;
    for (int chunk_begin = 0; chunk_begin < plan.n_kv; chunk_begin += window_chunk) {
        const int chunk_len = std::min(window_chunk, plan.n_kv - chunk_begin);
        const dim3 dequant_grid((uint32_t) chunk_len, (uint32_t) plan.n_kv_heads, (uint32_t) plan.n_stream);
        ggml_cuda_kernel_launch_params dequant_params(dequant_grid, dequant_block, 0, stream);
        ggml_cuda_kernel_launch(ggml_cuda_fattn_kvarn_window_dequant_kernel<DKQ>, dequant_params,
            k_desc.get(), v_desc.get(), k_f16.get(), v_f16.get(), chunk_begin, chunk_len, plan.n_kv_heads);

        const char * mask_data = mask ? (const char *) mask->data + (size_t) chunk_begin * mask->nb[0] : nullptr;
        ggml_cuda_kernel_launch_params partial_params(partial_grid, partial_block, nbytes_shared_total, stream);
        ggml_cuda_kernel_launch(ggml_cuda_fattn_kvarn_window_f16_partial_kernel<DKQ, DV, ncols1, ncols2, use_logit_softcap>,
            partial_params,
            (const char *) Q->data,
            (const char *) k_f16.get(),
            (const char *) v_f16.get(),
            mask_data,
            nullptr,
            partial.get(),
            scale, max_bias, m0, m1, n_head_log2, logit_softcap,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            DKQ, chunk_len, plan.n_kv_heads, plan.n_stream,
            DKQ * (int32_t) sizeof(half),
            chunk_len * DKQ * (int32_t) sizeof(half),
            (int64_t) plan.n_kv_heads * chunk_len * DKQ * (int64_t) sizeof(half),
            DV * (int32_t) sizeof(half),
            chunk_len * DV * (int32_t) sizeof(half),
            (int64_t) plan.n_kv_heads * chunk_len * DV * (int64_t) sizeof(half),
            mask ? (int32_t) mask->ne[3] : 1,
            mask ? (int32_t) mask->nb[1] : 0,
            mask ? (int64_t) mask->nb[3] : 0);

        ggml_cuda_kernel_launch_params merge_params(merge_grid, merge_block, 0, stream);
        ggml_cuda_kernel_launch(ggml_cuda_fattn_kvarn_window_merge_kernel<DV, ncols1, ncols2>, merge_params,
            partial.get(), (float *) dst->data, acc_meta.get(), init, ne01, Q->ne[2], plan.n_kv_heads, gqa_ratio, ntiles_dst);
        init = false;
    }

    ggml_cuda_kernel_launch_params finalize_params(finalize_grid, merge_block, 0, stream);
    ggml_cuda_kernel_launch(ggml_cuda_fattn_kvarn_window_finalize_kernel<DV>, finalize_params,
        (float *) dst->data, acc_meta.get(), dst_meta, n_rows);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <int DKQ, int DV, int ncols1, int ncols2>
static bool ggml_cuda_flash_attn_ext_mma_kvarn_windowed_case(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_cuda_fattn_kvarn_plan & plan,
        size_t nbytes_shared_total,
        bool use_logit_softcap) {
    if constexpr (DKQ != DV || (DKQ != 128 && DKQ != 256 && DKQ != 512)) {
        GGML_UNUSED_VARS(ctx, dst, plan, nbytes_shared_total, use_logit_softcap);
        return false;
    } else {
        if (use_logit_softcap) {
            return ggml_cuda_flash_attn_ext_mma_kvarn_windowed_case_impl<DKQ, DV, ncols1, ncols2, true>(
                ctx, dst, plan, nbytes_shared_total);
        }
        return ggml_cuda_flash_attn_ext_mma_kvarn_windowed_case_impl<DKQ, DV, ncols1, ncols2, false>(
            ctx, dst, plan, nbytes_shared_total);
    }
}

template <int DKQ, int DV, int ncols1, int ncols2>
static size_t ggml_cuda_fattn_kvarn_mma_shared_bytes(
        int cc,
        int warp_size_host,
        bool has_original_domain) {
    constexpr int ncols = ncols1 * ncols2;
    const int nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg      = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);
    const int nwarps         = nthreads / warp_size_host;
    const int cols_per_warp  = std::min(ncols, get_cols_per_warp(cc));

    const size_t nbytes_shared_KV = nbatch_fa * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q = ncols * (DKQ/2 + 4) * sizeof(half2);
    const size_t nbytes_shared_mask = ncols1 * (nbatch_fa/2 + 4) * sizeof(half2);
    const size_t nbytes_shared_combine = nwarps * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
    const size_t nbytes_shared_kvarn_rotated =
        6 * std::max(DKQ, DV) * sizeof(half) +
        2 * (std::max(DKQ, DV) / GGML_CUDA_FATTN_KVARN_DIM) * sizeof(int);
    const size_t nbytes_shared_kvarn_original =
        3 * GGML_CUDA_FATTN_KVARN_DIM * sizeof(half) +
        2 * nwarps * GGML_CUDA_FATTN_KVARN_DIM * sizeof(float);
    const size_t nbytes_shared_kvarn = has_original_domain ?
        nbytes_shared_kvarn_original : nbytes_shared_kvarn_rotated;
    const size_t nbytes_shared_KV_mask_kvarn = nbytes_shared_KV + nbytes_shared_mask + nbytes_shared_kvarn;
    return std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q, nbytes_shared_KV_mask_kvarn) :
                 nbytes_shared_Q + nbytes_shared_KV_mask_kvarn);
}

#if !defined(GGML_USE_MUSA)
template <int DKQ, int DV, int ncols1, int ncols2>
bool ggml_cuda_fattn_kvarn_wide_mma_supported(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst) {
    const int device = ctx.device;
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const bool k_original_domain = ggml_cuda_fattn_kvarn_k_original_domain(dst);
    const bool v_original_domain = ggml_cuda_fattn_kvarn_v_original_domain(dst);
    const int cache_key = 4 * (int) k_original_domain + 2 * (int) v_original_domain + (logit_softcap != 0.0f);
    static std::atomic<int> cached_active_blocks[GGML_CUDA_MAX_DEVICES][8] = {};
    const int cached = cached_active_blocks[device][cache_key].load(std::memory_order_relaxed);
    if (cached != 0) {
        return cached > 0;
    }

    const auto & device_info = ggml_cuda_info().devices[device];
    const size_t nbytes_shared_total = ggml_cuda_fattn_kvarn_mma_shared_bytes<DKQ, DV, ncols1, ncols2>(
        device_info.cc, device_info.warp_size, k_original_domain || v_original_domain);
    if (nbytes_shared_total > device_info.smpbo) {
        cached_active_blocks[device][cache_key].store(-1, std::memory_order_relaxed);
        return false;
    }

    fattn_kernel_t fattn_kernel = logit_softcap == 0.0f ?
        ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel<DKQ, DV, ncols1, ncols2, false>(
            k_original_domain, v_original_domain) :
        ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel<DKQ, DV, ncols1, ncols2, true>(
            k_original_domain, v_original_domain);
    CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<ggml_cuda_fattn_kernel_attr_ptr_t>(fattn_kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        nbytes_shared_total));

    const int nthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1 * ncols2, device_info.cc);
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        reinterpret_cast<ggml_cuda_fattn_kernel_attr_ptr_t>(fattn_kernel),
        nthreads,
        nbytes_shared_total));
    cached_active_blocks[device][cache_key].store(active_blocks > 0 ? active_blocks : -1, std::memory_order_relaxed);
    return active_blocks > 0;
}
#endif

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_kvarn_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_fattn_kvarn_plan plan;
    GGML_ASSERT(ggml_cuda_fattn_kvarn_view_supported(ctx.device, dst, &plan));

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps = nthreads / warp_size_host;
    const bool k_original_domain = ggml_cuda_fattn_kvarn_k_original_domain(dst);
    const bool v_original_domain = ggml_cuda_fattn_kvarn_v_original_domain(dst);
    const bool has_original_domain = k_original_domain || v_original_domain;

    const size_t nbytes_shared_KV = nbatch_fa * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q = ncols * (DKQ/2 + 4) * sizeof(half2);
    const size_t nbytes_shared_mask = ncols1 * (nbatch_fa/2 + 4) * sizeof(half2);
    const size_t nbytes_shared_combine = nwarps * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
    const size_t nbytes_shared_total = ggml_cuda_fattn_kvarn_mma_shared_bytes<DKQ, DV, ncols1, ncols2>(
        cc, warp_size_host, has_original_domain);
    const int nstages = ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, cc);
    const size_t nbytes_shared_KV_f16_1stage = nbatch_fa * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_KV_f16_2stage = nbatch_fa * (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_KV_f16 = nstages <= 1 ? nbytes_shared_KV_f16_1stage : nbytes_shared_KV_f16_2stage;
    const size_t nbytes_shared_total_f16 = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q, nbytes_shared_KV_f16 + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV_f16 + nbytes_shared_mask);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    const size_t n_desc = (size_t) plan.n_stream * plan.n_kv_heads;
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> k_desc(pool, n_desc);
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> v_desc(pool, n_desc);
    ggml_cuda_fattn_kvarn_init_descs(plan, k_desc.get(), v_desc.get(),
            k_original_domain ? 1 : 0, v_original_domain ? 1 : 0, stream);

    ggml_tensor K_desc = *dst->src[1];
    ggml_tensor V_desc = *dst->src[2];
    K_desc.data = k_desc.get();
    V_desc.data = v_desc.get();
    K_desc.type = GGML_TYPE_F16;
    V_desc.type = GGML_TYPE_F16;
    K_desc.view_src = nullptr;
    V_desc.view_src = nullptr;
    K_desc.view_offs = 0;
    V_desc.view_offs = 0;
    K_desc.nb[0] = sizeof(half);
    V_desc.nb[0] = sizeof(half);
    K_desc.nb[1] = 0;
    V_desc.nb[1] = 0;
    K_desc.nb[2] = sizeof(ggml_cuda_fattn_kvarn_desc);
    V_desc.nb[2] = sizeof(ggml_cuda_fattn_kvarn_desc);
    K_desc.nb[3] = sizeof(ggml_cuda_fattn_kvarn_desc) * plan.n_kv_heads;
    V_desc.nb[3] = sizeof(ggml_cuda_fattn_kvarn_desc) * plan.n_kv_heads;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif
    fattn_kernel_t fattn_kernel;
    fattn_kernel_t fattn_kernel_no_softcap = ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel<DKQ, DV, ncols1, ncols2, false>(
        k_original_domain, v_original_domain);
    fattn_kernel_t fattn_kernel_softcap = ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel<DKQ, DV, ncols1, ncols2, true>(
        k_original_domain, v_original_domain);
    if (logit_softcap == 0.0f) {
        fattn_kernel = fattn_kernel_no_softcap;
#if !defined(GGML_USE_MUSA)
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
#endif
    } else {
        fattn_kernel = fattn_kernel_softcap;
#if !defined(GGML_USE_MUSA)
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
#endif
    }

    if (ggml_cuda_flash_attn_ext_mma_kvarn_windowed_case<DKQ, DV, ncols1, ncols2>(
            ctx, dst, plan, nbytes_shared_total_f16, logit_softcap != 0.0f)) {
        return;
    }

    ggml_tensor * orig_k = dst->src[1];
    ggml_tensor * orig_v = dst->src[2];
    dst->src[1] = &K_desc;
    dst->src[2] = &V_desc;
    // need_f16_K=false, need_f16_V=false: KVarN K/V stay descriptor-backed.
    // Mixed prefill reconstructs original-domain V in the native loader.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true, warp_size_host);
    dst->src[1] = orig_k;
    dst->src[2] = orig_v;
}

#define DECL_FATTN_MMA_KVARN_CASE(DKQ, DV, ncols1, ncols2)                         \
    template void ggml_cuda_flash_attn_ext_mma_kvarn_case                          \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)
