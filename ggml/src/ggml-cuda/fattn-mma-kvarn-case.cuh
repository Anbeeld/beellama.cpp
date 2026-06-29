#pragma once

#include "fattn-mma-f16.cuh"
#include "fattn-mma-kvarn-case-decl.cuh"
#include "fattn-mma-kvarn-impl.cuh"

static __global__ void ggml_cuda_fattn_kvarn_materialize_v_original_kernel(
        const ggml_cuda_fattn_kvarn_desc * __restrict__ v_descs,
        half * __restrict__ v_f16,
        const int n_kv,
        const int n_kv_heads,
        const int head_dim,
        const int slices) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int tid  = threadIdx.x;
    const int warp = tid / warp_size;
    const int lane = tid - warp * warp_size;
    const int warps_per_block = blockDim.x / warp_size;

    const int row = blockIdx.x * warps_per_block + warp;
    if (row >= n_kv) {
        return;
    }

    const int stream = blockIdx.z;
    const int head_slice = blockIdx.y;
    const int head = head_slice / slices;
    const int slice = head_slice - head * slices;
    const ggml_cuda_fattn_kvarn_desc & desc = v_descs[(size_t) stream * n_kv_heads + head];

    extern __shared__ float scratch[];
    float * row0 = scratch + warp * GGML_CUDA_FATTN_KVARN_DIM;
    float * row1 = scratch + (warps_per_block + warp) * GGML_CUDA_FATTN_KVARN_DIM;

    const bool loaded_from_stage =
        ggml_cuda_fattn_kvarn_load_rotated_slice_warp(desc, row, slice, true, row0, lane);
    const bool stage_original = loaded_from_stage && desc.value;
    float * orig = stage_original ? row0 : ggml_cuda_fattn_kvarn_inverse_wht_128_warp(row0, row1, lane);

    const int dim_base = slice * GGML_CUDA_FATTN_KVARN_DIM;
    const int64_t out_base =
        (((int64_t) stream * n_kv_heads + head) * n_kv + row) * head_dim + dim_base;
    for (int d = lane; d < GGML_CUDA_FATTN_KVARN_DIM; d += warp_size) {
        v_f16[out_base + d] = __float2half(orig[d]);
    }
}

template <int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
static inline fattn_kernel_t ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel(
        bool k_original_domain,
        bool v_original_domain) {
    constexpr bool V_is_K_view = false;

    if (k_original_domain) {
        if (v_original_domain) {
            return flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view,
                GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE, GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE>;
        }

        return flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view,
            GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE, GGML_CUDA_FATTN_KVARN_TYPE>;
    }

    if (v_original_domain) {
        return flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view,
            GGML_CUDA_FATTN_KVARN_TYPE, GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE>;
    }

    return flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view,
        GGML_CUDA_FATTN_KVARN_TYPE, GGML_CUDA_FATTN_KVARN_TYPE>;
}

template <int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
static inline fattn_kernel_t ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel_materialized_v() {
    constexpr bool V_is_K_view = false;

    return flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view,
        GGML_CUDA_FATTN_KVARN_TYPE, GGML_TYPE_F16>;
}

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
    const bool materialize_v_original = !k_original_domain && v_original_domain;
    const bool has_original_domain = k_original_domain || (v_original_domain && !materialize_v_original);

    const size_t nbytes_shared_KV = nbatch_fa * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q = ncols * (DKQ/2 + 4) * sizeof(half2);
    const size_t nbytes_shared_mask = ncols1 * (nbatch_fa/2 + 4) * sizeof(half2);
    const size_t nbytes_shared_combine = nwarps * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
    const size_t nbytes_shared_kvarn_rotated =
        2 * GGML_CUDA_FATTN_KVARN_DIM * sizeof(half);
    const size_t nbytes_shared_kvarn_original =
        3 * GGML_CUDA_FATTN_KVARN_DIM * sizeof(half) +
        2 * nwarps * GGML_CUDA_FATTN_KVARN_DIM * sizeof(float);
    const size_t nbytes_shared_kvarn = has_original_domain ?
        nbytes_shared_kvarn_original : nbytes_shared_kvarn_rotated;
    const size_t nbytes_shared_KV_mask_kvarn = nbytes_shared_KV + nbytes_shared_mask + nbytes_shared_kvarn;
    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q, nbytes_shared_KV_mask_kvarn) :
                 nbytes_shared_Q + nbytes_shared_KV_mask_kvarn);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    const size_t n_desc = (size_t) plan.n_stream * plan.n_kv_heads;
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> k_desc(pool, n_desc);
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> v_desc(pool, n_desc);
    ggml_cuda_fattn_kvarn_init_descs(plan, k_desc.get(), v_desc.get(),
            k_original_domain ? 1 : 0, v_original_domain ? 1 : 0, stream);

    ggml_cuda_pool_alloc<half> v_materialized(pool);
    ggml_tensor V_f16;
    if (materialize_v_original) {
        v_materialized.alloc((size_t) plan.n_stream * plan.n_kv_heads * plan.n_kv * DV);
        const int materialize_warps = std::max(1, std::min(nwarps, 8));
        const dim3 blocks(
            (unsigned) ((plan.n_kv + materialize_warps - 1) / materialize_warps),
            (unsigned) (plan.n_kv_heads * plan.slices),
            (unsigned) plan.n_stream);
        const dim3 threads((unsigned) (warp_size_host * materialize_warps), 1, 1);
        const size_t nbytes_shared_materialize =
            2 * (size_t) materialize_warps * GGML_CUDA_FATTN_KVARN_DIM * sizeof(float);
        ggml_cuda_fattn_kvarn_materialize_v_original_kernel<<<blocks, threads, nbytes_shared_materialize, stream>>>(
            v_desc.get(), v_materialized.get(), plan.n_kv, plan.n_kv_heads, DV, plan.slices);
        CUDA_CHECK(cudaGetLastError());

        V_f16 = *dst->src[2];
        V_f16.data = v_materialized.get();
        V_f16.type = GGML_TYPE_F16;
        V_f16.view_src = nullptr;
        V_f16.view_offs = 0;
        V_f16.op = GGML_OP_NONE;
        V_f16.nb[0] = sizeof(half);
        V_f16.nb[1] = DV * sizeof(half);
        V_f16.nb[2] = (int64_t) plan.n_kv * DV * sizeof(half);
        V_f16.nb[3] = (int64_t) plan.n_kv_heads * plan.n_kv * DV * sizeof(half);
    }

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
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = materialize_v_original ?
            ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel_materialized_v<DKQ, DV, ncols1, ncols2, use_logit_softcap>() :
            ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel<DKQ, DV, ncols1, ncols2, use_logit_softcap>(
                k_original_domain, v_original_domain);
#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
#endif
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = materialize_v_original ?
            ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel_materialized_v<DKQ, DV, ncols1, ncols2, use_logit_softcap>() :
            ggml_cuda_flash_attn_ext_mma_kvarn_select_kernel<DKQ, DV, ncols1, ncols2, use_logit_softcap>(
                k_original_domain, v_original_domain);
#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
#endif
    }

    ggml_tensor * orig_k = dst->src[1];
    ggml_tensor * orig_v = dst->src[2];
    dst->src[1] = &K_desc;
    dst->src[2] = materialize_v_original ? &V_f16 : &V_desc;
    // need_f16_K=false, need_f16_V=false: KVarN K stays descriptor-backed; mixed
    // prefill materializes original-domain V above when the kernel consumes F16 V.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true, warp_size_host);
    dst->src[1] = orig_k;
    dst->src[2] = orig_v;
}

#define DECL_FATTN_MMA_KVARN_CASE(DKQ, DV, ncols1, ncols2)                         \
    template void ggml_cuda_flash_attn_ext_mma_kvarn_case                          \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)  \
