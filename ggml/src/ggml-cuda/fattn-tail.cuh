#pragma once

#include "fattn-common.cuh"
#include "fattn-kvarn-dispatch.cuh"

template<typename T>
static __global__ void k_flash_attn_ext_tail_pack_arenas(
        const char * src, const char * current, T * dst, const int32_t * run_desc,
        int d, int tail_stride, int n_head, int n_active,
        int desc_stride, int history_slots,
        size_t nb0, size_t nb1, size_t nb2,
        size_t current_nb0, size_t current_nb1, size_t current_nb2) {
    const size_t n = size_t(d)*tail_stride*n_head*n_active;
    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < n; i += size_t(blockDim.x)*gridDim.x) {
        size_t rem = i;
        const int id = int(rem % d); rem /= d;
        const int it = int(rem % tail_stride); rem /= tail_stride;
        const int ih = int(rem % n_head); rem /= n_head;
        const int ia = int(rem);
        const int32_t * desc = run_desc + desc_stride*ia;
        T value = {};
        if (it < desc[4]) {
            const int slot = desc[6 + it];
            const bool from_current = current && slot >= history_slots;
            const int row = from_current ? slot - history_slots : slot;
            const char * base = from_current ? current : src;
            value = *reinterpret_cast<const T *>(base +
                size_t(id)*(from_current ? current_nb0 : nb0) +
                size_t(row)*(from_current ? current_nb1 : nb1) +
                size_t(ih)*(from_current ? current_nb2 : nb2));
        }
        dst[i] = value;
    }
}

template<typename V>
static __global__ void k_flash_attn_ext_tail_pack_arenas_vec(
        const char * src, const char * current, V * dst, const int32_t * run_desc,
        int n_vec, int tail_stride, int n_head, int n_active,
        int desc_stride, int history_slots,
        size_t nb1, size_t nb2, size_t current_nb1, size_t current_nb2) {
    const size_t n = size_t(n_vec)*tail_stride*n_head*n_active;
    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < n; i += size_t(blockDim.x)*gridDim.x) {
        size_t rem = i;
        const int iv = int(rem % n_vec); rem /= n_vec;
        const int it = int(rem % tail_stride); rem /= tail_stride;
        const int ih = int(rem % n_head); rem /= n_head;
        const int ia = int(rem);
        const int32_t * desc = run_desc + desc_stride*ia;
        V value = {};
        if (it < desc[4]) {
            const int slot = desc[6 + it];
            const bool from_current = current && slot >= history_slots;
            const int row = from_current ? slot - history_slots : slot;
            const char * base = from_current ? current : src;
            value = *reinterpret_cast<const V *>(base + size_t(iv)*sizeof(V) +
                    size_t(row)*(from_current ? current_nb1 : nb1) +
                    size_t(ih)*(from_current ? current_nb2 : nb2));
        }
        dst[i] = value;
    }
}

static __global__ void k_flash_attn_ext_tail_pack_body_rows(
        const char * src, uint8_t * dst, const int32_t * run_desc,
        int row_bytes, int n_kv, int body_stride, int n_head, int n_active,
        int desc_stride, int body_map_offset, size_t nb1, size_t nb2, size_t nb3) {
    const size_t n = size_t(row_bytes)*body_stride*n_head*n_active;
    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < n; i += size_t(blockDim.x)*gridDim.x) {
        size_t rem = i;
        const int ib = int(rem % row_bytes); rem /= row_bytes;
        const int it = int(rem % body_stride); rem /= body_stride;
        const int ih = int(rem % n_head); rem /= n_head;
        const int ia = int(rem);
        const int32_t * desc = run_desc + desc_stride*ia;
        uint8_t value = 0;
        if (it < desc[5]) {
            const int flat = desc[body_map_offset + it];
            const int is = flat/n_kv;
            const int ik = flat - is*n_kv;
            value = *(reinterpret_cast<const uint8_t *>(src) +
                    size_t(ik)*nb1 + size_t(ih)*nb2 + size_t(is)*nb3 + ib);
        }
        dst[i] = value;
    }
}

static __global__ void k_flash_attn_ext_tail_pack_body_mask(
        const half * mask, const int32_t * query_order, const int32_t * run_desc,
        half * mask_packed, int n_kv, int n_query, int n_stream,
        int body_stride, int q_max, int n_active, int desc_stride, int body_map_offset,
        size_t m_nb1, size_t m_nb3) {
    const size_t n = size_t(body_stride)*q_max*n_active;
    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < n; i += size_t(blockDim.x)*gridDim.x) {
        size_t rem = i;
        const int it = int(rem % body_stride); rem /= body_stride;
        const int iq_packed = int(rem % q_max); rem /= q_max;
        const int ia = int(rem);
        const int iq_global = query_order[ia*q_max + iq_packed];
        const int32_t * desc = run_desc + desc_stride*ia;
        half value = __float2half(-INFINITY);
        if (iq_global >= 0 && iq_global < n_query*n_stream && it < desc[5]) {
            const int iq = iq_global % n_query;
            const int flat = desc[body_map_offset + it];
            const int is = flat/n_kv;
            const int ik = flat - is*n_kv;
            value = *reinterpret_cast<const half *>(reinterpret_cast<const char *>(mask) +
                    size_t(ik)*sizeof(half) + size_t(iq)*m_nb1 + size_t(is)*m_nb3);
        }
        mask_packed[i] = value;
    }
}

static __global__ void k_flash_attn_ext_tail_pack_q_mask(
        const float * q, const half * mask, const int32_t * query_order,
        float * q_packed, half * mask_packed,
        int d, int n_query, int n_head, int n_stream,
        int mask_stride, int tail_stride, int q_max, int n_active,
        size_t q_nb1, size_t q_nb2, size_t q_nb3,
        size_t m_nb1, size_t m_nb3) {
    const size_t n_q = size_t(d)*q_max*n_head*n_active;
    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < n_q; i += size_t(blockDim.x)*gridDim.x) {
        size_t rem = i;
        const int id = int(rem % d); rem /= d;
        const int iq_packed = int(rem % q_max); rem /= q_max;
        const int ih = int(rem % n_head); rem /= n_head;
        const int ia = int(rem);
        const int iq_global = query_order[ia*q_max + iq_packed];
        float value = 0.0f;
        if (iq_global >= 0 && iq_global < n_query*n_stream) {
            const int is = iq_global/n_query;
            const int iq = iq_global - is*n_query;
            value = *reinterpret_cast<const float *>(reinterpret_cast<const char *>(q) +
                size_t(id)*sizeof(float) + size_t(iq)*q_nb1 + size_t(ih)*q_nb2 + size_t(is)*q_nb3);
        }
        q_packed[i] = value;
    }

    const size_t n_m = size_t(tail_stride)*q_max*n_active;
    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < n_m; i += size_t(blockDim.x)*gridDim.x) {
        size_t rem = i;
        const int it = int(rem % tail_stride); rem /= tail_stride;
        const int iq_packed = int(rem % q_max); rem /= q_max;
        const int ia = int(rem);
        const int iq_global = query_order[ia*q_max + iq_packed];
        half value = __float2half(-INFINITY);
        if (iq_global >= 0 && iq_global < n_query*n_stream && it < mask_stride) {
            const int is = iq_global/n_query;
            const int iq = iq_global - is*n_query;
            value = *reinterpret_cast<const half *>(reinterpret_cast<const char *>(mask) +
                size_t(it)*sizeof(half) + size_t(iq)*m_nb1 + size_t(is)*m_nb3);
        }
        mask_packed[i] = value;
    }
}

template<typename T>
static __device__ __forceinline__ float tail_value_to_float(T value);

template<>
__device__ __forceinline__ float tail_value_to_float<half>(half value) {
    return __half2float(value);
}

template<>
__device__ __forceinline__ float tail_value_to_float<nv_bfloat16>(nv_bfloat16 value) {
    return __bfloat162float(value);
}

// Small indexed tails are faster when consumed in place.  This kernel avoids
// materializing compact K/V/Q/mask tensors and combines the tail partial with
// the already-computed body partial in one launch.  One warp computes each KQ
// dot while all threads cooperatively normalize and accumulate V.
template<typename TK, typename TV, int max_tail>
static __global__ void k_flash_attn_ext_tail_indexed_small(
        const float * q, const char * kt, const char * vt,
        const char * kt_current, const char * vt_current, const half * mt,
        const int32_t * query_order, const int32_t * run_desc,
        const float * body, const float2 * body_meta, float * dst,
        int d_k, int d_v, int n_query, int n_head, int n_stream,
        int q_max, int n_active, int n_head_k, int n_head_v, int desc_stride,
        float scale, float max_bias, float logit_softcap,
        size_t q_nb1, size_t q_nb2, size_t q_nb3,
        int history_slots,
        size_t kt_nb0, size_t kt_nb1, size_t kt_nb2,
        size_t vt_nb0, size_t vt_nb1, size_t vt_nb2,
        size_t ktc_nb0, size_t ktc_nb1, size_t ktc_nb2,
        size_t vtc_nb0, size_t vtc_nb1, size_t vtc_nb2,
        size_t mt_nb1, size_t mt_nb3,
        bool tail_bodyless, bool body_packed,
        size_t body_nb1, size_t body_nb2, size_t body_nb3,
        size_t dst_nb1, size_t dst_nb2, size_t dst_nb3) {
    const int iq_packed = blockIdx.x;
    const int ih = blockIdx.y;
    const int ia = blockIdx.z;
    if (iq_packed >= q_max || ih >= n_head || ia >= n_active) {
        return;
    }
    const int iq_global = query_order[ia*q_max + iq_packed];
    if (iq_global < 0 || iq_global >= n_query*n_stream) {
        return;
    }
    const int is = iq_global/n_query;
    const int iq = iq_global - is*n_query;
    const int32_t * desc = run_desc + desc_stride*ia;
    const int n_tail = desc[4];
    if (n_tail < 0 || n_tail > max_tail) {
        return;
    }

    __shared__ float scores[max_tail];
    __shared__ float reduction[256];
    __shared__ float tail_max_shared;
    __shared__ float tail_sum_shared;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int ih_k = ih/(n_head/n_head_k);
    const int ih_v = ih/(n_head/n_head_v);
    const float * qrow = reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(q) + size_t(iq)*q_nb1 + size_t(ih)*q_nb2 + size_t(is)*q_nb3);

    const uint32_t nh_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = exp2f(-max_bias/float(nh_log2));
    const float m1 = exp2f(-(max_bias/2.0f)/float(nh_log2));
    const float slope = max_bias > 0.0f ?
            (ih < int(nh_log2) ? powf(m0, ih + 1) : powf(m1, 2*(ih - int(nh_log2)) + 1)) : 1.0f;

    for (int token = warp; token < n_tail; token += 8) {
        const int slot = desc[6 + token];
        const bool from_current = kt_current && slot >= history_slots;
        const int row = from_current ? slot - history_slots : slot;
        const char * k_source = from_current ? kt_current : kt;
        float dot = 0.0f;
        for (int d = lane; d < d_k; d += 32) {
            const TK kval = *reinterpret_cast<const TK *>(
                    k_source +
                    size_t(d)*(from_current ? ktc_nb0 : kt_nb0) +
                    size_t(row)*(from_current ? ktc_nb1 : kt_nb1) +
                    size_t(ih_k)*(from_current ? ktc_nb2 : kt_nb2));
            dot += qrow[d]*tail_value_to_float(kval);
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            const float mask = __half2float(*reinterpret_cast<const half *>(
                    reinterpret_cast<const char *>(mt) + size_t(token)*sizeof(half) +
                    size_t(iq)*mt_nb1 + size_t(is)*mt_nb3));
            float score = dot*scale;
            if (logit_softcap != 0.0f) {
                score = logit_softcap*tanhf(score/logit_softcap);
            }
            scores[token] = score + slope*mask;
        }
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int token = tid; token < n_tail; token += blockDim.x) {
        local_max = fmaxf(local_max, scores[token]);
    }
    reduction[tid] = local_max;
    __syncthreads();
    for (int width = 128; width > 0; width >>= 1) {
        if (tid < width) {
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + width]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        tail_max_shared = reduction[0];
    }
    __syncthreads();

    float weight = 0.0f;
    for (int token = tid; token < n_tail; token += blockDim.x) {
        const float token_weight = isfinite(scores[token]) && isfinite(tail_max_shared) ?
                expf(scores[token] - tail_max_shared) : 0.0f;
        scores[token] = token_weight;
        weight += token_weight;
    }
    reduction[tid] = weight;
    __syncthreads();
    for (int width = 128; width > 0; width >>= 1) {
        if (tid < width) {
            reduction[tid] += reduction[tid + width];
        }
        __syncthreads();
    }
    if (tid == 0) {
        tail_sum_shared = reduction[0];
    }
    __syncthreads();

    const size_t body_row_index = body_packed ?
            (size_t(ia)*q_max + iq_packed)*n_head + ih :
            (size_t(is)*n_query + iq)*n_head + ih;
    const float2 bm = tail_bodyless ? make_float2(0.0f, 0.0f) : body_meta[body_row_index];
    const bool bv = bm.y > 0.0f && isfinite(bm.x) && isfinite(bm.y);
    const bool tv = tail_sum_shared > 0.0f && isfinite(tail_max_shared);
    const float global_max = bv && tv ? fmaxf(bm.x, tail_max_shared) :
            (bv ? bm.x : (tv ? tail_max_shared : -INFINITY));
    const float wb = bv ? bm.y*expf(bm.x - global_max) : 0.0f;
    const float wt = tv ? expf(tail_max_shared - global_max) : 0.0f;
    const float denom = wb + tail_sum_shared*wt;
    const char * brow = tail_bodyless ? nullptr : reinterpret_cast<const char *>(body) + size_t(ih)*body_nb1 +
            size_t(body_packed ? iq_packed : iq)*body_nb2 + size_t(body_packed ? ia : is)*body_nb3;
    char * drow = reinterpret_cast<char *>(dst) + size_t(ih)*dst_nb1 +
            size_t(iq)*dst_nb2 + size_t(is)*dst_nb3;
    for (int d = tid; d < d_v; d += blockDim.x) {
        float tail_acc = 0.0f;
        for (int token = 0; token < n_tail; ++token) {
            const int slot = desc[6 + token];
            const bool from_current = vt_current && slot >= history_slots;
            const int row = from_current ? slot - history_slots : slot;
            const char * v_source = from_current ? vt_current : vt;
            const TV vval = *reinterpret_cast<const TV *>(
                    v_source +
                    size_t(d)*(from_current ? vtc_nb0 : vt_nb0) +
                    size_t(row)*(from_current ? vtc_nb1 : vt_nb1) +
                    size_t(ih_v)*(from_current ? vtc_nb2 : vt_nb2));
            tail_acc += tail_value_to_float(vval)*scores[token];
        }
        const float body_value = bv ? *reinterpret_cast<const float *>(brow + size_t(d)*sizeof(float)) : 0.0f;
        *reinterpret_cast<float *>(drow + size_t(d)*sizeof(float)) =
                denom > 0.0f ? (body_value*wb + tail_acc*wt)/denom : 0.0f;
    }
}

static __global__ void k_flash_attn_ext_tail_partials_merge(
        const float * body, const float * tail, float * dst,
        const float2 * body_meta, const float2 * tail_meta, const int32_t * query_order,
        int d, int n_query, int n_head, int n_stream, int q_max, int n_active,
        bool body_packed,
        size_t body_nb1, size_t body_nb2, size_t body_nb3,
        size_t tail_nb1, size_t tail_nb2, size_t tail_nb3,
        size_t dst_nb1, size_t dst_nb2, size_t dst_nb3) {
    const int iq_packed = blockIdx.x;
    const int ih = blockIdx.y;
    const int ia = blockIdx.z;
    if (iq_packed >= q_max || ih >= n_head || ia >= n_active) {
        return;
    }
    const int iq_global = query_order[ia*q_max + iq_packed];
    if (iq_global < 0 || iq_global >= n_query*n_stream) {
        return;
    }
    const int is = iq_global/n_query;
    const int iq = iq_global - is*n_query;
    const size_t body_row_index = body_packed ?
        (size_t(ia)*q_max + iq_packed)*n_head + ih :
        (size_t(is)*n_query + iq)*n_head + ih;
    const size_t tail_row_index = (size_t(ia)*q_max + iq_packed)*n_head + ih;
    const float2 bm = body_meta[body_row_index];
    const float2 tm = tail_meta[tail_row_index];
    const bool bv = bm.y > 0.0f && isfinite(bm.x) && isfinite(bm.y);
    const bool tv = tm.y > 0.0f && isfinite(tm.x) && isfinite(tm.y);
    const float m = bv && tv ? fmaxf(bm.x, tm.x) : (bv ? bm.x : (tv ? tm.x : -INFINITY));
    const float wb = bv ? bm.y*expf(bm.x - m) : 0.0f;
    const float wt = tv ? tm.y*expf(tm.x - m) : 0.0f;
    const float denom = wb + wt;
    const char * brow = reinterpret_cast<const char *>(body) +
        size_t(ih)*body_nb1 +
        size_t(body_packed ? iq_packed : iq)*body_nb2 +
        size_t(body_packed ? ia : is)*body_nb3;
    const char * trow = reinterpret_cast<const char *>(tail) +
        size_t(ih)*tail_nb1 + size_t(iq_packed)*tail_nb2 + size_t(ia)*tail_nb3;
    char * drow = reinterpret_cast<char *>(dst) +
        size_t(ih)*dst_nb1 + size_t(iq)*dst_nb2 + size_t(is)*dst_nb3;
    for (int id = threadIdx.x; id < d; id += blockDim.x) {
        float numerator = 0.0f;
        if (bv) {
            numerator += *reinterpret_cast<const float *>(brow + size_t(id)*sizeof(float))*wb;
        }
        if (tv) {
            numerator += *reinterpret_cast<const float *>(trow + size_t(id)*sizeof(float))*wt;
        }
        *reinterpret_cast<float *>(drow + size_t(id)*sizeof(float)) = denom > 0.0f ? numerator/denom : 0.0f;
    }
}

static void ggml_cuda_tail_make_contiguous(ggml_tensor & t,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t element_size) {
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = ne3;
    t.nb[0] = element_size;
    t.nb[1] = size_t(ne0)*t.nb[0];
    t.nb[2] = size_t(ne1)*t.nb[1];
    t.nb[3] = size_t(ne2)*t.nb[2];
    t.view_src = nullptr;
    t.view_offs = 0;
}

static void ggml_cuda_tail_make_contiguous_type(ggml_tensor & t,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = ne3;
    t.nb[0] = ggml_type_size(t.type);
    t.nb[1] = ggml_row_size(t.type, ne0);
    t.nb[2] = size_t(ne1)*t.nb[1];
    t.nb[3] = size_t(ne2)*t.nb[2];
    t.view_src = nullptr;
    t.view_offs = 0;
}

static __global__ void k_flash_attn_ext_tail_scatter(
        const float * packed, float * dst, const int32_t * query_order,
        int d, int n_query, int n_head, int n_stream, int q_max, int n_active,
        size_t packed_nb1, size_t packed_nb2, size_t packed_nb3,
        size_t dst_nb1, size_t dst_nb2, size_t dst_nb3) {
    const int iq_packed = blockIdx.x;
    const int ih = blockIdx.y;
    const int ia = blockIdx.z;
    if (iq_packed >= q_max || ih >= n_head || ia >= n_active) {
        return;
    }
    const int iq_global = query_order[ia*q_max + iq_packed];
    if (iq_global < 0 || iq_global >= n_query*n_stream) {
        return;
    }
    const int is = iq_global/n_query;
    const int iq = iq_global - is*n_query;
    const char * src_row = reinterpret_cast<const char *>(packed) +
        size_t(ih)*packed_nb1 + size_t(iq_packed)*packed_nb2 + size_t(ia)*packed_nb3;
    char * dst_row = reinterpret_cast<char *>(dst) +
        size_t(ih)*dst_nb1 + size_t(iq)*dst_nb2 + size_t(is)*dst_nb3;
    for (int id = threadIdx.x; id < d; id += blockDim.x) {
        *reinterpret_cast<float *>(dst_row + size_t(id)*sizeof(float)) =
            *reinterpret_cast<const float *>(src_row + size_t(id)*sizeof(float));
    }
}

static int64_t ggml_cuda_tail_compute_stride(int64_t logical_stride) {
    // The direct indexed kernel accepts compact tails as-is. Larger tails use
    // the upstream FA kernels, whose D=512 path requires a 256-row-aligned KV
    // extent. Keep that padding graph-local: it must never leak into the
    // persistent N+R history allocation or the serialized state.
    return logical_stride <= 256 ? logical_stride : GGML_PAD(logical_stride, FATTN_KQ_STRIDE);
}

static size_t ggml_cuda_tail_pass_alloc_size(ggml_backend_cuda_context & ctx, ggml_tensor & pass) {
    pass.data = reinterpret_cast<void *>(uintptr_t(0x10000000));
    return ggml_cuda_flash_attn_ext_get_alloc_size(ctx.device, &pass) + 256;
}

static void ggml_cuda_flash_attn_ext_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q  = dst->src[0];
    const ggml_tensor * kb = dst->src[1];
    const ggml_tensor * vb = dst->src[2];
    const ggml_tensor * mb = dst->src[3];
    const ggml_tensor * kt = dst->src[5];
    const ggml_tensor * vt = dst->src[6];
    const ggml_tensor * mt = dst->src[7];
    const ggml_tensor * qo = dst->src[8];
    const ggml_tensor * rd = dst->src[9];
    const ggml_tensor * kt_current = dst->src[10];
    const ggml_tensor * vt_current = dst->src[11];
    GGML_ASSERT(q && kb && vb && mb && kt && vt && mt && qo && rd);
    GGML_ASSERT((kt_current == nullptr) == (vt_current == nullptr));
    if (kt_current) {
        GGML_ASSERT(kt_current->type == kt->type && vt_current->type == vt->type);
        GGML_ASSERT(kt_current->ne[0] == kt->ne[0] && kt_current->ne[2] == kt->ne[2] && kt_current->ne[3] == 1);
        GGML_ASSERT(vt_current->ne[0] == vt->ne[0] && vt_current->ne[2] == vt->ne[2] && vt_current->ne[3] == 1);
        GGML_ASSERT(kt_current->ne[1] == vt_current->ne[1]);
    }
    GGML_ASSERT(qo->type == GGML_TYPE_I32 && rd->type == GGML_TYPE_I32 && rd->ne[0] >= 4);
    GGML_ASSERT(qo->ne[1] == rd->ne[1]);
    GGML_ASSERT(kt->ne[0] == q->ne[0] && vt->ne[0] == dst->ne[0]);
    if (kt->ne[2] != kb->ne[2] || vt->ne[2] != vb->ne[2]) {
        GGML_LOG_ERROR(
                "precision-tail head shard mismatch: K body=%lldx%lld/tail=%lld, V body=%lldx%lld/tail=%lld; "
                "K ops=%s/%s/%s, V ops=%s/%s/%s\n",
                (long long) kb->ne[1], (long long) kb->ne[2], (long long) kt->ne[2],
                (long long) vb->ne[1], (long long) vb->ne[2], (long long) vt->ne[2],
                ggml_op_name(kb->op),
                kb->src[0] ? ggml_op_name(kb->src[0]->op) : "<null>",
                kb->src[0] && kb->src[0]->src[0] ? ggml_op_name(kb->src[0]->src[0]->op) : "<null>",
                ggml_op_name(vb->op),
                vb->src[0] ? ggml_op_name(vb->src[0]->op) : "<null>",
                vb->src[0] && vb->src[0]->src[0] ? ggml_op_name(vb->src[0]->src[0]->op) : "<null>");
    }
    GGML_ASSERT(kt->ne[2] == dst->src[1]->ne[2] && vt->ne[2] == dst->src[2]->ne[2]);
    GGML_ASSERT(mt->ne[0] > 0 &&
            kt->ne[1] + (kt_current ? kt_current->ne[1] : 0) >= mt->ne[0]);

    const int d_k = int(q->ne[0]);
    const int d_v = int(dst->ne[0]);
    const int n_query = int(q->ne[1]);
    const int n_head = int(q->ne[2]);
    const int n_stream = int(q->ne[3]);
    const int tail_stride = int(mt->ne[0]);
    const int q_max = int(qo->ne[0]);
    const int n_active = int(qo->ne[1]);
    const int n_head_k = int(kt->ne[2]);
    const int n_head_v = int(vt->ne[2]);
    const int body_map_offset = 6 + tail_stride;
    const bool body_packed = rd->ne[0] > body_map_offset;
    const int desc_stride = int(rd->ne[0]);
    const int body_stride = body_packed ? desc_stride - body_map_offset : 0;
    const int configured_history_slots = ggml_get_op_params_i32(
            dst, GGML_FLASH_ATTN_EXT_OP_PARAM_TAIL_HISTORY_SLOTS);
    const int history_slots = configured_history_slots > 0 ?
            configured_history_slots : int(kt->ne[1]);
    GGML_ASSERT(history_slots > 0 && history_slots <= kt->ne[1] && vt->ne[1] == kt->ne[1]);
    const bool tail_bodyless = ggml_get_op_params_i32(
            dst, GGML_FLASH_ATTN_EXT_OP_PARAM_TAIL_BODYLESS) != 0;
    const bool indexed_small = tail_stride <= 256;
    const int compute_stride = tail_bodyless ?
            int(GGML_PAD(tail_stride, FATTN_KQ_STRIDE)) :
            int(ggml_cuda_tail_compute_stride(tail_stride));
    const size_t n_tail_rows = size_t(q_max)*n_head*n_active;
    const size_t n_body_rows = body_packed ? n_tail_rows : size_t(n_query)*n_head*n_stream;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float2> body_meta_alloc(pool);
    ggml_cuda_pool_alloc<float2> tail_meta_alloc(pool);
    if (!tail_bodyless) {
        body_meta_alloc.alloc(n_body_rows);
        tail_meta_alloc.alloc(indexed_small ? 1 : n_tail_rows);
        CUDA_CHECK(cudaMemsetAsync(body_meta_alloc.get(), 0, n_body_rows*sizeof(float2), ctx.stream()));
    }

    const size_t kt_elements = size_t(d_k)*compute_stride*n_head_k*n_active;
    const size_t vt_elements = size_t(d_v)*compute_stride*n_head_v*n_active;
    const size_t q_elements = size_t(d_k)*q_max*n_head*n_active;
    const size_t mask_elements = size_t(compute_stride)*q_max*n_active;
    const size_t body_mask_elements = body_packed && !tail_bodyless ?
            size_t(body_stride)*q_max*n_active : 1;
    const size_t kb_row_bytes = ggml_row_size(kb->type, d_k);
    const size_t vb_row_bytes = ggml_row_size(vb->type, d_v);
    const size_t kb_packed_bytes = body_packed && !tail_bodyless ?
            kb_row_bytes*body_stride*kb->ne[2]*n_active : 1;
    const size_t vb_packed_bytes = body_packed && !tail_bodyless ?
            vb_row_bytes*body_stride*vb->ne[2]*n_active : 1;
    ggml_cuda_pool_alloc<uint8_t> kt_alloc(pool,
            indexed_small ? 1 : kt_elements*ggml_type_size(kt->type));
    ggml_cuda_pool_alloc<uint8_t> vt_alloc(pool,
            indexed_small ? 1 : vt_elements*ggml_type_size(vt->type));
    ggml_cuda_pool_alloc<float> q_alloc(pool,
            indexed_small && !body_packed ? 1 : q_elements);
    ggml_cuda_pool_alloc<half> mask_alloc(pool,
            indexed_small && !body_packed ? 1 : mask_elements);
    ggml_cuda_pool_alloc<uint8_t> kb_alloc(pool);
    ggml_cuda_pool_alloc<uint8_t> vb_alloc(pool);
    ggml_cuda_pool_alloc<half> body_mask_alloc(pool);
    if (body_packed && !tail_bodyless) {
        kb_alloc.alloc(kb_packed_bytes);
        vb_alloc.alloc(vb_packed_bytes);
        body_mask_alloc.alloc(body_mask_elements);
    }

    const int threads = 256;
    auto blocks_for = [threads](size_t n) { return int(std::min<size_t>((n + threads - 1)/threads, 65535)); };
    const bool kt_vec = !indexed_small && kt->nb[0] == ggml_type_size(kt->type) &&
            size_t(d_k)*kt->nb[0] % sizeof(uint4) == 0 &&
            kt->nb[1] % alignof(uint4) == 0 && kt->nb[2] % alignof(uint4) == 0 &&
            uintptr_t(kt->data) % alignof(uint4) == 0 && uintptr_t(kt_alloc.get()) % alignof(uint4) == 0 &&
            (!kt_current || (kt_current->nb[0] == kt->nb[0] &&
                kt_current->nb[1] % alignof(uint4) == 0 && kt_current->nb[2] % alignof(uint4) == 0 &&
                uintptr_t(kt_current->data) % alignof(uint4) == 0));
    const bool vt_vec = !indexed_small && vt->nb[0] == ggml_type_size(vt->type) &&
            size_t(d_v)*vt->nb[0] % sizeof(uint4) == 0 &&
            vt->nb[1] % alignof(uint4) == 0 && vt->nb[2] % alignof(uint4) == 0 &&
            uintptr_t(vt->data) % alignof(uint4) == 0 && uintptr_t(vt_alloc.get()) % alignof(uint4) == 0 &&
            (!vt_current || (vt_current->nb[0] == vt->nb[0] &&
                vt_current->nb[1] % alignof(uint4) == 0 && vt_current->nb[2] % alignof(uint4) == 0 &&
                uintptr_t(vt_current->data) % alignof(uint4) == 0));
    if (kt_vec) {
        const size_t n = kt_elements*ggml_type_size(kt->type)/sizeof(uint4);
        k_flash_attn_ext_tail_pack_arenas_vec<uint4><<<blocks_for(n), threads, 0, ctx.stream()>>>(
            (const char *) kt->data, kt_current ? (const char *) kt_current->data : nullptr,
            (uint4 *) kt_alloc.get(), (const int32_t *) rd->data,
            int(size_t(d_k)*kt->nb[0]/sizeof(uint4)), compute_stride, n_head_k, n_active,
            desc_stride, history_slots, kt->nb[1], kt->nb[2],
            kt_current ? kt_current->nb[1] : 0, kt_current ? kt_current->nb[2] : 0);
    } else if (!indexed_small && ggml_type_size(kt->type) == sizeof(uint16_t)) {
        k_flash_attn_ext_tail_pack_arenas<uint16_t><<<blocks_for(kt_elements), threads, 0, ctx.stream()>>>(
            (const char *) kt->data, kt_current ? (const char *) kt_current->data : nullptr,
            (uint16_t *) kt_alloc.get(), (const int32_t *) rd->data,
            d_k, compute_stride, n_head_k, n_active, desc_stride, history_slots,
            kt->nb[0], kt->nb[1], kt->nb[2],
            kt_current ? kt_current->nb[0] : 0,
            kt_current ? kt_current->nb[1] : 0,
            kt_current ? kt_current->nb[2] : 0);
    } else if (!indexed_small) {
        k_flash_attn_ext_tail_pack_arenas<float><<<blocks_for(kt_elements), threads, 0, ctx.stream()>>>(
            (const char *) kt->data, kt_current ? (const char *) kt_current->data : nullptr,
            (float *) kt_alloc.get(), (const int32_t *) rd->data,
            d_k, compute_stride, n_head_k, n_active, desc_stride, history_slots,
            kt->nb[0], kt->nb[1], kt->nb[2],
            kt_current ? kt_current->nb[0] : 0,
            kt_current ? kt_current->nb[1] : 0,
            kt_current ? kt_current->nb[2] : 0);
    }
    if (vt_vec) {
        const size_t n = vt_elements*ggml_type_size(vt->type)/sizeof(uint4);
        k_flash_attn_ext_tail_pack_arenas_vec<uint4><<<blocks_for(n), threads, 0, ctx.stream()>>>(
            (const char *) vt->data, vt_current ? (const char *) vt_current->data : nullptr,
            (uint4 *) vt_alloc.get(), (const int32_t *) rd->data,
            int(size_t(d_v)*vt->nb[0]/sizeof(uint4)), compute_stride, n_head_v, n_active,
            desc_stride, history_slots, vt->nb[1], vt->nb[2],
            vt_current ? vt_current->nb[1] : 0, vt_current ? vt_current->nb[2] : 0);
    } else if (!indexed_small && ggml_type_size(vt->type) == sizeof(uint16_t)) {
        k_flash_attn_ext_tail_pack_arenas<uint16_t><<<blocks_for(vt_elements), threads, 0, ctx.stream()>>>(
            (const char *) vt->data, vt_current ? (const char *) vt_current->data : nullptr,
            (uint16_t *) vt_alloc.get(), (const int32_t *) rd->data,
            d_v, compute_stride, n_head_v, n_active, desc_stride, history_slots,
            vt->nb[0], vt->nb[1], vt->nb[2],
            vt_current ? vt_current->nb[0] : 0,
            vt_current ? vt_current->nb[1] : 0,
            vt_current ? vt_current->nb[2] : 0);
    } else if (!indexed_small) {
        k_flash_attn_ext_tail_pack_arenas<float><<<blocks_for(vt_elements), threads, 0, ctx.stream()>>>(
            (const char *) vt->data, vt_current ? (const char *) vt_current->data : nullptr,
            (float *) vt_alloc.get(), (const int32_t *) rd->data,
            d_v, compute_stride, n_head_v, n_active, desc_stride, history_slots,
            vt->nb[0], vt->nb[1], vt->nb[2],
            vt_current ? vt_current->nb[0] : 0,
            vt_current ? vt_current->nb[1] : 0,
            vt_current ? vt_current->nb[2] : 0);
    }
    if (!indexed_small || body_packed) {
        k_flash_attn_ext_tail_pack_q_mask<<<blocks_for(std::max(q_elements, mask_elements)), threads, 0, ctx.stream()>>>(
            (const float *) q->data, (const half *) mt->data, (const int32_t *) qo->data,
            q_alloc.get(), mask_alloc.get(), d_k, n_query, n_head, n_stream,
            tail_stride, compute_stride, q_max, n_active,
            q->nb[1], q->nb[2], q->nb[3], mt->nb[1], mt->nb[3]);
    }
    if (body_packed && !tail_bodyless) {
        k_flash_attn_ext_tail_pack_body_rows<<<blocks_for(kb_packed_bytes), threads, 0, ctx.stream()>>>(
            (const char *) kb->data, kb_alloc.get(), (const int32_t *) rd->data,
            int(kb_row_bytes), int(kb->ne[1]), body_stride, int(kb->ne[2]), n_active,
            desc_stride, body_map_offset, kb->nb[1], kb->nb[2], kb->nb[3]);
        k_flash_attn_ext_tail_pack_body_rows<<<blocks_for(vb_packed_bytes), threads, 0, ctx.stream()>>>(
            (const char *) vb->data, vb_alloc.get(), (const int32_t *) rd->data,
            int(vb_row_bytes), int(vb->ne[1]), body_stride, int(vb->ne[2]), n_active,
            desc_stride, body_map_offset, vb->nb[1], vb->nb[2], vb->nb[3]);
        k_flash_attn_ext_tail_pack_body_mask<<<blocks_for(body_mask_elements), threads, 0, ctx.stream()>>>(
            (const half *) mb->data, (const int32_t *) qo->data, (const int32_t *) rd->data,
            body_mask_alloc.get(), int(kb->ne[1]), n_query, n_stream,
            body_stride, q_max, n_active, desc_stride, body_map_offset, mb->nb[1], mb->nb[3]);
    }
    CUDA_CHECK(cudaGetLastError());

    ggml_tensor q_packed = *q;
    q_packed.data = q_alloc.get();
    ggml_cuda_tail_make_contiguous(q_packed, d_k, q_max, n_head, n_active, sizeof(float));
    ggml_tensor k_packed = *kt;
    k_packed.data = kt_alloc.get();
    ggml_cuda_tail_make_contiguous(k_packed, d_k, compute_stride, n_head_k, n_active, ggml_type_size(kt->type));
    ggml_tensor v_packed = *vt;
    v_packed.data = vt_alloc.get();
    ggml_cuda_tail_make_contiguous(v_packed, d_v, compute_stride, n_head_v, n_active, ggml_type_size(vt->type));
    ggml_tensor mask_packed = *mt;
    mask_packed.data = mask_alloc.get();
    ggml_cuda_tail_make_contiguous(mask_packed, compute_stride, q_max, 1, n_active, sizeof(half));

    ggml_tensor body_meta = *qo;
    body_meta.type = GGML_TYPE_F32;
    body_meta.data = body_meta_alloc.get();
    ggml_cuda_tail_make_contiguous(body_meta, 2, n_head,
            body_packed ? q_max : n_query, body_packed ? n_active : n_stream, sizeof(float));
    ggml_tensor body_pass = *dst;
    ggml_tensor kb_packed = *kb;
    ggml_tensor vb_packed = *vb;
    ggml_tensor body_mask_packed = *mb;
    if (body_packed && !tail_bodyless) {
        kb_packed.data = kb_alloc.get();
        vb_packed.data = vb_alloc.get();
        body_mask_packed.data = body_mask_alloc.get();
        ggml_cuda_tail_make_contiguous_type(
                kb_packed, d_k, body_stride, kb->ne[2], n_active);
        ggml_cuda_tail_make_contiguous_type(
                vb_packed, d_v, body_stride, vb->ne[2], n_active);
        ggml_cuda_tail_make_contiguous(
                body_mask_packed, body_stride, q_max, 1, n_active, sizeof(half));
        body_pass.src[0] = &q_packed;
        body_pass.src[1] = &kb_packed;
        body_pass.src[2] = &vb_packed;
        body_pass.src[3] = &body_mask_packed;
        ggml_cuda_tail_make_contiguous(
                body_pass, d_v, n_head, q_max, n_active, sizeof(float));
    }
    for (int i = 5; i < GGML_MAX_SRC; ++i) {
        body_pass.src[i] = nullptr;
    }
    body_pass.src[8] = &body_meta;
    body_pass.view_src = nullptr;
    body_pass.view_offs = 0;
    ggml_cuda_pool_alloc<uint8_t> body_alloc(pool);
    if (!tail_bodyless) {
        const size_t body_alloc_size = ggml_cuda_tail_pass_alloc_size(ctx, body_pass);
        body_alloc.alloc(body_alloc_size);
        body_pass.data = body_alloc.get();
    }
    const uint64_t tail_pack_bytes = kt_alloc.actual_size + vt_alloc.actual_size +
            q_alloc.actual_size + mask_alloc.actual_size + kb_alloc.actual_size +
            vb_alloc.actual_size + body_mask_alloc.actual_size;
    const uint64_t tail_plan_input_bytes = ggml_nbytes(mt) + ggml_nbytes(qo) + ggml_nbytes(rd);
    // mt/qo/rd are scheduler-owned graph tensors and are already included in
    // the reported CUDA compute buffer. Keep their footprint visible as a
    // diagnostic, but do not double-count it as CUDA-pool high water.
    const uint64_t tail_base_bytes = body_meta_alloc.actual_size + tail_meta_alloc.actual_size +
            tail_pack_bytes + body_alloc.actual_size;
    if (!tail_bodyless) {
        if (ggml_cuda_flash_attn_ext_kvarn_uses_views(&body_pass)) {
            if (!ggml_cuda_flash_attn_ext_kvarn(
                    ctx, &body_pass, GGML_CUDA_FATTN_KVARN_ENTRY_COMPACT_TAIL)) {
                GGML_ABORT("unsupported structured body in exact-tail attention");
            }
        } else {
            ggml_cuda_flash_attn_ext_dispatch(ctx, &body_pass);
        }
    }

    if (indexed_small) {
        float scale = 1.0f;
        float max_bias = 0.0f;
        float logit_softcap = 0.0f;
        memcpy(&scale, dst->op_params + 0*sizeof(float), sizeof(float));
        memcpy(&max_bias, dst->op_params + 1*sizeof(float), sizeof(float));
        memcpy(&logit_softcap, dst->op_params + 2*sizeof(float), sizeof(float));
        const dim3 grid(q_max, n_head, n_active);
#define GGML_CUDA_LAUNCH_INDEXED(TK, TV, MAX_TAIL) \
        k_flash_attn_ext_tail_indexed_small<TK, TV, MAX_TAIL><<<grid, 256, 0, ctx.stream()>>>( \
            (const float *) q->data, (const char *) kt->data, (const char *) vt->data, \
            kt_current ? (const char *) kt_current->data : nullptr, \
            vt_current ? (const char *) vt_current->data : nullptr, (const half *) mt->data, \
            (const int32_t *) qo->data, (const int32_t *) rd->data, \
            (const float *) body_pass.data, body_meta_alloc.get(), (float *) dst->data, \
            d_k, d_v, n_query, n_head, n_stream, q_max, n_active, n_head_k, n_head_v, desc_stride, \
            scale, max_bias, logit_softcap, q->nb[1], q->nb[2], q->nb[3], \
            history_slots, kt->nb[0], kt->nb[1], kt->nb[2], vt->nb[0], vt->nb[1], vt->nb[2], \
            kt_current ? kt_current->nb[0] : 0, kt_current ? kt_current->nb[1] : 0, kt_current ? kt_current->nb[2] : 0, \
            vt_current ? vt_current->nb[0] : 0, vt_current ? vt_current->nb[1] : 0, vt_current ? vt_current->nb[2] : 0, \
            mt->nb[1], mt->nb[3], \
            tail_bodyless, body_packed, body_pass.nb[1], body_pass.nb[2], body_pass.nb[3], \
            dst->nb[1], dst->nb[2], dst->nb[3])
        if (kt->type == GGML_TYPE_F16 && vt->type == GGML_TYPE_F16) {
            GGML_CUDA_LAUNCH_INDEXED(half, half, 256);
        } else if (kt->type == GGML_TYPE_F16 && vt->type == GGML_TYPE_BF16) {
            GGML_CUDA_LAUNCH_INDEXED(half, nv_bfloat16, 256);
        } else if (kt->type == GGML_TYPE_BF16 && vt->type == GGML_TYPE_F16) {
            GGML_CUDA_LAUNCH_INDEXED(nv_bfloat16, half, 256);
        } else {
            GGML_ASSERT(kt->type == GGML_TYPE_BF16 && vt->type == GGML_TYPE_BF16);
            GGML_CUDA_LAUNCH_INDEXED(nv_bfloat16, nv_bfloat16, 256);
        }
#undef GGML_CUDA_LAUNCH_INDEXED
        CUDA_CHECK(cudaGetLastError());
        ggml_cuda_kv_memory_transient_stats_record_tail(
                body_meta_alloc.actual_size,
                tail_meta_alloc.actual_size,
                tail_pack_bytes,
                body_alloc.actual_size,
                0,
                tail_plan_input_bytes,
                tail_base_bytes);
        return;
    }

    ggml_tensor tail_meta = body_meta;
    if (!tail_bodyless) {
        tail_meta.data = tail_meta_alloc.get();
        ggml_cuda_tail_make_contiguous(tail_meta, 2, n_head, q_max, n_active, sizeof(float));
    }
    ggml_tensor tail_pass = *dst;
    tail_pass.src[0] = &q_packed;
    tail_pass.src[1] = &k_packed;
    tail_pass.src[2] = &v_packed;
    tail_pass.src[3] = &mask_packed;
    // Sinks are part of the body partial for a split source. For a bodyless
    // source the exact pass is the sole softmax and must consume them itself.
    tail_pass.src[4] = tail_bodyless ? dst->src[4] : nullptr;
    for (int i = 5; i < GGML_MAX_SRC; ++i) {
        tail_pass.src[i] = nullptr;
    }
    tail_pass.src[8] = tail_bodyless ? nullptr : &tail_meta;
    ggml_cuda_tail_make_contiguous(tail_pass, d_v, n_head, q_max, n_active, sizeof(float));
    if (q_max == 1 && n_active == 1 &&
            kt->type == GGML_TYPE_BF16 && vt->type == GGML_TYPE_BF16) {
        ggml_set_op_params_i32(&tail_pass, GGML_CUDA_FATTN_OP_PARAM_FORCE_VEC, 1);
    }
    const size_t tail_alloc_size = ggml_cuda_tail_pass_alloc_size(ctx, tail_pass);
    ggml_cuda_pool_alloc<uint8_t> tail_alloc(pool, tail_alloc_size);
    tail_pass.data = tail_alloc.get();
    ggml_cuda_kv_memory_transient_stats_record_tail(
            body_meta_alloc.actual_size,
            tail_meta_alloc.actual_size,
            tail_pack_bytes,
            body_alloc.actual_size,
            tail_alloc.actual_size,
            tail_plan_input_bytes,
            tail_base_bytes + tail_alloc.actual_size);
    ggml_cuda_flash_attn_ext_dispatch(ctx, &tail_pass);

    const dim3 grid(q_max, n_head, n_active);
    if (tail_bodyless) {
        k_flash_attn_ext_tail_scatter<<<grid, 256, 0, ctx.stream()>>>(
            (const float *) tail_pass.data, (float *) dst->data, (const int32_t *) qo->data,
            d_v, n_query, n_head, n_stream, q_max, n_active,
            tail_pass.nb[1], tail_pass.nb[2], tail_pass.nb[3],
            dst->nb[1], dst->nb[2], dst->nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    k_flash_attn_ext_tail_partials_merge<<<grid, 256, 0, ctx.stream()>>>(
        (const float *) body_pass.data, (const float *) tail_pass.data, (float *) dst->data,
        body_meta_alloc.get(), tail_meta_alloc.get(), (const int32_t *) qo->data,
        d_v, n_query, n_head, n_stream, q_max, n_active, body_packed,
        body_pass.nb[1], body_pass.nb[2], body_pass.nb[3],
        tail_pass.nb[1], tail_pass.nb[2], tail_pass.nb[3],
        dst->nb[1], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}
