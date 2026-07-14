#pragma once

#include "fattn-common.cuh"

template<typename T>
static __global__ void k_flash_attn_ext_tail_pack_arenas(
        const char * src, T * dst, const int32_t * run_desc,
        int d, int tail_stride, int n_head, int n_active,
        int desc_stride,
        size_t nb0, size_t nb1, size_t nb2) {
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
            value = *reinterpret_cast<const T *>(src + size_t(id)*nb0 +
                size_t(slot)*nb1 + size_t(ih)*nb2);
        }
        dst[i] = value;
    }
}

template<typename V>
static __global__ void k_flash_attn_ext_tail_pack_arenas_vec(
        const char * src, V * dst, const int32_t * run_desc,
        int n_vec, int tail_stride, int n_head, int n_active,
        int desc_stride, size_t nb1, size_t nb2) {
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
            value = *reinterpret_cast<const V *>(src + size_t(iv)*sizeof(V) +
                    size_t(slot)*nb1 + size_t(ih)*nb2);
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
        int tail_stride, int q_max, int n_active,
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
        if (iq_global >= 0 && iq_global < n_query*n_stream) {
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
        const float * q, const char * kt, const char * vt, const half * mt,
        const int32_t * query_order, const int32_t * run_desc,
        const float * body, const float2 * body_meta, float * dst,
        int d_k, int d_v, int n_query, int n_head, int n_stream,
        int q_max, int n_active, int n_head_k, int n_head_v, int desc_stride,
        float scale, float max_bias, float logit_softcap,
        size_t q_nb1, size_t q_nb2, size_t q_nb3,
        size_t kt_nb0, size_t kt_nb1, size_t kt_nb2,
        size_t vt_nb0, size_t vt_nb1, size_t vt_nb2,
        size_t mt_nb1, size_t mt_nb3,
        bool body_packed,
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
        float dot = 0.0f;
        for (int d = lane; d < d_k; d += 32) {
            const TK kval = *reinterpret_cast<const TK *>(
                    kt + size_t(d)*kt_nb0 + size_t(slot)*kt_nb1 + size_t(ih_k)*kt_nb2);
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

    reduction[tid] = tid < n_tail ? scores[tid] : -INFINITY;
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
    if (tid < n_tail && isfinite(scores[tid]) && isfinite(tail_max_shared)) {
        weight = expf(scores[tid] - tail_max_shared);
        scores[tid] = weight;
    } else if (tid < n_tail) {
        scores[tid] = 0.0f;
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
    const float2 bm = body_meta[body_row_index];
    const bool bv = bm.y > 0.0f && isfinite(bm.x) && isfinite(bm.y);
    const bool tv = tail_sum_shared > 0.0f && isfinite(tail_max_shared);
    const float global_max = bv && tv ? fmaxf(bm.x, tail_max_shared) :
            (bv ? bm.x : (tv ? tail_max_shared : -INFINITY));
    const float wb = bv ? bm.y*expf(bm.x - global_max) : 0.0f;
    const float wt = tv ? expf(tail_max_shared - global_max) : 0.0f;
    const float denom = wb + tail_sum_shared*wt;
    const char * brow = reinterpret_cast<const char *>(body) + size_t(ih)*body_nb1 +
            size_t(body_packed ? iq_packed : iq)*body_nb2 + size_t(body_packed ? ia : is)*body_nb3;
    char * drow = reinterpret_cast<char *>(dst) + size_t(ih)*dst_nb1 +
            size_t(iq)*dst_nb2 + size_t(is)*dst_nb3;
    for (int d = tid; d < d_v; d += blockDim.x) {
        float tail_acc = 0.0f;
        for (int token = 0; token < n_tail; ++token) {
            const int slot = desc[6 + token];
            const TV vval = *reinterpret_cast<const TV *>(
                    vt + size_t(d)*vt_nb0 + size_t(slot)*vt_nb1 + size_t(ih_v)*vt_nb2);
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
    GGML_ASSERT(q && kb && vb && mb && kt && vt && mt && qo && rd);
    GGML_ASSERT(qo->type == GGML_TYPE_I32 && rd->type == GGML_TYPE_I32 && rd->ne[0] >= 4);
    GGML_ASSERT(qo->ne[1] == rd->ne[1]);
    GGML_ASSERT(kt->ne[0] == q->ne[0] && vt->ne[0] == dst->ne[0]);
    GGML_ASSERT(kt->ne[2] == dst->src[1]->ne[2] && vt->ne[2] == dst->src[2]->ne[2]);
    GGML_ASSERT(mt->ne[0] > 0 && kt->ne[1] >= mt->ne[0]*rd->ne[1]);

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
    const bool indexed_small = tail_stride <= 128;
    const size_t n_tail_rows = size_t(q_max)*n_head*n_active;
    const size_t n_body_rows = body_packed ? n_tail_rows : size_t(n_query)*n_head*n_stream;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float2> body_meta_alloc(pool, n_body_rows);
    ggml_cuda_pool_alloc<float2> tail_meta_alloc(pool, indexed_small ? 1 : n_tail_rows);

    const size_t kt_elements = size_t(d_k)*tail_stride*n_head_k*n_active;
    const size_t vt_elements = size_t(d_v)*tail_stride*n_head_v*n_active;
    const size_t q_elements = size_t(d_k)*q_max*n_head*n_active;
    const size_t mask_elements = size_t(tail_stride)*q_max*n_active;
    const size_t body_mask_elements = body_packed ? size_t(body_stride)*q_max*n_active : 1;
    const size_t kb_row_bytes = ggml_row_size(kb->type, d_k);
    const size_t vb_row_bytes = ggml_row_size(vb->type, d_v);
    const size_t kb_packed_bytes = body_packed ? kb_row_bytes*body_stride*kb->ne[2]*n_active : 1;
    const size_t vb_packed_bytes = body_packed ? vb_row_bytes*body_stride*vb->ne[2]*n_active : 1;
    ggml_cuda_pool_alloc<uint8_t> kt_alloc(pool, indexed_small ? 1 : kt_elements*ggml_type_size(kt->type));
    ggml_cuda_pool_alloc<uint8_t> vt_alloc(pool, indexed_small ? 1 : vt_elements*ggml_type_size(vt->type));
    ggml_cuda_pool_alloc<float> q_alloc(pool, indexed_small && !body_packed ? 1 : q_elements);
    ggml_cuda_pool_alloc<half> mask_alloc(pool, indexed_small && !body_packed ? 1 : mask_elements);
    ggml_cuda_pool_alloc<uint8_t> kb_alloc(pool, kb_packed_bytes);
    ggml_cuda_pool_alloc<uint8_t> vb_alloc(pool, vb_packed_bytes);
    ggml_cuda_pool_alloc<half> body_mask_alloc(pool, body_mask_elements);

    const int threads = 256;
    auto blocks_for = [threads](size_t n) { return int(std::min<size_t>((n + threads - 1)/threads, 65535)); };
    const bool kt_vec = !indexed_small && kt->nb[0] == ggml_type_size(kt->type) &&
            size_t(d_k)*kt->nb[0] % sizeof(uint4) == 0 &&
            kt->nb[1] % alignof(uint4) == 0 && kt->nb[2] % alignof(uint4) == 0 &&
            uintptr_t(kt->data) % alignof(uint4) == 0 && uintptr_t(kt_alloc.get()) % alignof(uint4) == 0;
    const bool vt_vec = !indexed_small && vt->nb[0] == ggml_type_size(vt->type) &&
            size_t(d_v)*vt->nb[0] % sizeof(uint4) == 0 &&
            vt->nb[1] % alignof(uint4) == 0 && vt->nb[2] % alignof(uint4) == 0 &&
            uintptr_t(vt->data) % alignof(uint4) == 0 && uintptr_t(vt_alloc.get()) % alignof(uint4) == 0;
    if (kt_vec) {
        const size_t n = kt_elements*ggml_type_size(kt->type)/sizeof(uint4);
        k_flash_attn_ext_tail_pack_arenas_vec<uint4><<<blocks_for(n), threads, 0, ctx.stream()>>>(
            (const char *) kt->data, (uint4 *) kt_alloc.get(), (const int32_t *) rd->data,
            int(size_t(d_k)*kt->nb[0]/sizeof(uint4)), tail_stride, n_head_k, n_active,
            desc_stride, kt->nb[1], kt->nb[2]);
    } else if (!indexed_small && ggml_type_size(kt->type) == sizeof(uint16_t)) {
        k_flash_attn_ext_tail_pack_arenas<uint16_t><<<blocks_for(kt_elements), threads, 0, ctx.stream()>>>(
            (const char *) kt->data, (uint16_t *) kt_alloc.get(), (const int32_t *) rd->data,
            d_k, tail_stride, n_head_k, n_active, desc_stride, kt->nb[0], kt->nb[1], kt->nb[2]);
    } else if (!indexed_small) {
        k_flash_attn_ext_tail_pack_arenas<float><<<blocks_for(kt_elements), threads, 0, ctx.stream()>>>(
            (const char *) kt->data, (float *) kt_alloc.get(), (const int32_t *) rd->data,
            d_k, tail_stride, n_head_k, n_active, desc_stride, kt->nb[0], kt->nb[1], kt->nb[2]);
    }
    if (vt_vec) {
        const size_t n = vt_elements*ggml_type_size(vt->type)/sizeof(uint4);
        k_flash_attn_ext_tail_pack_arenas_vec<uint4><<<blocks_for(n), threads, 0, ctx.stream()>>>(
            (const char *) vt->data, (uint4 *) vt_alloc.get(), (const int32_t *) rd->data,
            int(size_t(d_v)*vt->nb[0]/sizeof(uint4)), tail_stride, n_head_v, n_active,
            desc_stride, vt->nb[1], vt->nb[2]);
    } else if (!indexed_small && ggml_type_size(vt->type) == sizeof(uint16_t)) {
        k_flash_attn_ext_tail_pack_arenas<uint16_t><<<blocks_for(vt_elements), threads, 0, ctx.stream()>>>(
            (const char *) vt->data, (uint16_t *) vt_alloc.get(), (const int32_t *) rd->data,
            d_v, tail_stride, n_head_v, n_active, desc_stride, vt->nb[0], vt->nb[1], vt->nb[2]);
    } else if (!indexed_small) {
        k_flash_attn_ext_tail_pack_arenas<float><<<blocks_for(vt_elements), threads, 0, ctx.stream()>>>(
            (const char *) vt->data, (float *) vt_alloc.get(), (const int32_t *) rd->data,
            d_v, tail_stride, n_head_v, n_active, desc_stride, vt->nb[0], vt->nb[1], vt->nb[2]);
    }
    if (!indexed_small || body_packed) {
        k_flash_attn_ext_tail_pack_q_mask<<<blocks_for(std::max(q_elements, mask_elements)), threads, 0, ctx.stream()>>>(
            (const float *) q->data, (const half *) mt->data, (const int32_t *) qo->data,
            q_alloc.get(), mask_alloc.get(), d_k, n_query, n_head, n_stream,
            tail_stride, q_max, n_active, q->nb[1], q->nb[2], q->nb[3], mt->nb[1], mt->nb[3]);
    }
    if (body_packed) {
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
    ggml_cuda_tail_make_contiguous(k_packed, d_k, tail_stride, n_head_k, n_active, ggml_type_size(kt->type));
    ggml_tensor v_packed = *vt;
    v_packed.data = vt_alloc.get();
    ggml_cuda_tail_make_contiguous(v_packed, d_v, tail_stride, n_head_v, n_active, ggml_type_size(vt->type));
    ggml_tensor mask_packed = *mt;
    mask_packed.data = mask_alloc.get();
    ggml_cuda_tail_make_contiguous(mask_packed, tail_stride, q_max, 1, n_active, sizeof(half));

    ggml_tensor body_meta = *qo;
    body_meta.type = GGML_TYPE_F32;
    body_meta.data = body_meta_alloc.get();
    ggml_cuda_tail_make_contiguous(body_meta, 2, n_head,
            body_packed ? q_max : n_query, body_packed ? n_active : n_stream, sizeof(float));
    ggml_tensor body_pass = *dst;
    ggml_tensor kb_packed = *kb;
    ggml_tensor vb_packed = *vb;
    ggml_tensor body_mask_packed = *mb;
    if (body_packed) {
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
    const size_t body_alloc_size = ggml_cuda_tail_pass_alloc_size(ctx, body_pass);
    ggml_cuda_pool_alloc<uint8_t> body_alloc(pool, body_alloc_size);
    body_pass.data = body_alloc.get();
    ggml_cuda_flash_attn_ext_dispatch(ctx, &body_pass);

    if (indexed_small) {
        float scale = 1.0f;
        float max_bias = 0.0f;
        float logit_softcap = 0.0f;
        memcpy(&scale, dst->op_params + 0*sizeof(float), sizeof(float));
        memcpy(&max_bias, dst->op_params + 1*sizeof(float), sizeof(float));
        memcpy(&logit_softcap, dst->op_params + 2*sizeof(float), sizeof(float));
        const dim3 grid(q_max, n_head, n_active);
#define GGML_CUDA_LAUNCH_INDEXED_SMALL(TK, TV) \
        k_flash_attn_ext_tail_indexed_small<TK, TV, 128><<<grid, 256, 0, ctx.stream()>>>( \
            (const float *) q->data, (const char *) kt->data, (const char *) vt->data, (const half *) mt->data, \
            (const int32_t *) qo->data, (const int32_t *) rd->data, \
            (const float *) body_pass.data, body_meta_alloc.get(), (float *) dst->data, \
            d_k, d_v, n_query, n_head, n_stream, q_max, n_active, n_head_k, n_head_v, desc_stride, \
            scale, max_bias, logit_softcap, q->nb[1], q->nb[2], q->nb[3], \
            kt->nb[0], kt->nb[1], kt->nb[2], vt->nb[0], vt->nb[1], vt->nb[2], mt->nb[1], mt->nb[3], \
            body_packed, body_pass.nb[1], body_pass.nb[2], body_pass.nb[3], \
            dst->nb[1], dst->nb[2], dst->nb[3])
        if (kt->type == GGML_TYPE_F16 && vt->type == GGML_TYPE_F16) {
            GGML_CUDA_LAUNCH_INDEXED_SMALL(half, half);
        } else if (kt->type == GGML_TYPE_F16 && vt->type == GGML_TYPE_BF16) {
            GGML_CUDA_LAUNCH_INDEXED_SMALL(half, nv_bfloat16);
        } else if (kt->type == GGML_TYPE_BF16 && vt->type == GGML_TYPE_F16) {
            GGML_CUDA_LAUNCH_INDEXED_SMALL(nv_bfloat16, half);
        } else {
            GGML_ASSERT(kt->type == GGML_TYPE_BF16 && vt->type == GGML_TYPE_BF16);
            GGML_CUDA_LAUNCH_INDEXED_SMALL(nv_bfloat16, nv_bfloat16);
        }
#undef GGML_CUDA_LAUNCH_INDEXED_SMALL
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    ggml_tensor tail_meta = body_meta;
    tail_meta.data = tail_meta_alloc.get();
    ggml_cuda_tail_make_contiguous(tail_meta, 2, n_head, q_max, n_active, sizeof(float));
    ggml_tensor tail_pass = *dst;
    tail_pass.src[0] = &q_packed;
    tail_pass.src[1] = &k_packed;
    tail_pass.src[2] = &v_packed;
    tail_pass.src[3] = &mask_packed;
    tail_pass.src[4] = nullptr;
    for (int i = 5; i < GGML_MAX_SRC; ++i) {
        tail_pass.src[i] = nullptr;
    }
    tail_pass.src[8] = &tail_meta;
    ggml_cuda_tail_make_contiguous(tail_pass, d_v, n_head, q_max, n_active, sizeof(float));
    const size_t tail_alloc_size = ggml_cuda_tail_pass_alloc_size(ctx, tail_pass);
    ggml_cuda_pool_alloc<uint8_t> tail_alloc(pool, tail_alloc_size);
    tail_pass.data = tail_alloc.get();
    ggml_cuda_flash_attn_ext_dispatch(ctx, &tail_pass);

    const dim3 grid(q_max, n_head, n_active);
    k_flash_attn_ext_tail_partials_merge<<<grid, 256, 0, ctx.stream()>>>(
        (const float *) body_pass.data, (const float *) tail_pass.data, (float *) dst->data,
        body_meta_alloc.get(), tail_meta_alloc.get(), (const int32_t *) qo->data,
        d_v, n_query, n_head, n_stream, q_max, n_active, body_packed,
        body_pass.nb[1], body_pass.nb[2], body_pass.nb[3],
        tail_pass.nb[1], tail_pass.nb[2], tail_pass.nb[3],
        dst->nb[1], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}
