#pragma once

#include "fattn-mma-kvarn-case-decl.cuh"
#include "fattn-mma-kvarn-load.cuh"

// Portable direct-record KVarN attention. It deliberately uses only ordinary
// CUDA/HIP block primitives, so RDNA/CDNA do not depend on NVIDIA MMA. The
// kernel stays in the rotated KVarN domain; the graph applies one inverse WHT
// to the output instead of materializing and inverse-transforming every KV row.
static __device__ __forceinline__ float ggml_cuda_fattn_kvarn_load_tail(
        const char * ptr,
        bool bf16) {
    if (bf16) {
        const uint16_t bits = *reinterpret_cast<const uint16_t *>(ptr);
        return __uint_as_float((uint32_t) bits << 16);
    }
    return __half2float(*reinterpret_cast<const half *>(ptr));
}

struct ggml_cuda_fattn_kvarn_portable_ref {
    bool valid;
    bool stage;
    int stage_pos;
};

static __device__ __forceinline__ ggml_cuda_fattn_kvarn_portable_ref
ggml_cuda_fattn_kvarn_portable_resolve(
        const ggml_cuda_fattn_kvarn_desc & desc, int token) {
    ggml_cuda_fattn_kvarn_portable_ref result = {};
    int group;
    int pos;
    bool explicitly_staged = false;
    if (desc.swa || desc.read_indirect) {
        const int64_t encoded = desc.indices[token];
        if (encoded == -1) {
            return result;
        }
        const int64_t absolute = ggml_cuda_fattn_kvarn_read_cell(desc, encoded, explicitly_staged);
        group = int(absolute/GGML_CUDA_FATTN_KVARN_DIM);
        pos = int(absolute - int64_t(group)*GGML_CUDA_FATTN_KVARN_DIM);
    } else {
        group = token/GGML_CUDA_FATTN_KVARN_DIM;
        pos = token - group*GGML_CUDA_FATTN_KVARN_DIM;
    }
    result.stage = explicitly_staged ||
        (!(desc.read_indirect && !desc.swa) &&
         ggml_cuda_fattn_kvarn_group_from_stage(desc, group));
    result.valid = result.stage || (!explicitly_staged &&
        (desc.read_indirect && !desc.swa ? true :
         ggml_cuda_fattn_kvarn_group_from_record(desc, group)));
    if (result.stage) {
        const int stage_base = desc.stream*GGML_CUDA_FATTN_KVARN_DIM*desc.stage_groups;
        result.stage_pos = desc.swa ?
            (group%desc.stage_groups)*GGML_CUDA_FATTN_KVARN_DIM + pos :
            stage_base + (group == 0 ? pos : GGML_CUDA_FATTN_KVARN_DIM +
                ((group - 1)%desc.tail_groups)*GGML_CUDA_FATTN_KVARN_DIM + pos);
    }
    return result;
}

template<int D>
static __device__ __forceinline__ void ggml_cuda_fattn_kvarn_portable_stage_rotated(
        const ggml_cuda_fattn_kvarn_desc & desc,
        int stage_pos,
        int tid,
        float (&values)[D/GGML_CUDA_FATTN_KVARN_DIM]) {
    constexpr int SLICES = D/GGML_CUDA_FATTN_KVARN_DIM;
#pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        const int head = desc.head_base + slice;
        values[slice] = __half2float(desc.stage[
            ((int64_t) stage_pos*desc.n_record_heads + head)*GGML_CUDA_FATTN_KVARN_DIM + tid]);
    }
}

template<int D>
static __global__ void ggml_cuda_fattn_kvarn_portable_kernel(
        const char * q_data,
        const ggml_cuda_fattn_kvarn_desc * k_descs,
        const ggml_cuda_fattn_kvarn_desc * v_descs,
        const char * mask_data,
        const float * sinks,
        const char * k_tail_data,
        const char * v_tail_data,
        const char * tail_mask_data,
        const int32_t * query_order,
        const int32_t * run_desc,
        float2 * body_meta,
        char * dst_data,
        float scale,
        float max_bias,
        float logit_softcap,
        int64_t nbq1,
        int64_t nbq2,
        int64_t nbq3,
        int64_t nbm0,
        int64_t nbm1,
        int64_t nbm2,
        int64_t nbm3,
        int64_t nmask2,
        int64_t nmask3,
        int64_t nbd1,
        int64_t nbd2,
        int64_t nbd3,
        int64_t nbkt1,
        int64_t nbkt2,
        int64_t nbvt1,
        int64_t nbvt2,
        int64_t nbmt0,
        int64_t nbmt1,
        int64_t nbmt3,
        int query_order_ne0,
        int query_order_nelements,
        int run_desc_ne0,
        int tail_mask_ne0,
        bool k_tail_bf16,
        bool v_tail_bf16,
        int n_kv,
        int n_query,
        int n_query_heads,
        int n_kv_heads) {
    static_assert(D == 128 || D == 256 || D == 512,
        "portable KVarN attention supports 128/256/512-wide heads");
    constexpr int THREADS = GGML_CUDA_FATTN_KVARN_DIM;
    constexpr int SLICES = D / GGML_CUDA_FATTN_KVARN_DIM;

    const int query = (int) blockIdx.x;
    const int query_head = (int) blockIdx.y;
    const int stream = (int) blockIdx.z;
    const int tid = (int) threadIdx.x;
    if (query >= n_query || query_head >= n_query_heads) {
        return;
    }

    const int gqa = n_query_heads / n_kv_heads;
    const int kv_head = query_head / gqa;
    const float * q = (const float *) (
        q_data + query * nbq1 + query_head * nbq2 + stream * nbq3);

    __shared__ float reduction[THREADS];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float old_scale_shared;
    __shared__ float weight_shared;

    float accumulator[SLICES] = {};
    if (tid == 0) {
        maximum = -FLT_MAX;
        denominator = 0.0f;
    }
    __syncthreads();

    uint32_t n_head_log2 = 1;
    while ((n_head_log2 << 1) <= (uint32_t) n_query_heads) {
        n_head_log2 <<= 1;
    }
    const float m0 = exp2f(-max_bias / float(n_head_log2));
    const float m1 = exp2f(-(max_bias / 2.0f) / float(n_head_log2));
    const float slope = max_bias > 0.0f ?
        (query_head < (int) n_head_log2 ?
            powf(m0, float(query_head + 1)) :
            powf(m1, float(2 * (query_head - (int) n_head_log2) + 1))) : 1.0f;

    const int32_t * desc = nullptr;
    bool body_packed = false;
    int n_body = n_kv;
    if (k_tail_data != nullptr) {
        const int query_id = stream * n_query + query;
        int active = -1;
        for (int packed = 0; packed < query_order_nelements; ++packed) {
            if (query_order[packed] == query_id) {
                active = packed / query_order_ne0;
                break;
            }
        }
        if (active < 0) {
            return;
        }
        desc = run_desc + (size_t) active * run_desc_ne0;
        body_packed = run_desc_ne0 > 6 + tail_mask_ne0;
        n_body = body_packed ? desc[5] : n_kv;
    }

    for (int packed = 0; packed < n_body; ++packed) {
        const int flat = body_packed ?
            desc[6 + tail_mask_ne0 + packed] : stream * n_kv + packed;
        const int body_stream = flat / n_kv;
        const int token = flat - body_stream * n_kv;
        const ggml_cuda_fattn_kvarn_desc & k_desc =
            k_descs[(size_t) body_stream * n_kv_heads + kv_head];
        const ggml_cuda_fattn_kvarn_desc & v_desc =
            v_descs[(size_t) body_stream * n_kv_heads + kv_head];
        const auto k_ref = ggml_cuda_fattn_kvarn_portable_resolve(k_desc, token);
        const auto v_ref = ggml_cuda_fattn_kvarn_portable_resolve(v_desc, token);
        float k_values[SLICES] = {};
        if (k_ref.stage) {
            ggml_cuda_fattn_kvarn_portable_stage_rotated<D>(
                    k_desc, k_ref.stage_pos, tid, k_values);
        } else {
#pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                k_values[slice] = ggml_cuda_fattn_kvarn_load_rotated(
                        k_desc, token, slice, tid);
            }
        }
        float partial = 0.0f;
#pragma unroll
        for (int slice = 0; slice < SLICES; ++slice) {
            const int dim = slice * GGML_CUDA_FATTN_KVARN_DIM + tid;
            partial += q[dim] * k_values[slice];
        }
        reduction[tid] = partial;
        __syncthreads();

        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float mask_value = 0.0f;
            if (mask_data != nullptr) {
                const half * mask = (const half *) (
                    mask_data + token * nbm0 + query * nbm1 +
                    (query_head % nmask2) * nbm2 + (body_stream % nmask3) * nbm3);
                mask_value = slope * __half2float(*mask);
            }

            float score = reduction[0] * scale;
            if (logit_softcap != 0.0f) {
                score = logit_softcap * tanhf(score);
            }
            score += mask_value;
            if (mask_value == -INFINITY) {
                old_scale_shared = 1.0f;
                weight_shared = 0.0f;
            } else {
                const float next_maximum = fmaxf(maximum, score);
                const float old_scale = maximum == -FLT_MAX ?
                    0.0f : expf(maximum - next_maximum);
                const float weight = expf(score - next_maximum);
                maximum = next_maximum;
                denominator = denominator * old_scale + weight;
                old_scale_shared = old_scale;
                weight_shared = weight;
            }
        }
        __syncthreads();

        float v_values[SLICES] = {};
        if (v_ref.stage) {
            ggml_cuda_fattn_kvarn_portable_stage_rotated<D>(
                    v_desc, v_ref.stage_pos, tid, v_values);
        } else {
            for (int slice = 0; slice < SLICES; ++slice) {
                v_values[slice] = ggml_cuda_fattn_kvarn_load_rotated(
                        v_desc, token, slice, tid);
            }
        }
#pragma unroll
        for (int slice = 0; slice < SLICES; ++slice) {
            accumulator[slice] = accumulator[slice] * old_scale_shared +
                v_values[slice] * weight_shared;
        }
        __syncthreads();
    }

    if (k_tail_data != nullptr) {
        const int n_tail = desc[4];
        for (int token = 0; token < n_tail; ++token) {
            const int slot = desc[6 + token];
            float partial = 0.0f;
#pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                const int dim = slice * GGML_CUDA_FATTN_KVARN_DIM + tid;
                const char * ptr = k_tail_data +
                    (size_t) slot * nbkt1 + (size_t) kv_head * nbkt2 +
                    (size_t) dim * sizeof(uint16_t);
                const float kval = ggml_cuda_fattn_kvarn_load_tail(
                    ptr, k_tail_bf16);
                partial += q[dim] * kval;
            }
            reduction[tid] = partial;
            __syncthreads();
            for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduction[tid] += reduction[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                const half * tail_mask = (const half *) (
                    tail_mask_data + (size_t) token * nbmt0 +
                    (size_t) query * nbmt1 + (size_t) stream * nbmt3);
                const float mask_value = slope * __half2float(*tail_mask);
                float score = reduction[0] * scale;
                if (logit_softcap != 0.0f) {
                    score = logit_softcap * tanhf(score);
                }
                score += mask_value;
                if (mask_value == -INFINITY) {
                    old_scale_shared = 1.0f;
                    weight_shared = 0.0f;
                } else {
                    const float next_maximum = fmaxf(maximum, score);
                    const float old_scale = maximum == -FLT_MAX ?
                        0.0f : expf(maximum - next_maximum);
                    const float weight = expf(score - next_maximum);
                    maximum = next_maximum;
                    denominator = denominator * old_scale + weight;
                    old_scale_shared = old_scale;
                    weight_shared = weight;
                }
            }
            __syncthreads();

#pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                const int dim = slice * GGML_CUDA_FATTN_KVARN_DIM + tid;
                const char * ptr = v_tail_data +
                    (size_t) slot * nbvt1 + (size_t) kv_head * nbvt2 +
                    (size_t) dim * sizeof(uint16_t);
                const float vval = ggml_cuda_fattn_kvarn_load_tail(
                    ptr, v_tail_bf16);
                accumulator[slice] =
                    accumulator[slice] * old_scale_shared + vval * weight_shared;
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        if (sinks != nullptr) {
            const float score = sinks[query_head];
            const float next_maximum = fmaxf(maximum, score);
            const float old_scale = maximum == -FLT_MAX ?
                0.0f : expf(maximum - next_maximum);
            const float weight = expf(score - next_maximum);
            denominator = denominator * old_scale + weight;
            maximum = next_maximum;
            old_scale_shared = old_scale;
        } else {
            old_scale_shared = 1.0f;
        }
        if (body_meta != nullptr) {
            const size_t row = ((size_t) stream * n_query + query) * n_query_heads + query_head;
            body_meta[row] = make_float2(maximum, denominator);
        }
        weight_shared = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    }
    __syncthreads();

    float * output = (float *) (
        dst_data + query_head * nbd1 + query * nbd2 + stream * nbd3);
#pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        const int dim = slice * GGML_CUDA_FATTN_KVARN_DIM + tid;
        output[dim] = accumulator[slice] * old_scale_shared * weight_shared;
    }
}

static inline bool ggml_cuda_fattn_kvarn_portable_supported(
        const ggml_cuda_fattn_kvarn_plan & plan,
        const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * kt = dst->src[5];
    const ggml_tensor * vt = dst->src[6];
    const ggml_tensor * mt = dst->src[7];
    const ggml_tensor * aux = dst->src[8];
    const ggml_tensor * rd = dst->src[9];
    const bool tail_attached = kt != nullptr;
    // Source 8 is query ordering for an attached exact tail, or the optional
    // body softmax metadata output for a standalone/coordinator body pass.
    const ggml_tensor * qo = tail_attached ? aux : nullptr;
    const ggml_tensor * body_meta = tail_attached ? nullptr : aux;
    const bool tail_ok = !tail_attached ||
        (vt != nullptr && mt != nullptr && qo != nullptr && rd != nullptr &&
         (kt->type == GGML_TYPE_F16 || kt->type == GGML_TYPE_BF16) &&
         (vt->type == GGML_TYPE_F16 || vt->type == GGML_TYPE_BF16) &&
         mt->type == GGML_TYPE_F16 && qo->type == GGML_TYPE_I32 &&
         rd->type == GGML_TYPE_I32 && kt->ne[0] == q->ne[0] &&
         vt->ne[0] == q->ne[0] && kt->ne[2] == plan.n_kv_heads &&
         vt->ne[2] == plan.n_kv_heads && qo->ne[1] == rd->ne[1] &&
         rd->ne[0] >= 6 + mt->ne[0]);
    const bool body_meta_ok = body_meta == nullptr ||
        (body_meta->type == GGML_TYPE_F32 && body_meta->ne[0] == 2 &&
         body_meta->ne[1] == q->ne[2] && body_meta->ne[2] == q->ne[1] &&
         body_meta->ne[3] == q->ne[3] && ggml_is_contiguous(body_meta));
    return ggml_cuda_fattn_kvarn_rotated_decode_domain(dst) &&
        (q->ne[0] == 128 || q->ne[0] == 256 || q->ne[0] == 512) &&
        q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        q->ne[0] == dst->src[1]->ne[0] && q->ne[0] == dst->src[2]->ne[0] &&
        q->ne[1] > 0 && q->ne[2] > 0 && q->ne[3] == plan.n_stream &&
        q->ne[2] % plan.n_kv_heads == 0 &&
        (mask == nullptr || mask->type == GGML_TYPE_F16) &&
        (sinks == nullptr || sinks->type == GGML_TYPE_F32) &&
        tail_ok && body_meta_ok;
}

template<int D>
static void ggml_cuda_fattn_kvarn_portable_launch(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_cuda_fattn_kvarn_plan & plan) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * kt = dst->src[5];
    const ggml_tensor * vt = dst->src[6];
    const ggml_tensor * mt = dst->src[7];
    const ggml_tensor * aux = dst->src[8];
    const ggml_tensor * rd = dst->src[9];
    const bool tail_attached = kt != nullptr;
    const ggml_tensor * qo = tail_attached ? aux : nullptr;
    const ggml_tensor * body_meta = tail_attached ? nullptr : aux;
    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    const size_t n_desc = (size_t) plan.n_stream * plan.n_kv_heads;
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> k_desc(pool, n_desc);
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> v_desc(pool, n_desc);
    ggml_cuda_fattn_kvarn_init_descs(
        plan, k_desc.get(), v_desc.get(), 0, 0, stream);
    ggml_cuda_kv_memory_transient_stats_record_kvarn(
        k_desc.actual_size + v_desc.actual_size, 0, 0,
        k_desc.actual_size + v_desc.actual_size);

    const dim3 blocks(
        (uint32_t) q->ne[1], (uint32_t) q->ne[2], (uint32_t) q->ne[3]);
    ggml_cuda_fattn_kvarn_portable_kernel<D>
        <<<blocks, GGML_CUDA_FATTN_KVARN_DIM, 0, stream>>>(
            (const char *) q->data,
            k_desc.get(), v_desc.get(),
            mask ? (const char *) mask->data : nullptr,
            sinks ? (const float *) sinks->data : nullptr,
            kt ? (const char *) kt->data : nullptr,
            vt ? (const char *) vt->data : nullptr,
            mt ? (const char *) mt->data : nullptr,
            qo ? (const int32_t *) qo->data : nullptr,
            rd ? (const int32_t *) rd->data : nullptr,
            body_meta ? (float2 *) body_meta->data : nullptr,
            (char *) dst->data,
            scale, max_bias, logit_softcap,
            q->nb[1], q->nb[2], q->nb[3],
            mask ? mask->nb[0] : 0,
            mask ? mask->nb[1] : 0,
            mask ? mask->nb[2] : 0,
            mask ? mask->nb[3] : 0,
            mask ? mask->ne[2] : 1,
            mask ? mask->ne[3] : 1,
            dst->nb[1], dst->nb[2], dst->nb[3],
            kt ? kt->nb[1] : 0,
            kt ? kt->nb[2] : 0,
            vt ? vt->nb[1] : 0,
            vt ? vt->nb[2] : 0,
            mt ? mt->nb[0] : 0,
            mt ? mt->nb[1] : 0,
            mt ? mt->nb[3] : 0,
            qo ? (int) qo->ne[0] : 0,
            qo ? (int) ggml_nelements(qo) : 0,
            rd ? (int) rd->ne[0] : 0,
            mt ? (int) mt->ne[0] : 0,
            kt && kt->type == GGML_TYPE_BF16,
            vt && vt->type == GGML_TYPE_BF16,
            plan.n_kv, (int) q->ne[1], (int) q->ne[2],
            plan.n_kv_heads);
    CUDA_CHECK(cudaGetLastError());
}

static bool ggml_cuda_flash_attn_ext_kvarn_portable(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_cuda_fattn_kvarn_plan & plan) {
    if (!ggml_cuda_fattn_kvarn_portable_supported(plan, dst)) {
        return false;
    }
    switch (dst->src[0]->ne[0]) {
        case 128: ggml_cuda_fattn_kvarn_portable_launch<128>(ctx, dst, plan); return true;
        case 256: ggml_cuda_fattn_kvarn_portable_launch<256>(ctx, dst, plan); return true;
        case 512: ggml_cuda_fattn_kvarn_portable_launch<512>(ctx, dst, plan); return true;
        default: return false;
    }
}
