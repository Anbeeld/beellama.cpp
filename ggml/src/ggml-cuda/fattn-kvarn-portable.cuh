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

static constexpr int GGML_CUDA_FATTN_KVARN_PORTABLE_SPLIT_TOKENS = 256;

struct ggml_cuda_fattn_kvarn_portable_ref {
    bool valid;
    bool stage;
    int stage_pos;
    int record_group;
    int pos;
};

static __device__ __forceinline__ ggml_cuda_fattn_kvarn_portable_ref
ggml_cuda_fattn_kvarn_portable_resolve(
        const ggml_cuda_fattn_kvarn_desc & desc, int token) {
    ggml_cuda_fattn_kvarn_portable_ref result = {};
    int group;
    int pos;
    bool explicitly_staged = false;
    int assigned_slot = -1;
    if (desc.swa || desc.read_indirect) {
        const int64_t encoded = desc.indices[token];
        if (encoded == -1) {
            return result;
        }
        const int64_t absolute = ggml_cuda_fattn_kvarn_read_cell(
                desc, encoded, explicitly_staged, &assigned_slot);
        group = int(absolute/GGML_CUDA_FATTN_KVARN_DIM);
        pos = int(absolute - int64_t(group)*GGML_CUDA_FATTN_KVARN_DIM);
    } else {
        group = token/GGML_CUDA_FATTN_KVARN_DIM;
        pos = token - group*GGML_CUDA_FATTN_KVARN_DIM;
    }
    result.pos = pos;
    result.record_group = desc.swa ? group % desc.groups_per_stream :
        desc.stream * desc.groups_per_stream + group;
    result.stage = explicitly_staged ||
        (!(desc.read_indirect && !desc.swa) &&
         ggml_cuda_fattn_kvarn_group_from_stage(desc, group));
    result.valid = result.stage || (!explicitly_staged &&
        (desc.read_indirect && !desc.swa ? true :
         ggml_cuda_fattn_kvarn_group_from_record(desc, group)));
    if (result.stage) {
        result.stage_pos = ggml_cuda_fattn_kvarn_stage_pos(
                desc, group, pos, assigned_slot);
    }
    return result;
}

static __device__ __forceinline__ float ggml_cuda_fattn_kvarn_portable_load_resolved(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const ggml_cuda_fattn_kvarn_portable_ref & ref,
        int slice,
        int dim) {
    const int record_head = desc.head_base + slice;
    if (ref.stage) {
        return ggml_cuda_fattn_kvarn_load_stage_rotated(
            desc, ref.stage_pos, record_head, dim);
    }
    if (!ref.valid) {
        return 0.0f;
    }
    const uint8_t * record = desc.records +
        ((int64_t) ref.record_group * desc.n_record_heads + record_head) *
            desc.record_bytes;
    const int rows = desc.value ? GGML_CUDA_FATTN_KVARN_DIM : desc.record_dim;
    const int cols = desc.value ? desc.record_dim : GGML_CUDA_FATTN_KVARN_DIM;
    const int payload_bytes = desc.record_dim * GGML_CUDA_FATTN_KVARN_DIM * desc.bits / 8;
    const half * scale_axis = (const half *) (record + payload_bytes);
    const half * zp_axis = scale_axis + rows;
    const half * other_axis = zp_axis + rows;
    const int row = desc.value ? ref.pos : dim;
    const int col = desc.value ? dim : ref.pos;
    const uint8_t q = ggml_cuda_fattn_kvarn_unpack_record(
        record, row * cols + col, desc.bits);
    return (float(q) * __half2float(scale_axis[row]) + __half2float(zp_axis[row])) *
        __half2float(other_axis[col]);
}

template<int D>
static __device__ __forceinline__ void ggml_cuda_fattn_kvarn_portable_stage_rotated(
        const ggml_cuda_fattn_kvarn_desc & desc,
        int stage_pos,
        int tid,
        float (&values)[D/(D == 64 ? 32 : GGML_CUDA_FATTN_KVARN_DIM)]) {
    constexpr int RECORD_DIM = D == 64 ? 64 : GGML_CUDA_FATTN_KVARN_DIM;
    constexpr int THREADS = D == 64 ? 32 : RECORD_DIM;
    constexpr int VALUES = D/THREADS;
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int dim = i * THREADS + tid;
        const int slice = dim / RECORD_DIM;
        const int lane = dim - slice * RECORD_DIM;
        const int head = desc.head_base + slice;
        values[i] = __half2float(desc.stage[
            ((int64_t) stage_pos*desc.n_record_heads + head)*RECORD_DIM + lane]);
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
        float2 * output_meta,
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
        bool v_original_domain,
        int n_kv,
        int n_query,
        int n_query_heads,
        int n_kv_heads,
        int n_splits) {
    static_assert(D == 64 || D == 128 || D == 256 || D == 512,
        "portable KVarN attention supports 64/128/256/512-wide heads");
    constexpr int RECORD_DIM = D == 64 ? 64 : GGML_CUDA_FATTN_KVARN_DIM;
    constexpr int THREADS = D == 64 ? 32 : RECORD_DIM;
    constexpr int VALUES = D / THREADS;

    const int query = (int) blockIdx.x % n_query;
    const int split = (int) blockIdx.x / n_query;
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

    __shared__ float reduction[RECORD_DIM];
    __shared__ float transform[RECORD_DIM];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float old_scale_shared;
    __shared__ float weight_shared;

    float accumulator[VALUES] = {};
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

    const int body_begin = (int) ((int64_t) n_body * split / n_splits);
    const int body_end = (int) ((int64_t) n_body * (split + 1) / n_splits);
    for (int packed = body_begin; packed < body_end; ++packed) {
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
        float k_values[VALUES] = {};
        if (k_ref.stage) {
            ggml_cuda_fattn_kvarn_portable_stage_rotated<D>(
                    k_desc, k_ref.stage_pos, tid, k_values);
        } else {
#pragma unroll
            for (int i = 0; i < VALUES; ++i) {
                const int dim = i * THREADS + tid;
                const int slice = dim / RECORD_DIM;
                const int lane = dim - slice * RECORD_DIM;
                k_values[i] = ggml_cuda_fattn_kvarn_portable_load_resolved(
                        k_desc, k_ref, slice, lane);
            }
        }
        float partial = 0.0f;
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            const int dim = i * THREADS + tid;
            partial += q[dim] * k_values[i];
        }
        if constexpr (D == 64) {
#pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1) {
                partial += __shfl_down_sync(0xffffffffu, partial, stride);
            }
            if (tid == 0) {
                reduction[0] = partial;
            }
        } else {
            reduction[tid] = partial;
            __syncthreads();
            for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduction[tid] += reduction[tid + stride];
                }
                __syncthreads();
            }
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
        if constexpr (D == 64) {
            __syncwarp();
        } else {
            __syncthreads();
        }

        float v_values[VALUES] = {};
        if (v_ref.stage) {
            ggml_cuda_fattn_kvarn_portable_stage_rotated<D>(
                    v_desc, v_ref.stage_pos, tid, v_values);
        } else {
#pragma unroll
            for (int i = 0; i < VALUES; ++i) {
                const int dim = i * THREADS + tid;
                const int slice = dim / RECORD_DIM;
                const int lane = dim - slice * RECORD_DIM;
                v_values[i] = ggml_cuda_fattn_kvarn_portable_load_resolved(
                        v_desc, v_ref, slice, lane);
            }
        }
        if (v_original_domain) {
#pragma unroll
            for (int i = 0; i < VALUES; ++i) {
                reduction[i * THREADS + tid] = v_values[i];
            }
            __syncthreads();
#pragma unroll
            for (int stride = 1; stride < D; stride <<= 1) {
#pragma unroll
                for (int i = 0; i < VALUES; ++i) {
                    const int dim = i * THREADS + tid;
                    const float self = reduction[dim];
                    const float other = reduction[dim ^ stride];
                    transform[dim] = (dim & stride) ? other - self : self + other;
                }
                __syncthreads();
#pragma unroll
                for (int i = 0; i < VALUES; ++i) {
                    const int dim = i * THREADS + tid;
                    reduction[dim] = transform[dim];
                }
                __syncthreads();
            }
#pragma unroll
            for (int i = 0; i < VALUES; ++i) {
                v_values[i] = reduction[i * THREADS + tid] * rsqrtf(float(D));
            }
        }
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            accumulator[i] = accumulator[i] * old_scale_shared +
                v_values[i] * weight_shared;
        }
        if constexpr (D == 64) {
            __syncwarp();
        } else {
            __syncthreads();
        }
    }

    if (split == 0 && k_tail_data != nullptr) {
        const int n_tail = desc[4];
        for (int token = 0; token < n_tail; ++token) {
            const int slot = desc[6 + token];
            float partial = 0.0f;
#pragma unroll
            for (int i = 0; i < VALUES; ++i) {
                const int dim = i * THREADS + tid;
                const char * ptr = k_tail_data +
                    (size_t) slot * nbkt1 + (size_t) kv_head * nbkt2 +
                    (size_t) dim * sizeof(uint16_t);
                const float kval = ggml_cuda_fattn_kvarn_load_tail(
                    ptr, k_tail_bf16);
                partial += q[dim] * kval;
            }
            if constexpr (D == 64) {
#pragma unroll
                for (int stride = 16; stride > 0; stride >>= 1) {
                    partial += __shfl_down_sync(0xffffffffu, partial, stride);
                }
                if (tid == 0) {
                    reduction[0] = partial;
                }
            } else {
                reduction[tid] = partial;
                __syncthreads();
                for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        reduction[tid] += reduction[tid + stride];
                    }
                    __syncthreads();
                }
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
            if constexpr (D == 64) {
                __syncwarp();
            } else {
                __syncthreads();
            }

#pragma unroll
            for (int i = 0; i < VALUES; ++i) {
                const int dim = i * THREADS + tid;
                const char * ptr = v_tail_data +
                    (size_t) slot * nbvt1 + (size_t) kv_head * nbvt2 +
                    (size_t) dim * sizeof(uint16_t);
                const float vval = ggml_cuda_fattn_kvarn_load_tail(
                    ptr, v_tail_bf16);
                accumulator[i] =
                    accumulator[i] * old_scale_shared + vval * weight_shared;
            }
            if constexpr (D == 64) {
                __syncwarp();
            } else {
                __syncthreads();
            }
        }
    }

    if (tid == 0) {
        if (split == 0 && sinks != nullptr) {
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
        if (output_meta != nullptr) {
            const size_t row = ((size_t) stream * n_query + query) * n_query_heads + query_head;
            output_meta[row * n_splits + split] = make_float2(maximum, denominator);
        }
        weight_shared = n_splits == 1 && denominator > 0.0f ? 1.0f / denominator : 1.0f;
    }
    if constexpr (D == 64) {
        __syncwarp();
    } else {
        __syncthreads();
    }

    if (n_splits == 1) {
        float * output = (float *) (
            dst_data + query_head * nbd1 + query * nbd2 + stream * nbd3);
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            const int dim = i * THREADS + tid;
            output[dim] = accumulator[i] * old_scale_shared * weight_shared;
        }
    } else {
        const size_t row = ((size_t) stream * n_query + query) * n_query_heads + query_head;
        float * partial = (float *) dst_data + (row * n_splits + split) * D;
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            const int dim = i * THREADS + tid;
            partial[dim] = accumulator[i] * old_scale_shared;
        }
    }
}

template<int GQA>
static __global__ void ggml_cuda_fattn_kvarn_d64_gqa_kernel(
        const char * q_data,
        const ggml_cuda_fattn_kvarn_desc * k_descs,
        const ggml_cuda_fattn_kvarn_desc * v_descs,
        const char * mask_data,
        const float * sinks,
        float2 * output_meta,
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
        int n_kv,
        int n_query,
        int n_query_heads,
        int n_kv_heads,
        int n_splits) {
    static_assert(GQA == 1 || GQA == 2 || GQA == 4 || GQA == 8,
        "D64 portable GQA kernel supports power-of-two ratios through 8");
    const int query = (int) blockIdx.x % n_query;
    const int split = (int) blockIdx.x / n_query;
    const int kv_head = (int) blockIdx.y;
    const int stream = (int) blockIdx.z;
    const int tid = (int) threadIdx.x;
    const int query_head0 = kv_head * GQA;
    if (query_head0 + GQA > n_query_heads || kv_head >= n_kv_heads) {
        return;
    }

    const ggml_cuda_fattn_kvarn_desc & k_desc =
        k_descs[(size_t) stream * n_kv_heads + kv_head];
    const ggml_cuda_fattn_kvarn_desc & v_desc =
        v_descs[(size_t) stream * n_kv_heads + kv_head];
    const int body_begin = (int) ((int64_t) n_kv * split / n_splits);
    const int body_end = (int) ((int64_t) n_kv * (split + 1) / n_splits);

    __shared__ float maximum[GQA];
    __shared__ float denominator[GQA];
    __shared__ float old_scale[GQA];
    __shared__ float weight[GQA];
    __shared__ float slopes[GQA];
    if (tid < GQA) {
        maximum[tid] = -FLT_MAX;
        denominator[tid] = 0.0f;
        uint32_t n_head_log2 = 1;
        while ((n_head_log2 << 1) <= (uint32_t) n_query_heads) {
            n_head_log2 <<= 1;
        }
        const int query_head = query_head0 + tid;
        const float m0 = exp2f(-max_bias / float(n_head_log2));
        const float m1 = exp2f(-(max_bias / 2.0f) / float(n_head_log2));
        slopes[tid] = max_bias > 0.0f ?
            (query_head < (int) n_head_log2 ?
                powf(m0, float(query_head + 1)) :
                powf(m1, float(2 * (query_head - (int) n_head_log2) + 1))) : 1.0f;
    }
    __syncwarp();

    float accumulator[GQA][2] = {};
    int cached_k_group = -1;
    int cached_v_group = -1;
    const uint8_t * k_record = nullptr;
    const uint8_t * v_record = nullptr;
    const half * k_other = nullptr;
    const half * v_scale = nullptr;
    const half * v_zp = nullptr;
    float k_scale[2] = {};
    float k_zp[2] = {};
    float v_other[2] = {};
    for (int token = body_begin; token < body_end; ++token) {
        const auto k_ref = ggml_cuda_fattn_kvarn_portable_resolve(k_desc, token);
        const auto v_ref = ggml_cuda_fattn_kvarn_portable_resolve(v_desc, token);
        if (!k_ref.stage && k_ref.valid && k_ref.record_group != cached_k_group) {
            cached_k_group = k_ref.record_group;
            k_record = k_desc.records +
                ((int64_t) cached_k_group * k_desc.n_record_heads + k_desc.head_base) *
                    k_desc.record_bytes;
            const int payload_bytes = 64 * GGML_CUDA_FATTN_KVARN_DIM * k_desc.bits / 8;
            const half * scale_axis = (const half *) (k_record + payload_bytes);
            const half * zp_axis = scale_axis + 64;
            k_other = zp_axis + 64;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int dim = 2 * tid + i;
                k_scale[i] = __half2float(scale_axis[dim]);
                k_zp[i] = __half2float(zp_axis[dim]);
            }
        }
        if (!v_ref.stage && v_ref.valid && v_ref.record_group != cached_v_group) {
            cached_v_group = v_ref.record_group;
            v_record = v_desc.records +
                ((int64_t) cached_v_group * v_desc.n_record_heads + v_desc.head_base) *
                    v_desc.record_bytes;
            const int payload_bytes = GGML_CUDA_FATTN_KVARN_DIM * 64 * v_desc.bits / 8;
            v_scale = (const half *) (v_record + payload_bytes);
            v_zp = v_scale + GGML_CUDA_FATTN_KVARN_DIM;
            const half * other_axis = v_zp + GGML_CUDA_FATTN_KVARN_DIM;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                v_other[i] = __half2float(other_axis[2 * tid + i]);
            }
        }

        float k_values[2];
        float v_values[2];
        if (k_ref.stage || !k_ref.valid) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                k_values[i] = ggml_cuda_fattn_kvarn_portable_load_resolved(
                    k_desc, k_ref, 0, 2 * tid + i);
            }
        } else {
            const float k_col = __half2float(k_other[k_ref.pos]);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int dim = 2 * tid + i;
                const uint8_t qv = ggml_cuda_fattn_kvarn_unpack_record(
                    k_record, dim * GGML_CUDA_FATTN_KVARN_DIM + k_ref.pos, k_desc.bits);
                k_values[i] = (float(qv) * k_scale[i] + k_zp[i]) * k_col;
            }
        }
        if (v_ref.stage || !v_ref.valid) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                v_values[i] = ggml_cuda_fattn_kvarn_portable_load_resolved(
                    v_desc, v_ref, 0, 2 * tid + i);
            }
        } else {
            const uint16_t pair = ggml_cuda_fattn_kvarn_unpack_record_pair(
                v_record, v_ref.pos * 64 + 2 * tid, v_desc.bits);
            const float scale_row = __half2float(v_scale[v_ref.pos]);
            const float zp_row = __half2float(v_zp[v_ref.pos]);
            v_values[0] = (float(pair & 0xffu) * scale_row + zp_row) * v_other[0];
            v_values[1] = (float(pair >> 8) * scale_row + zp_row) * v_other[1];
        }

        float scores[GQA];
#pragma unroll
        for (int h = 0; h < GQA; ++h) {
            const int query_head = query_head0 + h;
            const float * q = (const float *) (
                q_data + query * nbq1 + query_head * nbq2 + stream * nbq3);
            float dot = q[2 * tid] * k_values[0] + q[2 * tid + 1] * k_values[1];
#pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1) {
                dot += __shfl_down_sync(0xffffffffu, dot, stride);
            }
            scores[h] = dot;
        }
        if (tid == 0) {
#pragma unroll
            for (int h = 0; h < GQA; ++h) {
                const int query_head = query_head0 + h;
                float mask_value = 0.0f;
                if (mask_data != nullptr) {
                    const half * mask = (const half *) (
                        mask_data + token * nbm0 + query * nbm1 +
                        (query_head % nmask2) * nbm2 + (stream % nmask3) * nbm3);
                    mask_value = slopes[h] * __half2float(*mask);
                }
                float score = scores[h] * scale;
                if (logit_softcap != 0.0f) {
                    score = logit_softcap * tanhf(score);
                }
                score += mask_value;
                if (mask_value == -INFINITY) {
                    old_scale[h] = 1.0f;
                    weight[h] = 0.0f;
                } else {
                    const float next_maximum = fmaxf(maximum[h], score);
                    const float previous = maximum[h] == -FLT_MAX ?
                        0.0f : expf(maximum[h] - next_maximum);
                    const float next_weight = expf(score - next_maximum);
                    maximum[h] = next_maximum;
                    denominator[h] = denominator[h] * previous + next_weight;
                    old_scale[h] = previous;
                    weight[h] = next_weight;
                }
            }
        }
        __syncwarp();
#pragma unroll
        for (int h = 0; h < GQA; ++h) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                accumulator[h][i] = accumulator[h][i] * old_scale[h] +
                    v_values[i] * weight[h];
            }
        }
        __syncwarp();
    }

    if (tid == 0) {
#pragma unroll
        for (int h = 0; h < GQA; ++h) {
            if (split == 0 && sinks != nullptr) {
                const float score = sinks[query_head0 + h];
                const float next_maximum = fmaxf(maximum[h], score);
                const float previous = maximum[h] == -FLT_MAX ?
                    0.0f : expf(maximum[h] - next_maximum);
                const float next_weight = expf(score - next_maximum);
                denominator[h] = denominator[h] * previous + next_weight;
                maximum[h] = next_maximum;
                old_scale[h] = previous;
            } else {
                old_scale[h] = 1.0f;
            }
            const int query_head = query_head0 + h;
            const size_t row = ((size_t) stream * n_query + query) *
                n_query_heads + query_head;
            if (output_meta != nullptr) {
                output_meta[row * n_splits + split] =
                    make_float2(maximum[h], denominator[h]);
            }
        }
    }
    __syncwarp();

#pragma unroll
    for (int h = 0; h < GQA; ++h) {
        const int query_head = query_head0 + h;
        const size_t row = ((size_t) stream * n_query + query) *
            n_query_heads + query_head;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int dim = 2 * tid + i;
            const float numerator = accumulator[h][i] * old_scale[h];
            if (n_splits == 1) {
                float * output = (float *) (
                    dst_data + query_head * nbd1 + query * nbd2 + stream * nbd3);
                output[dim] = denominator[h] > 0.0f ?
                    numerator / denominator[h] : 0.0f;
            } else {
                ((float *) dst_data)[(row * n_splits + split) * 64 + dim] = numerator;
            }
        }
    }
}

template<int D>
static __global__ void ggml_cuda_fattn_kvarn_portable_combine_kernel(
        const float * partial,
        const float2 * partial_meta,
        char * dst_data,
        float2 * dst_meta,
        int64_t nbd1,
        int64_t nbd2,
        int64_t nbd3,
        int n_splits,
        int n_query,
        int n_query_heads) {
    const int query_head = (int) blockIdx.x;
    const int query = (int) blockIdx.y;
    const int stream = (int) blockIdx.z;
    const int tid = (int) threadIdx.x;
    const size_t row = ((size_t) stream * n_query + query) * n_query_heads + query_head;

    extern __shared__ float weights[];
    __shared__ float denominator;
    if (tid == 0) {
        float m = -FLT_MAX;
        for (int split = 0; split < n_splits; ++split) {
            const float2 meta = partial_meta[row * n_splits + split];
            if (meta.y > 0.0f) {
                m = fmaxf(m, meta.x);
            }
        }
        float denom = 0.0f;
        for (int split = 0; split < n_splits; ++split) {
            const float2 meta = partial_meta[row * n_splits + split];
            const float weight = meta.y > 0.0f ? expf(meta.x - m) : 0.0f;
            weights[split] = weight;
            denom += weight * meta.y;
        }
        denominator = denom;
        if (dst_meta != nullptr) {
            dst_meta[row] = make_float2(m, denom);
        }
    }
    __syncthreads();

    if (tid < D) {
        float value = 0.0f;
        for (int split = 0; split < n_splits; ++split) {
            value += weights[split] * partial[(row * n_splits + split) * D + tid];
        }
        float * output = (float *) (
            dst_data + query_head * nbd1 + query * nbd2 + stream * nbd3);
        output[tid] = denominator > 0.0f ? value / denominator : 0.0f;
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
    const bool domain_ok = ggml_cuda_fattn_kvarn_rotated_decode_domain(dst) ||
        (q->ne[0] == 64 &&
         ggml_cuda_fattn_kvarn_domain(dst) == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V);
    return domain_ok &&
        (q->ne[0] == 64 || q->ne[0] == 128 || q->ne[0] == 256 || q->ne[0] == 512) &&
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
    const int n_splits = D == 64 ?
        (plan.n_kv + GGML_CUDA_FATTN_KVARN_PORTABLE_SPLIT_TOKENS - 1) /
            GGML_CUDA_FATTN_KVARN_PORTABLE_SPLIT_TOKENS : 1;
    const size_t n_rows = (size_t) q->ne[1] * q->ne[2] * q->ne[3];
    const size_t partial_elements = n_splits > 1 ? n_rows * n_splits * D : 1;
    const size_t partial_meta_elements = n_splits > 1 ? n_rows * n_splits : 1;
    ggml_cuda_pool_alloc<float> partial(pool, partial_elements);
    ggml_cuda_pool_alloc<float2> partial_meta(pool, partial_meta_elements);
    const bool v_original_domain = ggml_cuda_fattn_kvarn_v_original_domain(dst);
    ggml_cuda_fattn_kvarn_init_descs(
        plan, k_desc.get(), v_desc.get(), 0, v_original_domain ? 1 : 0, stream);
    ggml_cuda_kv_memory_transient_stats_record_kvarn(
        k_desc.actual_size + v_desc.actual_size + partial.actual_size + partial_meta.actual_size,
        0, 0,
        k_desc.actual_size + v_desc.actual_size + partial.actual_size + partial_meta.actual_size);

    const dim3 blocks(
        (uint32_t) (q->ne[1] * n_splits), (uint32_t) q->ne[2], (uint32_t) q->ne[3]);
    constexpr int RECORD_DIM = D == 64 ? 64 : GGML_CUDA_FATTN_KVARN_DIM;
    constexpr int THREADS = D == 64 ? 32 : RECORD_DIM;
    bool launched_gqa = false;
    if constexpr (D == 64) {
        const int gqa = (int) (q->ne[2] / plan.n_kv_heads);
        if (!tail_attached && !v_original_domain &&
                (gqa == 1 || gqa == 2 || gqa == 4 || gqa == 8)) {
            const dim3 gqa_blocks(
                (uint32_t) (q->ne[1] * n_splits),
                (uint32_t) plan.n_kv_heads,
                (uint32_t) q->ne[3]);
#define GGML_CUDA_FATTN_KVARN_D64_GQA_LAUNCH(GQA) \
            ggml_cuda_fattn_kvarn_d64_gqa_kernel<GQA> \
                <<<gqa_blocks, 32, 0, stream>>>( \
                    (const char *) q->data, k_desc.get(), v_desc.get(), \
                    mask ? (const char *) mask->data : nullptr, \
                    sinks ? (const float *) sinks->data : nullptr, \
                    n_splits > 1 ? partial_meta.get() : \
                        (body_meta ? (float2 *) body_meta->data : nullptr), \
                    n_splits > 1 ? (char *) partial.get() : (char *) dst->data, \
                    scale, max_bias, logit_softcap, \
                    q->nb[1], q->nb[2], q->nb[3], \
                    mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, \
                    mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0, \
                    mask ? mask->ne[2] : 1, mask ? mask->ne[3] : 1, \
                    dst->nb[1], dst->nb[2], dst->nb[3], \
                    plan.n_kv, (int) q->ne[1], (int) q->ne[2], \
                    plan.n_kv_heads, n_splits)
            switch (gqa) {
                case 1: GGML_CUDA_FATTN_KVARN_D64_GQA_LAUNCH(1); break;
                case 2: GGML_CUDA_FATTN_KVARN_D64_GQA_LAUNCH(2); break;
                case 4: GGML_CUDA_FATTN_KVARN_D64_GQA_LAUNCH(4); break;
                case 8: GGML_CUDA_FATTN_KVARN_D64_GQA_LAUNCH(8); break;
                default: break;
            }
#undef GGML_CUDA_FATTN_KVARN_D64_GQA_LAUNCH
            launched_gqa = true;
            CUDA_CHECK(cudaGetLastError());
        }
    }
    if (!launched_gqa) {
        ggml_cuda_fattn_kvarn_portable_kernel<D>
            <<<blocks, THREADS, 0, stream>>>(
            (const char *) q->data,
            k_desc.get(), v_desc.get(),
            mask ? (const char *) mask->data : nullptr,
            sinks ? (const float *) sinks->data : nullptr,
            kt ? (const char *) kt->data : nullptr,
            vt ? (const char *) vt->data : nullptr,
            mt ? (const char *) mt->data : nullptr,
            qo ? (const int32_t *) qo->data : nullptr,
            rd ? (const int32_t *) rd->data : nullptr,
            n_splits > 1 ? partial_meta.get() :
                (body_meta ? (float2 *) body_meta->data : nullptr),
            n_splits > 1 ? (char *) partial.get() : (char *) dst->data,
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
            v_original_domain,
            plan.n_kv, (int) q->ne[1], (int) q->ne[2],
            plan.n_kv_heads, n_splits);
        CUDA_CHECK(cudaGetLastError());
    }

    if (n_splits > 1) {
        const dim3 combine_blocks(
            (uint32_t) q->ne[2], (uint32_t) q->ne[1], (uint32_t) q->ne[3]);
        ggml_cuda_fattn_kvarn_portable_combine_kernel<D>
            <<<combine_blocks, D, (size_t) n_splits * sizeof(float), stream>>>(
                partial.get(), partial_meta.get(), (char *) dst->data,
                body_meta ? (float2 *) body_meta->data : nullptr,
                dst->nb[1], dst->nb[2], dst->nb[3],
                n_splits, (int) q->ne[1], (int) q->ne[2]);
        CUDA_CHECK(cudaGetLastError());
    }
}

static bool ggml_cuda_flash_attn_ext_kvarn_portable(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_cuda_fattn_kvarn_plan & plan) {
    if (!ggml_cuda_fattn_kvarn_portable_supported(plan, dst)) {
        return false;
    }
    switch (dst->src[0]->ne[0]) {
        case  64: ggml_cuda_fattn_kvarn_portable_launch< 64>(ctx, dst, plan); return true;
        case 128: ggml_cuda_fattn_kvarn_portable_launch<128>(ctx, dst, plan); return true;
        case 256: ggml_cuda_fattn_kvarn_portable_launch<256>(ctx, dst, plan); return true;
        case 512: ggml_cuda_fattn_kvarn_portable_launch<512>(ctx, dst, plan); return true;
        default: return false;
    }
}
