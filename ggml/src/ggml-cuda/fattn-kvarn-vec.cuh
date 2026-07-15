#pragma once

#include "fattn-kvarn-vec-decl.cuh"
#include "fattn-mma-kvarn-decode.cuh"

enum {
    GGML_CUDA_FATTN_KVARN_VEC_INVALID = 0,
    GGML_CUDA_FATTN_KVARN_VEC_STAGE   = 1,
    GGML_CUDA_FATTN_KVARN_VEC_RECORD  = 2,
};

struct ggml_cuda_fattn_kvarn_vec_ref {
    int source;
    int pos;
    int stage_pos;
    int record_group;
};

static __device__ __forceinline__ ggml_cuda_fattn_kvarn_vec_ref
ggml_cuda_fattn_kvarn_vec_resolve(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int token,
        const int n_kv) {
    ggml_cuda_fattn_kvarn_vec_ref ref = {};
    if (token < 0 || token >= n_kv) {
        return ref;
    }

    int group;
    if (desc.swa) {
        const int64_t abs_pos = desc.indices[token];
        if (abs_pos < 0) {
            return ref;
        }
        group = (int) (abs_pos / GGML_CUDA_FATTN_KVARN_DIM);
        ref.pos = (int) (abs_pos - (int64_t) group * GGML_CUDA_FATTN_KVARN_DIM);
        if (ggml_cuda_fattn_kvarn_group_from_stage(desc, group)) {
            ref.source = GGML_CUDA_FATTN_KVARN_VEC_STAGE;
            ref.stage_pos = (group % desc.stage_groups) * GGML_CUDA_FATTN_KVARN_DIM + ref.pos;
        } else if (ggml_cuda_fattn_kvarn_group_from_record(desc, group)) {
            ref.source = GGML_CUDA_FATTN_KVARN_VEC_RECORD;
            ref.record_group = group % desc.groups_per_stream;
        }
        return ref;
    }

    group = token / GGML_CUDA_FATTN_KVARN_DIM;
    ref.pos = token - group * GGML_CUDA_FATTN_KVARN_DIM;
    const bool from_stage = ggml_cuda_fattn_kvarn_group_from_stage(desc, group);
    if (from_stage) {
        ref.source = GGML_CUDA_FATTN_KVARN_VEC_STAGE;
        const int stage_base = desc.stream * GGML_CUDA_FATTN_KVARN_DIM * desc.stage_groups;
        ref.stage_pos = stage_base + (group == 0 ? ref.pos :
            GGML_CUDA_FATTN_KVARN_DIM +
            ((group - 1) % desc.tail_groups) * GGML_CUDA_FATTN_KVARN_DIM + ref.pos);
    } else if (ggml_cuda_fattn_kvarn_group_from_record(desc, group)) {
        ref.source = GGML_CUDA_FATTN_KVARN_VEC_RECORD;
        ref.record_group = desc.stream * desc.groups_per_stream + group;
    }
    return ref;
}

template<int BITS, bool VALUE>
static __device__ __forceinline__ float ggml_cuda_fattn_kvarn_vec_load(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const ggml_cuda_fattn_kvarn_vec_ref & ref,
        const int slice,
        const int dim) {
    const int record_head = desc.head_base + slice;
    if (ref.source == GGML_CUDA_FATTN_KVARN_VEC_STAGE) {
        return ggml_cuda_fattn_kvarn_load_stage_rotated(desc, ref.stage_pos, record_head, dim);
    }
    if (ref.source != GGML_CUDA_FATTN_KVARN_VEC_RECORD) {
        return 0.0f;
    }

    const uint8_t * record = desc.records +
        ((int64_t) ref.record_group * desc.n_record_heads + record_head) * desc.record_bytes;
    constexpr int payload_bytes =
        GGML_CUDA_FATTN_KVARN_DIM * GGML_CUDA_FATTN_KVARN_DIM * BITS / 8;
    const half * scale_axis = (const half *) (record + payload_bytes);
    const half * zp_axis = scale_axis + GGML_CUDA_FATTN_KVARN_DIM;
    const half * other_axis = zp_axis + GGML_CUDA_FATTN_KVARN_DIM;
    const int row = VALUE ? ref.pos : dim;
    const int col = VALUE ? dim : ref.pos;
    const uint8_t q = ggml_cuda_fattn_kvarn_unpack_record(
        record, row * GGML_CUDA_FATTN_KVARN_DIM + col, BITS);
    return (float(q) * __half2float(scale_axis[row]) + __half2float(zp_axis[row])) *
        __half2float(other_axis[col]);
}

template<int D, int TOKENS_PER_SPLIT, int MAX_GQA, int K_BITS, int V_BITS>
static __global__ void ggml_cuda_fattn_kvarn_vec_kernel(
        const char * Q,
        const ggml_cuda_fattn_kvarn_desc * k_descs,
        const ggml_cuda_fattn_kvarn_desc * v_descs,
        const char * mask,
        float * partial,
        float2 * partial_meta,
        float scale,
        float logit_softcap,
        int64_t nb02,
        int64_t nb03,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int ne33,
        int n_kv,
        int n_q_heads,
        int n_kv_heads,
        int gqa_ratio,
        int n_gqa_blocks,
        int n_splits) {
    constexpr int SLICES = D / GGML_CUDA_FATTN_KVARN_DIM;
    static_assert(D == 256, "KVarN vec production route currently supports D256 heads");
    static_assert(TOKENS_PER_SPLIT == 8 || TOKENS_PER_SPLIT == 16 || TOKENS_PER_SPLIT == 32,
        "KVarN vec supports 8-, 16-, or 32-token partitions");
    static_assert(MAX_GQA == 2, "KVarN vec shares one dequantized K/V value across a bounded D256 GQA group");
    static_assert(WARP_SIZE == 32, "KVarN vec requires CUDA warp size 32");

    // Match the std fattn-vec block of 128 threads / 4 warps. The decode is latency-bound at ~6%
    // occupancy because each split-block only ran 2 warps (one per 128-wide slice), so the strided
    // per-element KVarN dequant loads had no sibling warps to hide behind. Splitting every slice
    // across DIM_GROUPS warps doubles the resident warps (each walks half the dims), without changing
    // the per-split math or the partial/combine layout. DIM_GROUPS==1 degenerates to the old kernel.
    constexpr int DIM_GROUPS    = 4 / SLICES;
    constexpr int DIM_PER_GROUP = GGML_CUDA_FATTN_KVARN_DIM / DIM_GROUPS;
    static_assert(4 % SLICES == 0, "KVarN vec targets 4 warps per block");
    static_assert(DIM_PER_GROUP >= WARP_SIZE && DIM_PER_GROUP % WARP_SIZE == 0,
        "KVarN vec needs each dim group to be a warp-aligned slab");

    const int split = blockIdx.x;
    const int gqa_block = blockIdx.y % n_gqa_blocks;
    const int kv_head = blockIdx.y / n_gqa_blocks;
    const int stream = blockIdx.z;
    const int lane = threadIdx.x;
    const int slice = threadIdx.y;
    const int dim_group = threadIdx.z;
    const int warp_id = dim_group * SLICES + slice;
    const int tid = warp_id * WARP_SIZE + lane;
    const int nthreads = SLICES * DIM_GROUPS * WARP_SIZE;
    const int dim_base = dim_group * DIM_PER_GROUP;
    const int q_head0 = kv_head * gqa_ratio + gqa_block * MAX_GQA;
    const int gqa_head_count = min(MAX_GQA, gqa_ratio - gqa_block * MAX_GQA);
    const int token_begin = split * TOKENS_PER_SPLIT;
    const int token_end = min(n_kv, token_begin + TOKENS_PER_SPLIT);

    const ggml_cuda_fattn_kvarn_desc & k_desc = k_descs[stream * n_kv_heads + kv_head];
    const ggml_cuda_fattn_kvarn_desc & v_desc = v_descs[stream * n_kv_heads + kv_head];

    __shared__ __align__(16) half q_sh[MAX_GQA][D];
    __shared__ ggml_cuda_fattn_kvarn_vec_ref k_refs[TOKENS_PER_SPLIT];
    __shared__ ggml_cuda_fattn_kvarn_vec_ref v_refs[TOKENS_PER_SPLIT];
    __shared__ float score_partial[SLICES][DIM_GROUPS][MAX_GQA][TOKENS_PER_SPLIT];
    __shared__ float weights[MAX_GQA][TOKENS_PER_SPLIT];
    __shared__ float max_score[MAX_GQA];
    __shared__ float denominator[MAX_GQA];

    for (int i = tid; i < MAX_GQA * D; i += nthreads) {
        const int h = i / D;
        const int dim = i % D;
        float value = 0.0f;
        if (h < gqa_head_count && q_head0 + h < n_q_heads) {
            const float * q = (const float *) (Q + nb03 * stream + nb02 * (q_head0 + h));
            value = q[dim] * scale;
        }
        q_sh[h][dim] = __float2half(value);
    }
    if (warp_id == 0 && lane < TOKENS_PER_SPLIT) {
        const int token = token_begin + lane;
        k_refs[lane] = ggml_cuda_fattn_kvarn_vec_resolve(k_desc, token, n_kv);
        v_refs[lane] = ggml_cuda_fattn_kvarn_vec_resolve(v_desc, token, n_kv);
    }
    __syncthreads();

    constexpr int DIM_WORKERS = WARP_SIZE / TOKENS_PER_SPLIT;
    const int token_lane = lane % TOKENS_PER_SPLIT;
    const int dim_worker = lane / TOKENS_PER_SPLIT;
    float dot[MAX_GQA] = {};
    if (token_begin + token_lane < token_end &&
            k_refs[token_lane].source != GGML_CUDA_FATTN_KVARN_VEC_INVALID) {
#pragma unroll
        for (int d = dim_worker; d < DIM_PER_GROUP; d += DIM_WORKERS) {
            const int dim = dim_base + d;
            const float k = ggml_cuda_fattn_kvarn_vec_load<K_BITS, false>(
                k_desc, k_refs[token_lane], slice, dim);
#pragma unroll
            for (int h = 0; h < MAX_GQA; ++h) {
                dot[h] += k * __half2float(q_sh[h][slice * GGML_CUDA_FATTN_KVARN_DIM + dim]);
            }
        }
    }
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset >= TOKENS_PER_SPLIT; offset >>= 1) {
#pragma unroll
        for (int h = 0; h < MAX_GQA; ++h) {
            dot[h] += __shfl_down_sync(0xFFFFFFFFu, dot[h], offset);
        }
    }
#pragma unroll
    for (int h = 0; h < MAX_GQA; ++h) {
        if (lane < TOKENS_PER_SPLIT) {
            score_partial[slice][dim_group][h][lane] = dot[h];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        const half * mask_h = mask != nullptr ?
            (const half *) (mask + nb33 * (stream % ne33)) : nullptr;
#pragma unroll
        for (int h = 0; h < MAX_GQA; ++h) {
            const int token = token_begin + lane;
            float score = -FLT_MAX / 2.0f;
            if (lane < TOKENS_PER_SPLIT && token < token_end &&
                    h < gqa_head_count && q_head0 + h < n_q_heads) {
                score = 0.0f;
#pragma unroll
                for (int s = 0; s < SLICES; ++s) {
#pragma unroll
                    for (int g = 0; g < DIM_GROUPS; ++g) {
                        score += score_partial[s][g][h][lane];
                    }
                }
                if (logit_softcap != 0.0f) {
                    score = logit_softcap * tanhf(score);
                }
                if (mask_h != nullptr) {
                    score += __half2float(*(const half *) (
                        (const char *) mask_h + nb30 * token + nb31 * 0));
                }
            }

            float m = score + FATTN_KQ_MAX_OFFSET;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                m = fmaxf(m, __shfl_down_sync(0xFFFFFFFFu, m, offset));
            }
            m = __shfl_sync(0xFFFFFFFFu, m, 0);

            const float diff = score - m;
            const float weight = diff >= SOFTMAX_FTZ_THRESHOLD ? expf(diff) : 0.0f;
            float denom = weight;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                denom += __shfl_down_sync(0xFFFFFFFFu, denom, offset);
            }
            denom = __shfl_sync(0xFFFFFFFFu, denom, 0);
            if (lane < TOKENS_PER_SPLIT) {
                weights[h][lane] = weight;
            }
            if (lane == 0) {
                max_score[h] = m;
                denominator[h] = denom;
            }
        }
    }
    __syncthreads();

    for (int d = lane; d < DIM_PER_GROUP; d += WARP_SIZE) {
        const int dim = dim_base + d;
        float out[MAX_GQA] = {};
#pragma unroll
        for (int t = 0; t < TOKENS_PER_SPLIT; ++t) {
            if (token_begin + t >= token_end ||
                    v_refs[t].source == GGML_CUDA_FATTN_KVARN_VEC_INVALID) {
                continue;
            }
            const float v = ggml_cuda_fattn_kvarn_vec_load<V_BITS, true>(
                v_desc, v_refs[t], slice, dim);
#pragma unroll
            for (int h = 0; h < MAX_GQA; ++h) {
                out[h] += weights[h][t] * v;
            }
        }

        const int global_dim = slice * GGML_CUDA_FATTN_KVARN_DIM + dim;
#pragma unroll
        for (int h = 0; h < MAX_GQA; ++h) {
            const int q_head = q_head0 + h;
            if (h < gqa_head_count && q_head < n_q_heads) {
                const size_t base =
                    ((size_t) stream * n_q_heads + q_head) * n_splits + split;
                partial[base * D + global_dim] = out[h];
            }
        }
    }

    if (warp_id == 0 && lane < gqa_head_count && q_head0 + lane < n_q_heads) {
        const int q_head = q_head0 + lane;
        const size_t base = ((size_t) stream * n_q_heads + q_head) * n_splits + split;
        partial_meta[base] = make_float2(max_score[lane], denominator[lane]);
    }
}

template<int D, int TOKENS_PER_SPLIT, int K_BITS, int V_BITS>
static void ggml_cuda_fattn_kvarn_vec_launch_tps(
        const ggml_cuda_fattn_kvarn_decode_args & args) {
    constexpr int max_gqa = ggml_cuda_fattn_kvarn_vec_max_gqa<D>();
    constexpr int slices = D / GGML_CUDA_FATTN_KVARN_DIM;
    constexpr int dim_groups = 4 / slices; // 128 threads / 4 warps, matching std fattn-vec
    const dim3 blocks_split(
        (uint32_t) args.n_splits,
        (uint32_t) (args.n_kv_heads * args.n_gqa_blocks),
        (uint32_t) args.n_stream);
    ggml_cuda_fattn_kvarn_vec_kernel<D, TOKENS_PER_SPLIT, max_gqa, K_BITS, V_BITS>
        <<<blocks_split, dim3(WARP_SIZE, slices, dim_groups), 0, args.stream>>>(
            args.Q, args.k_descs, args.v_descs, args.mask, args.partial, args.partial_meta,
            args.scale, args.logit_softcap, args.nb02, args.nb03,
            args.nb30, args.nb31, args.nb33, args.ne33, args.n_kv,
            args.n_q_heads, args.n_kv_heads, args.gqa_ratio, args.n_gqa_blocks,
            args.n_splits);
    CUDA_CHECK(cudaGetLastError());
}

template<int D, int K_BITS, int V_BITS>
void ggml_cuda_fattn_kvarn_vec_launch(const ggml_cuda_fattn_kvarn_decode_args & args) {
    switch (ggml_cuda_fattn_kvarn_vec_tokens_per_split()) {
        case 8:
            ggml_cuda_fattn_kvarn_vec_launch_tps<D, 8, K_BITS, V_BITS>(args);
            break;
        case 16:
            ggml_cuda_fattn_kvarn_vec_launch_tps<D, 16, K_BITS, V_BITS>(args);
            break;
        case 32:
            ggml_cuda_fattn_kvarn_vec_launch_tps<D, 32, K_BITS, V_BITS>(args);
            break;
        default:
            GGML_ABORT("invalid KVarN vec token partition");
    }

    const dim3 blocks_combine(
        (uint32_t) args.n_q_heads, 1, (uint32_t) args.n_stream);
    ggml_cuda_fattn_kvarn_decode_combine_kernel<D>
        <<<blocks_combine, GGML_CUDA_FATTN_KVARN_DECODE_THREADS,
            args.n_splits * sizeof(float), args.stream>>>(
            args.partial, args.partial_meta, args.dst, args.dst_meta,
            args.n_splits, 1, args.n_q_heads);
    CUDA_CHECK(cudaGetLastError());
}
