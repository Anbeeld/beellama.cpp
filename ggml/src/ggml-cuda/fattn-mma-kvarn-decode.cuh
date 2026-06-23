#pragma once

#include "fattn-mma-kvarn-decode-decl.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-kvarn-load.cuh"
#include "mma.cuh"

using namespace ggml_cuda_mma;

static constexpr int GGML_CUDA_FATTN_KVARN_DECODE_THREADS = 256;
static constexpr int GGML_CUDA_FATTN_KVARN_DECODE_CHUNK   = 16;

static __device__ __forceinline__ bool ggml_cuda_fattn_kvarn_decode_group_from_record(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int group) {
    if (desc.swa) {
        return false;
    }
    const int live_group = desc.live_group;
    const bool from_stage = group == 0 ||
        (group > 0 && group <= live_group && group + (desc.tail_groups - 1) >= live_group);
    return !from_stage && group < live_group && group < desc.groups_per_stream;
}

template<int BITS>
static __device__ __forceinline__ int ggml_cuda_fattn_kvarn_decode_unpack(
        const uint8_t * raw,
        const int index,
        const int bits) {
    if constexpr (BITS == 8) {
        return raw[index];
    } else if constexpr (BITS == 4) {
        return (raw[index >> 1] >> (4 * (index & 1))) & 0x0f;
    } else if constexpr (BITS == 2) {
        return (raw[index >> 2] >> (2 * (index & 3))) & 0x03;
    } else if constexpr (BITS > 0) {
        const int bit_offset = index * BITS;
        const int byte_offset = bit_offset >> 3;
        const int shift = bit_offset & 7;
        uint16_t packed = (uint16_t) raw[byte_offset];
        if (shift + BITS > 8) {
            packed |= (uint16_t) raw[byte_offset + 1] << 8;
        }
        return (packed >> shift) & ((1 << BITS) - 1);
    }

    if (bits == 8) {
        return raw[index];
    }
    if (bits == 4) {
        return (raw[index >> 1] >> (4 * (index & 1))) & 0x0f;
    }
    if (bits == 2) {
        return (raw[index >> 2] >> (2 * (index & 3))) & 0x03;
    }
    const int bit_offset = index * bits;
    const int byte_offset = bit_offset >> 3;
    const int shift = bit_offset & 7;
    uint16_t packed = (uint16_t) raw[byte_offset];
    if (shift + bits > 8) {
        packed |= (uint16_t) raw[byte_offset + 1] << 8;
    }
    return (packed >> shift) & ((1 << bits) - 1);
}

template<int D, int MAX_GQA, int SPLIT_TOKENS, int NWARPS, int K_BITS, int V_BITS>
static __global__ void ggml_cuda_fattn_kvarn_decode_mma_kernel(
        const char * Q,
        const ggml_cuda_fattn_kvarn_desc * k_descs,
        const ggml_cuda_fattn_kvarn_desc * v_descs,
        const char * mask,
        float * partial,
        float2 * partial_meta,
        float scale,
        float logit_softcap,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int ne33,
        int n_kv,
        int n_q,
        int n_q_heads,
        int n_kv_heads,
        int gqa_ratio,
        int n_gqa_blocks,
        int n_splits) {
    const int split = blockIdx.x;
    const int q_index = blockIdx.y % n_q;
    const int gqa_block = (blockIdx.y / n_q) % n_gqa_blocks;
    const int kv_head = blockIdx.y / (n_q * n_gqa_blocks);
    const int stream = blockIdx.z;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * WARP_SIZE + lane;
    constexpr int SLICES = D / GGML_CUDA_FATTN_KVARN_DIM;
    constexpr int TOKENS_PER_CHUNK = GGML_CUDA_FATTN_KVARN_DECODE_CHUNK;
    constexpr int TOKEN_CHUNKS = SPLIT_TOKENS / TOKENS_PER_CHUNK;
    constexpr int Q_STRIDE2 = D / 2 + 4;
    constexpr int P_STRIDE2 = SPLIT_TOKENS / 2 + 4;

    static_assert(D == 128 || D == 256 || D == 512, "KVarN decode MMA supports 128/256/512-wide heads");
    static_assert(MAX_GQA > 0 && MAX_GQA <= 8, "KVarN decode MMA expects at most eight GQA heads");
    static_assert(SPLIT_TOKENS == 32 || SPLIT_TOKENS == 64 || SPLIT_TOKENS == 128,
        "KVarN decode MMA supports 32-, 64-, or 128-token splits");
    static_assert(NWARPS >= (MAX_GQA + 1) / 2 && NWARPS <= 4,
        "KVarN decode MMA needs one half-warp per GQA head");
    static_assert(K_BITS == 2 || K_BITS == 3 || K_BITS == 4 ||
        K_BITS == 5 || K_BITS == 6 || K_BITS == 8, "invalid compile-time KVarN K bits");
    static_assert(V_BITS == 2 || V_BITS == 3 || V_BITS == 4 ||
        V_BITS == 5 || V_BITS == 6 || V_BITS == 8, "invalid compile-time KVarN V bits");
    static_assert(WARP_SIZE == 32, "KVarN decode MMA currently targets CUDA warp size 32");

    __shared__ __align__(16) half2 q_sh[MAX_GQA][Q_STRIDE2];
    __shared__ __align__(16) float score_sh[MAX_GQA][SPLIT_TOKENS];
    __shared__ __align__(16) half2 p_sh[MAX_GQA][P_STRIDE2];
    __shared__ float scale_axis_sh[SLICES][GGML_CUDA_FATTN_KVARN_DIM];
    __shared__ float zp_axis_sh[SLICES][GGML_CUDA_FATTN_KVARN_DIM];
    __shared__ float other_axis_sh[SLICES][GGML_CUDA_FATTN_KVARN_DIM];
    __shared__ float m_sh[MAX_GQA];
    __shared__ float denom_sh[MAX_GQA];

    const ggml_cuda_fattn_kvarn_desc & k_desc = k_descs[stream * n_kv_heads + kv_head];
    const ggml_cuda_fattn_kvarn_desc & v_desc = v_descs[stream * n_kv_heads + kv_head];
    const int q_head0 = kv_head * gqa_ratio + gqa_block * MAX_GQA;
    const int gqa_head_count = min(MAX_GQA, gqa_ratio - gqa_block * MAX_GQA);
    const int token_begin = split * SPLIT_TOKENS;
    const int token_end = min(n_kv, token_begin + SPLIT_TOKENS);
    const int group = token_begin / GGML_CUDA_FATTN_KVARN_DIM;
    const int group_pos_begin = token_begin - group * GGML_CUDA_FATTN_KVARN_DIM;
    const bool k_from_record = ggml_cuda_fattn_kvarn_decode_group_from_record(k_desc, group);
    const bool v_from_record = ggml_cuda_fattn_kvarn_decode_group_from_record(v_desc, group);
    const int k_payload_bytes = GGML_CUDA_FATTN_KVARN_DIM * GGML_CUDA_FATTN_KVARN_DIM * K_BITS / 8;
    const int v_payload_bytes = GGML_CUDA_FATTN_KVARN_DIM * GGML_CUDA_FATTN_KVARN_DIM * V_BITS / 8;
    const int k_row_bytes = GGML_CUDA_FATTN_KVARN_DIM * K_BITS / 8;
    const int v_row_bytes = GGML_CUDA_FATTN_KVARN_DIM * V_BITS / 8;
    const int record_group_k = k_desc.stream * k_desc.groups_per_stream + group;
    const int record_group_v = v_desc.stream * v_desc.groups_per_stream + group;

    const uint8_t * k_records[SLICES];
    const uint8_t * v_records[SLICES];
#pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        k_records[slice] = k_desc.records +
            ((int64_t) record_group_k * k_desc.n_record_heads + k_desc.head_base + slice) * k_desc.record_bytes;
        v_records[slice] = v_desc.records +
            ((int64_t) record_group_v * v_desc.n_record_heads + v_desc.head_base + slice) * v_desc.record_bytes;
    }

    half * q_h = (half *) q_sh;
    for (int i = tid; i < MAX_GQA * D; i += NWARPS * WARP_SIZE) {
        const int h = i / D;
        const int dim = i % D;
        float value = 0.0f;
        if (h < gqa_head_count && q_head0 + h < n_q_heads) {
            const float * q = (const float *) (Q + nb03 * stream + nb02 * (q_head0 + h) + nb01 * q_index);
            value = q[dim] * scale;
        }
        q_h[h * (2 * Q_STRIDE2) + dim] = __float2half(value);
    }

    if (k_from_record) {
        for (int i = tid; i < SLICES * GGML_CUDA_FATTN_KVARN_DIM; i += NWARPS * WARP_SIZE) {
            const int slice = i / GGML_CUDA_FATTN_KVARN_DIM;
            const int axis = i % GGML_CUDA_FATTN_KVARN_DIM;
            const half * scale_axis = (const half *) (k_records[slice] + k_payload_bytes);
            const half * zp_axis = scale_axis + GGML_CUDA_FATTN_KVARN_DIM;
            const half * other_axis = zp_axis + GGML_CUDA_FATTN_KVARN_DIM;
            scale_axis_sh[slice][axis] = __half2float(scale_axis[axis]);
            zp_axis_sh[slice][axis] = __half2float(zp_axis[axis]);
            other_axis_sh[slice][axis] = __half2float(other_axis[axis]);
        }
    }
    __syncthreads();

    using T_A = tile<16, 8, half2>;
    using T_B = tile<8, 8, half2>;
    using T_C = tile<16, 8, float>;

#pragma unroll
    for (int chunk = warp; chunk < TOKEN_CHUNKS; chunk += NWARPS) {
        const int token0 = token_begin + chunk * TOKENS_PER_CHUNK;
        T_C scores;
#pragma unroll
        for (int l = 0; l < T_C::ne; ++l) {
            scores.x[l] = 0.0f;
        }
#pragma unroll 1
        for (int dim0 = 0; dim0 < D; dim0 += 2 * T_A::J) {
            T_A k_a;
            T_B q_b;
#pragma unroll
            for (int l = 0; l < T_A::ne; ++l) {
                const int token_local = T_A::get_i(l);
                const int dim = dim0 + 2 * T_A::get_j(l);
                const int slice = dim / GGML_CUDA_FATTN_KVARN_DIM;
                const int local_dim = dim % GGML_CUDA_FATTN_KVARN_DIM;
                float x0;
                float x1;
                if (k_from_record) {
                    const int pos = group_pos_begin + chunk * TOKENS_PER_CHUNK + token_local;
                    const uint8_t * row0 = k_records[slice] + (local_dim + 0) * k_row_bytes;
                    const uint8_t * row1 = k_records[slice] + (local_dim + 1) * k_row_bytes;
                    const int q0 = ggml_cuda_fattn_kvarn_decode_unpack<K_BITS>(row0, pos, K_BITS);
                    const int q1 = ggml_cuda_fattn_kvarn_decode_unpack<K_BITS>(row1, pos, K_BITS);
                    const float other = other_axis_sh[slice][pos];
                    x0 = (float(q0) * scale_axis_sh[slice][local_dim + 0] +
                            zp_axis_sh[slice][local_dim + 0]) * other;
                    x1 = (float(q1) * scale_axis_sh[slice][local_dim + 1] +
                            zp_axis_sh[slice][local_dim + 1]) * other;
                } else {
                    const int token = token0 + token_local;
                    x0 = token < token_end ?
                        ggml_cuda_fattn_kvarn_load_rotated(k_desc, token, slice, local_dim + 0) : 0.0f;
                    x1 = token < token_end ?
                        ggml_cuda_fattn_kvarn_load_rotated(k_desc, token, slice, local_dim + 1) : 0.0f;
                }
                k_a.x[l] = make_half2(x0, x1);
            }
            load_ldmatrix(q_b, q_sh[0] + dim0 / 2, Q_STRIDE2);
            mma(scores, k_a, q_b);
        }

        const half * mask_h = mask != nullptr ? (const half *) (mask + nb33 * (stream % ne33)) : nullptr;
#pragma unroll
        for (int l = 0; l < T_C::ne; ++l) {
            const int j = T_C::get_i(l);
            const int h = T_C::get_j(l);
            const int token = token0 + j;
            float score = -FLT_MAX / 2.0f;
            if (h < gqa_head_count && q_head0 + h < n_q_heads && token < token_end) {
                score = scores.x[l];
                if (logit_softcap != 0.0f) {
                    score = logit_softcap * tanhf(score);
                }
                if (mask_h != nullptr) {
                    score += __half2float(*(const half *) ((const char *) mask_h + nb30 * token + nb31 * q_index));
                }
            }
            if (h < MAX_GQA) {
                score_sh[h][chunk * TOKENS_PER_CHUNK + j] = score;
            }
        }
        __syncwarp();
    }
    __syncthreads();

    const int h = tid / 16;
    const int lane_h = tid % 16;
    float m = -FLT_MAX / 2.0f;
    if (h < MAX_GQA) {
        for (int token = lane_h; token < SPLIT_TOKENS; token += 16) {
            m = fmaxf(m, score_sh[h][token] + FATTN_KQ_MAX_OFFSET);
        }
    }
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, offset, 16));
    }

    half * p_h = (half *) p_sh;
    float denom = 0.0f;
    if (h < MAX_GQA) {
        for (int token = lane_h; token < SPLIT_TOKENS; token += 16) {
            const float diff = score_sh[h][token] - m;
            const float weight = diff >= SOFTMAX_FTZ_THRESHOLD ? expf(diff) : 0.0f;
            denom += weight;
            p_h[h * (2 * P_STRIDE2) + token] = __float2half(weight);
        }
    }
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        denom += __shfl_xor_sync(0xFFFFFFFFu, denom, offset, 16);
    }
    if (h < MAX_GQA && lane_h == 0) {
        m_sh[h] = m;
        denom_sh[h] = denom;
    }
    __syncthreads();

    if (v_from_record) {
        for (int i = tid; i < SLICES * GGML_CUDA_FATTN_KVARN_DIM; i += NWARPS * WARP_SIZE) {
            const int slice = i / GGML_CUDA_FATTN_KVARN_DIM;
            const int axis = i % GGML_CUDA_FATTN_KVARN_DIM;
            const half * scale_axis = (const half *) (v_records[slice] + v_payload_bytes);
            const half * zp_axis = scale_axis + GGML_CUDA_FATTN_KVARN_DIM;
            const half * other_axis = zp_axis + GGML_CUDA_FATTN_KVARN_DIM;
            scale_axis_sh[slice][axis] = __half2float(scale_axis[axis]);
            zp_axis_sh[slice][axis] = __half2float(zp_axis[axis]);
            other_axis_sh[slice][axis] = __half2float(other_axis[axis]);
        }
    }
    __syncthreads();

#pragma unroll 1
    for (int dim0 = warp * TOKENS_PER_CHUNK; dim0 < D; dim0 += NWARPS * TOKENS_PER_CHUNK) {
        const int slice = dim0 / GGML_CUDA_FATTN_KVARN_DIM;
        const int local_dim0 = dim0 % GGML_CUDA_FATTN_KVARN_DIM;
        T_C out;
#pragma unroll
        for (int l = 0; l < T_C::ne; ++l) {
            out.x[l] = 0.0f;
        }

#pragma unroll 1
        for (int chunk = 0; chunk < TOKEN_CHUNKS; ++chunk) {
            const int token0 = token_begin + chunk * TOKENS_PER_CHUNK;
            T_A v_a;
            T_B p_b;
#pragma unroll
            for (int l = 0; l < T_A::ne; ++l) {
                const int local_dim = T_A::get_i(l);
                const int token_local = 2 * T_A::get_j(l);
                float x0;
                float x1;
                if (v_from_record) {
                    const int pos0 = group_pos_begin + chunk * TOKENS_PER_CHUNK + token_local + 0;
                    const int pos1 = group_pos_begin + chunk * TOKENS_PER_CHUNK + token_local + 1;
                    const uint8_t * row0 = v_records[slice] + pos0 * v_row_bytes;
                    const uint8_t * row1 = v_records[slice] + pos1 * v_row_bytes;
                    const int q0 = ggml_cuda_fattn_kvarn_decode_unpack<V_BITS>(row0, local_dim0 + local_dim, V_BITS);
                    const int q1 = ggml_cuda_fattn_kvarn_decode_unpack<V_BITS>(row1, local_dim0 + local_dim, V_BITS);
                    const float other = other_axis_sh[slice][local_dim0 + local_dim];
                    x0 = (float(q0) * scale_axis_sh[slice][pos0] +
                            zp_axis_sh[slice][pos0]) * other;
                    x1 = (float(q1) * scale_axis_sh[slice][pos1] +
                            zp_axis_sh[slice][pos1]) * other;
                } else {
                    const int token0_pair = token0 + token_local;
                    x0 = token0_pair + 0 < token_end ?
                        ggml_cuda_fattn_kvarn_load_rotated(v_desc, token0_pair + 0, slice, local_dim0 + local_dim) : 0.0f;
                    x1 = token0_pair + 1 < token_end ?
                        ggml_cuda_fattn_kvarn_load_rotated(v_desc, token0_pair + 1, slice, local_dim0 + local_dim) : 0.0f;
                }
                v_a.x[l] = make_half2(x0, x1);
            }
            load_ldmatrix(p_b, p_sh[0] + chunk * (TOKENS_PER_CHUNK / 2), P_STRIDE2);
            mma(out, v_a, p_b);
        }

#pragma unroll
        for (int l = 0; l < T_C::ne; ++l) {
            const int dim = dim0 + T_C::get_i(l);
            const int head = T_C::get_j(l);
            const int q_head = q_head0 + head;
            if (head < gqa_head_count && q_head < n_q_heads) {
                const size_t base = (((size_t) stream * n_q + q_index) * n_q_heads + q_head) * n_splits + split;
                partial[base * D + dim] = out.x[l];
            }
        }
        __syncwarp();
    }
    __syncthreads();

    if (tid < gqa_head_count && q_head0 + tid < n_q_heads) {
        const int q_head = q_head0 + tid;
        const size_t base = (((size_t) stream * n_q + q_index) * n_q_heads + q_head) * n_splits + split;
        partial_meta[base] = make_float2(m_sh[tid], denom_sh[tid]);
    }
}

template<int D>
static __global__ void ggml_cuda_fattn_kvarn_decode_combine_kernel(
        const float * partial,
        const float2 * partial_meta,
        float * dst,
        int n_splits,
        int n_q,
        int n_q_heads) {
    const int q_head = blockIdx.x;
    const int q_index = blockIdx.y;
    const int stream = blockIdx.z;
    const int tid = threadIdx.x;

    __shared__ float reduce_sh[GGML_CUDA_FATTN_KVARN_DECODE_THREADS];
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

    for (int dim = tid; dim < D; dim += blockDim.x) {
        float out = 0.0f;
        if (denom > 0.0f) {
            for (int split = 0; split < n_splits; ++split) {
                const size_t base = (((size_t) stream * n_q + q_index) * n_q_heads + q_head) * n_splits + split;
                out += split_weights[split] * partial[base * D + dim];
            }
            out /= denom;
        }
        dst[(((size_t) stream * n_q + q_index) * n_q_heads + q_head) * D + dim] = out;
    }
}

template<int D, int K_BITS, int V_BITS>
void ggml_cuda_fattn_kvarn_decode_launch(const ggml_cuda_fattn_kvarn_decode_args & args) {
    constexpr int split_tokens = 64;
    const dim3 blocks_split(
        (uint32_t) args.n_splits,
        (uint32_t) (args.n_kv_heads * args.n_gqa_blocks * args.n_q),
        (uint32_t) args.n_stream);

#define GGML_CUDA_FATTN_KVARN_LAUNCH(MAX_GQA) \
    ggml_cuda_fattn_kvarn_decode_mma_kernel<D, MAX_GQA, split_tokens, 4, K_BITS, V_BITS> \
        <<<blocks_split, dim3(WARP_SIZE, 4, 1), 0, args.stream>>>( \
            args.Q, args.k_descs, args.v_descs, args.mask, args.partial, args.partial_meta, \
            args.scale, args.logit_softcap, args.nb01, args.nb02, args.nb03, \
            args.nb30, args.nb31, args.nb33, args.ne33, args.n_kv, args.n_q, \
            args.n_q_heads, args.n_kv_heads, args.gqa_ratio, args.n_gqa_blocks, args.n_splits)

    if (args.gqa_ratio == 6) {
        GGML_CUDA_FATTN_KVARN_LAUNCH(6);
    } else {
        GGML_CUDA_FATTN_KVARN_LAUNCH(GGML_CUDA_FATTN_KVARN_DECODE_MAX_GQA);
    }
#undef GGML_CUDA_FATTN_KVARN_LAUNCH
    CUDA_CHECK(cudaGetLastError());

    const dim3 blocks_combine((uint32_t) args.n_q_heads, (uint32_t) args.n_q, (uint32_t) args.n_stream);
    ggml_cuda_fattn_kvarn_decode_combine_kernel<D>
        <<<blocks_combine, GGML_CUDA_FATTN_KVARN_DECODE_THREADS, args.n_splits * sizeof(float), args.stream>>>(
            args.partial, args.partial_meta, args.dst, args.n_splits, args.n_q, args.n_q_heads);
    CUDA_CHECK(cudaGetLastError());
}
