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
    return ggml_cuda_fattn_kvarn_group_from_record(desc, group);
}

// True when an absolute SWA group sits in the record store (not the live/stage tail and
// still inside the ring window). Mirrors the SWA branch of ggml_cuda_fattn_kvarn_load_rotated.
static __device__ __forceinline__ bool ggml_cuda_fattn_kvarn_decode_swa_record_backed(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int group_global) {
    if (group_global < 0) {
        return false;
    }
    return ggml_cuda_fattn_kvarn_group_from_record(desc, group_global);
}

// Per-split tile plan: whether the cooperative direct-record fast path is usable, the in-group
// position of the first tile token, and the ring record-group index for that first group.
//
// Non-SWA: token == in-stream position, so a 64-token split always lands inside one record group
// (pos_begin is 0 or 64) — identical to the original linear mapping.
//
// SWA: the ring presents window cells in piecewise-contiguous absolute-position order (one wrap
// discontinuity per decode). We take the fast path when the split's valid tokens are contiguous in
// absolute position (endpoint check) and the first absolute group is record-backed. Group-straddle,
// stage (live group), ring-wrap, and non-contiguous tiles fall back per element to load_rotated,
// which independently resolves group/pos for any cell. Empty/out-of-window cells carry idx < 0 and
// are -inf-masked downstream, so reads for them are harmless.
struct ggml_cuda_fattn_kvarn_decode_tile {
    bool fast;
    int  pos_begin;
    int  record_group;
};

static __device__ __forceinline__ ggml_cuda_fattn_kvarn_decode_tile
ggml_cuda_fattn_kvarn_decode_plan_tile(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int token_begin,
        const int token_end,
        const int group,
        const int group_pos_begin) {
    ggml_cuda_fattn_kvarn_decode_tile tile;
    if (desc.swa) {
        const int64_t a0 = desc.indices[token_begin];
        const int64_t a1 = desc.indices[token_end - 1];
        const int span = (token_end - 1) - token_begin;
        const int g0 = a0 >= 0 ? (int) (a0 / GGML_CUDA_FATTN_KVARN_DIM) : -1;
        const bool contiguous = a0 >= 0 && a1 >= 0 && (int) (a1 - a0) == span;
        const bool rec0 = ggml_cuda_fattn_kvarn_decode_swa_record_backed(desc, g0);
        tile.fast = contiguous && rec0;
        tile.pos_begin = tile.fast ? (int) (a0 % GGML_CUDA_FATTN_KVARN_DIM) : 0;
        tile.record_group = tile.fast ? (g0 % desc.groups_per_stream) : 0;
    } else {
        tile.fast = ggml_cuda_fattn_kvarn_decode_group_from_record(desc, group);
        tile.pos_begin = group_pos_begin;
        tile.record_group = desc.stream * desc.groups_per_stream + group;
    }
    return tile;
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
    constexpr int PHYSICAL_WAVE_SIZE = ggml_cuda_get_physical_warp_size();
    const int tid = warp * PHYSICAL_WAVE_SIZE + lane;
    constexpr int SLICES = D / GGML_CUDA_FATTN_KVARN_DIM;
    constexpr int TOKENS_PER_CHUNK = GGML_CUDA_FATTN_KVARN_DECODE_CHUNK;
    constexpr int TOKEN_CHUNKS = SPLIT_TOKENS / TOKENS_PER_CHUNK;
    constexpr int WARPS_PER_CHUNK = D / GGML_CUDA_FATTN_KVARN_DIM;
    constexpr int CHUNKS_PER_PASS = NWARPS / WARPS_PER_CHUNK;
    constexpr int Q_STRIDE2 = D / 2 + 4;
    constexpr int P_STRIDE2 = SPLIT_TOKENS / 2 + 4;

    using T_A = tile<16, 8, half2>;
    using T_B = tile<8, 8, half2>;
    using T_C = tile<16, 8, float>;

    static_assert(D == 128 || D == 256 || D == 512, "KVarN decode MMA supports 128/256/512-wide heads");
    static_assert(MAX_GQA > 0 && MAX_GQA <= 8, "KVarN decode MMA expects at most eight GQA heads");
    static_assert(SPLIT_TOKENS == 64, "KVarN decode MMA production splits use 64 KV tokens");
    static_assert(NWARPS % WARPS_PER_CHUNK == 0 && NWARPS >= WARPS_PER_CHUNK &&
        NWARPS <= TOKEN_CHUNKS * WARPS_PER_CHUNK && NWARPS <= 16,
        "KVarN decode MMA needs whole 128-dim slice groups and at most sixteen warps");
    static_assert(K_BITS == 2 || K_BITS == 3 || K_BITS == 4 ||
        K_BITS == 5 || K_BITS == 6 || K_BITS == 8, "invalid compile-time KVarN K bits");
    static_assert(V_BITS == 2 || V_BITS == 3 || V_BITS == 4 ||
        V_BITS == 5 || V_BITS == 6 || V_BITS == 8, "invalid compile-time KVarN V bits");
    static_assert(PHYSICAL_WAVE_SIZE == 32 || PHYSICAL_WAVE_SIZE == 64,
        "KVarN decode MMA requires a physical wave size of 32 or 64");

    __shared__ __align__(16) half2 q_sh[MAX_GQA][Q_STRIDE2];
    __shared__ __align__(16) float score_partial_sh[NWARPS][MAX_GQA * TOKENS_PER_CHUNK];
    __shared__ __align__(16) float score_sh[MAX_GQA][SPLIT_TOKENS];
    __shared__ __align__(16) half2 p_sh[MAX_GQA][P_STRIDE2];
    __shared__ float scale_axis_sh[SLICES][GGML_CUDA_FATTN_KVARN_DIM];
    __shared__ float zp_axis_sh[SLICES][GGML_CUDA_FATTN_KVARN_DIM];
    __shared__ float other_axis_sh[SLICES][GGML_CUDA_FATTN_KVARN_DIM];
    __shared__ float zq_sh[SLICES][MAX_GQA];
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
    const ggml_cuda_fattn_kvarn_decode_tile k_tile =
        ggml_cuda_fattn_kvarn_decode_plan_tile(k_desc, token_begin, token_end, group, group_pos_begin);
    const ggml_cuda_fattn_kvarn_decode_tile v_tile =
        ggml_cuda_fattn_kvarn_decode_plan_tile(v_desc, token_begin, token_end, group, group_pos_begin);
    const bool k_from_record = k_tile.fast;
    const bool v_from_record = v_tile.fast;
    const int k_payload_bytes = GGML_CUDA_FATTN_KVARN_DIM * GGML_CUDA_FATTN_KVARN_DIM * K_BITS / 8;
    const int v_payload_bytes = GGML_CUDA_FATTN_KVARN_DIM * GGML_CUDA_FATTN_KVARN_DIM * V_BITS / 8;
    const int k_row_bytes = GGML_CUDA_FATTN_KVARN_DIM * K_BITS / 8;
    const int v_row_bytes = GGML_CUDA_FATTN_KVARN_DIM * V_BITS / 8;
    const int record_group_k = k_tile.record_group;
    const int record_group_v = v_tile.record_group;

    const bool k_split_in_group =
        k_from_record && (k_tile.pos_begin + SPLIT_TOKENS) <= GGML_CUDA_FATTN_KVARN_DIM;

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
    for (int i = tid; i < MAX_GQA * D; i += NWARPS * PHYSICAL_WAVE_SIZE) {
        const int h = i / D;
        const int dim = i % D;
        float value = 0.0f;
        if (h < gqa_head_count && q_head0 + h < n_q_heads) {
            const float * q = (const float *) (Q + nb03 * stream + nb02 * (q_head0 + h) + nb01 * q_index);
            value = q[dim] * scale;
        }
        q_h[h * (2 * Q_STRIDE2) + dim] = __float2half(value);
    }

    if (k_split_in_group) {
        for (int i = tid; i < SLICES * GGML_CUDA_FATTN_KVARN_DIM; i += NWARPS * PHYSICAL_WAVE_SIZE) {
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

    if (k_split_in_group) {
        const int n_targets = SLICES * gqa_head_count;
        for (int target = tid; target < n_targets; target += NWARPS * PHYSICAL_WAVE_SIZE) {
            const int slice = target / gqa_head_count;
            const int h = target % gqa_head_count;
            const half * q_row = (const half *) q_sh + h * (2 * Q_STRIDE2) + slice * GGML_CUDA_FATTN_KVARN_DIM;
            float zq = 0.0f;
            for (int dim = 0; dim < GGML_CUDA_FATTN_KVARN_DIM; ++dim) {
                const float q_val = __half2float(q_row[dim]);
                zq += zp_axis_sh[slice][dim] * q_val;
                const float q_prime = scale_axis_sh[slice][dim] * q_val;
                ((half *) q_sh)[h * (2 * Q_STRIDE2) + slice * GGML_CUDA_FATTN_KVARN_DIM + dim] =
                    __float2half(q_prime);
            }
            zq_sh[slice][h] = zq;
        }
    }
    __syncthreads();

    const int local_chunk = warp / WARPS_PER_CHUNK;
    const int warp_in_chunk = warp % WARPS_PER_CHUNK;
    const half * mask_h = mask != nullptr ? (const half *) (mask + nb33 * (stream % ne33)) : nullptr;

#pragma unroll 1
    for (int chunk_base = 0; chunk_base < TOKEN_CHUNKS; chunk_base += CHUNKS_PER_PASS) {
        const int chunk = chunk_base + local_chunk;
        const bool chunk_active = chunk < TOKEN_CHUNKS;
        const int token0 = token_begin + chunk * TOKENS_PER_CHUNK;
        T_C scores;
#pragma unroll
        for (int l = 0; l < T_C::ne; ++l) {
            scores.x[l] = 0.0f;
        }

        if (chunk_active) {
#pragma unroll 1
            for (int dim0 = warp_in_chunk * GGML_CUDA_FATTN_KVARN_DIM;
                    dim0 < (warp_in_chunk + 1) * GGML_CUDA_FATTN_KVARN_DIM;
                    dim0 += 2 * T_A::J) {
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
                    const int pos = k_tile.pos_begin + chunk * TOKENS_PER_CHUNK + token_local;
                    if (k_split_in_group) {
                        const uint8_t * row0 = k_records[slice] + (local_dim + 0) * k_row_bytes;
                        const uint8_t * row1 = k_records[slice] + (local_dim + 1) * k_row_bytes;
                        x0 = (float) ggml_cuda_fattn_kvarn_decode_unpack<K_BITS>(row0, pos, K_BITS);
                        x1 = (float) ggml_cuda_fattn_kvarn_decode_unpack<K_BITS>(row1, pos, K_BITS);
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

#pragma unroll
            for (int l = 0; l < T_C::ne; ++l) {
                const int j = T_C::get_i(l);
                const int h = T_C::get_j(l);
                if (h < MAX_GQA) {
                    float v = scores.x[l];
                    if (k_split_in_group && h < gqa_head_count) {
                        const int pos = k_tile.pos_begin + chunk * TOKENS_PER_CHUNK + j;
                        v = other_axis_sh[warp_in_chunk][pos] * (v + zq_sh[warp_in_chunk][h]);
                    }
                    score_partial_sh[warp][h * TOKENS_PER_CHUNK + j] = v;
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int stride = WARPS_PER_CHUNK / 2; stride > 0; stride >>= 1) {
            if (chunk_active && warp_in_chunk < stride) {
#pragma unroll
                for (int l = 0; l < T_C::ne; ++l) {
                    const int j = T_C::get_i(l);
                    const int h = T_C::get_j(l);
                    if (h < MAX_GQA) {
                        score_partial_sh[warp][h * TOKENS_PER_CHUNK + j] +=
                            score_partial_sh[warp + stride][h * TOKENS_PER_CHUNK + j];
                    }
                }
            }
            __syncthreads();
        }

        if (chunk_active && warp_in_chunk == 0) {
#pragma unroll
            for (int l = 0; l < T_C::ne; ++l) {
                const int j = T_C::get_i(l);
                const int h = T_C::get_j(l);
                const int token = token0 + j;
                float score = -FLT_MAX / 2.0f;
                if (h < gqa_head_count && q_head0 + h < n_q_heads && token < token_end) {
                    score = score_partial_sh[warp][h * TOKENS_PER_CHUNK + j];
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
        }
        __syncthreads();
    }

    half * p_h = (half *) p_sh;
#if defined(GGML_USE_HIP) && defined(CDNA)
    if (tid < MAX_GQA) {
        const int h = tid;
        float m = -FLT_MAX / 2.0f;
        for (int token = 0; token < SPLIT_TOKENS; ++token) {
            m = fmaxf(m, score_sh[h][token] + FATTN_KQ_MAX_OFFSET);
        }
        float denom = 0.0f;
        for (int token = 0; token < SPLIT_TOKENS; ++token) {
            const float diff = score_sh[h][token] - m;
            const float weight = diff >= SOFTMAX_FTZ_THRESHOLD ? expf(diff) : 0.0f;
            denom += weight;
            p_h[h * (2 * P_STRIDE2) + token] = __float2half(weight);
        }
        m_sh[h] = m;
        denom_sh[h] = denom;
    }
#else
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
#endif
    __syncthreads();

    if (v_from_record) {
        for (int i = tid; i < SLICES * GGML_CUDA_FATTN_KVARN_DIM; i += NWARPS * PHYSICAL_WAVE_SIZE) {
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
                const int pos0 = v_tile.pos_begin + chunk * TOKENS_PER_CHUNK + token_local + 0;
                const int pos1 = v_tile.pos_begin + chunk * TOKENS_PER_CHUNK + token_local + 1;
                if (v_from_record && pos1 < GGML_CUDA_FATTN_KVARN_DIM) {
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
        float2 * dst_meta,
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

    const size_t output_row = ((size_t) stream * n_q + q_index) * n_q_heads + q_head;
    if (tid == 0 && dst_meta != nullptr) {
        dst_meta[output_row] = make_float2(m, denom);
    }

    for (int dim = tid; dim < D; dim += blockDim.x) {
        float out = 0.0f;
        if (denom > 0.0f) {
            for (int split = 0; split < n_splits; ++split) {
                const size_t base = (((size_t) stream * n_q + q_index) * n_q_heads + q_head) * n_splits + split;
                out += split_weights[split] * partial[base * D + dim];
            }
            out /= denom;
        }
        dst[output_row * D + dim] = out;
    }
}

static inline int ggml_cuda_fattn_kvarn_decode_div_up_i64(const int64_t x, const int y) {
    return (int) ((x + y - 1) / y);
}

static inline int ggml_cuda_fattn_kvarn_decode_wave_efficiency_percent(
        const int64_t blocks_total,
        const int blocks_per_wave,
        int * n_waves) {
    if (blocks_total <= 0 || blocks_per_wave <= 0) {
        *n_waves = 0;
        return 0;
    }
    *n_waves = ggml_cuda_fattn_kvarn_decode_div_up_i64(blocks_total, blocks_per_wave);
    return (int) (100 * blocks_total / ((int64_t) *n_waves * blocks_per_wave));
}

template<int D, int MAX_GQA, int SPLIT_TOKENS, int NWARPS, int K_BITS, int V_BITS>
static int ggml_cuda_fattn_kvarn_decode_active_blocks_per_sm() {
    // Occupancy depends only on the compiled kernel and the device arch, not the workload,
    // so cache it per device: the selector queries several candidates on every decode op.
    static int cache[GGML_CUDA_MAX_DEVICES] = {};
    const int device = ggml_cuda_get_device();
    if (device >= 0 && device < GGML_CUDA_MAX_DEVICES && cache[device] > 0) {
        return cache[device];
    }
    const int wave_size = ggml_cuda_info().devices[device].warp_size;
    const dim3 block_dim(wave_size, NWARPS, 1);
    int max_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        ggml_cuda_fattn_kvarn_decode_mma_kernel<D, MAX_GQA, SPLIT_TOKENS, NWARPS, K_BITS, V_BITS>,
        block_dim.x * block_dim.y * block_dim.z,
        0));
    if (device >= 0 && device < GGML_CUDA_MAX_DEVICES) {
        cache[device] = max_blocks_per_sm;
    }
    return max_blocks_per_sm;
}

template<int D, int MAX_GQA, int SPLIT_TOKENS, int NWARPS, int K_BITS, int V_BITS>
static void ggml_cuda_fattn_kvarn_decode_consider(
        ggml_cuda_fattn_kvarn_decode_geometry & best,
        int64_t & best_score,
        const int nsm,
        const int n_kv,
        const int n_q,
        const int n_q_heads,
        const int n_kv_heads,
        const int n_stream) {
    const int gqa_ratio = n_q_heads / n_kv_heads;
    const int n_splits = ggml_cuda_fattn_kvarn_decode_div_up_i64(n_kv, SPLIT_TOKENS);
    const int n_gqa_blocks = ggml_cuda_fattn_kvarn_decode_div_up_i64(gqa_ratio, MAX_GQA);
    const int max_blocks_per_sm =
        ggml_cuda_fattn_kvarn_decode_active_blocks_per_sm<D, MAX_GQA, SPLIT_TOKENS, NWARPS, K_BITS, V_BITS>();
    if (max_blocks_per_sm <= 0 || n_splits <= 0 || n_gqa_blocks <= 0) {
        return;
    }

    const int blocks_per_wave = nsm * max_blocks_per_sm;
    const int64_t blocks_total =
        (int64_t) n_splits * n_kv_heads * n_gqa_blocks * n_q * n_stream;
    int n_waves = 0;
    const int wave_efficiency_percent =
        ggml_cuda_fattn_kvarn_decode_wave_efficiency_percent(blocks_total, blocks_per_wave, &n_waves);

    // Upstream queries actual occupancy, then scores wave efficiency inside one kernel family.
    // KVarN is choosing between different CTA shapes; when split/combine work is equal, the
    // occupancy result is the architectural signal, and wave efficiency is secondary.
    const int64_t score =
        (int64_t) max_blocks_per_sm * 1000000 +
        (int64_t) wave_efficiency_percent * 10000 -
        (int64_t) n_gqa_blocks * 1000 -
        (int64_t) n_splits;

    if (score > best_score) {
        best_score = score;
        best.use_split = true;
        best.split_tokens = SPLIT_TOKENS;
        best.nwarps = NWARPS;
        best.gqa_per_block = MAX_GQA;
        best.n_splits = n_splits;
        best.n_gqa_blocks = n_gqa_blocks;
        best.max_blocks_per_sm = max_blocks_per_sm;
        best.wave_efficiency_percent = wave_efficiency_percent;
        best.n_waves = n_waves;
    }
}

template<int D, int K_BITS, int V_BITS>
ggml_cuda_fattn_kvarn_decode_geometry ggml_cuda_fattn_kvarn_decode_select(
        int device,
        int n_kv,
        int n_q,
        int n_q_heads,
        int n_kv_heads,
        int n_stream) {
    ggml_cuda_fattn_kvarn_decode_geometry best = {};
    if (n_kv <= 0 || n_q <= 0 || n_q_heads <= 0 || n_kv_heads <= 0 ||
            n_stream <= 0 || n_q_heads % n_kv_heads != 0) {
        return best;
    }

    const int nsm = ggml_cuda_info().devices[device].nsm;
    int64_t best_score = INT64_MIN;
    if constexpr (D == 512) {
        ggml_cuda_fattn_kvarn_decode_consider<D, 6, 64,  8, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q, n_q_heads, n_kv_heads, n_stream);
        ggml_cuda_fattn_kvarn_decode_consider<D, 8, 64,  8, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q, n_q_heads, n_kv_heads, n_stream);
        ggml_cuda_fattn_kvarn_decode_consider<D, 6, 64, 16, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q, n_q_heads, n_kv_heads, n_stream);
        ggml_cuda_fattn_kvarn_decode_consider<D, 8, 64, 16, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q, n_q_heads, n_kv_heads, n_stream);
    } else if constexpr (D == 256) {
        ggml_cuda_fattn_kvarn_decode_consider<D, 6, 64, 8, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q, n_q_heads, n_kv_heads, n_stream);
        ggml_cuda_fattn_kvarn_decode_consider<D, 8, 64, 8, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q, n_q_heads, n_kv_heads, n_stream);
    } else if constexpr (D == 128) {
        ggml_cuda_fattn_kvarn_decode_consider<D, 6, 64, 4, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q, n_q_heads, n_kv_heads, n_stream);
        ggml_cuda_fattn_kvarn_decode_consider<D, 8, 64, 4, K_BITS, V_BITS>(
            best, best_score, nsm, n_kv, n_q, n_q_heads, n_kv_heads, n_stream);
    }

    if (!best.use_split || best.n_splits <= 1) {
        best.use_split = false;
        return best;
    }

    const int direct_blocks_per_wave = nsm * best.max_blocks_per_sm;
    const int64_t direct_blocks =
        (int64_t) n_kv_heads * best.n_gqa_blocks * n_q * n_stream;
    int direct_waves = 0;
    const int direct_efficiency_percent =
        ggml_cuda_fattn_kvarn_decode_wave_efficiency_percent(
            direct_blocks, direct_blocks_per_wave, &direct_waves);

    // Mirror upstream's "don't add parallel work when the base grid is already efficient"
    // idea. Once the unsplit decode would occupy at least two efficient waves, the split
    // combine pass is usually worse than falling through to the generic native MMA path.
    if (direct_waves >= 2 && direct_efficiency_percent >= 75) {
        best.use_split = false;
    }
    return best;
}

template<int D, int MAX_GQA, int SPLIT_TOKENS, int NWARPS, int K_BITS, int V_BITS>
static void ggml_cuda_fattn_kvarn_decode_launch_geometry(
        const ggml_cuda_fattn_kvarn_decode_args & args,
        const dim3 blocks_split) {
    ggml_cuda_fattn_kvarn_decode_mma_kernel<D, MAX_GQA, SPLIT_TOKENS, NWARPS, K_BITS, V_BITS>
        <<<blocks_split, dim3(args.wave_size, NWARPS, 1), 0, args.stream>>>(
            args.Q, args.k_descs, args.v_descs, args.mask, args.partial, args.partial_meta,
            args.scale, args.logit_softcap, args.nb01, args.nb02, args.nb03,
            args.nb30, args.nb31, args.nb33, args.ne33, args.n_kv, args.n_q,
            args.n_q_heads, args.n_kv_heads, args.gqa_ratio, args.n_gqa_blocks, args.n_splits);
}

template<int D, int MAX_GQA, int K_BITS, int V_BITS>
static void ggml_cuda_fattn_kvarn_decode_launch_gqa(
        const ggml_cuda_fattn_kvarn_decode_args & args,
        const dim3 blocks_split) {
    if constexpr (D == 512) {
        if (args.split_tokens == 64 && args.nwarps == 8) {
            ggml_cuda_fattn_kvarn_decode_launch_geometry<D, MAX_GQA, 64, 8, K_BITS, V_BITS>(args, blocks_split);
            return;
        }
        if (args.split_tokens == 64 && args.nwarps == 16) {
            ggml_cuda_fattn_kvarn_decode_launch_geometry<D, MAX_GQA, 64, 16, K_BITS, V_BITS>(args, blocks_split);
            return;
        }
    } else if constexpr (D == 256) {
        if (args.split_tokens == 64 && args.nwarps == 8) {
            ggml_cuda_fattn_kvarn_decode_launch_geometry<D, MAX_GQA, 64, 8, K_BITS, V_BITS>(args, blocks_split);
            return;
        }
    } else if constexpr (D == 128) {
        if (args.split_tokens == 64 && args.nwarps == 4) {
            ggml_cuda_fattn_kvarn_decode_launch_geometry<D, MAX_GQA, 64, 4, K_BITS, V_BITS>(args, blocks_split);
            return;
        }
    }
    GGML_ABORT("unsupported KVarN decode geometry D=%d split=%d nwarps=%d max_gqa=%d",
        D, args.split_tokens, args.nwarps, MAX_GQA);
}

template<int D, int K_BITS, int V_BITS>
void ggml_cuda_fattn_kvarn_decode_launch(const ggml_cuda_fattn_kvarn_decode_args & args) {
    const dim3 blocks_split(
        (uint32_t) args.n_splits,
        (uint32_t) (args.n_kv_heads * args.n_gqa_blocks * args.n_q),
        (uint32_t) args.n_stream);

    if (args.gqa_per_block == 6) {
        ggml_cuda_fattn_kvarn_decode_launch_gqa<D, 6, K_BITS, V_BITS>(args, blocks_split);
    } else if (args.gqa_per_block == 8) {
        ggml_cuda_fattn_kvarn_decode_launch_gqa<D, 8, K_BITS, V_BITS>(args, blocks_split);
    } else {
        GGML_ABORT("unsupported KVarN decode GQA block size %d", args.gqa_per_block);
    }
    CUDA_CHECK(cudaGetLastError());

    const dim3 blocks_combine((uint32_t) args.n_q_heads, (uint32_t) args.n_q, (uint32_t) args.n_stream);
    ggml_cuda_fattn_kvarn_decode_combine_kernel<D>
        <<<blocks_combine, GGML_CUDA_FATTN_KVARN_DECODE_THREADS, args.n_splits * sizeof(float), args.stream>>>(
            args.partial, args.partial_meta, args.dst, args.dst_meta,
            args.n_splits, args.n_q, args.n_q_heads);
    CUDA_CHECK(cudaGetLastError());
}
