#include "gated_delta_net.cuh"

// RDNA3 wants 1 block/SM here: with 2 blocks/SM the compiler is forced under
// ~128 VGPR/lane and spills on this kernel. 1 block/SM raises the per-lane
// VGPR budget to ~256 and eliminates the spill entirely. CUDA keeps 2.
// See llama.cpp issue #20354 (GATED_DELTA_NET HIP underperforms on RDNA3).
// exp2 is a single SFU instruction (ex2.approx) vs expf's multiply + ex2 + range reduction
#define GDN_EXPF(x) exp2f((x) * 1.442695041f)

#if defined(GGML_USE_HIP)
#define GGML_GDN_MIN_BLOCKS_PER_SM 1
#else
#define GGML_GDN_MIN_BLOCKS_PER_SM 2
#endif

template <int S_v, bool KDA>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, GGML_GDN_MIN_BLOCKS_PER_SM)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_offset;
    curr_state += state_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = GDN_EXPF(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += GDN_EXPF(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = GDN_EXPF(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;
    }

    // Write state back to global memory (transposed layout)
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i          = r * warp_size + lane;
        state[col * S_v + i] = s_shard[r];
    }
}

template <bool KDA>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    switch (S_v) {
        case 16:
            gated_delta_net_cuda<16, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 32:
            gated_delta_net_cuda<32, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 64: {
            gated_delta_net_cuda<64, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        case 128: {
            gated_delta_net_cuda<128, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    if (kda) {
        launch_gated_delta_net<true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    } else {
        launch_gated_delta_net<false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    }
}

template <int S_v>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, GGML_GDN_MIN_BLOCKS_PER_SM)
dflash_gdn_state_replay_cuda(
        float * __restrict__ state,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const int n_tokens,
        const int H_k,
        const int H_v) {
    const int h_idx = blockIdx.x;
    const int lane  = threadIdx.x;
    const int col   = blockIdx.z * blockDim.y + threadIdx.y;

    if (h_idx >= H_v || col >= S_v) {
        return;
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;

    float * state_h = state + (size_t) h_idx * S_v * S_v + (size_t) col * S_v;
    float s_shard[rows_per_lane];

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        const int i = r * warp_size + lane;
        s_shard[r] = state_h[i];
    }

    const int hk = h_idx % H_k;
    for (int t = 0; t < n_tokens; ++t) {
        const float * k_t = k + ((size_t) t * H_k + hk) * S_v;
        const float * v_t = v + ((size_t) t * H_v + h_idx) * S_v;
        const float g_val = GDN_EXPF(g[(size_t) t * H_v + h_idx]);
        const float beta_val = 1.0f / (1.0f + expf(-beta[(size_t) t * H_v + h_idx]));

        float k_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
        }

        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        const float kv_col = warp_reduce_sum<warp_size>(kv_shard);
        const float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
        }
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        const int i = r * warp_size + lane;
        state_h[i] = s_shard[r];
    }
}

extern "C" bool dflash_replay_gdn_state_no_check(
        void * state,
        const void * k,
        const void * v,
        const void * g,
        const void * beta,
        int n_tokens,
        int S_v,
        int H_k,
        int H_v) {
    if (!state || !k || !v || !g || !beta) return false;
    if (n_tokens <= 0 || H_k <= 0 || H_v <= 0) return false;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    const dim3 grid_dims(H_v, 1, (S_v + num_warps - 1) / num_warps);
    const dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    switch (S_v) {
        case 16:
            dflash_gdn_state_replay_cuda<16><<<grid_dims, block_dims, 0, cudaStreamPerThread>>>(
                    (float *) state, (const float *) k, (const float *) v,
                    (const float *) g, (const float *) beta, n_tokens, H_k, H_v);
            break;
        case 32:
            dflash_gdn_state_replay_cuda<32><<<grid_dims, block_dims, 0, cudaStreamPerThread>>>(
                    (float *) state, (const float *) k, (const float *) v,
                    (const float *) g, (const float *) beta, n_tokens, H_k, H_v);
            break;
        case 64:
            dflash_gdn_state_replay_cuda<64><<<grid_dims, block_dims, 0, cudaStreamPerThread>>>(
                    (float *) state, (const float *) k, (const float *) v,
                    (const float *) g, (const float *) beta, n_tokens, H_k, H_v);
            break;
        case 128:
            dflash_gdn_state_replay_cuda<128><<<grid_dims, block_dims, 0, cudaStreamPerThread>>>(
                    (float *) state, (const float *) k, (const float *) v,
                    (const float *) g, (const float *) beta, n_tokens, H_k, H_v);
            break;
        default:
            return false;
    }

    return cudaGetLastError() == cudaSuccess;
}

// ============================================================================
// Tree-mode Gated Delta Net
// ============================================================================

#define GDN_TREE_ROOT_PARENT (-1)

template <int S_v, bool KDA>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, GGML_GDN_MIN_BLOCKS_PER_SM)
gated_delta_net_tree_cuda(const float * q,
                          const float * k,
                          const float * v,
                          const float * g,
                          const float * beta,
                          const float * curr_state,
                          const int32_t * parent_ids,
                          half *        persist_inter,
                          float *       dst,
                          int64_t       H,
                          int64_t       n_tokens,
                          int64_t       n_seqs,
                          int64_t       sq1,
                          int64_t       sq2,
                          int64_t       sq3,
                          int64_t       sv1,
                          int64_t       sv2,
                          int64_t       sv3,
                          int64_t       sb1,
                          int64_t       sb2,
                          int64_t       sb3,
                          const uint3   neqk1_magic,
                          const uint3   rq3_magic,
                          float         scale) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_offset;
    curr_state += state_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    // parent_ids layout: [n_tokens] per sequence (flat, not per-seq for now)
    const int32_t * parent_ids_seq = parent_ids;

    // persist_inter layout: [S_v, S_v, H, n_tokens, n_seqs]
    // For token t, head h: offset = ((seq * n_tokens + t) * H + h) * S_v * S_v
    const int64_t inter_stride_token = H * S_v * S_v;
    half * inter_seq_base = persist_inter + sequence * n_tokens * inter_stride_token;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];

    // Load initial state
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        // Tree branching: reload state from parent's intermediate if needed
        if (t > 0) {
            const int parent_t = parent_ids_seq[t];
            if (parent_t == GDN_TREE_ROOT_PARENT) {
                // Reload initial state (root token)
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    s_shard[r] = curr_state[i];
                }
            } else if (parent_t != t - 1) {
                // Branch transition: load from parent's intermediate (f16 → f32)
                const half * parent_inter = inter_seq_base + parent_t * inter_stride_token + h_idx * S_v * S_v;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    s_shard[r] = __half2float(parent_inter[col * S_v + i]);
                }
            }
            // else parent_t == t-1: sequential, state in registers is correct
        }

        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = GDN_EXPF(*g_t);

            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += GDN_EXPF(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            float delta_col = (v_t[col] - kv_col) * beta_val;

            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = GDN_EXPF(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        // Store intermediate state (f32 → f16) for this token
        {
            half * inter_t = inter_seq_base + t * inter_stride_token + h_idx * S_v * S_v;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                inter_t[col * S_v + i] = __float2half(s_shard[r]);
            }
        }

        attn_data += S_v * H;
    }

    // Write final state (state after last token in DFS order)
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i          = r * warp_size + lane;
        state[col * S_v + i] = s_shard[r];
    }
}

template <bool KDA>
static void launch_gated_delta_net_tree(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        const int32_t * parent_ids_d, half * persist_inter_d,
        float * dst_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    switch (S_v) {
        case 16:
            gated_delta_net_tree_cuda<16, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, parent_ids_d, persist_inter_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 32:
            gated_delta_net_tree_cuda<32, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, parent_ids_d, persist_inter_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 64:
            gated_delta_net_tree_cuda<64, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, parent_ids_d, persist_inter_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 128:
            gated_delta_net_tree_cuda<128, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, parent_ids_d, persist_inter_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_op_gated_delta_net_tree(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q       = dst->src[0];
    ggml_tensor * src_k       = dst->src[1];
    ggml_tensor * src_v       = dst->src[2];
    ggml_tensor * src_g       = dst->src[3];
    ggml_tensor * src_beta    = dst->src[4];
    ggml_tensor * src_state   = dst->src[5];
    ggml_tensor * src_parents = dst->src[6];
    ggml_tensor * src_inter   = dst->src[7];

    GGML_ASSERT(src_q       != nullptr);
    GGML_ASSERT(src_k       != nullptr);
    GGML_ASSERT(src_v       != nullptr);
    GGML_ASSERT(src_g       != nullptr);
    GGML_ASSERT(src_beta    != nullptr);
    GGML_ASSERT(src_state   != nullptr);
    GGML_ASSERT(src_parents != nullptr);
    GGML_ASSERT(src_inter   != nullptr);

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(src_q->type     == GGML_TYPE_F32);
    GGML_ASSERT(src_k->type     == GGML_TYPE_F32);
    GGML_ASSERT(src_v->type     == GGML_TYPE_F32);
    GGML_ASSERT(src_g->type     == GGML_TYPE_F32);
    GGML_ASSERT(src_beta->type  == GGML_TYPE_F32);
    GGML_ASSERT(src_state->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);
    GGML_ASSERT(n_tokens > 0);
    GGML_ASSERT(n_seqs > 0);
    GGML_ASSERT(neq1 == nek1);
    GGML_ASSERT(neq3 > 0);
    GGML_ASSERT(nev3 % neq3 == 0);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float *   s_d   = (const float *)   src_state->data;
    const int32_t * p_d   = (const int32_t *) src_parents->data;
    half *          i_d   = (half *)          src_inter->data;
    float *         dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));
    GGML_ASSERT(src_parents->type == GGML_TYPE_I32);
    GGML_ASSERT(src_inter->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_nelements(src_parents) >= n_tokens);
    GGML_ASSERT(ggml_nelements(src_state) == S_v * S_v * H * n_seqs);
    GGML_ASSERT(ggml_nelements(src_inter) >= S_v * S_v * H * n_tokens * n_seqs);
    GGML_ASSERT(dst->ne[0] == S_v * H);
    GGML_ASSERT(dst->ne[1] == n_tokens * n_seqs + S_v * n_seqs);

    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    if (kda) {
        launch_gated_delta_net_tree<true>(q_d, k_d, v_d, g_d, b_d, s_d, p_d, i_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    } else {
        launch_gated_delta_net_tree<false>(q_d, k_d, v_d, g_d, b_d, s_d, p_d, i_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    }

}
