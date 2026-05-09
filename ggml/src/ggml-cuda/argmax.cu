#include <algorithm>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

#if defined(__has_include)
#    if __has_include(<cub/cub.cuh>)
#        include <cub/cub.cuh>
#        if (CCCL_MAJOR_VERSION > 3) || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 2)
#            define GGML_CUDA_DFLASH_CUB_TOP_K_AVAILABLE
#        endif
#    endif
#endif

// philox-style counter-based PRNG: fast, stateless, deterministic per (seed, counter)
static __device__ __forceinline__ uint32_t philox_hash(uint64_t seed, uint64_t counter) {
    uint32_t lo = (uint32_t)(counter);
    uint32_t hi = (uint32_t)(counter >> 32);
    uint32_t k0 = (uint32_t)(seed);
    uint32_t k1 = (uint32_t)(seed >> 32);
    // 4 rounds of philox-like mixing
    for (int i = 0; i < 4; i++) {
        uint64_t prod = (uint64_t)lo * 0xD2511F53u;
        lo = (uint32_t)(prod >> 32) ^ hi ^ k0;
        hi = (uint32_t)(prod);
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return lo;
}

static __device__ __forceinline__ float gumbel_noise(uint64_t seed, int64_t row, int64_t col, int64_t ncols) {
    uint32_t h = philox_hash(seed, row * ncols + col);
    // uniform in (0, 1) — avoid exact 0 and 1
    float u = ((h >> 8) + 0.5f) * (1.0f / 16777216.0f);
    return -logf(-logf(u));
}

// Argmax kernel with optional Gumbel sampling and log-probability output.
// When output_logprob=true, also computes log_prob = log(softmax(logits/temp)[argmax_token])
//   = logits[argmax]/temp - logsumexp(logits/temp)
// using online softmax (single pass over data).
static __global__ void argmax_f32(
        const float * __restrict__ x,
        int32_t * __restrict__ dst,
        const int64_t ncols,
        const int64_t nrows,
        const float inv_temp,
        const uint64_t seed,
        const bool output_logprob) {

    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    // For log-prob: track the max scaled logit and sum of exp(scaled_logit - max)
    float logit_max = -FLT_MAX;  // max of logits[col] * inv_temp (without gumbel)
    float sum_exp = 0.0f;        // running sum for online softmax

    const float * rowx = x + row * ncols;
    const bool use_gumbel = (seed != 0);

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        float scaled = rowx[col] * inv_temp;

        // Online softmax accumulation (always on scaled logits, no gumbel)
        if (output_logprob) {
            if (scaled > logit_max) {
                sum_exp = sum_exp * expf(logit_max - scaled) + 1.0f;
                logit_max = scaled;
            } else {
                sum_exp += expf(scaled - logit_max);
            }
        }

        // Argmax (with optional gumbel perturbation)
        float val = scaled;
        if (use_gumbel) {
            val += gumbel_noise(seed, row, col, ncols);
        }
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    // Warp reduction for argmax
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFFULL, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFFULL, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    // Warp reduction for online softmax (merge logit_max and sum_exp)
    if (output_logprob) {
#pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            float other_max = __shfl_xor_sync(0xFFFFFFFFULL, logit_max, offset, WARP_SIZE);
            float other_sum = __shfl_xor_sync(0xFFFFFFFFULL, sum_exp, offset, WARP_SIZE);
            if (other_max > logit_max) {
                sum_exp = sum_exp * expf(logit_max - other_max) + other_sum;
                logit_max = other_max;
            } else {
                sum_exp = sum_exp + other_sum * expf(other_max - logit_max);
            }
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        __shared__ float shared_logit_max[max_warps];
        __shared__ float shared_sum_exp[max_warps];

        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
            if (output_logprob) {
                shared_logit_max[warp_id] = logit_max;
                shared_sum_exp[warp_id] = sum_exp;
            }
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
                if (output_logprob) {
                    logit_max = shared_logit_max[lane_id];
                    sum_exp = shared_sum_exp[lane_id];
                }
            } else {
                maxval = -FLT_MAX;
                argmax = -1;
                logit_max = -FLT_MAX;
                sum_exp = 0.0f;
            }

#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                float val = __shfl_xor_sync(0xFFFFFFFFULL, maxval, offset, WARP_SIZE);
                int   col = __shfl_xor_sync(0xFFFFFFFFULL, argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }

                if (output_logprob) {
                    float other_max = __shfl_xor_sync(0xFFFFFFFFULL, logit_max, offset, WARP_SIZE);
                    float other_sum = __shfl_xor_sync(0xFFFFFFFFULL, sum_exp, offset, WARP_SIZE);
                    if (other_max > logit_max) {
                        sum_exp = sum_exp * expf(logit_max - other_max) + other_sum;
                        logit_max = other_max;
                    } else {
                        sum_exp = sum_exp + other_sum * expf(other_max - logit_max);
                    }
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;

        if (output_logprob) {
            // log_prob = logits[argmax] * inv_temp - (logit_max + log(sum_exp))
            float log_prob = rowx[argmax] * inv_temp - logit_max - logf(sum_exp);
            int32_t prob_bits;
            memcpy(&prob_bits, &log_prob, sizeof(float));
            dst[nrows + row] = prob_bits;
        } else {
            int32_t logit_bits;
            const float raw_logit = rowx[argmax];
            memcpy(&logit_bits, &raw_logit, sizeof(float));
            dst[nrows + row] = logit_bits;
        }
    }
}

// Top-K kernel: each block handles one row, outputs K best tokens + log-probs.
// Uses a register-based min-heap per thread, then merges via shared memory.
// K is a runtime parameter but must be <= 32.
static __global__ void topk_f32(
        const float * __restrict__ x,
        int32_t * __restrict__ dst,
        const int64_t ncols,
        const int64_t nrows,
        const int K,
        const float inv_temp,
        const uint64_t seed,
        const bool output_logprob) {

    const int64_t row = blockIdx.x;
    const float * rowx = x + row * ncols;
    const bool use_gumbel = (seed != 0);

    // Per-thread top-K heap (min-heap: smallest score at index 0)
    // Max K=32, stored in registers
    float  heap_val[32];
    int32_t heap_idx[32];
    for (int i = 0; i < K; i++) {
        heap_val[i] = -FLT_MAX;
        heap_idx[i] = -1;
    }

    // Online softmax accumulators
    float logit_max = -FLT_MAX;
    float sum_exp = 0.0f;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        float scaled = rowx[col] * inv_temp;

        if (output_logprob) {
            if (scaled > logit_max) {
                sum_exp = sum_exp * expf(logit_max - scaled) + 1.0f;
                logit_max = scaled;
            } else {
                sum_exp += expf(scaled - logit_max);
            }
        }

        float val = scaled;
        if (use_gumbel) {
            val += gumbel_noise(seed, row, col, ncols);
        }

        // Insert into min-heap if larger than min
        if (val > heap_val[0]) {
            heap_val[0] = val;
            heap_idx[0] = col;
            // sift down
            int pos = 0;
            while (true) {
                int left = 2 * pos + 1;
                int right = 2 * pos + 2;
                int smallest = pos;
                if (left < K && heap_val[left] < heap_val[smallest]) smallest = left;
                if (right < K && heap_val[right] < heap_val[smallest]) smallest = right;
                if (smallest == pos) break;
                float tv = heap_val[pos]; heap_val[pos] = heap_val[smallest]; heap_val[smallest] = tv;
                int32_t ti = heap_idx[pos]; heap_idx[pos] = heap_idx[smallest]; heap_idx[smallest] = ti;
                pos = smallest;
            }
        }
    }

    // Cross-thread merge via shared memory
    // Strategy: iterative pairwise merge within warp, then across warps
    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    // Shared memory for cross-warp merge: need K * n_warps entries
    extern __shared__ char smem[];
    float  * s_val = (float  *)smem;
    int32_t * s_idx = (int32_t *)(smem + K * n_warps * sizeof(float));
    float  * s_logit_max = (float *)(smem + 2 * K * n_warps * sizeof(float));
    float  * s_sum_exp   = s_logit_max + n_warps;

    // Intra-warp merge: pair-reduce within warp using shuffle
    // Each step: lane gets partner's min element, if it beats our min, replace and re-heapify
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        for (int i = 0; i < K; i++) {
            float partner_val = __shfl_xor_sync(0xFFFFFFFFULL, heap_val[i], offset);
            int partner_idx = __shfl_xor_sync(0xFFFFFFFFULL, heap_idx[i], offset);
            if (partner_val > heap_val[0]) {
                heap_val[0] = partner_val;
                heap_idx[0] = partner_idx;
                // sift down
                int pos = 0;
                while (true) {
                    int left = 2 * pos + 1;
                    int right = 2 * pos + 2;
                    int smallest = pos;
                    if (left < K && heap_val[left] < heap_val[smallest]) smallest = left;
                    if (right < K && heap_val[right] < heap_val[smallest]) smallest = right;
                    if (smallest == pos) break;
                    float tv = heap_val[pos]; heap_val[pos] = heap_val[smallest]; heap_val[smallest] = tv;
                    int32_t ti = heap_idx[pos]; heap_idx[pos] = heap_idx[smallest]; heap_idx[smallest] = ti;
                    pos = smallest;
                }
            }
        }
    }

    // Warp 0 has best K for intra-warp; for multi-warp, merge across warps
    if (n_warps > 1) {
        // Reduce softmax within each warp first
        if (output_logprob) {
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                float other_max = __shfl_xor_sync(0xFFFFFFFFULL, logit_max, offset, WARP_SIZE);
                float other_sum = __shfl_xor_sync(0xFFFFFFFFULL, sum_exp, offset, WARP_SIZE);
                if (other_max > logit_max) {
                    sum_exp = sum_exp * expf(logit_max - other_max) + other_sum;
                    logit_max = other_max;
                } else {
                    sum_exp = sum_exp + other_sum * expf(other_max - logit_max);
                }
            }
        }

        // Each warp leader writes its K candidates to shared memory
        if (lane_id == 0) {
            for (int i = 0; i < K; i++) {
                s_val[warp_id * K + i] = heap_val[i];
                s_idx[warp_id * K + i] = heap_idx[i];
            }
            if (output_logprob) {
                s_logit_max[warp_id] = logit_max;
                s_sum_exp[warp_id] = sum_exp;
            }
        }
        __syncthreads();

        // Warp 0, lane 0 merges all warps' candidates
        if (warp_id == 0 && lane_id == 0) {
            // Start with warp 0's heap (already in registers)
            for (int w = 1; w < n_warps; w++) {
                for (int i = 0; i < K; i++) {
                    float cand_val = s_val[w * K + i];
                    int32_t cand_idx = s_idx[w * K + i];
                    if (cand_val > heap_val[0]) {
                        heap_val[0] = cand_val;
                        heap_idx[0] = cand_idx;
                        int pos = 0;
                        while (true) {
                            int left = 2 * pos + 1;
                            int right = 2 * pos + 2;
                            int smallest = pos;
                            if (left < K && heap_val[left] < heap_val[smallest]) smallest = left;
                            if (right < K && heap_val[right] < heap_val[smallest]) smallest = right;
                            if (smallest == pos) break;
                            float tv = heap_val[pos]; heap_val[pos] = heap_val[smallest]; heap_val[smallest] = tv;
                            int32_t ti = heap_idx[pos]; heap_idx[pos] = heap_idx[smallest]; heap_idx[smallest] = ti;
                            pos = smallest;
                        }
                    }
                }
            }

            // Merge online softmax across warps
            if (output_logprob) {
                for (int w = 1; w < n_warps; w++) {
                    float other_max = s_logit_max[w];
                    float other_sum = s_sum_exp[w];
                    if (other_max > logit_max) {
                        sum_exp = sum_exp * expf(logit_max - other_max) + other_sum;
                        logit_max = other_max;
                    } else {
                        sum_exp = sum_exp + other_sum * expf(other_max - logit_max);
                    }
                }
            }
        }
    } else {
        // Single warp: reduce softmax within warp
        if (output_logprob) {
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                float other_max = __shfl_xor_sync(0xFFFFFFFFULL, logit_max, offset, WARP_SIZE);
                float other_sum = __shfl_xor_sync(0xFFFFFFFFULL, sum_exp, offset, WARP_SIZE);
                if (other_max > logit_max) {
                    sum_exp = sum_exp * expf(logit_max - other_max) + other_sum;
                    logit_max = other_max;
                } else {
                    sum_exp = sum_exp + other_sum * expf(other_max - logit_max);
                }
            }
        }
    }

    // Output: sort heap descending and write
    if ((n_warps == 1 && lane_id == 0) || (n_warps > 1 && warp_id == 0 && lane_id == 0)) {
        // Sort K elements descending by val (insertion sort, K is small)
        for (int i = 1; i < K; i++) {
            float kv = heap_val[i];
            int32_t ki = heap_idx[i];
            int j = i - 1;
            while (j >= 0 && heap_val[j] < kv) {
                heap_val[j + 1] = heap_val[j];
                heap_idx[j + 1] = heap_idx[j];
                j--;
            }
            heap_val[j + 1] = kv;
            heap_idx[j + 1] = ki;
        }

        // Write token IDs: dst[row * K + 0..K-1]
        for (int i = 0; i < K; i++) {
            dst[row * K + i] = heap_idx[i];
        }

        // Write log-probs: dst[K*nrows + row * K + 0..K-1]
        if (output_logprob) {
            float log_norm = logit_max + logf(sum_exp);
            for (int i = 0; i < K; i++) {
                float log_prob = (heap_idx[i] >= 0) ? (rowx[heap_idx[i]] * inv_temp - log_norm) : -FLT_MAX;
                int32_t prob_bits;
                memcpy(&prob_bits, &log_prob, sizeof(float));
                dst[K * nrows + row * K + i] = prob_bits;
            }
        } else {
            for (int i = 0; i < K; i++) {
                int32_t logit_bits;
                const float raw_logit = heap_idx[i] >= 0 ? rowx[heap_idx[i]] : -FLT_MAX;
                memcpy(&logit_bits, &raw_logit, sizeof(float));
                dst[K * nrows + row * K + i] = logit_bits;
            }
        }
    }
}

#ifdef GGML_CUDA_DFLASH_CUB_TOP_K_AVAILABLE
static void topk_raw_cub(
        ggml_cuda_pool & pool,
        const float    * src,
        int32_t        * dst,
        const int64_t    ncols,
        const int64_t    nrows,
        const int        K,
        cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };
    auto indexes_in   = cuda::make_counting_iterator(0);

    float * vals_out = (float *) (dst + (size_t) K * (size_t) nrows);

    for (int64_t row = 0; row < nrows; ++row) {
        const float * row_src  = src + row * ncols;
        int32_t     * row_ids  = dst + row * K;
        float       * row_vals = vals_out + row * K;

        size_t temp_storage_bytes = 0;
        CUDA_CHECK(cub::DeviceTopK::MaxPairs(nullptr, temp_storage_bytes,
                    row_src, row_vals, indexes_in, row_ids, (int) ncols, K, env));

        ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
        void * d_temp_storage = temp_storage_alloc.get();

        CUDA_CHECK(cub::DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes,
                    row_src, row_vals, indexes_in, row_ids, (int) ncols, K, env));
    }
}
#endif

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    // read temperature and seed from op_params
    float temp = 0.0f;
    uint64_t seed = 0;
    memcpy(&temp, &dst->op_params[0], sizeof(float));
    uint32_t seed_lo = 0, seed_hi = 0;
    memcpy(&seed_lo, &dst->op_params[1], sizeof(uint32_t));
    memcpy(&seed_hi, &dst->op_params[2], sizeof(uint32_t));
    seed = ((uint64_t)seed_hi << 32) | seed_lo;

    // read K from op_params[3] (0 means K=1 for backward compat)
    int32_t K = 0;
    memcpy(&K, &dst->op_params[3], sizeof(int32_t));
    if (K <= 0) K = 1;

    const float inv_temp = (temp > 0.0f) ? (1.0f / temp) : 1.0f;
    // Seeded draft sampling needs full-vocab log-probs for acceptance tests.
    // Deterministic top-K verifier reduction only needs raw logits; adding or
    // subtracting a per-row constant does not affect the downstream sampler chain.
    const bool output_logprob = temp > 0.0f && seed != 0;

    ggml_cuda_pool & pool = ctx.pool();

#ifdef GGML_CUDA_DFLASH_CUB_TOP_K_AVAILABLE
    if (K > 1 && !output_logprob && seed == 0) {
        topk_raw_cub(pool, src0_d, dst_d, ne00, nrows, K, stream);
        return;
    }
#endif

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    if (K == 1) {
        argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00, nrows, inv_temp, seed, output_logprob);
    } else {
        // Shared memory: K * n_warps floats + K * n_warps ints + 2 * n_warps floats (softmax)
        const int n_warps = (int)(num_threads / WARP_SIZE);
        const size_t smem_size = K * n_warps * (sizeof(float) + sizeof(int32_t)) + 2 * n_warps * sizeof(float);
        topk_f32<<<blocks_num, blocks_dim, smem_size, stream>>>(src0_d, dst_d, ne00, nrows, K, inv_temp, seed, output_logprob);
    }
}
