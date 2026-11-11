#pragma once

#include <cstdint>

// Compare concrete kernels in units of warp-level tile passes, not locally
// calibrated microseconds. Occupancy supplies the device-dependent wave count.
// This is a structural heuristic, not a cross-device latency prediction.
inline int64_t ggml_cuda_kvarn_geometry_work(
        int waves, int split_tokens, int head_dim, int nwarps, int /*gqa_blocks*/) {
    const int slices = head_dim == 64 ? 1 : head_dim / 128;
    const int chunks = split_tokens / 16;
    const int k_passes = (chunks * slices + nwarps - 1) / nwarps;
    const int v_passes = ((head_dim + 16 * nwarps - 1) / (16 * nwarps)) * chunks;
    // Query/axis preparation is repeated for every CTA, independent of split
    // length. Count one pass for each of K and V, alongside their tile loops.
    return int64_t(waves) * (k_passes + v_passes + 2);
}

// Upstream stops adding parallel work once wave efficiency is at least 95%.
// Apply that same tolerance to the structural work estimate: near-equivalent
// geometries favor less partial-output/combine traffic, rather than oscillating
// with the rounded last wave. A substantial work difference still wins.
inline bool ggml_cuda_kvarn_geometry_better(
        int64_t work, int64_t best_work, int splits, int best_splits,
        int efficiency, int best_efficiency,
        bool populated = true, bool best_populated = true) {
    // As in the unsplit-grid guard, first retain at least two resident waves
    // of independent work when a candidate can provide them. Fewer CTAs do
    // not help if they remove the parallelism needed to hide memory latency.
    if (populated != best_populated) {
        return populated;
    }
    if (work * 100 < best_work * 95) {
        return true;
    }
    if (best_work * 100 < work * 95) {
        return false;
    }
    if (splits != best_splits) {
        return splits < best_splits;
    }
    return work < best_work || (work == best_work && efficiency > best_efficiency);
}

// Launch bounds are a compiler resource policy, not measured occupancy. Keep
// the qualified sm_86 register target there; elsewhere use the upstream MMA
// configuration's architecture-specific target instead of assuming 1536 threads.
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
constexpr int ggml_cuda_kvarn_min_blocks(int cc, int nthreads, int upstream_target) {
    return cc == 860 ? (1536 / nthreads < 4 ? 1536 / nthreads : 4) : upstream_target;
}
