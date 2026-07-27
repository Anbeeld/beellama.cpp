#include "kvarn.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

static constexpr int KVAR_N_DIM = 128;
// Dynamic stage/tail depth is carried explicitly in op_params[7]/[8].
static constexpr int KVAR_N_TILE_VALUES = KVAR_N_DIM * KVAR_N_DIM;
static constexpr int KVAR_N_REDUCE_FLOATS = 4 * 4;
static constexpr int KVAR_N_SHARED_FLOATS = KVAR_N_TILE_VALUES + 8 * KVAR_N_DIM + 2 + KVAR_N_REDUCE_FLOATS;
static constexpr int KVAR_N_SHARED_BYTES = KVAR_N_SHARED_FLOATS * sizeof(float);
static constexpr int KVAR_N_LOWSHMEM_FLOATS = 6 * KVAR_N_DIM + 2 + KVAR_N_REDUCE_FLOATS;
static constexpr int KVAR_N_LOWSHMEM_BYTES = KVAR_N_LOWSHMEM_FLOATS * sizeof(float);
static constexpr int KVAR_N_STAGE_CHUNK = 4;
static constexpr int KVAR_N_OP_PARAM_BITS = 0;
static constexpr int KVAR_N_OP_PARAM_ITERS = 1;
static constexpr int KVAR_N_OP_PARAM_MAT_VALUE = 1;
static constexpr int KVAR_N_OP_PARAM_STORE_VALUE = 2;
static constexpr int KVAR_N_OP_PARAM_MAT_STREAM_START = 2;
static constexpr int KVAR_N_OP_PARAM_MAT_N_STREAM = 3;
static constexpr int KVAR_N_OP_PARAM_TOKENS_PER_STREAM = 3;
static constexpr int KVAR_N_OP_PARAM_MAT_EMIT_ROTATED = 4;
static constexpr int KVAR_N_OP_PARAM_STORE_SWA = 4;     // store: SWA sliding-window ring mode
static constexpr int KVAR_N_OP_PARAM_HEAD_SLICES = 5;   // 1/2/4 slices per logical K/V head
static constexpr int KVAR_N_OP_PARAM_MAT_SWA = 6;
static constexpr int KVAR_N_OP_PARAM_STAGE_GROUPS = 7;  // dynamic F16 stage depth
static constexpr int KVAR_N_OP_PARAM_TAIL_GROUPS = 8;   // lossless tail groups within the stage
static constexpr int KVAR_N_OP_PARAM_EAGER_RECORDS = 9; // materialize a record when its closing token is stored

static std::atomic<uint64_t> g_kvarn_store_headwide_workspace{0};
static std::atomic<uint64_t> g_kvarn_store_headwide_monolithic{0};
static std::atomic<uint64_t> g_kvarn_store_single_slice_workspace{0};
static std::atomic<uint64_t> g_kvarn_store_direct_store{0};
static std::atomic<uint64_t> g_kvarn_store_high_shared_fallback{0};
static std::atomic<uint64_t> g_kvarn_store_low_shared_store{0};

void ggml_cuda_kvarn_store_route_stats_reset() {
    g_kvarn_store_headwide_workspace.store(0, std::memory_order_relaxed);
    g_kvarn_store_headwide_monolithic.store(0, std::memory_order_relaxed);
    g_kvarn_store_single_slice_workspace.store(0, std::memory_order_relaxed);
    g_kvarn_store_direct_store.store(0, std::memory_order_relaxed);
    g_kvarn_store_high_shared_fallback.store(0, std::memory_order_relaxed);
    g_kvarn_store_low_shared_store.store(0, std::memory_order_relaxed);
}

void ggml_cuda_kvarn_store_route_stats_get(ggml_cuda_kvarn_store_route_stats * stats) {
    if (stats == nullptr ||
            stats->struct_size < sizeof(ggml_cuda_kvarn_store_route_stats) ||
            stats->abi_version != GGML_CUDA_KVARN_STORE_ROUTE_STATS_ABI_VERSION) {
        return;
    }
    stats->struct_size = sizeof(*stats);
    stats->abi_version = GGML_CUDA_KVARN_STORE_ROUTE_STATS_ABI_VERSION;
    stats->headwide_workspace = g_kvarn_store_headwide_workspace.load(std::memory_order_relaxed);
    stats->headwide_monolithic = g_kvarn_store_headwide_monolithic.load(std::memory_order_relaxed);
    stats->single_slice_workspace = g_kvarn_store_single_slice_workspace.load(std::memory_order_relaxed);
    stats->direct_store = g_kvarn_store_direct_store.load(std::memory_order_relaxed);
    stats->high_shared_fallback = g_kvarn_store_high_shared_fallback.load(std::memory_order_relaxed);
    stats->low_shared_store = g_kvarn_store_low_shared_store.load(std::memory_order_relaxed);
}

// Resolve stage_groups from op_params[7]. Constructors set this explicitly;
// backends assert it before deriving stream counts.
static int kvarn_resolve_stage_groups(const ggml_tensor * dst) {
    return ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_STAGE_GROUPS);
}

static int kvarn_resolve_tail_groups(const ggml_tensor * dst, int stage_groups) {
    const int tail_groups = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_TAIL_GROUPS);
    return tail_groups > 0 ? tail_groups : stage_groups - 1;
}
enum class kvarn_prof_kind : uint8_t {
    STORE_HI = 0,
    STORE_LOW,
    LIVE_GROUPS,
    COUNT,
};

static bool kvarn_profile_enabled() {
    const char * value = std::getenv("GGML_KVARN_PROFILE");
    return value != nullptr && std::atoi(value) != 0;
}

static int kvarn_profile_dump_every() {
    const char * value = std::getenv("GGML_KVARN_PROFILE_DUMP_EVERY");
    return value != nullptr ? std::max(0, std::atoi(value)) : 0;
}

static bool kvarn_profile_cuda_graphs_disabled() {
    return false;
}

struct kvarn_prof_event_pair {
    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;
    int device = -1;
};

struct kvarn_prof_sample {
    kvarn_prof_kind kind = kvarn_prof_kind::STORE_HI;
    uint8_t side = 0;
    uint8_t bits = 0;
    int n_kv = 0;
    size_t bytes_out = 0;
    kvarn_prof_event_pair events;
};

struct kvarn_prof_bucket {
    uint64_t count = 0;
    double total_ms = 0.0;
    double max_ms = 0.0;
    size_t total_bytes = 0;
    int n_kv_min = INT_MAX;
    int n_kv_max = 0;
    std::array<uint64_t, 5> n_kv_buckets = {};
};

struct kvarn_prof_state {
    static constexpr int max_events_per_device = 8192;
    static constexpr int event_batch = 256;
    static constexpr size_t flush_threshold = 4096;

    std::mutex mutex;
    std::array<std::vector<kvarn_prof_event_pair>, GGML_CUDA_MAX_DEVICES> free_events;
    std::array<int, GGML_CUDA_MAX_DEVICES> created_events = {};
    std::vector<kvarn_prof_sample> pending;
    kvarn_prof_bucket aggregates[(int) kvarn_prof_kind::COUNT][2];
    uint64_t flush_count = 0;
    bool final_dump_done = false;

    bool make_event_pair_locked(int device, kvarn_prof_event_pair & pair) {
        if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
            return false;
        }

        if (free_events[device].empty() && created_events[device] < max_events_per_device) {
            ggml_cuda_set_device(device);
            const int remaining = max_events_per_device - created_events[device];
            const int n_create = std::min(event_batch, remaining);
            for (int i = 0; i < n_create; ++i) {
                kvarn_prof_event_pair cur;
                cur.device = device;
                if (cudaEventCreateWithFlags(&cur.ev0, cudaEventDefault) != cudaSuccess) {
                    break;
                }
                if (cudaEventCreateWithFlags(&cur.ev1, cudaEventDefault) != cudaSuccess) {
                    (void) cudaEventDestroy(cur.ev0);
                    break;
                }
                free_events[device].push_back(cur);
                ++created_events[device];
            }
        }

        if (free_events[device].empty()) {
            return false;
        }

        pair = free_events[device].back();
        free_events[device].pop_back();
        return true;
    }

    void release_event_pair_locked(kvarn_prof_event_pair pair) {
        if (pair.device >= 0 && pair.device < GGML_CUDA_MAX_DEVICES && pair.ev0 != nullptr && pair.ev1 != nullptr) {
            free_events[pair.device].push_back(pair);
        }
    }

    static int n_kv_bucket(int n_kv) {
        if (n_kv <= 4096) {
            return 0;
        }
        if (n_kv <= 16384) {
            return 1;
        }
        if (n_kv <= 32768) {
            return 2;
        }
        if (n_kv <= 65536) {
            return 3;
        }
        return 4;
    }

    void accumulate(const kvarn_prof_sample & sample, float ms) {
        const int side = sample.kind == kvarn_prof_kind::LIVE_GROUPS ? 0 : (sample.side ? 1 : 0);
        kvarn_prof_bucket & agg = aggregates[(int) sample.kind][side];
        agg.count += 1;
        agg.total_ms += ms;
        agg.max_ms = std::max<double>(agg.max_ms, ms);
        agg.total_bytes += sample.bytes_out;
        agg.n_kv_min = std::min(agg.n_kv_min, sample.n_kv);
        agg.n_kv_max = std::max(agg.n_kv_max, sample.n_kv);
        agg.n_kv_buckets[n_kv_bucket(sample.n_kv)] += 1;
    }

    void flush_locked(bool print_after) {
        if (pending.empty()) {
            if (print_after) {
                print_locked();
            }
            return;
        }

        std::array<cudaEvent_t, GGML_CUDA_MAX_DEVICES> latest = {};
        for (const kvarn_prof_sample & sample : pending) {
            const int device = sample.events.device;
            if (device >= 0 && device < GGML_CUDA_MAX_DEVICES) {
                latest[device] = sample.events.ev1;
            }
        }

        std::array<bool, GGML_CUDA_MAX_DEVICES> device_ready = {};
        for (int device = 0; device < GGML_CUDA_MAX_DEVICES; ++device) {
            if (latest[device] != nullptr && cudaEventSynchronize(latest[device]) == cudaSuccess) {
                device_ready[device] = true;
            }
        }

        std::vector<kvarn_prof_sample> still_pending;
        still_pending.reserve(pending.size());
        for (const kvarn_prof_sample & sample : pending) {
            const int device = sample.events.device;
            float ms = 0.0f;
            if (device >= 0 && device < GGML_CUDA_MAX_DEVICES && device_ready[device] &&
                    cudaEventElapsedTime(&ms, sample.events.ev0, sample.events.ev1) == cudaSuccess) {
                accumulate(sample, ms);
                release_event_pair_locked(sample.events);
            } else {
                still_pending.push_back(sample);
            }
        }
        pending.swap(still_pending);

        ++flush_count;
        const int dump_every = kvarn_profile_dump_every();
        if (print_after || (dump_every > 0 && flush_count % (uint64_t) dump_every == 0)) {
            print_locked();
        }
    }

    static const char * kind_name(kvarn_prof_kind kind) {
        switch (kind) {
            case kvarn_prof_kind::STORE_HI:     return "store_hi   ";
            case kvarn_prof_kind::STORE_LOW:    return "store_low  ";
            case kvarn_prof_kind::LIVE_GROUPS:  return "live_groups";
            case kvarn_prof_kind::COUNT:        break;
        }
        return "unknown    ";
    }

    void print_locked() const {
        for (int kind = 0; kind < (int) kvarn_prof_kind::COUNT; ++kind) {
            const int n_sides = kind == (int) kvarn_prof_kind::LIVE_GROUPS ? 1 : 2;
            for (int side = 0; side < n_sides; ++side) {
                const kvarn_prof_bucket & agg = aggregates[kind][side];
                if (agg.count == 0) {
                    continue;
                }

                const double mean_us = 1000.0 * agg.total_ms / (double) agg.count;
                const double gib = (double) agg.total_bytes / (1024.0 * 1024.0 * 1024.0);
                const double gbps = agg.total_ms > 0.0 ? gib * 1000.0 / agg.total_ms : 0.0;
                const int n_kv_min = agg.n_kv_min == INT_MAX ? 0 : agg.n_kv_min;
                std::fprintf(stderr,
                        "kvarn-prof: %s %c | count %llu | total %.3f ms | mean %.3f us | max %.3f ms | out %.3f GiB | %.3f GiB/s | n_kv %d..%d | buckets <=4k:%llu <=16k:%llu <=32k:%llu <=64k:%llu >64k:%llu\n",
                        kind_name((kvarn_prof_kind) kind),
                        kind == (int) kvarn_prof_kind::LIVE_GROUPS ? '-' : (side == 0 ? 'K' : 'V'),
                        (unsigned long long) agg.count,
                        agg.total_ms,
                        mean_us,
                        agg.max_ms,
                        gib,
                        gbps,
                        n_kv_min,
                        agg.n_kv_max,
                        (unsigned long long) agg.n_kv_buckets[0],
                        (unsigned long long) agg.n_kv_buckets[1],
                        (unsigned long long) agg.n_kv_buckets[2],
                        (unsigned long long) agg.n_kv_buckets[3],
                        (unsigned long long) agg.n_kv_buckets[4]);
            }
        }
    }
};

static kvarn_prof_state *& kvarn_prof_state_ptr() {
    static kvarn_prof_state * state = nullptr;
    return state;
}

static void kvarn_prof_dump_atexit() {
    ggml_cuda_kvarn_profile_dump();
}

void ggml_cuda_kvarn_profile_dump() {
    kvarn_prof_state * state = kvarn_prof_state_ptr();
    if (state == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->final_dump_done && state->pending.empty()) {
        return;
    }

    state->flush_locked(true);
    state->final_dump_done = true;
}

static kvarn_prof_state * kvarn_prof_get_state() {
    static std::mutex init_mutex;
    kvarn_prof_state *& state = kvarn_prof_state_ptr();
    if (state != nullptr) {
        return state;
    }

    std::lock_guard<std::mutex> lock(init_mutex);
    if (state == nullptr) {
        state = new kvarn_prof_state();
        state->pending.reserve(kvarn_prof_state::flush_threshold);
        std::atexit(kvarn_prof_dump_atexit);
    }
    return state;
}

struct kvarn_prof_scope {
    kvarn_prof_state * state = nullptr;
    kvarn_prof_sample sample;
    bool active = false;
};

static kvarn_prof_scope kvarn_prof_begin(
        ggml_backend_cuda_context & ctx,
        cudaStream_t stream,
        kvarn_prof_kind kind,
        bool value,
        int bits,
        int n_kv,
        size_t bytes_out) {
    if (!kvarn_profile_enabled()) {
        return {};
    }

#ifdef USE_CUDA_GRAPH
    if (ctx.any_cuda_graph_enabled() && !kvarn_profile_cuda_graphs_disabled()) {
        return {};
    }
#endif

    kvarn_prof_state * state = kvarn_prof_get_state();
    kvarn_prof_scope scope;
    scope.state = state;
    scope.sample.kind = kind;
    scope.sample.side = value ? 1 : 0;
    scope.sample.bits = (uint8_t) bits;
    scope.sample.n_kv = n_kv;
    scope.sample.bytes_out = bytes_out;

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->make_event_pair_locked(ctx.device, scope.sample.events)) {
            state->flush_locked(false);
            if (!state->make_event_pair_locked(ctx.device, scope.sample.events)) {
                return {};
            }
        }
    }

    ggml_cuda_set_device(ctx.device);
    if (cudaEventRecord(scope.sample.events.ev0, stream) != cudaSuccess) {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->release_event_pair_locked(scope.sample.events);
        return {};
    }

    scope.active = true;
    return scope;
}

static void kvarn_prof_end(kvarn_prof_scope & scope, cudaStream_t stream) {
    if (!scope.active) {
        return;
    }

    if (cudaEventRecord(scope.sample.events.ev1, stream) != cudaSuccess) {
        std::lock_guard<std::mutex> lock(scope.state->mutex);
        scope.state->release_event_pair_locked(scope.sample.events);
        scope.active = false;
        return;
    }

    std::lock_guard<std::mutex> lock(scope.state->mutex);
    scope.state->pending.push_back(scope.sample);
    scope.active = false;
    if (scope.state->pending.size() >= kvarn_prof_state::flush_threshold) {
        scope.state->flush_locked(false);
    }
}
static bool ggml_cuda_kvarn_valid_bits(int bits) {
    return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8;
}

size_t ggml_cuda_kvarn_required_shared_bytes() {
    return KVAR_N_SHARED_BYTES;
}

size_t ggml_cuda_kvarn_low_shared_bytes() {
    return KVAR_N_LOWSHMEM_BYTES;
}

static __device__ void kvarn_wht_128(float * values) {
    __syncthreads();
    for (int stride = 1; stride < KVAR_N_DIM; stride *= 2) {
        if (threadIdx.x < 64) {
            const int j = (threadIdx.x / stride) * (2 * stride) + (threadIdx.x % stride);
            const float a = values[j];
            const float b = values[j + stride];
            values[j] = a + b;
            values[j + stride] = a - b;
        }
        __syncthreads();
    }
    if (threadIdx.x < KVAR_N_DIM) {
        values[threadIdx.x] *= 0.08838834764831845f;
    }
    __syncthreads();
}

static __device__ void kvarn_wht_128_lane(float * values, int lane_dim) {
    __syncthreads();
    for (int stride = 1; stride < KVAR_N_DIM; stride *= 2) {
        if (lane_dim < 64) {
            const int j = (lane_dim / stride) * (2 * stride) + (lane_dim % stride);
            const float a = values[j];
            const float b = values[j + stride];
            values[j] = a + b;
            values[j + stride] = a - b;
        }
        __syncthreads();
    }
    if (lane_dim < KVAR_N_DIM) {
        values[lane_dim] *= 0.08838834764831845f;
    }
    __syncthreads();
}

static __device__ void kvarn_wht_cross_slices(float values[4], int head_slices) {
    if (head_slices == 1) {
        return;
    }

    for (int stride = 1; stride < head_slices; stride <<= 1) {
#pragma unroll
        for (int base = 0; base < 4; base += 2 * stride) {
            if (base + stride >= head_slices) {
                continue;
            }
#pragma unroll
            for (int i = 0; i < stride; ++i) {
                const int a_idx = base + i;
                const int b_idx = base + stride + i;
                if (b_idx >= head_slices) {
                    continue;
                }
                const float a = values[a_idx];
                const float b = values[b_idx];
                values[a_idx] = a + b;
                values[b_idx] = a - b;
            }
        }
    }

    const float scale = head_slices == 2 ? 0.7071067811865475f : 0.5f;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        if (slice < head_slices) {
            values[slice] *= scale;
        }
    }
}

static __device__ float kvarn_std_col(
        const float * tile,
        const float * s_col,
        const float * s_row,
        int col) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float sc = s_col[col];
    for (int row = 0; row < KVAR_N_DIM; ++row) {
        const float value = tile[row * KVAR_N_DIM + col] / (sc * s_row[row]);
        sum += value;
        sum_sq += value * value;
    }
    const float mean = sum / KVAR_N_DIM;
    return sqrtf(fmaxf((sum_sq - KVAR_N_DIM * mean * mean) / (KVAR_N_DIM - 1), 0.0f));
}

static __device__ float kvarn_std_row(
        const float * tile,
        const float * s_col,
        const float * s_row,
        int row) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float sr = s_row[row];
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float value = tile[row * KVAR_N_DIM + col] / (s_col[col] * sr);
        sum += value;
        sum_sq += value * value;
    }
    const float mean = sum / KVAR_N_DIM;
    return sqrtf(fmaxf((sum_sq - KVAR_N_DIM * mean * mean) / (KVAR_N_DIM - 1), 0.0f));
}

static __device__ __forceinline__ float kvarn_warp_min(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fminf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

static __device__ __forceinline__ float kvarn_warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

static __device__ void kvarn_reduce_std_ranges(
        const float * col_std,
        const float * row_std,
        float * reduce) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float col_min = kvarn_warp_min(col_std[threadIdx.x]);
    float col_max = kvarn_warp_max(col_std[threadIdx.x]);
    float row_min = kvarn_warp_min(row_std[threadIdx.x]);
    float row_max = kvarn_warp_max(row_std[threadIdx.x]);

    if (lane == 0) {
        reduce[warp * 4 + 0] = col_min;
        reduce[warp * 4 + 1] = col_max;
        reduce[warp * 4 + 2] = row_min;
        reduce[warp * 4 + 3] = row_max;
    }
    __syncthreads();

    if (threadIdx.x < 4) {
        const int metric = threadIdx.x;
        float value = reduce[metric];
        for (int w = 1; w < 4; ++w) {
            const float next = reduce[w * 4 + metric];
            value = (metric == 0 || metric == 2) ? fminf(value, next) : fmaxf(value, next);
        }
        reduce[metric] = value;
    }
    __syncthreads();
}

static __device__ void kvarn_update_best_from_std(
        const float * candidate_col,
        const float * candidate_row,
        bool candidate_is_log,
        float * best_col,
        float * best_row,
        float * col_std,
        float * row_std,
        float * best_imbalance,
        float * better,
        float * reduce) {
    const int i = threadIdx.x;
    kvarn_reduce_std_ranges(col_std, row_std, reduce);

    if (i == 0) {
        const float col_min = reduce[0];
        const float col_max = reduce[1];
        const float row_min = reduce[2];
        const float row_max = reduce[3];
        const float imbalance =
            col_max / fmaxf(col_min, 1e-8f) +
            row_max / fmaxf(row_min, 1e-8f);
        *better = imbalance <= *best_imbalance ? 1.0f : 0.0f;
        if (*better != 0.0f) {
            *best_imbalance = imbalance;
        }
    }
    __syncthreads();

    if (*better != 0.0f) {
        best_col[i] = candidate_is_log ? expf(candidate_col[i]) : candidate_col[i];
        best_row[i] = candidate_is_log ? expf(candidate_row[i]) : candidate_row[i];
    }
    __syncthreads();
}

static __device__ void kvarn_quantize_tile(
        uint8_t * record,
        int bits,
        int iterations,
        float * shared) {
    float * tile = shared;
    float * log_s_col = tile + KVAR_N_TILE_VALUES;
    float * log_s_row = log_s_col + KVAR_N_DIM;
    float * s_col = log_s_row + KVAR_N_DIM;
    float * s_row = s_col + KVAR_N_DIM;
    float * best_col = s_row + KVAR_N_DIM;
    float * best_row = best_col + KVAR_N_DIM;
    float * col_std = best_row + KVAR_N_DIM;
    float * row_std = col_std + KVAR_N_DIM;
    float * best_imbalance = row_std + KVAR_N_DIM;
    float * better = best_imbalance + 1;
    float * reduce = better + 1;

    log_s_col[threadIdx.x] = 0.0f;
    log_s_row[threadIdx.x] = 0.0f;
    s_col[threadIdx.x] = 1.0f;
    s_row[threadIdx.x] = 1.0f;
    best_col[threadIdx.x] = 1.0f;
    best_row[threadIdx.x] = 1.0f;
    __syncthreads();

    col_std[threadIdx.x] = kvarn_std_col(tile, s_col, s_row, threadIdx.x);
    row_std[threadIdx.x] = kvarn_std_row(tile, s_col, s_row, threadIdx.x);
    __syncthreads();
    if (threadIdx.x == 0) {
        *best_imbalance = 3.402823466e+38F;
    }
    __syncthreads();
    kvarn_update_best_from_std(
            s_col, s_row, false,
            best_col, best_row, col_std, row_std,
            best_imbalance, better, reduce);

    for (int iter = 0; iter < iterations; ++iter) {
        const float col = fminf(fmaxf(col_std[threadIdx.x], 1e-3f), 1e3f);
        log_s_col[threadIdx.x] = fminf(fmaxf(log_s_col[threadIdx.x] + logf(col), -0.3f), 10.0f);
        s_col[threadIdx.x] = expf(log_s_col[threadIdx.x]);
        __syncthreads();

        row_std[threadIdx.x] = kvarn_std_row(tile, s_col, s_row, threadIdx.x);
        __syncthreads();

        const float row = fminf(fmaxf(row_std[threadIdx.x], 1e-3f), 1e3f);
        const float log_s_row_new = fminf(fmaxf(log_s_row[threadIdx.x] + logf(row), -0.3f), 10.0f);
        log_s_row[threadIdx.x] = log_s_row_new;
        s_row[threadIdx.x] = expf(log_s_row_new);
        __syncthreads();

        row_std[threadIdx.x] = kvarn_std_row(tile, s_col, s_row, threadIdx.x);
        col_std[threadIdx.x] = kvarn_std_col(tile, s_col, s_row, threadIdx.x);
        __syncthreads();

        kvarn_update_best_from_std(
                s_col, s_row, false,
                best_col, best_row, col_std, row_std,
                best_imbalance, better, reduce);
    }

    const int row = threadIdx.x;
    float lo = 3.402823466e+38F;
    float hi = -3.402823466e+38F;
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float x = tile[row * KVAR_N_DIM + col] / (best_col[col] * best_row[row]);
        lo = fminf(lo, x);
        hi = fmaxf(hi, x);
    }

    const int qmax = (1 << bits) - 1;
    const float scale = fmaxf((hi - lo) / qmax, 1e-10f);
    const int row_bytes = KVAR_N_DIM * bits / 8;
    uint8_t * row_payload = record + row * row_bytes;
    for (int i = 0; i < row_bytes; ++i) {
        row_payload[i] = 0;
    }
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float x = tile[row * KVAR_N_DIM + col] / (best_col[col] * best_row[row]);
        const uint8_t q = (uint8_t) fminf(fmaxf(roundf((x - lo) / scale), 0.0f), (float) qmax);
        const int bit_offset = col * bits;
        for (int bit = 0; bit < bits; ++bit) {
            const int dst_bit = bit_offset + bit;
            row_payload[dst_bit / 8] |= ((q >> bit) & 1u) << (dst_bit % 8);
        }
    }

    const int payload_bytes = KVAR_N_TILE_VALUES * bits / 8;
    half * scale_axis = (half *) (record + payload_bytes);
    half * zp_axis = scale_axis + KVAR_N_DIM;
    half * other_axis = zp_axis + KVAR_N_DIM;
    scale_axis[row] = __float2half_rn(best_row[row] * scale);
    zp_axis[row] = __float2half_rn(best_row[row] * lo);
    other_axis[row] = __float2half_rn(best_col[row]);
    __syncthreads();
}

static __device__ void kvarn_quantize_stage(
        const half * stage,
        uint8_t * record,
        int n_heads,
        int head,
        int stage_base,
        int stage_group,
        int bits,
        int iterations,
        bool value,
        bool swa,
        int stage_groups,
        int tail_groups,
        float * shared) {
    float * tile = shared;
    // SWA uses a stage_groups-deep ping-pong over absolute tiles; non-SWA keeps
    // tile 0 as a permanent sink and ping-pongs the tail_groups newest tiles in
    // staging slots 1..stage_groups-1.
    const int stage_slot = swa ? (stage_group % stage_groups) : (1 + ((stage_group - 1) % tail_groups));
    for (int i = threadIdx.x; i < KVAR_N_TILE_VALUES; i += blockDim.x) {
        const int row = i / KVAR_N_DIM;
        const int col = i % KVAR_N_DIM;
        const int token = value ? row : col;
        const int dim = value ? col : row;
        const int stage_pos = stage_base + stage_slot * KVAR_N_DIM + token;
        tile[i] = __half2float(stage[(stage_pos * n_heads + head) * KVAR_N_DIM + dim]);
    }
    __syncthreads();
    // Stage and records are rotated-domain for both K and V.
    kvarn_quantize_tile(record, bits, iterations, shared);
}

static __device__ float kvarn_stage_rotated_value(
        const half * stage,
        int n_heads,
        int head,
        int stage_base,
        int stage_group,
        bool value,
        bool swa,
        int stage_groups,
        int tail_groups,
        int row,
        int col) {
    const int stage_slot = (
        swa ? stage_group % stage_groups : 1 + ((stage_group - 1) % tail_groups)) * KVAR_N_DIM;
    const int token = value ? row : col;
    const int dim = value ? col : row;
    const int stage_pos = stage_base + stage_slot + token;
    return __half2float(stage[(stage_pos * n_heads + head) * KVAR_N_DIM + dim]);
}

static __device__ float kvarn_std_col_lowshmem(
        const half * stage,
        int n_heads,
        int head,
        int stage_base,
        int stage_group,
        const float * log_s_col,
        const float * log_s_row,
        bool value,
        bool swa,
        int stage_groups,
        int tail_groups,
        int col) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float sc = expf(log_s_col[col]);
    for (int row = 0; row < KVAR_N_DIM; ++row) {
        const float raw = kvarn_stage_rotated_value(
                stage, n_heads, head, stage_base, stage_group, value, swa, stage_groups, tail_groups, row, col);
        const float scaled = raw / (sc * expf(log_s_row[row]));
        sum += scaled;
        sum_sq += scaled * scaled;
    }
    const float mean = sum / KVAR_N_DIM;
    return sqrtf(fmaxf((sum_sq - KVAR_N_DIM * mean * mean) / (KVAR_N_DIM - 1), 0.0f));
}

static __device__ float kvarn_std_row_lowshmem(
        const half * stage,
        int n_heads,
        int head,
        int stage_base,
        int stage_group,
        const float * log_s_col,
        const float * log_s_row,
        bool value,
        bool swa,
        int stage_groups,
        int tail_groups,
        int row) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float sr = expf(log_s_row[row]);
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float raw = kvarn_stage_rotated_value(
                stage, n_heads, head, stage_base, stage_group, value, swa, stage_groups, tail_groups, row, col);
        const float scaled = raw / (expf(log_s_col[col]) * sr);
        sum += scaled;
        sum_sq += scaled * scaled;
    }
    const float mean = sum / KVAR_N_DIM;
    return sqrtf(fmaxf((sum_sq - KVAR_N_DIM * mean * mean) / (KVAR_N_DIM - 1), 0.0f));
}

static __device__ void kvarn_quantize_stage_lowshmem(
        const half * stage,
        uint8_t * record,
        int n_heads,
        int head,
        int stage_base,
        int stage_group,
        int bits,
        int iterations,
        bool value,
        bool swa,
        int stage_groups,
        int tail_groups,
        float * shared) {
    float * log_s_col = shared;
    float * log_s_row = log_s_col + KVAR_N_DIM;
    float * best_col = log_s_row + KVAR_N_DIM;
    float * best_row = best_col + KVAR_N_DIM;
    float * col_std = best_row + KVAR_N_DIM;
    float * row_std = col_std + KVAR_N_DIM;
    float * best_imbalance = row_std + KVAR_N_DIM;
    float * better = best_imbalance + 1;
    float * reduce = better + 1;

    log_s_col[threadIdx.x] = 0.0f;
    log_s_row[threadIdx.x] = 0.0f;
    best_col[threadIdx.x] = 1.0f;
    best_row[threadIdx.x] = 1.0f;
    __syncthreads();

    col_std[threadIdx.x] = kvarn_std_col_lowshmem(
            stage, n_heads, head, stage_base, stage_group, log_s_col, log_s_row,
            value, swa, stage_groups, tail_groups, threadIdx.x);
    row_std[threadIdx.x] = kvarn_std_row_lowshmem(
            stage, n_heads, head, stage_base, stage_group, log_s_col, log_s_row,
            value, swa, stage_groups, tail_groups, threadIdx.x);
    __syncthreads();

    if (threadIdx.x == 0) {
        *best_imbalance = 3.402823466e+38F;
    }
    __syncthreads();
    kvarn_update_best_from_std(
            log_s_col, log_s_row, true,
            best_col, best_row, col_std, row_std,
            best_imbalance, better, reduce);

    for (int iter = 0; iter < iterations; ++iter) {
        const float col = fminf(fmaxf(col_std[threadIdx.x], 1e-3f), 1e3f);
        log_s_col[threadIdx.x] = fminf(fmaxf(log_s_col[threadIdx.x] + logf(col), -0.3f), 10.0f);
        __syncthreads();

        row_std[threadIdx.x] = kvarn_std_row_lowshmem(
                stage, n_heads, head, stage_base, stage_group,
                log_s_col, log_s_row, value, swa, stage_groups, tail_groups, threadIdx.x);
        __syncthreads();

        const float row = fminf(fmaxf(row_std[threadIdx.x], 1e-3f), 1e3f);
        const float log_s_row_new = fminf(fmaxf(log_s_row[threadIdx.x] + logf(row), -0.3f), 10.0f);
        log_s_row[threadIdx.x] = log_s_row_new;
        __syncthreads();

        row_std[threadIdx.x] = kvarn_std_row_lowshmem(
                stage, n_heads, head, stage_base, stage_group,
                log_s_col, log_s_row, value, swa, stage_groups, tail_groups, threadIdx.x);
        col_std[threadIdx.x] = kvarn_std_col_lowshmem(
                stage, n_heads, head, stage_base, stage_group,
                log_s_col, log_s_row, value, swa, stage_groups, tail_groups, threadIdx.x);
        __syncthreads();

        kvarn_update_best_from_std(
                log_s_col, log_s_row, true,
                best_col, best_row, col_std, row_std,
                best_imbalance, better, reduce);
    }

    const int row = threadIdx.x;
    float lo = 3.402823466e+38F;
    float hi = -3.402823466e+38F;
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float raw = kvarn_stage_rotated_value(
                stage, n_heads, head, stage_base, stage_group, value, swa, stage_groups, tail_groups, row, col);
        const float x = raw / (best_col[col] * best_row[row]);
        lo = fminf(lo, x);
        hi = fmaxf(hi, x);
    }

    const int qmax = (1 << bits) - 1;
    const float scale = fmaxf((hi - lo) / qmax, 1e-10f);
    const int row_bytes = KVAR_N_DIM * bits / 8;
    uint8_t * row_payload = record + row * row_bytes;
    for (int i = 0; i < row_bytes; ++i) {
        row_payload[i] = 0;
    }
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float raw = kvarn_stage_rotated_value(
                stage, n_heads, head, stage_base, stage_group, value, swa, stage_groups, tail_groups, row, col);
        const float x = raw / (best_col[col] * best_row[row]);
        const uint8_t q = (uint8_t) fminf(fmaxf(roundf((x - lo) / scale), 0.0f), (float) qmax);
        const int bit_offset = col * bits;
        for (int bit = 0; bit < bits; ++bit) {
            const int dst_bit = bit_offset + bit;
            row_payload[dst_bit / 8] |= ((q >> bit) & 1u) << (dst_bit % 8);
        }
    }

    const int payload_bytes = KVAR_N_TILE_VALUES * bits / 8;
    half * scale_axis = (half *) (record + payload_bytes);
    half * zp_axis = scale_axis + KVAR_N_DIM;
    half * other_axis = zp_axis + KVAR_N_DIM;
    scale_axis[row] = __float2half_rn(best_row[row] * scale);
    zp_axis[row] = __float2half_rn(best_row[row] * lo);
    other_axis[row] = __float2half_rn(best_col[row]);
    __syncthreads();
}

static __global__ void kvarn_store_kernel_hishmem(
        const float * current,
        const int64_t * indices,
        half * stage,
        uint8_t * records,
        int n_heads,
        int n_tokens,
        int n_stream,
        int groups_per_stream,
        int record_bytes,
        int bits,
        int iterations,
        bool value,
        bool swa,
        int stage_groups,
        int tail_groups,
        bool eager_records,
        const int * skip_if_workspace_valid) {
    extern __shared__ float shared[];
    const int head = blockIdx.x;
    if (skip_if_workspace_valid != nullptr && skip_if_workspace_valid[0] != 0) {
        return;
    }

    for (int token = 0; token < n_tokens; ++token) {
        const int64_t idx = indices[token];
        const int group_global = (int) (idx / KVAR_N_DIM);
        const int pos = (int) (idx % KVAR_N_DIM);
        // SWA: idx is the absolute token position; records form a ring and there
        // is no permanent group-0 sink (single stream).
        const int stream = swa ? 0 : group_global / groups_per_stream;
        const int group = swa ? group_global : group_global - stream * groups_per_stream;
        if (stream < 0 || stream >= n_stream || group < 0 || (!swa && group >= groups_per_stream)) {
            return;
        }

        const int stage_base = stream * KVAR_N_DIM * stage_groups;
        if (!eager_records && pos == 0 && (swa ? group >= tail_groups : group > tail_groups)) {
            const int flush_group = group - tail_groups;
            const int flush_ring = swa ? flush_group % groups_per_stream : flush_group;
            const int flush_record_group = stream * groups_per_stream + flush_ring;
            uint8_t * record = records + (flush_record_group * n_heads + head) * record_bytes;
            kvarn_quantize_stage(stage, record, n_heads, head, stage_base, flush_group, bits, iterations, value, swa, stage_groups, tail_groups, shared);
        }

        shared[threadIdx.x] = current[(token * n_heads + head) * KVAR_N_DIM + threadIdx.x];
        kvarn_wht_128(shared);
        const int stage_slot = swa ? (group % stage_groups) : (group == 0 ? 0 : 1 + ((group - 1) % tail_groups));
        const int stage_pos = stage_base + stage_slot * KVAR_N_DIM + pos;
        stage[(stage_pos * n_heads + head) * KVAR_N_DIM + threadIdx.x] =
            __float2half_rn(shared[threadIdx.x]);
        __syncthreads();
        if (eager_records && pos == KVAR_N_DIM - 1 && (swa || group > 0)) {
            const int record_ring = swa ? group % groups_per_stream : group;
            const int record_group = stream * groups_per_stream + record_ring;
            uint8_t * record = records + ((int64_t) record_group * n_heads + head) * record_bytes;
            kvarn_quantize_stage(stage, record, n_heads, head, stage_base, group,
                    bits, iterations, value, swa, stage_groups, tail_groups, shared);
        }
    }
}

static __global__ void kvarn_store_kernel_headwide(
        const float * current,
        const int64_t * indices,
        half * stage,
        uint8_t * records,
        int n_heads,
        int n_tokens,
        int n_stream,
        int groups_per_stream,
        int record_bytes,
        int bits,
        int iterations,
        bool value,
        bool swa,
        int stage_groups,
        int tail_groups,
        int head_slices,
        bool eager_records,
        const int * skip_if_workspace_valid) {
    extern __shared__ float shared[];
    const int head0 = blockIdx.x * head_slices;
    if (head0 + head_slices > n_heads) {
        return;
    }
    if (skip_if_workspace_valid != nullptr && skip_if_workspace_valid[0] != 0) {
        return;
    }
    const int dim = threadIdx.x;

    for (int token = 0; token < n_tokens; ++token) {
        const int64_t idx = indices[token];
        const int group_global = (int) (idx / KVAR_N_DIM);
        const int pos = (int) (idx % KVAR_N_DIM);
        const int stream = swa ? 0 : group_global / groups_per_stream;
        const int group = swa ? group_global : group_global - stream * groups_per_stream;
        if (stream < 0 || stream >= n_stream || group < 0 || (!swa && group >= groups_per_stream)) {
            return;
        }

        const int stage_base = stream * KVAR_N_DIM * stage_groups;
        if (!eager_records && pos == 0 && (swa ? group >= tail_groups : group > tail_groups)) {
            const int flush_group = group - tail_groups;
            const int flush_ring = swa ? flush_group % groups_per_stream : flush_group;
            const int flush_record_group = stream * groups_per_stream + flush_ring;
            for (int slice = 0; slice < head_slices; ++slice) {
                const int head = head0 + slice;
                uint8_t * record = records + ((int64_t) flush_record_group * n_heads + head) * record_bytes;
                kvarn_quantize_stage(stage, record, n_heads, head, stage_base, flush_group, bits, iterations, value, swa, stage_groups, tail_groups, shared);
            }
        }

        float values[4] = {};
        for (int slice = 0; slice < head_slices; ++slice) {
            const int head = head0 + slice;
            shared[dim] = current[((int64_t) token * n_heads + head) * KVAR_N_DIM + dim];
            kvarn_wht_128(shared);
            values[slice] = shared[dim];
        }

        kvarn_wht_cross_slices(values, head_slices);

        const int stage_slot = swa ? (group % stage_groups) : (group == 0 ? 0 : 1 + ((group - 1) % tail_groups));
        const int stage_pos = stage_base + stage_slot * KVAR_N_DIM + pos;
        for (int slice = 0; slice < head_slices; ++slice) {
            const int head = head0 + slice;
            stage[((int64_t) stage_pos * n_heads + head) * KVAR_N_DIM + dim] =
                __float2half_rn(values[slice]);
        }
        __syncthreads();
        if (eager_records && pos == KVAR_N_DIM - 1 && (swa || group > 0)) {
            const int record_ring = swa ? group % groups_per_stream : group;
            const int record_group = stream * groups_per_stream + record_ring;
            for (int slice = 0; slice < head_slices; ++slice) {
                const int head = head0 + slice;
                uint8_t * record = records + ((int64_t) record_group * n_heads + head) * record_bytes;
                kvarn_quantize_stage(stage, record, n_heads, head, stage_base, group,
                        bits, iterations, value, swa, stage_groups, tail_groups, shared);
            }
        }
    }
}

static __global__ void kvarn_store_kernel_lowshmem(
        const float * current,
        const int64_t * indices,
        half * stage,
        uint8_t * records,
        int n_heads,
        int n_tokens,
        int n_stream,
        int groups_per_stream,
        int record_bytes,
        int bits,
        int iterations,
        bool value,
        bool swa,
        int stage_groups,
        int tail_groups,
        int head_slices,
        bool eager_records) {
    extern __shared__ float shared[];
    const int head0 = blockIdx.x * head_slices;
    if (head0 + head_slices > n_heads) {
        return;
    }

    for (int token = 0; token < n_tokens; ++token) {
        const int64_t idx = indices[token];
        const int group_global = (int) (idx / KVAR_N_DIM);
        const int stream = swa ? 0 : group_global / groups_per_stream;
        const int group = swa ? group_global : group_global - stream * groups_per_stream;
        const int pos = (int) (idx % KVAR_N_DIM);
        if (stream < 0 || stream >= n_stream || group < 0 || (!swa && group >= groups_per_stream)) {
            return;
        }

        const int stage_base = stream * KVAR_N_DIM * stage_groups;
        if (!eager_records && pos == 0 && (swa ? group >= tail_groups : group > tail_groups)) {
            const int flush_group = group - tail_groups;
            const int flush_ring = swa ? flush_group % groups_per_stream : flush_group;
            const int flush_record_group = stream * groups_per_stream + flush_ring;
            for (int slice = 0; slice < head_slices; ++slice) {
                const int head = head0 + slice;
                uint8_t * record = records + ((int64_t) flush_record_group * n_heads + head) * record_bytes;
                kvarn_quantize_stage_lowshmem(
                        stage, record, n_heads, head, stage_base, flush_group, bits, iterations,
                        value, swa, stage_groups, tail_groups, shared);
            }
        }

        float values[4] = {};
        for (int slice = 0; slice < head_slices; ++slice) {
            const int head = head0 + slice;
            shared[threadIdx.x] = current[((int64_t) token * n_heads + head) * KVAR_N_DIM + threadIdx.x];
            kvarn_wht_128(shared);
            values[slice] = shared[threadIdx.x];
        }
        kvarn_wht_cross_slices(values, head_slices);

        const int stage_slot = swa ? group % stage_groups : (group == 0 ? 0 : 1 + ((group - 1) % tail_groups));
        const int stage_pos = stage_base + stage_slot * KVAR_N_DIM + pos;
        for (int slice = 0; slice < head_slices; ++slice) {
            const int head = head0 + slice;
            stage[((int64_t) stage_pos * n_heads + head) * KVAR_N_DIM + threadIdx.x] =
                __float2half_rn(values[slice]);
        }
        __syncthreads();
        if (eager_records && pos == KVAR_N_DIM - 1 && (swa || group > 0)) {
            const int record_ring = swa ? group % groups_per_stream : group;
            const int record_group = stream * groups_per_stream + record_ring;
            for (int slice = 0; slice < head_slices; ++slice) {
                const int head = head0 + slice;
                uint8_t * record = records + ((int64_t) record_group * n_heads + head) * record_bytes;
                kvarn_quantize_stage_lowshmem(
                        stage, record, n_heads, head, stage_base, group, bits, iterations,
                        value, swa, stage_groups, tail_groups, shared);
            }
        }
    }
}

static __global__ void kvarn_store_direct_flush_kernel(
        const int64_t * indices,
        const half * stage,
        uint8_t * records,
        int n_heads,
        int n_tokens,
        int n_stream,
        int groups_per_stream,
        int record_bytes,
        int bits,
        int iterations,
        bool value,
        bool swa,
        int stage_groups,
        int tail_groups) {
    extern __shared__ float shared[];
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (head >= n_heads || token >= n_tokens) {
        return;
    }

    const int64_t idx = indices[token];
    const int group_global = (int) (idx / KVAR_N_DIM);
    const int stream = swa ? 0 : group_global / groups_per_stream;
    const int group = swa ? group_global : group_global - stream * groups_per_stream;
    const int pos = (int) (idx % KVAR_N_DIM);
    if (stream < 0 || stream >= n_stream || group < 0 || (!swa && group >= groups_per_stream) || pos != 0) {
        return;
    }
    if (swa ? group < tail_groups : group <= tail_groups) {
        return;
    }

    const int flush_group = group - tail_groups;
    const int stage_base = stream * KVAR_N_DIM * stage_groups;
    const int flush_ring = swa ? flush_group % groups_per_stream : flush_group;
    const int flush_record_group = stream * groups_per_stream + flush_ring;
    uint8_t * record = records + ((int64_t) flush_record_group * n_heads + head) * record_bytes;
    kvarn_quantize_stage(stage, record, n_heads, head, stage_base, flush_group, bits, iterations, value, swa, stage_groups, tail_groups, shared);
}

static __global__ void kvarn_store_direct_stage_kernel(
        const float * current,
        const int64_t * indices,
        half * stage,
        int n_heads,
        int n_tokens,
        int n_stream,
        int groups_per_stream,
        int stage_groups,
        int tail_groups,
        bool value,
        bool swa) {
    const int head = blockIdx.x;
    const int chunk = blockIdx.y;
    const int lane = threadIdx.x / KVAR_N_DIM;
    const int dim = threadIdx.x - lane * KVAR_N_DIM;
    const int token = chunk * KVAR_N_STAGE_CHUNK + lane;
    if (head >= n_heads || lane >= KVAR_N_STAGE_CHUNK) {
        return;
    }

    bool valid = token < n_tokens;
    int stream = 0;
    int group = 0;
    int pos = 0;
    if (valid) {
        const int64_t idx = indices[token];
        const int group_global = (int) (idx / KVAR_N_DIM);
        stream = swa ? 0 : group_global / groups_per_stream;
        group = swa ? group_global : group_global - stream * groups_per_stream;
        pos = (int) (idx % KVAR_N_DIM);
        valid = stream >= 0 && stream < n_stream && group >= 0 && (swa || group < groups_per_stream);
    }

    __shared__ float shared[KVAR_N_STAGE_CHUNK * KVAR_N_DIM];
    float * values = shared + lane * KVAR_N_DIM;
    values[dim] = valid ? current[((int64_t) token * n_heads + head) * KVAR_N_DIM + dim] : 0.0f;
    kvarn_wht_128_lane(values, dim);
    if (!valid) {
        return;
    }

    const int stage_base = stream * KVAR_N_DIM * stage_groups;
    const int stage_pos = stage_base + (
        swa ? (group % stage_groups) * KVAR_N_DIM + pos :
        (group == 0 ? pos : KVAR_N_DIM + ((group - 1) % tail_groups) * KVAR_N_DIM + pos));
    stage[((int64_t) stage_pos * n_heads + head) * KVAR_N_DIM + dim] =
        __float2half_rn(values[dim]);
}

static __global__ void kvarn_store_workspace_stage_kernel(
        const float * current,
        half * workspace,
        int n_heads,
        int n_tokens,
        bool value) {
    const int head = blockIdx.x;
    const int chunk = blockIdx.y;
    const int lane = threadIdx.x / KVAR_N_DIM;
    const int dim = threadIdx.x - lane * KVAR_N_DIM;
    const int token = chunk * KVAR_N_STAGE_CHUNK + lane;
    if (head >= n_heads || lane >= KVAR_N_STAGE_CHUNK) {
        return;
    }

    __shared__ float shared[KVAR_N_STAGE_CHUNK * KVAR_N_DIM];
    float * values = shared + lane * KVAR_N_DIM;
    values[dim] = token < n_tokens ?
        current[((int64_t) token * n_heads + head) * KVAR_N_DIM + dim] : 0.0f;
    kvarn_wht_128_lane(values, dim);
    if (token < n_tokens) {
        workspace[((int64_t) token * n_heads + head) * KVAR_N_DIM + dim] =
            __float2half_rn(values[dim]);
    }
}

template<int HEAD_SLICES>
static __global__ void kvarn_store_workspace_stage_headwide_kernel(
        const float * current,
        half * workspace,
        int n_heads,
        int n_tokens,
        bool value) {
    GGML_UNUSED(value);

    const int head0 = blockIdx.x * HEAD_SLICES;
    const int chunk = blockIdx.y;
    const int lane = threadIdx.x / KVAR_N_DIM;
    const int dim = threadIdx.x - lane * KVAR_N_DIM;
    const int token = chunk * KVAR_N_STAGE_CHUNK + lane;
    if (head0 + HEAD_SLICES > n_heads || lane >= KVAR_N_STAGE_CHUNK) {
        return;
    }

    __shared__ float shared[KVAR_N_STAGE_CHUNK * HEAD_SLICES * KVAR_N_DIM];
    for (int slice = 0; slice < HEAD_SLICES; ++slice) {
        float * values = shared + (lane * HEAD_SLICES + slice) * KVAR_N_DIM;
        const int head = head0 + slice;
        values[dim] = token < n_tokens ? current[((int64_t) token * n_heads + head) * KVAR_N_DIM + dim] : 0.0f;
        kvarn_wht_128_lane(values, dim);
    }

    if (token >= n_tokens) {
        return;
    }

    float values[4] = {};
    for (int slice = 0; slice < HEAD_SLICES; ++slice) {
        values[slice] = shared[(lane * HEAD_SLICES + slice) * KVAR_N_DIM + dim];
    }
    kvarn_wht_cross_slices(values, HEAD_SLICES);

    for (int slice = 0; slice < HEAD_SLICES; ++slice) {
        const int head = head0 + slice;
        workspace[((int64_t) token * n_heads + head) * KVAR_N_DIM + dim] =
            __float2half_rn(values[slice]);
    }
}

static __global__ void kvarn_store_workspace_validate_kernel(
        const int64_t * indices,
        int n_tokens,
        int n_stream,
        int groups_per_stream,
        int tokens_per_stream,
        int active_streams,
        bool swa,
        int * workspace_valid) {
    if (blockIdx.x != 0) {
        return;
    }

    __shared__ int valid;
    if (threadIdx.x == 0) {
        valid =
        tokens_per_stream > 0 &&
        active_streams > 0 &&
        active_streams <= n_stream &&
        n_tokens == active_streams * tokens_per_stream ? 1 : 0;
    }
    __syncthreads();

    for (int active_stream = 0; active_stream < active_streams; ++active_stream) {
        if (valid == 0) {
            break;
        }
        const int token_base = active_stream * tokens_per_stream;
        const int64_t first_idx = indices[token_base];

        for (int t = threadIdx.x; t < tokens_per_stream; t += blockDim.x) {
            if (indices[token_base + t] != first_idx + t) {
                atomicExch(&valid, 0);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0 && valid != 0) {
            const int first_group_global = (int) (first_idx / KVAR_N_DIM);
            const int stream = swa ? 0 : first_group_global / groups_per_stream;
            const int first_group = swa ? first_group_global : first_group_global - stream * groups_per_stream;
            const int first_pos = (int) (first_idx % KVAR_N_DIM);
            const int start_local = first_group * KVAR_N_DIM + first_pos;
            const int end_local = start_local + tokens_per_stream;
            if (stream < 0 || stream >= n_stream ||
                    first_group < 0 || (!swa && first_group >= groups_per_stream) ||
                    first_pos < 0 || first_pos >= KVAR_N_DIM ||
                    (!swa && end_local > groups_per_stream * KVAR_N_DIM)) {
                valid = 0;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        workspace_valid[0] = valid;
    }
}

static __global__ void kvarn_store_workspace_flush_kernel(
        const int64_t * indices,
        const half * stage,
        const half * workspace,
        uint8_t * records,
        const int * workspace_valid,
        int n_heads,
        int n_tokens,
        int n_stream,
        int groups_per_stream,
        int record_bytes,
        int tokens_per_stream,
        int flush_candidates,
        int bits,
        int iterations,
        bool value,
        int stage_groups,
        int tail_groups,
        bool swa,
        bool eager_records) {
    extern __shared__ float shared[];
    const int head = blockIdx.x;
    const int active_stream = blockIdx.y / flush_candidates;
    const int candidate = blockIdx.y - active_stream * flush_candidates;
    const int token_base = active_stream * tokens_per_stream;
    if ((workspace_valid != nullptr && workspace_valid[0] == 0) || head >= n_heads || token_base >= n_tokens || tokens_per_stream <= 0) {
        return;
    }

    const int64_t first_idx = indices[token_base];
    const int64_t last_idx = indices[token_base + tokens_per_stream - 1];
    if (last_idx != first_idx + tokens_per_stream - 1) {
        return;
    }

    const int first_group_global = (int) (first_idx / KVAR_N_DIM);
    const int stream = swa ? 0 : first_group_global / groups_per_stream;
    const int first_group = swa ? first_group_global : first_group_global - stream * groups_per_stream;
    const int first_pos = (int) (first_idx % KVAR_N_DIM);
    if (stream < 0 || stream >= n_stream || first_group < 0 || (!swa && first_group >= groups_per_stream) || first_pos < 0) {
        return;
    }

    const int start_local = first_group * KVAR_N_DIM + first_pos;
    const int end_local = start_local + tokens_per_stream;

    // Delayed stores seal a group when it leaves the transient stage, so they
    // enumerate the group boundaries this store crosses. Exact-tail stores
    // instead need a compressed copy as soon as the source tile completes, so
    // they enumerate the groups that COMPLETE inside this store - starting at
    // the group containing start_local. An unaligned start leaves that group
    // straddling the store boundary: its prefix is still resident in the
    // transient stage slot and is picked up by the tile loader below. Anchoring
    // the eager scan on the first crossed boundary instead would skip that
    // group permanently, because no later store enumerates it either.
    const int record_group = eager_records ?
        start_local / KVAR_N_DIM + candidate :
        (start_local + KVAR_N_DIM - 1) / KVAR_N_DIM + candidate - tail_groups;
    if (eager_records) {
        // Only seal a complete tile. A trailing partial group stays in the stage
        // until a later store finishes it; quantizing it early would fold stale
        // stage rows past end_local into its scales.
        if ((record_group + 1) * KVAR_N_DIM > end_local) {
            return;
        }
    } else {
        const int boundary_group = record_group + tail_groups;
        if (boundary_group * KVAR_N_DIM >= end_local) {
            return;
        }
    }
    if (swa ? record_group < 0 : (record_group < 1 || record_group >= groups_per_stream)) {
        return;
    }
    if (swa) {
        if (eager_records) {
            // The ring slot is rewritten later in this same store; let that
            // candidate win instead of quantizing the older tile first.
            if ((record_group + groups_per_stream + 1) * KVAR_N_DIM <= end_local) {
                return;
            }
        } else if ((record_group + groups_per_stream + tail_groups) * KVAR_N_DIM < end_local) {
            return;
        }
    }

    const int flush_start = record_group * KVAR_N_DIM;
    const int stage_base = stream * KVAR_N_DIM * stage_groups;
    float * tile = shared;
    for (int i = threadIdx.x; i < KVAR_N_TILE_VALUES; i += blockDim.x) {
        const int row = i / KVAR_N_DIM;
        const int col = i % KVAR_N_DIM;
        const int token = value ? row : col;
        const int dim = value ? col : row;
        const int local_pos = flush_start + token;
        if (local_pos >= start_local && local_pos < end_local) {
            const int src_token = token_base + local_pos - start_local;
            tile[i] = __half2float(workspace[((int64_t) src_token * n_heads + head) * KVAR_N_DIM + dim]);
        } else {
            const int stage_slot = swa ? (record_group % stage_groups) : 1 + ((record_group - 1) % tail_groups);
            const int stage_pos = stage_base + stage_slot * KVAR_N_DIM + token;
            tile[i] = __half2float(stage[(stage_pos * n_heads + head) * KVAR_N_DIM + dim]);
        }
    }
    __syncthreads();

    const int flush_record_group = stream * groups_per_stream + (swa ? record_group % groups_per_stream : record_group);
    uint8_t * record = records + ((int64_t) flush_record_group * n_heads + head) * record_bytes;
    kvarn_quantize_tile(record, bits, iterations, shared);
}

static __global__ void kvarn_store_workspace_commit_kernel(
        const int64_t * indices,
        const half * workspace,
        half * stage,
        const int * workspace_valid,
        int n_heads,
        int n_tokens,
        int n_stream,
        int groups_per_stream,
        int tokens_per_stream,
        int stage_groups,
        int tail_groups,
        bool swa) {
    const int head = blockIdx.x;
    const int active_stream = blockIdx.y / (KVAR_N_DIM * stage_groups);
    const int stage_local = blockIdx.y - active_stream * (KVAR_N_DIM * stage_groups);
    const int token_base = active_stream * tokens_per_stream;
    if ((workspace_valid != nullptr && workspace_valid[0] == 0) || head >= n_heads || token_base >= n_tokens || tokens_per_stream <= 0) {
        return;
    }

    const int64_t first_idx = indices[token_base];
    const int64_t last_idx = indices[token_base + tokens_per_stream - 1];
    if (last_idx != first_idx + tokens_per_stream - 1) {
        return;
    }

    const int first_group_global = (int) (first_idx / KVAR_N_DIM);
    const int stream = swa ? 0 : first_group_global / groups_per_stream;
    const int first_group = swa ? first_group_global : first_group_global - stream * groups_per_stream;
    const int first_pos = (int) (first_idx % KVAR_N_DIM);
    if (stream < 0 || stream >= n_stream || first_group < 0 || (!swa && first_group >= groups_per_stream) || first_pos < 0) {
        return;
    }

    const int start_local = first_group * KVAR_N_DIM + first_pos;
    const int end_local = start_local + tokens_per_stream;
    const int pos = stage_local % KVAR_N_DIM;
    int group = 0;
    if (!swa && stage_local < KVAR_N_DIM) {
        const int local_pos = pos;
        if (local_pos < start_local || local_pos >= end_local) {
            return;
        }
    } else {
        const int slot = swa ? stage_local / KVAR_N_DIM : (stage_local - KVAR_N_DIM) / KVAR_N_DIM;
        int max_group = (end_local - 1 - pos) / KVAR_N_DIM;
        if (max_group < (swa ? 0 : 1)) {
            return;
        }
        while (max_group >= (swa ? 0 : 1) && (swa ? (max_group % stage_groups) != slot : ((max_group - 1) % tail_groups) != slot)) {
            --max_group;
        }
        if (max_group < (swa ? 0 : 1)) {
            return;
        }
        group = max_group;
        const int local_pos = group * KVAR_N_DIM + pos;
        if ((!swa && group >= groups_per_stream) || local_pos < start_local || local_pos >= end_local) {
            return;
        }
    }

    const int local_pos = group * KVAR_N_DIM + pos;
    const int token = token_base + local_pos - start_local;
    if (token < token_base || token >= token_base + tokens_per_stream) {
        return;
    }

    const int stage_base = stream * KVAR_N_DIM * stage_groups;
    const int stage_pos = stage_base + stage_local;
    stage[(stage_pos * n_heads + head) * KVAR_N_DIM + threadIdx.x] =
        workspace[((int64_t) token * n_heads + head) * KVAR_N_DIM + threadIdx.x];
}

void ggml_cuda_op_kvarn_store(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * current = dst->src[0];
    const ggml_tensor * indices = dst->src[1];
    ggml_tensor * stage = dst->src[2];
    ggml_tensor * records = dst->src[3];
    GGML_ASSERT(ggml_is_contiguous(current));
    GGML_ASSERT(ggml_is_contiguous(indices));
    GGML_ASSERT(ggml_is_contiguous(stage));
    GGML_ASSERT(ggml_is_contiguous(records));

    const int bits = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_BITS);
    const int iterations = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_ITERS);
    const bool value = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_STORE_VALUE) != 0;
    const int tokens_per_stream_hint = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_TOKENS_PER_STREAM);
    const bool swa = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_STORE_SWA) != 0;
    const int head_slices_param = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_HEAD_SLICES);
    const int head_slices = head_slices_param > 0 ? head_slices_param : 1;
    const int stage_groups = kvarn_resolve_stage_groups(dst);
    const int tail_groups = kvarn_resolve_tail_groups(dst, stage_groups);
    const bool eager_records = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_EAGER_RECORDS) != 0;
    GGML_ASSERT(ggml_cuda_kvarn_valid_bits(bits));
    GGML_ASSERT(head_slices == 1 || head_slices == 2 || head_slices == 4);
    GGML_ASSERT((KVAR_N_TILE_VALUES * bits) % 8 == 0);
    GGML_ASSERT((KVAR_N_DIM * bits) % 8 == 0);
    GGML_ASSERT(stage_groups >= 2);
    GGML_ASSERT(tail_groups >= 1 && tail_groups <= stage_groups);
    GGML_ASSERT(stage->ne[2] % (KVAR_N_DIM * stage_groups) == 0);
    const int n_stream = (int) (stage->ne[2] / (KVAR_N_DIM * stage_groups));
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(records->ne[2] % n_stream == 0);
    const int groups_per_stream = (int) (records->ne[2] / n_stream);
    if (swa) {
        GGML_ASSERT(n_stream == 1 && "SWA KVarN ring requires a single stream");
    }
    size_t smpbo = ggml_cuda_info().devices[ctx.device].smpbo;
    if (std::getenv("GGML_KVARN_TEST_FORCE_LOWSHMEM") != nullptr) {
        smpbo = std::min(smpbo, (size_t) KVAR_N_LOWSHMEM_BYTES);
    }
    cudaStream_t stream = ctx.stream();
    const int n_heads = (int) current->ne[1];
    GGML_ASSERT(n_heads % head_slices == 0);
    const int n_tokens = (int) current->ne[2];
    const size_t staged_bytes = (size_t) current->ne[0] * (size_t) current->ne[1] * (size_t) current->ne[2] * sizeof(half);

    const bool hint_well_formed =
        tokens_per_stream_hint > 0 &&
        tokens_per_stream_hint <= n_tokens &&
        n_tokens % tokens_per_stream_hint == 0;
    const int active_streams = hint_well_formed ? n_tokens / tokens_per_stream_hint : 0;
    const bool workspace_hint =
        hint_well_formed &&
        tokens_per_stream_hint >= 3 * KVAR_N_DIM &&
        (!swa || (n_stream == 1 && tokens_per_stream_hint <= groups_per_stream * KVAR_N_DIM));
    // The direct split is safe for the common ubatch=256 two-group case only
    // when there are at least three transient tail groups. With fewer tail
    // groups, an unaligned 256-token write can span far enough that the
    // pre-stage flush would read a group that belongs to the current write.
    const bool direct_hint =
        hint_well_formed &&
        tokens_per_stream_hint <= 2 * KVAR_N_DIM &&
        tail_groups >= 3;
    const int flush_candidates = workspace_hint ? (tokens_per_stream_hint + KVAR_N_DIM - 1) / KVAR_N_DIM + stage_groups : 0;
    const bool grid_fits = n_tokens <= 65535 && active_streams * flush_candidates <= 65535;
    const bool use_workspace =
        smpbo >= KVAR_N_SHARED_BYTES &&
        workspace_hint &&
        active_streams > 0 &&
        active_streams <= n_stream &&
        grid_fits;
    const bool use_direct =
        smpbo >= KVAR_N_SHARED_BYTES &&
        head_slices == 1 &&
        direct_hint &&
        active_streams > 0 &&
        active_streams <= n_stream &&
        n_tokens <= 65535 &&
        !eager_records;
    if (head_slices > 1 && use_workspace) {
        g_kvarn_store_headwide_workspace.fetch_add(1, std::memory_order_relaxed);
#if defined(GGML_USE_HIP)
        CUDA_CHECK(hipFuncSetAttribute(
            reinterpret_cast<const void *>(&kvarn_store_workspace_flush_kernel),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            KVAR_N_SHARED_BYTES));
        CUDA_CHECK(hipFuncSetAttribute(
            reinterpret_cast<const void *>(&kvarn_store_kernel_headwide),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            KVAR_N_SHARED_BYTES));
#elif !defined(GGML_USE_MUSA)
        CUDA_CHECK(cudaFuncSetAttribute(
            kvarn_store_workspace_flush_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            KVAR_N_SHARED_BYTES));
        CUDA_CHECK(cudaFuncSetAttribute(
            kvarn_store_kernel_headwide,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            KVAR_N_SHARED_BYTES));
#endif
        ggml_cuda_pool_alloc<half> workspace(ctx.pool(), (size_t) n_tokens * (size_t) n_heads * KVAR_N_DIM);
        ggml_cuda_pool_alloc<int> workspace_valid(ctx.pool(), 1);
        auto prof = kvarn_prof_begin(ctx, stream, kvarn_prof_kind::STORE_HI, value, bits, n_tokens, staged_bytes);
        kvarn_store_workspace_validate_kernel<<<1, 256, 0, stream>>>(
            (const int64_t *) indices->data,
            n_tokens,
            n_stream,
            groups_per_stream,
            tokens_per_stream_hint,
            active_streams,
            swa,
            workspace_valid.get());
        dim3 blocks_stage(n_heads / head_slices, (n_tokens + KVAR_N_STAGE_CHUNK - 1) / KVAR_N_STAGE_CHUNK, 1);
        switch (head_slices) {
            case 2:
                kvarn_store_workspace_stage_headwide_kernel<2><<<blocks_stage, KVAR_N_DIM * KVAR_N_STAGE_CHUNK, 0, stream>>>(
                    (const float *) current->data,
                    workspace.get(),
                    n_heads,
                    n_tokens,
                    value);
                break;
            case 4:
                kvarn_store_workspace_stage_headwide_kernel<4><<<blocks_stage, KVAR_N_DIM * KVAR_N_STAGE_CHUNK, 0, stream>>>(
                    (const float *) current->data,
                    workspace.get(),
                    n_heads,
                    n_tokens,
                    value);
                break;
            default:
                GGML_ABORT("unsupported KVarN head_slices");
        }
        dim3 blocks_flush(n_heads, active_streams * flush_candidates, 1);
        kvarn_store_workspace_flush_kernel<<<blocks_flush, KVAR_N_DIM, KVAR_N_SHARED_BYTES, stream>>>(
            (const int64_t *) indices->data,
            (const half *) stage->data,
            workspace.get(),
            (uint8_t *) records->data,
            workspace_valid.get(),
            n_heads,
            n_tokens,
            n_stream,
            groups_per_stream,
            (int) records->ne[0],
            tokens_per_stream_hint,
            flush_candidates,
            bits,
            iterations,
            value,
            stage_groups,
            tail_groups,
            swa,
            eager_records);
        dim3 blocks_commit(n_heads, active_streams * KVAR_N_DIM * stage_groups, 1);
        kvarn_store_workspace_commit_kernel<<<blocks_commit, KVAR_N_DIM, 0, stream>>>(
            (const int64_t *) indices->data,
            workspace.get(),
            (half *) stage->data,
            workspace_valid.get(),
            n_heads,
            n_tokens,
            n_stream,
            groups_per_stream,
            tokens_per_stream_hint,
            stage_groups,
            tail_groups,
            swa);
        kvarn_store_kernel_headwide<<<n_heads / head_slices, KVAR_N_DIM, KVAR_N_SHARED_BYTES, stream>>>(
            (const float *) current->data,
            (const int64_t *) indices->data,
            (half *) stage->data,
            (uint8_t *) records->data,
            n_heads,
            n_tokens,
            n_stream,
            groups_per_stream,
            (int) records->ne[0],
            bits,
            iterations,
            value,
            swa,
            stage_groups,
            tail_groups,
            head_slices,
            eager_records,
            workspace_valid.get());
        kvarn_prof_end(prof, stream);
        return;
    }
    if (head_slices > 1 && smpbo >= KVAR_N_SHARED_BYTES) {
        g_kvarn_store_headwide_monolithic.fetch_add(1, std::memory_order_relaxed);
#if defined(GGML_USE_HIP)
        CUDA_CHECK(hipFuncSetAttribute(
            reinterpret_cast<const void *>(&kvarn_store_kernel_headwide),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            KVAR_N_SHARED_BYTES));
#elif !defined(GGML_USE_MUSA)
        CUDA_CHECK(cudaFuncSetAttribute(
            kvarn_store_kernel_headwide,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            KVAR_N_SHARED_BYTES));
#endif
        auto prof = kvarn_prof_begin(ctx, stream, kvarn_prof_kind::STORE_HI, value, bits, (int) current->ne[2], staged_bytes);
        kvarn_store_kernel_headwide<<<n_heads / head_slices, KVAR_N_DIM, KVAR_N_SHARED_BYTES, stream>>>(
            (const float *) current->data,
            (const int64_t *) indices->data,
            (half *) stage->data,
            (uint8_t *) records->data,
            n_heads,
            n_tokens,
            n_stream,
            groups_per_stream,
            (int) records->ne[0],
            bits,
            iterations,
            value,
            swa,
            stage_groups,
            tail_groups,
            head_slices,
            eager_records,
            nullptr);
        kvarn_prof_end(prof, stream);
        return;
    }

    if (use_workspace) {
        g_kvarn_store_single_slice_workspace.fetch_add(1, std::memory_order_relaxed);
#if defined(GGML_USE_HIP)
        CUDA_CHECK(hipFuncSetAttribute(
            reinterpret_cast<const void *>(&kvarn_store_workspace_flush_kernel),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            KVAR_N_SHARED_BYTES));
        CUDA_CHECK(hipFuncSetAttribute(
            reinterpret_cast<const void *>(&kvarn_store_kernel_hishmem),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            KVAR_N_SHARED_BYTES));
#elif !defined(GGML_USE_MUSA)
        CUDA_CHECK(cudaFuncSetAttribute(
            kvarn_store_workspace_flush_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            KVAR_N_SHARED_BYTES));
        CUDA_CHECK(cudaFuncSetAttribute(
            kvarn_store_kernel_hishmem,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            KVAR_N_SHARED_BYTES));
#endif
        ggml_cuda_pool_alloc<half> workspace(ctx.pool(), (size_t) n_tokens * (size_t) n_heads * KVAR_N_DIM);
        ggml_cuda_pool_alloc<int> workspace_valid(ctx.pool(), 1);
        auto prof = kvarn_prof_begin(ctx, stream, kvarn_prof_kind::STORE_HI, value, bits, n_tokens, staged_bytes);
        kvarn_store_workspace_validate_kernel<<<1, 256, 0, stream>>>(
            (const int64_t *) indices->data,
            n_tokens,
            n_stream,
            groups_per_stream,
            tokens_per_stream_hint,
            active_streams,
            swa,
            workspace_valid.get());
        dim3 blocks_stage(n_heads, (n_tokens + KVAR_N_STAGE_CHUNK - 1) / KVAR_N_STAGE_CHUNK, 1);
        kvarn_store_workspace_stage_kernel<<<blocks_stage, KVAR_N_DIM * KVAR_N_STAGE_CHUNK, 0, stream>>>(
            (const float *) current->data,
            workspace.get(),
            n_heads,
            n_tokens,
            value);
        dim3 blocks_flush(n_heads, active_streams * flush_candidates, 1);
        kvarn_store_workspace_flush_kernel<<<blocks_flush, KVAR_N_DIM, KVAR_N_SHARED_BYTES, stream>>>(
            (const int64_t *) indices->data,
            (const half *) stage->data,
            workspace.get(),
            (uint8_t *) records->data,
            workspace_valid.get(),
            n_heads,
            n_tokens,
            n_stream,
            groups_per_stream,
            (int) records->ne[0],
            tokens_per_stream_hint,
            flush_candidates,
            bits,
            iterations,
            value,
            stage_groups,
            tail_groups,
            swa,
            eager_records);
        dim3 blocks_commit(n_heads, active_streams * KVAR_N_DIM * stage_groups, 1);
        kvarn_store_workspace_commit_kernel<<<blocks_commit, KVAR_N_DIM, 0, stream>>>(
            (const int64_t *) indices->data,
            workspace.get(),
            (half *) stage->data,
            workspace_valid.get(),
            n_heads,
            n_tokens,
            n_stream,
            groups_per_stream,
            tokens_per_stream_hint,
            stage_groups,
            tail_groups,
            swa);
        kvarn_store_kernel_hishmem<<<n_heads, KVAR_N_DIM, KVAR_N_SHARED_BYTES, stream>>>(
            (const float *) current->data,
            (const int64_t *) indices->data,
            (half *) stage->data,
            (uint8_t *) records->data,
            n_heads,
            n_tokens,
            n_stream,
            groups_per_stream,
            (int) records->ne[0],
            bits,
            iterations,
            value,
            swa,
            stage_groups,
            tail_groups,
            eager_records,
            workspace_valid.get());
        kvarn_prof_end(prof, stream);
        return;
    }

    if (use_direct) {
        g_kvarn_store_direct_store.fetch_add(1, std::memory_order_relaxed);
#if defined(GGML_USE_HIP)
        CUDA_CHECK(hipFuncSetAttribute(
            reinterpret_cast<const void *>(&kvarn_store_direct_flush_kernel),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            KVAR_N_SHARED_BYTES));
#elif !defined(GGML_USE_MUSA)
        CUDA_CHECK(cudaFuncSetAttribute(
            kvarn_store_direct_flush_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            KVAR_N_SHARED_BYTES));
#endif
        auto prof = kvarn_prof_begin(ctx, stream, kvarn_prof_kind::STORE_HI, value, bits, n_tokens, staged_bytes);
        dim3 blocks_flush(n_heads, n_tokens, 1);
        kvarn_store_direct_flush_kernel<<<blocks_flush, KVAR_N_DIM, KVAR_N_SHARED_BYTES, stream>>>(
            (const int64_t *) indices->data,
            (const half *) stage->data,
            (uint8_t *) records->data,
            n_heads,
            n_tokens,
            n_stream,
            groups_per_stream,
            (int) records->ne[0],
            bits,
            iterations,
            value,
            swa,
            stage_groups,
            tail_groups);
        dim3 blocks_stage(n_heads, (n_tokens + KVAR_N_STAGE_CHUNK - 1) / KVAR_N_STAGE_CHUNK, 1);
        kvarn_store_direct_stage_kernel<<<blocks_stage, KVAR_N_DIM * KVAR_N_STAGE_CHUNK, 0, stream>>>(
            (const float *) current->data,
            (const int64_t *) indices->data,
            (half *) stage->data,
            n_heads,
            n_tokens,
            n_stream,
            groups_per_stream,
            stage_groups,
            tail_groups,
            value,
            swa);
        kvarn_prof_end(prof, stream);
        return;
    }

    if (smpbo >= KVAR_N_SHARED_BYTES) {
        g_kvarn_store_high_shared_fallback.fetch_add(1, std::memory_order_relaxed);
#if defined(GGML_USE_HIP)
        CUDA_CHECK(hipFuncSetAttribute(
            reinterpret_cast<const void *>(&kvarn_store_kernel_hishmem),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            KVAR_N_SHARED_BYTES));
#elif !defined(GGML_USE_MUSA)
        CUDA_CHECK(cudaFuncSetAttribute(
            kvarn_store_kernel_hishmem,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            KVAR_N_SHARED_BYTES));
#endif
        // CUDA graph caveat: these host-side event records do not run for graph replays.
        // Use GGML_CUDA_DISABLE_GRAPHS=1 on profiling legs when every launch must be counted.
        auto prof = kvarn_prof_begin(ctx, stream, kvarn_prof_kind::STORE_HI, value, bits, (int) current->ne[2], staged_bytes);
        kvarn_store_kernel_hishmem<<<n_heads, KVAR_N_DIM, KVAR_N_SHARED_BYTES, stream>>>(
            (const float *) current->data,
            (const int64_t *) indices->data,
            (half *) stage->data,
            (uint8_t *) records->data,
            n_heads,
            n_tokens,
            n_stream,
            groups_per_stream,
            (int) records->ne[0],
            bits,
            iterations,
            value,
            swa,
            stage_groups,
            tail_groups,
            eager_records,
            nullptr);
        kvarn_prof_end(prof, stream);
    } else {
        g_kvarn_store_low_shared_store.fetch_add(1, std::memory_order_relaxed);
        GGML_ASSERT(smpbo >= KVAR_N_LOWSHMEM_BYTES);
        auto prof = kvarn_prof_begin(ctx, stream, kvarn_prof_kind::STORE_LOW, value, bits, (int) current->ne[2], staged_bytes);
        kvarn_store_kernel_lowshmem<<<n_heads / head_slices, KVAR_N_DIM, KVAR_N_LOWSHMEM_BYTES, stream>>>(
            (const float *) current->data,
            (const int64_t *) indices->data,
            (half *) stage->data,
            (uint8_t *) records->data,
            n_heads,
            n_tokens,
            n_stream,
            groups_per_stream,
            (int) records->ne[0],
            bits,
            iterations,
            value,
            swa,
            stage_groups,
            tail_groups,
            head_slices,
            eager_records);
        kvarn_prof_end(prof, stream);
    }
}

static __global__ void kvarn_materialize_live_kernel(
        const int64_t * indices,
        int n_indices,
        int64_t * live,
        int stream_start,
        int n_stream,
        int groups_per_stream,
        bool swa) {
    const int out_stream = blockIdx.x;
    if (out_stream >= n_stream || threadIdx.x != 0) {
        return;
    }
    int64_t live_group = 0;
    int64_t live_pos = 0;
    const int stream = stream_start + out_stream;
    for (int i = 0; i < n_indices; ++i) {
        const int64_t idx = indices[i];
        if (idx < 0) {
            continue;
        }
        const int64_t group_global = idx / KVAR_N_DIM;
        const int idx_stream = swa ? stream : int(group_global / groups_per_stream);
        if (idx_stream != stream) {
            continue;
        }
        const int64_t group = swa ? group_global : group_global - int64_t(stream) * groups_per_stream;
        const int64_t pos = idx % KVAR_N_DIM;
        if (group > live_group || (group == live_group && pos > live_pos)) {
            live_group = group;
            live_pos = pos;
        }
    }
    live[2*out_stream + 0] = live_group;
    live[2*out_stream + 1] = live_pos;
}

static __device__ __forceinline__ float kvarn_materialize_record_value(
        const uint8_t * record, int bits, bool value, int token, int dim) {
    const int payload_bytes = KVAR_N_TILE_VALUES * bits / 8;
    const int row = value ? token : dim;
    const int col = value ? dim : token;
    const size_t bit_offset = size_t(row * KVAR_N_DIM + col) * size_t(bits);
    uint8_t q = 0;
    for (int bit = 0; bit < bits; ++bit) {
        const size_t src_bit = bit_offset + size_t(bit);
        q |= uint8_t(((record[src_bit / 8] >> (src_bit % 8)) & 1u) << bit);
    }
    const half * scale_axis = reinterpret_cast<const half *>(record + payload_bytes);
    const half * zp_axis = scale_axis + KVAR_N_DIM;
    const half * other_axis = zp_axis + KVAR_N_DIM;
    return (float(q) * __half2float(scale_axis[row]) + __half2float(zp_axis[row])) *
        __half2float(other_axis[col]);
}

static __global__ void kvarn_materialize_kernel(
        const uint8_t * records,
        const half * stage,
        const int64_t * indices,
        const int64_t * live,
        half * output,
        int n_heads,
        int n_kv,
        int stream_start,
        int n_stream,
        int groups_per_stream,
        int record_bytes,
        int bits,
        bool value,
        bool emit_rotated,
        bool swa,
        int head_slices,
        int stage_groups,
        int tail_groups,
        bool eager_records) {
    const int cell = blockIdx.x;
    const int logical_head = blockIdx.y;
    const int out_stream = blockIdx.z;
    const int dim = threadIdx.x;
    if (cell >= n_kv || out_stream >= n_stream || dim >= KVAR_N_DIM) {
        return;
    }
    extern __shared__ float shared_rows[];
    float values[4] = {};
    const int64_t abs_pos = swa ? indices[cell] : cell;
    if (abs_pos >= 0) {
        const int64_t group = abs_pos / KVAR_N_DIM;
        const int64_t pos = abs_pos % KVAR_N_DIM;
        const int64_t live_group = live[2*out_stream + 0];
        const int64_t live_pos = live[2*out_stream + 1];
        const int stream = stream_start + out_stream;
        const int64_t stage_base = int64_t(stream) * KVAR_N_DIM * stage_groups;
        bool from_stage = false;
        bool from_record = false;
        int64_t stage_pos = 0;
        int64_t record_group = 0;
        if (eager_records) {
            from_stage = (!swa && group == 0) || (group == live_group && live_pos < KVAR_N_DIM - 1);
            const bool completed = group < live_group || (group == live_group && live_pos == KVAR_N_DIM - 1);
            from_record = !from_stage && completed && (swa ?
                (group >= 0 && live_group - group < groups_per_stream) :
                (group > 0 && group < groups_per_stream));
            stage_pos = stage_base + (swa ? group % stage_groups :
                (group == 0 ? 0 : 1 + ((group - 1) % tail_groups))) * KVAR_N_DIM + pos;
            record_group = int64_t(stream) * groups_per_stream + (swa ? group % groups_per_stream : group);
        } else if (swa) {
            const int64_t stage_begin = live_group >= tail_groups - 1 ? live_group - (tail_groups - 1) : 0;
            from_stage = group >= stage_begin && group <= live_group;
            from_record = !from_stage && group >= 0 && group < stage_begin &&
                live_group - group < groups_per_stream + tail_groups;
            stage_pos = stage_base + (group % stage_groups) * KVAR_N_DIM + pos;
            record_group = int64_t(stream) * groups_per_stream + group % groups_per_stream;
        } else {
            from_stage = group == 0 || (group > 0 && group <= live_group &&
                group + (tail_groups - 1) >= live_group);
            from_record = !from_stage && group < live_group && group < groups_per_stream;
            stage_pos = stage_base + (group == 0 ? pos :
                KVAR_N_DIM + ((group - 1) % tail_groups) * KVAR_N_DIM + pos);
            record_group = int64_t(stream) * groups_per_stream + group;
        }
        for (int slice = 0; slice < head_slices; ++slice) {
            const int h = logical_head * head_slices + slice;
            if (from_stage) {
                values[slice] = __half2float(stage[(stage_pos * n_heads + h) * KVAR_N_DIM + dim]);
            } else if (from_record) {
                const uint8_t * record = records + (record_group * n_heads + h) * record_bytes;
                values[slice] = kvarn_materialize_record_value(record, bits, value, int(pos), dim);
            }
        }
    }
    if (!emit_rotated) {
        for (int slice = 0; slice < head_slices; ++slice) {
            shared_rows[slice * KVAR_N_DIM + dim] = values[slice];
        }
        __syncthreads();
        for (int slice = 0; slice < head_slices; ++slice) {
            kvarn_wht_128_lane(shared_rows + slice * KVAR_N_DIM, dim);
            values[slice] = shared_rows[slice * KVAR_N_DIM + dim];
        }
        kvarn_wht_cross_slices(values, head_slices);
    }
    for (int slice = 0; slice < head_slices; ++slice) {
        const int h = logical_head * head_slices + slice;
        output[((int64_t(out_stream) * n_kv + cell) * n_heads + h) * KVAR_N_DIM + dim] =
            __float2half_rn(values[slice]);
    }
}

void ggml_cuda_op_kvarn_materialize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * records = dst->src[0];
    const ggml_tensor * stage = dst->src[1];
    const ggml_tensor * indices = dst->src[2];
    GGML_ASSERT(ggml_is_contiguous(records) && ggml_is_contiguous(stage) &&
        ggml_is_contiguous(indices) && ggml_is_contiguous(dst));
    const int bits = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_BITS);
    const bool value = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_MAT_VALUE) != 0;
    const int stream_start = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_MAT_STREAM_START);
    const int n_stream = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_MAT_N_STREAM);
    const bool emit_rotated = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_MAT_EMIT_ROTATED) != 0;
    const bool swa = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_MAT_SWA) != 0;
    const int head_slices_param = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_HEAD_SLICES);
    const int head_slices = head_slices_param > 0 ? head_slices_param : 1;
    const int stage_groups = kvarn_resolve_stage_groups(dst);
    const int tail_groups = kvarn_resolve_tail_groups(dst, stage_groups);
    const bool eager_records = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_EAGER_RECORDS) != 0;
    const int n_total_stream = int(stage->ne[2] / (KVAR_N_DIM * stage_groups));
    const int groups_per_stream = int(records->ne[2] / n_total_stream);
    const int n_heads = int(dst->ne[1]);
    const int n_kv = int(dst->ne[2]);
    GGML_ASSERT(n_heads % head_slices == 0 && stream_start + n_stream <= n_total_stream);
    ggml_cuda_pool_alloc<int64_t> live(ctx.pool(), size_t(2 * n_stream));
    cudaStream_t stream = ctx.stream();
    kvarn_materialize_live_kernel<<<n_stream, 1, 0, stream>>>(
        (const int64_t *) indices->data, int(indices->ne[0]), live.get(), stream_start,
        n_stream, groups_per_stream, swa);
    const dim3 blocks(n_kv, n_heads / head_slices, n_stream);
    kvarn_materialize_kernel<<<blocks, KVAR_N_DIM, size_t(head_slices) * KVAR_N_DIM * sizeof(float), stream>>>(
        (const uint8_t *) records->data, (const half *) stage->data, (const int64_t *) indices->data,
        live.get(), (half *) dst->data, n_heads, n_kv, stream_start, n_stream,
        groups_per_stream, int(records->ne[0]), bits, value, emit_rotated, swa,
        head_slices, stage_groups, tail_groups, eager_records);
}
