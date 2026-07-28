#include "fattn-kvarn-dispatch.cuh"
#include "fattn-kvarn-route-policy.h"

#include "fattn-common.cuh"
#include "fattn-kvarn-vec-decl.cuh"
#include "fattn-mma-kvarn-case-decl.cuh"
#include "fattn-mma-kvarn-decode-decl.cuh"
#include "fattn-mma-kvarn.cuh"
#include "fattn-kvarn-portable.cuh"
#include "kvarn.cuh"

#include <cstdio>
#include <cstdlib>
#include <atomic>

static std::atomic<uint64_t> g_kvarn_route_decode_split{0};
static std::atomic<uint64_t> g_kvarn_route_decode_vector{0};
static std::atomic<uint64_t> g_kvarn_route_generic_mma{0};
static std::atomic<uint64_t> g_kvarn_route_prompt_prefill{0};
static std::atomic<uint64_t> g_kvarn_route_portable_native{0};
static std::atomic<uint64_t> g_kvarn_route_amd_generic_mma{0};
static std::atomic<uint64_t> g_kvarn_route_amd_decode_split{0};
static std::atomic<uint64_t> g_kvarn_route_amd_decode_vector{0};
static std::atomic<uint64_t> g_kvarn_route_materialize_fallback{0};
static std::atomic<uint64_t> g_kvarn_route_split_reduce{0};
static std::atomic<uint64_t> g_kvarn_route_direct_entry{0};
static std::atomic<uint64_t> g_kvarn_route_compact_tail_entry{0};

static std::atomic<uint64_t> g_kv_mem_kvarn_descriptor{0};
static std::atomic<uint64_t> g_kv_mem_kvarn_partial_output{0};
static std::atomic<uint64_t> g_kv_mem_kvarn_partial_meta{0};
static std::atomic<uint64_t> g_kv_mem_kvarn_total{0};
static std::atomic<uint64_t> g_kv_mem_tail_body_meta{0};
static std::atomic<uint64_t> g_kv_mem_tail_exact_meta{0};
static std::atomic<uint64_t> g_kv_mem_tail_pack{0};
static std::atomic<uint64_t> g_kv_mem_tail_body_output{0};
static std::atomic<uint64_t> g_kv_mem_tail_exact_output{0};
static std::atomic<uint64_t> g_kv_mem_tail_plan_input{0};
static std::atomic<uint64_t> g_kv_mem_tail_total{0};
static std::atomic<bool> g_kv_mem_stats_enabled{false};

ggml_cuda_fattn_kvarn_capabilities ggml_cuda_fattn_kvarn_device_capabilities(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return {};
    }
    const auto & device_info = ggml_cuda_info().devices[device];
#if defined(GGML_USE_MUSA)
    constexpr ggml_cuda_fattn_kvarn_backend backend = GGML_CUDA_FATTN_KVARN_BACKEND_MUSA;
    const bool matrix_mma = false;
#elif defined(GGML_USE_HIP)
    constexpr ggml_cuda_fattn_kvarn_backend backend = GGML_CUDA_FATTN_KVARN_BACKEND_HIP;
    const bool matrix_mma = amd_wmma_available(device_info.cc) || amd_mfma_available(device_info.cc);
#else
    constexpr ggml_cuda_fattn_kvarn_backend backend = GGML_CUDA_FATTN_KVARN_BACKEND_CUDA;
    const char * force_portable_capability =
        getenv("GGML_KVARN_TEST_FORCE_PORTABLE_CAPABILITY");
    const bool matrix_mma =
        turing_mma_available(device_info.cc) &&
        !(force_portable_capability != nullptr && atoi(force_portable_capability) != 0);
#endif
#if defined(GGML_CUDA_KVARN)
    constexpr bool kvarn_instances = true;
    const uint64_t minimum_dynamic_shared_bytes = ggml_cuda_kvarn_low_shared_bytes();
#else
    constexpr bool kvarn_instances = false;
    constexpr uint64_t minimum_dynamic_shared_bytes = 0;
#endif
    return ggml_cuda_fattn_kvarn_select_capabilities({
        backend,
        device_info.warp_size,
        matrix_mma,
        kvarn_instances,
        device_info.max_threads_per_block,
        device_info.smpbo,
        minimum_dynamic_shared_bytes,
    });
}

uint32_t ggml_cuda_fattn_kvarn_decode_max_q() {
    return GGML_CUDA_FATTN_KVARN_SPECIALIZED_DECODE_MAX_Q;
}

static void ggml_cuda_atomic_max(std::atomic<uint64_t> & dst, uint64_t value) {
    uint64_t current = dst.load(std::memory_order_relaxed);
    while (current < value && !dst.compare_exchange_weak(
            current, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

void ggml_cuda_fattn_kvarn_route_stats_reset() {
    g_kvarn_route_decode_split.store(0, std::memory_order_relaxed);
    g_kvarn_route_decode_vector.store(0, std::memory_order_relaxed);
    g_kvarn_route_generic_mma.store(0, std::memory_order_relaxed);
    g_kvarn_route_prompt_prefill.store(0, std::memory_order_relaxed);
    g_kvarn_route_portable_native.store(0, std::memory_order_relaxed);
    g_kvarn_route_amd_generic_mma.store(0, std::memory_order_relaxed);
    g_kvarn_route_amd_decode_split.store(0, std::memory_order_relaxed);
    g_kvarn_route_amd_decode_vector.store(0, std::memory_order_relaxed);
    g_kvarn_route_materialize_fallback.store(0, std::memory_order_relaxed);
    g_kvarn_route_split_reduce.store(0, std::memory_order_relaxed);
    g_kvarn_route_direct_entry.store(0, std::memory_order_relaxed);
    g_kvarn_route_compact_tail_entry.store(0, std::memory_order_relaxed);
}

void ggml_cuda_fattn_kvarn_route_stats_get(ggml_cuda_fattn_kvarn_route_stats * stats) {
    if (stats == nullptr ||
            stats->struct_size < sizeof(ggml_cuda_fattn_kvarn_route_stats) ||
            stats->abi_version != GGML_CUDA_FATTN_KVARN_ROUTE_STATS_ABI_VERSION) {
        return;
    }
    const int device = ggml_cuda_get_device();
    const auto capabilities = ggml_cuda_fattn_kvarn_device_capabilities(device);
    stats->struct_size = sizeof(*stats);
    stats->abi_version = GGML_CUDA_FATTN_KVARN_ROUTE_STATS_ABI_VERSION;
    stats->route_families = capabilities.route_families;
    stats->reserved = 0;
    stats->decode_split = g_kvarn_route_decode_split.load(std::memory_order_relaxed);
    stats->decode_vector = g_kvarn_route_decode_vector.load(std::memory_order_relaxed);
    stats->generic_mma = g_kvarn_route_generic_mma.load(std::memory_order_relaxed);
    stats->prompt_prefill = g_kvarn_route_prompt_prefill.load(std::memory_order_relaxed);
    stats->portable_native = g_kvarn_route_portable_native.load(std::memory_order_relaxed);
    stats->amd_generic_mma = g_kvarn_route_amd_generic_mma.load(std::memory_order_relaxed);
    stats->amd_decode_split = g_kvarn_route_amd_decode_split.load(std::memory_order_relaxed);
    stats->amd_decode_vector = g_kvarn_route_amd_decode_vector.load(std::memory_order_relaxed);
    stats->materialize_fallback = g_kvarn_route_materialize_fallback.load(std::memory_order_relaxed);
    stats->split_reduce = g_kvarn_route_split_reduce.load(std::memory_order_relaxed);
    stats->direct_entry = g_kvarn_route_direct_entry.load(std::memory_order_relaxed);
    stats->compact_tail_entry = g_kvarn_route_compact_tail_entry.load(std::memory_order_relaxed);
}

void ggml_cuda_kv_memory_transient_stats_reset() {
    g_kv_mem_stats_enabled.store(false, std::memory_order_relaxed);
    g_kv_mem_kvarn_descriptor.store(0, std::memory_order_relaxed);
    g_kv_mem_kvarn_partial_output.store(0, std::memory_order_relaxed);
    g_kv_mem_kvarn_partial_meta.store(0, std::memory_order_relaxed);
    g_kv_mem_kvarn_total.store(0, std::memory_order_relaxed);
    g_kv_mem_tail_body_meta.store(0, std::memory_order_relaxed);
    g_kv_mem_tail_exact_meta.store(0, std::memory_order_relaxed);
    g_kv_mem_tail_pack.store(0, std::memory_order_relaxed);
    g_kv_mem_tail_body_output.store(0, std::memory_order_relaxed);
    g_kv_mem_tail_exact_output.store(0, std::memory_order_relaxed);
    g_kv_mem_tail_plan_input.store(0, std::memory_order_relaxed);
    g_kv_mem_tail_total.store(0, std::memory_order_relaxed);
    g_kv_mem_stats_enabled.store(true, std::memory_order_release);
}

void ggml_cuda_kv_memory_transient_stats_get(ggml_cuda_kv_memory_transient_stats * stats) {
    g_kv_mem_stats_enabled.store(false, std::memory_order_release);
    if (stats == nullptr) {
        return;
    }
    stats->kvarn_descriptor_bytes = g_kv_mem_kvarn_descriptor.load(std::memory_order_relaxed);
    stats->kvarn_partial_output_bytes = g_kv_mem_kvarn_partial_output.load(std::memory_order_relaxed);
    stats->kvarn_partial_meta_bytes = g_kv_mem_kvarn_partial_meta.load(std::memory_order_relaxed);
    stats->kvarn_total_bytes = g_kv_mem_kvarn_total.load(std::memory_order_relaxed);
    stats->tail_body_meta_bytes = g_kv_mem_tail_body_meta.load(std::memory_order_relaxed);
    stats->tail_exact_meta_bytes = g_kv_mem_tail_exact_meta.load(std::memory_order_relaxed);
    stats->tail_pack_bytes = g_kv_mem_tail_pack.load(std::memory_order_relaxed);
    stats->tail_body_output_bytes = g_kv_mem_tail_body_output.load(std::memory_order_relaxed);
    stats->tail_exact_output_bytes = g_kv_mem_tail_exact_output.load(std::memory_order_relaxed);
    stats->tail_plan_input_bytes = g_kv_mem_tail_plan_input.load(std::memory_order_relaxed);
    stats->tail_total_bytes = g_kv_mem_tail_total.load(std::memory_order_relaxed);
}

void ggml_cuda_kv_memory_transient_stats_record_kvarn(
        uint64_t descriptor_bytes,
        uint64_t partial_output_bytes,
        uint64_t partial_meta_bytes,
        uint64_t total_bytes) {
    if (!g_kv_mem_stats_enabled.load(std::memory_order_acquire)) {
        return;
    }
    ggml_cuda_atomic_max(g_kv_mem_kvarn_descriptor, descriptor_bytes);
    ggml_cuda_atomic_max(g_kv_mem_kvarn_partial_output, partial_output_bytes);
    ggml_cuda_atomic_max(g_kv_mem_kvarn_partial_meta, partial_meta_bytes);
    ggml_cuda_atomic_max(g_kv_mem_kvarn_total, total_bytes);
}

void ggml_cuda_kv_memory_transient_stats_record_tail(
        uint64_t body_meta_bytes,
        uint64_t exact_meta_bytes,
        uint64_t pack_bytes,
        uint64_t body_output_bytes,
        uint64_t exact_output_bytes,
        uint64_t plan_input_bytes,
        uint64_t total_bytes) {
    if (!g_kv_mem_stats_enabled.load(std::memory_order_acquire)) {
        return;
    }
    ggml_cuda_atomic_max(g_kv_mem_tail_body_meta, body_meta_bytes);
    ggml_cuda_atomic_max(g_kv_mem_tail_exact_meta, exact_meta_bytes);
    ggml_cuda_atomic_max(g_kv_mem_tail_pack, pack_bytes);
    ggml_cuda_atomic_max(g_kv_mem_tail_body_output, body_output_bytes);
    ggml_cuda_atomic_max(g_kv_mem_tail_exact_output, exact_output_bytes);
    ggml_cuda_atomic_max(g_kv_mem_tail_plan_input, plan_input_bytes);
    ggml_cuda_atomic_max(g_kv_mem_tail_total, total_bytes);
}

#if !defined(GGML_CUDA_KVARN)

bool ggml_cuda_flash_attn_ext_kvarn_uses_views(const ggml_tensor * dst) {
    return ggml_cuda_fattn_kvarn_uses_views(dst);
}

bool ggml_cuda_flash_attn_ext_kvarn_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}

bool ggml_cuda_flash_attn_ext_kvarn_portable_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}

bool ggml_cuda_flash_attn_ext_kvarn_direct_tail_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}

bool ggml_cuda_flash_attn_ext_kvarn(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        ggml_cuda_fattn_kvarn_entry_path entry_path) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_UNUSED(entry_path);
    return false;
}

#else

static __device__ __forceinline__ int ggml_cuda_fattn_kvarn_live_index_for_thread(
        const int64_t * indices,
        const int n_indices,
        const int stream,
        const int groups_per_stream,
        const bool swa) {
    int live_index = 0;
    for (int i = threadIdx.x; i < n_indices; i += blockDim.x) {
        const int64_t idx = indices[i];
        if (swa) {
            if (idx >= 0) {
                live_index = max(live_index, (int) idx);
            }
        } else {
            const int group_global = (int) (idx / GGML_CUDA_FATTN_KVARN_DIM);
            const int idx_stream = group_global / groups_per_stream;
            if (idx_stream == stream) {
                const int local_index = (group_global - stream * groups_per_stream) *
                    GGML_CUDA_FATTN_KVARN_DIM + (int) (idx % GGML_CUDA_FATTN_KVARN_DIM);
                live_index = max(live_index, local_index);
            }
        }
    }
    return live_index;
}

static __global__ void ggml_cuda_fattn_kvarn_init_descs_kernel(
        const uint8_t * k_records,
        const half * k_stage,
        const int64_t * k_indices,
        ggml_cuda_fattn_kvarn_desc * k_descs,
        int k_n_indices,
        int k_n_record_heads,
        int k_stream_start,
        int k_groups_per_stream,
        int k_record_bytes,
        int k_stage_groups,
        int k_tail_groups,
        int k_bits,
        bool k_swa,
        bool k_eager_records,
        const uint8_t * v_records,
        const half * v_stage,
        const int64_t * v_indices,
        ggml_cuda_fattn_kvarn_desc * v_descs,
        int v_n_indices,
        int v_n_record_heads,
        int v_stream_start,
        int v_groups_per_stream,
        int v_record_bytes,
        int v_stage_groups,
        int v_tail_groups,
        int v_bits,
        bool v_swa,
        bool v_eager_records,
        int n_stream,
        int n_kv_heads,
        int slices,
        int k_head_slices,
        int v_head_slices,
        int k_original_domain,
        int v_original_domain) {
    const int out_stream = blockIdx.x;
    if (out_stream >= n_stream) {
        return;
    }

    const int k_stream = k_stream_start + out_stream;
    const int v_stream = v_stream_start + out_stream;
    __shared__ int k_partial[GGML_CUDA_FATTN_KVARN_DIM];
    __shared__ int v_partial[GGML_CUDA_FATTN_KVARN_DIM];
    k_partial[threadIdx.x] = ggml_cuda_fattn_kvarn_live_index_for_thread(
        k_indices, k_n_indices, k_stream, k_groups_per_stream, k_swa);
    v_partial[threadIdx.x] = ggml_cuda_fattn_kvarn_live_index_for_thread(
        v_indices, v_n_indices, v_stream, v_groups_per_stream, v_swa);
    __syncthreads();

    for (int stride = GGML_CUDA_FATTN_KVARN_DIM / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            k_partial[threadIdx.x] = max(k_partial[threadIdx.x], k_partial[threadIdx.x + stride]);
            v_partial[threadIdx.x] = max(v_partial[threadIdx.x], v_partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x != 0) {
        return;
    }

    for (int h = 0; h < n_kv_heads; ++h) {
        ggml_cuda_fattn_kvarn_desc & k_desc = k_descs[(size_t) out_stream * n_kv_heads + h];
        k_desc.records = k_records;
        k_desc.stage = k_stage;
        k_desc.indices = k_indices;
        k_desc.n_record_heads = k_n_record_heads;
        k_desc.live_group = k_partial[0] / GGML_CUDA_FATTN_KVARN_DIM;
        k_desc.live_pos = k_partial[0] % GGML_CUDA_FATTN_KVARN_DIM;
        k_desc.stream = k_stream;
        k_desc.head_base = h * slices;
        k_desc.groups_per_stream = k_groups_per_stream;
        k_desc.record_bytes = k_record_bytes;
        k_desc.stage_groups = k_stage_groups;
        k_desc.tail_groups = k_tail_groups;
        k_desc.bits = k_bits;
        k_desc.value = 0;
        k_desc.swa = k_swa ? 1 : 0;
        k_desc.head_slices = k_head_slices;
        k_desc.eager_records = k_eager_records ? 1 : 0;
        k_desc.original_domain = k_original_domain;

        ggml_cuda_fattn_kvarn_desc & v_desc = v_descs[(size_t) out_stream * n_kv_heads + h];
        v_desc.records = v_records;
        v_desc.stage = v_stage;
        v_desc.indices = v_indices;
        v_desc.n_record_heads = v_n_record_heads;
        v_desc.live_group = v_partial[0] / GGML_CUDA_FATTN_KVARN_DIM;
        v_desc.live_pos = v_partial[0] % GGML_CUDA_FATTN_KVARN_DIM;
        v_desc.stream = v_stream;
        v_desc.head_base = h * slices;
        v_desc.groups_per_stream = v_groups_per_stream;
        v_desc.record_bytes = v_record_bytes;
        v_desc.stage_groups = v_stage_groups;
        v_desc.tail_groups = v_tail_groups;
        v_desc.bits = v_bits;
        v_desc.value = 1;
        v_desc.swa = v_swa ? 1 : 0;
        v_desc.head_slices = v_head_slices;
        v_desc.eager_records = v_eager_records ? 1 : 0;
        v_desc.original_domain = v_original_domain;
    }
}

void ggml_cuda_fattn_kvarn_init_descs(
        const ggml_cuda_fattn_kvarn_plan & plan,
        ggml_cuda_fattn_kvarn_desc * k_desc,
        ggml_cuda_fattn_kvarn_desc * v_desc,
        int k_original_domain,
        int v_original_domain,
        cudaStream_t stream) {
    ggml_cuda_fattn_kvarn_init_descs_kernel<<<plan.n_stream, GGML_CUDA_FATTN_KVARN_DIM, 0, stream>>>(
        (const uint8_t *) plan.k.records->data,
        (const half *) plan.k.stage->data,
        (const int64_t *) plan.k.indices->data,
        k_desc,
        (int) plan.k.indices->ne[0],
        (int) plan.k.view->ne[1],
        plan.k.stream_start,
        plan.k.groups_per_stream,
        (int) plan.k.records->ne[0],
        plan.k.stage_groups,
        plan.k.tail_groups,
        plan.k.bits,
        plan.k.swa,
        plan.k.eager_records,
        (const uint8_t *) plan.v.records->data,
        (const half *) plan.v.stage->data,
        (const int64_t *) plan.v.indices->data,
        v_desc,
        (int) plan.v.indices->ne[0],
        (int) plan.v.view->ne[1],
        plan.v.stream_start,
        plan.v.groups_per_stream,
        (int) plan.v.records->ne[0],
        plan.v.stage_groups,
        plan.v.tail_groups,
        plan.v.bits,
        plan.v.swa,
        plan.v.eager_records,
        plan.n_stream,
        plan.n_kv_heads,
        plan.slices,
        plan.k.head_slices,
        plan.v.head_slices,
        k_original_domain,
        v_original_domain);
    CUDA_CHECK(cudaGetLastError());
}

static inline bool ggml_cuda_fattn_kvarn_fast_decode_pair_enabled(int k_bits, int v_bits) {
#if defined(GGML_CUDA_FA_ALL_QUANTS)
    return ggml_cuda_fattn_kvarn_valid_bits(k_bits) && ggml_cuda_fattn_kvarn_valid_bits(v_bits);
#else
    switch (k_bits) {
        case 8: return v_bits == 8 || v_bits == 6 || v_bits == 5;
        case 6: return v_bits == 6 || v_bits == 5 || v_bits == 4;
        case 5: return v_bits == 5 || v_bits == 4 || v_bits == 3;
        case 4: return v_bits == 4 || v_bits == 3 || v_bits == 2;
        case 3: return v_bits == 3 || v_bits == 2;
        case 2: return v_bits == 2;
        default: return false;
    }
#endif
}

// Keep template references in sync with the decoder sources selected by CMake.
#if defined(GGML_CUDA_FA_ALL_QUANTS)
#define GGML_CUDA_FATTN_KVARN_FAST_DECODE_DISPATCH_K(DISPATCH_PAIR) \
    do { \
        switch (plan.k.bits) { \
            case 8: switch (plan.v.bits) { \
                case 8: DISPATCH_PAIR(8, 8); break; case 6: DISPATCH_PAIR(8, 6); break; \
                case 5: DISPATCH_PAIR(8, 5); break; case 4: DISPATCH_PAIR(8, 4); break; \
                case 3: DISPATCH_PAIR(8, 3); break; case 2: DISPATCH_PAIR(8, 2); break; default: return false; } break; \
            case 6: switch (plan.v.bits) { \
                case 8: DISPATCH_PAIR(6, 8); break; case 6: DISPATCH_PAIR(6, 6); break; \
                case 5: DISPATCH_PAIR(6, 5); break; case 4: DISPATCH_PAIR(6, 4); break; \
                case 3: DISPATCH_PAIR(6, 3); break; case 2: DISPATCH_PAIR(6, 2); break; default: return false; } break; \
            case 5: switch (plan.v.bits) { \
                case 8: DISPATCH_PAIR(5, 8); break; case 6: DISPATCH_PAIR(5, 6); break; \
                case 5: DISPATCH_PAIR(5, 5); break; case 4: DISPATCH_PAIR(5, 4); break; \
                case 3: DISPATCH_PAIR(5, 3); break; case 2: DISPATCH_PAIR(5, 2); break; default: return false; } break; \
            case 4: switch (plan.v.bits) { \
                case 8: DISPATCH_PAIR(4, 8); break; case 6: DISPATCH_PAIR(4, 6); break; \
                case 5: DISPATCH_PAIR(4, 5); break; case 4: DISPATCH_PAIR(4, 4); break; \
                case 3: DISPATCH_PAIR(4, 3); break; case 2: DISPATCH_PAIR(4, 2); break; default: return false; } break; \
            case 3: switch (plan.v.bits) { \
                case 8: DISPATCH_PAIR(3, 8); break; case 6: DISPATCH_PAIR(3, 6); break; \
                case 5: DISPATCH_PAIR(3, 5); break; case 4: DISPATCH_PAIR(3, 4); break; \
                case 3: DISPATCH_PAIR(3, 3); break; case 2: DISPATCH_PAIR(3, 2); break; default: return false; } break; \
            case 2: switch (plan.v.bits) { \
                case 8: DISPATCH_PAIR(2, 8); break; case 6: DISPATCH_PAIR(2, 6); break; \
                case 5: DISPATCH_PAIR(2, 5); break; case 4: DISPATCH_PAIR(2, 4); break; \
                case 3: DISPATCH_PAIR(2, 3); break; case 2: DISPATCH_PAIR(2, 2); break; default: return false; } break; \
            default: return false; \
        } \
    } while (0)
#else
#define GGML_CUDA_FATTN_KVARN_FAST_DECODE_DISPATCH_K(DISPATCH_PAIR) \
    do { \
        switch (plan.k.bits) { \
            case 8: switch (plan.v.bits) { \
                case 8: DISPATCH_PAIR(8, 8); break; case 6: DISPATCH_PAIR(8, 6); break; \
                case 5: DISPATCH_PAIR(8, 5); break; default: return false; } break; \
            case 6: switch (plan.v.bits) { \
                case 6: DISPATCH_PAIR(6, 6); break; case 5: DISPATCH_PAIR(6, 5); break; \
                case 4: DISPATCH_PAIR(6, 4); break; default: return false; } break; \
            case 5: switch (plan.v.bits) { \
                case 5: DISPATCH_PAIR(5, 5); break; case 4: DISPATCH_PAIR(5, 4); break; \
                case 3: DISPATCH_PAIR(5, 3); break; default: return false; } break; \
            case 4: switch (plan.v.bits) { \
                case 4: DISPATCH_PAIR(4, 4); break; case 3: DISPATCH_PAIR(4, 3); break; \
                case 2: DISPATCH_PAIR(4, 2); break; default: return false; } break; \
            case 3: switch (plan.v.bits) { \
                case 3: DISPATCH_PAIR(3, 3); break; case 2: DISPATCH_PAIR(3, 2); break; default: return false; } break; \
            case 2: if (plan.v.bits == 2) { DISPATCH_PAIR(2, 2); } else { return false; } break; \
            default: return false; \
        } \
    } while (0)
#endif

static bool ggml_cuda_flash_attn_ext_kvarn_vec_supported(
        const ggml_cuda_fattn_kvarn_plan & plan,
        const ggml_tensor * dst) {
    const char * enabled = getenv("GGML_KVARN_VEC");
    if (enabled != nullptr && atoi(enabled) == 0) {
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * sinks = dst->src[4];
    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const int head_dim = (int) Q->ne[0];
    if (head_dim != 256 || K->ne[0] != head_dim || V->ne[0] != head_dim ||
            Q->ne[1] != 1 || Q->ne[3] != plan.n_stream || plan.n_stream <= 0) {
        return false;
    }
    if (!ggml_cuda_fattn_kvarn_rotated_decode_domain(dst)) {
        return false;
    }
    if (sinks != nullptr || max_bias != 0.0f) {
        return false;
    }
    if (Q->ne[2] % plan.n_kv_heads != 0) {
        return false;
    }
    const int gqa_ratio = (int) (Q->ne[2] / plan.n_kv_heads);
    // D256 SWA/GQA2 is the proven vec geometry (benchmarked at k4v4); every KVarN bit pair
    // is wired through it. D512 vec regressed deep-context global layers and stays excluded.
    return plan.k.swa && plan.v.swa && gqa_ratio == 2 &&
        ggml_cuda_fattn_kvarn_fast_decode_pair_enabled(plan.k.bits, plan.v.bits);
}

template<int D>
static bool ggml_cuda_flash_attn_ext_kvarn_vec_d(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_cuda_fattn_kvarn_plan & plan) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * mask = dst->src[3];
    const int n_q_heads = (int) Q->ne[2];
    const int gqa_ratio = n_q_heads / plan.n_kv_heads;
    constexpr int gqa_per_block = ggml_cuda_fattn_kvarn_vec_max_gqa<D>();
    const int n_gqa_blocks = (gqa_ratio + gqa_per_block - 1) / gqa_per_block;
    const int split_tokens = ggml_cuda_fattn_kvarn_vec_tokens_per_split();
    const int n_splits = (plan.n_kv + split_tokens - 1) / split_tokens;
    float scale = 1.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    const size_t n_desc = (size_t) plan.n_stream * plan.n_kv_heads;
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> k_desc(pool, n_desc);
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> v_desc(pool, n_desc);
    ggml_cuda_fattn_kvarn_init_descs(plan, k_desc.get(), v_desc.get(), 0, 0, stream);

    const size_t partial_count = (size_t) plan.n_stream * n_q_heads * n_splits * D;
    const size_t meta_count = (size_t) plan.n_stream * n_q_heads * n_splits;
    ggml_cuda_pool_alloc<float> partial(pool, partial_count);
    ggml_cuda_pool_alloc<float2> partial_meta(pool, meta_count);
    ggml_cuda_kv_memory_transient_stats_record_kvarn(
            k_desc.actual_size + v_desc.actual_size,
            partial.actual_size,
            partial_meta.actual_size,
            k_desc.actual_size + v_desc.actual_size + partial.actual_size + partial_meta.actual_size);

    if (getenv("GGML_CUDA_FA_ROUTE_DEBUG") != nullptr) {
        fprintf(stderr,
            "CUDA_FA_ROUTE_EXEC_DISPATCH kernel=KVARN_DECODE_VEC "
            "Q=[%lld,%lld,%lld,%lld] bits=[%d,%d] n_kv=%d n_kv_heads=%d "
            "n_stream=%d gqa=%d gqa_blocks=%d n_splits=%d split_tokens=%d\n",
            (long long) Q->ne[0], (long long) Q->ne[1],
            (long long) Q->ne[2], (long long) Q->ne[3],
            plan.k.bits, plan.v.bits, plan.n_kv, plan.n_kv_heads,
            plan.n_stream, gqa_ratio, n_gqa_blocks, n_splits, split_tokens);
        fflush(stderr);
    }

    ggml_cuda_fattn_kvarn_decode_args args = {};
    args.Q = (const char *) Q->data;
    args.k_descs = k_desc.get();
    args.v_descs = v_desc.get();
    args.mask = mask ? (const char *) mask->data : nullptr;
    args.partial = partial.get();
    args.partial_meta = partial_meta.get();
    args.dst = (float *) dst->data;
    args.dst_meta = dst->src[8] != nullptr ? (float2 *) dst->src[8]->data : nullptr;
    args.scale = scale;
    args.logit_softcap = logit_softcap;
    args.nb01 = Q->nb[1];
    args.nb02 = Q->nb[2];
    args.nb03 = Q->nb[3];
    args.nb30 = mask ? mask->nb[0] : 0;
    args.nb31 = mask ? mask->nb[1] : 0;
    args.nb33 = mask ? mask->nb[3] : 0;
    args.ne33 = mask ? (int) mask->ne[3] : 1;
    args.n_kv = plan.n_kv;
    args.n_q = 1;
    args.n_q_heads = n_q_heads;
    args.n_kv_heads = plan.n_kv_heads;
    args.n_stream = plan.n_stream;
    args.gqa_ratio = gqa_ratio;
    args.gqa_per_block = gqa_per_block;
    args.n_gqa_blocks = n_gqa_blocks;
    args.n_splits = n_splits;
    args.split_tokens = split_tokens;
    args.nwarps = 0;
    args.stream = stream;

#define GGML_CUDA_FATTN_KVARN_VEC_LAUNCH(K_BITS, V_BITS) \
    ggml_cuda_fattn_kvarn_vec_launch<D, K_BITS, V_BITS>(args)

    GGML_CUDA_FATTN_KVARN_FAST_DECODE_DISPATCH_K(GGML_CUDA_FATTN_KVARN_VEC_LAUNCH);
#undef GGML_CUDA_FATTN_KVARN_VEC_LAUNCH
    return true;
}

static bool ggml_cuda_flash_attn_ext_kvarn_vec(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_cuda_fattn_kvarn_plan & plan) {
    if (!ggml_cuda_flash_attn_ext_kvarn_vec_supported(plan, dst)) {
        return false;
    }
    return ggml_cuda_flash_attn_ext_kvarn_vec_d<256>(ctx, dst, plan);
}


static bool ggml_cuda_flash_attn_ext_kvarn_decode_supported(
        const ggml_cuda_fattn_kvarn_plan & plan,
        const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * sinks = dst->src[4];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    if ((Q->ne[0] != 128 && Q->ne[0] != 256 && Q->ne[0] != 512) || V->ne[0] != Q->ne[0] || K->ne[0] != Q->ne[0]) {
        return false;
    }
    if (Q->ne[1] <= 0 || Q->ne[3] != plan.n_stream || plan.n_stream <= 0) {
        return false;
    }
    if (!ggml_cuda_fattn_kvarn_rotated_decode_domain(dst)) {
        return false;
    }
    if (Q->ne[1] > GGML_CUDA_FATTN_KVARN_SPECIALIZED_DECODE_MAX_Q) {
        return false;
    }
    if (sinks != nullptr || max_bias != 0.0f) {
        return false;
    }
    if (Q->ne[2] % plan.n_kv_heads != 0) {
        return false;
    }
    const int gqa_ratio = (int) (Q->ne[2] / plan.n_kv_heads);
    return gqa_ratio > 0 && ggml_cuda_fattn_kvarn_fast_decode_pair_enabled(plan.k.bits, plan.v.bits);
}

template<int D>
static bool ggml_cuda_flash_attn_ext_kvarn_decode_d(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_cuda_fattn_kvarn_plan & plan) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * mask = dst->src[3];
    const int n_q = (int) Q->ne[1];
    const int n_q_heads = (int) Q->ne[2];
    const int gqa_ratio = n_q_heads / plan.n_kv_heads;
    ggml_cuda_fattn_kvarn_decode_geometry geometry = {};

#define GGML_CUDA_FATTN_KVARN_SELECT(K_BITS, V_BITS) \
    geometry = ggml_cuda_fattn_kvarn_decode_select<D, K_BITS, V_BITS>( \
        ctx.device, plan.n_kv, n_q, n_q_heads, plan.n_kv_heads, plan.n_stream)

    GGML_CUDA_FATTN_KVARN_FAST_DECODE_DISPATCH_K(GGML_CUDA_FATTN_KVARN_SELECT);
#undef GGML_CUDA_FATTN_KVARN_SELECT

    if (!geometry.use_split) {
        return false;
    }

    const int gqa_per_block = geometry.gqa_per_block;
    const int n_gqa_blocks = geometry.n_gqa_blocks;
    const int split_tokens = geometry.split_tokens;
    const int n_splits = geometry.n_splits;
    float scale = 1.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    const size_t n_desc = (size_t) plan.n_stream * plan.n_kv_heads;
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> k_desc(pool, n_desc);
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> v_desc(pool, n_desc);
    ggml_cuda_fattn_kvarn_init_descs(plan, k_desc.get(), v_desc.get(), 0, 0, stream);

    const size_t partial_count = (size_t) plan.n_stream * n_q_heads * n_q * n_splits * D;
    const size_t meta_count = (size_t) plan.n_stream * n_q_heads * n_q * n_splits;
    ggml_cuda_pool_alloc<float> partial(pool, partial_count);
    ggml_cuda_pool_alloc<float2> partial_meta(pool, meta_count);
    ggml_cuda_kv_memory_transient_stats_record_kvarn(
            k_desc.actual_size + v_desc.actual_size,
            partial.actual_size,
            partial_meta.actual_size,
            k_desc.actual_size + v_desc.actual_size + partial.actual_size + partial_meta.actual_size);

    if (getenv("GGML_CUDA_FA_ROUTE_DEBUG") != nullptr) {
        fprintf(stderr,
            "CUDA_FA_ROUTE_EXEC_DISPATCH kernel=KVARN_DECODE_SPLIT "
            "Q=[%lld,%lld,%lld,%lld] bits=[%d,%d] n_kv=%d n_kv_heads=%d n_stream=%d "
            "gqa=%d gqa_blocks=%d max_gqa=%d n_splits=%d split_tokens=%d nwarps=%d "
            "active_blocks_per_sm=%d wave_efficiency=%d waves=%d\n",
            (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
            plan.k.bits, plan.v.bits, plan.n_kv, plan.n_kv_heads, plan.n_stream,
            gqa_ratio, n_gqa_blocks, gqa_per_block, n_splits, split_tokens, geometry.nwarps,
            geometry.max_blocks_per_sm, geometry.wave_efficiency_percent, geometry.n_waves);
        fflush(stderr);
    }

    ggml_cuda_fattn_kvarn_decode_args args = {};
    args.Q = (const char *) Q->data;
    args.k_descs = k_desc.get();
    args.v_descs = v_desc.get();
    args.mask = mask ? (const char *) mask->data : nullptr;
    args.partial = partial.get();
    args.partial_meta = partial_meta.get();
    args.dst = (float *) dst->data;
    args.dst_meta = dst->src[8] != nullptr ? (float2 *) dst->src[8]->data : nullptr;
    args.scale = scale;
    args.logit_softcap = logit_softcap;
    args.nb01 = Q->nb[1];
    args.nb02 = Q->nb[2];
    args.nb03 = Q->nb[3];
    args.nb30 = mask ? mask->nb[0] : 0;
    args.nb31 = mask ? mask->nb[1] : 0;
    args.nb33 = mask ? mask->nb[3] : 0;
    args.ne33 = mask ? (int) mask->ne[3] : 1;
    args.n_kv = plan.n_kv;
    args.n_q = n_q;
    args.n_q_heads = n_q_heads;
    args.n_kv_heads = plan.n_kv_heads;
    args.n_stream = plan.n_stream;
    args.gqa_ratio = gqa_ratio;
    args.gqa_per_block = gqa_per_block;
    args.n_gqa_blocks = n_gqa_blocks;
    args.n_splits = n_splits;
    args.split_tokens = split_tokens;
    args.nwarps = geometry.nwarps;
    args.wave_size = ggml_cuda_info().devices[ctx.device].warp_size;
    args.stream = stream;

#define GGML_CUDA_FATTN_KVARN_LAUNCH(K_BITS, V_BITS) \
    ggml_cuda_fattn_kvarn_decode_launch<D, K_BITS, V_BITS>(args)

    GGML_CUDA_FATTN_KVARN_FAST_DECODE_DISPATCH_K(GGML_CUDA_FATTN_KVARN_LAUNCH);
#undef GGML_CUDA_FATTN_KVARN_LAUNCH
    return true;
}

static bool ggml_cuda_flash_attn_ext_kvarn_decode(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_cuda_fattn_kvarn_plan & plan) {
    if (!ggml_cuda_flash_attn_ext_kvarn_decode_supported(plan, dst)) {
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    switch ((int) Q->ne[0]) {
        case 128: return ggml_cuda_flash_attn_ext_kvarn_decode_d<128>(ctx, dst, plan);
        case 256: return ggml_cuda_flash_attn_ext_kvarn_decode_d<256>(ctx, dst, plan);
        case 512: return ggml_cuda_flash_attn_ext_kvarn_decode_d<512>(ctx, dst, plan);
        default:  return false;
    }
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_kvarn_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_kvarn_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_kvarn_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) {
        ggml_cuda_flash_attn_ext_mma_kvarn_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_kvarn_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_kvarn_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

#if !defined(GGML_USE_MUSA)
    if constexpr ((DKQ == 128 && DV == 128) || (DKQ == 256 && DV == 256)) {
        const bool use_wide_shape = ggml_cuda_fattn_kvarn_use_wide_mma(
            (int) Q->ne[1], gqa_ratio, true);
        if (use_gqa_opt && use_wide_shape &&
            ggml_cuda_fattn_kvarn_wide_mma_supported<DKQ, DV, 16, 8>(ctx, dst)) {
            ggml_cuda_flash_attn_ext_mma_kvarn_case<DKQ, DV, 16, 8>(ctx, dst);
            return;
        }
    }
#endif

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_kvarn_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_kvarn_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_kvarn_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_kvarn_switch_ncols1<DKQ, DV, 1>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_mma_kvarn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * V = dst->src[2];

    switch (Q->ne[0]) {
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_kvarn_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_kvarn_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_kvarn_switch_ncols2<512, 512>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported KVarN native FlashAttention head_dim");
    }
}



bool ggml_cuda_flash_attn_ext_kvarn_uses_views(
        const ggml_tensor * dst) {
    return ggml_cuda_fattn_kvarn_uses_views(dst);
}

bool ggml_cuda_flash_attn_ext_kvarn_supported(
        int device,
        const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    ggml_cuda_fattn_kvarn_plan plan;
    if (!ggml_cuda_fattn_kvarn_supported(device, dst, &plan)) {
        return false;
    }
    const auto capabilities = ggml_cuda_fattn_kvarn_device_capabilities(device);
    return capabilities.generic_mma || capabilities.decode_split ||
        (capabilities.portable_native &&
         ggml_cuda_fattn_kvarn_portable_supported(plan, dst));
#endif // FLASH_ATTN_AVAILABLE
}

bool ggml_cuda_flash_attn_ext_kvarn_portable_supported(
        int device,
        const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    const auto capabilities = ggml_cuda_fattn_kvarn_device_capabilities(device);
    ggml_cuda_fattn_kvarn_plan plan;
    return capabilities.portable_native &&
        ggml_cuda_fattn_kvarn_supported(device, dst, &plan) &&
        ggml_cuda_fattn_kvarn_portable_supported(plan, dst);
#endif
}

bool ggml_cuda_flash_attn_ext_kvarn_direct_tail_supported(
        int device,
        const ggml_tensor * dst) {
    const auto capabilities = ggml_cuda_fattn_kvarn_device_capabilities(device);
    const char * force_portable = getenv("GGML_KVARN_TEST_FORCE_PORTABLE_FATTN");
    const bool portable_route =
        !capabilities.specialized_routes ||
        (force_portable != nullptr && atoi(force_portable) != 0);
    return dst != nullptr && dst->src[10] == nullptr &&
        capabilities.portable_native && portable_route &&
        ggml_cuda_flash_attn_ext_kvarn_portable_supported(device, dst);
}

static void ggml_cuda_fattn_kvarn_record_entry(ggml_cuda_fattn_kvarn_entry_path entry_path) {
    if (entry_path == GGML_CUDA_FATTN_KVARN_ENTRY_COMPACT_TAIL) {
        g_kvarn_route_compact_tail_entry.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_kvarn_route_direct_entry.fetch_add(1, std::memory_order_relaxed);
    }
}

static bool ggml_cuda_fattn_kvarn_debug_routes_enabled() {
    const char * value = getenv("GGML_KVARN_DEBUG_ROUTES");
    return value != nullptr && atoi(value) != 0;
}

static void ggml_cuda_fattn_kvarn_debug_route(
        int device,
        const ggml_cuda_fattn_kvarn_plan & plan,
        const ggml_tensor * dst,
        ggml_cuda_fattn_kvarn_entry_path entry_path,
        const char * route,
        const char * fallback_reason) {
    if (!ggml_cuda_fattn_kvarn_debug_routes_enabled()) {
        return;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * k_tail = dst->src[5];
    const ggml_tensor * k_tail_current = dst->src[10];
    const auto & device_info = ggml_cuda_info().devices[device];
    const int gqa = plan.n_kv_heads > 0 && Q->ne[2] % plan.n_kv_heads == 0 ?
        int(Q->ne[2] / plan.n_kv_heads) : 0;
    std::fprintf(stderr,
        "kvarn-route backend=%s cc=%d wave=%d D=%d k=%d v=%d gqa=%d nq=%d nkv=%d "
        "domain=%s tail_type=%s tail_history=%d tail_current=%d "
        "route=%s entry=%s fallback=%s\n",
#if defined(GGML_USE_MUSA)
        "MUSA",
#elif defined(GGML_USE_HIP)
        "HIP",
#else
        "CUDA",
#endif
        device_info.cc, device_info.warp_size, int(Q->ne[0]), plan.k.bits, plan.v.bits,
        gqa, int(Q->ne[1]), plan.n_kv, ggml_cuda_fattn_kvarn_domain_name(dst),
        k_tail != nullptr ? ggml_type_name(k_tail->type) : "none",
        k_tail != nullptr ? int(k_tail->ne[1]) : 0,
        k_tail_current != nullptr ? int(k_tail_current->ne[1]) : 0, route,
        entry_path == GGML_CUDA_FATTN_KVARN_ENTRY_COMPACT_TAIL ? "compact-tail" : "direct",
        fallback_reason != nullptr ? fallback_reason : "none");
}

bool ggml_cuda_flash_attn_ext_kvarn(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        ggml_cuda_fattn_kvarn_entry_path entry_path) {
    ggml_cuda_fattn_kvarn_plan plan;
    if (!ggml_cuda_fattn_kvarn_supported(ctx.device, dst, &plan)) {
        return false;
    }

    const auto capabilities = ggml_cuda_fattn_kvarn_device_capabilities(ctx.device);
    ggml_cuda_fattn_kvarn_record_entry(entry_path);

    const char * force_portable = getenv("GGML_KVARN_TEST_FORCE_PORTABLE_FATTN");
    if (capabilities.portable_native &&
            force_portable != nullptr && atoi(force_portable) != 0 &&
            ggml_cuda_fattn_kvarn_portable_supported(plan, dst)) {
        g_kvarn_route_portable_native.fetch_add(1, std::memory_order_relaxed);
        ggml_cuda_fattn_kvarn_debug_route(
            ctx.device, plan, dst, entry_path, "portable-native", "forced");
        return ggml_cuda_flash_attn_ext_kvarn_portable(ctx, dst, plan);
    }

    if (!capabilities.specialized_routes) {
        if (!ggml_cuda_fattn_kvarn_portable_supported(plan, dst)) {
            g_kvarn_route_materialize_fallback.fetch_add(1, std::memory_order_relaxed);
            ggml_cuda_fattn_kvarn_debug_route(
                ctx.device, plan, dst, entry_path,
                "materialize-fallback", "portable-shape-unsupported");
            return false;
        }
        g_kvarn_route_portable_native.fetch_add(1, std::memory_order_relaxed);
        ggml_cuda_fattn_kvarn_debug_route(
            ctx.device, plan, dst, entry_path, "portable-native", "no-matrix-capability");
        return ggml_cuda_flash_attn_ext_kvarn_portable(ctx, dst, plan);
    }

    const ggml_tensor * Q = dst->src[0];
    const bool prompt_prefill =
        Q->ne[1] > GGML_CUDA_FATTN_KVARN_SPECIALIZED_DECODE_MAX_Q;
    const bool vector_eligible = capabilities.decode_vector && !prompt_prefill &&
        ggml_cuda_flash_attn_ext_kvarn_vec_supported(plan, dst);
    const bool split_eligible = capabilities.decode_split && !prompt_prefill &&
        ggml_cuda_flash_attn_ext_kvarn_decode_supported(plan, dst);
    const int gqa = plan.n_kv_heads > 0 && Q->ne[2] % plan.n_kv_heads == 0 ?
        int(Q->ne[2] / plan.n_kv_heads) : 0;
    const ggml_cuda_fattn_kvarn_route route = ggml_cuda_fattn_kvarn_select_route({
        int(Q->ne[0]), int(Q->ne[1]), gqa, plan.k.bits, plan.v.bits,
        plan.k.swa && plan.v.swa, dst->src[8] != nullptr,
        vector_eligible, split_eligible, prompt_prefill,
    });

    if (route == GGML_CUDA_FATTN_KVARN_ROUTE_DECODE_VECTOR) {
        if (ggml_cuda_flash_attn_ext_kvarn_vec(ctx, dst, plan)) {
            g_kvarn_route_decode_vector.fetch_add(1, std::memory_order_relaxed);
#if defined(GGML_USE_HIP)
            g_kvarn_route_amd_decode_vector.fetch_add(1, std::memory_order_relaxed);
#endif
            ggml_cuda_fattn_kvarn_debug_route(
                ctx.device, plan, dst, entry_path, "decode-vector", nullptr);
            return true;
        }
    }
    if (route == GGML_CUDA_FATTN_KVARN_ROUTE_DECODE_SPLIT ||
            route == GGML_CUDA_FATTN_KVARN_ROUTE_DECODE_VECTOR) {
        if (ggml_cuda_flash_attn_ext_kvarn_decode(ctx, dst, plan)) {
            g_kvarn_route_decode_split.fetch_add(1, std::memory_order_relaxed);
            g_kvarn_route_split_reduce.fetch_add(1, std::memory_order_relaxed);
#if defined(GGML_USE_HIP)
            g_kvarn_route_amd_decode_split.fetch_add(1, std::memory_order_relaxed);
#endif
            ggml_cuda_fattn_kvarn_debug_route(
                ctx.device, plan, dst, entry_path, "decode-split", nullptr);
            return true;
        }
    }

    if (capabilities.generic_mma) {
        if (prompt_prefill) {
            g_kvarn_route_prompt_prefill.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_kvarn_route_generic_mma.fetch_add(1, std::memory_order_relaxed);
        }
#if defined(GGML_USE_HIP)
        g_kvarn_route_amd_generic_mma.fetch_add(1, std::memory_order_relaxed);
#endif
        ggml_cuda_fattn_kvarn_debug_route(
            ctx.device, plan, dst, entry_path,
            prompt_prefill ? "prompt-generic-mma" : "generic-mma",
            split_eligible ? "split-geometry-rejected" : nullptr);
        ggml_cuda_flash_attn_ext_mma_kvarn(ctx, dst);
        return true;
    }

    if (!ggml_cuda_fattn_kvarn_portable_supported(plan, dst)) {
        g_kvarn_route_materialize_fallback.fetch_add(1, std::memory_order_relaxed);
        ggml_cuda_fattn_kvarn_debug_route(
            ctx.device, plan, dst, entry_path,
            "materialize-fallback", "portable-shape-unsupported");
        return false;
    }
    g_kvarn_route_portable_native.fetch_add(1, std::memory_order_relaxed);
    ggml_cuda_fattn_kvarn_debug_route(
        ctx.device, plan, dst, entry_path, "portable-native", "shape-fallback");
    return ggml_cuda_flash_attn_ext_kvarn_portable(ctx, dst, plan);
}

#endif // GGML_CUDA_KVARN
