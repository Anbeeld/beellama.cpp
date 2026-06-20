#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-mma-kvarn.cuh"
#include "fattn-mma-turbo.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"

#include <atomic>

// InnerQ: update the fattn-side inverse scale array from host (all devices)
void turbo_innerq_update_fattn_scales(const float * scale_inv) {
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_innerq_channel_scale_inv_fattn, scale_inv, 128 * sizeof(float));
    }
    cudaSetDevice(cur_device);
}

void turbo_innerq_init_fattn() {
    float ones[128];
    for (int i = 0; i < 128; i++) ones[i] = 1.0f;
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_innerq_channel_scale_inv_fattn, ones, sizeof(ones));
    }
    cudaSetDevice(cur_device);
}

// Q² calibration: host-side management
static int q_calibrate_state = 0; // 0=off, 1=collecting, 2=done

static inline void ggml_cuda_fattn_kvarn_fill_descs_host(
        const ggml_cuda_fattn_kvarn_plan_side & side,
        const ggml_cuda_fattn_kvarn_plan & plan,
        const int * live_groups,
        std::vector<ggml_cuda_fattn_kvarn_desc> & descs) {
    descs.resize((size_t) plan.n_stream * plan.n_kv_heads);
    for (int s = 0; s < plan.n_stream; ++s) {
        for (int h = 0; h < plan.n_kv_heads; ++h) {
            ggml_cuda_fattn_kvarn_desc & desc = descs[(size_t) s * plan.n_kv_heads + h];
            desc.records = (const uint8_t *) side.records->data;
            desc.stage = (const half *) side.stage->data;
            desc.indices = (const int64_t *) side.indices->data;
            desc.live_groups = live_groups;
            desc.n_record_heads = (int) side.view->ne[1];
            desc.out_stream = s;
            desc.stream = side.stream_start + s;
            desc.head_base = h * plan.slices;
            desc.groups_per_stream = side.groups_per_stream;
            desc.record_bytes = (int) side.records->ne[0];
            desc.stage_groups = side.stage_groups;
            desc.tail_groups = side.stage_groups - 1;
            desc.bits = side.bits;
            desc.value = side.value ? 1 : 0;
            desc.swa = side.swa ? 1 : 0;
        }
    }
}

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_kvarn_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_fattn_kvarn_plan plan;
    GGML_ASSERT(ggml_cuda_fattn_kvarn_supported(ctx.device, dst, &plan));

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps = nthreads / warp_size_host;
    constexpr bool V_is_K_view = false;

    const size_t nbytes_shared_KV = nbatch_fa * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q = ncols * (DKQ/2 + 4) * sizeof(half2);
    const size_t nbytes_shared_mask = ncols1 * (nbatch_fa/2 + 4) * sizeof(half2);
    const size_t nbytes_shared_combine = nwarps * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q, nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int> live_groups_k(pool, plan.n_stream);
    ggml_cuda_pool_alloc<int> live_groups_v(pool, plan.n_stream);
    ggml_cuda_fattn_kvarn_live_groups_kernel<<<plan.n_stream, GGML_CUDA_FATTN_KVARN_DIM, 0, stream>>>(
        (const int64_t *) plan.k.indices->data,
        (int) plan.k.indices->ne[0],
        plan.k.stream_start,
        plan.n_stream,
        plan.k.groups_per_stream,
        plan.k.swa,
        live_groups_k.get());
    ggml_cuda_fattn_kvarn_live_groups_kernel<<<plan.n_stream, GGML_CUDA_FATTN_KVARN_DIM, 0, stream>>>(
        (const int64_t *) plan.v.indices->data,
        (int) plan.v.indices->ne[0],
        plan.v.stream_start,
        plan.n_stream,
        plan.v.groups_per_stream,
        plan.v.swa,
        live_groups_v.get());

    std::vector<ggml_cuda_fattn_kvarn_desc> k_desc_host;
    std::vector<ggml_cuda_fattn_kvarn_desc> v_desc_host;
    ggml_cuda_fattn_kvarn_fill_descs_host(plan.k, plan, live_groups_k.get(), k_desc_host);
    ggml_cuda_fattn_kvarn_fill_descs_host(plan.v, plan, live_groups_v.get(), v_desc_host);

    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> k_desc(pool, k_desc_host.size());
    ggml_cuda_pool_alloc<ggml_cuda_fattn_kvarn_desc> v_desc(pool, v_desc_host.size());
    CUDA_CHECK(cudaMemcpyAsync(k_desc.get(), k_desc_host.data(), k_desc_host.size() * sizeof(k_desc_host[0]), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(v_desc.get(), v_desc_host.data(), v_desc_host.size() * sizeof(v_desc_host[0]), cudaMemcpyHostToDevice, stream));

    ggml_tensor K_desc = *dst->src[1];
    ggml_tensor V_desc = *dst->src[2];
    K_desc.data = k_desc.get();
    V_desc.data = v_desc.get();
    K_desc.type = GGML_TYPE_F16;
    V_desc.type = GGML_TYPE_F16;
    K_desc.view_src = nullptr;
    V_desc.view_src = nullptr;
    K_desc.view_offs = 0;
    V_desc.view_offs = 0;
    K_desc.nb[0] = sizeof(half);
    V_desc.nb[0] = sizeof(half);
    K_desc.nb[1] = 0;
    V_desc.nb[1] = 0;
    K_desc.nb[2] = sizeof(ggml_cuda_fattn_kvarn_desc);
    V_desc.nb[2] = sizeof(ggml_cuda_fattn_kvarn_desc);
    K_desc.nb[3] = sizeof(ggml_cuda_fattn_kvarn_desc) * plan.n_kv_heads;
    V_desc.nb[3] = sizeof(ggml_cuda_fattn_kvarn_desc) * plan.n_kv_heads;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view,
            GGML_CUDA_FATTN_KVARN_TYPE, GGML_CUDA_FATTN_KVARN_TYPE>;
#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view,
            GGML_CUDA_FATTN_KVARN_TYPE, GGML_CUDA_FATTN_KVARN_TYPE>;
#if !defined(GGML_USE_MUSA) && !defined(GGML_USE_HIP)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif
    }

    ggml_tensor * orig_k = dst->src[1];
    ggml_tensor * orig_v = dst->src[2];
    dst->src[1] = &K_desc;
    dst->src[2] = &V_desc;
    // need_f16_K=false, need_f16_V=false: KVarN descriptors feed the MMA tile loader directly.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true, warp_size_host);
    dst->src[1] = orig_k;
    dst->src[2] = orig_v;
}

static __device__ __forceinline__ void ggml_cuda_fattn_kvarn_wht_128(float * sh) {
    const int tid = threadIdx.x;
    for (int stride = 1; stride < GGML_CUDA_FATTN_KVARN_DIM; stride <<= 1) {
        __syncthreads();
        if (tid < GGML_CUDA_FATTN_KVARN_DIM / 2) {
            const int lo = (tid / stride) * (2 * stride) + (tid % stride);
            const int hi = lo + stride;
            const float a = sh[lo];
            const float b = sh[hi];
            sh[lo] = a + b;
            sh[hi] = a - b;
        }
    }
    __syncthreads();
}

template<int BITS, bool VALUE, bool EMIT_ROTATED>
static __global__ void ggml_cuda_fattn_kvarn_materialize_kernel(
        const uint8_t * records,
        const half * stage,
        const int64_t * indices,
        const int * live_groups,
        half * dst,
        int n_record_heads,
        int n_kv,
        int stream_start,
        int n_stream,
        int groups_per_stream,
        int record_bytes,
        int stage_groups,
        int tail_groups,
        bool swa) {
    const int head = blockIdx.x;
    const int out_stream = blockIdx.z;
    const int dim = threadIdx.x;
    if (head >= n_record_heads || out_stream >= n_stream) {
        return;
    }

    const int stream = stream_start + out_stream;
    const int live_group = live_groups[out_stream];
    for (int cell = blockIdx.y; cell < n_kv; cell += gridDim.y) {
        const int64_t abs_pos = swa ? indices[cell] : cell;
        float rotated = 0.0f;

        if (abs_pos >= 0) {
            const int group = (int) (abs_pos / GGML_CUDA_FATTN_KVARN_DIM);
            const int pos   = (int) (abs_pos - (int64_t) group * GGML_CUDA_FATTN_KVARN_DIM);

            bool from_stage;
            bool from_record;
            int stage_pos;
            int record_group;

            if (swa) {
                const int stage_begin = live_group >= (tail_groups - 1) ? live_group - (tail_groups - 1) : 0;
                from_stage  = group >= stage_begin && group <= live_group;
                from_record = !from_stage && group >= 0 && group < stage_begin &&
                    (live_group - group) < groups_per_stream;
                stage_pos = (group % stage_groups) * GGML_CUDA_FATTN_KVARN_DIM + pos;
                record_group = group % groups_per_stream;
            } else {
                from_stage = group == 0 ||
                    (group > 0 && group <= live_group && group + (tail_groups - 1) >= live_group);
                from_record = !from_stage && group < live_group && group < groups_per_stream;
                const int stage_base = stream * GGML_CUDA_FATTN_KVARN_DIM * stage_groups;
                stage_pos = stage_base + (group == 0 ? pos :
                    GGML_CUDA_FATTN_KVARN_DIM + ((group - 1) % tail_groups) * GGML_CUDA_FATTN_KVARN_DIM + pos);
                record_group = stream * groups_per_stream + group;
            }

            if (from_stage) {
                rotated = __half2float(stage[((int64_t) stage_pos * n_record_heads + head) * GGML_CUDA_FATTN_KVARN_DIM + dim]);
            } else if (from_record) {
                const uint8_t * record = records + ((int64_t) record_group * n_record_heads + head) * record_bytes;
                constexpr int payload_bytes = GGML_CUDA_FATTN_KVARN_DIM * GGML_CUDA_FATTN_KVARN_DIM * BITS / 8;
                const half * scale_axis = (const half *) (record + payload_bytes);
                const half * zp_axis    = scale_axis + GGML_CUDA_FATTN_KVARN_DIM;
                const half * other_axis = zp_axis + GGML_CUDA_FATTN_KVARN_DIM;
                const int row = VALUE ? pos : dim;
                const int col = VALUE ? dim : pos;
                const uint8_t q = ggml_cuda_fattn_kvarn_unpack_record(record, row * GGML_CUDA_FATTN_KVARN_DIM + col, BITS);
                rotated = (float(q) * __half2float(scale_axis[row]) + __half2float(zp_axis[row])) * __half2float(other_axis[col]);
            }
        }

        if constexpr (EMIT_ROTATED) {
            dst[((int64_t) out_stream * n_kv * n_record_heads + (int64_t) cell * n_record_heads + head) * GGML_CUDA_FATTN_KVARN_DIM + dim] =
                __float2half_rn(rotated);
        } else {
            __shared__ float sh[GGML_CUDA_FATTN_KVARN_DIM];
            sh[dim] = rotated;
            ggml_cuda_fattn_kvarn_wht_128(sh);
            dst[((int64_t) out_stream * n_kv * n_record_heads + (int64_t) cell * n_record_heads + head) * GGML_CUDA_FATTN_KVARN_DIM + dim] =
                __float2half_rn(sh[dim] * 0.08838834764831845f);
        }
    }
}

template<int BITS, bool VALUE>
static void ggml_cuda_fattn_kvarn_launch_materialize(
        const ggml_cuda_fattn_kvarn_plan_side & side,
        const ggml_cuda_fattn_kvarn_plan & plan,
        const int * live_groups,
        half * dst,
        bool emit_rotated,
        cudaStream_t stream) {
    const int n_record_heads = (int) side.view->ne[1];
    const dim3 blocks((uint32_t) n_record_heads, (uint32_t) std::min(plan.n_kv, 65535), (uint32_t) plan.n_stream);
    if (emit_rotated) {
        ggml_cuda_fattn_kvarn_materialize_kernel<BITS, VALUE, true><<<blocks, GGML_CUDA_FATTN_KVARN_DIM, 0, stream>>>(
            (const uint8_t *) side.records->data,
            (const half *) side.stage->data,
            (const int64_t *) side.indices->data,
            live_groups,
            dst,
            n_record_heads,
            plan.n_kv,
            side.stream_start,
            plan.n_stream,
            side.groups_per_stream,
            (int) side.records->ne[0],
            side.stage_groups,
            side.stage_groups - 1,
            side.swa);
    } else {
        ggml_cuda_fattn_kvarn_materialize_kernel<BITS, VALUE, false><<<blocks, GGML_CUDA_FATTN_KVARN_DIM, 0, stream>>>(
            (const uint8_t *) side.records->data,
            (const half *) side.stage->data,
            (const int64_t *) side.indices->data,
            live_groups,
            dst,
            n_record_heads,
            plan.n_kv,
            side.stream_start,
            plan.n_stream,
            side.groups_per_stream,
            (int) side.records->ne[0],
            side.stage_groups,
            side.stage_groups - 1,
            side.swa);
    }
}

static void ggml_cuda_fattn_kvarn_materialize_side(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_fattn_kvarn_plan_side & side,
        const ggml_cuda_fattn_kvarn_plan & plan,
        half * dst,
        bool emit_rotated) {
    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int> live_groups(pool, plan.n_stream);
    ggml_cuda_fattn_kvarn_live_groups_kernel<<<plan.n_stream, GGML_CUDA_FATTN_KVARN_DIM, 0, stream>>>(
        (const int64_t *) side.indices->data,
        (int) side.indices->ne[0],
        side.stream_start,
        plan.n_stream,
        side.groups_per_stream,
        side.swa,
        live_groups.get());

    switch (side.bits) {
        case 2:
            if (side.value) ggml_cuda_fattn_kvarn_launch_materialize<2, true >(side, plan, live_groups.get(), dst, emit_rotated, stream);
            else            ggml_cuda_fattn_kvarn_launch_materialize<2, false>(side, plan, live_groups.get(), dst, emit_rotated, stream);
            break;
        case 3:
            if (side.value) ggml_cuda_fattn_kvarn_launch_materialize<3, true >(side, plan, live_groups.get(), dst, emit_rotated, stream);
            else            ggml_cuda_fattn_kvarn_launch_materialize<3, false>(side, plan, live_groups.get(), dst, emit_rotated, stream);
            break;
        case 4:
            if (side.value) ggml_cuda_fattn_kvarn_launch_materialize<4, true >(side, plan, live_groups.get(), dst, emit_rotated, stream);
            else            ggml_cuda_fattn_kvarn_launch_materialize<4, false>(side, plan, live_groups.get(), dst, emit_rotated, stream);
            break;
        case 5:
            if (side.value) ggml_cuda_fattn_kvarn_launch_materialize<5, true >(side, plan, live_groups.get(), dst, emit_rotated, stream);
            else            ggml_cuda_fattn_kvarn_launch_materialize<5, false>(side, plan, live_groups.get(), dst, emit_rotated, stream);
            break;
        case 6:
            if (side.value) ggml_cuda_fattn_kvarn_launch_materialize<6, true >(side, plan, live_groups.get(), dst, emit_rotated, stream);
            else            ggml_cuda_fattn_kvarn_launch_materialize<6, false>(side, plan, live_groups.get(), dst, emit_rotated, stream);
            break;
        case 8:
            if (side.value) ggml_cuda_fattn_kvarn_launch_materialize<8, true >(side, plan, live_groups.get(), dst, emit_rotated, stream);
            else            ggml_cuda_fattn_kvarn_launch_materialize<8, false>(side, plan, live_groups.get(), dst, emit_rotated, stream);
            break;
        default:
            GGML_ABORT("unsupported KVarN materialize bits %d", side.bits);
    }
}

void turbo_q_calibrate_init() {
    const char * env = getenv("TURBO_Q_CALIBRATE");
    if (!env || atoi(env) != 1) return;

    double zeros[128] = {};
    int zero = 0, one = 1;
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_q_channel_sq_fattn, zeros, sizeof(zeros));
        cudaMemcpyToSymbol(d_q_channel_count_fattn, &zero, sizeof(zero));
        cudaMemcpyToSymbol(d_q_calibrate_fattn, &one, sizeof(one));
    }
    cudaSetDevice(cur_device);
    q_calibrate_state = 1;
    fprintf(stderr, "TURBO_Q_CALIBRATE: collecting per-position Q² statistics\n");
}

void turbo_q_calibrate_finalize() {
    if (q_calibrate_state != 1) return;

    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);

    int zero = 0;
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_q_calibrate_fattn, &zero, sizeof(zero));
    }
    cudaSetDevice(cur_device);

    double sq[128];
    int count;
    cudaMemcpyFromSymbol(sq, d_q_channel_sq_fattn, sizeof(sq));
    cudaMemcpyFromSymbol(&count, d_q_channel_count_fattn, sizeof(count));

    if (count == 0) {
        fprintf(stderr, "TURBO_Q_CALIBRATE: no Q vectors seen, skipping\n");
        q_calibrate_state = 2;
        return;
    }

    // Compute E[Q²] per position and save as 128 float32 values
    float weights[128];
    fprintf(stderr, "TURBO_Q_CALIBRATE: %d Q groups accumulated\n", count);
    double total = 0;
    for (int i = 0; i < 128; i++) {
        weights[i] = (float)(sq[i] / count);
        total += weights[i];
    }
    float mean = (float)(total / 128.0);
    float maxw = 0, minw = 1e30f;
    for (int i = 0; i < 128; i++) {
        if (weights[i] > maxw) maxw = weights[i];
        if (weights[i] < minw) minw = weights[i];
    }
    fprintf(stderr, "  E[Q²] mean=%.6f min=%.6f max=%.6f ratio=%.2f\n",
            mean, minw, maxw, maxw / (minw > 1e-10f ? minw : 1e-10f));

    const char * path = "/tmp/q_weights.bin";
    FILE * fp = fopen(path, "wb");
    if (fp) {
        fwrite(weights, sizeof(float), 128, fp);
        fclose(fp);
        fprintf(stderr, "  Saved Q weights to %s (128 floats)\n", path);
    }

    q_calibrate_state = 2;
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        if (use_gqa_opt && gqa_ratio > 1) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }

        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

// Turbo MMA fused dispatch: ncols1 selection (mirrors f16 version but calls turbo case).
template <int DKQ, int DV, int ncols2, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 8/ncols2, ncols2, type_K, type_V>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 16/ncols2) {
        ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 16/ncols2, ncols2, type_K, type_V>(ctx, dst);
        return;
    }

    // Turing (sm_75) is capped at ncols=32 — the kernel has NO_DEVICE_CODE for ncols>32.
    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 32/ncols2, ncols2, type_K, type_V>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 64/ncols2, ncols2, type_K, type_V>(ctx, dst);
}

// Turbo MMA fused dispatch: ncols2 selection based on GQA ratio.
template <int DKQ, int DV, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 8, type_K, type_V>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 4, type_K, type_V>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 2, type_K, type_V>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 1, type_K, type_V>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
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

// Context-adaptive V alpha: logarithmic scaling based on current KV occupancy.
// 3-bit: alpha = 1.1484 - 0.01443 * ln(n_kv), calibrated on Qwen3.5-27B decode-time KLD sweeps
//   at 8K/16K/32K. Optima: 8K→1.020, 16K→1.005, 32K→1.000.
// 2-bit: alpha = 0.8865 + 0.0195 * ln(n_kv), calibrated on fine-grained 0.005-step decode-time
//   KLD sweeps at 2K/7K/16K/32K on Qwen3.5-27B. Optima: 2K→1.030, 7K→1.065, 16K→1.090, 32K→1.075.
// Override with TURBO_TCQ_DECODE_ALPHA_V env var to force a static alpha (disables adaptive).
static float d_tcq_decode_alpha_v_static = 0.0f; // 0 = use adaptive, >0 = static override
static float d_tcq_decode_alpha_k = 1.0f;       // K decode alpha, static (default 1.0)
static bool d_tcq_decode_alpha_loaded = false;

static inline float tcq_compute_alpha_v(ggml_type v_type, int64_t n_kv) {
    if (d_tcq_decode_alpha_v_static > 0.0f) return d_tcq_decode_alpha_v_static;
    if (n_kv < 1) n_kv = 1;
    const float ln_ctx = logf((float)n_kv);
    if (v_type == GGML_TYPE_TURBO4_TCQ) {
        return 1.0f;
    } else if (v_type == GGML_TYPE_TURBO3_TCQ) {
        // v3: 3-point fit from fine-grained KLD sweeps at 8K/16K/32K on Qwen3.5-27B.
        // Per-token: n_kv=512→1.044, 2K→1.024, 8K→1.004, 16K→0.994, 32K→0.984
        // Clamped [0.98, 1.06] — early tokens (small n_kv) use higher alpha.
        return fmaxf(0.98f, fminf(1.06f, 1.1484f - 0.01443f * ln_ctx));
    } else if (v_type == GGML_TYPE_TURBO2_TCQ) {
        // v2: 4-point fit from fine-grained 0.005-step KLD sweeps at 2K/7K/16K/32K.
        // Per-token: n_kv=512→1.008, 2K→1.035, 7K→1.060, 16K→1.076, 32K→1.089
        return fmaxf(1.00f, fminf(1.12f, 0.8865f + 0.0195f * ln_ctx));
    }
    return 1.0f;
}

static void load_tcq_decode_alpha(int device) {
    static bool loaded[GGML_CUDA_MAX_DEVICES] = {};
    if (loaded[device]) return;
    loaded[device] = true;
    if (!d_tcq_decode_alpha_loaded) {
        d_tcq_decode_alpha_loaded = true;
        const char * sk = getenv("TURBO_TCQ_DECODE_ALPHA_K");
        if (sk) {
            char * end;
            errno = 0;
            float a = strtof(sk, &end);
            if (end != sk && errno == 0 && a > 0.0f && a < 10.0f) {
                d_tcq_decode_alpha_k = a;
                fprintf(stderr, "TCQ decode: K alpha=%.4f\n", a);
            }
        }
        const char * s = getenv("TURBO_TCQ_DECODE_ALPHA_V");
        if (s) {
            char * end;
            errno = 0;
            float a = strtof(s, &end);
            if (end != s && errno == 0 && a > 0.0f && a < 10.0f) {
                d_tcq_decode_alpha_v_static = a;
            }
        }
    }
    if (d_tcq_decode_alpha_v_static > 0.0f) {
        cudaMemcpyToSymbol(d_tcq_decode_alpha_v_fattn, &d_tcq_decode_alpha_v_static, sizeof(float));
        if (device == 0) fprintf(stderr, "TCQ decode: V alpha=%.4f (static override)\n", d_tcq_decode_alpha_v_static);
    } else {
        if (device == 0) fprintf(stderr, "TCQ decode: context-adaptive V alpha enabled\n");
    }
}

static void load_tcq4_codebook_fattn(int device, const char * label) {
    static bool tcq4_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
    if (tcq4_cb_loaded[device]) {
        return;
    }
    tcq4_cb_loaded[device] = true;
    const char * cb_path = getenv("TURBO_TCQ_CB4");
    if (!cb_path) {
        return;
    }

    float cb[1024];
    FILE * f = fopen(cb_path, "rb");
    if (f && fread(cb, sizeof(float), 1024, f) == 1024) {
        fclose(f);
        cudaMemcpyToSymbol(d_turbo4_tcq_codebook_fattn, cb, 1024*sizeof(float));
        fprintf(stderr, "%s: loaded 4-bit codebook from %s (device %d)\n", label, cb_path, device);
    } else {
        if (f) {
            fclose(f);
        }
        fprintf(stderr, "%s: FAILED to load 4-bit codebook from %s\n", label, cb_path);
    }
}

// === Turbo prefill: bulk dequant to fp16 + MMA attention ===
// During prefill (Q->ne[1] > 1), dequantize turbo K/V to fp16 temp buffers
// and use the fast MMA kernel instead of the slower vec kernel.

static __global__ void k_turbo2_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO2;
    const int j_in_blk = j % QK_TURBO2;
    const block_turbo2_0 * blk = (const block_turbo2_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t idx = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
    const float val = d_turbo_centroids_2bit_fattn[idx] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo3_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO3;
    const int j_in_blk = j % QK_TURBO3;
    const block_turbo3_0 * blk = (const block_turbo3_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t low2 = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
    const uint8_t hi1  = (blk->signs[j_in_blk / 8] >> (j_in_blk % 8)) & 0x1;
    const float val = d_turbo_centroids_3bit_fattn[low2 | (hi1 << 2)] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo3_tcq_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x; // element index within row (0..ne0-1)
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO3_TCQ;
    const int t = j % QK_TURBO3_TCQ; // element index within 128-element block
    const block_turbo3_tcq * blk = (const block_turbo3_tcq *)src_row + blk_idx;

    const float norm = __half2float(blk->norm) * alpha;

    // Sliding window decode: read 9-bit state from bitstream at bit offset t*3
    const int bit_pos = t * 3;
    const int byte_idx = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
    const int state = (raw >> bit_off) & 0x1FF;
    const float val = d_turbo3_tcq_codebook_fattn[state] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo4_tcq_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO4_TCQ;
    const int t = j % QK_TURBO4_TCQ;
    const block_turbo4_tcq * blk = (const block_turbo4_tcq *)src_row + blk_idx;

    const float norm = __half2float(blk->norm) * alpha;
    const int bit_pos = t * 4;
    const int byte_idx = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
    const int state = (raw >> bit_off) & 0x3FF;
    const float val = d_turbo4_tcq_codebook_fattn[state] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo2_tcq_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO2_TCQ;
    const int t = j % QK_TURBO2_TCQ;
    const block_turbo2_tcq * blk = (const block_turbo2_tcq *)src_row + blk_idx;

    const float norm = __half2float(blk->norm) * alpha;

    // Sliding window decode: read 8-bit state from bitstream at bit offset t*2
    const int bit_pos = t * 2;
    const int byte_idx = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
    const int state = (raw >> bit_off) & 0xFF;
    const float val = d_turbo2_tcq_codebook_fattn[state] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo4_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO4;
    const int j_in_blk = j % QK_TURBO4;
    const block_turbo4_0 * blk = (const block_turbo4_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t idx = (j_in_blk & 1) ? (blk->qs[j_in_blk / 2] >> 4) : (blk->qs[j_in_blk / 2] & 0xF);
    const float val = d_turbo_centroids_4bit_fattn[idx] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

// turbo3 K dequant with inverse FWHT: produces K in original (unrotated) domain
// so Q does NOT need pre-rotation. 128 threads per block, loops over 128-element FWHT groups
// (each group spans 4 turbo3 storage blocks of 32 elements; all 4 blocks share the same norm).
// In-place 128-point inverse FWHT spread across 128 threads, one element per thread.
// h=1..16 use intra-warp __shfl_xor_sync (no smem, no syncthreads).
// h=32 and h=64 cross warp boundaries, so we fall back to smem with syncs.
// Caller supplies a __shared__ float[128] scratch buffer.
static __device__ __forceinline__ float fwht128_butterfly_inplace(float val, float * smem) {
    const int tid = threadIdx.x;

    // Intra-warp passes: shuffle xor with stride h, no smem, no sync.
    #pragma unroll
    for (int h = 1; h <= 16; h *= 2) {
        const float other = __shfl_xor_sync(0xFFFFFFFFULL, val, h);
        val = (tid & h) ? (other - val) : (val + other);
    }

    // h=32 (cross-warp within block, smem needed)
    smem[tid] = val;
    __syncthreads();
    val = (tid & 32) ? (smem[tid - 32] - val) : (val + smem[tid + 32]);
    __syncthreads();

    // h=64 (cross-warp within block, smem needed)
    smem[tid] = val;
    __syncthreads();
    val = (tid & 64) ? (smem[tid - 64] - val) : (val + smem[tid + 64]);
    __syncthreads();

    return val;
}

// Pair-pack two adjacent threads' fp16 outputs into a single 32-bit half2 store.
// Halves the global store count vs. one __float2half per thread.
static __device__ __forceinline__ void fwht128_store_half(
        float val, half * dst_base) {
    const int tid = threadIdx.x;
    const float neighbor = __shfl_xor_sync(0xFFFFFFFFULL, val, 1);
    if ((tid & 1) == 0) {
        const half2 packed = __floats2half2_rn(val, neighbor);
        *((half2 *)(dst_base + tid)) = packed;
    }
}

static __global__ void k_turbo3_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    // ne0 in elements, FWHT group = 128 elements = 4 turbo3 blocks of 32
    const int n_groups = (int)(ne0 / 128);
    constexpr int blocks_per_group = 128 / QK_TURBO3; // 4

    for (int g = 0; g < n_groups; g++) {
        // Element index within the FWHT group
        const int j_in_grp = tid;            // 0..127
        const int blk_in_grp = j_in_grp / QK_TURBO3;  // 0..3
        const int j_in_blk  = j_in_grp % QK_TURBO3;   // 0..31

        const block_turbo3_0 * blk = (const block_turbo3_0 *)src_row + g * blocks_per_group + blk_in_grp;
        const float norm = __half2float(blk->norm);   // same for all 4 blocks in this group

        const uint8_t low2 = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
        const uint8_t hi1  = (blk->signs[j_in_blk / 8] >> (j_in_blk % 8)) & 0x1;
        const float c = d_turbo_centroids_3bit_fattn[low2 | (hi1 << 2)];

        // Inverse FWHT: 5 intra-warp shfl passes + 2 cross-warp smem passes.
        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        // Normalize, apply signs1, undo InnerQ scaling, multiply by norm, cast to fp16
        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + g * 128);
        __syncthreads();
    }
}

// turbo4 K dequant with inverse FWHT: produces K in original (unrotated) domain
// so Q does NOT need pre-rotation. 128 threads per block, loops over 128-element turbo4 blocks.
static __global__ void k_turbo4_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO4);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo4_0 * blk = (const block_turbo4_0 *)src_row + blk_idx;
        const float norm = __half2float(blk->norm);

        const uint8_t idx = (tid & 1) ? (blk->qs[tid / 2] >> 4) : (blk->qs[tid / 2] & 0xF);

        float val = fwht128_butterfly_inplace(d_turbo_centroids_4bit_fattn[idx] * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo2 K dequant with inverse FWHT: produces K in original (unrotated) domain.
// 128 threads per block, loops over 128-element FWHT groups (each group spans
// 4 turbo2 storage blocks of 32 elements; all 4 blocks share the same norm).
static __global__ void k_turbo2_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_groups = (int)(ne0 / 128);
    constexpr int blocks_per_group = 128 / QK_TURBO2; // 4

    for (int g = 0; g < n_groups; g++) {
        const int j_in_grp = tid;            // 0..127
        const int blk_in_grp = j_in_grp / QK_TURBO2;  // 0..3
        const int j_in_blk  = j_in_grp % QK_TURBO2;   // 0..31

        const block_turbo2_0 * blk = (const block_turbo2_0 *)src_row + g * blocks_per_group + blk_in_grp;
        const float norm = __half2float(blk->norm);

        const uint8_t idx = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
        const float c = d_turbo_centroids_2bit_fattn[idx];

        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + g * 128);
        __syncthreads();
    }
}

// turbo3_tcq K dequant with inverse FWHT: produces K in original (unrotated) domain.
// 128 threads per block, loops over 128-element TCQ blocks (1 block per FWHT group).
static __global__ void k_turbo3_tcq_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO3_TCQ);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo3_tcq * blk = (const block_turbo3_tcq *)src_row + blk_idx;
        const float norm = __half2float(blk->norm) * alpha;

        // Sliding window decode: read 9-bit state from bitstream at bit offset tid*3
        const int bit_pos = tid * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        const float c = d_turbo3_tcq_codebook_fattn[state];

        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo4_tcq K dequant with inverse FWHT: produces K in original (unrotated) domain.
static __global__ void k_turbo4_tcq_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO4_TCQ);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo4_tcq * blk = (const block_turbo4_tcq *)src_row + blk_idx;
        const float norm = __half2float(blk->norm) * alpha;

        const int bit_pos = tid * 4;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x3FF;
        const float c = d_turbo4_tcq_codebook_fattn[state];

        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo2_tcq K dequant with inverse FWHT: produces K in original (unrotated) domain.
static __global__ void k_turbo2_tcq_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO2_TCQ);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo2_tcq * blk = (const block_turbo2_tcq *)src_row + blk_idx;
        const float norm = __half2float(blk->norm) * alpha;

        // Sliding window decode: read 8-bit state from bitstream at bit offset tid*2
        const int bit_pos = tid * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        const float c = d_turbo2_tcq_codebook_fattn[state];

        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// q8_0 K dequant to f16 in TKHE layout, matching the turbo K dequant kernels.
// Used at D=512 when K=q8_0 paired with V=turbo: produces (F16, F16) for the FA dispatch
// and bypasses the (Q8_0, TURBO*) D=512 native VEC templates which have buggy SASS on
// sm_120 PTX-JIT for some K/V combos. Q8_0 is in original (unrotated) domain → output too.
// 1 thread per element, 1 block per (token, head, batch).
static __global__ void k_q8_0_dequant_f16_tkhe(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + row * nb1 + head * nb2;
    const int blk_idx = j / QK8_0;
    const int j_in_blk = j % QK8_0;
    const block_q8_0 * blk = (const block_q8_0 *)src_row + blk_idx;
    const float d = __half2float(blk->d);
    const float val = d * (float)blk->qs[j_in_blk];

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

// Persistent Q rotation buffer per device (shared between prefill and decode paths)
static float * q_rot_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  q_rot_buf_size[GGML_CUDA_MAX_DEVICES] = {};

// Persistent K/V fp16 dequant buffers per device (shared between prefill and decode paths)
static half * kv_dequant_k_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_k_buf_size[GGML_CUDA_MAX_DEVICES] = {};
static half * kv_dequant_v_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_v_buf_size[GGML_CUDA_MAX_DEVICES] = {};

// === FWHT rotation kernels for pre-rotate-queries approach ===
// Forward rotation on Q before attention (both prefill and decode paths).
// One block per 128-element group, 128 threads per block.
static __global__ void k_turbo_fwht_forward(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t n_elements) {
    const int64_t offset = blockIdx.x * 128;
    if (offset >= n_elements) return;

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;

    __shared__ float buf[128];

    // InnerQ: apply inverse channel scale to Q before rotation
    float val = src[offset + threadIdx.x] * d_innerq_channel_scale_inv_fattn[threadIdx.x] * s1[threadIdx.x];

    val = fwht128_butterfly_inplace(val, buf);

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    val = val * inv_sqrt_128 * s2[threadIdx.x];
    dst[offset + threadIdx.x] = val;

    // Q² calibration: accumulate per-position squared values
    if (d_q_calibrate_fattn) {
        atomicAdd_double(&d_q_channel_sq_fattn[threadIdx.x], (double)(val * val));
        if (threadIdx.x == 0) atomicAdd(&d_q_channel_count_fattn, 1);
    }
}

static inline bool ggml_cuda_fattn_is_classic_non_q8_type(const ggml_type type) {
    return type == GGML_TYPE_BF16 ||
           type == GGML_TYPE_Q4_0 ||
           type == GGML_TYPE_Q4_1 ||
           type == GGML_TYPE_Q5_0 ||
           type == GGML_TYPE_Q5_1 ||
           type == GGML_TYPE_Q6_0 ||
           type == GGML_TYPE_Q6_1 ||
           type == GGML_TYPE_Q3_0 ||
           type == GGML_TYPE_Q3_1 ||
           type == GGML_TYPE_Q2_0 ||
           type == GGML_TYPE_Q2_1;
}

static void ggml_cuda_fattn_materialize_to_f16(
        const ggml_tensor * src, half * dst, cudaStream_t stream, ggml_tensor & src_f16) {
    const size_t bs = ggml_blck_size(src->type);
    const size_t ts = ggml_type_size(src->type);

    src_f16 = *src;
    src_f16.type = GGML_TYPE_F16;
    src_f16.data = dst;
    src_f16.nb[0] = sizeof(half);

    if (ggml_is_contiguously_allocated(src)) {
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(src->type);
        GGML_ASSERT(to_fp16 != nullptr);
        to_fp16(src->data, dst, ggml_nelements(src), stream);

        src_f16.nb[1] = src->nb[1] * bs * sizeof(half) / ts;
        src_f16.nb[2] = src->nb[2] * bs * sizeof(half) / ts;
        src_f16.nb[3] = src->nb[3] * bs * sizeof(half) / ts;
    } else {
        GGML_ASSERT(src->nb[0] == ts);
        const to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(src->type);
        GGML_ASSERT(to_fp16 != nullptr);
        const int64_t s01 = src->nb[1] / ts;
        const int64_t s02 = src->nb[2] / ts;
        const int64_t s03 = src->nb[3] / ts;
        to_fp16(src->data, dst, src->ne[0], src->ne[1], src->ne[2], src->ne[3], s01, s02, s03, stream);

        src_f16.nb[1] = src->ne[0] * sizeof(half);
        src_f16.nb[2] = src->ne[1] * src_f16.nb[1];
        src_f16.nb[3] = src->ne[2] * src_f16.nb[2];
    }
}

static void ggml_cuda_turbo_prefill_attend(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    load_tcq_decode_alpha(ctx.device);
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool turbo_k = K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0 || K->type == GGML_TYPE_TURBO4_TCQ || K->type == GGML_TYPE_TURBO3_TCQ || K->type == GGML_TYPE_TURBO2_TCQ;
    const bool turbo_v = V->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0 || V->type == GGML_TYPE_TURBO4_TCQ || V->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO2_TCQ;
    const bool classic_non_q8_v = !turbo_v && ggml_cuda_fattn_is_classic_non_q8_type(V->type);

    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    half * k_fp16 = nullptr;
    half * v_fp16 = nullptr;
    bool v_f16_layout_set = false;

    ggml_tensor K_f16 = *K;
    ggml_tensor V_f16 = *V;

    // Allocate and dequant K to fp16 (turbo2, turbo3, or turbo4)
    if (turbo_k) {
        // Size for full cache (kv_size from root) so we never realloc mid-session.
        const ggml_tensor * k_root = K;
        while (k_root->view_src) k_root = k_root->view_src;
        const size_t k_size = (size_t)k_root->ne[0] * k_root->ne[1] * k_root->ne[2] * sizeof(half);
        if (k_size > kv_dequant_k_buf_size[device]) {
            if (kv_dequant_k_buf[device]) CUDA_CHECK(cudaFree(kv_dequant_k_buf[device]));
            CUDA_CHECK(cudaMalloc(&kv_dequant_k_buf[device], k_size));
            kv_dequant_k_buf_size[device] = k_size;
        }
        k_fp16 = kv_dequant_k_buf[device];
        dim3 grid_k(K->ne[1], K->ne[2], K->ne[3]);
        if (K->type == GGML_TYPE_TURBO2_0) {
            k_turbo2_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO3_0) {
            k_turbo3_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO4_TCQ) {
            load_tcq4_codebook_fattn(device, "TCQ4 K prefill");
            k_turbo4_tcq_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
        } else if (K->type == GGML_TYPE_TURBO3_TCQ) {
            {
                static bool tcq_fattn_k_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                if (!tcq_fattn_k_cb_loaded[device]) {
                    tcq_fattn_k_cb_loaded[device] = true;
                    const char *cb_path = getenv("TURBO_TCQ_CB");
                    if (cb_path) {
                        float cb[512];
                        FILE *f = fopen(cb_path, "rb");
                        if (f && fread(cb, sizeof(float), 512, f) == 512) {
                            fclose(f);
                            cudaMemcpyToSymbol(d_turbo3_tcq_codebook_fattn, cb, 512*sizeof(float));
                            fprintf(stderr, "TCQ K prefill: loaded codebook from %s (device %d)\n", cb_path, device);
                        } else {
                            if (f) fclose(f);
                            fprintf(stderr, "TCQ K prefill: FAILED to load codebook from %s\n", cb_path);
                        }
                    }
                }
            }
            k_turbo3_tcq_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
        } else if (K->type == GGML_TYPE_TURBO2_TCQ) {
            {
                static bool tcq2_fattn_k_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                if (!tcq2_fattn_k_cb_loaded[device]) {
                    tcq2_fattn_k_cb_loaded[device] = true;
                    const char *cb_path = getenv("TURBO_TCQ_CB2");
                    if (cb_path) {
                        float cb[256];
                        FILE *f = fopen(cb_path, "rb");
                        if (f && fread(cb, sizeof(float), 256, f) == 256) {
                            fclose(f);
                            cudaMemcpyToSymbol(d_turbo2_tcq_codebook_fattn, cb, 256*sizeof(float));
                            fprintf(stderr, "TCQ2 K prefill: loaded 2-bit codebook from %s (device %d)\n", cb_path, device);
                        } else {
                            if (f) fclose(f);
                            fprintf(stderr, "TCQ2 K prefill: FAILED to load codebook from %s\n", cb_path);
                        }
                    }
                }
            }
            k_turbo2_tcq_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
        } else {
            // turbo4 K: inverse FWHT dequant → produces K in original domain (no Q rotation needed)
            k_turbo4_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        }
    }

    // Allocate and materialize V to fp16 (turbo2, turbo3, turbo4, or classic non-q8).
    if (turbo_v || classic_non_q8_v) {
        // Size for full cache (kv_size from root) so we never realloc mid-session.
        const ggml_tensor * v_root = V;
        while (v_root->view_src) v_root = v_root->view_src;
        const size_t v_size = (size_t)v_root->ne[0] * v_root->ne[1] * v_root->ne[2] * sizeof(half);
        if (v_size > kv_dequant_v_buf_size[device]) {
            if (kv_dequant_v_buf[device]) CUDA_CHECK(cudaFree(kv_dequant_v_buf[device]));
            CUDA_CHECK(cudaMalloc(&kv_dequant_v_buf[device], v_size));
            kv_dequant_v_buf_size[device] = v_size;
        }
        v_fp16 = kv_dequant_v_buf[device];
        dim3 grid_v(V->ne[1], V->ne[2], V->ne[3]);
        if (V->type == GGML_TYPE_TURBO2_0) {
            k_turbo2_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TURBO3_0) {
            k_turbo3_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TURBO4_TCQ) {
            load_tcq4_codebook_fattn(device, "TCQ4 V decode");
            k_turbo4_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]));
        } else if (V->type == GGML_TYPE_TURBO3_TCQ) {
            // Runtime codebook loading for 3-bit V decode (in case K is a different type)
            {
                static bool tcq_fattn_v_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                if (!tcq_fattn_v_cb_loaded[device]) {
                    tcq_fattn_v_cb_loaded[device] = true;
                    const char *cb_path = getenv("TURBO_TCQ_CB");
                    if (cb_path) {
                        float cb[512];
                        FILE *f = fopen(cb_path, "rb");
                        if (f && fread(cb, sizeof(float), 512, f) == 512) {
                            fclose(f);
                            cudaMemcpyToSymbol(d_turbo3_tcq_codebook_fattn, cb, 512*sizeof(float));
                            fprintf(stderr, "TCQ V decode: loaded 3-bit codebook from %s (device %d)\n", cb_path, device);
                        } else {
                            if (f) fclose(f);
                        }
                    }
                }
            }
            k_turbo3_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]));
        } else if (V->type == GGML_TYPE_TURBO2_TCQ) {
            // Runtime codebook loading for 2-bit V decode (in case K is a different type)
            {
                static bool tcq2_fattn_v_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                if (!tcq2_fattn_v_cb_loaded[device]) {
                    tcq2_fattn_v_cb_loaded[device] = true;
                    const char *cb_path = getenv("TURBO_TCQ_CB2");
                    if (cb_path) {
                        float cb[256];
                        FILE *f = fopen(cb_path, "rb");
                        if (f && fread(cb, sizeof(float), 256, f) == 256) {
                            fclose(f);
                            cudaMemcpyToSymbol(d_turbo2_tcq_codebook_fattn, cb, 256*sizeof(float));
                            fprintf(stderr, "TCQ2 V decode: loaded 2-bit codebook from %s (device %d)\n", cb_path, device);
                        } else {
                            if (f) fclose(f);
                        }
                    }
                }
            }
            k_turbo2_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]));
        } else if (V->type == GGML_TYPE_TURBO4_0) {
            k_turbo4_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else {
            ggml_cuda_fattn_materialize_to_f16(V, v_fp16, stream, V_f16);
            v_f16_layout_set = true;
        }
    }

    if (k_fp16) {
        K_f16.type = GGML_TYPE_F16;
        K_f16.data = k_fp16;
        K_f16.nb[0] = sizeof(half);
        K_f16.nb[1] = K->ne[0] * K->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
        K_f16.nb[2] = K->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
        K_f16.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
    }

    if (v_fp16 && !v_f16_layout_set) {
        V_f16.type = GGML_TYPE_F16;
        V_f16.data = v_fp16;
        V_f16.nb[0] = sizeof(half);
        V_f16.nb[1] = V->ne[0] * V->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
        V_f16.nb[2] = V->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
        V_f16.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
    }

    // Rotate Q for turbo pre-rotate-queries (only when K is in rotated space)
    // turbo4 K is dequanted via inverse FWHT → original domain, so Q stays unrotated
    const ggml_tensor * Q = dst->src[0];
    float * q_rotated = nullptr;
    if (turbo_k && K->type != GGML_TYPE_TURBO4_0 && Q->ne[0] % 128 == 0) {
        const size_t q_size = ggml_nelements(Q) * sizeof(float);
        if (q_size > q_rot_buf_size[device]) {
            if (q_rot_buf[device]) CUDA_CHECK(cudaFree(q_rot_buf[device]));
            CUDA_CHECK(cudaMalloc(&q_rot_buf[device], q_size));
            q_rot_buf_size[device] = q_size;
        }
        q_rotated = q_rot_buf[device];
        const int64_t n_q_groups = ggml_nelements(Q) / 128;
        k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
            (const float *)Q->data, q_rotated, ggml_nelements(Q));
    }

    // Temporarily swap src pointers to fp16 K/V and rotated Q
    ggml_tensor * orig_q = dst->src[0];
    ggml_tensor * orig_k = dst->src[1];
    ggml_tensor * orig_v = dst->src[2];

    ggml_tensor Q_rot;
    if (q_rotated) {
        Q_rot = *Q;
        Q_rot.data = q_rotated;
        dst->src[0] = &Q_rot;
    }
    dst->src[1] = k_fp16 ? &K_f16 : orig_k;
    dst->src[2] = v_fp16 ? &V_f16 : orig_v;

    // Dispatch to MMA kernel (sees rotated Q, fp16 K/V, uses tensor cores)
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);

    // Restore original tensor pointers
    dst->src[0] = orig_q;
    dst->src[1] = orig_k;
    dst->src[2] = orig_v;

    // K/V fp16 buffers are persistent (grow-only), no free needed
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

#define FATTN_VEC_CASES_ALL_D_512(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

static inline bool ggml_cuda_fattn_is_turbo_kv_type(const ggml_type type) {
    return type == GGML_TYPE_TURBO2_0   ||
           type == GGML_TYPE_TURBO3_0   ||
           type == GGML_TYPE_TURBO4_0   ||
           type == GGML_TYPE_TURBO4_TCQ ||
           type == GGML_TYPE_TURBO3_TCQ ||
           type == GGML_TYPE_TURBO2_TCQ;
}

static inline bool ggml_cuda_turbo_prefill_mma_can_make_f16(const ggml_type type) {
    return type == GGML_TYPE_F16 || ggml_cuda_fattn_is_turbo_kv_type(type);
}

static inline int ggml_cuda_fattn_kv_rank(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:         return 0;
        case GGML_TYPE_BF16:        return 1;
        case GGML_TYPE_Q8_0:        return 2;
        case GGML_TYPE_Q6_1:        return 3;
        case GGML_TYPE_Q6_0:        return 4;
        case GGML_TYPE_Q5_1:        return 5;
        case GGML_TYPE_Q5_0:        return 6;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_TURBO4_TCQ:
        case GGML_TYPE_TURBO4_0:    return 7;
        case GGML_TYPE_Q4_0:        return 8;
        case GGML_TYPE_Q3_1:
        case GGML_TYPE_TURBO3_TCQ:
        case GGML_TYPE_TURBO3_0:    return 9;
        case GGML_TYPE_Q3_0:        return 10;
        case GGML_TYPE_Q2_1:
        case GGML_TYPE_TURBO2_TCQ:
        case GGML_TYPE_TURBO2_0:    return 11;
        case GGML_TYPE_Q2_0:        return 12;
        default:                    return -1;
    }
}

static inline bool ggml_cuda_fattn_is_ranked_kv_type(const ggml_type type) {
    return ggml_cuda_fattn_kv_rank(type) >= 0;
}

static inline ggml_type ggml_cuda_fattn_canonical_kv_type(const ggml_type type) {
    return type == GGML_TYPE_F32 ? GGML_TYPE_F16 : type;
}

#if !defined(GGML_CUDA_FA_ALL_QUANTS) && !defined(GGML_CUDA_FA_HALF_QUANTS)
static inline int ggml_cuda_fattn_default_kv_tier(const ggml_type type) {
    switch (ggml_cuda_fattn_canonical_kv_type(type)) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return 0;
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q6_1:
        case GGML_TYPE_Q6_0:
            return 1;
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
            return 2;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return 3;
        case GGML_TYPE_Q3_1:
        case GGML_TYPE_Q3_0:
            return 4;
        case GGML_TYPE_Q2_1:
        case GGML_TYPE_Q2_0:
            return 5;
        default:
            return -1;
    }
}

static inline bool ggml_cuda_fattn_default_pair_compiled(const ggml_type type_K, const ggml_type type_V) {
    if (ggml_cuda_fattn_is_turbo_kv_type(type_K) || ggml_cuda_fattn_is_turbo_kv_type(type_V)) {
        return false;
    }

    const int rank_K = ggml_cuda_fattn_kv_rank(type_K);
    const int rank_V = ggml_cuda_fattn_kv_rank(type_V);
    const int tier_K = ggml_cuda_fattn_default_kv_tier(type_K);
    const int tier_V = ggml_cuda_fattn_default_kv_tier(type_V);

    if (rank_K < 0 || rank_V < 0 || tier_K < 0 || tier_V < 0) {
        return false;
    }

    if ((type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16) &&
        (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) &&
        type_K != type_V) {
        return false;
    }

    return rank_K <= rank_V && tier_V - tier_K <= 2;
}
#endif

static inline bool ggml_cuda_fattn_pair_compiled(const ggml_type type_K_in, const ggml_type type_V_in) {
    const ggml_type type_K = ggml_cuda_fattn_canonical_kv_type(type_K_in);
    const ggml_type type_V = ggml_cuda_fattn_canonical_kv_type(type_V_in);

#if defined(GGML_CUDA_FA_ALL_QUANTS)
    return ggml_cuda_fattn_is_ranked_kv_type(type_K) &&
           ggml_cuda_fattn_is_ranked_kv_type(type_V);
#elif defined(GGML_CUDA_FA_HALF_QUANTS)
    const int rank_K = ggml_cuda_fattn_kv_rank(type_K);
    const int rank_V = ggml_cuda_fattn_kv_rank(type_V);

    return rank_K >= 0 && rank_V >= 0 &&
           (rank_K <= rank_V || type_K == GGML_TYPE_F16 || type_V == GGML_TYPE_F16);
#else
    return ggml_cuda_fattn_default_pair_compiled(type_K, type_V);
#endif
}

const char * ggml_cuda_fa_build_policy() {
#if defined(GGML_CUDA_FA_ALL_QUANTS)
    return "all";
#elif defined(GGML_CUDA_FA_HALF_QUANTS)
    return "half";
#else
    return "default";
#endif
}

bool ggml_cuda_fa_pair_compiled(ggml_type type_K, ggml_type type_V) {
    return ggml_cuda_fattn_pair_compiled(type_K, type_V);
}

static inline bool ggml_cuda_fattn_prefers_native_vec_for_turbo_k_classic_v(
        const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V) {
    return ggml_cuda_fattn_is_turbo_kv_type(K->type) &&
           !ggml_cuda_fattn_is_turbo_kv_type(V->type) &&
           ggml_cuda_fattn_is_classic_non_q8_type(V->type) &&
           Q->ne[0] <= 512 &&
           Q->ne[0] % 64 == 0 &&
           ggml_cuda_fattn_pair_compiled(K->type, V->type);
}

static inline bool ggml_cuda_fattn_prefill_mma_can_materialize_turbo_k_classic_v(
        const ggml_tensor * K, const ggml_tensor * V) {
    return ggml_cuda_fattn_is_turbo_kv_type(K->type) &&
           !ggml_cuda_fattn_is_turbo_kv_type(V->type) &&
           ggml_cuda_fattn_is_classic_non_q8_type(V->type);
}

static inline bool ggml_cuda_fattn_is_turbo_v_decode_unsafe_k_type(const ggml_type type) {
    return type == GGML_TYPE_Q8_0 ||
           ggml_cuda_fattn_is_classic_non_q8_type(type);
}

// Shape guard for the effective K/V pair after Turbo V decode-dequant.
// D>=256 with classic-or-q8 K/f16 V is unsafe on the vec path.
// Only applied when V was actually decoded from Turbo — explicit q5_0/f16
// at D>=256 is unaffected. D=128 is safe on vec and not gated.
static inline bool ggml_cuda_fattn_effective_vec_shape_unsafe(
        const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V) {
    return Q->ne[0] >= 256 &&
           ggml_cuda_fattn_is_turbo_v_decode_unsafe_k_type(K->type) &&
           V->type == GGML_TYPE_F16;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#if defined(GGML_CUDA_FA_ALL_QUANTS)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_Q2_0)
#elif defined(GGML_CUDA_FA_HALF_QUANTS)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_Q2_0)
#else
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q6_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q6_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q3_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q2_1)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_Q2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_Q2_0)
#endif

    fprintf(stderr,
        "CUDA FA vec dispatch missing compiled K/V pair: K=%s V=%s D=%lld. "
        "Use standard q/KVarN fallback cache types in the default build, or rebuild with "
        "GGML_CUDA_FA_HALF_QUANTS or GGML_CUDA_FA_ALL_QUANTS for Turbo/TCQ or arbitrary pairs.\n",
        ggml_type_name(K->type), ggml_type_name(V->type), (long long) Q->ne[0]);
    GGML_ABORT("missing CUDA FA vec K/V pair");
}

// Very verbose CUDA FA route tracing. Intentionally env-gated; do not enable
// for performance measurements or wire into normal -lv verbosity levels.
static inline bool ggml_cuda_fattn_route_debug_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_FA_ROUTE_DEBUG");
        return e != nullptr && atoi(e) != 0;
    }();
    return enabled;
}

static inline bool ggml_cuda_fattn_ignore_uncompiled_pairs() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_FA_IGNORE_UNCOMPILED_PAIRS");
        return e != nullptr && atoi(e) != 0;
    }();
    return enabled;
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
};

static const char * ggml_cuda_fattn_kernel_name(const best_fattn_kernel kernel) {
    switch (kernel) {
        case BEST_FATTN_KERNEL_NONE:     return "NONE";
        case BEST_FATTN_KERNEL_VEC:      return "VEC";
        case BEST_FATTN_KERNEL_TILE:     return "TILE";
        case BEST_FATTN_KERNEL_WMMA_F16: return "WMMA_F16";
        case BEST_FATTN_KERNEL_MMA_F16:  return "MMA_F16";
    }
    return "UNKNOWN";
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst, bool allow_vec = true) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst); GGML_UNUSED(allow_vec);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_Q6_1:
        case GGML_TYPE_Q3_0:
        case GGML_TYPE_Q3_1:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q2_1:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TURBO4_TCQ:
        case GGML_TYPE_TURBO3_TCQ:
        case GGML_TYPE_TURBO2_TCQ:
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    const bool pair_compiled = ggml_cuda_fattn_pair_compiled(K->type, V->type);
    if (!pair_compiled && !ggml_cuda_fattn_ignore_uncompiled_pairs()) {
        return BEST_FATTN_KERNEL_NONE;
    }
    allow_vec = allow_vec && pair_compiled;

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // TurboQuant: only the vec kernel has native turbo dequant support.
    if (K->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO2_0 ||
        K->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO3_0 ||
        K->type == GGML_TYPE_TURBO4_0 || V->type == GGML_TYPE_TURBO4_0 ||
        K->type == GGML_TYPE_TURBO4_TCQ || V->type == GGML_TYPE_TURBO4_TCQ ||
        K->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO3_TCQ ||
        K->type == GGML_TYPE_TURBO2_TCQ || V->type == GGML_TYPE_TURBO2_TCQ) {
        if (allow_vec && Q->ne[0] <= 512 && Q->ne[0] % 64 == 0)
            return BEST_FATTN_KERNEL_VEC;
        return BEST_FATTN_KERNEL_NONE;
    }

    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (allow_vec && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (allow_vec && Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (allow_vec && Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (allow_vec && !gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (allow_vec && can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use the WMMA kernel if possible:
    if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 192 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        if (allow_vec && can_use_vector_kernel && Q->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (allow_vec && Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (allow_vec && Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

struct ggml_cuda_fattn_route_plan {
    ggml_type effective_type_K;
    ggml_type effective_type_V;
    ggml_cuda_fattn_kvarn_plan kvarn_plan;
    bool decode_dequant;
    bool decode_dequant_K;
    bool decode_dequant_V;
    bool need_generic_f16_K;
    bool need_generic_f16_V;
    bool allow_vec;
    bool unsafe_vec_after_turbo_v_decode;
    bool native_kvarn_mma;
    bool materialize_kvarn_f16;
    best_fattn_kernel kernel;
};

static ggml_cuda_fattn_route_plan ggml_cuda_fattn_make_route_plan(const int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(dst->src[0] != nullptr);
    GGML_ASSERT(dst->src[1] != nullptr);
    GGML_ASSERT(dst->src[2] != nullptr);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    ggml_cuda_fattn_route_plan plan = {};

    if (ggml_cuda_fattn_kvarn_supported(device, dst, &plan.kvarn_plan)) {
        plan.effective_type_K = GGML_TYPE_F16;
        plan.effective_type_V = GGML_TYPE_F16;
        plan.allow_vec = false;
        plan.native_kvarn_mma = true;
        plan.kernel = BEST_FATTN_KERNEL_MMA_F16;
        return plan;
    }

    if (ggml_cuda_fattn_kvarn_view_supported(device, dst, &plan.kvarn_plan)) {
        ggml_tensor K_eff = *K;
        ggml_tensor V_eff = *V;
        K_eff.type = GGML_TYPE_F16;
        V_eff.type = GGML_TYPE_F16;

        ggml_tensor dst_eff = *dst;
        ggml_tensor * src_eff[GGML_MAX_SRC];
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            src_eff[i] = dst->src[i];
        }
        src_eff[1] = &K_eff;
        src_eff[2] = &V_eff;
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            dst_eff.src[i] = src_eff[i];
        }

        plan.effective_type_K = GGML_TYPE_F16;
        plan.effective_type_V = GGML_TYPE_F16;
        plan.allow_vec = true;
        plan.materialize_kvarn_f16 = true;
        plan.kernel = ggml_cuda_get_best_fattn_kernel(device, &dst_eff, plan.allow_vec);
        if (plan.kernel == BEST_FATTN_KERNEL_NONE) {
            // KVarN views only expose 128/256/512-wide heads. The stock selector
            // rejects D=512 without GQA, but the F16/F16 vec kernel has a compiled
            // D=512 case and is the only stock FA fallback for that shape.
            plan.kernel = BEST_FATTN_KERNEL_VEC;
        }
        return plan;
    }

    if (ggml_cuda_fattn_kvarn_uses_views(dst)) {
        plan.kernel = BEST_FATTN_KERNEL_NONE;
        return plan;
    }

    const bool turbo_kv =
        ggml_cuda_fattn_is_turbo_kv_type(K->type) ||
        ggml_cuda_fattn_is_turbo_kv_type(V->type);

#if defined(GGML_USE_HIP)
    const bool hip_native_tcq_decode =
        K->type == GGML_TYPE_TURBO4_TCQ || K->type == GGML_TYPE_TURBO3_TCQ || K->type == GGML_TYPE_TURBO2_TCQ ||
        V->type == GGML_TYPE_TURBO4_TCQ || V->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO2_TCQ;
#else
    const bool hip_native_tcq_decode = false;
#endif

    const bool turbo_decode_native = getenv("GGML_TURBO_DECODE_NATIVE") != nullptr;
    const bool prefer_native_vec =
        ggml_cuda_fattn_prefers_native_vec_for_turbo_k_classic_v(Q, K, V);

    const bool turbo_k_only = ggml_cuda_fattn_is_turbo_kv_type(K->type);
    const bool turbo_v_only = ggml_cuda_fattn_is_turbo_kv_type(V->type);

    const bool classic_non_q8_K_turbo_V =
        ggml_cuda_fattn_is_classic_non_q8_type(K->type) &&
        turbo_v_only &&
        !turbo_k_only;

    // Decode-dequant policy:
    // - Compiled Turbo K + classic non-q8 V fallback routes stay native vec
    //   instead of taking generic f16 materialization after Turbo K decode.
    // - D <= 256: Turbo K/V sides may be decoded to f16.
    // - D = 512: f16/q8_0/Turbo pairs keep the existing f16 route.
    // - D = 512 classic non-q8 K + Turbo V decodes Turbo V to f16 and keeps
    //   K in its original classic type for generic f16 conversion.
    // - D > 512: unchanged.
    const bool k_f16_q8_or_turbo =
        K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || turbo_k_only;
    const bool v_f16_q8_or_turbo =
        V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || turbo_v_only;

    plan.decode_dequant =
        !hip_native_tcq_decode &&
        !turbo_decode_native &&
        !prefer_native_vec &&
        turbo_kv &&
        (Q->ne[0] <= 256 ||
         (Q->ne[0] <= 512 && k_f16_q8_or_turbo && v_f16_q8_or_turbo) ||
         (Q->ne[0] == 512 && classic_non_q8_K_turbo_V));

    plan.decode_dequant_K =
        plan.decode_dequant &&
        (turbo_k_only || (K->type == GGML_TYPE_Q8_0 && Q->ne[0] > 256));

    plan.decode_dequant_V =
        plan.decode_dequant &&
        (turbo_v_only || (V->type == GGML_TYPE_Q8_0 && Q->ne[0] > 256));

    plan.effective_type_K = plan.decode_dequant_K ? GGML_TYPE_F16 : K->type;
    plan.effective_type_V = plan.decode_dequant_V ? GGML_TYPE_F16 : V->type;

    ggml_tensor K_eff = *K;
    ggml_tensor V_eff = *V;

    if (plan.decode_dequant_K) {
        K_eff.type = GGML_TYPE_F16;
        K_eff.nb[0] = sizeof(half);
        K_eff.nb[1] = K->ne[0] * K->ne[2] * sizeof(half);
        K_eff.nb[2] = K->ne[0] * sizeof(half);
        K_eff.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
    }

    if (plan.decode_dequant_V) {
        V_eff.type = GGML_TYPE_F16;
        V_eff.nb[0] = sizeof(half);
        V_eff.nb[1] = V->ne[0] * V->ne[2] * sizeof(half);
        V_eff.nb[2] = V->ne[0] * sizeof(half);
        V_eff.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
    }

    ggml_tensor dst_eff = *dst;
    ggml_tensor * src_eff[GGML_MAX_SRC];
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        src_eff[i] = dst->src[i];
    }
    src_eff[1] = &K_eff;
    src_eff[2] = &V_eff;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        dst_eff.src[i] = src_eff[i];
    }

    // If V was decoded from Turbo to f16 and the effective pair is
    // classic-or-q8 K/f16 at D>=256, the vec path is unsafe. Only gate vec
    // for Turbo-originated f16 V — explicit q5_0/f16 at D>=256 is unaffected.
    // Disable vec so the existing kernel selector picks MMA_F16 or tile with
    // generic f16 K conversion. D=128 is fine on vec and is not affected.
    plan.unsafe_vec_after_turbo_v_decode =
        plan.decode_dequant_V &&
        ggml_cuda_fattn_effective_vec_shape_unsafe(Q, &K_eff, &V_eff);

    plan.allow_vec = !plan.unsafe_vec_after_turbo_v_decode;

    plan.kernel = ggml_cuda_get_best_fattn_kernel(device, &dst_eff, plan.allow_vec);

    plan.need_generic_f16_K = false;
    plan.need_generic_f16_V = false;
    switch (plan.kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_WMMA_F16:
        case BEST_FATTN_KERNEL_MMA_F16:
            plan.need_generic_f16_K = !plan.decode_dequant_K && plan.effective_type_K != GGML_TYPE_F16;
            plan.need_generic_f16_V = !plan.decode_dequant_V && plan.effective_type_V != GGML_TYPE_F16;
            break;
        case BEST_FATTN_KERNEL_VEC:
            plan.need_generic_f16_K = plan.effective_type_K == GGML_TYPE_F32;
            plan.need_generic_f16_V = plan.effective_type_V == GGML_TYPE_F32;
            break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    if (ggml_cuda_fattn_route_debug_enabled()) {
        fprintf(stderr,
            "CUDA_FA_ROUTE_PLAN "
            "device=%d "
            "Q=[%lld,%lld,%lld,%lld] "
            "Kraw=%s Vraw=%s "
            "Kshape=[%lld,%lld,%lld,%lld] Vshape=[%lld,%lld,%lld,%lld] "
            "Keff=%s Veff=%s "
            "turbo_kv=%d "
            "prefer_native_vec=%d "
            "decode=%d dk=%d dv=%d "
            "unsafe_vec_after_turbo_v_decode=%d allow_vec=%d "
            "kernel=%s "
            "need_f16_K=%d need_f16_V=%d "
            "Knb=[%lld,%lld,%lld,%lld] Vnb=[%lld,%lld,%lld,%lld] "
            "Keff_nb=[%lld,%lld,%lld,%lld] Veff_nb=[%lld,%lld,%lld,%lld]\n",
            device,
            (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
            ggml_type_name(K->type), ggml_type_name(V->type),
            (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3],
            (long long) V->ne[0], (long long) V->ne[1], (long long) V->ne[2], (long long) V->ne[3],
            ggml_type_name(plan.effective_type_K), ggml_type_name(plan.effective_type_V),
            (int) turbo_kv,
            (int) prefer_native_vec,
            (int) plan.decode_dequant,
            (int) plan.decode_dequant_K,
            (int) plan.decode_dequant_V,
            (int) plan.unsafe_vec_after_turbo_v_decode,
            (int) plan.allow_vec,
            ggml_cuda_fattn_kernel_name(plan.kernel),
            (int) plan.need_generic_f16_K,
            (int) plan.need_generic_f16_V,
            (long long) K->nb[0], (long long) K->nb[1], (long long) K->nb[2], (long long) K->nb[3],
            (long long) V->nb[0], (long long) V->nb[1], (long long) V->nb[2], (long long) V->nb[3],
            (long long) K_eff.nb[0], (long long) K_eff.nb[1], (long long) K_eff.nb[2], (long long) K_eff.nb[3],
            (long long) V_eff.nb[0], (long long) V_eff.nb[1], (long long) V_eff.nb[2], (long long) V_eff.nb[3]);
        fflush(stderr);
    }

    return plan;
}

static void ggml_cuda_flash_attn_ext_materialize_kvarn_f16(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_cuda_fattn_route_plan & plan) {
    ggml_cuda_pool & pool = ctx.pool();
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    GGML_ASSERT(K != nullptr && V != nullptr);
    GGML_ASSERT(plan.materialize_kvarn_f16);

    ggml_cuda_pool_alloc<half> K_f16(pool, ggml_nelements(K));
    ggml_cuda_pool_alloc<half> V_f16(pool, ggml_nelements(V));

    ggml_cuda_fattn_kvarn_materialize_side(ctx, plan.kvarn_plan.k, plan.kvarn_plan, K_f16.get(), /*emit_rotated=*/true);
    ggml_cuda_fattn_kvarn_materialize_side(ctx, plan.kvarn_plan.v, plan.kvarn_plan, V_f16.get(), /*emit_rotated=*/true);

    ggml_tensor K_mat = *K;
    ggml_tensor V_mat = *V;
    K_mat.data = K_f16.get();
    V_mat.data = V_f16.get();
    K_mat.type = GGML_TYPE_F16;
    V_mat.type = GGML_TYPE_F16;
    K_mat.op = GGML_OP_NONE;
    V_mat.op = GGML_OP_NONE;
    K_mat.view_src = nullptr;
    V_mat.view_src = nullptr;
    K_mat.view_offs = 0;
    V_mat.view_offs = 0;

    const int64_t k_head_dim = K->ne[0];
    const int64_t k_n_kv = K->ne[1];
    const int64_t k_n_heads = K->ne[2];
    K_mat.nb[0] = sizeof(half);
    K_mat.nb[2] = k_head_dim * sizeof(half);
    K_mat.nb[1] = k_head_dim * k_n_heads * sizeof(half);
    K_mat.nb[3] = k_head_dim * k_n_heads * k_n_kv * sizeof(half);

    const int64_t v_head_dim = V->ne[0];
    const int64_t v_n_kv = V->ne[1];
    const int64_t v_n_heads = V->ne[2];
    V_mat.nb[0] = sizeof(half);
    V_mat.nb[2] = v_head_dim * sizeof(half);
    V_mat.nb[1] = v_head_dim * v_n_heads * sizeof(half);
    V_mat.nb[3] = v_head_dim * v_n_heads * v_n_kv * sizeof(half);

    if (ggml_cuda_fattn_route_debug_enabled()) {
        const ggml_tensor * Q = dst->src[0];
        fprintf(stderr,
            "CUDA_FA_ROUTE_EXEC_DISPATCH kernel=%s materialize_kvarn_f16=1 "
            "Q=[%lld,%lld,%lld,%lld] K=[%lld,%lld,%lld,%lld] V=[%lld,%lld,%lld,%lld]\n",
            ggml_cuda_fattn_kernel_name(plan.kernel),
            (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
            (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3],
            (long long) V->ne[0], (long long) V->ne[1], (long long) V->ne[2], (long long) V->ne[3]);
        fflush(stderr);
    }

    ggml_tensor * orig_k = dst->src[1];
    ggml_tensor * orig_v = dst->src[2];
    dst->src[1] = &K_mat;
    dst->src[2] = &V_mat;

    switch (plan.kernel) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("no CUDA FA kernel selected for materialized KVarN fallback");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_WMMA_F16:
            ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }

    dst->src[1] = orig_k;
    dst->src[2] = orig_v;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_cuda_fattn_route_plan plan = ggml_cuda_fattn_make_route_plan(device, dst);

    const bool alloc_f16_K = plan.need_generic_f16_K;
    const bool alloc_f16_V = plan.need_generic_f16_V;

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, alloc_f16_K, alloc_f16_V);

    if (ggml_cuda_fattn_route_debug_enabled()) {
        fprintf(stderr,
            "CUDA_FA_ROUTE_ALLOC "
            "kernel=%s "
            "need_f16_K=%d need_f16_V=%d "
            "workspace_end_offset=%llu "
            "dst_data=%p f16_K=%p f16_V=%p f16_end=%p\n",
            ggml_cuda_fattn_kernel_name(plan.kernel),
            (int) plan.need_generic_f16_K,
            (int) plan.need_generic_f16_V,
            (unsigned long long) (f16_extra.end - (uintptr_t) dst->data),
            dst->data,
            (void *) f16_extra.K,
            (void *) f16_extra.V,
            (void *) f16_extra.end);
        fflush(stderr);
    }

    return f16_extra.end - (uintptr_t) dst->data;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const ggml_cuda_fattn_route_plan plan =
        ggml_cuda_fattn_make_route_plan(ctx.device, dst);

    if ((ggml_cuda_fattn_is_ranked_kv_type(plan.effective_type_K) ||
         ggml_cuda_fattn_is_ranked_kv_type(plan.effective_type_V)) &&
        !ggml_cuda_fattn_pair_compiled(plan.effective_type_K, plan.effective_type_V)) {
        if (!ggml_cuda_fattn_ignore_uncompiled_pairs()) {
            fprintf(stderr,
                "CUDA FA effective K/V pair was not compiled in this build: raw K=%s raw V=%s effective K=%s effective V=%s. "
                "Use standard q/KVarN fallback cache types in the default build, or rebuild with "
                "GGML_CUDA_FA_HALF_QUANTS or GGML_CUDA_FA_ALL_QUANTS for Turbo/TCQ or arbitrary pairs.\n",
                ggml_type_name(K->type), ggml_type_name(V->type),
                ggml_type_name(plan.effective_type_K), ggml_type_name(plan.effective_type_V));
            GGML_ABORT("CUDA FA effective K/V pair not compiled");
        }
        fprintf(stderr,
            "WARNING: GGML_CUDA_FA_IGNORE_UNCOMPILED_PAIRS=1: CUDA FA effective K/V pair was not compiled: "
            "raw K=%s raw V=%s effective K=%s effective V=%s. Continuing with non-vec fallback if available.\n",
            ggml_type_name(K->type), ggml_type_name(V->type),
            ggml_type_name(plan.effective_type_K), ggml_type_name(plan.effective_type_V));
    }

    if (plan.native_kvarn_mma) {
        if (ggml_cuda_fattn_route_debug_enabled()) {
            fprintf(stderr,
                "CUDA_FA_ROUTE_EXEC_DISPATCH kernel=MMA_KVARN need_f16_K=0 need_f16_V=0 "
                "Q=[%lld,%lld,%lld,%lld] K=[%lld,%lld,%lld,%lld] V=[%lld,%lld,%lld,%lld]\n",
                (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
                (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3],
                (long long) V->ne[0], (long long) V->ne[1], (long long) V->ne[2], (long long) V->ne[3]);
            fflush(stderr);
        }
        ggml_cuda_flash_attn_ext_mma_kvarn(ctx, dst);
        return;
    }

    if (plan.materialize_kvarn_f16) {
        ggml_cuda_flash_attn_ext_materialize_kvarn_f16(ctx, dst, plan);
        return;
    }

    // Turbo prefill: dequant to fp16 and use tensor core MMA for batched attention.
    // turbo4 K uses inverse FWHT during dequant — mixes centroids in float32 shmem before
    // fp16 cast, so precision is fine. turbo2/turbo3 use simple centroid×norm dequant.
    // Set TURBO_PREFILL_VEC=1 to force vec kernel for all turbo types (debug override).
    static const bool turbo_prefill_vec = [] {
        const char * e = getenv("TURBO_PREFILL_VEC");
        if (e) fprintf(stderr, "TURBO_PREFILL_VEC=%s: forcing vec prefill for turbo types\n", e);
        return e != nullptr;
    }();
    const bool turbo_kv =
        ggml_cuda_fattn_is_turbo_kv_type(K->type) ||
        ggml_cuda_fattn_is_turbo_kv_type(V->type);

    // Fused MMA turbo: reads raw turbo bytes directly in the MMA kernel, no intermediate fp16 buffers.
    // Fused straight TurboQuant matched K/V. Set GGML_TURBO_MMA_FUSED=0 to disable.
    static const bool turbo_mma_fused = [] {
        const char * e = getenv("GGML_TURBO_MMA_FUSED");
        if (e && atoi(e) == 0) {
            fprintf(stderr, "GGML_TURBO_MMA_FUSED=0: fused turbo MMA kernel disabled\n");
            return false;
        }
        return true;
    }();
    static const bool turbo_fa_debug = [] {
        const char * e = getenv("GGML_TURBO_FA_DEBUG");
        return e && atoi(e) != 0;
    }();
    const bool turbo_matched = K->type == V->type && turbo_kv;
    const bool turbo_mma_supported =
        turbo_matched &&
        (K->type == GGML_TYPE_TURBO4_0 ||
         K->type == GGML_TYPE_TURBO3_0 ||
         K->type == GGML_TYPE_TURBO2_0);
    if (turbo_mma_fused && turbo_mma_supported && Q->ne[1] <= 4 &&
        (Q->ne[0] == 128 || Q->ne[0] == 256) &&
        turing_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        cudaStream_t stream = ctx.stream();
        int device_fused;
        CUDA_CHECK(cudaGetDevice(&device_fused));

        if (turbo_fa_debug) {
            fprintf(stderr,
                "GGML_TURBO_FA_DEBUG: path=fused-mma K=%s V=%s Q=[%lld,%lld,%lld,%lld] K=[%lld,%lld,%lld,%lld] V=[%lld,%lld,%lld,%lld]\n",
                ggml_type_name(K->type), ggml_type_name(V->type),
                (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
                (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3],
                (long long) V->ne[0], (long long) V->ne[1], (long long) V->ne[2], (long long) V->ne[3]);
        }

        // Pre-rotate Q: all turbo K types stay in WHT-rotated domain, so Q must be rotated.
        ggml_tensor Q_rot_fused;
        ggml_tensor * orig_q_fused = nullptr;
        if (Q->ne[0] % 128 == 0) {
            const size_t q_size = ggml_nelements(Q) * sizeof(float);
            if (q_size > q_rot_buf_size[device_fused]) {
                if (q_rot_buf[device_fused]) CUDA_CHECK(cudaFree(q_rot_buf[device_fused]));
                CUDA_CHECK(cudaMalloc(&q_rot_buf[device_fused], q_size));
                q_rot_buf_size[device_fused] = q_size;
            }
            const int64_t n_q_groups = ggml_nelements(Q) / 128;
            k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
                (const float *)Q->data, q_rot_buf[device_fused], ggml_nelements(Q));
            Q_rot_fused = *Q;
            Q_rot_fused.data = q_rot_buf[device_fused];
            orig_q_fused = dst->src[0];
            dst->src[0] = &Q_rot_fused;
        }

#define TURBO_FUSED_DISPATCH(tK, tV) \
        if (K->type == tK && V->type == tV) { \
            if (Q->ne[0] == 128) \
                ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, tK, tV>(ctx, dst); \
            else \
                ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, tK, tV>(ctx, dst); \
        }
        TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO4_0)
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO3_0,   GGML_TYPE_TURBO3_0)
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO2_0,   GGML_TYPE_TURBO2_0)
#undef TURBO_FUSED_DISPATCH

        if (orig_q_fused) dst->src[0] = orig_q_fused;
        return;
    }

    const bool turbo_k_classic_v_prefill =
        ggml_cuda_fattn_prefill_mma_can_materialize_turbo_k_classic_v(K, V);
    const bool turbo_prefill_can_make_f16_K = ggml_cuda_turbo_prefill_mma_can_make_f16(K->type);
    const bool turbo_prefill_can_make_f16_V = ggml_cuda_turbo_prefill_mma_can_make_f16(V->type);
    const bool turbo_prefill_mma_safe =
        turbo_kv &&
        ((turbo_prefill_can_make_f16_K &&
          turbo_prefill_can_make_f16_V) ||
         turbo_k_classic_v_prefill);
    const bool turbo_prefill_turing_mma =
        turing_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);

    if (turbo_prefill_mma_safe && !turbo_prefill_vec && Q->ne[1] > 1 && Q->ne[0] <= 512 && turbo_prefill_turing_mma) {
        if (turbo_fa_debug) {
            fprintf(stderr,
                "GGML_TURBO_FA_DEBUG: path=prefill-dequant K=%s V=%s Q=[%lld,%lld,%lld,%lld] K=[%lld,%lld,%lld,%lld] V=[%lld,%lld,%lld,%lld]\n",
                ggml_type_name(K->type), ggml_type_name(V->type),
                (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
                (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3],
                (long long) V->ne[0], (long long) V->ne[1], (long long) V->ne[2], (long long) V->ne[3]);
        }
        // Prefill path: turbo4 K uses inverse FWHT dequant (original domain, no Q rotation),
        // turbo2/3 K uses simple dequant (rotated domain, Q pre-rotated). V un-rotation at graph level.
        ggml_cuda_turbo_prefill_attend(ctx, dst);
    } else {
        if (turbo_fa_debug && turbo_kv) {
            fprintf(stderr,
                "GGML_TURBO_FA_DEBUG: path=decode-dequant-or-vec K=%s V=%s Q=[%lld,%lld,%lld,%lld] K=[%lld,%lld,%lld,%lld] V=[%lld,%lld,%lld,%lld] prefill_safe=%d can_make_f16_K=%d can_make_f16_V=%d turbo_k_classic_v_prefill=%d vec_override=%d batch_ok=%d dim_ok=%d turing_mma=%d\n",
                ggml_type_name(K->type), ggml_type_name(V->type),
                (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
                (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3],
                (long long) V->ne[0], (long long) V->ne[1], (long long) V->ne[2], (long long) V->ne[3],
                (int) turbo_prefill_mma_safe,
                (int) turbo_prefill_can_make_f16_K,
                (int) turbo_prefill_can_make_f16_V,
                (int) turbo_k_classic_v_prefill,
                (int) turbo_prefill_vec,
                (int) (Q->ne[1] > 1),
                (int) (Q->ne[0] <= 512),
                (int) turbo_prefill_turing_mma);
        }
        load_tcq_decode_alpha(ctx.device);

        // Update VEC __constant__ alpha for context-adaptive mode
        if (d_tcq_decode_alpha_v_static == 0.0f &&
            (V->type == GGML_TYPE_TURBO4_TCQ || V->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO2_TCQ)) {
            float alpha = tcq_compute_alpha_v(V->type, V->ne[1]);
            cudaMemcpyToSymbol(d_tcq_decode_alpha_v_fattn, &alpha, sizeof(float));
        }

        // Load runtime codebooks for TCQ types (needed by both dequant and native VEC paths)
        if (K->type == GGML_TYPE_TURBO4_TCQ || V->type == GGML_TYPE_TURBO4_TCQ) {
            load_tcq4_codebook_fattn(ctx.device, "TCQ4 decode");
        }
        if (K->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO3_TCQ) {
            static bool tcq3_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
            if (!tcq3_cb_loaded[ctx.device]) {
                tcq3_cb_loaded[ctx.device] = true;
                const char *cb_path = getenv("TURBO_TCQ_CB");
                if (cb_path) {
                    float cb[512];
                    FILE *f = fopen(cb_path, "rb");
                    if (f && fread(cb, sizeof(float), 512, f) == 512) {
                        fclose(f);
                        cudaMemcpyToSymbol(d_turbo3_tcq_codebook_fattn, cb, 512*sizeof(float));
                        fprintf(stderr, "TCQ decode: loaded 3-bit codebook from %s (device %d)\n", cb_path, ctx.device);
                    } else {
                        if (f) fclose(f);
                        fprintf(stderr, "TCQ decode: FAILED to load 3-bit codebook from %s\n", cb_path);
                    }
                }
            }
        }
        if (K->type == GGML_TYPE_TURBO2_TCQ || V->type == GGML_TYPE_TURBO2_TCQ) {
            static bool tcq2_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
            if (!tcq2_cb_loaded[ctx.device]) {
                tcq2_cb_loaded[ctx.device] = true;
                const char *cb_path = getenv("TURBO_TCQ_CB2");
                if (cb_path) {
                    float cb[256];
                    FILE *f = fopen(cb_path, "rb");
                    if (f && fread(cb, sizeof(float), 256, f) == 256) {
                        fclose(f);
                        cudaMemcpyToSymbol(d_turbo2_tcq_codebook_fattn, cb, 256*sizeof(float));
                        fprintf(stderr, "TCQ decode: loaded 2-bit codebook from %s (device %d)\n", cb_path, ctx.device);
                    } else {
                        if (f) fclose(f);
                        fprintf(stderr, "TCQ decode: FAILED to load 2-bit codebook from %s\n", cb_path);
                    }
                }
            }
        }

        cudaStream_t stream = ctx.stream();

        // Use the unified route plan from raw K/V types. The same plan drives
        // allocation, support checks, and execution. Mixed classic K + Turbo V
        // uses decoded Turbo V plus MMA/tile fallback for D>=256 when the
        // effective classic_K/f16 vec route is unsafe. D=512 classic_non_q8 K
        // + Turbo V is also forced through this path to avoid the raw
        // classic_K/Turbo_V vec crash seen on Gemma 4.
        int device_dec;
        CUDA_CHECK(cudaGetDevice(&device_dec));

        if (ggml_cuda_fattn_route_debug_enabled()) {
            fprintf(stderr,
                "CUDA_FA_ROUTE_EXEC_BEGIN "
                "Q=[%lld,%lld,%lld,%lld] "
                "Kraw=%s Vraw=%s "
                "decode=%d dk=%d dv=%d "
                "allow_vec=%d kernel=%s "
                "need_f16_K=%d need_f16_V=%d "
                "Kdata=%p Vdata=%p dst=%p\n",
                (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
                ggml_type_name(K->type), ggml_type_name(V->type),
                (int) plan.decode_dequant,
                (int) plan.decode_dequant_K,
                (int) plan.decode_dequant_V,
                (int) plan.allow_vec,
                ggml_cuda_fattn_kernel_name(plan.kernel),
                (int) plan.need_generic_f16_K,
                (int) plan.need_generic_f16_V,
                K->data,
                V->data,
                dst->data);
            fflush(stderr);
        }

        const bool do_decode_dequant = plan.decode_dequant;
        const bool k_needs_dequant    = plan.decode_dequant_K;
        const bool v_needs_dequant    = plan.decode_dequant_V;

        half * k_fp16_dec = nullptr;
        half * v_fp16_dec = nullptr;
        ggml_tensor K_f16_dec, V_f16_dec;
        ggml_tensor * orig_k_decode = nullptr;
        ggml_tensor * orig_v_decode = nullptr;

        if (do_decode_dequant) {
            if (k_needs_dequant) {
                // Size the dequant buffer for the FULL cache (kv_size from the underlying root
                // tensor), not just the current n_kv. This prevents per-token reallocations as
                // the cache fills, which would invalidate any in-flight CUDA graph capture
                // pointing at the old device pointer (defensive against PR #21635-class bugs).
                // The first call sizes the buffer for the worst-case cache; subsequent calls
                // (including layers with smaller caches) reuse the same allocation.
                const ggml_tensor * k_root = K;
                while (k_root->view_src) k_root = k_root->view_src;
                const size_t k_max_bytes = (size_t)k_root->ne[0] * k_root->ne[1] * k_root->ne[2] * sizeof(half);
                if (k_max_bytes > kv_dequant_k_buf_size[device_dec]) {
                    if (kv_dequant_k_buf[device_dec]) CUDA_CHECK(cudaFree(kv_dequant_k_buf[device_dec]));
                    CUDA_CHECK(cudaMalloc(&kv_dequant_k_buf[device_dec], k_max_bytes));
                    kv_dequant_k_buf_size[device_dec] = k_max_bytes;
                }
                k_fp16_dec = kv_dequant_k_buf[device_dec];
                // K dequant to fp16 in ORIGINAL (unrotated) domain via inverse FWHT.
                // All turbo K types use inv-FWHT kernels so K matches native f16/q8_0 layout
                // and Q stays unrotated. This mirrors the prefill path's encode→decode chain
                // and is the only path that works on Gemma 4 ISWA + K=V global layers.
                //
                // Bug #31 exception: K=turbo2 inv-FWHT decode produces correct values for V in
                // {turbo2, *_tcq} but a (still-undiagnosed) divergence with V in {turbo3, turbo4,
                // q8_0, f16} on Gemma 4 26B-A4B (degenerate single-token output, attention scores
                // collapse). The PREFILL path uses the rotated-domain kernel + Q rotation for
                // turbo2 K and works for every V type. Mirror that here for the failing V types.
                const bool k_t2_use_rotated = (K->type == GGML_TYPE_TURBO2_0) &&
                    (V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0 ||
                     V->type == GGML_TYPE_Q8_0    || V->type == GGML_TYPE_F16);
                const bool k_t3_use_rotated = (K->type == GGML_TYPE_TURBO3_0) &&
                    (V->type == GGML_TYPE_TURBO2_0);
                dim3 grid_k(K->ne[1], K->ne[2], K->ne[3]);
                if (K->type == GGML_TYPE_TURBO2_0 && k_t2_use_rotated) {
                    // Rotated-domain dequant: K stays in WHT-rotated space; Q is pre-rotated below.
                    k_turbo2_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO2_0) {
                    k_turbo2_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO3_0 && k_t3_use_rotated) {
                    // Rotated-domain dequant for K=t3 + V=t2 (same Bug #31 pattern, V side).
                    k_turbo3_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO3_0) {
                    k_turbo3_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO4_0) {
                    k_turbo4_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO4_TCQ) {
                    k_turbo4_tcq_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
                } else if (K->type == GGML_TYPE_TURBO3_TCQ) {
                    k_turbo3_tcq_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
                } else if (K->type == GGML_TYPE_TURBO2_TCQ) {
                    k_turbo2_tcq_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
                } else if (K->type == GGML_TYPE_Q8_0) {
                    // Q8_0 K dequant: only fires at D=512 when V is turbo (no F16/Q8_0 D=512
                    // template, and (Q8_0, TURBO4_0) D=512 has buggy SASS on sm_120 PTX-JIT).
                    // Output goes into TKHE layout matching V_f16_dec → dispatches as F16/F16 D=512.
                    k_q8_0_dequant_f16_tkhe<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                }
                K_f16_dec = *K;
                K_f16_dec.type = GGML_TYPE_F16;
                K_f16_dec.data = k_fp16_dec;
                K_f16_dec.nb[0] = sizeof(half);
                K_f16_dec.nb[1] = K->ne[0] * K->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
                K_f16_dec.nb[2] = K->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
                K_f16_dec.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
                orig_k_decode = dst->src[1];
                dst->src[1] = &K_f16_dec;
            }
            if (v_needs_dequant) {
                // Same kv_size-based sizing as K above — see comment there.
                const ggml_tensor * v_root = V;
                while (v_root->view_src) v_root = v_root->view_src;
                const size_t v_max_bytes = (size_t)v_root->ne[0] * v_root->ne[1] * v_root->ne[2] * sizeof(half);
                if (v_max_bytes > kv_dequant_v_buf_size[device_dec]) {
                    if (kv_dequant_v_buf[device_dec]) CUDA_CHECK(cudaFree(kv_dequant_v_buf[device_dec]));
                    CUDA_CHECK(cudaMalloc(&kv_dequant_v_buf[device_dec], v_max_bytes));
                    kv_dequant_v_buf_size[device_dec] = v_max_bytes;
                }
                v_fp16_dec = kv_dequant_v_buf[device_dec];
                // V dequant to fp16. All turbo V stays in rotated domain — the graph-level
                // ggml_turbo_wht inverse op (added in build_attn) un-rotates the attention output.
                dim3 grid_v(V->ne[1], V->ne[2], V->ne[3]);
                if (V->type == GGML_TYPE_TURBO2_0) {
                    k_turbo2_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_TURBO4_TCQ) {
                    k_turbo4_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]));
                } else if (V->type == GGML_TYPE_TURBO3_TCQ) {
                    k_turbo3_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]));
                } else if (V->type == GGML_TYPE_TURBO2_TCQ) {
                    k_turbo2_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]));
                } else if (V->type == GGML_TYPE_TURBO4_0) {
                    k_turbo4_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_TURBO3_0) {
                    k_turbo3_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_Q8_0) {
                    // Q8_0 V dequant: only fires at D=512 when K is turbo (mirror of K=Q8_0 path).
                    k_q8_0_dequant_f16_tkhe<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                }
                V_f16_dec = *V;
                V_f16_dec.type = GGML_TYPE_F16;
                V_f16_dec.data = v_fp16_dec;
                V_f16_dec.nb[0] = sizeof(half);
                V_f16_dec.nb[1] = V->ne[0] * V->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
                V_f16_dec.nb[2] = V->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
                V_f16_dec.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
                orig_v_decode = dst->src[2];
                dst->src[2] = &V_f16_dec;
                // Bug A1: nvcc 13 on sm_120a reorders these V_f16_dec.nb[*] stores past the FA
                // dispatcher → stale strides → <unused49> garbage. signal_fence is a pure
                // host-compiler barrier (zero machine instructions).
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }
        }

        // Pre-rotate Q for turbo K stored in rotated domain.
        // When do_decode_dequant fires, all turbo K types are dequanted via inv-FWHT into
        // ORIGINAL domain → Q stays unrotated. When decode dequant is skipped (D>256 or
        // GGML_TURBO_DECODE_NATIVE), turbo K is consumed by the native vec turbo dot product,
        // which expects a pre-rotated Q — so rotate Q in that case.
        ggml_tensor Q_rot_decode;
        ggml_tensor * orig_q_decode = nullptr;
        const bool turbo_k_any = ggml_cuda_fattn_is_turbo_kv_type(K->type);
        // Bug #31 exception: when K=turbo2/turbo3 dequant fell back to the rotated kernel (see K
        // dispatch above), K is in WHT-rotated space, not original space, so Q must be pre-rotated.
        const bool k_uses_rotated_path = do_decode_dequant && (
            ((K->type == GGML_TYPE_TURBO2_0) &&
             (V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0 ||
              V->type == GGML_TYPE_Q8_0    || V->type == GGML_TYPE_F16)) ||
            ((K->type == GGML_TYPE_TURBO3_0) && (V->type == GGML_TYPE_TURBO2_0)));
        const bool turbo_k_in_orig_domain = do_decode_dequant && turbo_k_any && !k_uses_rotated_path;
        if (turbo_k_any && !turbo_k_in_orig_domain && Q->ne[0] % 128 == 0) {
            const size_t q_size = ggml_nelements(Q) * sizeof(float);
            if (q_size > q_rot_buf_size[device_dec]) {
                if (q_rot_buf[device_dec]) CUDA_CHECK(cudaFree(q_rot_buf[device_dec]));
                CUDA_CHECK(cudaMalloc(&q_rot_buf[device_dec], q_size));
                q_rot_buf_size[device_dec] = q_size;
            }
            const int64_t n_q_groups = ggml_nelements(Q) / 128;
            k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
                (const float *)Q->data, q_rot_buf[device_dec], ggml_nelements(Q));
            Q_rot_decode = *Q;
            Q_rot_decode.data = q_rot_buf[device_dec];
            orig_q_decode = dst->src[0];
            dst->src[0] = &Q_rot_decode;
        }

        if (ggml_cuda_fattn_route_debug_enabled()) {
            const ggml_tensor * K_run = dst->src[1];
            const ggml_tensor * V_run = dst->src[2];
            const ggml_tensor * Q_run = dst->src[0];

            fprintf(stderr,
                "CUDA_FA_ROUTE_EXEC_DISPATCH "
                "Qrun=%s Q=[%lld,%lld,%lld,%lld] Qdata=%p "
                "Krun=%s K=[%lld,%lld,%lld,%lld] Kdata=%p Knb=[%lld,%lld,%lld,%lld] "
                "Vrun=%s V=[%lld,%lld,%lld,%lld] Vdata=%p Vnb=[%lld,%lld,%lld,%lld] "
                "kernel=%s compiled_pair=%d\n",
                ggml_type_name(Q_run->type),
                (long long) Q_run->ne[0], (long long) Q_run->ne[1], (long long) Q_run->ne[2], (long long) Q_run->ne[3],
                Q_run->data,
                ggml_type_name(K_run->type),
                (long long) K_run->ne[0], (long long) K_run->ne[1], (long long) K_run->ne[2], (long long) K_run->ne[3],
                K_run->data,
                (long long) K_run->nb[0], (long long) K_run->nb[1], (long long) K_run->nb[2], (long long) K_run->nb[3],
                ggml_type_name(V_run->type),
                (long long) V_run->ne[0], (long long) V_run->ne[1], (long long) V_run->ne[2], (long long) V_run->ne[3],
                V_run->data,
                (long long) V_run->nb[0], (long long) V_run->nb[1], (long long) V_run->nb[2], (long long) V_run->nb[3],
                ggml_cuda_fattn_kernel_name(plan.kernel),
                (int) ggml_cuda_fattn_pair_compiled(K_run->type, V_run->type));
            fflush(stderr);
        }

        const best_fattn_kernel selected_kernel = plan.kernel;

        switch (selected_kernel) {
            case BEST_FATTN_KERNEL_NONE:
                fprintf(stderr, "No CUDA FA kernel selected: K=%s V=%s D=%lld\n",
                    ggml_type_name(K->type), ggml_type_name(V->type), (long long) Q->ne[0]);
                GGML_ABORT("no CUDA FA kernel selected");
            case BEST_FATTN_KERNEL_TILE:
                ggml_cuda_flash_attn_ext_tile(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_VEC:
                if (!ggml_cuda_fattn_pair_compiled(dst->src[1]->type, dst->src[2]->type)) {
                    if (!ggml_cuda_fattn_ignore_uncompiled_pairs()) {
                        fprintf(stderr,
                            "CUDA FA effective K/V pair was not compiled: K=%s V=%s. "
                            "Use standard q/KVarN fallback cache types in the default build, or rebuild with "
                            "GGML_CUDA_FA_HALF_QUANTS or GGML_CUDA_FA_ALL_QUANTS for Turbo/TCQ or arbitrary pairs.\n",
                            ggml_type_name(dst->src[1]->type), ggml_type_name(dst->src[2]->type));
                        GGML_ABORT("CUDA FA effective K/V pair not compiled");
                    }
                    fprintf(stderr,
                        "WARNING: GGML_CUDA_FA_IGNORE_UNCOMPILED_PAIRS=1: CUDA FA effective K/V pair was not compiled: "
                        "K=%s V=%s. Continuing with vec dispatch.\n",
                        ggml_type_name(dst->src[1]->type), ggml_type_name(dst->src[2]->type));
                }
                ggml_cuda_flash_attn_ext_vec(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_WMMA_F16:
                ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_MMA_F16:
                ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
                break;
        }

        if (orig_q_decode) dst->src[0] = orig_q_decode;
        if (orig_k_decode) dst->src[1] = orig_k_decode;
        if (orig_v_decode) dst->src[2] = orig_v_decode;
        // K/V fp16 buffers are persistent (grow-only), no free needed
    }

    // Output inverse rotation for turbo V types is handled at graph level
    // (ggml_turbo_wht op in llama-graph.cpp) to maintain CUDA graph compatibility.
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    const ggml_cuda_fattn_route_plan plan = ggml_cuda_fattn_make_route_plan(device, dst);

    if (plan.kernel == BEST_FATTN_KERNEL_NONE) {
        return false;
    }

    if (plan.kernel == BEST_FATTN_KERNEL_VEC) {
        return ggml_cuda_fattn_pair_compiled(plan.effective_type_K, plan.effective_type_V);
    }

    if ((ggml_cuda_fattn_is_ranked_kv_type(plan.effective_type_K) || ggml_cuda_fattn_is_ranked_kv_type(plan.effective_type_V)) &&
        !ggml_cuda_fattn_pair_compiled(plan.effective_type_K, plan.effective_type_V)) {
        return false;
    }

    if (ggml_cuda_fattn_route_debug_enabled()) {
        fprintf(stderr,
            "CUDA_FA_ROUTE_SUPPORTED "
            "kernel=%s effK=%s effV=%s supported=%d\n",
            ggml_cuda_fattn_kernel_name(plan.kernel),
            ggml_type_name(plan.effective_type_K),
            ggml_type_name(plan.effective_type_V),
            plan.kernel != BEST_FATTN_KERNEL_NONE);
        fflush(stderr);
    }

    return true;
}
