#pragma once

#include <vector>

static constexpr int GGML_CUDA_FATTN_KVARN_DIM = 128;
static constexpr int GGML_CUDA_FATTN_KVARN_NATIVE_MAX_Q = 8;
static constexpr ggml_type GGML_CUDA_FATTN_KVARN_TYPE = GGML_TYPE_COUNT;

enum {
    GGML_CUDA_FATTN_KVARN_OP_PARAM_BITS              = 0,
    GGML_CUDA_FATTN_KVARN_OP_PARAM_VIEW_VALUE        = 1,
    GGML_CUDA_FATTN_KVARN_OP_PARAM_VIEW_STREAM_START = 2,
    GGML_CUDA_FATTN_KVARN_OP_PARAM_VIEW_N_STREAM     = 3,
    GGML_CUDA_FATTN_KVARN_OP_PARAM_VIEW_SWA          = 6,
    GGML_CUDA_FATTN_KVARN_OP_PARAM_STAGE_GROUPS      = 7,
};

struct ggml_cuda_fattn_kvarn_desc {
    const uint8_t * records;
    const half    * stage;
    const int64_t * indices;
    const int     * live_groups;
    int n_record_heads;
    int out_stream;
    int stream;
    int head_base;
    int groups_per_stream;
    int record_bytes;
    int stage_groups;
    int tail_groups;
    int bits;
    int value;
    int swa;
};

struct ggml_cuda_fattn_kvarn_plan_side {
    const ggml_tensor * view        = nullptr;
    const ggml_tensor * records     = nullptr;
    const ggml_tensor * stage       = nullptr;
    const ggml_tensor * indices     = nullptr;
    int bits          = 0;
    int stream_start  = 0;
    int n_stream      = 0;
    int stage_groups  = 0;
    int groups_per_stream = 0;
    bool value        = false;
    bool swa          = false;
};

struct ggml_cuda_fattn_kvarn_plan {
    ggml_cuda_fattn_kvarn_plan_side k;
    ggml_cuda_fattn_kvarn_plan_side v;
    int head_dim   = 0;
    int n_kv       = 0;
    int n_kv_heads = 0;
    int n_stream   = 0;
    int slices     = 0;
};

static __device__ __forceinline__ uint8_t ggml_cuda_fattn_kvarn_unpack_record(
        const uint8_t * record, const int index, const int bits) {
    if (bits == 8) {
        return record[index];
    }
    if (bits == 4) {
        const uint8_t packed = record[index >> 1];
        return (packed >> ((index & 1) << 2)) & 0x0fu;
    }
    if (bits == 2) {
        const uint8_t packed = record[index >> 2];
        return (packed >> ((index & 3) << 1)) & 0x03u;
    }
    const int bit_offset = index * bits;
    const int byte_offset = bit_offset >> 3;
    const int bit_in_byte = bit_offset & 7;
    const uint16_t packed = (uint16_t) record[byte_offset] | ((uint16_t) record[byte_offset + 1] << 8);
    return (packed >> bit_in_byte) & ((1u << bits) - 1u);
}

static __device__ __forceinline__ float ggml_cuda_fattn_kvarn_load_rotated(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int token,
        const int slice,
        const int dim) {
    const int record_head = desc.head_base + slice;

    int group;
    int pos;
    bool from_stage;
    bool from_record;
    int stage_pos;
    int record_group;

    if (desc.swa) {
        const int64_t abs_pos = desc.indices[token];
        if (abs_pos < 0) {
            return 0.0f;
        }
        const int live_group = desc.live_groups[0];
        group = (int) (abs_pos / GGML_CUDA_FATTN_KVARN_DIM);
        pos   = (int) (abs_pos - (int64_t) group * GGML_CUDA_FATTN_KVARN_DIM);
        const int stage_begin = live_group >= (desc.tail_groups - 1) ? live_group - (desc.tail_groups - 1) : 0;
        from_stage  = group >= stage_begin && group <= live_group;
        from_record = !from_stage && group >= 0 && group < stage_begin &&
            (live_group - group) < desc.groups_per_stream;
        stage_pos = (group % desc.stage_groups) * GGML_CUDA_FATTN_KVARN_DIM + pos;
        record_group = group % desc.groups_per_stream;
    } else {
        const int live_group = desc.live_groups[desc.out_stream];
        group = token / GGML_CUDA_FATTN_KVARN_DIM;
        pos   = token - group * GGML_CUDA_FATTN_KVARN_DIM;
        from_stage = group == 0 ||
            (group > 0 && group <= live_group && group + (desc.tail_groups - 1) >= live_group);
        from_record = !from_stage && group < live_group && group < desc.groups_per_stream;
        const int stage_base = desc.stream * GGML_CUDA_FATTN_KVARN_DIM * desc.stage_groups;
        stage_pos = stage_base + (group == 0 ? pos :
            GGML_CUDA_FATTN_KVARN_DIM + ((group - 1) % desc.tail_groups) * GGML_CUDA_FATTN_KVARN_DIM + pos);
        record_group = desc.stream * desc.groups_per_stream + group;
    }

    if (from_stage) {
        return __half2float(desc.stage[((int64_t) stage_pos * desc.n_record_heads + record_head) * GGML_CUDA_FATTN_KVARN_DIM + dim]);
    }

    if (!from_record) {
        return 0.0f;
    }

    const uint8_t * record = desc.records + ((int64_t) record_group * desc.n_record_heads + record_head) * desc.record_bytes;
    const int payload_bytes = GGML_CUDA_FATTN_KVARN_DIM * GGML_CUDA_FATTN_KVARN_DIM * desc.bits / 8;
    const half * scale_axis = (const half *) (record + payload_bytes);
    const half * zp_axis    = scale_axis + GGML_CUDA_FATTN_KVARN_DIM;
    const half * other_axis = zp_axis + GGML_CUDA_FATTN_KVARN_DIM;
    const int row = desc.value ? pos : dim;
    const int col = desc.value ? dim : pos;
    const uint8_t q = ggml_cuda_fattn_kvarn_unpack_record(record, row * GGML_CUDA_FATTN_KVARN_DIM + col, desc.bits);
    return (float(q) * __half2float(scale_axis[row]) + __half2float(zp_axis[row])) * __half2float(other_axis[col]);
}

template<int D, int stride_tile, int nbatch_fa, int nthreads, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_kvarn_load_tile(
        const char * __restrict__ desc_raw,
        half2      * __restrict__ tile_KV,
        const int k_start,
        const int i_sup,
        const int dim2_start,
        const int dim2_count) {
    const ggml_cuda_fattn_kvarn_desc & desc = *(const ggml_cuda_fattn_kvarn_desc *) desc_raw;
    constexpr int slices = D / GGML_CUDA_FATTN_KVARN_DIM;
    static_assert(D % GGML_CUDA_FATTN_KVARN_DIM == 0 && D <= 512, "KVarN native MMA supports 128-wide slices through D=512");
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int tid = threadIdx.y * warp_size + threadIdx.x;
    const int dim2_end = dim2_start + dim2_count;
    __shared__ float sh[GGML_CUDA_FATTN_KVARN_DIM];

    for (int row = 0; row < nbatch_fa; ++row) {
        const bool valid_row = !oob_check || row < i_sup;
        if (!valid_row) {
            for (int b = tid; b < dim2_count; b += nthreads) {
                tile_KV[row * stride_tile + b] = make_half2(0.0f, 0.0f);
            }
            __syncthreads();
            continue;
        }

        const int token = k_start + row;
        for (int slice = 0; slice < slices; ++slice) {
            const int slice_dim2_start = slice * (GGML_CUDA_FATTN_KVARN_DIM / 2);
            const int slice_dim2_end   = slice_dim2_start + GGML_CUDA_FATTN_KVARN_DIM / 2;
            const int out_dim2_start   = max(dim2_start, slice_dim2_start);
            const int out_dim2_end     = min(dim2_end, slice_dim2_end);
            if (out_dim2_start >= out_dim2_end) {
                continue;
            }

            for (int dim = tid; dim < GGML_CUDA_FATTN_KVARN_DIM; dim += nthreads) {
                sh[dim] = ggml_cuda_fattn_kvarn_load_rotated(desc, token, slice, dim);
            }
            __syncthreads();

            for (int global_b = out_dim2_start + tid; global_b < out_dim2_end; global_b += nthreads) {
                const int local_b = global_b - slice_dim2_start;
                tile_KV[row * stride_tile + global_b - dim2_start] =
                    make_half2(sh[2 * local_b], sh[2 * local_b + 1]);
            }
            __syncthreads();
        }
    }
}

static __global__ void ggml_cuda_fattn_kvarn_live_groups_kernel(
        const int64_t * indices,
        int n_indices,
        int stream_start,
        int n_stream,
        int groups_per_stream,
        bool swa,
        int * live_groups) {
    const int out_stream = blockIdx.x;
    if (out_stream >= n_stream) {
        return;
    }

    const int stream = stream_start + out_stream;
    int live_group = 0;
    for (int i = threadIdx.x; i < n_indices; i += blockDim.x) {
        const int64_t idx = indices[i];
        if (swa) {
            if (idx >= 0) {
                live_group = max(live_group, (int) (idx / GGML_CUDA_FATTN_KVARN_DIM));
            }
        } else {
            const int group_global = (int) (idx / GGML_CUDA_FATTN_KVARN_DIM);
            const int idx_stream = group_global / groups_per_stream;
            if (idx_stream == stream) {
                live_group = max(live_group, group_global - stream * groups_per_stream);
            }
        }
    }

    __shared__ int partial[GGML_CUDA_FATTN_KVARN_DIM];
    partial[threadIdx.x] = live_group;
    __syncthreads();
    for (int stride = GGML_CUDA_FATTN_KVARN_DIM / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = max(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        live_groups[out_stream] = partial[0];
    }
}

static inline bool ggml_cuda_fattn_kvarn_valid_bits(const int bits) {
    return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8;
}

static inline const ggml_tensor * ggml_cuda_fattn_kvarn_view_base(const ggml_tensor * t) {
    while (t != nullptr && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE)) {
        t = t->src[0];
    }
    return t != nullptr && t->op == GGML_OP_KVARN_VIEW ? t : nullptr;
}

static inline bool ggml_cuda_fattn_kvarn_uses_views(const ggml_tensor * dst) {
    return dst != nullptr &&
        (ggml_cuda_fattn_kvarn_view_base(dst->src[1]) != nullptr ||
         ggml_cuda_fattn_kvarn_view_base(dst->src[2]) != nullptr);
}

static inline bool ggml_cuda_fattn_kvarn_unwrap_view(
        const ggml_tensor * t,
        ggml_cuda_fattn_kvarn_plan_side & side) {
    if (t == nullptr || t->type != GGML_TYPE_F16) {
        return false;
    }

    const ggml_tensor * cur = t;
    if (cur->op == GGML_OP_PERMUTE) {
        if (ggml_get_op_params_i32(cur, 0) != 0 ||
            ggml_get_op_params_i32(cur, 1) != 2 ||
            ggml_get_op_params_i32(cur, 2) != 1 ||
            ggml_get_op_params_i32(cur, 3) != 3) {
            return false;
        }
        cur = cur->src[0];
    } else {
        return false;
    }

    if (cur != nullptr && cur->op == GGML_OP_RESHAPE) {
        cur = cur->src[0];
    }

    if (cur == nullptr || cur->op != GGML_OP_KVARN_VIEW) {
        return false;
    }

    side.view = cur;
    side.records = cur->src[0];
    side.stage   = cur->src[1];
    side.indices = cur->src[2];
    if (side.records == nullptr || side.stage == nullptr || side.indices == nullptr) {
        return false;
    }

    side.bits = ggml_get_op_params_i32(cur, GGML_CUDA_FATTN_KVARN_OP_PARAM_BITS);
    side.value = ggml_get_op_params_i32(cur, GGML_CUDA_FATTN_KVARN_OP_PARAM_VIEW_VALUE) != 0;
    side.stream_start = ggml_get_op_params_i32(cur, GGML_CUDA_FATTN_KVARN_OP_PARAM_VIEW_STREAM_START);
    side.n_stream = ggml_get_op_params_i32(cur, GGML_CUDA_FATTN_KVARN_OP_PARAM_VIEW_N_STREAM);
    side.swa = ggml_get_op_params_i32(cur, GGML_CUDA_FATTN_KVARN_OP_PARAM_VIEW_SWA) != 0;
    side.stage_groups = ggml_get_op_params_i32(cur, GGML_CUDA_FATTN_KVARN_OP_PARAM_STAGE_GROUPS);

    if (!ggml_cuda_fattn_kvarn_valid_bits(side.bits) || side.n_stream <= 0 || side.stage_groups < 2) {
        return false;
    }
    if (side.records->type != GGML_TYPE_I8 || side.stage->type != GGML_TYPE_F16 || side.indices->type != GGML_TYPE_I64) {
        return false;
    }
    if (side.stage->ne[2] % (GGML_CUDA_FATTN_KVARN_DIM * side.stage_groups) != 0) {
        return false;
    }
    const int total_streams = (int) (side.stage->ne[2] / (GGML_CUDA_FATTN_KVARN_DIM * side.stage_groups));
    if (total_streams <= 0 || side.records->ne[2] % total_streams != 0) {
        return false;
    }
    side.groups_per_stream = (int) (side.records->ne[2] / total_streams);
    return side.stream_start >= 0 && side.stream_start + side.n_stream <= total_streams;
}

static inline bool ggml_cuda_fattn_kvarn_view_supported(
        const int device,
        const ggml_tensor * dst,
        ggml_cuda_fattn_kvarn_plan * out = nullptr) {
    if (dst == nullptr || dst->op != GGML_OP_FLASH_ATTN_EXT || dst->src[0] == nullptr ||
            dst->src[1] == nullptr || dst->src[2] == nullptr) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return false;
    }
    if (!turing_mma_available(ggml_cuda_info().devices[device].cc)) {
        return false;
    }
    if (!((Q->ne[0] == 128 && V->ne[0] == 128) ||
          (Q->ne[0] == 256 && V->ne[0] == 256) ||
          (Q->ne[0] == 512 && V->ne[0] == 512))) {
        return false;
    }
    if (K->ne[0] != Q->ne[0] || V->ne[0] != Q->ne[0]) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0 || V->ne[2] != K->ne[2] || K->ne[1] != V->ne[1] || K->ne[3] != V->ne[3]) {
        return false;
    }

    ggml_cuda_fattn_kvarn_plan plan;
    if (!ggml_cuda_fattn_kvarn_unwrap_view(K, plan.k) ||
        !ggml_cuda_fattn_kvarn_unwrap_view(V, plan.v)) {
        return false;
    }
    if (plan.k.value || !plan.v.value) {
        return false;
    }
    if (plan.k.indices->ne[0] != plan.v.indices->ne[0] ||
        plan.k.stream_start != plan.v.stream_start ||
        plan.k.n_stream != plan.v.n_stream ||
        plan.k.swa != plan.v.swa) {
        return false;
    }

    plan.head_dim = (int) Q->ne[0];
    plan.n_kv = (int) K->ne[1];
    plan.n_kv_heads = (int) K->ne[2];
    plan.n_stream = (int) K->ne[3];
    plan.slices = plan.head_dim / GGML_CUDA_FATTN_KVARN_DIM;
    if (plan.n_stream != plan.k.n_stream || plan.n_stream != plan.v.n_stream) {
        return false;
    }
    if (plan.k.view->ne[1] != (int64_t) plan.n_kv_heads * plan.slices ||
        plan.v.view->ne[1] != (int64_t) plan.n_kv_heads * plan.slices ||
        plan.k.view->ne[2] != plan.n_kv || plan.v.view->ne[2] != plan.n_kv ||
        plan.k.view->ne[3] != plan.n_stream || plan.v.view->ne[3] != plan.n_stream) {
        return false;
    }

    if (out != nullptr) {
        *out = plan;
    }
    return true;
}

static inline bool ggml_cuda_fattn_kvarn_supported(
        const int device,
        const ggml_tensor * dst,
        ggml_cuda_fattn_kvarn_plan * out = nullptr) {
    const ggml_tensor * Q = dst != nullptr ? dst->src[0] : nullptr;
    if (Q == nullptr || !(Q->ne[1] <= GGML_CUDA_FATTN_KVARN_NATIVE_MAX_Q)) {
        return false;
    }

    return ggml_cuda_fattn_kvarn_view_supported(device, dst, out);
}

#ifdef GGML_CUDA_FATTN_MMA_KVARN_DEFINE_CASE
#ifndef GGML_CUDA_FATTN_MMA_KVARN_CASE_DEFINED
#define GGML_CUDA_FATTN_MMA_KVARN_CASE_DEFINED

static inline void ggml_cuda_fattn_kvarn_fill_descs(
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
    constexpr int nstages = 0;

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps = nthreads / warp_size_host;
    constexpr bool V_is_K_view = false;

    const size_t nbytes_shared_KV = nbatch_fa * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q = ncols * (DKQ / 2 + 4) * sizeof(half2);
    const size_t nbytes_shared_mask = ncols1 * (nbatch_fa / 2 + 4) * sizeof(half2);
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
    ggml_cuda_fattn_kvarn_fill_descs(plan.k, plan, live_groups_k.get(), k_desc_host);
    ggml_cuda_fattn_kvarn_fill_descs(plan.v, plan, live_groups_v.get(), v_desc_host);

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

    GGML_UNUSED(nstages);
}

#endif
#endif
