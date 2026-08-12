#pragma once

static constexpr int GGML_CUDA_FATTN_KVARN_DIM = 128;
static constexpr ggml_type GGML_CUDA_FATTN_KVARN_TYPE = GGML_TYPE_COUNT;
static constexpr ggml_type GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE = (ggml_type) (GGML_TYPE_COUNT + 1);

static constexpr __host__ __device__ bool ggml_cuda_fattn_kvarn_template_type(ggml_type type) {
    return type == GGML_CUDA_FATTN_KVARN_TYPE || type == GGML_CUDA_FATTN_KVARN_ORIGINAL_TYPE;
}

enum {
    GGML_CUDA_FATTN_KVARN_OP_PARAM_BITS              = 0,
    GGML_CUDA_FATTN_KVARN_OP_PARAM_VIEW_VALUE        = 1,
    GGML_CUDA_FATTN_KVARN_OP_PARAM_VIEW_STREAM_START = 2,
    GGML_CUDA_FATTN_KVARN_OP_PARAM_VIEW_N_STREAM     = 3,
    GGML_CUDA_FATTN_KVARN_OP_PARAM_HEAD_SLICES       = 5,
    GGML_CUDA_FATTN_KVARN_OP_PARAM_VIEW_SWA          = 6,
    GGML_CUDA_FATTN_KVARN_OP_PARAM_STAGE_GROUPS      = 7,
    GGML_CUDA_FATTN_KVARN_OP_PARAM_TAIL_GROUPS       = 8,
    GGML_CUDA_FATTN_KVARN_OP_PARAM_EAGER_RECORDS     = 9,
};

struct ggml_cuda_fattn_kvarn_desc {
    const uint8_t * records;
    const half    * stage;
    const int64_t * indices;
    int n_record_heads;
    int live_group;
    int live_pos;
    int stream;
    int head_base;
    int groups_per_stream;
    int record_bytes;
    int stage_groups;
    int tail_groups;
    int bits;
    int value;
    int swa;
    int head_slices;
    int eager_records;
    int read_indirect;
    // Large prefill can reconstruct one side in the original domain in the MMA tile loader.
    // Decode-width paths keep rotated-domain K/V and rotate Q/output in the graph.
    int original_domain;
};

static __device__ __forceinline__ int64_t ggml_cuda_fattn_kvarn_read_cell(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int64_t encoded,
        bool & explicitly_staged) {
    explicitly_staged = false;
    if (encoded >= -1) {
        return encoded;
    }
    explicitly_staged = true;
    return -encoded - 2;
}

static __device__ __forceinline__ bool ggml_cuda_fattn_kvarn_group_from_stage(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int group) {
    if (desc.eager_records) {
        if (!desc.swa && group == 0) {
            return true;
        }
        return group == desc.live_group && desc.live_pos < GGML_CUDA_FATTN_KVARN_DIM - 1;
    }
    if (desc.swa) {
        const int stage_begin = desc.live_group >= (desc.tail_groups - 1) ?
            desc.live_group - (desc.tail_groups - 1) : 0;
        return group >= stage_begin && group <= desc.live_group;
    }
    return group == 0 ||
        (group > 0 && group <= desc.live_group && group + (desc.tail_groups - 1) >= desc.live_group);
}

static __device__ __forceinline__ bool ggml_cuda_fattn_kvarn_group_from_record(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int group) {
    if (group < 0 || ggml_cuda_fattn_kvarn_group_from_stage(desc, group)) {
        return false;
    }
    if (desc.eager_records) {
        const bool completed = group < desc.live_group ||
            (group == desc.live_group && desc.live_pos == GGML_CUDA_FATTN_KVARN_DIM - 1);
        if (!completed) {
            return false;
        }
        return desc.swa ? desc.live_group - group < desc.groups_per_stream :
            group > 0 && group < desc.groups_per_stream;
    }
    if (desc.swa) {
        const int distance = desc.live_group - group;
        return distance >= desc.tail_groups && distance < desc.groups_per_stream + desc.tail_groups;
    }
    return group < desc.live_group && group < desc.groups_per_stream;
}

static __device__ __forceinline__ bool ggml_cuda_fattn_kvarn_swa_group_from_record(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int group,
        const int stage_begin) {
    GGML_UNUSED(stage_begin);
    return desc.swa && ggml_cuda_fattn_kvarn_group_from_record(desc, group);
}

struct ggml_cuda_fattn_kvarn_plan_side {
    const ggml_tensor * view        = nullptr;
    const ggml_tensor * records     = nullptr;
    const ggml_tensor * stage       = nullptr;
    const ggml_tensor * indices     = nullptr;
    int bits          = 0;
    int stream_start  = 0;
    int n_stream      = 0;
    int stage_groups  = 0;
    int tail_groups   = 0;
    int groups_per_stream = 0;
    int head_slices   = 1;
    bool value        = false;
    bool swa          = false;
    bool eager_records = false;
    bool read_indirect = false;
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

// Definition lives in fattn-mma-kvarn-impl.cuh. fattn-mma-f16.cuh needs only this declaration:
// its call sites are in if-constexpr branches discarded for non-KVarN types, while fattn.cu
// includes the implementation for the generic native-MMA fallback.
template<int D, int stride_tile, int nbatch_fa, int nthreads, bool oob_check,
    bool original_domain = false, bool dim_major_K = false, bool cache_record_axes = false>
static __device__ __forceinline__ void flash_attn_ext_kvarn_load_tile(
        const char * __restrict__ desc_raw,
        half2      * __restrict__ tile_KV,
        const int k_start,
        const int i_sup,
        const int dim2_start,
        const int dim2_count,
        half      * __restrict__ scale_smem);

static inline bool ggml_cuda_fattn_kvarn_valid_bits(const int bits) {
    return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8;
}

static __device__ __forceinline__ float ggml_cuda_fattn_kvarn_hslice_sign(
        const int row,
        const int col) {
    return (__popc((unsigned) (row & col)) & 1) ? -1.0f : 1.0f;
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
    side.eager_records = ggml_get_op_params_i32(cur, GGML_CUDA_FATTN_KVARN_OP_PARAM_EAGER_RECORDS) != 0;
    side.read_indirect = ggml_get_op_params_i32(cur, 10) != 0;
    const int head_slices_param = ggml_get_op_params_i32(cur, GGML_CUDA_FATTN_KVARN_OP_PARAM_HEAD_SLICES);
    side.head_slices = head_slices_param > 0 ? head_slices_param : 1;
    side.stage_groups = ggml_get_op_params_i32(cur, GGML_CUDA_FATTN_KVARN_OP_PARAM_STAGE_GROUPS);
    const int tail_groups_param = ggml_get_op_params_i32(cur, GGML_CUDA_FATTN_KVARN_OP_PARAM_TAIL_GROUPS);
    side.tail_groups = tail_groups_param > 0 ? tail_groups_param : side.stage_groups - 1;

    if (!ggml_cuda_fattn_kvarn_valid_bits(side.bits) || side.n_stream <= 0 || side.stage_groups < 2 ||
            side.tail_groups <= 0 || side.tail_groups > side.stage_groups ||
            !(side.head_slices == 1 || side.head_slices == 2 || side.head_slices == 4)) {
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
    if (side.groups_per_stream <= 0) {
        return false;
    }
    return side.stream_start >= 0 && side.stream_start + side.n_stream <= total_streams;
}

static inline bool ggml_cuda_fattn_kvarn_view_supported(
        const int device,
        const ggml_tensor * dst,
        ggml_cuda_fattn_kvarn_plan * out = nullptr) {
    GGML_UNUSED(device);
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
    if (plan.k.head_slices != plan.v.head_slices ||
            (plan.k.head_slices != 1 && plan.k.head_slices != plan.slices)) {
        return false;
    }
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
    return ggml_cuda_fattn_kvarn_view_supported(device, dst, out);
}
