#pragma once

#include "fattn-mma-kvarn-load.cuh"

// Generic native-MMA tile loader. The shared record/SWA element loader lives in
// fattn-mma-kvarn-load.cuh so decode instance TUs do not parse this larger fallback path.

static __device__ __forceinline__ int ggml_cuda_fattn_kvarn_mma_record_payload_bytes(const int bits) {
    return GGML_CUDA_FATTN_KVARN_DIM * GGML_CUDA_FATTN_KVARN_DIM * bits / 8;
}

static __device__ __forceinline__ bool ggml_cuda_fattn_kvarn_mma_group_from_record(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int group) {
    return ggml_cuda_fattn_kvarn_group_from_record(desc, group);
}

struct ggml_cuda_fattn_kvarn_mma_tile {
    bool fast;
    bool stage;
    int pos_begin;
    int record_group;
    int stage_pos_begin;
};

static __device__ __forceinline__ ggml_cuda_fattn_kvarn_mma_tile ggml_cuda_fattn_kvarn_mma_plan_tile(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int k_start,
        const int valid_count,
        const int group_first,
        const int group_last) {
    ggml_cuda_fattn_kvarn_mma_tile tile = { false, false, 0, 0, 0 };
    if (valid_count <= 0) {
        return tile;
    }

    if (desc.swa || desc.read_indirect) {
        const int64_t first_encoded = desc.indices[k_start];
        const int64_t last_encoded  = desc.indices[k_start + valid_count - 1];
        if (first_encoded == -1 || last_encoded == -1) {
            return tile;
        }
        bool first_staged;
        bool last_staged;
        const int64_t first_idx = ggml_cuda_fattn_kvarn_read_cell(desc, first_encoded, first_staged);
        const int64_t last_idx = ggml_cuda_fattn_kvarn_read_cell(desc, last_encoded, last_staged);
        if (last_idx != first_idx + valid_count - 1 || first_staged != last_staged) {
            return tile;
        }
        const int group0 = (int) (first_idx / GGML_CUDA_FATTN_KVARN_DIM);
        const int group1 = (int) (last_idx  / GGML_CUDA_FATTN_KVARN_DIM);
        if (group0 != group1) {
            return tile;
        }
        const bool from_stage = first_staged ||
            (!(desc.read_indirect && !desc.swa) && ggml_cuda_fattn_kvarn_group_from_stage(desc, group0));
        const bool from_record = !first_staged && (desc.read_indirect && !desc.swa ? true :
            ggml_cuda_fattn_kvarn_group_from_record(desc, group0));
        tile.pos_begin = (int) (first_idx - (int64_t) group0 * GGML_CUDA_FATTN_KVARN_DIM);
        if (from_stage) {
            const int stage_base = desc.stream * GGML_CUDA_FATTN_KVARN_DIM * desc.stage_groups;
            tile.stage = true;
            tile.stage_pos_begin = desc.swa ?
                (group0 % desc.stage_groups) * GGML_CUDA_FATTN_KVARN_DIM + tile.pos_begin :
                stage_base + (group0 == 0 ? tile.pos_begin :
                    GGML_CUDA_FATTN_KVARN_DIM +
                    ((group0 - 1) % desc.tail_groups) * GGML_CUDA_FATTN_KVARN_DIM + tile.pos_begin);
            return tile;
        }
        if (!from_record) {
            return tile;
        }
        tile.fast = true;
        tile.record_group = desc.swa ? group0 % desc.groups_per_stream :
            desc.stream * desc.groups_per_stream + group0;
        return tile;
    }

    if (group_first != group_last) {
        return tile;
    }

    tile.pos_begin = k_start - group_first * GGML_CUDA_FATTN_KVARN_DIM;
    const bool from_stage = ggml_cuda_fattn_kvarn_group_from_stage(desc, group_first);
    if (from_stage) {
        const int stage_base = desc.stream * GGML_CUDA_FATTN_KVARN_DIM * desc.stage_groups;
        tile.stage = true;
        tile.stage_pos_begin = stage_base + (group_first == 0 ? tile.pos_begin :
            GGML_CUDA_FATTN_KVARN_DIM +
            ((group_first - 1) % desc.tail_groups) * GGML_CUDA_FATTN_KVARN_DIM + tile.pos_begin);
        return tile;
    }

    if (!ggml_cuda_fattn_kvarn_mma_group_from_record(desc, group_first)) {
        return tile;
    }
    tile.fast = true;
    tile.record_group = desc.stream * desc.groups_per_stream + group_first;
    return tile;
}

static __device__ __forceinline__ void ggml_cuda_fattn_kvarn_load_rotated_slice_record_warp(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const uint8_t * __restrict__ record,
        const int pos,
        float * __restrict__ out,
        const half * __restrict__ axis_smem,
        const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const half * scale_axis = axis_smem;
    const half * zp_axis    = scale_axis + GGML_CUDA_FATTN_KVARN_DIM;
    const half * other_axis = zp_axis + GGML_CUDA_FATTN_KVARN_DIM;

    for (int d = lane; d < GGML_CUDA_FATTN_KVARN_DIM; d += warp_size) {
        float x = 0.0f;
        if (desc.value) {
            const uint8_t q = ggml_cuda_fattn_kvarn_unpack_record(
                record, pos * GGML_CUDA_FATTN_KVARN_DIM + d, desc.bits);
            x = (float(q) * __half2float(scale_axis[pos]) + __half2float(zp_axis[pos])) *
                __half2float(other_axis[d]);
        } else {
            const uint8_t q = ggml_cuda_fattn_kvarn_unpack_record(
                record, d * GGML_CUDA_FATTN_KVARN_DIM + pos, desc.bits);
            x = (float(q) * __half2float(scale_axis[d]) + __half2float(zp_axis[d])) *
                __half2float(other_axis[pos]);
        }
        out[d] = x;
    }
}

static __device__ __forceinline__ bool ggml_cuda_fattn_kvarn_load_rotated_slice_warp(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int token,
        const int slice,
        const bool valid_row,
        float * __restrict__ out,
        const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int record_head = desc.head_base + slice;

    int pos = 0;
    int stage_pos = 0;
    int record_group = 0;
    bool from_stage = false;
    bool from_record = false;

    if (valid_row) {
        if (desc.swa || desc.read_indirect) {
            const int64_t encoded = desc.indices[token];
            if (encoded != -1) {
                bool explicitly_staged;
                const int64_t abs_pos = ggml_cuda_fattn_kvarn_read_cell(desc, encoded, explicitly_staged);
                const int group = (int) (abs_pos / GGML_CUDA_FATTN_KVARN_DIM);
                pos = (int) (abs_pos - (int64_t) group * GGML_CUDA_FATTN_KVARN_DIM);
                from_stage = explicitly_staged ||
                    (!(desc.read_indirect && !desc.swa) && ggml_cuda_fattn_kvarn_group_from_stage(desc, group));
                from_record = !explicitly_staged && (desc.read_indirect && !desc.swa ? true :
                    ggml_cuda_fattn_kvarn_group_from_record(desc, group));
                const int stage_base = desc.stream * GGML_CUDA_FATTN_KVARN_DIM * desc.stage_groups;
                stage_pos = desc.swa ?
                    (group % desc.stage_groups) * GGML_CUDA_FATTN_KVARN_DIM + pos :
                    stage_base + (group == 0 ? pos : GGML_CUDA_FATTN_KVARN_DIM +
                        ((group - 1) % desc.tail_groups) * GGML_CUDA_FATTN_KVARN_DIM + pos);
                record_group = desc.swa ? group % desc.groups_per_stream :
                    desc.stream * desc.groups_per_stream + group;
            }
        } else {
            const int group = token / GGML_CUDA_FATTN_KVARN_DIM;
            pos = token - group * GGML_CUDA_FATTN_KVARN_DIM;
            from_stage = ggml_cuda_fattn_kvarn_group_from_stage(desc, group);
            from_record = ggml_cuda_fattn_kvarn_group_from_record(desc, group);
            const int stage_base = desc.stream * GGML_CUDA_FATTN_KVARN_DIM * desc.stage_groups;
            stage_pos = stage_base + (group == 0 ? pos :
                GGML_CUDA_FATTN_KVARN_DIM + ((group - 1) % desc.tail_groups) * GGML_CUDA_FATTN_KVARN_DIM + pos);
            record_group = desc.stream * desc.groups_per_stream + group;
        }
    }

    const uint8_t * record = nullptr;
    const half * scale_axis = nullptr;
    const half * zp_axis = nullptr;
    const half * other_axis = nullptr;
    if (from_record) {
        record = desc.records +
            ((int64_t) record_group * desc.n_record_heads + record_head) * desc.record_bytes;
        const int payload_bytes = ggml_cuda_fattn_kvarn_mma_record_payload_bytes(desc.bits);
        scale_axis = (const half *) (record + payload_bytes);
        zp_axis    = scale_axis + GGML_CUDA_FATTN_KVARN_DIM;
        other_axis = zp_axis + GGML_CUDA_FATTN_KVARN_DIM;
    }

    for (int d = lane; d < GGML_CUDA_FATTN_KVARN_DIM; d += warp_size) {
        float x = 0.0f;
        if (from_stage) {
            x = __half2float(desc.stage[((int64_t) stage_pos * desc.n_record_heads + record_head) *
                GGML_CUDA_FATTN_KVARN_DIM + d]);
        } else if (from_record) {
            const int row = desc.value ? pos : d;
            const int col = desc.value ? d : pos;
            const uint8_t q = ggml_cuda_fattn_kvarn_unpack_record(
                record, row * GGML_CUDA_FATTN_KVARN_DIM + col, desc.bits);
            x = (float(q) * __half2float(scale_axis[row]) + __half2float(zp_axis[row])) *
                __half2float(other_axis[col]);
        }
        out[d] = x;
    }
    return from_stage;
}

static __device__ __forceinline__ float * ggml_cuda_fattn_kvarn_inverse_wht_128_warp(
        float * __restrict__ buf0,
        float * __restrict__ buf1,
        const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    float * src = buf0;
    float * dst = buf1;
    __syncwarp();
    for (int stride = 1; stride < GGML_CUDA_FATTN_KVARN_DIM; stride *= 2) {
        for (int d = lane; d < GGML_CUDA_FATTN_KVARN_DIM; d += warp_size) {
            const float a = src[d];
            const float b = src[d ^ stride];
            dst[d] = (d & stride) ? (b - a) : (a + b);
        }
        __syncwarp();
        float * tmp = src;
        src = dst;
        dst = tmp;
    }
    for (int d = lane; d < GGML_CUDA_FATTN_KVARN_DIM; d += warp_size) {
        src[d] *= 0.08838834764831845f; // 1/sqrt(128)
    }
    __syncwarp();
    return src;
}

template<int D, int stride_tile, int nbatch_fa, int nthreads, bool oob_check,
    bool original_domain, bool dim_major_K, bool cache_record_axes>
static __device__ __forceinline__ void flash_attn_ext_kvarn_load_tile(
        const char * __restrict__ desc_raw,
        half2      * __restrict__ tile_KV,
        const int k_start,
        const int i_sup,
        const int dim2_start,
        const int dim2_count,
        half      * __restrict__ scale_smem) {
    const ggml_cuda_fattn_kvarn_desc & desc = *(const ggml_cuda_fattn_kvarn_desc *) desc_raw;
    static_assert(D % GGML_CUDA_FATTN_KVARN_DIM == 0 && D <= 512, "KVarN native MMA supports 128-wide slices through D=512");
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int dim2_per_slice = GGML_CUDA_FATTN_KVARN_DIM / 2;
    const int tid = threadIdx.y * warp_size + threadIdx.x;
    const int dim2_end = dim2_start + dim2_count;
    const int valid_count = oob_check ? min(i_sup, nbatch_fa) : nbatch_fa;
    const int group_first = valid_count > 0 ? k_start / GGML_CUDA_FATTN_KVARN_DIM : -1;
    const int group_last  = valid_count > 0 ? (k_start + valid_count - 1) / GGML_CUDA_FATTN_KVARN_DIM : -2;
    const ggml_cuda_fattn_kvarn_mma_tile tile =
        ggml_cuda_fattn_kvarn_mma_plan_tile(desc, k_start, valid_count, group_first, group_last);
    const bool fast_record = scale_smem != nullptr && tile.fast && !tile.stage;
    const bool fast_stage = tile.stage;
    const bool stream_record = fast_record;
    const int record_group = tile.record_group;

    for (int slice = dim2_start / dim2_per_slice; slice < (dim2_end + dim2_per_slice - 1) / dim2_per_slice; ++slice) {
        const int slice_dim2_start = slice * dim2_per_slice;
        const int out_dim2_start = max(dim2_start, slice_dim2_start);
        const int out_dim2_end = min(dim2_end, slice_dim2_start + dim2_per_slice);
        if (out_dim2_start >= out_dim2_end) {
            continue;
        }

        const bool fast_stage_direct = fast_stage && !original_domain;
        if (fast_stage_direct) {
            const int dim2_count_local = out_dim2_end - out_dim2_start;
            constexpr int h2_per_chunk = 16 / sizeof(half2);
            const int dst_h2_begin = out_dim2_start - dim2_start;
            const int src_h2_begin = out_dim2_start - slice_dim2_start;
            const bool chunk_aligned = (dst_h2_begin % h2_per_chunk) == 0 &&
                (src_h2_begin % h2_per_chunk) == 0;
            const int record_head = desc.head_base + slice;

            if constexpr (stride_tile % h2_per_chunk == 0) {
                if (chunk_aligned) {
                    const half2 zero[4] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
                    const int chunks_per_row = dim2_count_local / h2_per_chunk;
                    for (int idx = tid; idx < nbatch_fa * chunks_per_row; idx += nthreads) {
                        const int row = idx / chunks_per_row;
                        const int chunk = idx - row * chunks_per_row;
                        const int dst_h2 = dst_h2_begin + chunk * h2_per_chunk;
                        const int src_h2 = src_h2_begin + chunk * h2_per_chunk;
                        const int stage_pos = tile.stage_pos_begin + row;
                        const int64_t base = ((int64_t) stage_pos * desc.n_record_heads + record_head) *
                            GGML_CUDA_FATTN_KVARN_DIM;
                        const half2 * src = row < valid_count ?
                            (const half2 *) (desc.stage + base + 2 * src_h2) : zero;
                        ggml_cuda_memcpy_1<16>(tile_KV + row * stride_tile + dst_h2, src);
                    }

                    const int copied_h2 = chunks_per_row * h2_per_chunk;
                    const int tail_h2 = dim2_count_local - copied_h2;
                    if (tail_h2 > 0) {
                        for (int idx = tid; idx < nbatch_fa * tail_h2; idx += nthreads) {
                            const int row = idx / tail_h2;
                            const int dim2_tail = copied_h2 + idx - row * tail_h2;
                            const int dst_h2 = dst_h2_begin + dim2_tail;
                            const int src_h2 = src_h2_begin + dim2_tail;
                            if (row >= valid_count) {
                                tile_KV[row * stride_tile + dst_h2] = make_half2(0.0f, 0.0f);
                                continue;
                            }
                            const int stage_pos = tile.stage_pos_begin + row;
                            const int64_t base = ((int64_t) stage_pos * desc.n_record_heads + record_head) *
                                GGML_CUDA_FATTN_KVARN_DIM;
                            tile_KV[row * stride_tile + dst_h2] =
                                ((const half2 *) (desc.stage + base))[src_h2];
                        }
                    }
                    continue;
                }
            }

            for (int idx = tid; idx < nbatch_fa * dim2_count_local; idx += nthreads) {
                const int row = idx / dim2_count_local;
                const int dim2_local = idx - row * dim2_count_local;
                const int dst_h2 = dst_h2_begin + dim2_local;
                const int src_h2 = src_h2_begin + dim2_local;
                if (row >= valid_count) {
                    tile_KV[row * stride_tile + dst_h2] = make_half2(0.0f, 0.0f);
                    continue;
                }
                const int stage_pos = tile.stage_pos_begin + row;
                const int64_t base = ((int64_t) stage_pos * desc.n_record_heads + record_head) *
                    GGML_CUDA_FATTN_KVARN_DIM;
                tile_KV[row * stride_tile + dst_h2] =
                    ((const half2 *) (desc.stage + base))[src_h2];
            }
            continue;
        }

        if (fast_stage) {
            constexpr int warps_per_block = nthreads / warp_size;
            constexpr int slices = D/GGML_CUDA_FATTN_KVARN_DIM;
            const int warp = tid/warp_size;
            const int lane = tid - warp*warp_size;
            half * scratch_base_h = scale_smem + 3*GGML_CUDA_FATTN_KVARN_DIM;
            float * scratch0 = (float *) scratch_base_h;
            float * scratch1 = scratch0 + warps_per_block*GGML_CUDA_FATTN_KVARN_DIM;
            const float slice_scale = slices == 1 ? 1.0f :
                (slices == 2 ? 0.7071067811865475f : 0.5f);

            for (int row = warp; row < nbatch_fa; row += warps_per_block) {
                float * row0 = scratch0 + warp*GGML_CUDA_FATTN_KVARN_DIM;
                float * row1 = scratch1 + warp*GGML_CUDA_FATTN_KVARN_DIM;
                const bool valid_row = row < valid_count;
                const int stage_pos = tile.stage_pos_begin + row;
                for (int d = lane; d < GGML_CUDA_FATTN_KVARN_DIM; d += warp_size) {
                    row0[d] = 0.0f;
                }
                __syncwarp();

                for (int src_slice = 0; src_slice < slices; ++src_slice) {
                    const int record_head = desc.head_base + src_slice;
                    const int64_t base = ((int64_t) stage_pos*desc.n_record_heads + record_head)*
                        GGML_CUDA_FATTN_KVARN_DIM;
                    for (int d = lane; d < GGML_CUDA_FATTN_KVARN_DIM; d += warp_size) {
                        row1[d] = valid_row ? __half2float(desc.stage[base + d]) : 0.0f;
                    }
                    __syncwarp();
                    const float sign = ggml_cuda_fattn_kvarn_hslice_sign(slice, src_slice);
                    for (int d = lane; d < GGML_CUDA_FATTN_KVARN_DIM; d += warp_size) {
                        row0[d] += sign*row1[d];
                    }
                    __syncwarp();
                }
                for (int d = lane; d < GGML_CUDA_FATTN_KVARN_DIM; d += warp_size) {
                    row0[d] *= slice_scale;
                }
                __syncwarp();
                float * orig = ggml_cuda_fattn_kvarn_inverse_wht_128_warp(row0, row1, lane);
                for (int global_b = out_dim2_start + lane;
                        global_b < out_dim2_end; global_b += warp_size) {
                    const int dim = 2*(global_b - slice_dim2_start);
                    tile_KV[row*stride_tile + global_b - dim2_start] =
                        make_half2(orig[dim], orig[dim + 1]);
                }
                __syncwarp();
            }
            continue;
        }

        if constexpr (original_domain) {
            static_assert(!dim_major_K, "KVarN original-domain loader expects row-major MMA tiles");
            constexpr int warps_per_block = nthreads / warp_size;
            static_assert(warps_per_block > 0, "bad KVarN original-domain warp count");
            const int warp = tid / warp_size;
            const int lane = tid - warp * warp_size;
            float * scratch0 = (float *) (scale_smem + 3 * GGML_CUDA_FATTN_KVARN_DIM);
            float * scratch1 = scratch0 + warps_per_block * GGML_CUDA_FATTN_KVARN_DIM;

            if (desc.head_slices > 1) {
                const float inv_sqrt_slices = desc.head_slices == 2 ? 0.7071067811865475f : 0.5f;

                for (int row = warp; row < nbatch_fa; row += warps_per_block) {
                    const bool valid_row = !oob_check || row < i_sup;
                    float * row0 = scratch0 + warp * GGML_CUDA_FATTN_KVARN_DIM;
                    float * row1 = scratch1 + warp * GGML_CUDA_FATTN_KVARN_DIM;
                    for (int d = lane; d < GGML_CUDA_FATTN_KVARN_DIM; d += warp_size) {
                        row0[d] = 0.0f;
                    }
                    __syncwarp();

                    for (int src_slice = 0; src_slice < desc.head_slices; ++src_slice) {
                        ggml_cuda_fattn_kvarn_load_rotated_slice_warp(
                                desc, k_start + row, src_slice, valid_row, row1, lane);
                        __syncwarp();
                        const float sign = ggml_cuda_fattn_kvarn_hslice_sign(slice, src_slice);
                        for (int d = lane; d < GGML_CUDA_FATTN_KVARN_DIM; d += warp_size) {
                            row0[d] += sign * row1[d];
                        }
                        __syncwarp();
                    }

                    for (int d = lane; d < GGML_CUDA_FATTN_KVARN_DIM; d += warp_size) {
                        row0[d] *= inv_sqrt_slices;
                    }
                    __syncwarp();

                    float * orig = ggml_cuda_fattn_kvarn_inverse_wht_128_warp(row0, row1, lane);
                    for (int global_b = out_dim2_start + lane; global_b < out_dim2_end; global_b += warp_size) {
                        const int dim = 2 * (global_b - slice_dim2_start);
                        tile_KV[row * stride_tile + global_b - dim2_start] = make_half2(orig[dim], orig[dim + 1]);
                    }
                    __syncwarp();
                }
                continue;
            }

            const uint8_t * record = nullptr;
            if (fast_record) {
                record = desc.records + ((int64_t) record_group * desc.n_record_heads + desc.head_base + slice) * desc.record_bytes;
                const int payload_bytes = ggml_cuda_fattn_kvarn_mma_record_payload_bytes(desc.bits);
                const half * scale_axis = (const half *) (record + payload_bytes);
                const half * zp_axis    = scale_axis + GGML_CUDA_FATTN_KVARN_DIM;
                const half * other_axis = zp_axis + GGML_CUDA_FATTN_KVARN_DIM;
                for (int i = tid; i < GGML_CUDA_FATTN_KVARN_DIM; i += nthreads) {
                    scale_smem[i] = scale_axis[i];
                    scale_smem[GGML_CUDA_FATTN_KVARN_DIM + i] = zp_axis[i];
                    scale_smem[2 * GGML_CUDA_FATTN_KVARN_DIM + i] = other_axis[i];
                }
                __syncthreads();
            }

            for (int row = warp; row < nbatch_fa; row += warps_per_block) {
                const bool valid_row = !oob_check || row < i_sup;
                float * row0 = scratch0 + warp * GGML_CUDA_FATTN_KVARN_DIM;
                float * row1 = scratch1 + warp * GGML_CUDA_FATTN_KVARN_DIM;
                if (fast_record && valid_row) {
                    ggml_cuda_fattn_kvarn_load_rotated_slice_record_warp(
                        desc, record, tile.pos_begin + row, row0, scale_smem, lane);
                } else {
                    ggml_cuda_fattn_kvarn_load_rotated_slice_warp(
                        desc, k_start + row, slice, valid_row, row0, lane);
                }
                float * orig = ggml_cuda_fattn_kvarn_inverse_wht_128_warp(row0, row1, lane);
                for (int global_b = out_dim2_start + lane; global_b < out_dim2_end; global_b += warp_size) {
                    const int dim = 2 * (global_b - slice_dim2_start);
                    tile_KV[row * stride_tile + global_b - dim2_start] = make_half2(orig[dim], orig[dim + 1]);
                }
                __syncwarp();
            }
            if (fast_record) {
                __syncthreads();
            }
            continue;
        }

        const uint8_t * record = nullptr;
        const half * scale_axis = nullptr;
        const half * zp_axis = nullptr;
        const half * other_axis = nullptr;

        if (stream_record) {
            record = desc.records + ((int64_t) record_group * desc.n_record_heads + desc.head_base + slice) * desc.record_bytes;
            const int payload_bytes = ggml_cuda_fattn_kvarn_mma_record_payload_bytes(desc.bits);
            scale_axis = (const half *) (record + payload_bytes);
            zp_axis    = scale_axis + GGML_CUDA_FATTN_KVARN_DIM;
            other_axis = zp_axis + GGML_CUDA_FATTN_KVARN_DIM;
            if constexpr (!dim_major_K) {
                if constexpr (cache_record_axes) {
                    constexpr int slices = D / GGML_CUDA_FATTN_KVARN_DIM;
                    half * axis_smem = scale_smem +
                        (desc.value ? 3 * D : 0) + slice * 3 * GGML_CUDA_FATTN_KVARN_DIM;
                    int * axis_group_tags = (int *) (scale_smem + 6 * D);
                    const int axis_tag = (desc.value ? slices : 0) + slice;
                    const bool load_axes = axis_group_tags[axis_tag] != record_group;
                    if (load_axes) {
                        for (int i = tid; i < GGML_CUDA_FATTN_KVARN_DIM; i += nthreads) {
                            axis_smem[i] = scale_axis[i];
                            axis_smem[GGML_CUDA_FATTN_KVARN_DIM + i] = zp_axis[i];
                            axis_smem[2 * GGML_CUDA_FATTN_KVARN_DIM + i] = other_axis[i];
                        }
                        __syncthreads();
                        if (tid == 0) {
                            axis_group_tags[axis_tag] = record_group;
                        }
                    }
                    scale_axis = axis_smem;
                    zp_axis    = axis_smem + GGML_CUDA_FATTN_KVARN_DIM;
                    other_axis = zp_axis + GGML_CUDA_FATTN_KVARN_DIM;
                } else {
                    for (int i = tid; i < GGML_CUDA_FATTN_KVARN_DIM; i += nthreads) {
                        if (desc.value) {
                            scale_smem[i] = other_axis[i];
                        } else {
                            scale_smem[i] = scale_axis[i];
                            scale_smem[GGML_CUDA_FATTN_KVARN_DIM + i] = zp_axis[i];
                        }
                    }
                    __syncthreads();
                    if (desc.value) {
                        other_axis = scale_smem;
                    } else {
                        scale_axis = scale_smem;
                        zp_axis = scale_smem + GGML_CUDA_FATTN_KVARN_DIM;
                    }
                }
            }
        }

        if constexpr (dim_major_K) {
            if (!desc.value) {
                const int dim_begin = 2 * (out_dim2_start - slice_dim2_start);
                const int dim_end   = 2 * (out_dim2_end   - slice_dim2_start);
                constexpr int token_pairs = nbatch_fa / 2;
                static_assert(nthreads >= token_pairs && nthreads % token_pairs == 0, "bad KVarN dim-major loader shape");
                constexpr int dim_workers = nthreads / token_pairs;
                const int tok2 = tid % token_pairs;
                const int dim_lane = tid / token_pairs;
                if (dim_lane < dim_workers) {
                    const int row0 = 2 * tok2 + 0;
                    const int row1 = 2 * tok2 + 1;
                    const bool valid0 = !oob_check || row0 < i_sup;
                    const bool valid1 = !oob_check || row1 < i_sup;
                    const int pos0 = tile.pos_begin + row0;
                    const int pos1 = tile.pos_begin + row1;
                    const float other0 = stream_record && valid0 ? __half2float(other_axis[pos0]) : 0.0f;
                    const float other1 = stream_record && valid1 ? __half2float(other_axis[pos1]) : 0.0f;
                    for (int dim = dim_begin + dim_lane; dim < dim_end; dim += dim_workers) {
                        const int tile_dim = 2 * (out_dim2_start - dim2_start) + (dim - dim_begin);
                        const float dim_scale = stream_record ? __half2float(scale_axis[dim]) : 0.0f;
                        const float dim_zp    = stream_record ? __half2float(zp_axis[dim]) : 0.0f;
                        float x0 = 0.0f;
                        float x1 = 0.0f;
                        if (stream_record) {
                            if (valid0) {
                                const uint8_t q0 = ggml_cuda_fattn_kvarn_unpack_record(record, dim * GGML_CUDA_FATTN_KVARN_DIM + pos0, desc.bits);
                                x0 = (float(q0) * dim_scale + dim_zp) * other0;
                            }
                            if (valid1) {
                                const uint8_t q1 = ggml_cuda_fattn_kvarn_unpack_record(record, dim * GGML_CUDA_FATTN_KVARN_DIM + pos1, desc.bits);
                                x1 = (float(q1) * dim_scale + dim_zp) * other1;
                            }
                        } else {
                            if (valid0) {
                                x0 = ggml_cuda_fattn_kvarn_load_rotated(desc, k_start + row0, slice, dim);
                            }
                            if (valid1) {
                                x1 = ggml_cuda_fattn_kvarn_load_rotated(desc, k_start + row1, slice, dim);
                            }
                        }
                        tile_KV[tile_dim * stride_tile + tok2] = make_half2(x0, x1);
                    }
                }
            } else {
                GGML_UNUSED(scale_smem);
            }
        } else {
            const int dim2_count_local = out_dim2_end - out_dim2_start;
            if constexpr (cache_record_axes) {
                if (stream_record && nthreads % dim2_count_local == 0) {
                    const int dim2_lane = tid % dim2_count_local;
                    const int row_lane = tid / dim2_count_local;
                    const int row_stride = nthreads / dim2_count_local;
                    const int global_b = out_dim2_start + dim2_lane;
                    const int dim = 2 * (global_b - slice_dim2_start);

                    if (desc.value) {
                        const float2 other = __half22float2(*(const half2 *) (other_axis + dim));
                        for (int row = row_lane; row < nbatch_fa; row += row_stride) {
                            const bool valid_row = !oob_check || row < i_sup;
                            if (!valid_row) {
                                tile_KV[row * stride_tile + global_b - dim2_start] = make_half2(0.0f, 0.0f);
                                continue;
                            }
                            const int pos = tile.pos_begin + row;
                            const float token_scale = __half2float(scale_axis[pos]);
                            const float token_zp = __half2float(zp_axis[pos]);
                            const uint16_t q01 = ggml_cuda_fattn_kvarn_unpack_record_pair(
                                record, pos * GGML_CUDA_FATTN_KVARN_DIM + dim, desc.bits);
                            const float x0 = (float(q01 & 0xffu) * token_scale + token_zp) * other.x;
                            const float x1 = (float(q01 >> 8) * token_scale + token_zp) * other.y;
                            tile_KV[row * stride_tile + global_b - dim2_start] = make_half2(x0, x1);
                        }
                    } else {
                        const float2 dim_scale = __half22float2(*(const half2 *) (scale_axis + dim));
                        const float2 dim_zp = __half22float2(*(const half2 *) (zp_axis + dim));
                        if ((tile.pos_begin & 1) == 0) {
                            // Key records are dimension-major, so adjacent tokens share a
                            // packed bit window. Pair them while keeping a dimension pair
                            // stationary in each thread; this halves address/bit extraction
                            // work without changing reconstruction arithmetic.
                            const int row_pair_lane = row_lane;
                            const int row_pair_stride = row_stride;
                            for (int row_pair = row_pair_lane; row_pair < (nbatch_fa + 1) / 2; row_pair += row_pair_stride) {
                                const int row0 = 2 * row_pair;
                                const int row1 = row0 + 1;
                                const bool valid0 = !oob_check || row0 < i_sup;
                                const bool valid1 = row1 < nbatch_fa && (!oob_check || row1 < i_sup);
                                const int pos0 = tile.pos_begin + row0;
                                const uint16_t q0_pair = ggml_cuda_fattn_kvarn_unpack_record_pair(
                                    record, (dim + 0) * GGML_CUDA_FATTN_KVARN_DIM + pos0, desc.bits);
                                const uint16_t q1_pair = ggml_cuda_fattn_kvarn_unpack_record_pair(
                                    record, (dim + 1) * GGML_CUDA_FATTN_KVARN_DIM + pos0, desc.bits);
                                const float2 token_other = __half22float2(*(const half2 *) (other_axis + pos0));
                                if (valid0) {
                                    const float x0 = (float(q0_pair & 0xffu) * dim_scale.x + dim_zp.x) * token_other.x;
                                    const float x1 = (float(q1_pair & 0xffu) * dim_scale.y + dim_zp.y) * token_other.x;
                                    tile_KV[row0 * stride_tile + global_b - dim2_start] = make_half2(x0, x1);
                                } else {
                                    tile_KV[row0 * stride_tile + global_b - dim2_start] = make_half2(0.0f, 0.0f);
                                }
                                if (row1 < nbatch_fa) {
                                    if (valid1) {
                                        const float x0 = (float(q0_pair >> 8) * dim_scale.x + dim_zp.x) * token_other.y;
                                        const float x1 = (float(q1_pair >> 8) * dim_scale.y + dim_zp.y) * token_other.y;
                                        tile_KV[row1 * stride_tile + global_b - dim2_start] = make_half2(x0, x1);
                                    } else {
                                        tile_KV[row1 * stride_tile + global_b - dim2_start] = make_half2(0.0f, 0.0f);
                                    }
                                }
                            }
                        } else {
                            for (int row = row_lane; row < nbatch_fa; row += row_stride) {
                                const bool valid_row = !oob_check || row < i_sup;
                                if (!valid_row) {
                                    tile_KV[row * stride_tile + global_b - dim2_start] = make_half2(0.0f, 0.0f);
                                    continue;
                                }
                                const int pos = tile.pos_begin + row;
                                const float token_other = __half2float(other_axis[pos]);
                                const uint8_t q0 = ggml_cuda_fattn_kvarn_unpack_record(
                                    record, (dim + 0) * GGML_CUDA_FATTN_KVARN_DIM + pos, desc.bits);
                                const uint8_t q1 = ggml_cuda_fattn_kvarn_unpack_record(
                                    record, (dim + 1) * GGML_CUDA_FATTN_KVARN_DIM + pos, desc.bits);
                                const float x0 = (float(q0) * dim_scale.x + dim_zp.x) * token_other;
                                const float x1 = (float(q1) * dim_scale.y + dim_zp.y) * token_other;
                                tile_KV[row * stride_tile + global_b - dim2_start] = make_half2(x0, x1);
                            }
                        }
                    }
                    continue;
                }
            }

            for (int idx = tid; idx < nbatch_fa * dim2_count_local; idx += nthreads) {
                const int row = idx / dim2_count_local;
                const int global_b = out_dim2_start + idx - row * dim2_count_local;
                const bool valid_row = !oob_check || row < i_sup;
                if (!valid_row) {
                    tile_KV[row * stride_tile + global_b - dim2_start] = make_half2(0.0f, 0.0f);
                    continue;
                }

                const int token = k_start + row;
                const int pos = stream_record ? tile.pos_begin + row :
                    token - group_first * GGML_CUDA_FATTN_KVARN_DIM;
                const float token_scale = stream_record && desc.value ? __half2float(scale_axis[pos]) : 0.0f;
                const float token_zp    = stream_record && desc.value ? __half2float(zp_axis[pos])    : 0.0f;
                const float token_other = stream_record && !desc.value ? __half2float(other_axis[pos]) : 0.0f;
                const int dim = 2 * (global_b - slice_dim2_start);
                if (stream_record) {
                    if (desc.value) {
                        const uint16_t q01 = ggml_cuda_fattn_kvarn_unpack_record_pair(
                            record, pos * GGML_CUDA_FATTN_KVARN_DIM + dim, desc.bits);
                        const float2 other = __half22float2(*(const half2 *) (other_axis + dim));
                        const float x0 = (float(q01 & 0xffu) * token_scale + token_zp) * other.x;
                        const float x1 = (float(q01 >> 8) * token_scale + token_zp) * other.y;
                        tile_KV[row * stride_tile + global_b - dim2_start] = make_half2(x0, x1);
                    } else {
                        const uint8_t q0 = ggml_cuda_fattn_kvarn_unpack_record(record, (dim + 0) * GGML_CUDA_FATTN_KVARN_DIM + pos, desc.bits);
                        const uint8_t q1 = ggml_cuda_fattn_kvarn_unpack_record(record, (dim + 1) * GGML_CUDA_FATTN_KVARN_DIM + pos, desc.bits);
                        const float2 dim_scale = __half22float2(*(const half2 *) (scale_axis + dim));
                        const float2 dim_zp = __half22float2(*(const half2 *) (zp_axis + dim));
                        const float x0 = (float(q0) * dim_scale.x + dim_zp.x) * token_other;
                        const float x1 = (float(q1) * dim_scale.y + dim_zp.y) * token_other;
                        tile_KV[row * stride_tile + global_b - dim2_start] = make_half2(x0, x1);
                    }
                } else {
                    const float x0 = ggml_cuda_fattn_kvarn_load_rotated(desc, token, slice, dim + 0);
                    const float x1 = ggml_cuda_fattn_kvarn_load_rotated(desc, token, slice, dim + 1);
                    tile_KV[row * stride_tile + global_b - dim2_start] = make_half2(x0, x1);
                }
            }
        }

        if (stream_record && !cache_record_axes) {
            __syncthreads();
        }
    }
}
