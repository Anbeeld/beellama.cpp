#pragma once

#include "fattn-mma-kvarn.cuh"

// Device-side KVarN dequant loader definitions (unpack/load_rotated/load_tile + helpers).
// Compiled ONLY in the FA dispatch TU (fattn.cu) -- the sole place the native KVarN kernel
// is instantiated. Keeping these out of fattn-mma-kvarn.cuh means editing the loader does
// NOT recompile every FA MMA instance (each includes fattn-mma-f16.cuh -> fattn-mma-kvarn.cuh,
// which now carries only the declaration). See fattn-mma-kvarn.cuh for the declaration.

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
        const int live_group = desc.live_group;
        group = (int) (abs_pos / GGML_CUDA_FATTN_KVARN_DIM);
        pos   = (int) (abs_pos - (int64_t) group * GGML_CUDA_FATTN_KVARN_DIM);
        const int stage_begin = live_group >= (desc.tail_groups - 1) ? live_group - (desc.tail_groups - 1) : 0;
        from_stage  = group >= stage_begin && group <= live_group;
        from_record = !from_stage && group >= 0 && group < stage_begin &&
            (live_group - group) < desc.groups_per_stream;
        stage_pos = (group % desc.stage_groups) * GGML_CUDA_FATTN_KVARN_DIM + pos;
        record_group = group % desc.groups_per_stream;
    } else {
        const int live_group = desc.live_group;
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

static __device__ __forceinline__ int ggml_cuda_fattn_kvarn_mma_record_payload_bytes(const int bits) {
    return GGML_CUDA_FATTN_KVARN_DIM * GGML_CUDA_FATTN_KVARN_DIM * bits / 8;
}

static __device__ __forceinline__ bool ggml_cuda_fattn_kvarn_mma_group_from_record(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int group) {
    const int live_group = desc.live_group;
    const bool from_stage = group == 0 ||
        (group > 0 && group <= live_group && group + (desc.tail_groups - 1) >= live_group);
    return !from_stage && group < live_group && group < desc.groups_per_stream;
}

template<int D, int stride_tile, int nbatch_fa, int nthreads, bool oob_check, bool dim_major_K>
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
    const bool stream_record = scale_smem != nullptr && !desc.swa && group_first == group_last &&
        ggml_cuda_fattn_kvarn_mma_group_from_record(desc, group_first);
    const int record_group = desc.stream * desc.groups_per_stream + group_first;

    for (int slice = dim2_start / dim2_per_slice; slice < (dim2_end + dim2_per_slice - 1) / dim2_per_slice; ++slice) {
        const int slice_dim2_start = slice * dim2_per_slice;
        const int out_dim2_start = max(dim2_start, slice_dim2_start);
        const int out_dim2_end = min(dim2_end, slice_dim2_start + dim2_per_slice);
        if (out_dim2_start >= out_dim2_end) {
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
                for (int i = tid; i < GGML_CUDA_FATTN_KVARN_DIM; i += nthreads) {
                    if (desc.value) {
                        scale_smem[i] = other_axis[i];
                    } else {
                        scale_smem[i] = scale_axis[i];
                        scale_smem[GGML_CUDA_FATTN_KVARN_DIM + i] = zp_axis[i];
                    }
                }
                __syncthreads();
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
                    const int pos0 = k_start + row0 - group_first * GGML_CUDA_FATTN_KVARN_DIM;
                    const int pos1 = k_start + row1 - group_first * GGML_CUDA_FATTN_KVARN_DIM;
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
            for (int row = tid; row < nbatch_fa; row += nthreads) {
                const bool valid_row = !oob_check || row < i_sup;
                if (!valid_row) {
                    for (int global_b = out_dim2_start; global_b < out_dim2_end; ++global_b) {
                        tile_KV[row * stride_tile + global_b - dim2_start] = make_half2(0.0f, 0.0f);
                    }
                    continue;
                }

                const int token = k_start + row;
                const int pos = token - group_first * GGML_CUDA_FATTN_KVARN_DIM;
                const float token_scale = stream_record && desc.value ? __half2float(scale_axis[pos]) : 0.0f;
                const float token_zp    = stream_record && desc.value ? __half2float(zp_axis[pos])    : 0.0f;
                const float token_other = stream_record && !desc.value ? __half2float(other_axis[pos]) : 0.0f;
                for (int global_b = out_dim2_start; global_b < out_dim2_end; ++global_b) {
                    const int dim = 2 * (global_b - slice_dim2_start);
                    if (stream_record) {
                        if (desc.value) {
                            const uint8_t q0 = ggml_cuda_fattn_kvarn_unpack_record(record, pos * GGML_CUDA_FATTN_KVARN_DIM + dim + 0, desc.bits);
                            const uint8_t q1 = ggml_cuda_fattn_kvarn_unpack_record(record, pos * GGML_CUDA_FATTN_KVARN_DIM + dim + 1, desc.bits);
                            const float x0 = (float(q0) * token_scale + token_zp) * __half2float(scale_smem[dim + 0]);
                            const float x1 = (float(q1) * token_scale + token_zp) * __half2float(scale_smem[dim + 1]);
                            tile_KV[row * stride_tile + global_b - dim2_start] = make_half2(x0, x1);
                        } else {
                            const uint8_t q0 = ggml_cuda_fattn_kvarn_unpack_record(record, (dim + 0) * GGML_CUDA_FATTN_KVARN_DIM + pos, desc.bits);
                            const uint8_t q1 = ggml_cuda_fattn_kvarn_unpack_record(record, (dim + 1) * GGML_CUDA_FATTN_KVARN_DIM + pos, desc.bits);
                            const float x0 = (float(q0) * __half2float(scale_smem[dim + 0]) +
                                    __half2float(scale_smem[GGML_CUDA_FATTN_KVARN_DIM + dim + 0])) * token_other;
                            const float x1 = (float(q1) * __half2float(scale_smem[dim + 1]) +
                                    __half2float(scale_smem[GGML_CUDA_FATTN_KVARN_DIM + dim + 1])) * token_other;
                            tile_KV[row * stride_tile + global_b - dim2_start] = make_half2(x0, x1);
                        }
                    } else {
                        const float x0 = ggml_cuda_fattn_kvarn_load_rotated(desc, token, slice, dim + 0);
                        const float x1 = ggml_cuda_fattn_kvarn_load_rotated(desc, token, slice, dim + 1);
                        tile_KV[row * stride_tile + global_b - dim2_start] = make_half2(x0, x1);
                    }
                }
            }
        }

        if (stream_record) {
            __syncthreads();
        }
    }
}
