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
