#pragma once

#include "fattn-mma-kvarn.cuh"

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

// Decode two consecutive values at an even element index. KVarN value records
// are row-major and the MMA loader consumes dimensions in pairs, so one packed
// window and one bit-offset calculation serve both output lanes.
static __device__ __forceinline__ uint16_t ggml_cuda_fattn_kvarn_unpack_record_pair(
        const uint8_t * record, const int index, const int bits) {
    if (bits == 8) {
        return (uint16_t) record[index] | ((uint16_t) record[index + 1] << 8);
    }
    if (bits == 4) {
        const uint8_t packed = record[index >> 1];
        return (uint16_t) (packed & 0x0fu) | ((uint16_t) (packed >> 4) << 8);
    }
    if (bits == 2) {
        const uint8_t packed = record[index >> 2];
        const int shift = (index & 3) << 1;
        return (uint16_t) ((packed >> shift) & 0x03u) |
            ((uint16_t) ((packed >> (shift + 2)) & 0x03u) << 8);
    }
    const int bit_offset = index * bits;
    const int byte_offset = bit_offset >> 3;
    const int bit_in_byte = bit_offset & 7;
    const uint16_t packed = (uint16_t) record[byte_offset] | ((uint16_t) record[byte_offset + 1] << 8);
    const uint16_t mask = (1u << bits) - 1u;
    return (uint16_t) ((packed >> bit_in_byte) & mask) |
        (uint16_t) (((packed >> (bit_in_byte + bits)) & mask) << 8);
}

static __device__ __forceinline__ float ggml_cuda_fattn_kvarn_load_stage_rotated(
        const ggml_cuda_fattn_kvarn_desc & desc,
        const int stage_pos,
        const int record_head,
        const int dim) {
    const int64_t base = ((int64_t) stage_pos * desc.n_record_heads + record_head) *
        GGML_CUDA_FATTN_KVARN_DIM;
    return __half2float(desc.stage[base + dim]);
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
        group = (int) (abs_pos / GGML_CUDA_FATTN_KVARN_DIM);
        pos   = (int) (abs_pos - (int64_t) group * GGML_CUDA_FATTN_KVARN_DIM);
        from_stage  = ggml_cuda_fattn_kvarn_group_from_stage(desc, group);
        from_record = ggml_cuda_fattn_kvarn_group_from_record(desc, group);
        stage_pos = (group % desc.stage_groups) * GGML_CUDA_FATTN_KVARN_DIM + pos;
        record_group = group % desc.groups_per_stream;
    } else {
        group = token / GGML_CUDA_FATTN_KVARN_DIM;
        pos   = token - group * GGML_CUDA_FATTN_KVARN_DIM;
        from_stage = ggml_cuda_fattn_kvarn_group_from_stage(desc, group);
        from_record = ggml_cuda_fattn_kvarn_group_from_record(desc, group);
        const int stage_base = desc.stream * GGML_CUDA_FATTN_KVARN_DIM * desc.stage_groups;
        stage_pos = stage_base + (group == 0 ? pos :
            GGML_CUDA_FATTN_KVARN_DIM + ((group - 1) % desc.tail_groups) * GGML_CUDA_FATTN_KVARN_DIM + pos);
        record_group = desc.stream * desc.groups_per_stream + group;
    }

    if (from_stage) {
        return ggml_cuda_fattn_kvarn_load_stage_rotated(desc, stage_pos, record_head, dim);
    }

    if (!from_record) {
        return 0.0f;
    }

    const uint8_t * record = desc.records +
        ((int64_t) record_group * desc.n_record_heads + record_head) * desc.record_bytes;
    const int payload_bytes = GGML_CUDA_FATTN_KVARN_DIM * GGML_CUDA_FATTN_KVARN_DIM * desc.bits / 8;
    const half * scale_axis = (const half *) (record + payload_bytes);
    const half * zp_axis    = scale_axis + GGML_CUDA_FATTN_KVARN_DIM;
    const half * other_axis = zp_axis + GGML_CUDA_FATTN_KVARN_DIM;
    const int row = desc.value ? pos : dim;
    const int col = desc.value ? dim : pos;
    const uint8_t q = ggml_cuda_fattn_kvarn_unpack_record(
        record, row * GGML_CUDA_FATTN_KVARN_DIM + col, desc.bits);
    return (float(q) * __half2float(scale_axis[row]) + __half2float(zp_axis[row])) *
        __half2float(other_axis[col]);
}
