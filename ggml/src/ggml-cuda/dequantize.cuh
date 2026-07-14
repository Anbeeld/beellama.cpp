#include "common.cuh"

static __device__ __forceinline__ void dequantize_iq4_nl(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq4_nl * x = (const block_iq4_nl *) vx;
    const float d = x[ib].d;
    const uint8_t q = x[ib].qs[iqs];

    v.x = d*kvalues_iq4nl[q & 0x0f];
    v.y = d*kvalues_iq4nl[q >> 4];
}

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q6_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q6_0 * x = (const block_q6_0 *) vx;

    const float d = x[ib].d;

    const uint8_t h = (x[ib].qh[iqs % (QK6_0 / 4)] >> (4 * (iqs / (QK6_0 / 4)))) & 0x0F;

    v.x = (x[ib].qs[iqs] & 0x0F) | ((h & 0x03) << 4);
    v.y = (x[ib].qs[iqs] >>   4) | ((h & 0x0C) << 2);

    v.x = (v.x - 32.0f) * d;
    v.y = (v.y - 32.0f) * d;
}

static __device__ __forceinline__ void dequantize_q6_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q6_1 * x = (const block_q6_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const uint8_t h = (x[ib].qh[iqs % (QK6_1 / 4)] >> (4 * (iqs / (QK6_1 / 4)))) & 0x0F;

    v.x = (x[ib].qs[iqs] & 0x0F) | ((h & 0x03) << 4);
    v.y = (x[ib].qs[iqs] >>   4) | ((h & 0x0C) << 2);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

// 2-bit planes: byte j holds elements j, j+8, j+16, j+24; iqs in [0, 16) yields elements iqs and iqs+16
static __device__ __forceinline__ void dequantize_q3_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q3_0 * x = (const block_q3_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const uint8_t b = x[ib].qs[iqs % (QK3_0 / 4)];
    const int     p = iqs / (QK3_0 / 4); // plane 0 or 1; element iqs+16 lives in plane p+2

    v.x = ((b >> (2*p))     & 0x03) | (((qh >> (iqs +  0)) & 1) << 2);
    v.y = ((b >> (2*p + 4)) & 0x03) | (((qh >> (iqs + 16)) & 1) << 2);

    v.x = (v.x - 4.0f) * d;
    v.y = (v.y - 4.0f) * d;
}

static __device__ __forceinline__ void dequantize_q3_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q3_1 * x = (const block_q3_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const uint8_t b = x[ib].qs[iqs % (QK3_1 / 4)];
    const int     p = iqs / (QK3_1 / 4);

    v.x = ((b >> (2*p))     & 0x03) | (((qh >> (iqs +  0)) & 1) << 2);
    v.y = ((b >> (2*p + 4)) & 0x03) | (((qh >> (iqs + 16)) & 1) << 2);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q2_0s(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q2_0s * x = (const block_q2_0s *) vx;

    const float d = x[ib].d;

    const uint8_t b = x[ib].qs[iqs % (QK2_0S / 4)];
    const int     p = iqs / (QK2_0S / 4);

    v.x = (b >> (2*p))     & 0x03;
    v.y = (b >> (2*p + 4)) & 0x03;

    v.x = (v.x - 2.0f) * d;
    v.y = (v.y - 2.0f) * d;
}

static __device__ __forceinline__ void dequantize_q2_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q2_1 * x = (const block_q2_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const uint8_t b = x[ib].qs[iqs % (QK2_1 / 4)];
    const int     p = iqs / (QK2_1 / 4);

    v.x = (b >> (2*p))     & 0x03;
    v.y = (b >> (2*p + 4)) & 0x03;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}
