#pragma once

#include "common.h"

void quantize_q1_0(device const float * src, device block_q1_0 & dst) {
    float sum_abs = 0.0f;
    for (int j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    for (int j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    for (int j = 0; j < QK1_0; j++) {
        if (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

void quantize_q6_0(device const float * src, device block_q6_0 & dst) {
    float amax = 0.0f;
    float vmax = 0.0f;
    for (int j = 0; j < QK6_0; ++j) {
        if (fabs(src[j]) > amax) {
            amax = fabs(src[j]);
            vmax = src[j];
        }
    }
    float d = vmax / -32.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    float sumqx = 0.0f;
    float sumq2 = 0.0f;
    for (int j = 0; j < QK6_0 / 4; ++j) dst.qh[j] = 0;
    for (int j = 0; j < QK6_0 / 2; ++j) {
        const uint8_t q0 = (uint8_t) clamp((int) floor(src[j] * id + 32.5f), 0, 63);
        const uint8_t q1 = (uint8_t) clamp((int) floor(src[j + 16] * id + 32.5f), 0, 63);
        dst.qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        dst.qh[j % 8] |= ((q0 >> 4) | ((q1 >> 4) << 2)) << (4 * (j / 8));

        const float v0 = (float) q0 - 32.0f;
        const float v1 = (float) q1 - 32.0f;
        const float w0 = src[j] * src[j];
        const float w1 = src[j + 16] * src[j + 16];
        sumqx += w0 * v0 * src[j] + w1 * v1 * src[j + 16];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }
    if (sumq2 > 0.0f) d = sumqx / sumq2;
    dst.d = d;
}

void quantize_q6_1(device const float * src, device block_q6_1 & dst) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;
    for (int j = 0; j < QK6_1; ++j) {
        vmin = min(vmin, src[j]);
        vmax = max(vmax, src[j]);
    }
    const float d = (vmax - vmin) / 63.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    dst.d = d;
    dst.m = vmin;
    for (int j = 0; j < QK6_1 / 4; ++j) dst.qh[j] = 0;
    for (int j = 0; j < QK6_1 / 2; ++j) {
        const uint8_t q0 = (uint8_t) clamp((int) floor((src[j] - vmin) * id + 0.5f), 0, 63);
        const uint8_t q1 = (uint8_t) clamp((int) floor((src[j + 16] - vmin) * id + 0.5f), 0, 63);
        dst.qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        dst.qh[j % 8] |= ((q0 >> 4) | ((q1 >> 4) << 2)) << (4 * (j / 8));
    }
}

template <int levels, bool affine>
void quantize_planar_values(device const float * src, thread float & d, thread float & m,
                            thread uint8_t (&qs)[8], thread uint8_t (&qh)[4]) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;
    float amax = 0.0f;
    float signed_max = 0.0f;
    for (int j = 0; j < 32; ++j) {
        vmin = min(vmin, src[j]);
        vmax = max(vmax, src[j]);
        if (fabs(src[j]) > amax) {
            amax = fabs(src[j]);
            signed_max = src[j];
        }
    }
    m = affine ? vmin : 0.0f;
    d = affine ? (vmax - vmin) / (levels - 1) : signed_max / -(levels / 2);
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    for (int j = 0; j < 8; ++j) qs[j] = 0;
    for (int j = 0; j < 4; ++j) qh[j] = 0;
    for (int j = 0; j < 32; ++j) {
        const float shifted = affine ? (src[j] - m) * id + 0.5f : src[j] * id + levels / 2 + 0.5f;
        const uint8_t q = (uint8_t) clamp((int) floor(shifted), 0, levels - 1);
        qs[j % 8] |= (q & 3) << (2 * (j / 8));
        if (levels == 8) qh[j / 8] |= ((q >> 2) & 1) << (j % 8);
    }
}

void quantize_q3_0(device const float * src, device block_q3_0 & dst) {
    float d, m; uint8_t qs[8], qh[4];
    quantize_planar_values<8, false>(src, d, m, qs, qh);
    dst.d = d;
    for (int j = 0; j < 8; ++j) dst.qs[j] = qs[j];
    for (int j = 0; j < 4; ++j) dst.qh[j] = qh[j];
}
void quantize_q3_1(device const float * src, device block_q3_1 & dst) {
    float d, m; uint8_t qs[8], qh[4];
    quantize_planar_values<8, true>(src, d, m, qs, qh);
    dst.d = d; dst.m = m;
    for (int j = 0; j < 8; ++j) dst.qs[j] = qs[j];
    for (int j = 0; j < 4; ++j) dst.qh[j] = qh[j];
}
void quantize_q2_0(device const float * src, device block_q2_0 & dst) {
    float amax = 0.0f;
    for (int j = 0; j < QK2_0; j++) {
        float a = fabs(src[j]);
        if (a > amax) amax = a;
    }
    const float d = amax;
    dst.d = d;

    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    for (int j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    for (int j = 0; j < QK2_0; j++) {
        int q = (int)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

void quantize_q4_0(device const float * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK4_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / -8;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = src[0       + j]*id;
        const float x1 = src[QK4_0/2 + j]*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

void quantize_q4_1(device const float * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int j = 0; j < QK4_1; j++) {
        const float v = src[j];
        if (min > v) min = v;
        if (max < v) max = v;
    }

    const float d = (max - min) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (src[0       + j] - min)*id;
        const float x1 = (src[QK4_1/2 + j] - min)*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

void quantize_q5_0(device const float * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK5_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / -16;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = src[0       + j]*id;
        const float x1 = src[QK5_0/2 + j]*id;

        const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread const uint8_t * qh8 = (thread const uint8_t *)&qh;

    for (int j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

void quantize_q5_1(device const float * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    float max = src[0];
    float min = src[0];

    for (int j = 1; j < QK5_1; j++) {
        const float v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (src[0       + j] - min)*id;
        const float x1 = (src[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread const uint8_t * qh8 = (thread const uint8_t *)&qh;

    for (int j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

void quantize_q8_0(device const float * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = src[j];
        amax = MAX(amax, fabs(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

void quantize_iq4_nl(device const float * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK4_NL; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / kvalues_iq4nl_f[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = src[0        + j]*id;
        const float x1 = src[QK4_NL/2 + j]*id;

        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        const float v0 = kvalues_iq4nl_f[xi0];
        const float v1 = kvalues_iq4nl_f[xi1];
        const float w0 = src[0        + j]*src[0        + j];
        const float w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

void quantize_tq2_0(device const float * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK_K; j++) {
        const float v = src[j];
        amax = MAX(amax, fabs(v));
    }

    const float d = amax;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    for (int j = 0; j < QK_K/4; j += 32) {
        for (int m = 0; m < 32; ++m) {
            uint8_t q = 0;
            for (int n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                int xi = (int)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
