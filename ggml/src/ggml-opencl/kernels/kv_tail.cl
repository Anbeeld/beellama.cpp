#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Backend-native exact-tail operators for the standard KV-cache formats.
// `layout == 0` is the ordinary ggml block layout; `layout == 1` is the
// OpenCL backend's split q/qh/d/m layout used for loaded legacy quants.

enum kv_type {
    KV_F32 = 0,
    KV_F16 = 1,
    KV_BF16 = 2, // BF16 tensors are represented as F16 in OpenCL device memory.
    KV_Q8_0 = 3,
    KV_Q4_0 = 4,
    KV_Q4_1 = 5,
    KV_IQ4_NL = 6,
    KV_Q5_0 = 7,
    KV_Q5_1 = 8,
    KV_Q6_0 = 9,
    KV_Q6_1 = 10,
    KV_Q3_0 = 11,
    KV_Q3_1 = 12,
    KV_Q2_0S = 13,
    KV_Q2_1 = 14,
};

constant char kv_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

inline float kv_load_half(global const uchar * p, ulong off) {
    return (float) vload_half(0, (global const half *) (p + off));
}

inline void kv_store_half(global uchar * p, ulong off, float v) {
    vstore_half_rte(v, 0, (global half *) (p + off));
}

inline uint kv_load_u32(global const uchar * p, ulong off) {
    return (uint) p[off] | ((uint) p[off + 1] << 8) |
           ((uint) p[off + 2] << 16) | ((uint) p[off + 3] << 24);
}

inline void kv_store_u32(global uchar * p, ulong off, uint v) {
    p[off] = (uchar) v;
    p[off + 1] = (uchar) (v >> 8);
    p[off + 2] = (uchar) (v >> 16);
    p[off + 3] = (uchar) (v >> 24);
}

inline int kv_block_size(int type) {
    switch (type) {
        case KV_Q8_0:   return 34;
        case KV_Q4_0:   return 18;
        case KV_Q4_1:   return 20;
        case KV_IQ4_NL: return 18;
        case KV_Q5_0:   return 22;
        case KV_Q5_1:   return 24;
        case KV_Q6_0:   return 26;
        case KV_Q6_1:   return 28;
        case KV_Q3_0:   return 14;
        case KV_Q3_1:   return 16;
        case KV_Q2_0S:  return 10;
        case KV_Q2_1:   return 12;
        default:        return 0;
    }
}

inline float kv_dequant(
        global const uchar * raw,
        global const uchar * q,
        global const uchar * qh,
        global const uchar * d,
        global const uchar * m,
        ulong raw_off,
        ulong block,
        int element,
        int type,
        int layout) {
    const int bsz = kv_block_size(type);
    global const uchar * b = raw + raw_off + block*(ulong) bsz;
    float scale;
    float minimum = 0.0f;
    uint quant;

    if (layout == 1) {
        scale = kv_load_half(d, 2*block);
        if (type == KV_Q4_1 || type == KV_Q5_1) {
            minimum = kv_load_half(m, 2*block);
        }
    } else {
        scale = kv_load_half(b, 0);
        if (type == KV_Q4_1 || type == KV_Q5_1 || type == KV_Q6_1 ||
                type == KV_Q3_1 || type == KV_Q2_1) {
            minimum = kv_load_half(b, 2);
        }
    }

    switch (type) {
        case KV_Q8_0:
            quant = (uint) (uchar) (layout ? q[block*32 + element] : b[2 + element]);
            return scale*(float) ((int) quant - (quant >= 128 ? 256 : 0));
        case KV_Q4_0:
        case KV_Q4_1:
        case KV_IQ4_NL: {
            const ulong qo = block*16 + (ulong) (element & 15);
            const uchar packed = layout ? q[qo] : b[(type == KV_Q4_1 ? 4 : 2) + (element & 15)];
            quant = element < 16 ? packed & 15 : packed >> 4;
            if (type == KV_IQ4_NL) {
                return scale*(float) kv_iq4nl[quant];
            }
            return scale*(float) ((int) quant - (type == KV_Q4_0 ? 8 : 0)) + minimum;
        }
        case KV_Q5_0:
        case KV_Q5_1: {
            const uchar packed = layout ? q[block*16 + (element & 15)] :
                b[(type == KV_Q5_0 ? 6 : 8) + (element & 15)];
            const uint high = layout ?
                ((qh[block*4 + (element >> 3)] >> (element & 7)) & 1) :
                ((kv_load_u32(b, type == KV_Q5_0 ? 2 : 4) >> element) & 1);
            quant = (element < 16 ? packed & 15 : packed >> 4) | (high << 4);
            return scale*(float) ((int) quant - (type == KV_Q5_0 ? 16 : 0)) + minimum;
        }
        case KV_Q6_0:
        case KV_Q6_1: {
            const int j = element & 15;
            const int base = type == KV_Q6_0 ? 2 : 4;
            const uchar packed = b[base + 8 + j];
            const uchar high = b[base + j%8] >> (4*(j/8));
            quant = element < 16 ? (packed & 15) | ((high & 3) << 4) :
                                   (packed >> 4) | ((high & 12) << 2);
            return scale*(float) ((int) quant - (type == KV_Q6_0 ? 32 : 0)) + minimum;
        }
        case KV_Q3_0:
        case KV_Q3_1: {
            const int base = type == KV_Q3_0 ? 2 : 4;
            const uint high = kv_load_u32(b, base);
            quant = ((b[base + 4 + element%8] >> (2*(element/8))) & 3) |
                    (((high >> element) & 1) << 2);
            return scale*(float) ((int) quant - (type == KV_Q3_0 ? 4 : 0)) + minimum;
        }
        case KV_Q2_0S:
        case KV_Q2_1: {
            const int base = type == KV_Q2_0S ? 2 : 4;
            quant = (b[base + element%8] >> (2*(element/8))) & 3;
            return scale*(float) ((int) quant - (type == KV_Q2_0S ? 2 : 0)) + minimum;
        }
        default:
            return 0.0f;
    }
}

inline uint kv_nearest_iq4(float x) {
    uint best = 0;
    float error = fabs(x - (float) kv_iq4nl[0]);
    for (uint i = 1; i < 16; ++i) {
        const float next = fabs(x - (float) kv_iq4nl[i]);
        if (next < error) {
            error = next;
            best = i;
        }
    }
    return best;
}

inline void kv_quantize_block(
        global const float * x,
        global uchar * raw,
        global uchar * q,
        global uchar * qh,
        global uchar * d,
        global uchar * m,
        ulong raw_off,
        ulong block,
        int type,
        int layout) {
    global uchar * b = raw + raw_off + block*(ulong) kv_block_size(type);
    float vmin = x[0];
    float vmax = x[0];
    float amax = fabs(x[0]);
    float signed_max = x[0];
    for (int j = 1; j < 32; ++j) {
        vmin = fmin(vmin, x[j]);
        vmax = fmax(vmax, x[j]);
        if (fabs(x[j]) > amax) {
            amax = fabs(x[j]);
            signed_max = x[j];
        }
    }

    const int affine = type == KV_Q4_1 || type == KV_Q5_1 || type == KV_Q6_1 ||
                       type == KV_Q3_1 || type == KV_Q2_1;
    int levels = 0;
    int midpoint = 0;
    if (type == KV_Q2_0S || type == KV_Q2_1) { levels = 4; midpoint = 2; }
    if (type == KV_Q3_0  || type == KV_Q3_1) { levels = 8; midpoint = 4; }
    if (type == KV_Q4_0  || type == KV_Q4_1) { levels = 16; midpoint = 8; }
    if (type == KV_Q5_0  || type == KV_Q5_1) { levels = 32; midpoint = 16; }
    if (type == KV_Q6_0  || type == KV_Q6_1) { levels = 64; midpoint = 32; }

    float scale;
    float minimum = 0.0f;
    if (type == KV_Q8_0) {
        scale = amax/127.0f;
    } else if (type == KV_IQ4_NL) {
        scale = signed_max/(float) kv_iq4nl[0];
    } else if (affine) {
        minimum = vmin;
        scale = (vmax - vmin)/(float) (levels - 1);
    } else {
        scale = signed_max/(float) -midpoint;
    }
    const float inv = scale != 0.0f ? 1.0f/scale : 0.0f;

    if (type == KV_Q8_0) {
        if (layout) kv_store_half(d, 2*block, scale); else kv_store_half(b, 0, scale);
        for (int j = 0; j < 32; ++j) {
            const char value = (char) clamp((int) rint(x[j]*inv), -127, 127);
            if (layout) q[block*32 + j] = (uchar) value; else b[2 + j] = (uchar) value;
        }
        return;
    }

    if (type == KV_IQ4_NL) {
        float sumqx = 0.0f;
        float sumq2 = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const uint lo = kv_nearest_iq4(x[j]*inv);
            const uint hi = kv_nearest_iq4(x[j + 16]*inv);
            const float v0 = (float) kv_iq4nl[lo];
            const float v1 = (float) kv_iq4nl[hi];
            const float w0 = x[j]*x[j];
            const float w1 = x[j + 16]*x[j + 16];
            sumqx += w0*v0*x[j] + w1*v1*x[j + 16];
            sumq2 += w0*v0*v0 + w1*v1*v1;
            if (layout) q[block*16 + j] = (uchar) (lo | (hi << 4));
            else b[2 + j] = (uchar) (lo | (hi << 4));
        }
        if (sumq2 > 0.0f) scale = sumqx/sumq2;
        if (layout) kv_store_half(d, 2*block, scale); else kv_store_half(b, 0, scale);
        return;
    }

    uint values[32];
    for (int j = 0; j < 32; ++j) {
        const float shifted = affine ? (x[j] - minimum)*inv : x[j]*inv + (float) midpoint;
        values[j] = (uint) clamp((int) floor(shifted + 0.5f), 0, levels - 1);
    }

    if (type == KV_Q6_0) {
        float sumqx = 0.0f;
        float sumq2 = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const float v = (float) ((int) values[j] - 32);
            const float w = x[j]*x[j];
            sumqx += w*v*x[j];
            sumq2 += w*v*v;
        }
        if (sumq2 > 0.0f) scale = sumqx/sumq2;
    }

    if (layout) {
        kv_store_half(d, 2*block, scale);
        if (type == KV_Q4_1 || type == KV_Q5_1) kv_store_half(m, 2*block, minimum);
    } else {
        kv_store_half(b, 0, scale);
        if (affine) kv_store_half(b, 2, minimum);
    }

    if (type == KV_Q4_0 || type == KV_Q4_1 || type == KV_Q5_0 || type == KV_Q5_1 || type == KV_Q6_0 || type == KV_Q6_1) {
        const int qoff = type == KV_Q4_0 ? 2 : type == KV_Q4_1 ? 4 :
                         type == KV_Q5_0 ? 6 : type == KV_Q5_1 ? 8 :
                         type == KV_Q6_0 ? 10 : 12;
        uint high5 = 0;
        uchar high6[8] = { 0,0,0,0,0,0,0,0 };
        for (int j = 0; j < 16; ++j) {
            const uint lo = values[j];
            const uint hi = values[j + 16];
            const uchar packed = (uchar) ((lo & 15) | ((hi & 15) << 4));
            if (layout) q[block*16 + j] = packed; else b[qoff + j] = packed;
            if (type == KV_Q5_0 || type == KV_Q5_1) {
                high5 |= ((lo >> 4) & 1) << j;
                high5 |= ((hi >> 4) & 1) << (j + 16);
            } else if (type == KV_Q6_0 || type == KV_Q6_1) {
                high6[j%8] |= (uchar) (((lo >> 4) | ((hi >> 4) << 2)) << (4*(j/8)));
            }
        }
        if (type == KV_Q5_0 || type == KV_Q5_1) {
            if (layout) kv_store_u32(qh, 4*block, high5);
            else kv_store_u32(b, type == KV_Q5_0 ? 2 : 4, high5);
        } else if (type == KV_Q6_0 || type == KV_Q6_1) {
            const int hoff = type == KV_Q6_0 ? 2 : 4;
            for (int j = 0; j < 8; ++j) b[hoff + j] = high6[j];
        }
        return;
    }

    if (type == KV_Q3_0 || type == KV_Q3_1) {
        const int base = type == KV_Q3_0 ? 2 : 4;
        uint high = 0;
        for (int j = 0; j < 8; ++j) b[base + 4 + j] = 0;
        for (int j = 0; j < 32; ++j) {
            b[base + 4 + j%8] |= (uchar) ((values[j] & 3) << (2*(j/8)));
            high |= ((values[j] >> 2) & 1) << j;
        }
        kv_store_u32(b, base, high);
        return;
    }

    const int base = type == KV_Q2_0S ? 2 : 4;
    for (int j = 0; j < 8; ++j) b[base + j] = 0;
    for (int j = 0; j < 32; ++j) {
        b[base + j%8] |= (uchar) (values[j] << (2*(j/8)));
    }
}

kernel void kernel_kv_get_rows(
        global const uchar * raw, global const uchar * q, global const uchar * qh,
        global const uchar * d, global const uchar * m, ulong raw_off,
        global const int * indices, ulong indices_off, global uchar * dst, ulong dst_off,
        int ne00, ulong nb01, ulong nb02, ulong nb03,
        int ne01, int ne02, int ne10, ulong nb10, ulong nb11, ulong nb12,
        ulong nb1, ulong nb2, ulong nb3, int src_type, int dst_type, int layout) {
    const int i10 = get_group_id(0);
    const int i11 = get_group_id(1);
    const int i12 = get_group_id(2);
    const int r = *(global const int *) ((global const uchar *) indices + indices_off +
                                         i10*nb10 + i11*nb11 + i12*nb12);
    global uchar * dst_row = dst + dst_off + i10*nb1 + i11*nb2 + i12*nb3;
    global const uchar * src_row = raw + raw_off + r*nb01 + i11*nb02 + i12*nb03;
    const ulong soa_row = ((ulong) i12*ne02 + i11)*ne01 + r;
    const int nblk = ne00/32;
    for (int e = get_local_id(0); e < ne00; e += get_local_size(0)) {
        float value;
        if (src_type == KV_F32) value = *(global const float *) (src_row + 4*(ulong)e);
        else if (src_type == KV_F16 || src_type == KV_BF16) value = kv_load_half(src_row, 2*(ulong)e);
        else if (layout) value = kv_dequant(raw, q, qh, d, m, raw_off, soa_row*nblk + e/32, e%32, src_type, layout);
        else value = kv_dequant(raw, q, qh, d, m,
                                raw_off + r*nb01 + i11*nb02 + i12*nb03,
                                e/32, e%32, src_type, layout);
        if (dst_type == KV_F32) *(global float *) (dst_row + 4*(ulong)e) = value;
        else kv_store_half(dst_row, 2*(ulong)e, value);
    }
}

kernel void kernel_kv_set_rows(
        global const float * src, ulong src_off, global const uchar * indices, ulong indices_off,
        global uchar * raw, global uchar * q, global uchar * qh, global uchar * d, global uchar * m,
        ulong raw_off, int ne01, ulong nb01, ulong nb02, ulong nb03,
        int ne11, int ne12, ulong nb10, ulong nb11, ulong nb12,
        int dne0, int nblk, int dne1, int dne2, ulong dnb1, ulong dnb2, ulong dnb3,
        int index_i64, int dst_type, int layout) {
    const int blk = get_global_id(0);
    const int i01 = get_global_id(1);
    const int i23 = get_global_id(2);
    if (blk >= nblk || i01 >= ne01) return;
    const int i02 = i23 % dne2;
    const int i03 = i23 / dne2;
    const int i11 = i02 % ne11;
    const int i12 = i03 % ne12;
    global const uchar * ip = indices + indices_off + i01*nb10 + i11*nb11 + i12*nb12;
    const long row = index_i64 ? *(global const long *) ip : (long) *(global const int *) ip;
    global const float * x = (global const float *) ((global const uchar *) src + src_off +
                                                     i01*nb01 + i02*nb02 + i03*nb03) + 32*blk;
    const ulong soa_block = (((ulong) i03*dne2 + i02)*dne1 + (ulong) row)*nblk + blk;
    if (dst_type >= KV_Q8_0) {
        if (layout) kv_quantize_block(x, raw, q, qh, d, m, raw_off, soa_block, dst_type, layout);
        else kv_quantize_block(x, raw, q, qh, d, m,
                               raw_off + row*dnb1 + i02*dnb2 + i03*dnb3,
                               blk, dst_type, layout);
    } else {
        global uchar * dst_row = raw + raw_off + row*dnb1 + i02*dnb2 + i03*dnb3;
        for (int j = 0; j < 32 && 32*blk + j < dne0; ++j) {
            if (dst_type == KV_F32) *(global float *) (dst_row + 4*(ulong)(32*blk + j)) = x[j];
            else kv_store_half(dst_row, 2*(ulong)(32*blk + j), x[j]);
        }
    }
}

kernel void kernel_kv_out_prod(
        global const uchar * raw, global const uchar * q, global const uchar * qh,
        global const uchar * d, global const uchar * m, ulong raw_off,
        global const float * src1, ulong src1_off, global float * dst, ulong dst_off,
        int ne00, int ne01, int ne02, int ne03, ulong nb01, ulong nb02, ulong nb03,
        int ne10, ulong nb10, ulong nb11, ulong nb12, ulong nb13,
        int ne22, int ne23, ulong nb20, ulong nb21, ulong nb22, ulong nb23,
        int src_type, int layout) {
    const int i0 = get_global_id(0);
    const int i1 = get_global_id(1);
    const int i23 = get_global_id(2);
    if (i0 >= ne00 || i1 >= ne10 || i23 >= ne22*ne23) return;
    const int i2 = i23 % ne22;
    const int i3 = i23 / ne22;
    const int dps2 = ne22/ne02;
    const int dps3 = ne23/ne03;
    const int a2 = i2/dps2;
    const int a3 = i3/dps3;
    const int nblk = ne00/32;
    float sum = 0.0f;
    for (int k = 0; k < ne01; ++k) {
        const ulong soa_block = (((ulong) a3*ne02 + a2)*ne01 + k)*nblk + i0/32;
        const float av = layout ?
            kv_dequant(raw, q, qh, d, m, raw_off, soa_block, i0%32, src_type, layout) :
            kv_dequant(raw, q, qh, d, m, raw_off + k*nb01 + a2*nb02 + a3*nb03,
                       i0/32, i0%32, src_type, layout);
        const float bv = *(global const float *) ((global const uchar *) src1 + src1_off +
                                                  i1*nb10 + k*nb11 + i2*nb12 + i3*nb13);
        sum = fma(av, bv, sum);
    }
    *(global float *) ((global uchar *) dst + dst_off + i0*nb20 + i1*nb21 + i2*nb22 + i3*nb23) = sum;
}

kernel void kernel_kv_dequant_copy(
        global const uchar * raw, global const uchar * q, global const uchar * qh,
        global const uchar * d, global const uchar * m, ulong raw_off,
        global uchar * dst, ulong dst_off, ulong ne, int src_type, int dst_type, int layout) {
    const ulong e = get_global_id(0);
    if (e >= ne) return;
    const float value = kv_dequant(raw, q, qh, d, m, raw_off, e/32, e%32, src_type, layout);
    if (dst_type == KV_F32) *(global float *) (dst + dst_off + 4*e) = value;
    else kv_store_half(dst, dst_off + 2*e, value);
}
