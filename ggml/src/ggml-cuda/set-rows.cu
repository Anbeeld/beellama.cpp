#include "set-rows.cuh"
#include "cpy-utils.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    if (dst_row < 0) {
        return;
    }

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template <typename block_type, int qk, void (*quantize_func)(const float *, block_type *), typename shadow_t>
static __global__ void k_set_rows_quant_shadow(
        const float * __restrict__ src,
        const int64_t * __restrict__ body_indices,
        block_type * __restrict__ body,
        shadow_t * __restrict__ shadow,
        const int64_t * __restrict__ shadow_indices,
        int64_t row_size, int64_t n_rows, int64_t n_shadow_levels,
        int64_t src_row_stride, int64_t body_row_stride_bytes,
        int64_t shadow_row_stride, int64_t shadow_level_stride) {
    const int64_t blocks_per_row = row_size/qk;
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n_rows*blocks_per_row) {
        return;
    }
    const int64_t row = i/blocks_per_row;
    const int64_t block = i - row*blocks_per_row;
    const int64_t element = block*qk;
    const float * src_block = src + row*src_row_stride + element;
    const int64_t body_row = body_indices[row];
    block_type * body_block = reinterpret_cast<block_type *>(
            reinterpret_cast<char *>(body) + body_row*body_row_stride_bytes) + block;
    quantize_func(src_block, body_block);
    for (int64_t level = 0; level < n_shadow_levels; ++level) {
        const int64_t shadow_row = shadow_indices[row + level*shadow_level_stride];
        shadow_t * shadow_block = shadow + shadow_row*shadow_row_stride + element;
        for (int j = 0; j < qk; ++j) {
            shadow_block[j] = ggml_cuda_cast<shadow_t>(src_block[j]);
        }
    }
}

template <typename block_type, int qk, void (*quantize_func)(const float *, block_type *), typename shadow_t>
static void launch_set_rows_quant_shadow(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src,
        const ggml_tensor * body_indices, ggml_tensor * body,
        ggml_tensor * shadow, const ggml_tensor * shadow_indices) {
    GGML_ASSERT(src->ne[2] == 1 && src->ne[3] == 1 && body_indices->ne[0] == src->ne[1]);
    GGML_ASSERT(src->ne[0] % qk == 0);
    const int64_t blocks = src->ne[1]*(src->ne[0]/qk);
    const int threads = CUDA_SET_ROWS_BLOCK_SIZE;
    k_set_rows_quant_shadow<block_type, qk, quantize_func, shadow_t>
            <<<(blocks + threads - 1)/threads, threads, 0, ctx.stream()>>>(
            (const float *) src->data, (const int64_t *) body_indices->data,
            (block_type *) body->data, (shadow_t *) shadow->data,
            (const int64_t *) shadow_indices->data,
            src->ne[0], src->ne[1], shadow_indices->ne[1], src->nb[1]/sizeof(float),
            body->nb[1], shadow->nb[1]/sizeof(shadow_t),
            shadow_indices->nb[1]/sizeof(int64_t));
}

template <typename shadow_t>
static void set_rows_quant_shadow_cuda(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src,
        const ggml_tensor * body_indices, ggml_tensor * body,
        ggml_tensor * shadow, const ggml_tensor * shadow_indices) {
#define LAUNCH_CASE(type, block_type, qk, quantize_func) \
    case type: \
        launch_set_rows_quant_shadow<block_type, qk, quantize_func, shadow_t>( \
                ctx, src, body_indices, body, shadow, shadow_indices); \
        break
    switch (body->type) {
        LAUNCH_CASE(GGML_TYPE_Q8_0,  block_q8_0,   QK8_0,  quantize_f32_q8_0_block);
        LAUNCH_CASE(GGML_TYPE_Q6_0,  block_q6_0,   QK6_0,  quantize_f32_q6_0_block);
        LAUNCH_CASE(GGML_TYPE_Q6_1,  block_q6_1,   QK6_1,  quantize_f32_q6_1_block);
        LAUNCH_CASE(GGML_TYPE_Q5_0,  block_q5_0,   QK5_0,  quantize_f32_q5_0_block);
        LAUNCH_CASE(GGML_TYPE_Q5_1,  block_q5_1,   QK5_1,  quantize_f32_q5_1_block);
        LAUNCH_CASE(GGML_TYPE_Q4_0,  block_q4_0,   QK4_0,  quantize_f32_q4_0_block);
        LAUNCH_CASE(GGML_TYPE_Q4_1,  block_q4_1,   QK4_1,  quantize_f32_q4_1_block);
        LAUNCH_CASE(GGML_TYPE_IQ4_NL, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block);
        LAUNCH_CASE(GGML_TYPE_Q3_0,  block_q3_0,   QK3_0,  quantize_f32_q3_0_block);
        LAUNCH_CASE(GGML_TYPE_Q3_1,  block_q3_1,   QK3_1,  quantize_f32_q3_1_block);
        LAUNCH_CASE(GGML_TYPE_Q2_0S, block_q2_0s, QK2_0S, quantize_f32_q2_0s_block);
        LAUNCH_CASE(GGML_TYPE_Q2_1,  block_q2_1,   QK2_1,  quantize_f32_q2_1_block);
        default:
            GGML_ABORT("unsupported fused shadow body type %s", ggml_type_name(body->type));
    }
#undef LAUNCH_CASE
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows_quant<idx_t, block_type, qk, quantize_func>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * src0_ptr,
                                  const idx_t * src1_ptr,
                                  dst_t * dst_ptr,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const src_t * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const idx_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    ggml_cuda_pdl_lc();
    if (dst_row < 0) {
        return;
    }

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows<src_t, idx_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
            s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
            ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q6_0) {
        set_rows_cuda_quant<idx_t, block_q6_0, QK6_0, quantize_f32_q6_0_block>(
            src0_d, src1_d, (block_q6_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q6_1) {
        set_rows_cuda_quant<idx_t, block_q6_1, QK6_1, quantize_f32_q6_1_block>(
            src0_d, src1_d, (block_q6_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q3_0) {
        set_rows_cuda_quant<idx_t, block_q3_0, QK3_0, quantize_f32_q3_0_block>(
            src0_d, src1_d, (block_q3_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q3_1) {
        set_rows_cuda_quant<idx_t, block_q3_1, QK3_1, quantize_f32_q3_1_block>(
            src0_d, src1_d, (block_q3_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q2_0S) {
        set_rows_cuda_quant<idx_t, block_q2_0s, QK2_0S, quantize_f32_q2_0s_block>(
            src0_d, src1_d, (block_q2_0s*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q2_1) {
        set_rows_cuda_quant<idx_t, block_q2_1, QK2_1, quantize_f32_q2_1_block>(
            src0_d, src1_d, (block_q2_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int32_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int32_t * src1_d = (const int32_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int64_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (dst->src[3] != nullptr) {
        GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I64 &&
                dst->src[4]->type == GGML_TYPE_I64);
        if (dst->src[3]->type == GGML_TYPE_F16) {
            set_rows_quant_shadow_cuda<half>(ctx, src0, src1, dst->src[2], dst->src[3], dst->src[4]);
        } else if (dst->src[3]->type == GGML_TYPE_BF16) {
            set_rows_quant_shadow_cuda<nv_bfloat16>(ctx, src0, src1, dst->src[2], dst->src[3], dst->src[4]);
        } else {
            GGML_ABORT("unsupported fused shadow type %s", ggml_type_name(dst->src[3]->type));
        }
        return;
    }

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16));
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src0->type == GGML_TYPE_F32) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
        }
    } else if (src0->type == GGML_TYPE_F16) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<half, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<half, int32_t>(ctx, src0, src1, dst);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
    }
}
