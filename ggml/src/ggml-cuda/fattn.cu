#include "common.cuh"
#include "fattn.cuh"
#include "fattn-common.cuh"
#include "fattn-kvarn-dispatch.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

static ggml_type ggml_cuda_fattn_canonical_kv_type(ggml_type type) {
    return type == GGML_TYPE_F32 ? GGML_TYPE_F16 : type;
}

static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
    switch (ggml_cuda_fattn_canonical_kv_type(type)) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q6_1:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q3_1:
        case GGML_TYPE_Q3_0:
        case GGML_TYPE_Q2_1:
        case GGML_TYPE_Q2_0S:
        case GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}

static int ggml_cuda_fattn_quant_bits(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:                 return 8;
        case GGML_TYPE_Q6_1:
        case GGML_TYPE_Q6_0:                 return 6;
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:                 return 5;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:                 return 4;
        case GGML_TYPE_Q3_1:
        case GGML_TYPE_Q3_0:                 return 3;
        case GGML_TYPE_Q2_1:
        case GGML_TYPE_Q2_0S:                return 2;
        default:                              return -1;
    }
}

static int ggml_cuda_fattn_quant_variant(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q3_0:
        case GGML_TYPE_Q2_0S: return 1;
        default:               return 0;
    }
}

static bool ggml_cuda_fattn_default_quant_pair(ggml_type type_K, ggml_type type_V) {
    const int bits_K = ggml_cuda_fattn_quant_bits(type_K);
    const int bits_V = ggml_cuda_fattn_quant_bits(type_V);
    if (bits_K < 0 || bits_V < 0) {
        return false;
    }

    bool bit_pair = false;
    switch (bits_K) {
        case 8: bit_pair = bits_V == 8 || bits_V == 6 || bits_V == 5; break;
        case 6: bit_pair = bits_V == 6 || bits_V == 5 || bits_V == 4; break;
        case 5: bit_pair = bits_V == 5 || bits_V == 4 || bits_V == 3; break;
        case 4: bit_pair = bits_V == 4 || bits_V == 3 || bits_V == 2; break;
        case 3: bit_pair = bits_V == 3 || bits_V == 2;                break;
        case 2: bit_pair = bits_V == 2;                               break;
        default: break;
    }

    return bit_pair && (bits_K != bits_V ||
        ggml_cuda_fattn_quant_variant(type_K) <= ggml_cuda_fattn_quant_variant(type_V));
}

static bool ggml_cuda_fattn_pair_compiled(ggml_type type_K, ggml_type type_V) {
    type_K = ggml_cuda_fattn_canonical_kv_type(type_K);
    type_V = ggml_cuda_fattn_canonical_kv_type(type_V);

    if (!ggml_cuda_fattn_kv_type_supported(type_K) || !ggml_cuda_fattn_kv_type_supported(type_V) ||
        type_K == GGML_TYPE_IQ4_NL || type_V == GGML_TYPE_IQ4_NL) {
        return false;
    }

#if defined(GGML_CUDA_FA_ALL_QUANTS)
    return true;
#else
    if (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16 ||
        type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) {
        return type_K == type_V;
    }
    return ggml_cuda_fattn_default_quant_pair(type_K, type_V);
#endif
}

bool ggml_cuda_fa_pair_compiled(ggml_type type_K, ggml_type type_V) {
    return ggml_cuda_fattn_pair_compiled(type_K, type_V);
}

bool ggml_cuda_flash_attn_ext_tail_supported(
        ggml_type body_k, ggml_type body_v, ggml_type tail_k, ggml_type tail_v, int64_t d_k, int64_t d_v) {
    const bool body_supported = ggml_cuda_fattn_pair_compiled(body_k, body_v) ||
        (body_k == GGML_TYPE_IQ4_NL && body_v == GGML_TYPE_IQ4_NL);
    return body_supported &&
        (tail_k == GGML_TYPE_F16 || tail_k == GGML_TYPE_BF16) &&
        (tail_v == GGML_TYPE_F16 || tail_v == GGML_TYPE_BF16) &&
        ggml_cuda_fattn_pair_compiled(tail_k, tail_v) &&
        d_k > 0 && d_k <= 512 && d_v > 0 && d_v <= 512;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#include "fattn-vec-dispatch.cuh"

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// Internal hint used by the compact exact-tail pass. On pre-Ada tensor-core
// GPUs, q=1 BF16 attention otherwise converts the complete K/V source to F16
// on every token even though the compiled vector kernel consumes BF16
// directly. The hint is applied only after the tail wrapper has produced an
// aligned, contiguous pass that satisfies the ordinary vector eligibility
// contract below.
static constexpr int GGML_CUDA_FATTN_OP_PARAM_FORCE_VEC = 7;

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel =
        ggml_cuda_fattn_pair_compiled(K->type, V->type) &&
        Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 &&
        K->ne[1] % FATTN_KQ_STRIDE == 0;

    const bool force_vector_kernel =
        ggml_get_op_params_i32(KQV, GGML_CUDA_FATTN_OP_PARAM_FORCE_VEC) != 0;
    if (force_vector_kernel && turing_mma_available(cc) &&
            can_use_vector_kernel && Q->ne[1] == 1 && Q->ne[3] == 1) {
        return BEST_FATTN_KERNEL_VEC;
    }

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    if (ggml_cuda_flash_attn_ext_kvarn_uses_views(dst)) {
        // Descriptor-native KVarN does not need materialized F16 K/V buffers,
        // but the upstream MMA kernels still use the fixup workspace placed
        // after the output tensor when attention spans multiple KV batches.
        // Allocation planning can run before backend route selection, so it
        // must be structural and must not assert execution eligibility.
        const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
            ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, false, false);
        return f16_extra.end - (uintptr_t) dst->data;
    }

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC:
            need_f16_K = K->type == GGML_TYPE_F32;
            need_f16_V = V->type == GGML_TYPE_F32;
            break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

static void ggml_cuda_flash_attn_ext_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

#include "fattn-tail.cuh"

static bool ggml_cuda_flash_attn_ext_tail_pass_supported(int device, const ggml_tensor * dst) {
    const ggml_tensor * qo = dst->src[8];
    const ggml_tensor * rd = dst->src[9];
    const ggml_tensor * kt = dst->src[5];
    const ggml_tensor * vt = dst->src[6];
    const ggml_tensor * kt_current = dst->src[10];
    const ggml_tensor * vt_current = dst->src[11];
    const int64_t tail_stride = dst->src[7]->ne[0];
    const int64_t compute_stride = ggml_cuda_tail_compute_stride(tail_stride);
    const int64_t body_map_offset = 6 + tail_stride;
    if (!qo || !rd || qo->ne[0] <= 0 || qo->ne[1] <= 0 ||
            rd->ne[0] < body_map_offset || rd->ne[1] != qo->ne[1]) {
        return false;
    }
    if ((kt_current == nullptr) != (vt_current == nullptr)) {
        return false;
    }
    if (kt_current != nullptr &&
            (kt_current->type != kt->type || vt_current->type != vt->type ||
             kt_current->ne[0] != kt->ne[0] || vt_current->ne[0] != vt->ne[0] ||
             kt_current->ne[1] != vt_current->ne[1] ||
             kt_current->ne[2] != kt->ne[2] || vt_current->ne[2] != vt->ne[2] ||
             kt_current->ne[3] != 1 || vt_current->ne[3] != 1 ||
             kt->ne[1] + kt_current->ne[1] < tail_stride)) {
        return false;
    }
    ggml_tensor q = *dst->src[0];
    ggml_cuda_tail_make_contiguous(q, q.ne[0], qo->ne[0], q.ne[2], qo->ne[1], sizeof(float));
    ggml_tensor k = *kt;
    ggml_cuda_tail_make_contiguous(k, k.ne[0], compute_stride, k.ne[2], qo->ne[1], ggml_type_size(k.type));
    ggml_tensor v = *vt;
    ggml_cuda_tail_make_contiguous(v, v.ne[0], compute_stride, v.ne[2], qo->ne[1], ggml_type_size(v.type));
    ggml_tensor mask = *dst->src[7];
    ggml_cuda_tail_make_contiguous(mask, compute_stride, qo->ne[0], 1, qo->ne[1], sizeof(half));
    ggml_tensor pass = *dst;
    pass.src[0] = &q;
    pass.src[1] = &k;
    pass.src[2] = &v;
    pass.src[3] = &mask;
    pass.src[4] = nullptr;
    for (int i = 5; i < GGML_MAX_SRC; ++i) {
        pass.src[i] = nullptr;
    }
    ggml_cuda_tail_make_contiguous(pass, pass.ne[0], pass.ne[1], qo->ne[0], qo->ne[1], sizeof(float));
    // Compact decode tails up to two native KVarN groups use the direct
    // indexed-small kernel
    // below, so they do not need to satisfy the padded upstream FA geometry.
    // This matters for D512, whose generic FA route requires a 256-token KV
    // stride even though the direct exact-tail kernel supports 128 tokens.
    if (tail_stride > 256 &&
            ggml_cuda_get_best_fattn_kernel(device, &pass) == BEST_FATTN_KERNEL_NONE) {
        return false;
    }
    if (ggml_cuda_flash_attn_ext_kvarn_uses_views(dst)) {
        return rd->ne[0] == body_map_offset &&
            ggml_cuda_flash_attn_ext_kvarn_supported(device, dst);
    }
    if (rd->ne[0] > body_map_offset) {
        const int64_t body_stride = rd->ne[0] - body_map_offset;
        ggml_tensor kb = *dst->src[1];
        ggml_cuda_tail_make_contiguous_type(kb, kb.ne[0], body_stride, kb.ne[2], qo->ne[1]);
        ggml_tensor vb = *dst->src[2];
        ggml_cuda_tail_make_contiguous_type(vb, vb.ne[0], body_stride, vb.ne[2], qo->ne[1]);
        ggml_tensor mb = *dst->src[3];
        ggml_cuda_tail_make_contiguous(mb, body_stride, qo->ne[0], 1, qo->ne[1], sizeof(half));
        ggml_tensor body = pass;
        body.src[1] = &kb;
        body.src[2] = &vb;
        body.src[3] = &mb;
        body.src[4] = dst->src[4];
        return ggml_cuda_get_best_fattn_kernel(device, &body) != BEST_FATTN_KERNEL_NONE;
    }
    return true;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    const bool has_exact_tail = dst->src[5] != nullptr && dst->src[6] != nullptr && dst->src[7] != nullptr &&
        dst->src[8] != nullptr && dst->src[9] != nullptr;
    const bool uses_kvarn = ggml_cuda_flash_attn_ext_kvarn_uses_views(dst);
    const bool portable_kvarn_tail = uses_kvarn && has_exact_tail &&
        ggml_cuda_flash_attn_ext_kvarn_direct_tail_supported(ctx.device, dst);
    if (portable_kvarn_tail) {
        if (!ggml_cuda_flash_attn_ext_kvarn(ctx, dst)) {
            GGML_ABORT("unsupported portable KVarN exact-tail FlashAttention route");
        }
        return;
    }
    if (has_exact_tail) {
        ggml_cuda_flash_attn_ext_tail(ctx, dst);
        return;
    }

    if (uses_kvarn) {
        if (!ggml_cuda_flash_attn_ext_kvarn(ctx, dst)) {
            GGML_ABORT("unsupported KVarN CUDA FlashAttention route");
        }
        return;
    }

    ggml_cuda_flash_attn_ext_dispatch(ctx, dst);
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    const bool has_exact_tail = dst->src[5] != nullptr && dst->src[6] != nullptr && dst->src[7] != nullptr &&
        dst->src[8] != nullptr && dst->src[9] != nullptr;
    const bool uses_kvarn = ggml_cuda_flash_attn_ext_kvarn_uses_views(dst);
    const bool portable_kvarn_tail = uses_kvarn && has_exact_tail &&
        ggml_cuda_flash_attn_ext_kvarn_direct_tail_supported(device, dst);
    if (portable_kvarn_tail) {
        return ggml_cuda_flash_attn_ext_kvarn_supported(device, dst);
    }
    if (has_exact_tail) {
        if (!ggml_cuda_flash_attn_ext_tail_supported(
                    dst->src[1]->type, dst->src[2]->type, dst->src[5]->type, dst->src[6]->type,
                    dst->src[0]->ne[0], dst->ne[0]) ||
                !ggml_cuda_flash_attn_ext_tail_pass_supported(device, dst)) {
            return false;
        }
        return true;
    }

    if (uses_kvarn) {
        return ggml_cuda_flash_attn_ext_kvarn_supported(device, dst);
    }
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
