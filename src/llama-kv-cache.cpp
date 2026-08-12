#include "llama-kv-cache.h"
#include "llama-kv-cache-update.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <vector>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

namespace {

using backend_kv_tail_attention_supported_t = bool (*)(
        ggml_type, ggml_type, ggml_type, ggml_type, int64_t, int64_t);

struct kv_tail_backend_probe_spec {
    uint32_t layer_id;
    ggml_backend_buffer_type_t buft;
    ggml_type body_type_k;
    ggml_type body_type_v;
    int64_t head_dim_k;
    int64_t head_dim_v;
    bool has_v;
    bool explicit_bias;
};

static bool backend_supports_native_kv_tail(
        ggml_backend_buffer_type_t buft,
        ggml_type body_k, ggml_type body_v,
        ggml_type tail_k, ggml_type tail_v,
        int64_t d_k, int64_t d_v,
        bool segmented) {
    auto dev = buft ? ggml_backend_buft_get_device(buft) : nullptr;
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    const auto supports = [&](ggml_backend_dev_t candidate) {
        const auto reg = candidate ? ggml_backend_dev_backend_reg(candidate) : nullptr;
        const auto fn = reg ? reinterpret_cast<backend_kv_tail_attention_supported_t>(
                ggml_backend_reg_get_proc_address(reg, segmented ?
                        "ggml_backend_kv_tail_segmented_attention_supported" :
                        "ggml_backend_kv_tail_attention_supported")) : nullptr;
        return fn && fn(body_k, body_v, tail_k, tail_v, d_k, d_v);
    };
    if (!ggml_backend_dev_is_meta(dev)) {
        return supports(dev);
    }
    const size_t count = ggml_backend_meta_device_count(dev);
    for (size_t i = 0; i < count; ++i) {
        if (!supports(ggml_backend_meta_device_get(dev, i))) {
            return false;
        }
    }
    return count > 0;
}

static llama_kv_tail_route_capability probe_standard_kv_tail_route(
        const kv_tail_backend_probe_spec & spec,
        ggml_type exact_k,
        ggml_type exact_v,
        bool v_transposed,
        bool flash_attn,
        bool segmented) {
    if (!spec.has_v) {
        return { false, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_WRITE_V };
    }
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * dev = ggml_backend_buft_get_device(spec.buft);
    if (!dev) {
        dev = cpu;
    }
    if (!dev || !cpu) {
        return { false, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_WRITE_K };
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 2*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error("failed to create standard KV tail capability context");
    }
    constexpr int64_t n_body = 256;
    constexpr int64_t n_tail = 16;
    auto * idx64 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 1);
    auto * tail_idx64 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 1);
    auto * idx32 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    auto * src_k = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, spec.head_dim_k, 1);
    auto * src_v = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, spec.head_dim_v, 1);
    auto * body_k = ggml_new_tensor_2d(ctx.get(), spec.body_type_k, spec.head_dim_k, n_body);
    auto * body_v = ggml_new_tensor_2d(ctx.get(), spec.body_type_v, spec.head_dim_v, n_body);
    auto * tail_k = ggml_new_tensor_2d(ctx.get(), exact_k, spec.head_dim_k, n_tail);
    auto * tail_v = ggml_new_tensor_2d(ctx.get(), exact_v, spec.head_dim_v, n_tail);
    auto * commit_dependency = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);

    const auto owner_supports = [&](ggml_tensor * op) {
        return op && ggml_backend_dev_supports_op(dev, op);
    };
    const auto compact_supports = [&](ggml_tensor * op) {
        return owner_supports(op) || (op && ggml_backend_dev_supports_op(cpu, op));
    };

    llama_kv_tail_route_requirements requirements;
    const bool fused_k = ggml_is_quantized(spec.body_type_k) &&
            (exact_k == GGML_TYPE_F16 || exact_k == GGML_TYPE_BF16);
    if (segmented) {
        requirements.write_k = owner_supports(ggml_set_rows(
                ctx.get(), body_k, src_k, idx64)) &&
            owner_supports(ggml_set_rows_ordered(
                ctx.get(), tail_k, src_k, tail_idx64, commit_dependency));
    } else if (fused_k) {
        requirements.write_k = owner_supports(ggml_set_rows_with_shadow(
                ctx.get(), body_k, src_k, idx64, tail_k, tail_idx64));
    } else {
        requirements.write_k = owner_supports(ggml_set_rows(ctx.get(), body_k, src_k, idx64)) &&
                compact_supports(ggml_set_rows(ctx.get(), tail_k, src_k, tail_idx64));
    }

    const bool fused_v = !v_transposed && ggml_is_quantized(spec.body_type_v) &&
            (exact_v == GGML_TYPE_F16 || exact_v == GGML_TYPE_BF16);
    if (segmented && !v_transposed) {
        requirements.write_v = owner_supports(ggml_set_rows(
                ctx.get(), body_v, src_v, idx64)) &&
            owner_supports(ggml_set_rows_ordered(
                ctx.get(), tail_v, src_v, tail_idx64, commit_dependency));
    } else if (fused_v) {
        requirements.write_v = owner_supports(ggml_set_rows_with_shadow(
                ctx.get(), body_v, src_v, idx64, tail_v, tail_idx64));
    } else if (v_transposed && ggml_is_quantized(spec.body_type_v)) {
        requirements.write_v = false;
    } else {
        ggml_tensor * body_write = nullptr;
        if (v_transposed) {
            auto * body_scalar = ggml_reshape_2d(ctx.get(), body_v, 1, ggml_nelements(body_v));
            auto * src_scalar = ggml_reshape_2d(ctx.get(), src_v, 1, ggml_nelements(src_v));
            auto * scalar_idxs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, spec.head_dim_v);
            body_write = ggml_set_rows(ctx.get(), body_scalar, src_scalar, scalar_idxs);
        } else {
            body_write = ggml_set_rows(ctx.get(), body_v, src_v, idx64);
        }
        requirements.write_v = owner_supports(body_write) &&
                compact_supports(ggml_set_rows(ctx.get(), tail_v, src_v, tail_idx64));
    }

    requirements.gather_k = compact_supports(ggml_get_rows_as(ctx.get(), tail_k, idx32, exact_k));
    requirements.gather_v = compact_supports(ggml_get_rows_as(ctx.get(), tail_v, idx32, exact_v));

    auto * q = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, spec.head_dim_k, 1);
    auto * body_scores = ggml_mul_mat(ctx.get(), body_k, q);
    auto * exact_scores = ggml_mul_mat(ctx.get(), tail_k, q);
    requirements.body_score = owner_supports(body_scores);
    requirements.exact_score = compact_supports(exact_scores);

    auto * body_weights = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_body, 1);
    auto * exact_weights = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_tail, 1);
    ggml_tensor * body_value = nullptr;
    if (ggml_is_quantized(spec.body_type_v)) {
        body_value = ggml_out_prod(ctx.get(), body_v, ggml_transpose(ctx.get(), body_weights));
    } else {
        auto * body_v_transposed = ggml_new_tensor_2d(
                ctx.get(), spec.body_type_v, n_body, spec.head_dim_v);
        body_value = ggml_mul_mat(ctx.get(), body_v_transposed, body_weights);
    }
    auto * exact_v_transposed = ggml_new_tensor_2d(ctx.get(), exact_v, n_tail, spec.head_dim_v);
    auto * exact_value = ggml_mul_mat(ctx.get(), exact_v_transposed, exact_weights);
    requirements.body_value = owner_supports(body_value);
    requirements.exact_value = compact_supports(exact_value);

    auto * scores = ggml_concat(ctx.get(), body_scores, exact_scores, 0);
    auto * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_body + n_tail, 1);
    if (spec.explicit_bias) {
        auto * bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_body + n_tail, 1);
        scores = ggml_add(ctx.get(), scores, bias);
    }
    auto * normalized = ggml_soft_max_ext(ctx.get(), scores, mask, 1.0f, 0.0f);
    auto * merged = ggml_add(ctx.get(), body_value, exact_value);
    requirements.generic_merge = owner_supports(scores) && owner_supports(normalized) && owner_supports(merged);
    requirements.native_attention = flash_attn && !spec.explicit_bias &&
            backend_supports_native_kv_tail(spec.buft,
                    spec.body_type_k, spec.body_type_v, exact_k, exact_v,
                    spec.head_dim_k, spec.head_dim_v, segmented);
    return llama_kv_tail_select_route(requirements);
}

static llama_kv_tail_route_capability probe_standard_native_exact_route(
        const kv_tail_backend_probe_spec & spec,
        ggml_type actual_k,
        ggml_type actual_v,
        bool v_transposed,
        bool flash_attn) {
    if (!spec.has_v) {
        return { false, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_WRITE_V };
    }
    auto * dev = ggml_backend_buft_get_device(spec.buft);
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!dev) {
        return { false, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_WRITE_K };
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 2*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error("failed to create native-exact KV capability context");
    }
    constexpr int64_t n_body = 256;
    auto * idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 1);
    auto * src_k = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, spec.head_dim_k, 1);
    auto * src_v = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, spec.head_dim_v, 1);
    auto * body_k = ggml_new_tensor_2d(ctx.get(), actual_k, spec.head_dim_k, n_body);
    auto * body_v = v_transposed ?
            ggml_new_tensor_2d(ctx.get(), actual_v, n_body, spec.head_dim_v) :
            ggml_new_tensor_2d(ctx.get(), actual_v, spec.head_dim_v, n_body);
    const auto supports = [&](ggml_tensor * op) {
        return op && ggml_backend_dev_supports_op(dev, op);
    };

    if (!supports(ggml_set_rows(ctx.get(), body_k, src_k, idx))) {
        return { false, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_WRITE_K };
    }
    ggml_tensor * v_write = nullptr;
    if (v_transposed) {
        auto * body_scalar = ggml_reshape_2d(ctx.get(), body_v, 1, ggml_nelements(body_v));
        auto * src_scalar = ggml_reshape_2d(ctx.get(), src_v, 1, ggml_nelements(src_v));
        auto * scalar_idxs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, spec.head_dim_v);
        v_write = ggml_set_rows(ctx.get(), body_scalar, src_scalar, scalar_idxs);
    } else {
        v_write = ggml_set_rows(ctx.get(), body_v, src_v, idx);
    }
    if (!supports(v_write)) {
        return { false, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_WRITE_V };
    }

    if (flash_attn && !spec.explicit_bias) {
        auto * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, spec.head_dim_k, 1, 1, 1);
        auto * k = ggml_reshape_4d(ctx.get(), body_k, spec.head_dim_k, n_body, 1, 1);
        auto * v = ggml_reshape_4d(ctx.get(), body_v, spec.head_dim_v, n_body, 1, 1);
        auto * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_body, 1, 1, 1);
        auto * attn = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, 1.0f, 0.0f, 0.0f);
        return supports(attn) ?
                llama_kv_tail_route_capability { true, LLAMA_KV_TAIL_ROUTE_NATIVE, LLAMA_KV_TAIL_OP_NONE } :
                llama_kv_tail_route_capability { false, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_NATIVE_ATTENTION };
    }

    auto * q = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, spec.head_dim_k, 1);
    auto * scores = ggml_mul_mat(ctx.get(), body_k, q);
    if (!supports(scores)) {
        return { false, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_BODY_SCORE };
    }
    if (spec.explicit_bias) {
        auto * bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_body, 1);
        scores = ggml_add(ctx.get(), scores, bias);
        if (!supports(scores)) {
            return { false, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_BODY_SCORE };
        }
    }
    auto * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_body, 1);
    if (!supports(ggml_soft_max_ext(ctx.get(), scores, mask, 1.0f, 0.0f))) {
        return { false, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_GENERIC_MERGE };
    }
    auto * weights = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_body, 1);
    if (!supports(ggml_mul_mat(ctx.get(), body_v, weights))) {
        return { false, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_BODY_VALUE };
    }
    return { true, LLAMA_KV_TAIL_ROUTE_GENERIC, LLAMA_KV_TAIL_OP_NONE };
}

class scoped_kv_tail_planner_timer {
public:
    scoped_kv_tail_planner_timer(bool enabled, std::atomic<uint64_t> & total) :
            total(enabled ? &total : nullptr), start(enabled ? clock::now() : clock::time_point {}) {}

    ~scoped_kv_tail_planner_timer() {
        if (total) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - start).count();
            total->fetch_add(uint64_t(elapsed), std::memory_order_relaxed);
        }
    }

private:
    using clock = std::chrono::steady_clock;
    std::atomic<uint64_t> * total;
    clock::time_point start;
};

}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
                 uint32_t   n_ubatch,
                 uint32_t   tail_tokens,
                ggml_type   tail_type_requested,
                 uint32_t   tail_tokens_requested,
                     bool   tail_metadata_only,
                 uint32_t   tail_rollback_tokens,
                 uint32_t   tail_visibility_window) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa),
    tail_tokens(tail_tokens), tail_rollback_tokens(tail_rollback_tokens),
    tail_metadata_only(tail_metadata_only),
    tail_type(tail_type_requested), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl) {

    // Shared cells view the source cache's K/V tensors, so resolve the source
    // allocation before sizing tail generations or any cell-indexed metadata.
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer = hparams.n_layer_all;
    const uint32_t n_layer_kv = hparams.n_layer_kv();
    const bool is_mla = hparams.is_mla();

    if (tail_tokens > 0 && tail_type != GGML_TYPE_COUNT &&
            tail_type != GGML_TYPE_F16 && tail_type != GGML_TYPE_BF16) {
        throw std::invalid_argument("standard KV tail type must be F16 or BF16");
    }
    const bool tail_type_auto = tail_type == GGML_TYPE_COUNT;
    if (tail_type_auto) {
        tail_type = tail_metadata_only ? GGML_TYPE_F16 : GGML_TYPE_BF16;
    }

    uint64_t promotion_bytes_per_row = 0;
    uint64_t overlay_bytes_per_row = 0;
    uint64_t requested_body_bytes_per_row = 0;
    bool native_capable = true;
    bool already_exact = true;
    bool has_owned_layer = false;
    bool has_shared_layer = false;
    bool shadow_k_capable = false;
    bool shadow_v_capable = false;
    bool have_source_type = false;
    ggml_type plan_body_type_k = type_k;
    ggml_type plan_body_type_v = type_v;
    std::vector<kv_tail_backend_probe_spec> route_probe_specs;

    const auto add_row_bytes = [](uint64_t & total, size_t value) {
        if (uint64_t(value) > std::numeric_limits<uint64_t>::max() - total) {
            throw std::overflow_error("standard KV tail row byte count overflows uint64_t");
        }
        total += uint64_t(value);
    };

    if (tail_tokens > 0) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (!hparams.has_kv(il) || (filter && !filter(il))) {
                continue;
            }

            const uint32_t owned_k_dim = hparams.n_embd_k_gqa(il);
            const uint32_t owned_v_dim = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();
            const bool has_v = !is_mla;
            bool shared_layer = false;
            ggml_type actual_type_k = type_k;
            ggml_type actual_type_v = type_v;
            uint32_t actual_k_dim = owned_k_dim;
            uint32_t actual_v_dim = owned_v_dim;

            if (share && other) {
                const int32_t il_share = share(il);
                if (il_share >= 0) {
                    const auto & source = other->layers[other->map_layer_ids.at(il_share)];
                    shared_layer = true;
                    actual_type_k = source.k->type;
                    actual_k_dim = uint32_t(source.k->ne[0]);
                    if (has_v && source.v) {
                        actual_type_v = source.v->type;
                        actual_v_dim = uint32_t(source.v->ne[0]);
                    }
                }
            }

            if (shared_layer) {
                has_shared_layer = true;
                if (!have_source_type) {
                    plan_body_type_k = actual_type_k;
                    plan_body_type_v = actual_type_v;
                    have_source_type = true;
                }
            } else {
                has_owned_layer = true;
            }

            const bool quant_k = ggml_is_quantized(actual_type_k);
            const bool quant_v = has_v && ggml_is_quantized(actual_type_v);
            const bool layer_has_quant = quant_k || quant_v;
            add_row_bytes(requested_body_bytes_per_row, ggml_row_size(actual_type_k, actual_k_dim));
            if (has_v) {
                add_row_bytes(requested_body_bytes_per_row, ggml_row_size(actual_type_v, actual_v_dim));
            }
            already_exact = already_exact && !layer_has_quant;
            if (shared_layer && layer_has_quant) {
                native_capable = false;
            }

            if (!shared_layer) {
                if (quant_k) {
                    const size_t body = ggml_row_size(actual_type_k, actual_k_dim);
                    const size_t exact = ggml_row_size(tail_type, actual_k_dim);
                    GGML_ASSERT(exact >= body);
                    add_row_bytes(promotion_bytes_per_row, exact - body);
                }
                if (quant_v) {
                    const size_t body = ggml_row_size(actual_type_v, actual_v_dim);
                    const size_t exact = ggml_row_size(tail_type, actual_v_dim);
                    GGML_ASSERT(exact >= body);
                    add_row_bytes(promotion_bytes_per_row, exact - body);
                }
            }

            if (layer_has_quant) {
                if (quant_k || actual_type_k == GGML_TYPE_F16 || actual_type_k == GGML_TYPE_BF16 ||
                        actual_type_k == GGML_TYPE_F32) {
                    shadow_k_capable = true;
                    const ggml_type shadow_type = quant_k ? tail_type : actual_type_k;
                    add_row_bytes(overlay_bytes_per_row, ggml_row_size(shadow_type, actual_k_dim));
                }
                if (has_v && (quant_v || actual_type_v == GGML_TYPE_F16 || actual_type_v == GGML_TYPE_BF16 ||
                        actual_type_v == GGML_TYPE_F32)) {
                    shadow_v_capable = true;
                    const ggml_type shadow_type = quant_v ? tail_type : actual_type_v;
                    add_row_bytes(overlay_bytes_per_row, ggml_row_size(shadow_type, actual_v_dim));
                }
            }

            ggml_backend_buffer_type_t route_buft = nullptr;
            if (shared_layer) {
                const int32_t il_share = share(il);
                const auto & source = other->layers[other->map_layer_ids.at(il_share)];
                route_buft = ggml_backend_buffer_get_type(source.k->buffer);
            } else if (offload) {
                route_buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
            } else {
                route_buft = ggml_backend_cpu_buffer_type();
            }
            route_probe_specs.push_back({
                    il, route_buft, actual_type_k, actual_type_v,
                    int64_t(hparams.n_embd_head_k(il)),
                    int64_t(has_v ? hparams.n_embd_head_v(il) : 0), has_v,
                    model.self_attention_uses_explicit_bias(il),
            });
        }

        if (has_owned_layer) {
            plan_body_type_k = type_k;
            plan_body_type_v = type_v;
        }
    }

    const uint32_t storage_visibility_window = n_swa > 0 ? std::min(n_swa, kv_size) : kv_size;
    const uint32_t visibility_window = tail_visibility_window > 0 ?
            std::min(tail_visibility_window, storage_visibility_window) : storage_visibility_window;
    if (tail_tokens_requested == UINT32_MAX) {
        tail_tokens_requested = tail_tokens;
    }
    llama_kv_tail_storage_request storage_request {
        plan_body_type_k,
        plan_body_type_v,
        tail_type,
        tail_tokens_requested,
        tail_tokens,
        n_seq_max,
        n_ubatch,
        visibility_window,
        uint64_t(kv_size)*n_stream,
        requested_body_bytes_per_row,
        promotion_bytes_per_row,
        overlay_bytes_per_row,
        native_capable,
        already_exact,
        has_owned_layer,
        has_shared_layer,
        shadow_k_capable,
        shadow_v_capable,
        model.graph_consumes_exact_kv_tail(),
        true,
        llm_arch_name(model.arch),
        n_swa > 0 ? "swa" : "full",
        tail_rollback_tokens,
        true,
        true,
        true,
        n_swa > 0 && !has_shared_layer,
    };
    tail_plan = llama_kv_tail_storage_plan_for(storage_request);

    const auto resolve_overlay_routes = [&](ggml_type candidate,
                                            std::vector<llama_kv_tail_layer_route> & routes,
                                            llama_kv_tail_route_capability & first_failure) {
        routes.clear();
        first_failure = { true, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_NONE };
        for (const auto & spec : route_probe_specs) {
            auto route_spec = spec;
            if (tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT) {
                // The body is a masked one-row anchor, but its realized tensor
                // types still participate in backend support checks. Match the
                // exact source so a bodyless route never depends on a compiled
                // mixed pair that performs no useful work.
                route_spec.body_type_k = candidate;
                route_spec.body_type_v = candidate;
            }
            const ggml_type exact_k = ggml_is_quantized(route_spec.body_type_k) ? candidate : route_spec.body_type_k;
            const ggml_type exact_v = ggml_is_quantized(route_spec.body_type_v) ? candidate : route_spec.body_type_v;
            const auto capability = probe_standard_kv_tail_route(
                    route_spec, exact_k, exact_v, v_trans, !v_trans,
                    tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_OVERLAY ||
                    tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT);
            const bool has_body = tail_plan.kind != LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT;
            const uint32_t body_execution_rows = !has_body ? 0 :
                    tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_OVERLAY ?
                        llama_kv_tail_packed_body_stride(
                                tail_plan.compact_layout.history_stride, 256) :
                        tail_plan.kind == LLAMA_KV_TAIL_STORAGE_OVERLAY ?
                            tail_plan.layout.arena_stride : 0;
            auto * dev = ggml_backend_buft_get_device(spec.buft);
            if (!dev) {
                dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            }
            routes.push_back({
                    spec.layer_id,
                    dev ? ggml_backend_dev_name(dev) : ggml_backend_buft_name(spec.buft),
                    route_spec.body_type_k, route_spec.body_type_v, exact_k, exact_v,
                    v_trans, hparams.causal_attn, n_swa > 0, spec.explicit_bias,
                    has_body,
                    (tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_OVERLAY ||
                     tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT) &&
                        capability.route == LLAMA_KV_TAIL_ROUTE_NATIVE,
                    body_execution_rows, dev,
                    capability,
            });
            if (!capability.supported && first_failure.supported) {
                first_failure = capability;
            }
        }
        return first_failure.supported;
    };

    const auto resolve_native_exact_routes = [&](ggml_type candidate,
                                                 std::vector<llama_kv_tail_layer_route> & routes,
                                                 llama_kv_tail_route_capability & first_failure) {
        routes.clear();
        first_failure = { true, LLAMA_KV_TAIL_ROUTE_NONE, LLAMA_KV_TAIL_OP_NONE };
        for (const auto & spec : route_probe_specs) {
            const ggml_type actual_k = ggml_is_quantized(spec.body_type_k) ? candidate : spec.body_type_k;
            const ggml_type actual_v = ggml_is_quantized(spec.body_type_v) ? candidate : spec.body_type_v;
            const auto capability = probe_standard_native_exact_route(
                    spec, actual_k, actual_v, v_trans, !v_trans);
            auto * dev = ggml_backend_buft_get_device(spec.buft);
            if (!dev) {
                dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            }
            routes.push_back({
                    spec.layer_id,
                    dev ? ggml_backend_dev_name(dev) : ggml_backend_buft_name(spec.buft),
                    actual_k, actual_v, actual_k, actual_v,
                    v_trans, hparams.causal_attn, n_swa > 0, spec.explicit_bias,
                    true, false, 0, dev, capability,
            });
            if (!capability.supported && first_failure.supported) {
                first_failure = capability;
            }
        }
        return first_failure.supported;
    };

    const auto route_failure = [&](const char * representation,
                                   const std::vector<llama_kv_tail_layer_route> & routes) {
        const auto it = std::find_if(routes.begin(), routes.end(), [](const auto & route) {
            return !route.capability.supported;
        });
        if (it == routes.end()) {
            return std::string(representation) + " route failed without a missing operation";
        }
        return format(
                "%s KV tail route is unsupported for architecture %s group %s layer %u backend %s "
                "body %s/%s exact %s/%s: missing %s",
                representation, llm_arch_name(model.arch), n_swa > 0 ? "swa" : "full",
                it->layer_id, it->backend.c_str(),
                ggml_type_name(it->body_type_k), ggml_type_name(it->body_type_v),
                ggml_type_name(it->exact_type_k), ggml_type_name(it->exact_type_v),
                llama_kv_tail_operation_name(it->capability.missing_operation));
    };

    if (tail_plan.kind == LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT && !route_probe_specs.empty()) {
        llama_kv_tail_route_capability failure;
        if (!resolve_native_exact_routes(tail_type, tail_plan.layer_routes, failure)) {
            if (tail_type_auto && tail_type == GGML_TYPE_BF16) {
                const std::string bf16_failure = route_failure("standard native-exact", tail_plan.layer_routes);
                std::vector<llama_kv_tail_layer_route> f16_routes;
                llama_kv_tail_route_capability f16_failure;
                const bool f16_supported = resolve_native_exact_routes(GGML_TYPE_F16, f16_routes, f16_failure);
                const auto type_resolution = llama_kv_tail_resolve_type(
                        tail_type_requested, GGML_TYPE_BF16, failure,
                        f16_supported ? llama_kv_tail_route_capability {
                            true, f16_routes.front().capability.route, LLAMA_KV_TAIL_OP_NONE } : f16_failure);
                if (!type_resolution.supported) {
                    throw std::runtime_error("standard KV tail auto type has no complete route; BF16: " +
                            bf16_failure + "; F16: " + route_failure("standard native-exact", f16_routes));
                }
                GGML_ASSERT(type_resolution.downgraded && type_resolution.actual_type == GGML_TYPE_F16);
                tail_type = type_resolution.actual_type;
                storage_request.exact_type = tail_type;
                tail_plan = llama_kv_tail_storage_plan_for(storage_request);
                GGML_ASSERT(tail_plan.kind == LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT);
                tail_plan.layer_routes = std::move(f16_routes);
                LLAMA_LOG_WARN("KV tail: auto BF16 native-exact route unavailable; using F16 for the complete %s group\n",
                        n_swa > 0 ? "SWA" : "full-context");
            } else {
                throw std::runtime_error(route_failure("standard native-exact", tail_plan.layer_routes));
            }
        }
    }

    if (tail_metadata_only && tail_tokens > 0) {
        // A metadata-only cache owns cells, generations, and source selection,
        // while its caller owns the exact payload tensors. Keep this generic so
        // every structured cache can reuse the same logical tail machinery.
        tail_plan = llama_kv_tail_storage_plan_for({
            type_k,
            type_v,
            tail_type,
            tail_tokens_requested,
            tail_tokens,
            n_seq_max,
            n_ubatch,
            visibility_window,
            uint64_t(kv_size)*n_stream,
            1,
            0,
            1,
            false,
            false,
            true,
            false,
            true,
            true,
            true,
            true,
            llm_arch_name(model.arch),
            n_swa > 0 ? "swa" : "full",
            tail_rollback_tokens,
            true,
            true,
            true,
            false,
        });
    }

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t((2u*(1 + n_stream) +
                        (tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT ? 2u : 0u))*
                        n_layer_kv*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }
    allocation_seq_heads.assign(n_seq_max, 0);

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                // exact-tail storage is never shared between layers
                layers.back().k_tail = nullptr;
                layers.back().v_tail = nullptr;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (!is_mla) {
            if (n_embd_head_v_all == 0) {
                n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
            } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
                n_embd_head_v_all = -1;
            }
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        const bool native_exact = tail_plan.kind == LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT;
        const bool compact_native_exact =
                tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT;
        ggml_type layer_type_k = native_exact ? tail_plan.actual_body_type_k : type_k;
        ggml_type layer_type_v = native_exact ? tail_plan.actual_body_type_v : type_v;

        ggml_tensor * k = has_k && !compact_native_exact ?
                ggml_new_tensor_3d(ctx, layer_type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v && !compact_native_exact ?
                ggml_new_tensor_3d(ctx, layer_type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * k_tail = compact_native_exact ?
                ggml_new_tensor_2d(ctx, tail_plan.actual_body_type_k, n_embd_k_gqa,
                        tail_plan.compact_layout.history_slots) : nullptr;
        ggml_tensor * v_tail = has_v && compact_native_exact ?
                ggml_new_tensor_2d(ctx, tail_plan.actual_body_type_v, n_embd_v_gqa,
                        tail_plan.compact_layout.history_slots) : nullptr;

        if (k) {
            ggml_format_name(k, "cache_k_l%d", il);
        }
        if (v) {
            ggml_format_name(v, "cache_v_l%d", il);
        }
        if (k_tail) {
            ggml_format_name(k_tail, "cache_k_tail_l%d", il);
        }
        if (v_tail) {
            ggml_format_name(v_tail, "cache_v_tail_l%d", il);
        }

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_tail, v_tail, k_stream, v_stream });
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);

        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    const auto tensor_buft = [](const ggml_tensor * tensor) -> ggml_backend_buffer_type_t {
        return tensor && tensor->buffer ? ggml_backend_buffer_get_type(tensor->buffer) : nullptr;
    };
    const auto buft_is_meta = [](ggml_backend_buffer_type_t buft) {
        auto * dev = buft ? ggml_backend_buft_get_device(buft) : nullptr;
        return dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_META;
    };
    const auto validate_meta_body = [&](const kv_layer & layer, const ggml_tensor * tensor, const char * side) {
        const auto split = llama_meta_device_get_split_state(tensor,
                const_cast<llama_meta_device_get_split_state_userdata *>(&model.get_split_state_ud));
        if (split.axis != GGML_BACKEND_SPLIT_AXIS_0 || split.n_segments == 0) {
            throw std::runtime_error(format(
                    "standard KV tensor/meta split is invalid for layer %u %s body; "
                    "the ordinary KV path did not produce a valid row split",
                    layer.il, side));
        }
    };

    // Route and placement decisions are based on the realized persistent body,
    // not on split-mode text or an intended device. This is deliberately after
    // body allocation and before any tail arena, shadow tensor, or generations.
    for (auto & spec : route_probe_specs) {
        const auto mapped = map_layer_ids.find(int32_t(spec.layer_id));
        if (mapped == map_layer_ids.end()) {
            throw std::logic_error("KV tail route references an unallocated body layer");
        }
        const auto & layer = layers.at(mapped->second);
        const bool compact_native_exact =
                tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT;
        const auto * owner_k = compact_native_exact ? layer.k_tail : layer.k;
        const auto * owner_v = compact_native_exact ? layer.v_tail : layer.v;
        const auto k_buft = tensor_buft(owner_k);
        const auto v_buft = tensor_buft(owner_v);
        if (!k_buft || (owner_v && !v_buft)) {
            throw std::runtime_error(format("standard KV layer %u has no realized backend buffer", spec.layer_id));
        }
        if (owner_v && k_buft != v_buft) {
            throw std::runtime_error(format(
                    "standard KV body layer %u places K and V on different owners; exact-tail routing requires one layer owner",
                    spec.layer_id));
        }
        spec.buft = k_buft;

        const bool k_meta = buft_is_meta(k_buft);
        const bool v_meta = buft_is_meta(v_buft);
        if (k_meta) {
            validate_meta_body(layer, owner_k, "K");
        }
        if (v_meta) {
            validate_meta_body(layer, owner_v, "V");
        }
    }

    if ((tail_plan.kind == LLAMA_KV_TAIL_STORAGE_OVERLAY ||
            tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_OVERLAY ||
            tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT) && !route_probe_specs.empty()) {
        llama_kv_tail_route_capability failure;
        if (!resolve_overlay_routes(tail_type, tail_plan.layer_routes, failure)) {
            if (tail_type_auto && tail_type == GGML_TYPE_BF16) {
                const std::string bf16_failure = route_failure("standard overlay", tail_plan.layer_routes);
                std::vector<llama_kv_tail_layer_route> f16_routes;
                llama_kv_tail_route_capability f16_failure;
                const bool f16_supported = resolve_overlay_routes(GGML_TYPE_F16, f16_routes, f16_failure);
                const auto type_resolution = llama_kv_tail_resolve_type(
                        tail_type_requested, GGML_TYPE_BF16, failure,
                        f16_supported ? llama_kv_tail_route_capability {
                            true, f16_routes.front().capability.route, LLAMA_KV_TAIL_OP_NONE } : f16_failure);
                if (!type_resolution.supported) {
                    throw std::runtime_error("standard KV tail auto type has no complete route; BF16: " +
                            bf16_failure + "; F16: " + route_failure("standard overlay", f16_routes));
                }
                GGML_ASSERT(type_resolution.downgraded && type_resolution.actual_type == GGML_TYPE_F16);
                tail_type = type_resolution.actual_type;
                storage_request.exact_type = tail_type;
                tail_plan = llama_kv_tail_storage_plan_for(storage_request);
                tail_plan.layer_routes = std::move(f16_routes);
                LLAMA_LOG_WARN("KV tail: auto BF16 route unavailable; using F16 for the complete %s group\n",
                        n_swa > 0 ? "SWA" : "full-context");
            } else {
                throw std::runtime_error(route_failure("standard overlay", tail_plan.layer_routes));
            }
        }
    }

    if (has_tail_overlay() && !tail_metadata_only) {
        finalize_tail_overlay_metadata();

        std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> tail_ctx_map;
        const auto tail_ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
            auto it = tail_ctx_map.find(buft);
            if (it != tail_ctx_map.end()) {
                return it->second.get();
            }
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*n_layer_kv*ggml_tensor_overhead()),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr ctx { ggml_init(params) };
            if (!ctx) {
                return nullptr;
            }
            auto * result = ctx.get();
            tail_ctx_map.emplace(buft, std::move(ctx));
            return result;
        };

        for (auto & layer : layers) {
            if (tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT) {
                continue;
            }
            const bool layer_has_quant = ggml_is_quantized(layer.k->type) ||
                    (layer.v && ggml_is_quantized(layer.v->type));
            if (!layer_has_quant) {
                continue;
            }
            const auto k_buft = tensor_buft(layer.k);
            const auto v_buft = tensor_buft(layer.v);
            if (tail_plan.shadow_k && (ggml_is_quantized(layer.k->type) ||
                    layer.k->type == GGML_TYPE_F16 || layer.k->type == GGML_TYPE_BF16 || layer.k->type == GGML_TYPE_F32)) {
                auto * ctx = tail_ctx_for_buft(k_buft);
                if (!ctx) {
                    throw std::runtime_error("failed to create standard K tail context");
                }
                const ggml_type shadow_type = ggml_is_quantized(layer.k->type) ? tail_type : layer.k->type;
                layer.k_tail = ggml_new_tensor_2d(ctx, shadow_type, layer.k->ne[0], tail_slots);
                ggml_format_name(layer.k_tail, "cache_k_tail_l%d", layer.il);
            }
            if (tail_plan.shadow_v && layer.v && (ggml_is_quantized(layer.v->type) ||
                    layer.v->type == GGML_TYPE_F16 || layer.v->type == GGML_TYPE_BF16 || layer.v->type == GGML_TYPE_F32)) {
                auto * ctx = tail_ctx_for_buft(v_buft);
                if (!ctx) {
                    throw std::runtime_error("failed to create standard V tail context");
                }
                const ggml_type shadow_type = ggml_is_quantized(layer.v->type) ? tail_type : layer.v->type;
                layer.v_tail = ggml_new_tensor_2d(ctx, shadow_type, layer.v->ne[0], tail_slots);
                ggml_format_name(layer.v_tail, "cache_v_tail_l%d", layer.il);
            }
        }

        for (auto & [buft, ctx] : tail_ctx_map) {
            ggml_backend_buffer_t buf;
            if (hparams.no_alloc) {
                buf = ggml_backend_buft_alloc_buffer(buft, 0);
                for (auto * tensor = ggml_get_first_tensor(ctx.get()); tensor != nullptr; tensor = ggml_get_next_tensor(ctx.get(), tensor)) {
                    tensor->buffer = buf;
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
            }
            if (!buf) {
                throw std::runtime_error("failed to allocate standard KV tail buffer");
            }
            ggml_backend_buffer_clear(buf, 0);
            LLAMA_LOG_INFO("%s: %10s KV tail buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
            ctxs_bufs.emplace_back(std::move(ctx), buf);
        }

        for (const auto & layer : layers) {
            const bool compact_native_exact =
                    tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT;
            const uintptr_t k_owner = reinterpret_cast<uintptr_t>(
                    tensor_buft(compact_native_exact ? layer.k_tail : layer.k));
            const uintptr_t v_owner = reinterpret_cast<uintptr_t>(
                    tensor_buft(compact_native_exact ? layer.v_tail : layer.v));
            auto ownership = llama_kv_tail_plan_layer_ownership(
                    layer.il, k_owner, v_owner, layer.k_tail != nullptr, layer.v_tail != nullptr);
            ownership.shadow_k_owner = reinterpret_cast<uintptr_t>(tensor_buft(layer.k_tail));
            ownership.shadow_v_owner = reinterpret_cast<uintptr_t>(tensor_buft(layer.v_tail));
            const auto error = llama_kv_tail_validate_layer_ownership(ownership);
            if (error != LLAMA_KV_TAIL_OWNERSHIP_OK) {
                throw std::runtime_error(format("standard KV tail ownership validation failed for layer %u (error %d)",
                        layer.il, int(error)));
            }
        }
    }

    if (tail_plan.requested_tokens > 0) {
        const char * storage =
                tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT ? "compact_native_exact" :
                tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_OVERLAY ? "compact_overlay" :
                tail_plan.kind == LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT ? "native_exact" :
                tail_plan.kind == LLAMA_KV_TAIL_STORAGE_OVERLAY ? "overlay" : "disabled";
        LLAMA_LOG_INFO("%s: standard KV tail plan: storage=%s requested=%u effective=%u window=%u body=%s/%s actual=%s/%s physical_rows=%llu arena=%u sink=%u overlay_rows=%u promotion=%.2f MiB overlay=%.2f MiB\n",
                __func__, storage, tail_plan.requested_tokens, tail_plan.effective_tokens, tail_plan.visibility_window,
                ggml_type_name(type_k), ggml_type_name(type_v),
                ggml_type_name(tail_plan.actual_body_type_k), ggml_type_name(tail_plan.actual_body_type_v),
                (unsigned long long) tail_plan.physical_body_rows,
                tail_plan.layout.arena_stride, tail_plan.layout.sink_slots, tail_plan.layout.total_slots,
                tail_plan.promotion_increment/1024.0/1024.0, tail_plan.overlay_increment/1024.0/1024.0);
    }

    {
        const size_t memory_size_k     = size_k_bytes();
        const size_t memory_size_v     = size_v_bytes();
        const char * actual_type_k_name = layers.empty() || !layers[0].k ? "none" : ggml_type_name(layers[0].k->type);
        const char * actual_type_v_name = layers.empty() || !layers[0].v ? "none" : ggml_type_name(layers[0].v->type);

        constexpr float mib = 1024.0f * 1024.0f;

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / mib, kv_size, (int) layers.size(), n_seq_max, n_stream,
                actual_type_k_name, (float)memory_size_k / mib,
                actual_type_v_name, (float)memory_size_v / mib);

        if (tail_plan.kind == LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT) {
            std::ostringstream ids;
            for (size_t i = 0; i < layers.size(); ++i) {
                ids << (i ? "," : "") << layers[i].il;
            }
            LLAMA_LOG_INFO("%s: tail type=%s requested=%u effective=%u storage=native_exact physical_rows=%llu shadow=0.00 MiB sink=0 planner=0 state_tail=0 total=%.2f MiB\n",
                    __func__, ggml_type_name(tail_type), tail_plan.requested_tokens, tail_plan.effective_tokens,
                    (unsigned long long) uint64_t(kv_size)*n_stream,
                    (memory_size_k + memory_size_v)/1024.0/1024.0);
            LLAMA_LOG_INFO("KV tail: group=%s layers=%s representation=native_exact route=body body=%s/%s shadow=none requested=%u effective=%u\n",
                    n_swa > 0 ? "swa" : "full", ids.str().c_str(),
                    actual_type_k_name, actual_type_v_name, tail_plan.requested_tokens, tail_plan.effective_tokens);
        }

        if (tail) {
            size_t tail_k_bytes = 0;
            size_t tail_v_bytes = 0;
            for (const auto & layer : layers) {
                tail_k_bytes += layer.k_tail ? ggml_nbytes(layer.k_tail) : 0;
                tail_v_bytes += layer.v_tail ? ggml_nbytes(layer.v_tail) : 0;
            }
            const size_t generation_bytes = size_t(n_stream)*kv_size*sizeof(uint64_t);
            const uint64_t arena_slots = uint64_t(tail_arena_stride)*n_seq_max;
            const size_t tail_k_arena_bytes = tail_slots ? size_t(uint64_t(tail_k_bytes)*arena_slots/tail_slots) : 0;
            const size_t tail_v_arena_bytes = tail_slots ? size_t(uint64_t(tail_v_bytes)*arena_slots/tail_slots) : 0;
            const size_t tail_k_sink_bytes = tail_k_bytes - tail_k_arena_bytes;
            const size_t tail_v_sink_bytes = tail_v_bytes - tail_v_arena_bytes;
            LLAMA_LOG_INFO("%s: tail type=%s requested=%u effective=%u slots=%u "
                    "(arena=%llu = %u*%u, sink=%u), "
                    "K=%.2f MiB (arena=%.2f sink=%.2f) V=%.2f MiB (arena=%.2f sink=%.2f) "
                    "metadata>=%.2f MiB total=%.2f MiB\n",
                    __func__, ggml_type_name(tail_type), tail_plan.requested_tokens, tail_plan.effective_tokens, tail_slots,
                    (unsigned long long) arena_slots, tail_arena_stride, n_seq_max, tail_sink_slots,
                    tail_k_bytes/1024.0/1024.0, tail_k_arena_bytes/1024.0/1024.0,
                    tail_k_sink_bytes/1024.0/1024.0, tail_v_bytes/1024.0/1024.0,
                    tail_v_arena_bytes/1024.0/1024.0, tail_v_sink_bytes/1024.0/1024.0,
                    generation_bytes/1024.0/1024.0,
                    (memory_size_k + memory_size_v + tail_k_bytes + tail_v_bytes + generation_bytes)/1024.0/1024.0);

            std::map<std::string, std::vector<uint32_t>> route_layers;
            for (const auto & layer_route : tail_plan.layer_routes) {
                GGML_ASSERT(layer_route.capability.supported);
                const bool native = layer_route.capability.route == LLAMA_KV_TAIL_ROUTE_NATIVE;
                std::ostringstream key;
                key << layer_route.backend << '|' << (native ? "native" : "generic")
                    << '|' << ggml_type_name(layer_route.body_type_k) << '/'
                    << ggml_type_name(layer_route.body_type_v)
                    << '|' << ggml_type_name(layer_route.exact_type_k) << '/'
                    << ggml_type_name(layer_route.exact_type_v)
                    << '|' << (layer_route.has_body ? "body" : "bodyless")
                    << '|' << (layer_route.has_current ? "current" : "no-current")
                    << '|' << layer_route.body_execution_rows;
                route_layers[key.str()].push_back(layer_route.layer_id);
            }
            for (const auto & [key, layer_ids] : route_layers) {
                const size_t p0 = key.find('|');
                const size_t p1 = key.find('|', p0 + 1);
                const size_t p2 = key.find('|', p1 + 1);
                const size_t p3 = key.find('|', p2 + 1);
                const size_t p4 = key.find('|', p3 + 1);
                const size_t p5 = key.find('|', p4 + 1);
                std::ostringstream ids;
                for (size_t i = 0; i < layer_ids.size(); ++i) {
                    ids << (i ? "," : "") << layer_ids[i];
                }
                const std::string dev_name = key.substr(0, p0);
                const std::string route = key.substr(p0 + 1, p1 - p0 - 1);
                const std::string body_types = key.substr(p1 + 1, p2 - p1 - 1);
                const std::string shadow_types = key.substr(p2 + 1, p3 - p2 - 1);
                const std::string body_presence = key.substr(p3 + 1, p4 - p3 - 1);
                const std::string current_presence = key.substr(p4 + 1, p5 - p4 - 1);
                const std::string execution_rows = key.substr(p5 + 1);
                LLAMA_LOG_INFO("KV tail: group=%s layers=%s dev=%s route=%s body=%s shadow=%s "
                        "presence=%s current=%s execution_rows=%s requested=%u effective=%u\n",
                        n_swa > 0 ? "swa" : "full", ids.str().c_str(), dev_name.c_str(), route.c_str(),
                        body_types.c_str(), shadow_types.c_str(), body_presence.c_str(),
                        current_presence.c_str(), execution_rows.c_str(),
                        tail_plan.requested_tokens, tail_plan.effective_tokens);
                if (route == "generic") {
                    LLAMA_LOG_WARN("KV tail: explicit positive request uses catastrophic generic attention for "
                            "group=%s layers=%s dev=%s; correctness is preserved but long-context performance may collapse\n",
                            n_swa > 0 ? "swa" : "full", ids.str().c_str(), dev_name.c_str());
                }
            }
        }
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        attn_rot_k = other->attn_rot_k;
        attn_rot_v = other->attn_rot_v;
    } else {
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
        }

        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek lightning indexers
        if ((model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4 || model.arch == LLM_ARCH_GLM_DSA) &&
                hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;
    }

    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

void llama_kv_cache::clear(bool data) {
    sc_info = {};
    if (tail) {
        tail->clear();
        tail_ordinal = 0;
        tail_write_slots.clear();
        for (auto & generations : tail_generations) {
            std::fill(generations.begin(), generations.end(), 0);
        }
    }
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }
    for (llama_seq_id seq_id = 0; uint32_t(seq_id) < n_seq_max; ++seq_id) {
        reset_allocation_head(seq_id);
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }

    }
}

llama_memory_i::seq_rm_capability llama_kv_cache::get_seq_rm_capability() const {
    if (has_compact_tail()) {
        return {
            /* .full_clear = */ true,
            /* .arbitrary_ranges = */ false,
            /* .suffix_rollback_tokens = */ tail_rollback_tokens,
        };
    }
    return {};
}

bool llama_kv_cache::can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    if (!has_compact_tail()) {
        return true;
    }
    if (seq_id < 0 || size_t(seq_id) >= seq_to_stream.size()) {
        return p0 <= 0 && p1 < 0;
    }
    if (p0 <= 0 && p1 < 0) {
        return true;
    }
    if (p0 <= 0 || p1 >= 0) {
        return false;
    }
    // A compact tail that shadows a body is an optimisation overlay: the body
    // still holds every surviving position, so dropping exact rows to a deep
    // suffix removal costs precision until the tail refills, never data. Only a
    // bodyless tail (COMPACT_NATIVE_EXACT, where the body was omitted because
    // the tail covers the whole window) is the sole copy of those rows and has
    // to keep its removal inside the persistent reserve.
    //
    // tail_metadata_only covers the structured-cache case, where the body is
    // owned by the enclosing cache (KVarN records) rather than by this one.
    if (tail_metadata_only || tail_plan.has_owned_body || tail_plan.has_shared_body) {
        return true;
    }
    return llama_kv_tail_can_remove_suffix(
            seq_pos_max(seq_id), p0, p1, tail_rollback_tokens);
}

bool llama_kv_cache::seq_rm_plan(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1,
        llama_pos & planned_p0, llama_pos & planned_p1) const {
    if (!can_seq_rm(seq_id, p0, p1)) {
        return false;
    }
    planned_p0 = p0;
    planned_p1 = p1;
    return true;
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (other) {
        return true;
    }
    if (!can_seq_rm(seq_id, p0, p1)) {
        LLAMA_LOG_WARN("%s: compact KV tail supports complete clear or at most %u suffix tokens\n",
                __func__, tail_rollback_tokens);
        return false;
    }
    return seq_rm_unchecked(seq_id, p0, p1);
}

bool llama_kv_cache::seq_rm_unchecked(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }
    materialize_pending_copies();

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (tail) {
        if (seq_id >= 0) {
            tail->seq_rm(seq_id, p0, p1);
        } else {
            for (llama_seq_id current = 0; current < int32_t(n_seq_max); ++current) {
                tail->seq_rm(current, p0, p1);
            }
        }
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    if (tail) {
        const auto mark_if_incomplete = [&](llama_seq_id current) {
            const auto & cells_cur = v_cells[seq_to_stream[current]];
            const uint32_t available = cells_cur.seq_size(current);
            const auto coverage = tail->coverage(current, available);
            if (coverage.exact < coverage.requested) {
                tail->mark_degraded(current, LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
            }
        };
        if (seq_id >= 0) {
            mark_if_incomplete(seq_id);
        } else {
            for (llama_seq_id current = 0; current < int32_t(n_seq_max); ++current) {
                mark_if_incomplete(current);
            }
        }
    }

    if (seq_id >= 0) {
        rebuild_allocation_head(seq_id);
    } else {
        for (llama_seq_id current = 0; current < int32_t(n_seq_max); ++current) {
            rebuild_allocation_head(current);
        }
    }

    return true;
}

bool llama_kv_cache::seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (cells.seq_rm_cell(cell_idx, seq_id)) {
        if (tail) {
            tail->seq_rm_cell(seq_id, seq_to_stream[seq_id], cell_idx);
            const uint32_t available = cells.seq_size(seq_id);
            const auto coverage = tail->coverage(seq_id, available);
            if (coverage.exact < coverage.requested) {
                tail->mark_degraded(seq_id, LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
            }
        }
        if (cell_idx < head) {
            head = cell_idx;
        }
        rebuild_allocation_head(seq_id);
    }

    return true;
}

int llama_kv_cache::cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];
    auto result = cells.cells_at(seq_id, pos);

    if (cell_indices && n_max > 0) {
        int n_copied = std::min((int)result.size(), n_max);
        for (int i = 0; i < n_copied; ++i) {
            cell_indices[i] = result[i];
        }
    }

    return (int)result.size();
}

void llama_kv_cache::materialize_pending_copies() {
    if (sc_info.empty()) {
        return;
    }
    for (size_t i = 0; i < sc_info.ssrc.size(); ++i) {
        const uint32_t src = sc_info.ssrc[i];
        const uint32_t dst = sc_info.sdst[i];
        for (const auto & layer : layers) {
            ggml_backend_tensor_copy(layer.k_stream[src], layer.k_stream[dst]);
            if (layer.v_stream[src]) {
                ggml_backend_tensor_copy(layer.v_stream[src], layer.v_stream[dst]);
            }
        }
    }
    for (size_t i = 0; i < sc_info.tail_src_slots.size(); ++i) {
        const int32_t src = sc_info.tail_src_slots[i];
        const int32_t dst = sc_info.tail_dst_slots[i];
        for (const auto & layer : layers) {
            for (ggml_tensor * tensor : { layer.k_tail, layer.v_tail }) {
                if (!tensor) {
                    continue;
                }
                const size_t row = ggml_row_size(tensor->type, tensor->ne[0]);
                std::vector<uint8_t> payload(row);
                ggml_backend_tensor_get(tensor, payload.data(), size_t(src)*row, row);
                ggml_backend_tensor_set(tensor, payload.data(), size_t(dst)*row, row);
            }
        }
    }
    if (tail && sc_info.tail_transaction) {
        tail->commit_seq_cp();
    }
    sc_info = {};
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }
    materialize_pending_copies();

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    const auto queue_tail_copies = [&]() {
        if (!tail) {
            return;
        }
        const auto copies = tail->prepare_seq_cp(seq_id_src, seq_id_dst, s0, s1, p0, p1);
        for (const auto & copy : copies) {
            sc_info.tail_src_slots.push_back(copy.src_slot);
            sc_info.tail_dst_slots.push_back(copy.dst_slot);
        }
        sc_info.tail_transaction = tail->has_pending_seq_cp();
    };

    if (s0 == s1) {
        queue_tail_copies();
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        if (tail) {
            const uint32_t available = cells.seq_size(seq_id_dst);
            const auto coverage = tail->coverage(seq_id_dst, available);
            if (coverage.exact < coverage.requested) {
                tail->mark_degraded(seq_id_dst, LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
            }
        }

        rebuild_allocation_head(seq_id_dst);

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    if (tail) {
        std::fill(tail_generations[s1].begin(), tail_generations[s1].end(), 0);
    }
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);

            if (tail) {
                tail_generations[s1][i] = tail_generations[s0][i];
            }
        }
    }

    if (tail) {
        queue_tail_copies();
        const uint32_t available = v_cells[s1].seq_size(seq_id_dst);
        const auto coverage = tail->coverage(seq_id_dst, available);
        if (coverage.exact < coverage.requested) {
            tail->mark_degraded(seq_id_dst, LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
        }
    }

    v_heads[s1] = v_heads[s0];
    rebuild_allocation_head(seq_id_dst);

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }
    materialize_pending_copies();

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    if (tail) {
        tail->seq_keep(seq_id);
    }

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
    for (llama_seq_id current = 0; current < int32_t(n_seq_max); ++current) {
        rebuild_allocation_head(current);
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }
    materialize_pending_copies();

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    const bool full_range = p0 <= 0 && p1 < 0;
    if (tail) {
        tail->seq_add(seq_id, p0, p1, shift);
        if (shift != 0 && !full_range) {
            tail->mark_degraded(seq_id, LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
        }
    }

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
    rebuild_allocation_head(seq_id);
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }
    materialize_pending_copies();

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    const bool full_range = p0 <= 0 && p1 < 0;
    if (tail) {
        tail->seq_div(seq_id, p0, p1, d);
        if (d != 1 && !full_range) {
            tail->mark_degraded(seq_id, LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
        }
    }

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
    rebuild_allocation_head(seq_id);
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    auto pending = std::move(sc_info);
    sc_info = {};
    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(pending));
}

uint32_t llama_kv_cache::get_kv_n_stream() const {
    return get_n_stream();
}

uint32_t llama_kv_cache::get_kv_size() const {
    return get_size();
}

llama_memory_context_ptr llama_kv_cache::init_kv_batch(const std::vector<llama_ubatch> & ubatches) {
    auto sinfos = prepare(ubatches);
    if (sinfos.empty()) {
        return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    return std::make_unique<llama_kv_cache_context>(this, std::move(sinfos), ubatches);
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    scoped_kv_tail_planner_timer timer(tail_planner_timing_enabled, tail_planner_timing_ns);
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    tail_preparing = true;
    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }
    tail_preparing = false;

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

llama_kv_memory_stats llama_kv_cache::kv_memory_stats() const {
    llama_kv_memory_stats result;
    llama_kv_memory_component_stats & component = n_swa > 0 ? result.swa : result.global;
    for (const auto & route : tail_plan.layer_routes) {
        const bool cpu = !route.owner || ggml_backend_dev_type(route.owner) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if (cpu) {
            component.tail_cpu_layers++;
        } else if (route.capability.route != LLAMA_KV_TAIL_ROUTE_NATIVE) {
            component.tail_device_fallback_layers++;
        } else if (route.has_body) {
            component.tail_native_mixed_layers++;
        } else {
            component.tail_native_bodyless_layers++;
        }
    }

    std::unordered_set<const ggml_tensor *> seen;
    const auto account = [&seen](const ggml_tensor * tensor, uint64_t & total) {
        if (tensor != nullptr && seen.insert(tensor).second) {
            total += ggml_nbytes(tensor);
        }
    };
    const auto account_payload = [&account](const ggml_tensor * tensor, uint64_t & payload, uint64_t & native_exact) {
        if (tensor != nullptr && (tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16)) {
            account(tensor, native_exact);
        } else {
            account(tensor, payload);
        }
    };
    const auto account_tail = [&](const ggml_tensor * tensor) {
        if (tensor == nullptr || !seen.insert(tensor).second) {
            return;
        }
        const uint64_t bytes = ggml_nbytes(tensor);
        if (!has_compact_tail()) {
            component.exact_tail_bytes += bytes;
            return;
        }
        const uint64_t slots = tail_plan.compact_layout.history_slots;
        const uint64_t rollback_slots = uint64_t(tail_plan.compact_layout.rollback_tokens)*n_seq_max;
        GGML_ASSERT(slots > 0 && rollback_slots <= slots && uint64_t(tensor->ne[1]) >= slots);
        const uint64_t row_bytes = tensor->nb[1];
        GGML_ASSERT(row_bytes*uint64_t(tensor->ne[1]) <= bytes);
        const uint64_t rollback_bytes = row_bytes*rollback_slots;
        const uint64_t history_bytes = row_bytes*(slots - rollback_slots);
        if (tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT) {
            component.native_exact_bytes += history_bytes;
        } else {
            component.exact_tail_bytes += history_bytes;
        }
        component.rollback_reserve_bytes += rollback_bytes;
        const uint32_t current_capacity = tail_plan.compact_layout.attention_stride - tail_plan.effective_tokens;
        component.transient_estimate_bytes += row_bytes*current_capacity;
    };
    for (const auto & layer : layers) {
        account_payload(layer.k, component.k_payload_bytes, component.native_exact_bytes);
        account_payload(layer.v, component.v_payload_bytes, component.native_exact_bytes);
        account_tail(layer.k_tail);
        account_tail(layer.v_tail);
    }

    uint64_t allocated = 0;
    for (const auto & [buft, size] : memory_breakdown()) {
        GGML_UNUSED(buft);
        allocated += size;
    }
    const uint64_t accounted = component.k_payload_bytes + component.v_payload_bytes +
            component.exact_tail_bytes + component.native_exact_bytes + component.rollback_reserve_bytes;
    component.padding_bytes = allocated > accounted ? allocated - accounted : 0;
    component.allocated_capacity_tokens =
            tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT ?
                tail_plan.compact_layout.history_stride : get_size();
    return result;
}

llama_memory_status llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return LLAMA_MEMORY_STATUS_NO_UPDATE;
    }

    llama_memory_status result = LLAMA_MEMORY_STATUS_NO_UPDATE;

    auto * sched = lctx->get_sched();
    const auto fail_pending_tail_copy = [&]() {
        if (!tail || !sc_info.tail_transaction) {
            return;
        }
        const llama_seq_id dst = tail->pending_seq_cp_destination();
        tail->cancel_seq_cp();
        if (dst >= 0) {
            seq_rm(dst, -1, -1);
        }
    };

    if (!sc_info.empty()) {
        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();
        if (n_copy > 0) {
            assert(n_stream > 1 && "whole-stream copy should never happen with a single stream");
        }

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
            result = LLAMA_MEMORY_STATUS_SUCCESS;
        }

        if (!sc_info.tail_src_slots.empty()) {
            const size_t n_tail_copy = sc_info.tail_src_slots.size();
            const size_t n_nodes = 4*layers.size() + 8;
            std::vector<uint8_t> meta(
                    ggml_tensor_overhead()*(2 + 4*layers.size()) + ggml_graph_overhead_custom(n_nodes, false));
            ggml_init_params params = { meta.size(), meta.data(), true };
            ggml_context_ptr copy_ctx { ggml_init(params) };
            ggml_tensor * src_idxs = ggml_new_tensor_1d(copy_ctx.get(), GGML_TYPE_I32, n_tail_copy);
            ggml_tensor * dst_idxs = ggml_new_tensor_1d(copy_ctx.get(), GGML_TYPE_I64, n_tail_copy);
            ggml_set_input(src_idxs);
            ggml_set_input(dst_idxs);
            ggml_cgraph * gf = ggml_new_graph_custom(copy_ctx.get(), n_nodes, false);
            for (const auto & layer : layers) {
                for (ggml_tensor * tensor : { layer.k_tail, layer.v_tail }) {
                    if (!tensor) {
                        continue;
                    }
                    const ggml_type copy_type = tensor->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
                    ggml_tensor * gathered = ggml_get_rows_as(copy_ctx.get(), tensor, src_idxs, copy_type);
                    ggml_tensor * copied = ggml_set_rows(copy_ctx.get(), tensor, gathered, dst_idxs);
                    ggml_build_forward_expand(gf, copied);
                }
            }
            std::vector<int64_t> dst_slots(sc_info.tail_dst_slots.begin(), sc_info.tail_dst_slots.end());
            ggml_backend_sched_reset(sched);
            const auto copy_status = llama_kv_tail_copy_transaction(
                    [&] {
                        return ggml_backend_sched_alloc_graph(sched, gf);
                    },
                    [&] {
                        ggml_backend_tensor_set(src_idxs, sc_info.tail_src_slots.data(), 0,
                                n_tail_copy*sizeof(sc_info.tail_src_slots[0]));
                        ggml_backend_tensor_set(dst_idxs, dst_slots.data(), 0,
                                n_tail_copy*sizeof(dst_slots[0]));
                        return true;
                    },
                    [&] {
                        return lctx->graph_compute(gf, false) == GGML_STATUS_SUCCESS;
                    },
                    fail_pending_tail_copy);
            if (llama_memory_status_is_fail(copy_status)) {
                LLAMA_LOG_ERROR("%s: failed to %s tail row-copy graph\n", __func__,
                        copy_status == LLAMA_MEMORY_STATUS_FAILED_PREPARE ? "prepare" : "compute");
                return copy_status;
            }
            result = LLAMA_MEMORY_STATUS_SUCCESS;
        }
        if (tail && sc_info.tail_transaction) {
            tail->commit_seq_cp();
            result = LLAMA_MEMORY_STATUS_SUCCESS;
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return LLAMA_MEMORY_STATUS_FAILED_PREPARE;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return LLAMA_MEMORY_STATUS_FAILED_COMPUTE;
            }

            result = LLAMA_MEMORY_STATUS_SUCCESS;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
        result = LLAMA_MEMORY_STATUS_SUCCESS;
    }

    return result;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    if (allocation_group_size > 1 && n_stream == 1 && !cont) {
        res.s0 = 0;
        res.s1 = 0;
        res.resize(1);
        res.strm[0] = 0;
        res.idxs[0].reserve(ubatch.n_tokens);

        const auto & cells = v_cells[0];
        std::vector<uint32_t> heads = allocation_seq_heads;
        std::vector<bool> reserved(cells.size(), false);
        const uint32_t n_groups = uint32_t(cells.size()/allocation_group_size);
        std::vector<uint32_t> group_used(n_groups, 0);
        for (uint32_t cell = 0; cell < cells.size(); ++cell) {
            group_used[cell/allocation_group_size] += !cells.is_empty(cell);
        }
        std::vector<int32_t> stage_owners(allocation_stage_groups + 1u, -1);
        auto stage_slot = [&](uint32_t group) {
            return group == 0 ? 0u : 1u + ((group - 1u)%allocation_stage_groups);
        };
        if (!cells.is_empty(0)) {
            stage_owners[0] = 0;
        }
        for (llama_seq_id seq_id = 0; uint32_t(seq_id) < n_seq_max; ++seq_id) {
            const uint32_t head = heads[size_t(seq_id)];
            if (head == 0 || head%allocation_group_size == 0) {
                continue;
            }
            const uint32_t group = (head - 1u)/allocation_group_size;
            if (group > 0) {
                const uint32_t slot = stage_slot(group);
                if (stage_owners[slot] >= 0 && uint32_t(stage_owners[slot]) != group) {
                    throw std::runtime_error("structured KV live groups alias one F16 stage slot");
                }
                stage_owners[slot] = int32_t(group);
            }
        }
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            const int32_t n_cell_seqs = ubatch.n_seq_id[i];
            if (n_cell_seqs <= 0) {
                throw std::runtime_error("structured KV allocation token has no sequence owner");
            }
            const llama_seq_id * cell_seqs = ubatch.seq_id[i];
            llama_seq_id owner_seq = cell_seqs[0];
            for (int32_t j = 0; j < n_cell_seqs; ++j) {
                const llama_seq_id owner = cell_seqs[j];
                if (owner < 0 || uint32_t(owner) >= n_seq_max) {
                    throw std::runtime_error("structured KV allocation token has an invalid sequence owner");
                }
                owner_seq = std::min(owner_seq, owner);
            }

            uint32_t head_cur = heads[size_t(owner_seq)];
            bool found = false;
            for (uint32_t tested = 0; tested < cells.size(); ++tested) {
                if (head_cur >= cells.size()) {
                    head_cur = 0;
                }
                const uint32_t idx = head_cur++;
                const uint32_t group = idx/allocation_group_size;
                if (reserved[idx] || !cells.is_empty(idx)) {
                    continue;
                }

                bool compatible = true;
                const uint32_t group_begin = group*allocation_group_size;
                const uint32_t group_end = std::min<uint32_t>(
                        group_begin + allocation_group_size, cells.size());
                for (uint32_t group_cell = group_begin;
                        compatible && group_cell < group_end; ++group_cell) {
                    if (cells.is_empty(group_cell)) {
                        continue;
                    }
                    if (cells.seq_count(group_cell) != n_cell_seqs) {
                        compatible = false;
                        break;
                    }
                    for (int32_t j = 0; j < n_cell_seqs; ++j) {
                        if (!cells.seq_has(group_cell, cell_seqs[j])) {
                            compatible = false;
                            break;
                        }
                    }
                }
                if (!compatible) {
                    continue;
                }

                const uint32_t slot = stage_slot(group);
                if (group_used[group] == 0 && stage_owners[slot] >= 0 &&
                        uint32_t(stage_owners[slot]) != group) {
                    continue;
                }
                if (group_used[group] == 0) {
                    stage_owners[slot] = int32_t(group);
                }

                res.idxs[0].push_back(idx);
                reserved[idx] = true;
                if (++group_used[group] == allocation_group_size && group > 0) {
                    stage_owners[slot] = -1;
                }
                for (int32_t j = 0; j < n_cell_seqs; ++j) {
                    heads[size_t(cell_seqs[j])] = head_cur;
                }
                found = true;
                break;
            }
            if (!found) {
                return {};
            }
        }
        return res;
    }

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                // Structured KVarN records quantize a complete physical group
                // as one indivisible unit. Never append private rows to a
                // group shared with, or owned by, another sequence.
                if (can_use && allocation_group_size > 1) {
                    // A unified ubatch may interleave tokens from several
                    // logical sequences while using one physical stream.  The
                    // next free cell belongs to the next token, not
                    // necessarily ubatch.seq_id_unq[0].  Keep every structured
                    // record group owned by one exact sequence-id set so a
                    // per-sequence state never captures unrelated rows.
                    const uint32_t token_index = uint32_t(res.idxs[s].size());
                    const int32_t n_cell_seqs = n_stream == 1 ?
                            ubatch.n_seq_id[token_index] : 1;
                    if (n_cell_seqs <= 0) {
                        throw std::runtime_error("structured KV allocation token has no sequence owner");
                    }
                    const llama_seq_id * cell_seqs = n_stream == 1 ?
                            ubatch.seq_id[token_index] : &seq_id;
                    llama_seq_id owner_seq = cell_seqs[0];
                    for (int32_t cell_seq = 1; cell_seq < n_cell_seqs; ++cell_seq) {
                        owner_seq = std::min(owner_seq, cell_seqs[cell_seq]);
                    }
                    if (owner_seq < 0) {
                        throw std::runtime_error("structured KV allocation token has an invalid sequence owner");
                    }
                    const uint32_t group_begin = (idx/allocation_group_size)*allocation_group_size;
                    const uint32_t group_end = std::min<uint32_t>(
                            group_begin + allocation_group_size, cells.size());
                    for (uint32_t group_cell = group_begin; can_use && group_cell < group_end; ++group_cell) {
                        if (!cells.is_empty(group_cell)) {
                            if (cells.seq_count(group_cell) != n_cell_seqs) {
                                can_use = false;
                                break;
                            }
                            for (int32_t cell_seq = 0; cell_seq < n_cell_seqs; ++cell_seq) {
                                if (!cells.seq_has(group_cell, cell_seqs[cell_seq])) {
                                    can_use = false;
                                    break;
                                }
                            }
                        }
                    }
                }

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    scoped_kv_tail_planner_timer timer(
            tail_planner_timing_enabled && !tail_preparing, tail_planner_timing_ns);
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    if (tail && !tail_preparing) {
        if (has_compact_tail()) {
            tail_generations_before_batch = tail_generations;
            tail_graph_started = false;
        }
        tail->begin_batch();
        tail_write_levels = 0;
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            tail_write_levels = std::max(tail_write_levels, uint32_t(ubatch.n_seq_id[i]));
        }
        GGML_ASSERT(tail_write_levels <= n_seq_max);
        tail_write_slots.assign(uint64_t(ubatch.n_tokens)*tail_write_levels, LLAMA_KV_TAIL_BODY_SLOT);
    }

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (tail && !tail_preparing) {
                const uint64_t generation = ++tail_generations[sinfo.strm[s]][idx];
                tail->recycle(sinfo.strm[s], idx, generation);
            }
            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
                cells.ext_set(idx, ext);
            }

            for (int32_t iseq = 0; iseq < ubatch.n_seq_id[i]; iseq++) {
                const llama_seq_id owner = ubatch.seq_id[i][iseq];
                cells.seq_add(idx, owner);
                if (allocation_group_size > 1 && n_stream == 1) {
                    allocation_seq_heads[size_t(owner)] = idx + 1;
                }
                if (tail && !tail_preparing) {
                    const int32_t slot = tail->commit(
                            owner,
                            { uint32_t(sinfo.strm[s]), idx, tail_generations[sinfo.strm[s]][idx] },
                            ubatch.pos[i], tail_ordinal++, has_compact_tail() ? i : UINT32_MAX);
                    tail_write_slots[uint64_t(iseq)*ubatch.n_tokens + i] = slot;
                }
            }
        }

    }

    if (tail && !tail_preparing) {
        const uint32_t sink_base = tail_arena_stride*n_seq_max;
        for (uint32_t level = 0; level < tail_write_levels; ++level) {
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                int64_t & slot = tail_write_slots[uint64_t(level)*ubatch.n_tokens + i];
                if (slot == LLAMA_KV_TAIL_BODY_SLOT && !has_compact_tail()) {
                    GGML_ASSERT(i < tail_sink_slots);
                    slot = int64_t(sink_base + i);
                }
                GGML_ASSERT((has_compact_tail() && slot == LLAMA_KV_TAIL_BODY_SLOT) ||
                        (slot >= 0 && uint64_t(slot) < tail_slots));
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            // Slot preparation has already committed to this eviction. Avoid a
            // second capability preflight after the cell metadata was mutated.
            GGML_ASSERT(seq_rm_unchecked(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1));
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

void llama_kv_cache::finish_tail_batch(bool success, bool payload_may_be_modified) {
    if (!tail || tail_preparing) {
        return;
    }
    tail->finish_batch(success, payload_may_be_modified);
    if (!success && has_compact_tail() && !tail_generations_before_batch.empty()) {
        tail_generations = tail_generations_before_batch;
    }
    tail_generations_before_batch.clear();
    tail_graph_started = false;
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

std::vector<uint32_t> llama_kv_cache::cells_at(llama_seq_id seq_id, llama_pos p) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];
    return cells.cells_at(seq_id, p);
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

uint32_t llama_kv_cache::get_stream_for_seq(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    return seq_to_stream[seq_id];
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k ? layers[0].k->type : layers[0].k_tail->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v ? layers[0].v->type : layers[0].v_tail->type;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

const llama_kv_cells & llama_kv_cache::get_cells(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return v_cells[seq_to_stream[seq_id]];
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;
    if (!k) {
        return nullptr;
    }

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    GGML_ASSERT(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t n_head_kv     = hparams.n_head_kv(il);
    const uint32_t n_embd_head_k = n_embd_k_gqa / n_head_kv;

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            n_embd_head_k, n_head_kv, n_kv, ns,
            ggml_row_size(k->type, n_embd_head_k),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;
    if (!v) {
        return nullptr;
    }

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // use the head dimension stored in the cache tensor
        const uint32_t n_head_kv     = hparams.n_head_kv(il);
        const uint32_t n_embd_head_v = n_embd_v_gqa / n_head_kv;

        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                n_embd_head_v, n_head_kv, n_kv, ns,
                ggml_row_size(v->type, n_embd_head_v),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;
    if (!k) {
        return nullptr;
    }

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    const int64_t n_embd_gqa = n_embd_head * n_head;
    GGML_ASSERT(n_embd_gqa == k->ne[0]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;
    if (!v) {
        return nullptr;
    }

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        GGML_ASSERT(n_embd_gqa == v->ne[0]);

        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        return ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::cpy_k_with_tail(
        ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs,
        ggml_tensor * tail_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);
    ggml_tensor * body = layers[ikv].k;
    ggml_tensor * shadow = layers[ikv].k_tail;
    if (has_compact_tail() || !shadow || !tail_idxs || !ggml_is_quantized(body->type) ||
            (shadow->type != GGML_TYPE_F16 && shadow->type != GGML_TYPE_BF16) || k_cur->type != GGML_TYPE_F32 ||
            k_idxs->type != GGML_TYPE_I64 || tail_idxs->type != GGML_TYPE_I64) {
        return nullptr;
    }

    const int64_t n_embd = k_cur->ne[0]*k_cur->ne[1];
    const int64_t n_tokens = k_cur->ne[2];
    GGML_ASSERT(n_embd == body->ne[0] && n_embd == shadow->ne[0]);
    GGML_ASSERT(k_idxs->ne[0] == n_tokens && tail_idxs->ne[0] == n_tokens);
    GGML_ASSERT(ggml_row_size(k_cur->type, k_cur->ne[0]) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd, n_tokens, k_cur->nb[2], 0);
    if (body->ne[2] > 1) {
        body = ggml_reshape_2d(ctx, body, n_embd, body->ne[1]*body->ne[2]);
    }

    ggml_tensor * written = ggml_set_rows_with_shadow(
            ctx, body, k_cur, k_idxs, shadow, tail_idxs);
    return ggml_view_4d(ctx, written,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), tail_slots, 1,
            ggml_row_size(written->type, hparams.n_embd_head_k(il)), written->nb[1], written->nb[2], 0);
}

ggml_tensor * llama_kv_cache::cpy_v_with_tail(
        ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs,
        ggml_tensor * tail_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);
    ggml_tensor * body = layers[ikv].v;
    ggml_tensor * shadow = layers[ikv].v_tail;
    if (has_compact_tail() || v_trans || !shadow || !tail_idxs || !ggml_is_quantized(body->type) ||
            (shadow->type != GGML_TYPE_F16 && shadow->type != GGML_TYPE_BF16) || v_cur->type != GGML_TYPE_F32 ||
            v_idxs->type != GGML_TYPE_I64 || tail_idxs->type != GGML_TYPE_I64) {
        return nullptr;
    }

    const int64_t n_embd = v_cur->ne[0]*v_cur->ne[1];
    const int64_t n_tokens = v_cur->ne[2];
    GGML_ASSERT(n_embd == body->ne[0] && n_embd == shadow->ne[0]);
    GGML_ASSERT(v_idxs->ne[0] == n_tokens && tail_idxs->ne[0] == n_tokens);
    GGML_ASSERT(ggml_row_size(v_cur->type, v_cur->ne[0]) == v_cur->nb[1]);

    v_cur = ggml_view_2d(ctx, v_cur, n_embd, n_tokens, v_cur->nb[2], 0);
    if (body->ne[2] > 1) {
        body = ggml_reshape_2d(ctx, body, n_embd, body->ne[1]*body->ne[2]);
    }

    ggml_tensor * written = ggml_set_rows_with_shadow(
            ctx, body, v_cur, v_idxs, shadow, tail_idxs);
    return ggml_view_4d(ctx, written,
            hparams.n_embd_head_v(il), hparams.n_head_kv(il), tail_slots, 1,
            ggml_row_size(written->type, hparams.n_embd_head_v(il)), written->nb[1], written->nb[2], 0);
}

bool llama_kv_cache::get_kv_tail_coverage(
        uint32_t group_index, llama_seq_id seq_id, llama_kv_tail_coverage_info & out) const {
    if (group_index != 0 || seq_id < 0 || size_t(seq_id) >= seq_to_stream.size()) {
        return false;
    }
    const auto & cells = v_cells[seq_to_stream[seq_id]];
    const uint32_t available = cells.seq_size(seq_id);
    if (tail_plan.kind == LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT) {
        const uint32_t exact = std::min(tail_plan.effective_tokens, available);
        out = {
            exact > 0 ? LLAMA_KV_TAIL_COVERAGE_COMPLETE : LLAMA_KV_TAIL_COVERAGE_NONE,
            exact,
            exact,
            0,
        };
        return true;
    }
    if (tail_plan.kind == LLAMA_KV_TAIL_STORAGE_DISABLED) {
        out = { LLAMA_KV_TAIL_COVERAGE_NONE, 0, 0, 0 };
        return true;
    }
    if (!tail_plan.graph_consumes_exact_tail) {
        return false;
    }
    GGML_ASSERT(tail);
    const auto coverage = tail->coverage(seq_id, available);
    out = { coverage.state, coverage.requested, coverage.exact, coverage.degradation_flags };
    return true;
}

void llama_kv_cache::reset_kv_tail_planner_timing() {
    tail_planner_timing_ns.store(0, std::memory_order_relaxed);
}

uint64_t llama_kv_cache::get_kv_tail_planner_timing_ns() const {
    return tail_planner_timing_ns.load(std::memory_order_relaxed);
}

ggml_tensor * llama_kv_cache::get_k_tail(ggml_context * ctx, int32_t il) const {
    const auto * tensor = layers[map_layer_ids.at(il)].k_tail;
    if (!tensor) {
        return nullptr;
    }
    const uint32_t n_head = hparams.n_head_kv(il);
    return ggml_view_4d(ctx, const_cast<ggml_tensor *>(tensor),
            hparams.n_embd_head_k(il), n_head, tail_slots, 1,
            ggml_row_size(tensor->type, hparams.n_embd_head_k(il)), tensor->nb[1], tensor->nb[2], 0);
}

ggml_tensor * llama_kv_cache::get_v_tail(ggml_context * ctx, int32_t il) const {
    const auto * tensor = layers[map_layer_ids.at(il)].v_tail;
    if (!tensor) {
        return nullptr;
    }
    const uint32_t n_head = hparams.n_head_kv(il);
    return ggml_view_4d(ctx, const_cast<ggml_tensor *>(tensor),
            hparams.n_embd_head_v(il), n_head, tail_slots, 1,
            ggml_row_size(tensor->type, hparams.n_embd_head_v(il)), tensor->nb[1], tensor->nb[2], 0);
}

ggml_tensor * llama_kv_cache::get_k_tail_fallback(
        ggml_context * ctx, int32_t il, ggml_tensor * body_idxs) const {
    ggml_tensor * storage = layers[map_layer_ids.at(il)].k;
    storage = ggml_reshape_2d(ctx, storage, storage->ne[0], storage->ne[1]*storage->ne[2]);
    ggml_tensor * flat_idxs = ggml_reshape_1d(ctx, body_idxs, body_idxs->ne[0]*body_idxs->ne[1]);
    ggml_tensor * rows = ggml_get_rows(ctx, storage, flat_idxs);
    return ggml_reshape_4d(ctx, rows,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), body_idxs->ne[0], body_idxs->ne[1]);
}

ggml_tensor * llama_kv_cache::get_v_tail_fallback(
        ggml_context * ctx, int32_t il, ggml_tensor * body_idxs) const {
    GGML_ASSERT(!v_trans && "one-sided KV tail fallback requires ordinary non-transposed V storage");
    ggml_tensor * storage = layers[map_layer_ids.at(il)].v;
    storage = ggml_reshape_2d(ctx, storage, storage->ne[0], storage->ne[1]*storage->ne[2]);
    ggml_tensor * flat_idxs = ggml_reshape_1d(ctx, body_idxs, body_idxs->ne[0]*body_idxs->ne[1]);
    ggml_tensor * rows = ggml_get_rows(ctx, storage, flat_idxs);
    return ggml_reshape_4d(ctx, rows,
            hparams.n_embd_head_v(il), hparams.n_head_kv(il), body_idxs->ne[0], body_idxs->ne[1]);
}

ggml_tensor * llama_kv_cache::cpy_k_tail(
        ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * tail_idxs,
        int32_t il, ggml_tensor * dependency) const {
    const int32_t ikv = map_layer_ids.at(il);
    ggml_tensor * dst = layers[ikv].k_tail;
    if (!dst || !tail_idxs) {
        return nullptr;
    }
    const int64_t n_embd = k_cur->ne[0]*k_cur->ne[1];
    const int64_t n_tokens = k_cur->ne[2];
    GGML_ASSERT(n_embd == dst->ne[0]);
    GGML_ASSERT(tail_idxs->ne[0] == n_tokens);
    k_cur = ggml_is_contiguous(k_cur)
        ? ggml_reshape_2d(ctx, k_cur, n_embd, n_tokens)
        : ggml_cont_2d(ctx, k_cur, n_embd, n_tokens);
    ggml_tensor * written = dst;
    for (int64_t level = 0; level < tail_idxs->ne[1]; ++level) {
        ggml_tensor * level_idxs = tail_idxs->ne[1] == 1
            ? tail_idxs
            : ggml_view_1d(ctx, tail_idxs, n_tokens, level*tail_idxs->nb[1]);
        written = ggml_set_rows_ordered(ctx, written, k_cur, level_idxs, dependency);
        dependency = written;
    }
    return ggml_view_4d(ctx, written,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), tail_slots, 1,
            ggml_row_size(written->type, hparams.n_embd_head_k(il)), written->nb[1], written->nb[2], 0);
}

ggml_tensor * llama_kv_cache::cpy_v_tail(
        ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * tail_idxs,
        int32_t il, ggml_tensor * dependency) const {
    const int32_t ikv = map_layer_ids.at(il);
    ggml_tensor * dst = layers[ikv].v_tail;
    if (!dst || !tail_idxs) {
        return nullptr;
    }
    const int64_t n_embd = v_cur->ne[0]*v_cur->ne[1];
    const int64_t n_tokens = v_cur->ne[2];
    GGML_ASSERT(n_embd <= dst->ne[0]);
    GGML_ASSERT(tail_idxs->ne[0] == n_tokens);
    v_cur = ggml_is_contiguous(v_cur)
        ? ggml_reshape_2d(ctx, v_cur, n_embd, n_tokens)
        : ggml_cont_2d(ctx, v_cur, n_embd, n_tokens);
    if (n_embd < dst->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, dst->ne[0] - n_embd, 0, 0, 0);
    }
    ggml_tensor * written = dst;
    for (int64_t level = 0; level < tail_idxs->ne[1]; ++level) {
        ggml_tensor * level_idxs = tail_idxs->ne[1] == 1
            ? tail_idxs
            : ggml_view_1d(ctx, tail_idxs, n_tokens, level*tail_idxs->nb[1]);
        written = ggml_set_rows_ordered(ctx, written, v_cur, level_idxs, dependency);
        dependency = written;
    }
    return ggml_view_4d(ctx, written,
            hparams.n_embd_head_v(il), hparams.n_head_kv(il), tail_slots, 1,
            ggml_row_size(written->type, hparams.n_embd_head_v(il)), written->nb[1], written->nb[2], 0);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_tail_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    if (!tail) {
        return nullptr;
    }
    uint32_t n_levels = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        n_levels = std::max(n_levels, uint32_t(ubatch.n_seq_id[i]));
    }
    ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, ubatch.n_tokens, n_levels);
    ggml_set_input(result);
    return result;
}

ggml_tensor * llama_kv_cache::build_input_tail_body_idxs(ggml_context * ctx) const {
    if (!tail) {
        return nullptr;
    }
    ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tail_slots);
    ggml_set_input(result);
    return result;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        int nrot = 64;

        // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        do {
            nrot *= 2;
        } while (n_embd_head_k_all % nrot == 0);
        nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::set_input_k_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    std::vector<int64_t> data(n_tokens);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }

    ggml_backend_tensor_set(dst, data.data(), 0, data.size()*sizeof(int64_t));
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_tail_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (!dst) {
        return;
    }
    scoped_kv_tail_planner_timer timer(tail_planner_timing_enabled, tail_planner_timing_ns);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(uint32_t(dst->ne[1]) == tail_write_levels);
    GGML_ASSERT(tail_write_slots.size() == size_t(ubatch->n_tokens)*tail_write_levels);
    std::memcpy(dst->data, tail_write_slots.data(), tail_write_slots.size()*sizeof(int64_t));
}

void llama_kv_cache::set_input_tail_body_idxs(ggml_tensor * dst) const {
    if (!dst) {
        return;
    }
    scoped_kv_tail_planner_timer timer(tail_planner_timing_enabled, tail_planner_timing_ns);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    const auto indices = tail->body_indices(get_size());
    GGML_ASSERT(indices.size() == size_t(dst->ne[0]));
    std::memcpy(dst->data, indices.data(), indices.size()*sizeof(int32_t));
}

void llama_kv_cache::set_input_v_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    std::vector<int64_t> data(ggml_nelements(dst));

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        const int64_t kv_size = get_size();
        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }

    ggml_backend_tensor_set(dst, data.data(), 0, data.size()*sizeof(int64_t));
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
    const std::vector<int64_t> * read_cells;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                uint32_t cell;
                const int64_t mapped = args.read_cells ? args.read_cells->at(j) : int64_t(j);
                if (mapped < 0) {
                    goto skip;
                }
                cell = uint32_t(mapped);
                if (cell >= cells.size() || cells.is_empty(cell)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(cell, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(cell);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(cell);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    set_input_kq_mask_mapped(dst, ubatch, causal_attn, {});
}

void llama_kv_cache::set_input_kq_mask_mapped(
        ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn,
        const std::vector<int64_t> & read_cells) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch
    GGML_ASSERT(read_cells.empty() || (n_stream == 1 && int64_t(read_cells.size()) == n_kv));

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
        /*.read_cells       =*/ read_cells.empty() ? nullptr : &read_cells,
    };

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_k_rot_backend(ggml_tensor * dst) const {
    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    ggml_backend_tensor_set(dst, attn_rot_hadamard.at(n_rot).data(), 0, ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot_backend(ggml_tensor * dst) const {
    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    ggml_backend_tensor_set(dst, attn_rot_hadamard.at(n_rot).data(), 0, ggml_nbytes(dst));
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += layer.k ? ggml_nbytes(layer.k) : 0;
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]
    ggml_tensor * k_shift_tail = nullptr; // I32 [tail_slots]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_shift_tail) {
        kv_self->set_input_k_shift_tail(k_shift_tail);
    }

    if (k_rot) {
        kv_self->set_input_k_rot(k_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    if (tail) {
        inp->k_shift_tail = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tail_slots);
        ggml_set_input(inp->k_shift_tail);
    }

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        if (!hparams.has_rope(il)) {
            continue;
        }

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        if (!layer.k) {
            GGML_ASSERT(layer.k_tail);
            ggml_tensor * k_tail = ggml_view_3d(ctx, layer.k_tail,
                    n_rot, n_head_kv, tail_slots,
                    ggml_row_size(layer.k_tail->type, n_embd_head_k),
                    ggml_row_size(layer.k_tail->type, n_embd_k_gqa),
                    ggml_row_size(layer.k_tail->type, n_embd_nope));
            ggml_tensor * tail_cur = build_rope_shift(cparams, ctx, k_tail, inp->k_shift_tail,
                    inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);
            ggml_build_forward_expand(gf, tail_cur);
            continue;
        }

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);

        if (layer.k_tail) {
            ggml_tensor * k_tail = ggml_view_3d(ctx, layer.k_tail,
                    n_rot, n_head_kv, tail_slots,
                    ggml_row_size(layer.k_tail->type, n_embd_head_k),
                    ggml_row_size(layer.k_tail->type, n_embd_k_gqa),
                    ggml_row_size(layer.k_tail->type, n_embd_nope));
            ggml_tensor * tail_cur = build_rope_shift(cparams, ctx, k_tail, inp->k_shift_tail,
                    inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);
            ggml_build_forward_expand(gf, tail_cur);
        }
    }

    res->add_input(std::move(inp));

    return gf;
}

namespace {

constexpr uint32_t LLAMA_KV_TAIL_STATE_MAGIC = 0x4c54564b; // KVTL
constexpr uint32_t LLAMA_KV_TAIL_STATE_VERSION_V1 = 1;
constexpr uint32_t LLAMA_KV_TAIL_STATE_VERSION_V2 = 2;
constexpr uint32_t LLAMA_KV_TAIL_STATE_VERSION_V3 = 3;
constexpr uint32_t LLAMA_KV_TAIL_STATE_VERSION_V4 = 4;
constexpr uint32_t LLAMA_KV_TAIL_STATE_VERSION = 5;
constexpr uint32_t LLAMA_KV_TAIL_STATE_BODY_ONLY = 1;

class llama_io_write_counter final : public llama_io_write_i {
public:
    explicit llama_io_write_counter(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void *, size_t size) override { count += size; }
    void write_tensor(ggml_tensor *, size_t, size_t size) override {
        if (!skip_tensors) {
            count += size;
        }
    }
    size_t n_bytes() override { return count; }

private:
    bool skip_tensors;
    size_t count = 0;
};

}

struct llama_kv_cache::state_v2_manifest {
    struct cell {
        llama_pos pos = 0;
        llama_kv_cell_ext ext {};
        std::vector<llama_seq_id> seq_ids;
        uint32_t source_cell = 0;
        uint64_t generation = 0;
    };
    struct stream {
        std::vector<cell> cells;
        std::vector<uint32_t> payload_runs;
    };
    struct body_layer {
        uint32_t il = 0;
        int32_t k_type = GGML_TYPE_COUNT;
        uint64_t k_row = 0;
        uint32_t has_v = 0;
        int32_t v_type = GGML_TYPE_COUNT;
        uint64_t v_unit = 0;
        uint32_t v_embd = 0;
    };
    struct tail_record {
        llama_seq_id seq_id = -1;
        uint32_t stream = 0;
        uint32_t body_ordinal = 0;
        uint64_t generation = 0;
        llama_pos position = 0;
        uint64_t insertion_ordinal = 0;
        uint32_t payload = 0;
        uint32_t local_slot = UINT32_MAX;
    };
    struct tail_layer {
        uint32_t il = 0;
        uint32_t has_k = 0;
        uint32_t has_v = 0;
        uint64_t k_row = 0;
        uint64_t v_row = 0;
    };

    uint32_t v_trans = 0;
    uint32_t n_pos_per_embd = 0;
    uint32_t saved_n_seq_max = 0;
    bool body_only = false; // framing property, not serialized in the manifest body
    std::vector<stream> streams;
    std::vector<body_layer> body_layers;
    uint64_t tail_ordinal = 0;
    uint32_t tail_payload_count = 0;
    std::vector<tail_record> tail_records;
    std::vector<tail_layer> tail_layers;
    std::vector<llama_kv_tail_provenance> provenance;
    std::vector<uint32_t> tail_write_cursors;
    std::vector<int32_t> tail_payload_slots; // writer-only, never serialized
};

llama_kv_cache::state_v2_manifest llama_kv_cache::state_v2_collect(
        llama_seq_id seq_id, bool body_only) const {
    state_v2_manifest result;
    result.body_only = body_only;
    result.v_trans = v_trans ? 1u : 0u;
    result.n_pos_per_embd = hparams.n_pos_per_embd();
    result.saved_n_seq_max = n_seq_max;
    result.streams.resize(n_stream);

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        auto & saved = result.streams[s];
        uint32_t previous = UINT32_MAX;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool include = !cells.is_empty(i) && (seq_id == -1 || cells.seq_has(i, seq_id));
            if (include && seq_id != -1) {
                include = !llama_hparams::is_masked_swa(
                        n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));
            }
            if (!include) {
                continue;
            }
            state_v2_manifest::cell cell;
            cell.pos = cells.pos_get(i);
            if (result.n_pos_per_embd > 1) {
                cell.ext = cells.ext_get(i);
            }
            for (llama_seq_id current = 0; current < int32_t(n_seq_max); ++current) {
                if ((seq_id == -1 || current == seq_id) && cells.seq_has(i, current)) {
                    cell.seq_ids.push_back(current);
                }
            }
            cell.source_cell = i;
            if (tail) {
                cell.generation = tail_generations[s][i];
            }
            saved.cells.push_back(std::move(cell));
            if (previous == UINT32_MAX || i != previous + 1) {
                saved.payload_runs.push_back(1);
            } else {
                ++saved.payload_runs.back();
            }
            previous = i;
        }
    }

    if (has_kv_body()) {
        result.body_layers.reserve(layers.size());
        for (const auto & layer : layers) {
            const auto * k = layer.k_stream[0];
            const auto * v = layer.v_stream[0];
            state_v2_manifest::body_layer saved;
            saved.il = layer.il;
            saved.k_type = int32_t(k->type);
            saved.k_row = ggml_row_size(k->type, hparams.n_embd_k_gqa(layer.il));
            saved.has_v = v ? 1u : 0u;
            if (v) {
                saved.v_type = int32_t(v->type);
                saved.v_unit = v_trans ? ggml_type_size(v->type) :
                        ggml_row_size(v->type, hparams.n_embd_v_gqa(layer.il));
                saved.v_embd = v_trans ? hparams.n_embd_v_gqa(layer.il) : 0;
            }
            result.body_layers.push_back(saved);
        }
    }

    if (body_only || !has_tail_overlay()) {
        return result;
    }

    GGML_ASSERT(tail);
    result.tail_ordinal = tail_ordinal;
    result.tail_payload_slots = state_tail_payload_slots(seq_id);
    result.tail_payload_count = uint32_t(result.tail_payload_slots.size());
    std::unordered_map<int32_t, uint32_t> payload_by_slot;
    for (uint32_t i = 0; i < result.tail_payload_slots.size(); ++i) {
        payload_by_slot.emplace(result.tail_payload_slots[i], i);
    }
    for (const auto & entry : tail->snapshot(seq_id)) {
        if (entry.identity.stream >= result.streams.size()) {
            continue;
        }
        const auto & body = result.streams[entry.identity.stream].cells;
        const auto found = std::find_if(body.begin(), body.end(), [&](const state_v2_manifest::cell & cell) {
            return cell.source_cell == entry.identity.cell;
        });
        if (found == body.end()) {
            continue;
        }
        const auto payload = payload_by_slot.find(entry.slot);
        if (payload == payload_by_slot.end()) {
            throw std::logic_error("KV tail snapshot payload is missing from the manifest");
        }
        result.tail_records.push_back({
                entry.seq_id,
                entry.identity.stream,
                uint32_t(std::distance(body.begin(), found)),
                entry.identity.generation,
                entry.position,
                entry.insertion_ordinal,
                payload->second,
                uint32_t(entry.slot)%tail_arena_stride });
    }
    result.tail_layers.reserve(layers.size());
    for (const auto & layer : layers) {
        result.tail_layers.push_back({
                uint32_t(layer.il),
                layer.k_tail ? 1u : 0u,
                layer.v_tail ? 1u : 0u,
                layer.k_tail ? uint64_t(ggml_row_size(layer.k_tail->type, layer.k_tail->ne[0])) : 0,
                layer.v_tail ? uint64_t(ggml_row_size(layer.v_tail->type, layer.v_tail->ne[0])) : 0 });
    }
    result.provenance = tail->snapshot_provenance(seq_id);
    result.tail_write_cursors.reserve(result.provenance.size());
    for (const auto & provenance : result.provenance) {
        result.tail_write_cursors.push_back(tail->state_write_cursor(provenance.seq_id));
    }
    return result;
}

void llama_kv_cache::state_v2_write_manifest(
        llama_io_write_i & io, const state_v2_manifest & manifest) const {
    const uint32_t stream_count = uint32_t(manifest.streams.size());
    const uint32_t layer_count = uint32_t(manifest.body_layers.size());
    io.write(&stream_count, sizeof(stream_count));
    io.write(&manifest.v_trans, sizeof(manifest.v_trans));
    io.write(&manifest.n_pos_per_embd, sizeof(manifest.n_pos_per_embd));
    io.write(&manifest.saved_n_seq_max, sizeof(manifest.saved_n_seq_max));
    io.write(&layer_count, sizeof(layer_count));
    for (const auto & layer : manifest.body_layers) {
        io.write(&layer.il, sizeof(layer.il));
        io.write(&layer.k_type, sizeof(layer.k_type));
        io.write(&layer.k_row, sizeof(layer.k_row));
        io.write(&layer.has_v, sizeof(layer.has_v));
        io.write(&layer.v_type, sizeof(layer.v_type));
        io.write(&layer.v_unit, sizeof(layer.v_unit));
        io.write(&layer.v_embd, sizeof(layer.v_embd));
    }
    for (const auto & stream : manifest.streams) {
        const uint32_t cell_count = uint32_t(stream.cells.size());
        const uint32_t run_count = uint32_t(stream.payload_runs.size());
        io.write(&cell_count, sizeof(cell_count));
        io.write(&run_count, sizeof(run_count));
        for (uint32_t run : stream.payload_runs) {
            io.write(&run, sizeof(run));
        }
        for (const auto & cell : stream.cells) {
            io.write(&cell.source_cell, sizeof(cell.source_cell));
            io.write(&cell.generation, sizeof(cell.generation));
            io.write(&cell.pos, sizeof(cell.pos));
            if (manifest.n_pos_per_embd > 1) {
                io.write(&cell.ext, sizeof(cell.ext));
            }
            const uint32_t seq_count = uint32_t(cell.seq_ids.size());
            io.write(&seq_count, sizeof(seq_count));
            for (llama_seq_id saved_seq : cell.seq_ids) {
                io.write(&saved_seq, sizeof(saved_seq));
            }
        }
    }

    const uint32_t record_count = uint32_t(manifest.tail_records.size());
    const uint32_t tail_layer_count = uint32_t(manifest.tail_layers.size());
    const uint32_t provenance_count = uint32_t(manifest.provenance.size());
    io.write(&manifest.tail_ordinal, sizeof(manifest.tail_ordinal));
    io.write(&record_count, sizeof(record_count));
    io.write(&manifest.tail_payload_count, sizeof(manifest.tail_payload_count));
    io.write(&tail_layer_count, sizeof(tail_layer_count));
    io.write(&provenance_count, sizeof(provenance_count));
    for (const auto & record : manifest.tail_records) {
        io.write(&record.seq_id, sizeof(record.seq_id));
        io.write(&record.stream, sizeof(record.stream));
        io.write(&record.body_ordinal, sizeof(record.body_ordinal));
        io.write(&record.generation, sizeof(record.generation));
        io.write(&record.position, sizeof(record.position));
        io.write(&record.insertion_ordinal, sizeof(record.insertion_ordinal));
        io.write(&record.payload, sizeof(record.payload));
        io.write(&record.local_slot, sizeof(record.local_slot));
    }
    for (const auto & layer : manifest.tail_layers) {
        io.write(&layer.il, sizeof(layer.il));
        io.write(&layer.has_k, sizeof(layer.has_k));
        io.write(&layer.has_v, sizeof(layer.has_v));
        io.write(&layer.k_row, sizeof(layer.k_row));
        io.write(&layer.v_row, sizeof(layer.v_row));
    }
    for (const auto & provenance : manifest.provenance) {
        io.write(&provenance.seq_id, sizeof(provenance.seq_id));
        io.write(&provenance.degradation_flags, sizeof(provenance.degradation_flags));
        io.write(&provenance.recovery_commits, sizeof(provenance.recovery_commits));
    }
    GGML_ASSERT(manifest.tail_write_cursors.size() == manifest.provenance.size());
    for (uint32_t cursor : manifest.tail_write_cursors) {
        io.write(&cursor, sizeof(cursor));
    }
}

llama_kv_cache::state_v2_manifest llama_kv_cache::state_v2_read_manifest(
        llama_io_read_i & io, llama_seq_id seq_id, bool body_only, uint32_t version) const {
    state_v2_manifest result;
    result.body_only = body_only;
    uint32_t stream_count;
    uint32_t layer_count;
    io.read(&stream_count, sizeof(stream_count));
    io.read(&result.v_trans, sizeof(result.v_trans));
    io.read(&result.n_pos_per_embd, sizeof(result.n_pos_per_embd));
    io.read(&result.saved_n_seq_max, sizeof(result.saved_n_seq_max));
    io.read(&layer_count, sizeof(layer_count));
    const uint32_t expected_body_layer_count = has_kv_body() ? uint32_t(layers.size()) : 0u;
    if (stream_count != n_stream || layer_count != expected_body_layer_count ||
            result.v_trans != uint32_t(v_trans) || result.n_pos_per_embd != hparams.n_pos_per_embd()) {
        throw std::runtime_error("KV tail state body layout does not match the context");
    }
    if (result.saved_n_seq_max == 0 || (seq_id == -1 && result.saved_n_seq_max > n_seq_max)) {
        throw std::runtime_error("KV tail state sequence capacity does not fit the context");
    }
    result.body_layers.resize(layer_count);
    for (uint32_t i = 0; i < layer_count; ++i) {
        auto & saved = result.body_layers[i];
        io.read(&saved.il, sizeof(saved.il));
        io.read(&saved.k_type, sizeof(saved.k_type));
        io.read(&saved.k_row, sizeof(saved.k_row));
        io.read(&saved.has_v, sizeof(saved.has_v));
        io.read(&saved.v_type, sizeof(saved.v_type));
        io.read(&saved.v_unit, sizeof(saved.v_unit));
        io.read(&saved.v_embd, sizeof(saved.v_embd));
        const auto & current = layers[i];
        const auto * k = current.k_stream[0];
        const auto * v = current.v_stream[0];
        const uint64_t expected_v_unit = !v ? 0 : v_trans ? ggml_type_size(v->type) :
                ggml_row_size(v->type, hparams.n_embd_v_gqa(current.il));
        const uint32_t expected_v_embd = v && v_trans ? hparams.n_embd_v_gqa(current.il) : 0;
        if (saved.il != uint32_t(current.il) || saved.k_type != int32_t(k->type) ||
                saved.k_row != ggml_row_size(k->type, hparams.n_embd_k_gqa(current.il)) ||
                saved.has_v != uint32_t(v != nullptr) ||
                saved.v_type != (v ? int32_t(v->type) : int32_t(GGML_TYPE_COUNT)) ||
                saved.v_unit != expected_v_unit || saved.v_embd != expected_v_embd) {
            throw std::runtime_error("KV tail state body layer layout mismatch");
        }
    }

    result.streams.resize(stream_count);
    uint64_t per_sequence_cells = 0;
    uint32_t nonempty_streams = 0;
    for (uint32_t s = 0; s < stream_count; ++s) {
        uint32_t cell_count;
        uint32_t run_count;
        io.read(&cell_count, sizeof(cell_count));
        io.read(&run_count, sizeof(run_count));
        if (cell_count > v_cells[s].size() || run_count > cell_count ||
                (cell_count == 0) != (run_count == 0)) {
            throw std::runtime_error("invalid KV tail state body dimensions");
        }
        if (cell_count > 0) {
            ++nonempty_streams;
        }
        per_sequence_cells += cell_count;
        auto & stream = result.streams[s];
        stream.payload_runs.resize(run_count);
        uint64_t run_total = 0;
        for (uint32_t & run : stream.payload_runs) {
            io.read(&run, sizeof(run));
            if (run == 0 || run_total + run > cell_count) {
                throw std::runtime_error("invalid KV tail state body payload run");
            }
            run_total += run;
        }
        if (run_total != cell_count) {
            throw std::runtime_error("KV tail state body payload runs do not cover the cells");
        }
        stream.cells.resize(cell_count);
        for (auto & cell : stream.cells) {
            if (version >= LLAMA_KV_TAIL_STATE_VERSION_V4) {
                io.read(&cell.source_cell, sizeof(cell.source_cell));
                io.read(&cell.generation, sizeof(cell.generation));
                if (cell.source_cell >= v_cells[s].size()) {
                    throw std::runtime_error("invalid KV tail state source cell");
                }
            }
            io.read(&cell.pos, sizeof(cell.pos));
            if (result.n_pos_per_embd > 1) {
                io.read(&cell.ext, sizeof(cell.ext));
            }
            uint32_t seq_count;
            io.read(&seq_count, sizeof(seq_count));
            if (seq_count == 0 || seq_count > result.saved_n_seq_max || (seq_id >= 0 && seq_count != 1)) {
                throw std::runtime_error("invalid KV tail state cell sequence count");
            }
            cell.seq_ids.resize(seq_count);
            std::set<llama_seq_id> unique;
            for (llama_seq_id & saved_seq : cell.seq_ids) {
                io.read(&saved_seq, sizeof(saved_seq));
                if (saved_seq < 0 || uint32_t(saved_seq) >= result.saved_n_seq_max ||
                        (seq_id == -1 && uint32_t(saved_seq) >= n_seq_max) || !unique.insert(saved_seq).second) {
                    throw std::runtime_error("invalid KV tail state cell sequence ID");
                }
            }
        }
    }
    if (seq_id >= 0 && (nonempty_streams > 1 ||
            per_sequence_cells > v_cells[seq_to_stream.at(seq_id)].size())) {
        throw std::runtime_error("KV tail state sequence body does not fit the destination stream");
    }

    uint32_t record_count;
    uint32_t tail_layer_count;
    uint32_t provenance_count;
    io.read(&result.tail_ordinal, sizeof(result.tail_ordinal));
    io.read(&record_count, sizeof(record_count));
    io.read(&result.tail_payload_count, sizeof(result.tail_payload_count));
    io.read(&tail_layer_count, sizeof(tail_layer_count));
    io.read(&provenance_count, sizeof(provenance_count));
    const uint64_t max_records = uint64_t(result.tail_payload_count)*result.saved_n_seq_max;
    if (body_only) {
        if (record_count != 0 || result.tail_payload_count != 0 ||
                tail_layer_count != 0 || provenance_count != 0) {
            throw std::runtime_error("body-only KV tail state contains exact-tail metadata");
        }
    } else if (!tail || result.tail_payload_count > tail_slots || record_count > max_records ||
            tail_layer_count != layers.size() ||
            provenance_count != (seq_id == -1 ? result.saved_n_seq_max : 1u)) {
        throw std::runtime_error("invalid KV tail state manifest dimensions");
    }

    result.tail_records.resize(record_count);
    std::vector<bool> payload_referenced(result.tail_payload_count, false);
    std::vector<uint32_t> records_per_destination(n_seq_max, 0);
    std::set<std::tuple<llama_seq_id, uint32_t, uint32_t>> unique_identity;
    for (auto & record : result.tail_records) {
        io.read(&record.seq_id, sizeof(record.seq_id));
        io.read(&record.stream, sizeof(record.stream));
        io.read(&record.body_ordinal, sizeof(record.body_ordinal));
        io.read(&record.generation, sizeof(record.generation));
        io.read(&record.position, sizeof(record.position));
        io.read(&record.insertion_ordinal, sizeof(record.insertion_ordinal));
        io.read(&record.payload, sizeof(record.payload));
        if (version >= LLAMA_KV_TAIL_STATE_VERSION) {
            io.read(&record.local_slot, sizeof(record.local_slot));
        }
        if (record.seq_id < 0 || uint32_t(record.seq_id) >= result.saved_n_seq_max ||
                (seq_id == -1 && uint32_t(record.seq_id) >= n_seq_max) ||
                record.stream >= result.streams.size() ||
                record.body_ordinal >= result.streams[record.stream].cells.size() ||
                record.payload >= result.tail_payload_count ||
                (version >= LLAMA_KV_TAIL_STATE_VERSION && record.local_slot >= tail_arena_stride)) {
            throw std::runtime_error("invalid KV tail state identity mapping");
        }
        const auto & body_cell = result.streams[record.stream].cells[record.body_ordinal];
        if (std::find(body_cell.seq_ids.begin(), body_cell.seq_ids.end(), record.seq_id) == body_cell.seq_ids.end()) {
            throw std::runtime_error("KV tail state exact record does not belong to its body cell");
        }
        const llama_seq_id dst_seq = seq_id == -1 ? record.seq_id : seq_id;
        if (!unique_identity.emplace(dst_seq, record.stream, record.body_ordinal).second) {
            throw std::runtime_error("duplicate KV tail state identity");
        }
        // Compact state includes the rollback reserve as persistent history.
        // It is not visible to ordinary attention, but it must survive a
        // checkpoint so suffix rollback remains valid after restore.
        if (++records_per_destination[size_t(dst_seq)] > tail->history_capacity()) {
            throw std::runtime_error("over-capacity KV tail state identity");
        }
        payload_referenced[record.payload] = true;
    }
    if (std::find(payload_referenced.begin(), payload_referenced.end(), false) != payload_referenced.end()) {
        throw std::runtime_error("KV tail state contains an unreferenced payload");
    }

    result.tail_layers.resize(tail_layer_count);
    for (uint32_t i = 0; i < tail_layer_count; ++i) {
        auto & saved = result.tail_layers[i];
        io.read(&saved.il, sizeof(saved.il));
        io.read(&saved.has_k, sizeof(saved.has_k));
        io.read(&saved.has_v, sizeof(saved.has_v));
        io.read(&saved.k_row, sizeof(saved.k_row));
        io.read(&saved.v_row, sizeof(saved.v_row));
        const auto & current = layers[i];
        const uint64_t expected_k = current.k_tail ?
                ggml_row_size(current.k_tail->type, current.k_tail->ne[0]) : 0;
        const uint64_t expected_v = current.v_tail ?
                ggml_row_size(current.v_tail->type, current.v_tail->ne[0]) : 0;
        if (saved.il != uint32_t(current.il) || saved.has_k != uint32_t(current.k_tail != nullptr) ||
                saved.has_v != uint32_t(current.v_tail != nullptr) ||
                saved.k_row != expected_k || saved.v_row != expected_v) {
            throw std::runtime_error("KV tail state exact layer layout mismatch");
        }
    }

    constexpr uint32_t known_flags =
            LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE |
            LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP |
            LLAMA_KV_TAIL_DEGRADED_STATE_RESTORE;
    result.provenance.resize(provenance_count);
    std::vector<bool> provenance_seen(result.saved_n_seq_max, false);
    for (auto & saved : result.provenance) {
        io.read(&saved.seq_id, sizeof(saved.seq_id));
        io.read(&saved.degradation_flags, sizeof(saved.degradation_flags));
        io.read(&saved.recovery_commits, sizeof(saved.recovery_commits));
        if (saved.seq_id < 0 || uint32_t(saved.seq_id) >= result.saved_n_seq_max ||
                (seq_id == -1 && uint32_t(saved.seq_id) >= n_seq_max) ||
                provenance_seen[size_t(saved.seq_id)] ||
                (saved.degradation_flags & ~known_flags) != 0 ||
                saved.recovery_commits > tail->retention() ||
                (saved.degradation_flags == 0 && saved.recovery_commits != 0)) {
            throw std::runtime_error("invalid KV tail state degradation provenance");
        }
        provenance_seen[size_t(saved.seq_id)] = true;
    }
    if (seq_id == -1 && !body_only &&
            std::find(provenance_seen.begin(), provenance_seen.end(), false) != provenance_seen.end()) {
        throw std::runtime_error("incomplete KV tail state degradation provenance");
    }
    if (version >= LLAMA_KV_TAIL_STATE_VERSION) {
        result.tail_write_cursors.resize(provenance_count);
        for (uint32_t & cursor : result.tail_write_cursors) {
            io.read(&cursor, sizeof(cursor));
            if (cursor >= tail_arena_stride) {
                throw std::runtime_error("invalid KV tail state write cursor");
            }
        }
    }
    return result;
}

void llama_kv_cache::state_v2_write_body_payload(
        llama_io_write_i & io, const state_v2_manifest & manifest) const {
    for (uint32_t s = 0; s < manifest.streams.size(); ++s) {
        const auto & stream = manifest.streams[s];
        for (uint32_t l = 0; l < manifest.body_layers.size(); ++l) {
            auto * k = layers[l].k_stream[s];
            const uint64_t row = manifest.body_layers[l].k_row;
            size_t ordinal = 0;
            for (uint32_t run : stream.payload_runs) {
                io.write_tensor(k, size_t(stream.cells[ordinal].source_cell)*row, size_t(run)*row);
                ordinal += run;
            }
        }
        if (!v_trans) {
            for (uint32_t l = 0; l < manifest.body_layers.size(); ++l) {
                auto * v = layers[l].v_stream[s];
                if (!v) {
                    continue;
                }
                const uint64_t row = manifest.body_layers[l].v_unit;
                size_t ordinal = 0;
                for (uint32_t run : stream.payload_runs) {
                    io.write_tensor(v, size_t(stream.cells[ordinal].source_cell)*row, size_t(run)*row);
                    ordinal += run;
                }
            }
        } else {
            const uint32_t kv_size = v_cells[s].size();
            for (uint32_t l = 0; l < manifest.body_layers.size(); ++l) {
                auto * v = layers[l].v_stream[s];
                if (!v) {
                    continue;
                }
                const uint64_t element = manifest.body_layers[l].v_unit;
                for (uint32_t j = 0; j < manifest.body_layers[l].v_embd; ++j) {
                    size_t ordinal = 0;
                    for (uint32_t run : stream.payload_runs) {
                        const size_t offset = (size_t(stream.cells[ordinal].source_cell) + size_t(j)*kv_size)*element;
                        io.write_tensor(v, offset, size_t(run)*element);
                        ordinal += run;
                    }
                }
            }
        }
    }
}

void llama_kv_cache::state_v2_write_tail_payload(
        llama_io_write_i & io, const state_v2_manifest & manifest) const {
    for (uint32_t l = 0; l < manifest.tail_layers.size(); ++l) {
        const auto & saved = manifest.tail_layers[l];
        const auto & layer = layers[l];
        for (int32_t slot : manifest.tail_payload_slots) {
            if (layer.k_tail) {
                io.write_tensor(layer.k_tail, size_t(slot)*saved.k_row, saved.k_row);
            }
            if (layer.v_tail) {
                io.write_tensor(layer.v_tail, size_t(slot)*saved.v_row, saved.v_row);
            }
        }
    }
}

void llama_kv_cache::state_v2_read_payload_and_install(
        llama_io_read_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        state_v2_manifest & manifest,
        uint64_t body_payload_size,
        uint64_t tail_payload_size,
        uint32_t version) {
    struct body_chunk {
        uint32_t stream;
        uint32_t layer;
        uint32_t ordinal;
        uint32_t count;
        uint32_t v_element;
        bool key;
        std::vector<uint8_t> data;
    };
    struct tail_chunk {
        uint32_t layer;
        uint32_t payload;
        bool key;
        std::vector<uint8_t> data;
    };
    const bool on_device = (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) != 0;
    const bool self_contained = (flags & LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED) != 0;
    const bool reference_only_partial =
            (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0 && body_payload_size == 0;
    state_cell_remap.clear();
    std::vector<body_chunk> body_chunks;
    std::vector<tail_chunk> tail_chunks;
    auto checked_bytes = [](uint64_t unit, uint32_t count) -> size_t {
        if (count != 0 && unit > std::numeric_limits<size_t>::max()/count) {
            throw std::runtime_error("KV tail state payload size overflows size_t");
        }
        return size_t(unit)*count;
    };

    const size_t body_begin = io.n_bytes();
    if (!on_device && !reference_only_partial) {
        for (uint32_t s = 0; s < manifest.streams.size(); ++s) {
            const auto & stream = manifest.streams[s];
            for (uint32_t l = 0; l < manifest.body_layers.size(); ++l) {
                uint32_t ordinal = 0;
                for (uint32_t run : stream.payload_runs) {
                    body_chunk chunk { s, l, ordinal, run, 0, true, {} };
                    chunk.data.resize(checked_bytes(manifest.body_layers[l].k_row, run));
                    io.read(chunk.data.data(), chunk.data.size());
                    body_chunks.push_back(std::move(chunk));
                    ordinal += run;
                }
            }
            if (!v_trans) {
                for (uint32_t l = 0; l < manifest.body_layers.size(); ++l) {
                    if (!manifest.body_layers[l].has_v) {
                        continue;
                    }
                    uint32_t ordinal = 0;
                    for (uint32_t run : stream.payload_runs) {
                        body_chunk chunk { s, l, ordinal, run, 0, false, {} };
                        chunk.data.resize(checked_bytes(manifest.body_layers[l].v_unit, run));
                        io.read(chunk.data.data(), chunk.data.size());
                        body_chunks.push_back(std::move(chunk));
                        ordinal += run;
                    }
                }
            } else {
                for (uint32_t l = 0; l < manifest.body_layers.size(); ++l) {
                    if (!manifest.body_layers[l].has_v) {
                        continue;
                    }
                    for (uint32_t j = 0; j < manifest.body_layers[l].v_embd; ++j) {
                        uint32_t ordinal = 0;
                        for (uint32_t run : stream.payload_runs) {
                            body_chunk chunk { s, l, ordinal, run, j, false, {} };
                            chunk.data.resize(checked_bytes(manifest.body_layers[l].v_unit, run));
                            io.read(chunk.data.data(), chunk.data.size());
                            body_chunks.push_back(std::move(chunk));
                            ordinal += run;
                        }
                    }
                }
            }
        }
    }
    if (io.n_bytes() - body_begin != body_payload_size) {
        throw std::runtime_error("invalid KV tail state body payload size");
    }

    const size_t tail_begin = io.n_bytes();
    if (!on_device) {
        for (uint32_t l = 0; l < manifest.tail_layers.size(); ++l) {
            for (uint32_t payload = 0; payload < manifest.tail_payload_count; ++payload) {
                if (manifest.tail_layers[l].has_k) {
                    tail_chunk chunk { l, payload, true, {} };
                    chunk.data.resize(size_t(manifest.tail_layers[l].k_row));
                    io.read(chunk.data.data(), chunk.data.size());
                    tail_chunks.push_back(std::move(chunk));
                }
                if (manifest.tail_layers[l].has_v) {
                    tail_chunk chunk { l, payload, false, {} };
                    chunk.data.resize(size_t(manifest.tail_layers[l].v_row));
                    io.read(chunk.data.data(), chunk.data.size());
                    tail_chunks.push_back(std::move(chunk));
                }
            }
        }
    }
    if (io.n_bytes() - tail_begin != tail_payload_size) {
        throw std::runtime_error("invalid KV tail state exact payload size");
    }

    const uint64_t live_ordinal = tail_ordinal;
    std::vector<std::vector<uint32_t>> restored_cells(n_stream);
    const bool exact_source_cells = version >= LLAMA_KV_TAIL_STATE_VERSION_V4;
    if (seq_id == -1) {
        clear(true);
        for (uint32_t s = 0; s < manifest.streams.size(); ++s) {
            auto & cells = v_cells[s];
            const auto & saved = manifest.streams[s].cells;
            restored_cells[s].resize(saved.size());
            for (uint32_t i = 0; i < saved.size(); ++i) {
                const uint32_t dst_cell = exact_source_cells ? saved[i].source_cell : i;
                if (!cells.is_empty(dst_cell)) {
                    throw std::runtime_error("duplicate KV tail state source cell");
                }
                cells.pos_set(dst_cell, saved[i].pos);
                if (manifest.n_pos_per_embd > 1) {
                    cells.ext_set(dst_cell, saved[i].ext);
                }
                for (llama_seq_id saved_seq : saved[i].seq_ids) {
                    cells.seq_add(dst_cell, saved_seq);
                }
                if (tail) {
                    tail_generations[s][dst_cell] = saved[i].generation;
                }
                restored_cells[s][i] = dst_cell;
            }
            v_heads[s] = 0;
        }
    } else if (exact_source_cells && self_contained) {
        const uint32_t dst_stream = seq_to_stream.at(seq_id);
        auto & cells = v_cells[dst_stream];
        if (!seq_rm(seq_id, -1, -1)) {
            throw std::runtime_error("failed to clear KV state destination sequence");
        }

        std::set<uint32_t> source_groups;
        std::set<uint32_t> source_cells;
        for (const auto & stream : manifest.streams) {
            for (const auto & saved : stream.cells) {
                if (saved.source_cell >= cells.size() || !source_cells.insert(saved.source_cell).second) {
                    throw std::runtime_error("invalid self-contained KV state source cell");
                }
                source_groups.insert(saved.source_cell/state_remap_group_size);
            }
        }

        std::unordered_map<uint32_t, uint32_t> destination_groups;
        std::set<uint32_t> reserved_groups;
        const uint32_t n_groups = cells.size()/state_remap_group_size;
        for (const uint32_t source_group : source_groups) {
            uint32_t destination_group = UINT32_MAX;
            for (uint32_t pass = 0; pass < 2 && destination_group == UINT32_MAX; ++pass) {
                for (uint32_t candidate = 0; candidate < n_groups; ++candidate) {
                    if ((pass == 0 && candidate != source_group) ||
                            (pass == 1 && candidate == source_group) ||
                            reserved_groups.count(candidate) != 0) {
                        continue;
                    }
                    bool empty = true;
                    const uint32_t first = candidate*state_remap_group_size;
                    for (uint32_t offset = 0; offset < state_remap_group_size; ++offset) {
                        if (!cells.is_empty(first + offset)) {
                            empty = false;
                            break;
                        }
                    }
                    if (empty) {
                        destination_group = candidate;
                        break;
                    }
                }
            }
            if (destination_group == UINT32_MAX) {
                throw std::runtime_error("failed to allocate a complete KV state destination group");
            }
            destination_groups.emplace(source_group, destination_group);
            reserved_groups.insert(destination_group);
        }

        for (uint32_t s = 0; s < manifest.streams.size(); ++s) {
            const auto & saved_stream = manifest.streams[s].cells;
            restored_cells[s].reserve(saved_stream.size());
            for (const auto & saved : saved_stream) {
                const uint32_t source_group = saved.source_cell/state_remap_group_size;
                const uint32_t source_offset = saved.source_cell%state_remap_group_size;
                const uint32_t dst_cell = destination_groups.at(source_group)*state_remap_group_size + source_offset;
                if (!cells.is_empty(dst_cell)) {
                    throw std::runtime_error("self-contained KV state destination cell is occupied");
                }
                cells.pos_set(dst_cell, saved.pos);
                if (manifest.n_pos_per_embd > 1) {
                    cells.ext_set(dst_cell, saved.ext);
                }
                cells.seq_add(dst_cell, seq_id);
                if (tail) {
                    tail_generations[dst_stream][dst_cell] = saved.generation;
                }
                restored_cells[s].push_back(dst_cell);
                state_cell_remap.emplace_back(saved.source_cell, dst_cell);
            }
        }
    } else if (exact_source_cells) {
        const uint32_t dst_stream = seq_to_stream.at(seq_id);
        auto & cells = v_cells[dst_stream];
        std::set<uint32_t> destinations;
        std::unordered_map<uint32_t, uint32_t> reference_destinations;
        for (const auto & stream : manifest.streams) {
            for (const auto & saved : stream.cells) {
                uint32_t destination = saved.source_cell;
                if (reference_only_partial) {
                    destination = UINT32_MAX;
                    for (uint32_t candidate = 0; candidate < cells.size(); ++candidate) {
                        if (!cells.seq_has(candidate, seq_id) || cells.pos_get(candidate) != saved.pos ||
                                (manifest.n_pos_per_embd > 1 &&
                                 !(cells.ext_get(candidate).x == saved.ext.x &&
                                   cells.ext_get(candidate).y == saved.ext.y))) {
                            continue;
                        }
                        destination = candidate;
                        break;
                    }
                    if (destination == UINT32_MAX) {
                        throw std::runtime_error(
                                "partial KV state no longer has its self-contained anchor record");
                    }
                    reference_destinations.emplace(saved.source_cell, destination);
                }
                if (!destinations.insert(destination).second) {
                    throw std::runtime_error("duplicate per-sequence KV state source cell");
                }
                // An empty destination is safe to initialize from this self-contained
                // snapshot, including its generation and K/V payload.  An occupied
                // destination must still name the same physical generation: accepting
                // a mismatched occupied cell would turn an ABA reuse into silent state
                // corruption even when its logical position happens to match.
                if (!reference_only_partial && tail && !cells.is_empty(saved.source_cell) &&
                        tail_generations[dst_stream][saved.source_cell] != saved.generation) {
                    throw std::runtime_error(
                            "selective KV state source cell generation is stale");
                }
                if (!reference_only_partial && !cells.is_empty(saved.source_cell) &&
                        (cells.pos_get(saved.source_cell) != saved.pos ||
                        (manifest.n_pos_per_embd > 1 &&
                         !(cells.ext_get(saved.source_cell).x == saved.ext.x &&
                           cells.ext_get(saved.source_cell).y == saved.ext.y)))) {
                    throw std::runtime_error(
                            "selective KV state source cell conflicts with live data");
                }
            }
        }

        seq_rm(seq_id, -1, -1);
        for (uint32_t s = 0; s < manifest.streams.size(); ++s) {
            const auto & saved_stream = manifest.streams[s].cells;
            restored_cells[s].reserve(saved_stream.size());
            for (const auto & saved : saved_stream) {
                const uint32_t dst_cell = reference_only_partial ?
                        reference_destinations.at(saved.source_cell) : saved.source_cell;
                if (cells.is_empty(dst_cell)) {
                    cells.pos_set(dst_cell, saved.pos);
                    if (manifest.n_pos_per_embd > 1) {
                        cells.ext_set(dst_cell, saved.ext);
                    }
                } else if (cells.pos_get(dst_cell) != saved.pos ||
                        (manifest.n_pos_per_embd > 1 &&
                         !(cells.ext_get(dst_cell).x == saved.ext.x &&
                           cells.ext_get(dst_cell).y == saved.ext.y))) {
                    throw std::runtime_error("selective KV state conflicts with another logical sequence");
                }
                cells.seq_add(dst_cell, seq_id);
                if (tail) {
                    tail_generations[dst_stream][dst_cell] = saved.generation;
                }
                restored_cells[s].push_back(dst_cell);
                if (reference_only_partial) {
                    state_cell_remap.emplace_back(saved.source_cell, dst_cell);
                }
            }
        }
    } else {
        seq_rm(seq_id, -1, -1);
        const uint32_t dst_stream = seq_to_stream.at(seq_id);
        for (uint32_t s = 0; s < manifest.streams.size(); ++s) {
            const auto & saved = manifest.streams[s].cells;
            if (saved.empty()) {
                continue;
            }
            llama_batch_allocr balloc(hparams.n_pos_per_embd());
            llama_ubatch ubatch = balloc.ubatch_reserve(uint32_t(saved.size()), 1);
            ubatch.seq_id_unq[0] = seq_id;
            for (uint32_t i = 0; i < saved.size(); ++i) {
                ubatch.pos[i] = saved[i].pos;
                ubatch.n_seq_id[i] = 1;
                ubatch.seq_id[i] = &seq_id;
                if (manifest.n_pos_per_embd > 1) {
                    ubatch.pos[i + ubatch.n_tokens] = saved[i].ext.y;
                    ubatch.pos[i + ubatch.n_tokens*2] = saved[i].ext.x;
                }
            }
            const slot_info sinfo = find_slot(ubatch, false);
            if (sinfo.empty() || sinfo.n_stream() != 1 || sinfo.idxs[0].size() != saved.size()) {
                seq_rm(seq_id, -1, -1);
                throw std::runtime_error("failed to allocate KV tail state destination cells");
            }
            struct tail_preparing_guard {
                bool & value;
                bool old;
                ~tail_preparing_guard() { value = old; }
            } guard { tail_preparing, tail_preparing };
            tail_preparing = true;
            apply_ubatch(sinfo, ubatch);
            restored_cells[s].assign(sinfo.idxs[0].begin(), sinfo.idxs[0].end());
            GGML_ASSERT(sinfo.strm[0] == int32_t(dst_stream));
        }
    }

    auto body_tensor = [&](const body_chunk & chunk) -> ggml_tensor * {
        const uint32_t dst_stream = seq_id == -1 ? chunk.stream : seq_to_stream.at(seq_id);
        return chunk.key ? layers[chunk.layer].k_stream[dst_stream] : layers[chunk.layer].v_stream[dst_stream];
    };
    auto body_offset = [&](const body_chunk & chunk, uint32_t i) -> size_t {
        const uint32_t dst_stream = seq_id == -1 ? chunk.stream : seq_to_stream.at(seq_id);
        const uint32_t cell = restored_cells[chunk.stream][chunk.ordinal + i];
        if (chunk.key) {
            return size_t(cell)*manifest.body_layers[chunk.layer].k_row;
        }
        if (!v_trans) {
            return size_t(cell)*manifest.body_layers[chunk.layer].v_unit;
        }
        return (size_t(cell) + size_t(chunk.v_element)*v_cells[dst_stream].size())*
                manifest.body_layers[chunk.layer].v_unit;
    };
    if (!on_device) {
        for (const auto & chunk : body_chunks) {
            const size_t unit = chunk.key ? manifest.body_layers[chunk.layer].k_row :
                    manifest.body_layers[chunk.layer].v_unit;
            for (uint32_t i = 0; i < chunk.count; ++i) {
                io.stage_tensor_set(body_tensor(chunk), chunk.data.data() + size_t(i)*unit,
                        body_offset(chunk, i), unit);
            }
        }
    } else if (!reference_only_partial) {
        for (uint32_t s = 0; s < manifest.streams.size(); ++s) {
            const auto & stream = manifest.streams[s];
            auto read_runs = [&](uint32_t l, bool key, uint32_t v_element, uint64_t unit) {
                uint32_t ordinal = 0;
                for (uint32_t run : stream.payload_runs) {
                    body_chunk chunk { s, l, ordinal, run, v_element, key, {} };
                    const size_t first = body_offset(chunk, 0);
                    for (uint32_t i = 1; i < run; ++i) {
                        if (body_offset(chunk, i) != first + size_t(i)*unit) {
                            seq_rm(seq_id, -1, -1);
                            throw std::runtime_error("on-device KV tail state destination is not contiguous");
                        }
                    }
                    io.read_tensor(body_tensor(chunk), first, checked_bytes(unit, run));
                    ordinal += run;
                }
            };
            for (uint32_t l = 0; l < manifest.body_layers.size(); ++l) {
                read_runs(l, true, 0, manifest.body_layers[l].k_row);
            }
            if (!v_trans) {
                for (uint32_t l = 0; l < manifest.body_layers.size(); ++l) {
                    if (manifest.body_layers[l].has_v) {
                        read_runs(l, false, 0, manifest.body_layers[l].v_unit);
                    }
                }
            } else {
                for (uint32_t l = 0; l < manifest.body_layers.size(); ++l) {
                    if (!manifest.body_layers[l].has_v) {
                        continue;
                    }
                    for (uint32_t j = 0; j < manifest.body_layers[l].v_embd; ++j) {
                        read_runs(l, false, j, manifest.body_layers[l].v_unit);
                    }
                }
            }
        }
    }

    restored_tail_payload_slots.clear();
    if (manifest.body_only) {
        if (tail) {
            if (seq_id == -1) {
                tail->clear();
            } else {
                tail->seq_rm(seq_id, -1, -1);
            }
            tail->mark_degraded(seq_id, LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE);
        }
        if (seq_id == -1) {
            for (llama_seq_id current = 0; current < int32_t(n_seq_max); ++current) {
                rebuild_allocation_head(current);
            }
        } else {
            rebuild_allocation_head(seq_id);
        }
        return;
    }

    if (seq_id == -1) {
        tail->clear();
    } else {
        tail->seq_rm(seq_id, -1, -1);
    }
    std::vector<std::vector<int32_t>> payload_slots(manifest.tail_payload_count);
    for (const auto & record : manifest.tail_records) {
        const llama_seq_id dst_seq = seq_id == -1 ? record.seq_id : seq_id;
        const uint32_t dst_stream = seq_id == -1 ? record.stream : seq_to_stream.at(seq_id);
        const uint32_t dst_cell = restored_cells[record.stream][record.body_ordinal];
        if (dst_stream >= tail_generations.size() || dst_cell >= tail_generations[dst_stream].size()) {
            throw std::runtime_error("KV tail state destination identity is out of range");
        }
        tail_generations[dst_stream][dst_cell] = record.generation;
        const int32_t slot = version >= LLAMA_KV_TAIL_STATE_VERSION ?
                tail->restore(dst_seq, { dst_stream, dst_cell, record.generation },
                        record.position, record.insertion_ordinal, record.local_slot) :
                tail->commit(dst_seq, { dst_stream, dst_cell, record.generation },
                        record.position, record.insertion_ordinal);
        if (slot < 0) {
            throw std::runtime_error("KV tail state could not reserve an exact payload slot");
        }
        payload_slots[record.payload].push_back(slot);
    }
    tail_ordinal = seq_id == -1 ? manifest.tail_ordinal : std::max(live_ordinal, manifest.tail_ordinal);
    tail->restore_provenance(manifest.provenance, seq_id);
    if (version >= LLAMA_KV_TAIL_STATE_VERSION) {
        GGML_ASSERT(manifest.tail_write_cursors.size() == manifest.provenance.size());
        for (size_t i = 0; i < manifest.provenance.size(); ++i) {
            const llama_seq_id dst_seq = seq_id == -1 ? manifest.provenance[i].seq_id : seq_id;
            tail->restore_write_cursor(dst_seq, manifest.tail_write_cursors[i]);
        }
    }

    if (!on_device) {
        for (const auto & chunk : tail_chunks) {
            const auto & destinations = payload_slots[chunk.payload];
            const uint64_t row = chunk.key ? manifest.tail_layers[chunk.layer].k_row :
                    manifest.tail_layers[chunk.layer].v_row;
            ggml_tensor * tensor = chunk.key ? layers[chunk.layer].k_tail : layers[chunk.layer].v_tail;
            for (int32_t slot : destinations) {
                io.stage_tensor_set(tensor, chunk.data.data(), size_t(slot)*row, row);
            }
        }
    } else {
        for (uint32_t l = 0; l < manifest.tail_layers.size(); ++l) {
            for (uint32_t payload = 0; payload < manifest.tail_payload_count; ++payload) {
                if (payload_slots[payload].size() != 1) {
                    throw std::runtime_error("on-device KV tail state requires one destination slot per payload");
                }
                const int32_t slot = payload_slots[payload][0];
                if (manifest.tail_layers[l].has_k) {
                    io.read_tensor(layers[l].k_tail, size_t(slot)*manifest.tail_layers[l].k_row,
                            manifest.tail_layers[l].k_row);
                }
                if (manifest.tail_layers[l].has_v) {
                    io.read_tensor(layers[l].v_tail, size_t(slot)*manifest.tail_layers[l].v_row,
                            manifest.tail_layers[l].v_row);
                }
            }
        }
    }
    restored_tail_payload_slots = std::move(payload_slots);
    if (seq_id == -1) {
        for (llama_seq_id current = 0; current < int32_t(n_seq_max); ++current) {
            rebuild_allocation_head(current);
        }
    } else {
        rebuild_allocation_head(seq_id);
    }
}

bool llama_kv_cache::requires_state_for_partial_restore() const {
    return has_tail_overlay();
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    const bool requested_body_only = (flags & LLAMA_STATE_SEQ_FLAGS_BODY_ONLY) != 0;
    // A compact-native cache has no lower-fidelity body to serialize. Preserve
    // the logical cache state (including exact K/V) even when a caller asks for
    // a body-only snapshot.
    const bool body_only = requested_body_only && has_kv_body();
    if (!has_tail_overlay() && !body_only) {
        state_write_body(io, seq_id);
        return;
    }

    const auto manifest = state_v2_collect(seq_id, body_only);
    const bool skip_tensors = (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) != 0;
    // Partial checkpoints need the exact tail as canonical state, but the
    // ordinary body already remains live and position-addressable.  Record its
    // logical manifest as the rollback anchor without copying its tensor rows.
    const bool reference_only_partial =
            (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0 && seq_id >= 0;
    llama_io_write_counter manifest_counter(false);
    state_v2_write_manifest(manifest_counter, manifest);
    llama_io_write_counter body_counter(skip_tensors || reference_only_partial);
    if (!reference_only_partial) {
        state_v2_write_body_payload(body_counter, manifest);
    }

    llama_io_write_counter tail_counter(skip_tensors);
    state_v2_write_tail_payload(tail_counter, manifest);

    const uint32_t state_flags = body_only ? LLAMA_KV_TAIL_STATE_BODY_ONLY : 0;
    const int32_t state_tail_type = int32_t(tail_type);
    const uint32_t state_storage_kind = uint32_t(tail_plan.kind);
    const uint32_t state_rollback_tokens = tail_plan.compact_layout.rollback_tokens;
    uint32_t lowest_layer = UINT32_MAX;
    for (const auto & layer : layers) {
        lowest_layer = std::min(lowest_layer, uint32_t(layer.il));
    }
    const std::string group_id = std::string(n_swa > 0 ? "swa@l" : "full@l") +
            std::to_string(lowest_layer == UINT32_MAX ? 0 : lowest_layer);
    const uint64_t manifest_size = manifest_counter.n_bytes();
    const uint64_t body_payload_size = body_counter.n_bytes();
    const uint64_t tail_payload_size = tail_counter.n_bytes();
    io.write(&LLAMA_KV_TAIL_STATE_MAGIC, sizeof(LLAMA_KV_TAIL_STATE_MAGIC));
    io.write(&LLAMA_KV_TAIL_STATE_VERSION, sizeof(LLAMA_KV_TAIL_STATE_VERSION));
    io.write(&state_flags, sizeof(state_flags));
    io.write(&tail_plan.effective_tokens, sizeof(tail_plan.effective_tokens));
    io.write(&state_tail_type, sizeof(state_tail_type));
    io.write(&state_storage_kind, sizeof(state_storage_kind));
    io.write(&state_rollback_tokens, sizeof(state_rollback_tokens));
    io.write_string(group_id);
    io.write(&manifest_size, sizeof(manifest_size));
    io.write(&body_payload_size, sizeof(body_payload_size));
    io.write(&tail_payload_size, sizeof(tail_payload_size));

    const size_t manifest_begin = io.n_bytes();
    state_v2_write_manifest(io, manifest);
    if (io.n_bytes() - manifest_begin != manifest_size) {
        throw std::runtime_error("KV tail state manifest size changed while writing");
    }
    const size_t body_begin = io.n_bytes();
    if (!reference_only_partial) {
        state_v2_write_body_payload(io, manifest);
    }
    if (io.n_bytes() - body_begin != body_payload_size) {
        throw std::runtime_error("KV tail state body payload size changed while writing");
    }
    const size_t tail_begin = io.n_bytes();
    state_v2_write_tail_payload(io, manifest);
    if (io.n_bytes() - tail_begin != tail_payload_size) {
        throw std::runtime_error("KV tail state tail payload size changed while writing");
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // KVarN restores into a private metadata-only clone and commits that clone
    // atomically with its record payload. Keep that internal parser immediate.
    if (tail_metadata_only) {
        state_read_impl(io, seq_id, flags);
        return;
    }

    struct logical_state {
        llama_kv_cells_vec cells;
        std::vector<uint32_t> heads;
        std::vector<uint32_t> allocation_heads;
        std::vector<uint32_t> seq_streams;
        std::vector<std::vector<uint64_t>> generations;
        std::vector<std::vector<uint64_t>> generations_before_batch;
        uint64_t ordinal;
        std::vector<int64_t> write_slots;
        std::vector<std::vector<int32_t>> restored_slots;
        std::vector<std::pair<uint32_t, uint32_t>> remap;
        stream_copy_info stream_copy;
        std::shared_ptr<llama_kv_tail_store> tail_state;
    };

    auto capture = [&]() {
        auto state = std::make_shared<logical_state>();
        state->cells = v_cells;
        state->heads = v_heads;
        state->allocation_heads = allocation_seq_heads;
        state->seq_streams = seq_to_stream;
        state->generations = tail_generations;
        state->generations_before_batch = tail_generations_before_batch;
        state->ordinal = tail_ordinal;
        state->write_slots = tail_write_slots;
        state->restored_slots = restored_tail_payload_slots;
        state->remap = state_cell_remap;
        state->stream_copy = sc_info;
        if (tail) {
            state->tail_state = std::shared_ptr<llama_kv_tail_store>(tail->clone_logical_state().release());
        }
        return state;
    };
    auto install = [this](const logical_state & state) {
        GGML_ASSERT(v_cells.size() == state.cells.size());
        for (uint32_t stream = 0; stream < v_cells.size(); ++stream) {
            v_cells[stream].set(0, state.cells[stream]);
        }
        v_heads = state.heads;
        allocation_seq_heads = state.allocation_heads;
        seq_to_stream = state.seq_streams;
        tail_generations = state.generations;
        tail_generations_before_batch = state.generations_before_batch;
        tail_ordinal = state.ordinal;
        tail_write_slots = state.write_slots;
        restored_tail_payload_slots = state.restored_slots;
        state_cell_remap = state.remap;
        sc_info = state.stream_copy;
        if (tail) {
            GGML_ASSERT(state.tail_state);
            tail->clone_logical_state_from(*state.tail_state);
        }
    };

    auto original = capture();
    try {
        state_read_impl(io, seq_id, flags);
    } catch (...) {
        install(*original);
        throw;
    }
    auto prepared = capture();
    install(*original);
    io.on_commit([this, prepared]() {
        GGML_ASSERT(v_cells.size() == prepared->cells.size());
        for (uint32_t stream = 0; stream < v_cells.size(); ++stream) {
            v_cells[stream].set(0, prepared->cells[stream]);
        }
        v_heads = prepared->heads;
        allocation_seq_heads = prepared->allocation_heads;
        seq_to_stream = prepared->seq_streams;
        tail_generations = prepared->generations;
        tail_generations_before_batch = prepared->generations_before_batch;
        tail_ordinal = prepared->ordinal;
        tail_write_slots = prepared->write_slots;
        restored_tail_payload_slots = prepared->restored_slots;
        state_cell_remap = prepared->remap;
        sc_info = prepared->stream_copy;
        if (tail) {
            GGML_ASSERT(prepared->tail_state);
            tail->clone_logical_state_from(*prepared->tail_state);
        }
    });
}

void llama_kv_cache::state_read_impl(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    uint32_t marker;
    io.read(&marker, sizeof(marker));
    if (marker != LLAMA_KV_TAIL_STATE_MAGIC) {
        if (tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_OVERLAY ||
                tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT) {
            throw std::runtime_error(
                    "legacy KV state lacks compact-tail representation metadata");
        }
        state_read_body(io, seq_id, marker);
        if (has_tail_overlay()) {
            if (seq_id == -1) {
                tail->clear();
            } else {
                tail->seq_rm(seq_id, -1, -1);
            }
            tail->mark_degraded(seq_id, LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE);
            LLAMA_LOG_WARN("%s: loaded preceding body-only KV state; exact tail coverage is degraded until refilled\n", __func__);
        }
        return;
    }

    uint32_t version;
    uint32_t state_flags;
    uint32_t state_tail_tokens;
    int32_t state_tail_type;
    uint32_t state_storage_kind = uint32_t(LLAMA_KV_TAIL_STORAGE_DISABLED);
    uint32_t state_rollback_tokens = 0;
    std::string state_group_id;
    io.read(&version, sizeof(version));
    if (version != LLAMA_KV_TAIL_STATE_VERSION_V1 &&
            version != LLAMA_KV_TAIL_STATE_VERSION_V2 &&
            version != LLAMA_KV_TAIL_STATE_VERSION_V3 &&
            version != LLAMA_KV_TAIL_STATE_VERSION_V4 &&
            version != LLAMA_KV_TAIL_STATE_VERSION) {
        throw std::runtime_error("unsupported KV tail state version");
    }
    io.read(&state_flags, sizeof(state_flags));
    io.read(&state_tail_tokens, sizeof(state_tail_tokens));
    io.read(&state_tail_type, sizeof(state_tail_type));
    if (version >= LLAMA_KV_TAIL_STATE_VERSION_V3) {
        io.read(&state_storage_kind, sizeof(state_storage_kind));
        io.read(&state_rollback_tokens, sizeof(state_rollback_tokens));
    }
    io.read_string(state_group_id);
    uint32_t lowest_layer = UINT32_MAX;
    for (const auto & layer : layers) {
        lowest_layer = std::min(lowest_layer, uint32_t(layer.il));
    }
    const std::string group_id = std::string(n_swa > 0 ? "swa@l" : "full@l") +
            std::to_string(lowest_layer == UINT32_MAX ? 0 : lowest_layer);
    if (state_group_id != group_id) {
        throw std::runtime_error("KV tail state cache-group ID does not match the context");
    }
    const bool body_only = (state_flags & LLAMA_KV_TAIL_STATE_BODY_ONLY) != 0;
    const bool compact_context =
            tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_OVERLAY ||
            tail_plan.kind == LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT;
    if (version < LLAMA_KV_TAIL_STATE_VERSION_V3 && compact_context) {
        throw std::runtime_error(
                "KV tail state version predates compact representation metadata");
    }
    if (version >= LLAMA_KV_TAIL_STATE_VERSION_V3 &&
            (state_storage_kind != uint32_t(tail_plan.kind) ||
             state_rollback_tokens != tail_plan.compact_layout.rollback_tokens)) {
        throw std::runtime_error(
                "KV tail state representation does not match the context: state kind=" +
                std::to_string(state_storage_kind) + " rollback=" +
                std::to_string(state_rollback_tokens) + ", context kind=" +
                std::to_string(uint32_t(tail_plan.kind)) + " rollback=" +
                std::to_string(tail_plan.compact_layout.rollback_tokens));
    }
    if (body_only && !has_kv_body()) {
        throw std::runtime_error("body-only KV tail state cannot restore into a bodyless context");
    }
    if (!body_only && (!has_tail_overlay() ||
            state_tail_tokens != tail_plan.effective_tokens ||
            state_tail_type != int32_t(tail_type))) {
        throw std::runtime_error("KV tail state configuration does not match the context");
    }

    if (version >= LLAMA_KV_TAIL_STATE_VERSION_V2) {
        uint64_t manifest_size;
        uint64_t body_payload_size;
        uint64_t tail_payload_size;
        io.read(&manifest_size, sizeof(manifest_size));
        io.read(&body_payload_size, sizeof(body_payload_size));
        io.read(&tail_payload_size, sizeof(tail_payload_size));
        const size_t manifest_begin = io.n_bytes();
        auto manifest = state_v2_read_manifest(io, seq_id, body_only, version);
        if (io.n_bytes() - manifest_begin != manifest_size) {
            throw std::runtime_error("invalid KV tail state manifest size");
        }
        state_v2_read_payload_and_install(
                io, seq_id, flags, manifest, body_payload_size, tail_payload_size, version);
        return;
    }

    uint64_t body_size;
    uint64_t tail_size;
    io.read(&body_size, sizeof(body_size));
    io.read(&tail_size, sizeof(tail_size));

    uint32_t body_n_stream;
    const size_t body_begin = io.n_bytes();
    io.read(&body_n_stream, sizeof(body_n_stream));
    const auto restored_cells = state_read_body(io, seq_id, body_n_stream);
    if (io.n_bytes() - body_begin != body_size) {
        throw std::runtime_error("invalid KV tail state body section size");
    }

    const size_t tail_begin = io.n_bytes();
    if (tail_size > 0) {
        state_read_tail(io, seq_id, restored_cells, flags);
        tail->mark_degraded(seq_id, LLAMA_KV_TAIL_DEGRADED_STATE_RESTORE);
        LLAMA_LOG_WARN("%s: loaded KV tail state v1 without coverage provenance; exact coverage is conservatively degraded until refilled\n", __func__);
    } else if (has_tail_overlay()) {
        if (seq_id == -1) {
            tail->clear();
        } else {
            tail->seq_rm(seq_id, -1, -1);
        }
        tail->mark_degraded(seq_id, LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE);
        LLAMA_LOG_WARN("%s: loaded explicit body-only KV state; exact tail coverage is degraded until refilled\n", __func__);
    }
    if (io.n_bytes() - tail_begin != tail_size) {
        throw std::runtime_error("invalid KV tail state section size");
    }
}

void llama_kv_cache::state_write_body(llama_io_write_i & io, llama_seq_id seq_id) const {
    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

std::vector<std::vector<uint32_t>> llama_kv_cache::state_read_body(
        llama_io_read_i & io, llama_seq_id seq_id, uint32_t n_stream_cur) {
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    if (seq_id == -1) {
        clear(true);
    }

    std::vector<std::vector<uint32_t>> restored_cells(n_stream);

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id);

        try {
            res = res && state_read_data(io, strm, cell_count, sinfo);
        } catch (...) {
            res = false;
        }

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }

        restored_cells[s].assign(sinfo.idxs[0].begin(), sinfo.idxs[0].end());
    }

    if (seq_id == -1) {
        for (llama_seq_id current = 0; current < int32_t(n_seq_max); ++current) {
            rebuild_allocation_head(current);
        }
    } else {
        rebuild_allocation_head(seq_id);
    }

    return restored_cells;
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (hparams.n_pos_per_embd() > 1) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }

}

std::vector<int32_t> llama_kv_cache::state_tail_payload_slots(llama_seq_id seq_id) const {
    GGML_ASSERT(tail);

    std::vector<int32_t> result;
    std::unordered_map<int32_t, uint32_t> payload_by_slot;
    for (const auto & entry : tail->snapshot(seq_id)) {
        if (entry.identity.stream >= v_cells.size()) {
            continue;
        }
        const auto & cells = v_cells[entry.identity.stream];
        bool found = false;
        for (uint32_t cell = 0; cell < cells.size(); ++cell) {
            bool included = !cells.is_empty(cell) && (seq_id == -1 || cells.seq_has(cell, seq_id));
            if (included && seq_id != -1) {
                included = !llama_hparams::is_masked_swa(
                        n_swa, swa_type, cells.pos_get(cell), cells.seq_pos_max(seq_id));
            }
            if (!included) {
                continue;
            }
            if (cell == entry.identity.cell) {
                found = true;
                break;
            }
        }
        if (!found) {
            continue;
        }
        if (payload_by_slot.emplace(entry.slot, uint32_t(result.size())).second) {
            result.push_back(entry.slot);
        }
    }
    return result;
}

std::vector<std::vector<int32_t>> llama_kv_cache::take_restored_tail_payload_slots() {
    std::vector<std::vector<int32_t>> result;
    result.swap(restored_tail_payload_slots);
    return result;
}

std::vector<uint32_t> llama_kv_cache::state_source_cells(llama_seq_id seq_id) const {
    if (seq_id < 0 || uint32_t(seq_id) >= n_seq_max) {
        throw std::invalid_argument("invalid KV state sequence ID");
    }
    const uint32_t stream = seq_to_stream.at(seq_id);
    const auto & cells = v_cells[stream];
    const llama_pos pos_max = cells.seq_pos_max(seq_id);
    std::vector<uint32_t> result;
    result.reserve(cells.get_used());
    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
        if (!cells.is_empty(cell) && cells.seq_has(cell, seq_id) &&
                !llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(cell), pos_max)) {
            result.push_back(cell);
        }
    }
    return result;
}

void llama_kv_cache::set_state_remap_group_size(uint32_t group_size) {
    if (group_size == 0 || get_size() % group_size != 0) {
        throw std::invalid_argument("invalid KV state remap group size");
    }
    state_remap_group_size = group_size;
}

void llama_kv_cache::set_allocation_group_size(uint32_t group_size, uint32_t stage_groups) {
    if (group_size == 0 || stage_groups == 0 || get_size() % group_size != 0 ||
            get_size()/group_size < stage_groups) {
        throw std::invalid_argument("invalid KV allocation group size");
    }
    allocation_group_size = group_size;
    allocation_stage_groups = stage_groups;
    for (llama_seq_id seq_id = 0; uint32_t(seq_id) < n_seq_max; ++seq_id) {
        reset_allocation_head(seq_id);
    }
}

bool llama_kv_cache::allocation_cell_uses_stage(uint32_t cell) const {
    if (allocation_group_size <= 1 || n_stream != 1 || cell >= get_size()) {
        return false;
    }
    const uint32_t group = cell/allocation_group_size;
    if (group == 0) {
        return true;
    }
    for (const uint32_t head : allocation_seq_heads) {
        if (head != 0 && head%allocation_group_size != 0 &&
                (head - 1u)/allocation_group_size == group) {
            return true;
        }
    }
    return false;
}

void llama_kv_cache::reset_allocation_head(llama_seq_id seq_id) {
    if (seq_id < 0 || uint32_t(seq_id) >= n_seq_max || allocation_seq_heads.empty()) {
        return;
    }
    allocation_seq_heads[size_t(seq_id)] = 0;
}

void llama_kv_cache::rebuild_allocation_head(llama_seq_id seq_id) {
    if (seq_id < 0 || uint32_t(seq_id) >= n_seq_max ||
            allocation_group_size <= 1 || n_stream != 1) {
        return;
    }
    const auto & cells = v_cells[0];
    llama_pos pos_max = -1;
    uint32_t newest_cell = cells.size();
    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
        if (cells.seq_has(cell, seq_id) && cells.pos_get(cell) > pos_max) {
            pos_max = cells.pos_get(cell);
            newest_cell = cell;
        }
    }
    if (newest_cell == cells.size()) {
        reset_allocation_head(seq_id);
    } else {
        allocation_seq_heads[size_t(seq_id)] = newest_cell + 1;
    }
}

const std::vector<std::pair<uint32_t, uint32_t>> & llama_kv_cache::get_state_cell_remap() const {
    return state_cell_remap;
}

void llama_kv_cache::clone_logical_state_from(const llama_kv_cache & source) {
    if (n_seq_max != source.n_seq_max || n_stream != source.n_stream ||
            v_cells.size() != source.v_cells.size() ||
            tail_generations.size() != source.tail_generations.size() ||
            bool(tail) != bool(source.tail)) {
        throw std::runtime_error("cannot clone incompatible KV cache logical state");
    }
    if (source.tail_preparing || source.tail_graph_started ||
            (source.tail && (source.tail->has_batch_transaction() || source.tail->has_pending_seq_cp()))) {
        throw std::runtime_error("cannot clone KV cache logical state during a transaction");
    }

    for (uint32_t stream = 0; stream < n_stream; ++stream) {
        if (v_cells[stream].size() != source.v_cells[stream].size()) {
            throw std::runtime_error("cannot clone KV cache with a different stream capacity");
        }
        v_cells[stream].set(0, source.v_cells[stream].cp(0, source.v_cells[stream].size()));
    }
    v_heads = source.v_heads;
    allocation_seq_heads = source.allocation_seq_heads;
    seq_to_stream = source.seq_to_stream;
    tail_generations = source.tail_generations;
    tail_generations_before_batch = source.tail_generations_before_batch;
    tail_ordinal = source.tail_ordinal;
    restored_tail_payload_slots.clear();
    state_cell_remap.clear();
    sc_info = {};
    if (tail) {
        tail->clone_logical_state_from(*source.tail);
    }
}

void llama_kv_cache::state_write_tail(llama_io_write_i & io, llama_seq_id seq_id) const {
    GGML_ASSERT(tail);

    struct record {
        llama_seq_id seq_id;
        uint32_t stream;
        uint32_t body_ordinal;
        uint64_t generation;
        llama_pos position;
        uint64_t insertion_ordinal;
        uint32_t payload;
    };

    std::vector<record> records;
    std::vector<int32_t> payload_slots = state_tail_payload_slots(seq_id);
    std::unordered_map<int32_t, uint32_t> payload_by_slot;
    for (uint32_t payload = 0; payload < payload_slots.size(); ++payload) {
        payload_by_slot.emplace(payload_slots[payload], payload);
    }

    for (const auto & entry : tail->snapshot(seq_id)) {
        if (entry.identity.stream >= v_cells.size()) {
            continue;
        }
        const auto & cells = v_cells[entry.identity.stream];
        uint32_t ordinal = 0;
        bool found = false;
        for (uint32_t cell = 0; cell < cells.size(); ++cell) {
            bool included = !cells.is_empty(cell) && (seq_id == -1 || cells.seq_has(cell, seq_id));
            if (included && seq_id != -1) {
                included = !llama_hparams::is_masked_swa(
                        n_swa, swa_type, cells.pos_get(cell), cells.seq_pos_max(seq_id));
            }
            if (!included) {
                continue;
            }
            if (cell == entry.identity.cell) {
                found = true;
                break;
            }
            ++ordinal;
        }
        if (!found) {
            continue;
        }

        const auto it = payload_by_slot.find(entry.slot);
        GGML_ASSERT(it != payload_by_slot.end());
        records.push_back({ entry.seq_id, entry.identity.stream, ordinal, entry.identity.generation,
                entry.position, entry.insertion_ordinal, it->second });
    }

    const uint32_t n_records = uint32_t(records.size());
    const uint32_t n_payload = uint32_t(payload_slots.size());
    const uint32_t n_layers = uint32_t(layers.size());
    io.write(&tail_ordinal, sizeof(tail_ordinal));
    io.write(&n_records, sizeof(n_records));
    io.write(&n_payload, sizeof(n_payload));
    io.write(&n_layers, sizeof(n_layers));
    for (const auto & record : records) {
        io.write(&record.seq_id, sizeof(record.seq_id));
        io.write(&record.stream, sizeof(record.stream));
        io.write(&record.body_ordinal, sizeof(record.body_ordinal));
        io.write(&record.generation, sizeof(record.generation));
        io.write(&record.position, sizeof(record.position));
        io.write(&record.insertion_ordinal, sizeof(record.insertion_ordinal));
        io.write(&record.payload, sizeof(record.payload));
    }

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;
        const uint32_t has_k = layer.k_tail ? 1 : 0;
        const uint32_t has_v = layer.v_tail ? 1 : 0;
        const uint64_t k_row = layer.k_tail ? ggml_row_size(layer.k_tail->type, layer.k_tail->ne[0]) : 0;
        const uint64_t v_row = layer.v_tail ? ggml_row_size(layer.v_tail->type, layer.v_tail->ne[0]) : 0;
        io.write(&il, sizeof(il));
        io.write(&has_k, sizeof(has_k));
        io.write(&has_v, sizeof(has_v));
        io.write(&k_row, sizeof(k_row));
        io.write(&v_row, sizeof(v_row));
        for (int32_t slot : payload_slots) {
            if (layer.k_tail) {
                io.write_tensor(layer.k_tail, size_t(slot)*k_row, k_row);
            }
            if (layer.v_tail) {
                io.write_tensor(layer.v_tail, size_t(slot)*v_row, v_row);
            }
        }
    }
}

void llama_kv_cache::state_read_tail(
        llama_io_read_i & io,
        llama_seq_id seq_id,
        const std::vector<std::vector<uint32_t>> & restored_cells,
        llama_state_seq_flags flags) {
    GGML_ASSERT(tail);

    struct record {
        llama_seq_id seq_id;
        uint32_t stream;
        uint32_t body_ordinal;
        uint64_t generation;
        llama_pos position;
        uint64_t insertion_ordinal;
        uint32_t payload;
    };

    uint64_t restored_ordinal;
    uint32_t n_records;
    uint32_t n_payload;
    uint32_t n_layers;
    io.read(&restored_ordinal, sizeof(restored_ordinal));
    io.read(&n_records, sizeof(n_records));
    io.read(&n_payload, sizeof(n_payload));
    io.read(&n_layers, sizeof(n_layers));
    if (n_payload > tail_slots || uint64_t(n_records) > uint64_t(n_payload)*n_seq_max || n_layers != layers.size()) {
        throw std::runtime_error("invalid KV tail state dimensions");
    }

    std::vector<record> records(n_records);
    for (auto & record : records) {
        io.read(&record.seq_id, sizeof(record.seq_id));
        io.read(&record.stream, sizeof(record.stream));
        io.read(&record.body_ordinal, sizeof(record.body_ordinal));
        io.read(&record.generation, sizeof(record.generation));
        io.read(&record.position, sizeof(record.position));
        io.read(&record.insertion_ordinal, sizeof(record.insertion_ordinal));
        io.read(&record.payload, sizeof(record.payload));
    }

    if (seq_id == -1) {
        tail->clear();
    } else {
        tail->seq_rm(seq_id, -1, -1);
    }
    restored_tail_payload_slots.clear();
    std::vector<std::vector<int32_t>> slots(n_payload);
    for (const auto & record : records) {
        if (record.stream >= restored_cells.size() ||
                record.body_ordinal >= restored_cells[record.stream].size() || record.payload >= n_payload) {
            throw std::runtime_error("invalid KV tail state identity mapping");
        }
        const llama_seq_id dst_seq = seq_id == -1 ? record.seq_id : seq_id;
        const uint32_t dst_stream = seq_id == -1 ? record.stream : seq_to_stream.at(dst_seq);
        const uint32_t dst_cell = restored_cells[record.stream][record.body_ordinal];
        if (dst_stream >= tail_generations.size() || dst_cell >= tail_generations[dst_stream].size()) {
            throw std::runtime_error("KV tail state destination is out of range");
        }
        tail_generations[dst_stream][dst_cell] = record.generation;
        const int32_t slot = tail->commit(dst_seq, { dst_stream, dst_cell, record.generation },
                record.position, record.insertion_ordinal);
        if (slot < 0) {
            throw std::runtime_error("KV tail state v1 could not reserve an exact payload slot");
        }
        slots[record.payload].push_back(slot);
    }
    tail_ordinal = restored_ordinal;

    const bool on_device = (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) != 0;
    for (const auto & layer : layers) {
        uint32_t il;
        uint32_t has_k;
        uint32_t has_v;
        uint64_t k_row;
        uint64_t v_row;
        io.read(&il, sizeof(il));
        io.read(&has_k, sizeof(has_k));
        io.read(&has_v, sizeof(has_v));
        io.read(&k_row, sizeof(k_row));
        io.read(&v_row, sizeof(v_row));
        const uint64_t expected_k_row = layer.k_tail ? ggml_row_size(layer.k_tail->type, layer.k_tail->ne[0]) : 0;
        const uint64_t expected_v_row = layer.v_tail ? ggml_row_size(layer.v_tail->type, layer.v_tail->ne[0]) : 0;
        if (il != uint32_t(layer.il) || has_k != uint32_t(layer.k_tail != nullptr) ||
                has_v != uint32_t(layer.v_tail != nullptr) || k_row != expected_k_row || v_row != expected_v_row) {
            throw std::runtime_error("KV tail state layer layout mismatch");
        }
        for (uint32_t payload = 0; payload < n_payload; ++payload) {
            if (slots[payload].empty()) {
                throw std::runtime_error("KV tail state contains an unreferenced payload");
            }
            if (on_device && slots[payload].size() != 1) {
                throw std::runtime_error("on-device KV tail state requires one destination slot per payload");
            }
            if (layer.k_tail) {
                if (on_device) {
                    io.read_tensor(layer.k_tail, size_t(slots[payload][0])*k_row, k_row);
                } else {
                    std::vector<uint8_t> row(k_row);
                    io.read(row.data(), row.size());
                    for (int32_t slot : slots[payload]) {
                        io.stage_tensor_set(layer.k_tail, row.data(), size_t(slot)*k_row, k_row);
                    }
                }
            }
            if (layer.v_tail) {
                if (on_device) {
                    io.read_tensor(layer.v_tail, size_t(slots[payload][0])*v_row, v_row);
                } else {
                    std::vector<uint8_t> row(v_row);
                    io.read(row.data(), row.size());
                    for (int32_t slot : slots[payload]) {
                        io.stage_tensor_set(layer.v_tail, row.data(), size_t(slot)*v_row, v_row);
                    }
                }
            }
        }
    }
    restored_tail_payload_slots = std::move(slots);
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        sinfo = find_slot(ubatch, false);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
            return false;
        }

        // TODO: we cannot yet restore llama_kv_cell_ext as the apply_ubatch() does not support it yet
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        // Body-state placement must not synthesize exact shadows. The framed
        // tail section restores original payloads immediately afterwards; a
        // body-only restore explicitly reports degraded coverage instead.
        struct tail_preparing_guard {
            bool & value;
            bool old;
            ~tail_preparing_guard() { value = old; }
        } guard { tail_preparing, tail_preparing };
        tail_preparing = true;
        apply_ubatch(sinfo, ubatch);

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    uint32_t v_trans;
    uint32_t n_layer;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            if (sinfo.is_contiguous()) {
                // Fast path: contiguous cells, single memcpy
                io.read_tensor(k, sinfo.head() * k_size_row, cell_count * k_size_row);
            } else {
                // Slow path: scatter to non-contiguous positions
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const size_t dst_offset = sinfo.idxs[0][i] * k_size_row;
                    io.read_tensor(k, dst_offset, k_size_row);
                }
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells, single memcpy
                    io.read_tensor(v, sinfo.head() * v_size_row, cell_count * v_size_row);
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = sinfo.idxs[0][i] * v_size_row;
                        io.read_tensor(v, dst_offset, v_size_row);
                    }
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells
                    const uint32_t h = sinfo.head();
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (h + j * cells.size()) * v_size_el;
                        io.read_tensor(v, dst_offset, cell_count * v_size_el);
                    }
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        for (uint32_t i = 0; i < cell_count; ++i) {
                            const size_t dst_offset = (sinfo.idxs[0][i] + j * cells.size()) * v_size_el;
                            io.read_tensor(v, dst_offset, v_size_el);
                        }
                    }
                }
            }
        }
    }

    return true;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        status = kv->update(lctx, do_shift, sc_info);
        return !llama_memory_status_is_fail(status);
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = kv->get_n_kv(sinfos[i_cur]);

    return true;
}

void llama_kv_cache_context::graph_compute_start() {
    graph_started = true;
}

void llama_kv_cache_context::graph_compute_finish(ggml_status compute_status) {
    if (!ubatches.empty()) {
        kv->finish_tail_batch(compute_status == GGML_STATUS_SUCCESS,
                graph_started && compute_status != GGML_STATUS_SUCCESS);
    }
    graph_started = false;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

llama_kv_cache * llama_kv_cache_context::get_kv() const {
    return kv;
}

const llama_kv_cache::slot_info & llama_kv_cache_context::current_sinfo() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    assert(i_cur < sinfos.size());

    return sinfos[i_cur];
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

void llama_kv_cache::set_input_k_shift_tail(ggml_tensor * dst) const {
    GGML_ASSERT(tail && ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I32 && uint32_t(dst->ne[0]) == tail_slots);

    int32_t * data = static_cast<int32_t *>(dst->data);
    std::fill(data, data + tail_slots, 0);
    for (const auto & [slot, identity] : tail->active_slots()) {
        GGML_ASSERT(identity.stream < v_cells.size() && identity.cell < v_cells[identity.stream].size());
        const auto & cells = v_cells[identity.stream];
        if (!cells.is_empty(identity.cell) && tail_generations[identity.stream][identity.cell] == identity.generation) {
            data[slot] = cells.get_shift(identity.cell);
        }
    }
}

ggml_tensor * llama_kv_cache_context::get_k_tail(ggml_context * ctx, int32_t il) const {
    return kv->get_k_tail(ctx, il);
}

ggml_tensor * llama_kv_cache_context::get_v_tail(ggml_context * ctx, int32_t il) const {
    return kv->get_v_tail(ctx, il);
}

ggml_tensor * llama_kv_cache_context::get_k_tail_fallback(
        ggml_context * ctx, int32_t il, ggml_tensor * body_idxs) const {
    return kv->get_k_tail_fallback(ctx, il, body_idxs);
}

ggml_tensor * llama_kv_cache_context::get_v_tail_fallback(
        ggml_context * ctx, int32_t il, ggml_tensor * body_idxs) const {
    return kv->get_v_tail_fallback(ctx, il, body_idxs);
}

uint32_t llama_kv_cache_context::get_tail_slots() const {
    return kv->get_tail_slots();
}

ggml_type llama_kv_cache_context::get_tail_type() const {
    return kv->get_tail_type();
}

uint32_t llama_kv_cache_context::get_tail_tokens() const {
    return kv->get_tail_tokens();
}

uint32_t llama_kv_cache_context::get_tail_arena_stride() const {
    return kv->get_tail_arena_stride();
}

uint32_t llama_kv_cache_context::get_tail_body_execution_stride() const {
    return kv->get_tail_body_execution_stride();
}

uint32_t llama_kv_cache_context::get_tail_body_execution_rows(int32_t il) const {
    return kv->get_tail_body_execution_rows(il);
}

bool llama_kv_cache_context::has_compact_tail() const {
    return kv->has_compact_tail();
}

bool llama_kv_cache_context::has_kv_body() const {
    return kv->has_kv_body();
}

bool llama_kv_cache_context::has_kv_body(int32_t il) const {
    return kv->has_kv_body(il);
}

bool llama_kv_cache_context::has_tail_current(int32_t il) const {
    return kv->has_tail_current(il);
}

ggml_backend_dev_t llama_kv_cache_context::get_tail_backend(int32_t il) const {
    return kv->get_tail_backend(il);
}

llama_kv_tail_storage_kind llama_kv_cache_context::get_tail_storage_kind() const {
    return kv->get_tail_storage_kind();
}

uint32_t llama_kv_cache_context::get_tail_rollback_tokens() const {
    return kv->get_tail_rollback_tokens();
}

uint32_t llama_kv_cache::get_tail_attention_stride(uint32_t n_query_tokens) const {
    if (!has_tail_overlay()) {
        return 0;
    }
    if (has_compact_tail()) {
        if (n_query_tokens == 0) {
            return tail_attention_stride;
        }
        const uint64_t required = uint64_t(tail_tokens) + n_query_tokens;
        GGML_ASSERT(required <= uint64_t(INT32_MAX));
        return uint32_t(required);
    }
    if (n_query_tokens == 0) {
        return tail_attention_stride;
    }
    // A single-query ubatch needs only its configured newest N entries.  The
    // extra in-flight rows are a rollback reserve and are required only by a
    // multi-query ubatch whose early and late queries have different windows.
    const uint64_t required = n_query_tokens == 1 ? tail_tokens : uint64_t(tail_tokens) + n_query_tokens;
    GGML_ASSERT(required <= tail_arena_stride);
    // The indexed small-tail CUDA path consumes arbitrary extents directly.
    // Larger tails use the regular FlashAttention kernels, whose K/V tiles are
    // 256 rows wide; padding only the attention view avoids a costly partial
    // tile without increasing persistent arena storage.
    constexpr uint32_t small_tail_extent = 128;
    constexpr uint32_t fa_tile = 256;
    if (required <= small_tail_extent) {
        return uint32_t(required);
    }
    return uint32_t((required + fa_tile - 1)/fa_tile*fa_tile);
}

llama_kv_tail_route llama_kv_cache::get_tail_route(int32_t il) const {
    const auto * route = get_tail_layer_route(il);
    return route ? route->capability.route : LLAMA_KV_TAIL_ROUTE_NONE;
}

const llama_kv_tail_layer_route * llama_kv_cache::get_tail_layer_route(int32_t il) const {
    const auto it = std::find_if(tail_plan.layer_routes.begin(), tail_plan.layer_routes.end(),
            [&](const llama_kv_tail_layer_route & route) { return route.layer_id == uint32_t(il); });
    if (it == tail_plan.layer_routes.end()) {
        return nullptr;
    }
    GGML_ASSERT(it->capability.supported);
    return &*it;
}

uint32_t llama_kv_cache::get_tail_body_execution_stride() const {
    uint32_t result = 0;
    for (const auto & route : tail_plan.layer_routes) {
        result = std::max(result, route.body_execution_rows);
    }
    return result;
}

uint32_t llama_kv_cache::get_tail_body_execution_rows(int32_t il) const {
    if (!has_tail_overlay()) {
        return 0;
    }
    const auto it = std::find_if(tail_plan.layer_routes.begin(), tail_plan.layer_routes.end(),
            [&](const llama_kv_tail_layer_route & route) { return route.layer_id == uint32_t(il); });
    if (it == tail_plan.layer_routes.end()) {
        throw std::logic_error(format("KV tail has no execution descriptor for layer %d", il));
    }
    return it->body_execution_rows;
}

bool llama_kv_cache::has_kv_body(int32_t il) const {
    if (!has_tail_overlay()) {
        return true;
    }
    const auto it = std::find_if(tail_plan.layer_routes.begin(), tail_plan.layer_routes.end(),
            [&](const llama_kv_tail_layer_route & route) { return route.layer_id == uint32_t(il); });
    if (it == tail_plan.layer_routes.end()) {
        throw std::logic_error(format("KV tail has no execution descriptor for layer %d", il));
    }
    return it->has_body;
}

bool llama_kv_cache::has_tail_current(int32_t il) const {
    if (!has_tail_overlay()) {
        return false;
    }
    const auto it = std::find_if(tail_plan.layer_routes.begin(), tail_plan.layer_routes.end(),
            [&](const llama_kv_tail_layer_route & route) { return route.layer_id == uint32_t(il); });
    if (it == tail_plan.layer_routes.end()) {
        throw std::logic_error(format("KV tail has no execution descriptor for layer %d", il));
    }
    return it->has_current;
}

ggml_backend_dev_t llama_kv_cache::get_tail_backend(int32_t il) const {
    if (!has_tail_overlay()) {
        return nullptr;
    }
    const auto it = std::find_if(tail_plan.layer_routes.begin(), tail_plan.layer_routes.end(),
            [&](const llama_kv_tail_layer_route & route) { return route.layer_id == uint32_t(il); });
    if (it == tail_plan.layer_routes.end()) {
        throw std::logic_error(format("KV tail has no execution descriptor for layer %d", il));
    }
    return it->owner;
}

bool llama_kv_cache::get_tail_explicit_bias(int32_t il) const {
    if (!has_tail_overlay() && tail_plan.kind != LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT) {
        return false;
    }
    const auto it = std::find_if(tail_plan.layer_routes.begin(), tail_plan.layer_routes.end(),
            [&](const llama_kv_tail_layer_route & route) { return route.layer_id == uint32_t(il); });
    return it != tail_plan.layer_routes.end() && it->explicit_bias;
}

const std::vector<llama_kv_tail_layer_route> & llama_kv_cache::get_tail_layer_routes() const {
    return tail_plan.layer_routes;
}

void llama_kv_cache::set_tail_routes(std::vector<llama_kv_tail_layer_route> routes) {
    if (!has_tail_overlay()) {
        if (!routes.empty()) {
            throw std::invalid_argument("cannot attach routes to a non-overlay KV tail plan");
        }
        return;
    }
    if (!tail_plan.layer_routes.empty()) {
        throw std::logic_error("KV tail routes were already finalized");
    }
    for (const auto & route : routes) {
        if (!route.capability.supported || route.capability.route == LLAMA_KV_TAIL_ROUTE_NONE) {
            throw std::invalid_argument("cannot attach an incomplete KV tail route");
        }
    }
    tail_plan.layer_routes = std::move(routes);
}

void llama_kv_cache::finalize_tail_overlay_metadata() {
    if (!has_tail_overlay() || tail) {
        return;
    }
    if (tail_plan.layer_routes.empty()) {
        throw std::logic_error("cannot finalize KV tail metadata before layer routes");
    }
    for (const auto & route : tail_plan.layer_routes) {
        if (!route.capability.supported || route.capability.route == LLAMA_KV_TAIL_ROUTE_NONE) {
            throw std::logic_error("cannot finalize KV tail metadata with an incomplete route");
        }
    }

    if (has_compact_tail()) {
        tail_arena_stride = tail_plan.compact_layout.history_stride;
        tail_attention_stride = tail_plan.compact_layout.attention_stride;
        tail_sink_slots = 0;
        tail_slots = tail_plan.compact_layout.history_slots;
        tail = std::make_unique<llama_kv_tail_store>(
                tail_plan.effective_tokens, tail_plan.compact_layout.rollback_tokens,
                n_seq_max, tail_arena_stride, 0);
    } else {
        tail_arena_stride = tail_plan.layout.arena_stride;
        tail_attention_stride = tail_plan.effective_tokens;
        tail_sink_slots = tail_plan.layout.sink_slots;
        tail_slots = tail_plan.layout.total_slots;
        tail = std::make_unique<llama_kv_tail_store>(
                tail_plan.effective_tokens, n_seq_max, tail_arena_stride, tail_sink_slots);
    }
    const char * timing = std::getenv("LLAMA_KV_TAIL_PLANNER_TIMING");
    tail_planner_timing_enabled = timing && std::strcmp(timing, "1") == 0;
    tail_generations.assign(n_stream, std::vector<uint64_t>(get_size(), 0));
}

llama_kv_cache::stream_copy_info llama_kv_cache::take_pending_tail_copies() {
    stream_copy_info result;
    result.tail_src_slots.swap(sc_info.tail_src_slots);
    result.tail_dst_slots.swap(sc_info.tail_dst_slots);
    result.tail_transaction = sc_info.tail_transaction;
    sc_info.tail_transaction = false;
    return result;
}

void llama_kv_cache::commit_pending_tail_copy() {
    if (tail) {
        tail->commit_seq_cp();
    }
}

void llama_kv_cache::cancel_pending_tail_copy() {
    if (tail) {
        tail->cancel_seq_cp();
    }
}

uint32_t llama_kv_cache_context::get_tail_attention_stride(uint32_t n_query_tokens) const {
    return kv->get_tail_attention_stride(n_query_tokens);
}

llama_kv_tail_route llama_kv_cache_context::get_tail_route(int32_t il) const {
    return kv->get_tail_route(il);
}

const llama_kv_tail_layer_route * llama_kv_cache_context::get_tail_layer_route(int32_t il) const {
    return kv->get_tail_layer_route(il);
}

bool llama_kv_cache_context::get_tail_explicit_bias(int32_t il) const {
    return kv->get_tail_explicit_bias(il);
}

bool llama_kv_cache_context::can_pack_tail_body(const llama_ubatch & ubatch) const {
    return kv->can_pack_tail_body(ubatch);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k_with_tail(
        ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs,
        ggml_tensor * tail_idxs, int32_t il) const {
    return kv->cpy_k_with_tail(ctx, k_cur, k_idxs, tail_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v_with_tail(
        ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs,
        ggml_tensor * tail_idxs, int32_t il) const {
    return kv->cpy_v_with_tail(ctx, v_cur, v_idxs, tail_idxs, il, sinfos[i_cur]);
}

template <typename T>
static bool kv_tail_mask_finite(T value) {
    return std::isfinite(llama_cast<float>(value));
}

template <>
bool kv_tail_mask_finite(ggml_fp16_t value) {
    // Attention masks contain finite scores or +/-infinity.  Inspecting the
    // IEEE-754 half exponent avoids millions of scalar half-to-float table
    // conversions in large multi-query tail plans.
    return (value & 0x7c00u) != 0x7c00u;
}

template <typename T>
static void set_input_kq_mask_tail_impl(
        const llama_kv_tail_store & tail,
        const std::vector<std::vector<uint64_t>> & generations,
        const std::vector<uint32_t> & seq_to_stream,
        T * body_data,
        T * exact_data,
        int32_t * read_data,
        int32_t * body_read_data,
        int32_t * bias_read_data,
        const llama_ubatch * ubatch,
        uint32_t n_kv,
        uint32_t attention_stride,
        bool causal_attn,
        const std::vector<int32_t> * physical_to_read) {
    const T drop = llama_cast<T>(-INFINITY);
    std::fill(exact_data, exact_data + uint64_t(attention_stride)*ubatch->n_tokens, drop);
    if (read_data) {
        std::fill(read_data, read_data + uint64_t(attention_stride)*ubatch->n_tokens, 0);
    }
    if (body_read_data) {
        std::fill(body_read_data, body_read_data + uint64_t(attention_stride)*ubatch->n_tokens, 0);
    }
    if (bias_read_data) {
        std::fill(bias_read_data, bias_read_data + uint64_t(attention_stride)*ubatch->n_tokens, 0);
    }

    struct exact_entry {
        llama_kv_tail_mask_entry mask;
        int32_t slot;
        int32_t body_row;
    };
    struct copy_run {
        uint32_t exact;
        uint32_t stream;
        uint32_t cell;
        uint32_t length;
    };
    struct sequence_plan {
        std::vector<exact_entry> entries;
        std::vector<llama_kv_tail_mask_entry> mask_entries;
    };

    // Physical cache allocation and tail insertion are both position ordered
    // in the steady state.  Collapse their matching spans into memcpy/fill
    // runs once per active sequence instead of repeating candidate metadata,
    // generation and finiteness branches for every query row.
    std::unordered_map<llama_seq_id, sequence_plan> plans_by_seq;
    std::unordered_map<llama_seq_id, uint32_t> query_count_by_seq;
    std::vector<uint8_t> finite;
    std::vector<uint32_t> selected;
    std::vector<copy_run> selected_runs;
    finite.reserve(attention_stride);
    selected.reserve(tail.retention());
    selected_runs.reserve(tail.retention());
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        ++query_count_by_seq[ubatch->seq_id[i][0]];
    }
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];
        const uint32_t stream = seq_to_stream.at(seq_id);
        const bool compact_single = query_count_by_seq.at(seq_id) == 1;
        auto [plan_it, inserted] = plans_by_seq.try_emplace(seq_id);
        if (inserted) {
            auto & plan = plan_it->second;
            const auto candidates = tail.source_candidates(seq_id);
            const size_t first = compact_single && candidates.size() > tail.retention() ?
                    candidates.size() - std::min<size_t>(tail.retention(), candidates.size()) : 0;
            GGML_ASSERT(candidates.size() - first <= attention_stride);
            plan.entries.reserve(candidates.size() - first);
            plan.mask_entries.reserve(candidates.size() - first);
            for (size_t j = first; j < candidates.size(); ++j) {
                const auto & candidate = candidates[j];
                const auto & identity = candidate.identity;
                if (identity.stream != stream ||
                        identity.cell >= generations.at(stream).size() ||
                        generations.at(stream)[identity.cell] != identity.generation) {
                    continue;
                }
                const int32_t read_cell = physical_to_read ?
                        physical_to_read->at(identity.cell) : int32_t(identity.cell);
                if (read_cell < 0 || uint32_t(read_cell) >= n_kv) {
                    continue;
                }
                const uint64_t body_row = uint64_t(stream)*n_kv + uint32_t(read_cell);
                GGML_ASSERT(body_row <= uint64_t(INT32_MAX));
                const llama_kv_tail_mask_entry mask_entry {
                    uint32_t(j - first), uint32_t(read_cell), candidate.position,
                };
                plan.entries.push_back({ mask_entry, candidate.slot, int32_t(body_row) });
                plan.mask_entries.push_back(mask_entry);
            }
        }
        const auto & plan = plan_it->second;
        T * body_row = body_data + uint64_t(i)*n_kv;
        T * exact_row = exact_data + uint64_t(i)*attention_stride;

        finite.assign(plan.entries.size(), 0);
        for (size_t j = 0; j < plan.entries.size(); ++j) {
            finite[j] = kv_tail_mask_finite(body_row[plan.entries[j].mask.cell]);
        }
        llama_kv_tail_select_masked_entries(
                plan.mask_entries, finite, ubatch->pos[i], tail.retention(), causal_attn, selected);
        GGML_ASSERT(selected.size() <= tail.retention());

        selected_runs.clear();
        for (uint32_t index : selected) {
            const auto & entry = plan.entries[index].mask;
            if (!selected_runs.empty()) {
                auto & run = selected_runs.back();
                if (run.exact + run.length == entry.exact && run.stream == stream &&
                        run.cell + run.length == entry.cell) {
                    ++run.length;
                    continue;
                }
            }
            selected_runs.push_back({ entry.exact, stream, entry.cell, 1 });
        }
        for (const auto & run : selected_runs) {
            GGML_ASSERT(run.exact + run.length <= attention_stride && run.cell + run.length <= n_kv);
            std::memcpy(exact_row + run.exact, body_row + run.cell, size_t(run.length)*sizeof(T));
            std::fill_n(body_row + run.cell, run.length, drop);
        }
        if (read_data || body_read_data || bias_read_data) {
            for (uint32_t index : selected) {
                const auto & entry = plan.entries[index];
                const uint64_t exact_index = uint64_t(i)*attention_stride + entry.mask.exact;
                if (read_data) {
                    read_data[exact_index] = entry.slot;
                }
                if (body_read_data) {
                    body_read_data[exact_index] = entry.body_row;
                }
                const uint64_t body_index = uint64_t(i)*n_kv + entry.mask.cell;
                GGML_ASSERT(body_index <= uint64_t(INT32_MAX));
                if (bias_read_data) {
                    bias_read_data[exact_index] = int32_t(body_index);
                }
            }
        }
    }

}

void llama_kv_cache::set_input_kq_mask_tail(
        ggml_tensor * body, ggml_tensor * exact,
        ggml_tensor * read_idxs, ggml_tensor * body_read_idxs, ggml_tensor * bias_read_idxs,
        const llama_ubatch * ubatch, bool causal_attn) const {
    set_input_kq_mask_tail_mapped(
            body, exact, read_idxs, body_read_idxs, bias_read_idxs,
            ubatch, causal_attn, {});
}

void llama_kv_cache::set_input_kq_mask_tail_mapped(
        ggml_tensor * body, ggml_tensor * exact,
        ggml_tensor * read_idxs, ggml_tensor * body_read_idxs, ggml_tensor * bias_read_idxs,
        const llama_ubatch * ubatch, bool causal_attn,
        const std::vector<int64_t> & read_cells) const {
    const uint32_t n_kv = uint32_t(body ? body->ne[0] : read_cells.size());
    if (!tail || !exact || !exact->buffer) {
        return;
    }
    scoped_kv_tail_planner_timer timer(tail_planner_timing_enabled, tail_planner_timing_ns);
    GGML_ASSERT(body && body->buffer);
    GGML_ASSERT(ggml_backend_buffer_is_host(body->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(exact->buffer));
    GGML_ASSERT(!read_idxs || !read_idxs->buffer || ggml_backend_buffer_is_host(read_idxs->buffer));
    GGML_ASSERT(!body_read_idxs || !body_read_idxs->buffer || ggml_backend_buffer_is_host(body_read_idxs->buffer));
    GGML_ASSERT(!bias_read_idxs || !bias_read_idxs->buffer || ggml_backend_buffer_is_host(bias_read_idxs->buffer));
    const uint32_t attention_stride = uint32_t(exact->ne[0]);
    GGML_ASSERT(body->type == exact->type);
    if (has_compact_tail()) {
        // Compact graphs size the exact input to N + the largest active query
        // cohort.  That runtime extent can exceed the persistent N + R arena,
        // but may not exceed the graph's reserved N + U extent.
        GGML_ASSERT(attention_stride >= tail->retention() &&
                attention_stride <= tail_attention_stride);
    } else {
        GGML_ASSERT(attention_stride >= tail_attention_stride &&
                attention_stride <= tail_arena_stride);
    }
    GGML_ASSERT(!read_idxs ||
            (read_idxs->type == GGML_TYPE_I32 && read_idxs->ne[0] == attention_stride));
    GGML_ASSERT(!body_read_idxs ||
            (body_read_idxs->type == GGML_TYPE_I32 && body_read_idxs->ne[0] == attention_stride));
    GGML_ASSERT(!bias_read_idxs ||
            (bias_read_idxs->type == GGML_TYPE_I32 && bias_read_idxs->ne[0] == attention_stride));
    GGML_ASSERT(body->ne[0] == n_kv);
    std::vector<int32_t> physical_to_read;
    if (!read_cells.empty()) {
        GGML_ASSERT(n_stream == 1 && read_cells.size() == n_kv);
        physical_to_read.assign(v_cells[0].size(), -1);
        for (uint32_t read = 0; read < n_kv; ++read) {
            if (read_cells[read] >= 0) {
                GGML_ASSERT(uint64_t(read_cells[read]) < physical_to_read.size());
                physical_to_read[size_t(read_cells[read])] = int32_t(read);
            }
        }
    }
    if (body->type == GGML_TYPE_F16) {
        set_input_kq_mask_tail_impl(*tail, tail_generations, seq_to_stream,
                static_cast<ggml_fp16_t *>(body->data), static_cast<ggml_fp16_t *>(exact->data),
                read_idxs && read_idxs->buffer ? static_cast<int32_t *>(read_idxs->data) : nullptr,
                body_read_idxs && body_read_idxs->buffer ? static_cast<int32_t *>(body_read_idxs->data) : nullptr,
                bias_read_idxs && bias_read_idxs->buffer ? static_cast<int32_t *>(bias_read_idxs->data) : nullptr,
                ubatch, n_kv, attention_stride, causal_attn,
                physical_to_read.empty() ? nullptr : &physical_to_read);
    } else {
        GGML_ASSERT(body->type == GGML_TYPE_F32);
        set_input_kq_mask_tail_impl(*tail, tail_generations, seq_to_stream,
                static_cast<float *>(body->data), static_cast<float *>(exact->data),
                read_idxs && read_idxs->buffer ? static_cast<int32_t *>(read_idxs->data) : nullptr,
                body_read_idxs && body_read_idxs->buffer ? static_cast<int32_t *>(body_read_idxs->data) : nullptr,
                bias_read_idxs && bias_read_idxs->buffer ? static_cast<int32_t *>(bias_read_idxs->data) : nullptr,
                ubatch, n_kv, attention_stride, causal_attn,
                physical_to_read.empty() ? nullptr : &physical_to_read);
    }
}

bool llama_kv_cache::can_pack_tail_body(const llama_ubatch & ubatch) const {
    if (!has_kv_body() || !tail || tail_arena_stride == 0 ||
            ubatch.n_seq_id == nullptr || ubatch.seq_id == nullptr) {
        return false;
    }

    std::vector<bool> seen(seq_to_stream.size(), false);
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // Graph reservation uses a synthetic equal-sequence ubatch whose trailing
        // tokens do not have per-token sequence metadata.  Sparse body packing is
        // an execution-time optimization and requires an explicit sequence for
        // every token; the normal body path remains valid for the reserve graph.
        if (ubatch.n_seq_id[i] <= 0 || ubatch.seq_id[i] == nullptr) {
            return false;
        }
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        if (seq_id < 0 || size_t(seq_id) >= seen.size() || seen[seq_id]) {
            continue;
        }
        seen[seq_id] = true;
        const uint32_t stream = seq_to_stream[seq_id];
        const auto & cells = v_cells[stream];
        // Graphs are cached across cache growth, state restoration, and tail
        // degradation.  Eligibility therefore cannot depend on current live
        // occupancy or coverage: a graph selected while the cache is small
        // must remain valid at its full physical capacity.  When the complete
        // stream fits, every possible visible-body subset fits as well.
        if (!llama_kv_tail_sparse_body_capacity_safe(cells.size(), tail_arena_stride)) {
            return false;
        }
    }
    return true;
}

template <typename T>
static void set_input_tail_body_plan_impl(
        const llama_kv_tail_store & tail,
        const std::vector<llama_kv_cells> & v_cells,
        const std::vector<uint32_t> & seq_to_stream,
        const int32_t * query_order, int32_t * run_desc, const T * body_mask,
        const llama_ubatch * ubatch, int64_t q_max, int64_t n_active,
        int64_t desc_stride, int64_t n_kv, int64_t n_query, int64_t n_stream,
        uint32_t attention_stride, uint32_t arena_stride,
        uint32_t n_swa, llama_swa_type swa_type, bool causal_attn) {
    std::vector<llama_kv_tail_body_row> rows;
    std::vector<llama_kv_tail_query_window> query_windows;
    std::vector<uint8_t> selected_rows;
    rows.reserve(arena_stride);
    query_windows.reserve(q_max);
    selected_rows.reserve(arena_stride);

    for (int64_t ia = 0; ia < n_active; ++ia) {
        int32_t * desc = run_desc + ia*desc_stride;
        const llama_seq_id seq_id = desc[0];
        const int32_t n_queries = desc[2];
        GGML_ASSERT(seq_id >= 0 && size_t(seq_id) < seq_to_stream.size());
        GGML_ASSERT(n_queries > 0 && n_queries <= q_max);
        const int32_t iq_global_last = query_order[ia*q_max + n_queries - 1];
        GGML_ASSERT(iq_global_last >= 0 && iq_global_last < n_query*n_stream);
        const int64_t is = iq_global_last/n_query;
        const int64_t iq = iq_global_last - is*n_query;
        const uint32_t stream = seq_to_stream[seq_id];
        const auto & cells = v_cells[stream];

        const auto candidates = tail.source_candidates(seq_id);
        const size_t candidate_first = q_max == 1 && candidates.size() > tail.retention() ?
                candidates.size() - std::min<size_t>(tail.retention(), candidates.size()) : 0;
        GGML_ASSERT(candidates.size() - candidate_first <= attention_stride);
        desc[4] = int32_t(candidates.size() - candidate_first);
        desc[5] = -1;
        std::fill(desc + 6, desc + 6 + attention_stride, -1);
        for (size_t i = candidate_first; i < candidates.size(); ++i) {
            desc[6 + i - candidate_first] = candidates[i].slot;
        }

        const int64_t body_map_offset = 6 + attention_stride;
        const bool pack_body = desc_stride > body_map_offset;
        if (!pack_body) {
            GGML_ASSERT(desc_stride == body_map_offset);
            continue;
        }
        GGML_ASSERT(desc_stride >= body_map_offset + arena_stride);

        rows.clear();
        for (uint32_t cell = 0; cell < uint32_t(n_kv) && cell < cells.size(); ++cell) {
            if (cells.is_empty(cell) || !cells.seq_has(cell, seq_id)) {
                continue;
            }
            const uint64_t flat = uint64_t(is)*n_kv + cell;
            GGML_ASSERT(flat <= uint64_t(INT32_MAX));
            rows.push_back({ cells.pos_get(cell), cell, int32_t(flat) });
        }
        std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
            return a.position < b.position || (a.position == b.position && a.cell < b.cell);
        });
        if (n_swa == 0 || swa_type == LLAMA_SWA_TYPE_NONE) {
            // Full-context causal visibility is monotonic, so the final query
            // remains the O(n_kv) union oracle used before sparse SWA packing.
            rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const auto & row) {
                const uint64_t mask_index = (uint64_t(is)*n_query + iq)*n_kv + row.cell;
                return !kv_tail_mask_finite(body_mask[mask_index]);
            }), rows.end());
        } else {
            // Candidate intervals bound work by the union of SWA windows. The
            // authoritative post-tail mask still decides membership, including
            // causal/M-RoPE details, without scanning query*n_kv cells.
            query_windows.clear();
            for (int32_t iq_run = 0; iq_run < n_queries; ++iq_run) {
                const int32_t iq_global = query_order[ia*q_max + iq_run];
                GGML_ASSERT(iq_global >= 0 && iq_global < n_query*n_stream);
                const int64_t query_stream = iq_global/n_query;
                GGML_ASSERT(query_stream == is);
                query_windows.push_back({ uint32_t(iq_global), ubatch->pos[iq_global] });
            }
            llama_kv_tail_union_swa_rows(rows, query_windows, n_swa, swa_type, causal_attn,
                    [&](uint32_t mask_row, uint32_t cell) {
                        return kv_tail_mask_finite(body_mask[uint64_t(mask_row)*n_kv + cell]);
                    }, selected_rows);
        }
        GGML_ASSERT(rows.size() <= arena_stride);
        desc[5] = int32_t(rows.size());
        std::fill(desc + body_map_offset, desc + desc_stride, -1);
        for (size_t i = 0; i < rows.size(); ++i) {
            desc[body_map_offset + i] = rows[i].flat;
        }
    }
}

void llama_kv_cache::set_input_tail_body_plan(
        ggml_tensor * query_order, ggml_tensor * run_desc,
        ggml_tensor * body_mask, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t attention_stride = get_tail_attention_stride(uint32_t(query_order ? query_order->ne[0] : 0));
    if (!query_order || !run_desc || !body_mask || run_desc->ne[0] < int64_t(6 + attention_stride) ||
            !query_order->buffer || !run_desc->buffer || !body_mask->buffer) {
        return;
    }
    GGML_ASSERT(tail && (run_desc->ne[0] == int64_t(6 + attention_stride) ||
            run_desc->ne[0] == int64_t(6 + attention_stride + get_tail_body_execution_stride())));
    GGML_ASSERT(query_order->type == GGML_TYPE_I32 && run_desc->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_backend_buffer_is_host(query_order->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(run_desc->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(body_mask->buffer));
    GGML_ASSERT(query_order->ne[1] == run_desc->ne[1]);
    if (body_mask->type == GGML_TYPE_F16) {
        set_input_tail_body_plan_impl(*tail, v_cells, seq_to_stream,
                static_cast<const int32_t *>(query_order->data), static_cast<int32_t *>(run_desc->data),
                static_cast<const ggml_fp16_t *>(body_mask->data), ubatch,
                query_order->ne[0], query_order->ne[1], run_desc->ne[0],
                body_mask->ne[0], body_mask->ne[1], body_mask->ne[3],
                attention_stride, tail_arena_stride, n_swa, swa_type, causal_attn);
    } else {
        GGML_ASSERT(body_mask->type == GGML_TYPE_F32);
        set_input_tail_body_plan_impl(*tail, v_cells, seq_to_stream,
                static_cast<const int32_t *>(query_order->data), static_cast<int32_t *>(run_desc->data),
                static_cast<const float *>(body_mask->data), ubatch,
                query_order->ne[0], query_order->ne[1], run_desc->ne[0],
                body_mask->ne[0], body_mask->ne[1], body_mask->ne[3],
                attention_stride, tail_arena_stride, n_swa, swa_type, causal_attn);
    }
}

ggml_tensor * llama_kv_cache_context::cpy_k_tail(
        ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * tail_idxs,
        int32_t il, ggml_tensor * dependency) const {
    return kv->cpy_k_tail(ctx, k_cur, tail_idxs, il, dependency);
}

ggml_tensor * llama_kv_cache_context::cpy_v_tail(
        ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * tail_idxs,
        int32_t il, ggml_tensor * dependency) const {
    return kv->cpy_v_tail(ctx, v_cur, tail_idxs, il, dependency);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_tail_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_tail_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_tail_body_idxs(ggml_context * ctx) const {
    return kv->build_input_tail_body_idxs(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_tail_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_tail_idxs(dst, ubatch);
}

void llama_kv_cache_context::set_input_tail_body_idxs(ggml_tensor * dst) const {
    kv->set_input_tail_body_idxs(dst);
}

void llama_kv_cache_context::set_input_k_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs_backend(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs_backend(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_kq_mask_tail(
        ggml_tensor * body, ggml_tensor * exact,
        ggml_tensor * read_idxs, ggml_tensor * body_read_idxs, ggml_tensor * bias_read_idxs,
        const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask_tail(
            body, exact, read_idxs, body_read_idxs, bias_read_idxs, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_tail_body_plan(
        ggml_tensor * query_order, ggml_tensor * run_desc,
        ggml_tensor * body_mask, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_tail_body_plan(query_order, run_desc, body_mask, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}

void llama_kv_cache_context::set_input_k_rot_backend(ggml_tensor * dst) const {
    kv->set_input_k_rot_backend(dst);
}

void llama_kv_cache_context::set_input_v_rot_backend(ggml_tensor * dst) const {
    kv->set_input_v_rot_backend(dst);
}
