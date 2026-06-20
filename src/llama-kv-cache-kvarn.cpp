#include "llama-kv-cache-kvarn.h"

#include "ggml-backend.h"
#include "llama-context.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace {

constexpr uint32_t KVAR_N_GROUP = 128;
constexpr uint32_t KVAR_N_STAGE_GROUPS = 3; // legacy default; production caches carry stage_groups in op_params[7]
constexpr uint32_t KVAR_N_STATE_MAGIC = 0x4e52564b; // "KVRN"
// Version 5: adds stage_groups to the state header so restore can validate the
// saved stage layout against the current cache. The initial W2 gate rejects any
// stage_groups mismatch (including version-4 states restored into a cache whose
// configured ubatch is not 256) rather than silently corrupting the stage. Stage-
// slot remap for differing save/restore ubatch settings is future work.
constexpr uint32_t KVAR_N_STATE_VERSION = 5;
constexpr uint32_t KVAR_N_STATE_RECORDS_FULL = 0;
constexpr uint32_t KVAR_N_STATE_STAGE_ONLY_PARTIAL = 1;

} // namespace

bool llama_kvarn_backend_supports_native_ops(ggml_backend_dev_t dev) {
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return false;
    }

    using ggml_backend_kvarn_native_ops_t = bool (*)(ggml_backend_dev_t dev);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto * fn = reg ? (ggml_backend_kvarn_native_ops_t) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_kvarn_native_ops") : nullptr;
    return fn != nullptr && fn(dev);
}

namespace {

size_t kvarn_record_bytes(int bits) {
    return llama_kvarn_packed_bytes(KVAR_N_GROUP * KVAR_N_GROUP, bits) +
        3 * KVAR_N_GROUP * sizeof(ggml_fp16_t);
}

void write_kvarn_tensor(llama_io_write_i & io, ggml_tensor * tensor) {
    const uint64_t size = ggml_nbytes(tensor);
    io.write(&size, sizeof(size));
    io.write_tensor(tensor, 0, size);
}

void write_kvarn_tensor_slice(llama_io_write_i & io, ggml_tensor * tensor, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= (size_t) ggml_nbytes(tensor));
    const uint64_t size64 = size;
    io.write(&size64, sizeof(size64));
    io.write_tensor(tensor, offset, size);
}

void read_kvarn_tensor(llama_io_read_i & io, ggml_tensor * tensor) {
    uint64_t size;
    io.read(&size, sizeof(size));
    if (size != (uint64_t) ggml_nbytes(tensor)) {
        throw std::runtime_error("mismatched KVarN cache tensor size");
    }
    io.read_tensor(tensor, 0, size);
}

void read_kvarn_tensor_slice(llama_io_read_i & io, ggml_tensor * tensor, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= (size_t) ggml_nbytes(tensor));
    uint64_t saved_size;
    io.read(&saved_size, sizeof(saved_size));
    if (saved_size != size) {
        throw std::runtime_error("mismatched KVarN cache tensor slice size");
    }
    io.read_tensor(tensor, offset, size);
}

void zero_kvarn_tensor_range(ggml_tensor * tensor, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    ggml_backend_tensor_memset(tensor, 0, offset, size);
}

int32_t kvarn_contiguous_tokens_per_stream_hint(const llama_kv_cache::slot_info & sinfo) {
    if (sinfo.empty() || sinfo.idxs.empty() || sinfo.idxs[0].empty()) {
        return 0;
    }

    const size_t n_tokens = sinfo.idxs[0].size();
    if (n_tokens > (size_t) std::numeric_limits<int32_t>::max()) {
        return 0;
    }

    for (const auto & idxs : sinfo.idxs) {
        if (idxs.size() != n_tokens || idxs.empty()) {
            return 0;
        }
        const uint32_t first = idxs[0];
        for (size_t i = 1; i < idxs.size(); ++i) {
            if (idxs[i] != first + (uint32_t) i) {
                return 0;
            }
        }
    }

    return (int32_t) n_tokens;
}

void kvarn_gen_hadamard_128(std::vector<float> & data) {
    constexpr int n = 128;
    data.assign(n * n, 0.0f);
    data[0] = 1.0f / std::sqrt(float(n));

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                const float val = data[i * n + j];

                data[(i + s) * n + j]       =  val;
                data[i * n + (j + s)]       =  val;
                data[(i + s) * n + (j + s)] = -val;
            }
        }
    }
}

const std::vector<float> & kvarn_hadamard_128() {
    static const std::vector<float> data = [] {
        std::vector<float> result;
        kvarn_gen_hadamard_128(result);
        return result;
    }();

    return data;
}

} // namespace

llama_kv_cache_kvarn_context::llama_kv_cache_kvarn_context(
        llama_kv_cache_kvarn * cache,
        llama_memory_context_ptr base,
        llama_context * update_lctx) :
    llama_kv_cache_context(base ? base->get_status() : LLAMA_MEMORY_STATUS_FAILED_PREPARE),
    cache(cache),
    base_ctx(std::move(base)),
    update_lctx(update_lctx) {
}

llama_kv_cache_context * llama_kv_cache_kvarn_context::base() const {
    return static_cast<llama_kv_cache_context *>(base_ctx.get());
}

bool llama_kv_cache_kvarn_context::next() {
    return base()->next();
}

bool llama_kv_cache_kvarn_context::apply() {
    if (!base()->apply()) {
        return false;
    }

    return !update_lctx || cache->apply_pending_stream_copies(update_lctx);
}

llama_memory_status llama_kv_cache_kvarn_context::get_status() const {
    const auto status = base_ctx ? base_ctx->get_status() : LLAMA_MEMORY_STATUS_FAILED_PREPARE;
    if (status == LLAMA_MEMORY_STATUS_NO_UPDATE && cache->has_pending_stream_copies()) {
        return LLAMA_MEMORY_STATUS_SUCCESS;
    }
    return status;
}

const llama_ubatch & llama_kv_cache_kvarn_context::get_ubatch() const {
    return base()->get_ubatch();
}

uint32_t llama_kv_cache_kvarn_context::get_n_kv() const {
    return base()->get_n_kv();
}

llama_kv_cache * llama_kv_cache_kvarn_context::get_kv() const {
    return cache->get_metadata_cache();
}

const llama_kv_cache::slot_info & llama_kv_cache_kvarn_context::current_sinfo() const {
    return base()->current_sinfo();
}

ggml_type llama_kv_cache_kvarn_context::type_k() const {
    return GGML_TYPE_F16;
}

ggml_type llama_kv_cache_kvarn_context::type_v() const {
    return GGML_TYPE_F16;
}

ggml_tensor * llama_kv_cache_kvarn_context::get_k(ggml_context * ctx, int32_t il) const {
    return get_k_native(ctx, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v(ggml_context * ctx, int32_t il) const {
    return get_v_native(ctx, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_k_native(ggml_context * ctx, int32_t il) const {
    const auto it = stored_k.find(cache->mapped_layer_id(il));
    GGML_ASSERT(it != stored_k.end());
    return cache->view(ctx, it->second, il, get_n_kv(), current_sinfo(), false, mat_idxs);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v_native(ggml_context * ctx, int32_t il) const {
    const auto it = stored_v.find(cache->mapped_layer_id(il));
    GGML_ASSERT(it != stored_v.end());
    return cache->view(ctx, it->second, il, get_n_kv(), current_sinfo(), true, mat_idxs);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_turbo_rotation() const {
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::get_turbo_rotation_inv() const {
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::get_turbo_rot_forward() const {
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::get_turbo_rot_inverse() const {
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_kvarn_rot(ggml_context * ctx) const {
    ggml_tensor * res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, KVAR_N_GROUP, KVAR_N_GROUP);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kvarn_rot");
    return res;
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_k(
        ggml_context * ctx,
        ggml_tensor * k_cur,
        ggml_tensor * k_idxs,
        int32_t il) const {
    auto * result = cache->store(ctx, k_cur, k_idxs, il, current_sinfo(), false);
    stored_k[cache->mapped_layer_id(il)] = result;
    return result;
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_v(
        ggml_context * ctx,
        ggml_tensor * v_cur,
        ggml_tensor * v_idxs,
        int32_t il) const {
    auto * result = cache->store(ctx, v_cur, v_idxs, il, current_sinfo(), true);
    stored_v[cache->mapped_layer_id(il)] = result;
    return result;
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return base()->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return base()->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_k_rot(ggml_context * ctx) const {
    return base()->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_v_rot(ggml_context * ctx) const {
    return base()->build_input_v_rot(ctx);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_kvarn_mat_idxs(ggml_context * ctx) const {
    // SWA ring view: one absolute token position per output cache cell.
    // Sized to the padded n_kv so the graph is reusable across batches; empty
    // window cells are marked with idx < 0 and produce zero output.
    const uint32_t n_kv = get_n_kv();
    ggml_tensor * res = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kvarn_mat_idxs_swa");
    return res;
}

void llama_kv_cache_kvarn_context::set_input_kvarn_mat_idxs(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    const auto * kv = cache->get_metadata_cache();
    const uint32_t n_kv = (uint32_t) dst->ne[0];
    int64_t * data = (int64_t *) dst->data;

    // single stream for SWA ring
    const auto & cells = kv->get_cells(0);
    for (uint32_t cell = 0; cell < n_kv; ++cell) {
        if (cells.is_empty(cell)) {
            data[cell] = -1; // empty window cell
        } else {
            data[cell] = (int64_t) cells.pos_get(cell);
        }
    }
}

void llama_kv_cache_kvarn_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (cache->is_swa()) {
        // SWA ring store: indices carry absolute token positions (one per input token).
        // The store op decodes group = idx/128, pos = idx%128 and maps the group into a
        // ring slot. The metadata cache's own cell ordering is irrelevant to the records.
        GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
        int64_t * data = (int64_t *) dst->data;
        const uint32_t n_tokens = ubatch->n_tokens;
        for (uint32_t i = 0; i < n_tokens; ++i) {
            data[i] = (int64_t) ubatch->pos[i];
        }
        return;
    }
    base()->set_input_k_idxs(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (cache->is_swa()) {
        GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
        int64_t * data = (int64_t *) dst->data;
        const uint32_t n_tokens = ubatch->n_tokens;
        for (uint32_t i = 0; i < n_tokens; ++i) {
            data[i] = (int64_t) ubatch->pos[i];
        }
        return;
    }
    base()->set_input_v_idxs(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_k_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (cache->is_swa()) {
        std::vector<int64_t> data(ubatch->n_tokens);
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            data[i] = (int64_t) ubatch->pos[i];
        }
        ggml_backend_tensor_set(dst, data.data(), 0, data.size() * sizeof(int64_t));
        return;
    }
    base()->set_input_k_idxs_backend(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_v_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (cache->is_swa()) {
        std::vector<int64_t> data(ubatch->n_tokens);
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            data[i] = (int64_t) ubatch->pos[i];
        }
        ggml_backend_tensor_set(dst, data.data(), 0, data.size() * sizeof(int64_t));
        return;
    }
    base()->set_input_v_idxs_backend(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_k_shift(ggml_tensor * dst) const {
    base()->set_input_k_shift(dst);
}

void llama_kv_cache_kvarn_context::set_input_kq_mask(
        ggml_tensor * dst,
        const llama_ubatch * ubatch,
        bool causal_attn) const {
    base()->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_kvarn_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    base()->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_k_rot(ggml_tensor * dst) const {
    base()->set_input_k_rot(dst);
}

void llama_kv_cache_kvarn_context::set_input_v_rot(ggml_tensor * dst) const {
    base()->set_input_v_rot(dst);
}

void llama_kv_cache_kvarn_context::set_input_k_rot_backend(ggml_tensor * dst) const {
    base()->set_input_k_rot_backend(dst);
}

void llama_kv_cache_kvarn_context::set_input_v_rot_backend(ggml_tensor * dst) const {
    base()->set_input_v_rot_backend(dst);
}

void llama_kv_cache_kvarn_context::set_input_kvarn_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->ne[0] == KVAR_N_GROUP && dst->ne[1] == KVAR_N_GROUP);

    const auto & data = kvarn_hadamard_128();
    memcpy(dst->data, data.data(), ggml_nbytes(dst));
}

llama_kv_cache_kvarn::llama_kv_cache_kvarn(
        const llama_model & model,
        const llama_hparams & hparams,
        llama_kvarn_params params,
        bool offload,
        bool unified,
        uint32_t kv_size,
        uint32_t n_seq_max,
        uint32_t n_ubatch,
        uint32_t n_pad,
        uint32_t n_swa,
        llama_swa_type swa_type,
        const layer_filter_cb & filter,
        const layer_reuse_cb & reuse) :
    hparams(hparams),
    params(params),
    n_stream(unified ? 1u : n_seq_max),
    // SWA: the metadata window of up to kv_size cells spans kv_size/128 + 1 tiles
    // (a sliding window is rarely tile-aligned), so the record ring needs 2 extra
    // slots over the non-SWA count to represent the oldest in-window tile without
    // a slot collision and to keep (live_group - group) < groups_per_stream.
    n_groups_per_stream(((kv_size + KVAR_N_GROUP - 1) / KVAR_N_GROUP) +
        ((n_swa > 0 && swa_type != LLAMA_SWA_TYPE_NONE) ? 2u : 0u)),
    // Dynamic staging: size the lossless F16 ring from the configured physical
    // ubatch. Non-SWA KVarN keeps at least four transient tail groups; Qwen KLD
    // is sensitive to the compression boundary at common -ub 256 settings. SWA
    // keeps the smaller ring because it has no permanent sink and stores a
    // sliding window rather than a full-context cache.
    tail_groups(std::max<uint32_t>(
        (n_swa > 0 && swa_type != LLAMA_SWA_TYPE_NONE) ? 1u : 4u,
        (n_ubatch + KVAR_N_GROUP - 1u) / KVAR_N_GROUP)),
    stage_groups(tail_groups + 1u),
    swa(n_swa > 0 && swa_type != LLAMA_SWA_TYPE_NONE),
    metadata(std::make_unique<llama_kv_cache>(
        model,
        hparams,
        GGML_TYPE_F16,
        GGML_TYPE_F16,
        false,
        false,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        nullptr,
        [](int32_t) { return false; },
        nullptr,
        nullptr)) {
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(swa || kv_size % KVAR_N_GROUP == 0);
    GGML_ASSERT(stage_groups >= 2 && "KVarN stage depth must be at least 2 (sink + one transient)");
    if (swa) {
        GGML_ASSERT(n_stream == 1 && "SWA KVarN ring requires a unified (single-stream) cache");
        // Backstop for the ring-size invariant above: the record ring must have
        // strictly more slots than the metadata window's worst-case tile span so
        // the oldest in-window tile still decodes from records.
        GGML_ASSERT(n_groups_per_stream > (kv_size + KVAR_N_GROUP - 1) / KVAR_N_GROUP &&
            "SWA KVarN record ring is too small for the sliding window");
    }
    // Dynamic staging keeps the F16/compressed mix independent of physical
    // ubatch. Log the configured stage depth and its memory cost at cache
    // creation so regressions in the propagation are visible at startup.
    LLAMA_LOG_INFO("KVarN cache: stage_groups=%u tail_groups=%u n_ubatch=%u%s\n",
            stage_groups, tail_groups, n_ubatch, swa ? " (SWA ring)" : "");

    struct buft_comparator {
        bool operator()(ggml_backend_buffer_type_t lhs, ggml_backend_buffer_type_t rhs) const {
            return std::strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        const auto it = ctx_map.find(buft);
        if (it != ctx_map.end()) {
            return it->second.get();
        }

        ggml_init_params ctx_params = {
            /*.mem_size   =*/ size_t((4u + 4u * n_stream) * hparams.n_layer_kv() * ggml_tensor_overhead()),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx { ggml_init(ctx_params) };
        if (!ctx) {
            return nullptr;
        }

        auto * result = ctx.get();
        ctx_map.emplace(buft, std::move(ctx));
        return result;
    };

    const size_t k_record_size = kvarn_record_bytes(params.key_bits);
    const size_t v_record_size = kvarn_record_bytes(params.value_bits);
    const int64_t n_record_groups = int64_t(n_groups_per_stream) * n_stream;
    // Stage depth is now a cache property derived from the configured physical
    // ubatch. Non-SWA caches keep at least four transient tail groups; SWA uses
    // the raw ubatch-derived tail because its staging area is a sliding ring.
    // Backends read stage_groups from op_params[7] instead of assuming 3.
    const int64_t n_stage_tokens = int64_t(KVAR_N_GROUP) * int64_t(stage_groups) * n_stream;
    size_t raw_bytes = 0;

    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        if (!hparams.has_kv(il)) {
            continue;
        }
        if (filter && !filter(il)) {
            continue;
        }

        auto * dev = offload ? model.dev_layer(il) : nullptr;
        if (offload && !llama_kvarn_backend_supports_native_ops(dev)) {
            throw std::runtime_error(format(
                "KVarN cache layer %u is assigned to backend %s, which has no native KVarN operations "
                "or does not meet KVarN kernel limits; use a backend with native KVarN support, "
                "disable KV offload, or enable KVarN CPU fallback",
                il, dev ? ggml_backend_dev_name(dev) : "unknown"));
        }

        auto * buft = offload ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type();
        auto * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create KVarN cache tensor context");
        }

        const uint32_t n_head_kv = hparams.n_head_kv(il);
        const uint32_t head_dim_k = hparams.n_embd_head_k(il);
        const uint32_t head_dim_v = hparams.n_embd_head_v(il);
        const int k_slices = llama_kvarn_head_slices(head_dim_k);
        const int v_slices = llama_kvarn_head_slices(head_dim_v);
        if (k_slices <= 0 || v_slices <= 0) {
            throw std::runtime_error(format(
                "KVarN cache layer %u has unsupported K/V head dimensions %u/%u",
                il, head_dim_k, head_dim_v));
        }

        const uint32_t n_head_k_sliced = n_head_kv * (uint32_t) k_slices;
        const uint32_t n_head_v_sliced = n_head_kv * (uint32_t) v_slices;
        auto * k_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, k_record_size, n_head_k_sliced, n_record_groups);
        auto * v_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, v_record_size, n_head_v_sliced, n_record_groups);
        auto * k_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, KVAR_N_GROUP, n_head_k_sliced, n_stage_tokens);
        auto * v_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, KVAR_N_GROUP, n_head_v_sliced, n_stage_tokens);

        ggml_format_name(k_records, "cache_kvarn_k_records_l%d", il);
        ggml_format_name(v_records, "cache_kvarn_v_records_l%d", il);
        ggml_format_name(k_stage, "cache_kvarn_k_stage_l%d", il);
        ggml_format_name(v_stage, "cache_kvarn_v_stage_l%d", il);

        std::vector<ggml_tensor *> k_records_stream;
        std::vector<ggml_tensor *> v_records_stream;
        std::vector<ggml_tensor *> k_stage_stream;
        std::vector<ggml_tensor *> v_stage_stream;
        k_records_stream.reserve(n_stream);
        v_records_stream.reserve(n_stream);
        k_stage_stream.reserve(n_stream);
        v_stage_stream.reserve(n_stream);

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto * k_records_view = ggml_view_3d(
                    ctx, k_records,
                    k_record_size, n_head_k_sliced, n_groups_per_stream,
                    k_records->nb[1], k_records->nb[2],
                    size_t(s) * n_groups_per_stream * k_records->nb[2]);
            auto * v_records_view = ggml_view_3d(
                    ctx, v_records,
                    v_record_size, n_head_v_sliced, n_groups_per_stream,
                    v_records->nb[1], v_records->nb[2],
                    size_t(s) * n_groups_per_stream * v_records->nb[2]);
            auto * k_stage_view = ggml_view_3d(
                    ctx, k_stage,
                    KVAR_N_GROUP, n_head_k_sliced, KVAR_N_GROUP * stage_groups,
                    k_stage->nb[1], k_stage->nb[2],
                    size_t(s) * KVAR_N_GROUP * stage_groups * k_stage->nb[2]);
            auto * v_stage_view = ggml_view_3d(
                    ctx, v_stage,
                    KVAR_N_GROUP, n_head_v_sliced, KVAR_N_GROUP * stage_groups,
                    v_stage->nb[1], v_stage->nb[2],
                    size_t(s) * KVAR_N_GROUP * stage_groups * v_stage->nb[2]);

            ggml_format_name(k_records_view, "cache_kvarn_k_records_l%d_s%d", il, s);
            ggml_format_name(v_records_view, "cache_kvarn_v_records_l%d_s%d", il, s);
            ggml_format_name(k_stage_view, "cache_kvarn_k_stage_l%d_s%d", il, s);
            ggml_format_name(v_stage_view, "cache_kvarn_v_stage_l%d_s%d", il, s);

            k_records_stream.push_back(k_records_view);
            v_records_stream.push_back(v_records_view);
            k_stage_stream.push_back(k_stage_view);
            v_stage_stream.push_back(v_stage_view);
        }

        map_layer_ids[il] = layers.size();
        layers.push_back({
            il,
            n_head_kv,
            head_dim_k,
            head_dim_v,
            (uint32_t) k_slices,
            (uint32_t) v_slices,
            k_records,
            v_records,
            k_stage,
            v_stage,
            std::move(k_records_stream),
            std::move(v_records_stream),
            std::move(k_stage_stream),
            std::move(v_stage_stream),
        });

        raw_bytes += size_t(kv_size) * n_stream * n_head_kv * (head_dim_k + head_dim_v) * sizeof(ggml_fp16_t);
    }

    if (reuse) {
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            const int32_t il_reuse = reuse(il);
            if (il_reuse < 0) {
                continue;
            }
            if (filter && !filter(il)) {
                continue;
            }
            const auto src = map_layer_ids.find(il_reuse);
            if (src == map_layer_ids.end()) {
                throw std::runtime_error(format("KVarN cache layer %u cannot reuse missing layer %d", il, il_reuse));
            }

            const auto & reused = layers.at(src->second);
            if (hparams.n_head_kv(il) != reused.n_head_kv ||
                hparams.n_embd_head_k(il) != reused.head_dim_k ||
                hparams.n_embd_head_v(il) != reused.head_dim_v) {
                throw std::runtime_error(format(
                    "KVarN cache layer %u cannot reuse layer %d with different KV shape",
                    il, il_reuse));
            }

            map_layer_ids[il] = src->second;
        }
    }

    size_t total_bytes = 0;
    for (auto & [buft, ctx] : ctx_map) {
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
            throw std::runtime_error("failed to allocate KVarN cache buffer");
        }

        ggml_backend_buffer_clear(buf, 0);
        total_bytes += ggml_backend_buffer_get_size(buf);
        LLAMA_LOG_INFO("%s: %10s KVarN buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    LLAMA_LOG_INFO("%s: type = %s, layers = %zu, groups/stream = %u, streams = %u, KVarN = %.2f MiB, equivalent F16 = %.2f MiB\n",
            __func__, llama_kvarn_type_name(params.type), layers.size(), n_groups_per_stream, n_stream,
            total_bytes / 1024.0 / 1024.0, raw_bytes / 1024.0 / 1024.0);
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    return std::make_unique<llama_kv_cache_kvarn_context>(
        this, metadata->init_batch(balloc, n_ubatch, embd_all));
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_full() {
    return std::make_unique<llama_kv_cache_kvarn_context>(this, metadata->init_full());
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_kvarn_context>(this, metadata->init_update(lctx, optimize), lctx);
}

uint32_t llama_kv_cache_kvarn::get_kv_n_stream() const {
    return metadata->get_n_stream();
}

uint32_t llama_kv_cache_kvarn::get_kv_size() const {
    return metadata->get_size();
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_kv_batch(const std::vector<llama_ubatch> & ubatches) {
    auto sinfos = metadata->prepare(ubatches);
    if (sinfos.empty()) {
        return std::make_unique<llama_kv_cache_kvarn_context>(
                this, std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE));
    }

    return std::make_unique<llama_kv_cache_kvarn_context>(
            this, std::make_unique<llama_kv_cache_context>(metadata.get(), std::move(sinfos), ubatches));
}

bool llama_kv_cache_kvarn::get_can_shift() const {
    return false;
}

void llama_kv_cache_kvarn::clear(bool data) {
    metadata->clear(false);
    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache_kvarn::can_remove(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    if (seq_id < 0) {
        return p0 <= 0 && p1 < 0;
    }

    if (swa) {
        // SWA ring: eviction is implicit (ring overwrite). The metadata cache
        // already enforces the SWA window, so defer all removal decisions to it.
        return true;
    }

    const llama_pos pos_max = metadata->seq_pos_max(seq_id);
    return llama_kvarn_can_remove_range(pos_max, p0, p1, KVAR_N_GROUP);
}

bool llama_kv_cache_kvarn::can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    return can_remove(seq_id, p0, p1);
}

bool llama_kv_cache_kvarn::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!can_seq_rm(seq_id, p0, p1)) {
        LLAMA_LOG_WARN("%s: KVarN can only remove a complete sequence or the current/previous fp16 tail groups\n", __func__);
        return false;
    }
    return metadata->seq_rm(seq_id, p0, p1);
}

bool llama_kv_cache_kvarn::seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) {
    if (swa) {
        // SWA ring: the metadata cache manages window eviction; records follow the ring.
        return metadata->seq_rm_cell(seq_id, cell_idx);
    }
    const llama_pos pos_max = metadata->seq_pos_max(seq_id);
    if (pos_max >= 0) {
        const uint32_t earliest_exact = uint32_t(std::max<llama_pos>(0, pos_max / KVAR_N_GROUP - 1) * KVAR_N_GROUP);
        if (cell_idx < earliest_exact) {
            return false;
        }
    }
    return metadata->seq_rm_cell(seq_id, cell_idx);
}

int llama_kv_cache_kvarn::cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) {
    return metadata->cells_at_pos(seq_id, pos, cell_indices, n_max);
}

void llama_kv_cache_kvarn::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    const uint32_t stream_src = metadata->get_stream_for_seq(seq_id_src);
    const uint32_t stream_dst = metadata->get_stream_for_seq(seq_id_dst);

    if (stream_src != stream_dst) {
        bool is_full = true;

        if (p0 > 0 && p0 + 1 < (int) get_kv_size()) {
            is_full = false;
        }

        if (p1 > 0 && p1 + 1 < (int) get_kv_size()) {
            is_full = false;
        }

        GGML_ASSERT(is_full && "KVarN cross-stream seq_cp() is only supported for full KV buffers");

        pending_stream_copies.ssrc.push_back(stream_src);
        pending_stream_copies.sdst.push_back(stream_dst);
    }

    metadata->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_kvarn::seq_keep(llama_seq_id seq_id) {
    metadata->seq_keep(seq_id);
}

GGML_NORETURN void llama_kv_cache_kvarn::seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) {
    GGML_ABORT("KVarN does not support position shifts");
}

GGML_NORETURN void llama_kv_cache_kvarn::seq_div(llama_seq_id, llama_pos, llama_pos, int) {
    GGML_ABORT("KVarN does not support position division");
}

llama_pos llama_kv_cache_kvarn::seq_pos_min(llama_seq_id seq_id) const {
    return metadata->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_kvarn::seq_pos_max(llama_seq_id seq_id) const {
    return metadata->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_kvarn::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        auto * buft = ggml_backend_buffer_get_type(buf.get());
        result[buft] += hparams.no_alloc
            ? ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft)
            : ggml_backend_buffer_get_size(buf.get());
    }
    return result;
}

bool llama_kv_cache_kvarn::requires_state_for_partial_restore() const {
    return true;
}

bool llama_kv_cache_kvarn::state_seq_restore_requires_exclusive_kv_stream() const {
    return get_kv_n_stream() == 1;
}

bool llama_kv_cache_kvarn::has_pending_stream_copies() const {
    return !pending_stream_copies.empty();
}

void llama_kv_cache_kvarn::copy_kvarn_stream(uint32_t stream_src, uint32_t stream_dst) {
    GGML_ASSERT(stream_src < n_stream);
    GGML_ASSERT(stream_dst < n_stream);
    GGML_ASSERT(stream_src != stream_dst);

    LLAMA_LOG_DEBUG("%s: copying KVarN stream %u to stream %u\n", __func__, stream_src, stream_dst);

    for (auto & layer : layers) {
        ggml_backend_tensor_copy(layer.k_records_stream[stream_src], layer.k_records_stream[stream_dst]);
        ggml_backend_tensor_copy(layer.v_records_stream[stream_src], layer.v_records_stream[stream_dst]);
        ggml_backend_tensor_copy(layer.k_stage_stream[stream_src], layer.k_stage_stream[stream_dst]);
        ggml_backend_tensor_copy(layer.v_stage_stream[stream_src], layer.v_stage_stream[stream_dst]);
    }
}

bool llama_kv_cache_kvarn::apply_pending_stream_copies(llama_context * lctx) {
    if (pending_stream_copies.empty()) {
        return true;
    }

    GGML_ASSERT(pending_stream_copies.ssrc.size() == pending_stream_copies.sdst.size());
    llama_synchronize(lctx);

    const size_t n_copy = pending_stream_copies.ssrc.size();
    for (size_t i = 0; i < n_copy; ++i) {
        copy_kvarn_stream(pending_stream_copies.ssrc[i], pending_stream_copies.sdst[i]);
    }

    pending_stream_copies.ssrc.clear();
    pending_stream_copies.sdst.clear();
    return true;
}

void llama_kv_cache_kvarn::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    metadata->state_write(io, seq_id, flags);
    const bool partial_state = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0 && seq_id >= 0;

    std::vector<uint32_t> saved_streams;
    if (seq_id == -1) {
        saved_streams.reserve(n_stream);
        for (uint32_t stream = 0; stream < n_stream; ++stream) {
            saved_streams.push_back(stream);
        }
    } else {
        const uint32_t stream = metadata->get_stream_for_seq(seq_id);
        GGML_ASSERT(stream < n_stream);
        saved_streams.push_back(stream);
    }

    io.write(&KVAR_N_STATE_MAGIC, sizeof(KVAR_N_STATE_MAGIC));
    io.write(&KVAR_N_STATE_VERSION, sizeof(KVAR_N_STATE_VERSION));
    const int32_t type = params.type;
    const uint32_t n_layers = layers.size();
    const uint32_t n_saved_streams = saved_streams.size();
    io.write(&type, sizeof(type));
    io.write(&n_layers, sizeof(n_layers));
    io.write(&n_saved_streams, sizeof(n_saved_streams));
    for (const uint32_t stream : saved_streams) {
        io.write(&stream, sizeof(stream));
    }
    const uint32_t state_kind = partial_state ? KVAR_N_STATE_STAGE_ONLY_PARTIAL : KVAR_N_STATE_RECORDS_FULL;
    io.write(&state_kind, sizeof(state_kind));
    // Version 5: record the stage depth so the reader can validate that the saved
    // stage tensor layout matches the current cache. The W2 gate rejects any
    // mismatch; remap for differing save/restore ubatch is future work.
    io.write(&stage_groups, sizeof(stage_groups));

    // n_groups_used is single-valued across all saved streams. This is correct
    // because when seq_id >= 0, saved_streams has exactly 1 entry (the stream
    // for that sequence), so n_groups_used applies to that one stream only.
    // When seq_id == -1, n_groups_used equals n_groups_per_stream (no compression).
    // If multi-stream partial writes are ever added, n_groups_used must become
    // per-stream.
    uint32_t n_groups_used = n_groups_per_stream;
    if (seq_id >= 0 && !swa) {
        const llama_pos pos_max = metadata->seq_pos_max(seq_id);
        if (pos_max >= 0) {
            // ceil((pos_max + 1) / KVAR_N_GROUP) — covers all groups that the
            // native view might read. After restore the next store will
            // compute live_group ≤ pos_max / KVAR_N_GROUP < n_groups_used, so
            // every accessed group stays within the restored range.
            n_groups_used = std::min(n_groups_per_stream, (uint32_t) ((pos_max + KVAR_N_GROUP) / KVAR_N_GROUP));
        }
    }
    // SWA ring: always serialize all ring slots — the live window may wrap around
    // and occupy any slot, so partial serialization by group index is not meaningful.
    io.write(&n_groups_used, sizeof(n_groups_used));

    for (const auto & layer : layers) {
        io.write(&layer.il, sizeof(layer.il));
        for (const uint32_t stream : saved_streams) {
            io.write(&stream, sizeof(stream));

            if (state_kind == KVAR_N_STATE_RECORDS_FULL) {
                const size_t k_records_used = n_groups_used * layer.k_records_stream[stream]->nb[2];
                const size_t v_records_used = n_groups_used * layer.v_records_stream[stream]->nb[2];
                write_kvarn_tensor_slice(io, layer.k_records_stream[stream], 0, k_records_used);
                write_kvarn_tensor_slice(io, layer.v_records_stream[stream], 0, v_records_used);
            }
            write_kvarn_tensor(io, layer.k_stage_stream[stream]);
            write_kvarn_tensor(io, layer.v_stage_stream[stream]);
        }
    }
}

void llama_kv_cache_kvarn::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    metadata->state_read(io, seq_id, flags);

    uint32_t magic;
    uint32_t version;
    int32_t type;
    uint32_t n_layers;
    io.read(&magic, sizeof(magic));
    io.read(&version, sizeof(version));
    io.read(&type, sizeof(type));
    io.read(&n_layers, sizeof(n_layers));
    if (magic != KVAR_N_STATE_MAGIC || version > KVAR_N_STATE_VERSION ||
        type != params.type || n_layers != layers.size()) {
        throw std::runtime_error("incompatible KVarN cache state");
    }

    if (version == 1) {
        for (const auto & layer : layers) {
            uint32_t il;
            io.read(&il, sizeof(il));
            if (il != layer.il) {
                throw std::runtime_error("mismatched KVarN cache layer");
            }

            for (auto * tensor : { layer.k_records, layer.v_records, layer.k_stage, layer.v_stage }) {
                read_kvarn_tensor(io, tensor);
            }
        }
        return;
    }

    uint32_t n_saved_streams;
    io.read(&n_saved_streams, sizeof(n_saved_streams));
    if (n_saved_streams == 0 || n_saved_streams > n_stream) {
        throw std::runtime_error("invalid KVarN cache stream count");
    }

    std::vector<uint32_t> saved_streams(n_saved_streams);
    for (uint32_t & stream : saved_streams) {
        io.read(&stream, sizeof(stream));
        if (stream >= n_stream) {
            throw std::runtime_error("invalid KVarN cache stream");
        }
    }

    uint32_t state_kind = KVAR_N_STATE_RECORDS_FULL;
    if (version >= 4) {
        io.read(&state_kind, sizeof(state_kind));
        if (state_kind != KVAR_N_STATE_RECORDS_FULL && state_kind != KVAR_N_STATE_STAGE_ONLY_PARTIAL) {
            throw std::runtime_error("invalid KVarN cache state kind");
        }
    }

    // Version 5 carries stage_groups. Version 4 states were written with the
    // legacy three-slot stage layout. The initial W2 gate does not implement
    // stage-slot remap on restore (the plan's full save/restore matrix allows
    // remapping logical live groups into differently-sized stages). For now,
    // any stage_groups mismatch — including version-4 states restored into a
    // cache whose configured ubatch is not 256 — is rejected explicitly so the
    // stage is never silently corrupted. Remap support is a follow-up.
    uint32_t saved_stage_groups = KVAR_N_STAGE_GROUPS;
    if (version >= 5) {
        io.read(&saved_stage_groups, sizeof(saved_stage_groups));
        if (saved_stage_groups < 2) {
            throw std::runtime_error("invalid KVarN cache stage depth");
        }
    }
    if (saved_stage_groups != stage_groups) {
        throw std::runtime_error(format(
            "KVarN cache stage depth mismatch: state has %u stage groups, cache has %u; "
            "re-save the prompt cache with the current --ubatch setting",
            saved_stage_groups, stage_groups));
    }

    uint32_t n_groups_used = n_groups_per_stream;
    if (version >= 3) {
        io.read(&n_groups_used, sizeof(n_groups_used));
        if (n_groups_used == 0 || n_groups_used > n_groups_per_stream) {
            throw std::runtime_error("invalid KVarN cache group count");
        }
    }

    const uint32_t seq_stream = seq_id == -1 ? 0 : metadata->get_stream_for_seq(seq_id);
    if (seq_id != -1 && seq_stream >= n_stream) {
        throw std::runtime_error("invalid KVarN sequence stream");
    }
    if (state_kind == KVAR_N_STATE_STAGE_ONLY_PARTIAL) {
        if (seq_id < 0) {
            throw std::runtime_error("KVarN stage-only state requires a destination sequence");
        }
    }

    for (const auto & layer : layers) {
        uint32_t il;
        io.read(&il, sizeof(il));
        if (il != layer.il) {
            throw std::runtime_error("mismatched KVarN cache layer");
        }

        for (uint32_t i = 0; i < n_saved_streams; ++i) {
            uint32_t stream;
            io.read(&stream, sizeof(stream));
            if (stream != saved_streams[i]) {
                throw std::runtime_error("mismatched KVarN cache stream");
            }

            const uint32_t stream_dst = seq_id == -1 ? stream : seq_stream;

            if (version >= 3) {
                const size_t k_records_used = n_groups_used * layer.k_records_stream[stream_dst]->nb[2];
                const size_t v_records_used = n_groups_used * layer.v_records_stream[stream_dst]->nb[2];
                const size_t k_records_total = n_groups_per_stream * layer.k_records_stream[stream_dst]->nb[2];
                const size_t v_records_total = n_groups_per_stream * layer.v_records_stream[stream_dst]->nb[2];

                if (state_kind == KVAR_N_STATE_RECORDS_FULL) {
                    read_kvarn_tensor_slice(io, layer.k_records_stream[stream_dst], 0, k_records_used);
                    zero_kvarn_tensor_range(layer.k_records_stream[stream_dst], k_records_used, k_records_total - k_records_used);

                    read_kvarn_tensor_slice(io, layer.v_records_stream[stream_dst], 0, v_records_used);
                    zero_kvarn_tensor_range(layer.v_records_stream[stream_dst], v_records_used, v_records_total - v_records_used);
                }

                read_kvarn_tensor(io, layer.k_stage_stream[stream_dst]);
                read_kvarn_tensor(io, layer.v_stage_stream[stream_dst]);
            } else {
                for (auto * tensor : {
                        layer.k_records_stream[stream_dst],
                        layer.v_records_stream[stream_dst],
                        layer.k_stage_stream[stream_dst],
                        layer.v_stage_stream[stream_dst],
                    }) {
                    read_kvarn_tensor(io, tensor);
                }
            }
        }
    }
}

llama_kv_cache * llama_kv_cache_kvarn::get_metadata_cache() const {
    return metadata.get();
}

int32_t llama_kv_cache_kvarn::mapped_layer_id(int32_t il) const {
    return map_layer_ids.at(il);
}

const llama_kv_cache_kvarn::layer & llama_kv_cache_kvarn::layer_for(int32_t il) const {
    return layers.at(map_layer_ids.at(il));
}

ggml_tensor * llama_kv_cache_kvarn::store(
        ggml_context * ctx,
        ggml_tensor * current,
        ggml_tensor * indices,
        int32_t il,
        const llama_kv_cache::slot_info & sinfo,
        bool value) const {
    const auto & layer = layer_for(il);
    if (!ggml_is_contiguous(current)) {
        current = ggml_cont(ctx, current);
    }

    const uint32_t head_dim = value ? layer.head_dim_v : layer.head_dim_k;
    const uint32_t slices = value ? layer.v_slices : layer.k_slices;
    GGML_ASSERT((uint32_t) current->ne[0] == head_dim);
    GGML_ASSERT((uint32_t) current->ne[1] == layer.n_head_kv);
    if (slices > 1) {
        current = ggml_reshape_3d(ctx, current, KVAR_N_GROUP, layer.n_head_kv * slices, current->ne[2]);
    }

    ggml_tensor * result = ggml_kvarn_store(
        ctx,
        current,
        indices,
        value ? layer.v_stage : layer.k_stage,
        value ? layer.v_records : layer.k_records,
        value ? params.value_bits : params.key_bits,
        params.sinkhorn_iters,
        value,
        int32_t(stage_groups));
    result->op_params[3] = kvarn_contiguous_tokens_per_stream_hint(sinfo);
    result->op_params[4] = swa ? 1 : 0; // SWA sliding-window ring store
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::view(
        ggml_context * ctx,
        ggml_tensor * stored,
        int32_t il,
        uint32_t n_kv,
        const llama_kv_cache::slot_info & sinfo,
        bool value,
        ggml_tensor * mat_idxs) const {
    const auto & layer = layer_for(il);
    const uint32_t stream_start = sinfo.s0;
    const uint32_t stream_count = sinfo.s1 - sinfo.s0 + 1;
    ggml_tensor * indices = swa ? mat_idxs : stored->src[1];

    ggml_tensor * result = ggml_kvarn_view(
        ctx,
        value ? layer.v_records : layer.k_records,
        stored,
        indices,
        n_kv,
        stream_start,
        stream_count,
        value ? params.value_bits : params.key_bits,
        value,
        int32_t(stage_groups));
    result->op_params[6] = swa ? 1 : 0;
    const uint32_t slices = value ? layer.v_slices : layer.k_slices;
    if (slices > 1) {
        result = ggml_reshape_4d(
                ctx,
                result,
                value ? layer.head_dim_v : layer.head_dim_k,
                layer.n_head_kv,
                n_kv,
                stream_count);
    }

    return result;
}
