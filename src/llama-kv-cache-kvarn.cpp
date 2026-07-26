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

using backend_kv_tail_attention_supported_t = bool (*)(
        ggml_type, ggml_type, ggml_type, ggml_type, int64_t, int64_t);
using backend_kvarn_tail_attention_supported_t = bool (*)(
        ggml_backend_dev_t,
        ggml_type, ggml_type, ggml_type, ggml_type, int64_t, int64_t);

bool kvarn_backend_supports_native_tail(
        ggml_backend_dev_t dev, ggml_type exact_type, int64_t d_k, int64_t d_v) {
    if (dev == nullptr) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto * segmented_fn = reg ? reinterpret_cast<backend_kv_tail_attention_supported_t>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_kv_tail_segmented_attention_supported")) : nullptr;
    if (!segmented_fn || !segmented_fn(
            GGML_TYPE_F16, GGML_TYPE_F16, exact_type, exact_type, d_k, d_v)) {
        return false;
    }
    auto * kvarn_fn = reg ? reinterpret_cast<backend_kvarn_tail_attention_supported_t>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_kvarn_tail_attention_supported")) : nullptr;
    if (kvarn_fn) {
        return kvarn_fn(dev, GGML_TYPE_F16, GGML_TYPE_F16,
            exact_type, exact_type, d_k, d_v);
    }
    auto * fn = reg ? reinterpret_cast<backend_kv_tail_attention_supported_t>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_kv_tail_attention_supported")) : nullptr;
    return fn && fn(GGML_TYPE_F16, GGML_TYPE_F16, exact_type, exact_type, d_k, d_v);
}

bool kvarn_backend_supports_tail_write(
        ggml_backend_dev_t dev, ggml_type exact_type, int64_t n_embd) {
    if (!dev) {
        return false;
    }
    ggml_init_params params = {
        /*.mem_size   =*/ 16*ggml_tensor_overhead() + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error("failed to create KVarN exact-tail capability context");
    }
    auto * dst = ggml_new_tensor_2d(ctx.get(), exact_type, n_embd, 16);
    auto * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, 1);
    auto * idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 1);
    return ggml_backend_dev_supports_op(dev, ggml_set_rows(ctx.get(), dst, src, idx));
}

// SWA keeps only local tail groups in F16; older window groups are served from
// records. Keep this low enough that KVarN remains a KV-memory win over q5_0.
constexpr uint32_t KVAR_N_SWA_TAIL_GROUPS = 2;
constexpr uint32_t KVAR_N_STATE_MAGIC = 0x4e52564b; // "KVRN"
// Version 13 stores exact-tail payloads component-major in contiguous physical
// slot runs. Version 12 stores canonical exact-tail payloads interleaved by row
// and remaps SWA record rings across physical ubatch layouts. Version 11: tail_groups is explicit and SWA
// stages no longer allocate a non-existent sink slot. Version 9: D256/D512 records use the full logical-head Hadamard instead of
// independent 128-wide slice rotations. Version 8: KVarN K/V stage rows and compressed records are all
// rotated-domain. Older states are rejected because their staged V rows may
// otherwise be restored in the wrong domain. Version 5 added stage_groups
// validation. Version 10 rejects states with the pre-dedup SWA record-ring layout.
constexpr uint32_t KVAR_N_STATE_VERSION_MIN = 12;
constexpr uint32_t KVAR_N_STATE_VERSION = 13;
constexpr uint32_t KVAR_N_STATE_RECORDS_FULL = 0;
constexpr uint32_t KVAR_N_STATE_STAGE_ONLY_PARTIAL = 1;

} // namespace

bool llama_kvarn_backend_supports_native_ops(ggml_backend_dev_t dev) {
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return true; // the built-in CPU backend consumes KVarN views directly
    }

    using ggml_backend_kvarn_native_ops_t = bool (*)(ggml_backend_dev_t dev);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto * fn = reg ? (ggml_backend_kvarn_native_ops_t) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_kvarn_native_ops") : nullptr;
    return fn != nullptr && fn(dev);
}

bool llama_kvarn_backend_native_attention_uses_original_v(ggml_backend_dev_t dev) {
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return false;
    }

    using ggml_backend_kvarn_native_original_v_t = bool (*)(ggml_backend_dev_t dev);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto * fn = reg ? (ggml_backend_kvarn_native_original_v_t) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_kvarn_native_original_v") : nullptr;
    return fn != nullptr && fn(dev);
}

uint32_t llama_kvarn_backend_native_rotated_max_query_tokens(ggml_backend_dev_t dev) {
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return 0;
    }

    using ggml_backend_kvarn_native_rotated_max_query_tokens_t = uint32_t (*)(ggml_backend_dev_t dev);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto * fn = reg ? (ggml_backend_kvarn_native_rotated_max_query_tokens_t)
        ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_kvarn_native_rotated_max_query_tokens") : nullptr;
    return fn != nullptr ? fn(dev) : 0;
}

bool llama_kvarn_backend_mixed_tail_native_preferred(ggml_backend_dev_t dev) {
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return true;
    }

    using ggml_backend_kvarn_mixed_tail_native_preferred_t = bool (*)(
        ggml_backend_dev_t dev);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto * fn = reg ? (ggml_backend_kvarn_mixed_tail_native_preferred_t)
        ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_kvarn_mixed_tail_native_preferred") : nullptr;
    return fn == nullptr || fn(dev);
}

bool llama_kvarn_backend_supports_ops(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return true; // the built-in CPU backend implements store + materialize
    }

    using ggml_backend_kvarn_ops_t = bool (*)(ggml_backend_dev_t dev);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto * fn = reg ? (ggml_backend_kvarn_ops_t) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_kvarn_ops") : nullptr;
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

void read_kvarn_swa_records(
        llama_io_read_i & io,
        ggml_tensor * tensor,
        uint32_t saved_groups,
        uint32_t current_groups,
        llama_pos saved_pos_max,
        bool on_device) {
    const size_t group_bytes = tensor->nb[2];
    uint64_t saved_size;
    io.read(&saved_size, sizeof(saved_size));
    if (saved_size != uint64_t(saved_groups)*group_bytes) {
        throw std::runtime_error("mismatched KVarN SWA record tensor size");
    }
    if (on_device) {
        if (saved_groups != current_groups) {
            throw std::runtime_error("on-device KVarN SWA state cannot remap record-ring depth");
        }
        io.read_tensor(tensor, 0, saved_size);
        return;
    }
    std::vector<uint8_t> saved(saved_size);
    if (!saved.empty()) {
        io.read(saved.data(), saved.size());
    }
    io.stage_tensor_clear(tensor, 0, ggml_nbytes(tensor));
    if (saved_pos_max < llama_pos(KVAR_N_GROUP - 1)) {
        return;
    }

    const int64_t last_complete = (int64_t(saved_pos_max) + 1) / KVAR_N_GROUP - 1;
    const int64_t first_saved = std::max<int64_t>(0, last_complete - int64_t(saved_groups) + 1);
    const int64_t first_current = std::max<int64_t>(first_saved, last_complete - int64_t(current_groups) + 1);
    for (int64_t group = first_current; group <= last_complete; ++group) {
        const uint32_t src = uint32_t(group % saved_groups);
        const uint32_t dst = uint32_t(group % current_groups);
        io.stage_tensor_set(
                tensor,
                saved.data() + size_t(src)*group_bytes,
                size_t(dst)*group_bytes,
                group_bytes);
    }
}

void zero_kvarn_tensor_range(llama_io_read_i & io, ggml_tensor * tensor, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    io.stage_tensor_clear(tensor, offset, size);
}

struct kvarn_tail_tensor_span {
    size_t offset;
    size_t size;
};

kvarn_tail_tensor_span kvarn_tail_checked_span(
        ggml_tensor * tensor, int32_t slot_begin, uint32_t length, uint64_t row_size) {
    if (!tensor || slot_begin < 0 || length == 0 || row_size == 0) {
        throw std::runtime_error("invalid KVarN exact-tail tensor span");
    }
    const uint64_t slot = uint64_t(slot_begin);
    if (slot > uint64_t(std::numeric_limits<size_t>::max())/row_size ||
            uint64_t(length) > uint64_t(std::numeric_limits<size_t>::max())/row_size) {
        throw std::overflow_error("KVarN exact-tail tensor span overflows size_t");
    }
    const size_t offset = size_t(slot*row_size);
    const size_t size = size_t(uint64_t(length)*row_size);
    if (offset > size_t(ggml_nbytes(tensor)) || size > size_t(ggml_nbytes(tensor)) - offset) {
        throw std::runtime_error("KVarN exact-tail tensor span exceeds its allocation");
    }
    return { offset, size };
}

uint64_t kvarn_tail_checked_bytes(uint32_t payloads, uint64_t row_size) {
    if (row_size != 0 && uint64_t(payloads) > std::numeric_limits<uint64_t>::max()/row_size) {
        throw std::overflow_error("KVarN exact-tail state byte count overflows uint64_t");
    }
    return uint64_t(payloads)*row_size;
}

void kvarn_tail_add_bytes(uint64_t & total, uint64_t bytes) {
    if (bytes > std::numeric_limits<uint64_t>::max() - total) {
        throw std::overflow_error("KVarN exact-tail state byte total overflows uint64_t");
    }
    total += bytes;
}

size_t read_kvarn_exact_tail_v12_interleaved(
        llama_io_read_i & io,
        ggml_tensor * k_tail,
        ggml_tensor * v_tail,
        uint64_t k_tail_row,
        uint64_t v_tail_row,
        const std::vector<std::vector<int32_t>> & destinations,
        bool on_device) {
    size_t tensor_ops = 0;
    for (const auto & payload_destinations : destinations) {
        if (k_tail) {
            if (on_device) {
                const auto span = kvarn_tail_checked_span(k_tail, payload_destinations[0], 1, k_tail_row);
                io.read_tensor(k_tail, span.offset, span.size);
                ++tensor_ops;
            } else {
                std::vector<uint8_t> row(static_cast<size_t>(k_tail_row));
                io.read(row.data(), row.size());
                for (const int32_t slot : payload_destinations) {
                    const auto span = kvarn_tail_checked_span(k_tail, slot, 1, k_tail_row);
                    io.stage_tensor_set(k_tail, row.data(), span.offset, span.size);
                    ++tensor_ops;
                }
            }
        }
        if (v_tail) {
            if (on_device) {
                const auto span = kvarn_tail_checked_span(v_tail, payload_destinations[0], 1, v_tail_row);
                io.read_tensor(v_tail, span.offset, span.size);
                ++tensor_ops;
            } else {
                std::vector<uint8_t> row(static_cast<size_t>(v_tail_row));
                io.read(row.data(), row.size());
                for (const int32_t slot : payload_destinations) {
                    const auto span = kvarn_tail_checked_span(v_tail, slot, 1, v_tail_row);
                    io.stage_tensor_set(v_tail, row.data(), span.offset, span.size);
                    ++tensor_ops;
                }
            }
        }
    }
    return tensor_ops;
}

size_t read_kvarn_exact_tail_v13_component(
        llama_io_read_i & io,
        ggml_tensor * tensor,
        uint64_t row_size,
        const std::vector<std::vector<int32_t>> & destinations,
        bool on_device) {
    if (!tensor) {
        if (row_size != 0) {
            throw std::runtime_error("KVarN exact-tail state has a row for an absent tensor");
        }
        return 0;
    }

    const uint64_t expected_bytes = kvarn_tail_checked_bytes(uint32_t(destinations.size()), row_size);
    if (expected_bytes > uint64_t(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("KVarN exact-tail component exceeds size_t");
    }
    const size_t begin = io.n_bytes();
    size_t tensor_ops = 0;
    if (on_device) {
        // Device state buffers retain one tensor per write operation. Keep a
        // canonical row topology so a wrapped source can restore into a
        // destination with different physical run boundaries.
        for (const auto & payload_destinations : destinations) {
            if (payload_destinations.size() != 1) {
                throw std::runtime_error("on-device KVarN exact-tail state requires one destination slot per payload");
            }
            const auto span = kvarn_tail_checked_span(tensor, payload_destinations[0], 1, row_size);
            io.read_tensor(tensor, span.offset, span.size);
            ++tensor_ops;
        }
        return tensor_ops;
    }

    for (size_t payload = 0; payload < destinations.size();) {
        if (destinations[payload].size() == 1) {
            uint32_t length = 1;
            while (payload + length < destinations.size() &&
                    destinations[payload + length].size() == 1 &&
                    int64_t(destinations[payload + length][0]) ==
                            int64_t(destinations[payload + length - 1][0]) + 1) {
                ++length;
            }
            const auto span = kvarn_tail_checked_span(tensor, destinations[payload][0], length, row_size);
            io.read_tensor(tensor, span.offset, span.size);
            ++tensor_ops;
            payload += length;
            continue;
        }

        std::vector<uint8_t> row(static_cast<size_t>(row_size));
        io.read(row.data(), row.size());
        for (const int32_t slot : destinations[payload]) {
            const auto span = kvarn_tail_checked_span(tensor, slot, 1, row_size);
            io.stage_tensor_set(tensor, row.data(), span.offset, span.size);
            ++tensor_ops;
        }
        ++payload;
    }

    if (io.n_bytes() < begin || io.n_bytes() - begin != size_t(expected_bytes)) {
        throw std::runtime_error("KVarN exact-tail component byte count mismatch");
    }
    return tensor_ops;
}

size_t kvarn_exact_tail_destination_runs(
        const std::vector<std::vector<int32_t>> & destinations) {
    size_t runs = 0;
    for (size_t payload = 0; payload < destinations.size();) {
        if (destinations[payload].size() != 1) {
            runs += destinations[payload].size();
            ++payload;
            continue;
        }
        ++runs;
        do {
            ++payload;
        } while (payload < destinations.size() && destinations[payload].size() == 1 &&
                int64_t(destinations[payload][0]) == int64_t(destinations[payload - 1][0]) + 1);
    }
    return runs;
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

void kvarn_gen_hadamard(std::vector<float> & data, int n) {
    GGML_ASSERT(n == 128 || n == 256 || n == 512);
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

const std::vector<float> & kvarn_hadamard(int n) {
    static const std::vector<float> h128 = [] {
        std::vector<float> result;
        kvarn_gen_hadamard(result, 128);
        return result;
    }();
    static const std::vector<float> h256 = [] {
        std::vector<float> result;
        kvarn_gen_hadamard(result, 256);
        return result;
    }();
    static const std::vector<float> h512 = [] {
        std::vector<float> result;
        kvarn_gen_hadamard(result, 512);
        return result;
    }();

    switch (n) {
        case 128: return h128;
        case 256: return h256;
        case 512: return h512;
        default:  GGML_ABORT("unsupported KVarN Hadamard width");
    }
}

uint32_t kvarn_stage_tail_groups(uint32_t n_batch, uint32_t n_ubatch, bool is_swa) {
    if (is_swa) {
        return KVAR_N_SWA_TAIL_GROUPS;
    }

    return llama_kvarn_non_swa_tail_groups(n_batch, n_ubatch);
}

uint32_t kvarn_swa_visible_groups(uint32_t kv_size, uint32_t n_swa) {
    const uint32_t window_cells = n_swa > 0 ? std::min(kv_size, n_swa) : kv_size;
    return ((window_cells + KVAR_N_GROUP - 1u) / KVAR_N_GROUP) + 1u;
}

uint32_t kvarn_record_groups_per_stream(uint32_t kv_size, uint32_t n_ubatch, uint32_t n_swa, bool is_swa, uint32_t tail_groups) {
    if (!is_swa) {
        return (kv_size + KVAR_N_GROUP - 1u) / KVAR_N_GROUP;
    }

    GGML_UNUSED(tail_groups);
    const uint32_t visible_groups = kvarn_swa_visible_groups(kv_size, n_swa);
    const uint32_t in_flight_groups = std::max<uint32_t>(1u, (n_ubatch + KVAR_N_GROUP - 1u) / KVAR_N_GROUP);
    return std::max<uint32_t>(1u, visible_groups + in_flight_groups - 1u);
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

void llama_kv_cache_kvarn_context::graph_compute_start() {
    base()->graph_compute_start();
}

void llama_kv_cache_kvarn_context::graph_compute_finish(ggml_status compute_status) {
    base()->graph_compute_finish(compute_status);
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
    return get_k_for_attention(ctx, il, uses_native_attention(il));
}

ggml_tensor * llama_kv_cache_kvarn_context::get_k_for_attention(
        ggml_context * ctx, int32_t il, bool native_attention) const {
    const auto it = stored_k.find(cache->mapped_layer_id(il));
    GGML_ASSERT(it != stored_k.end());
    return native_attention ? get_k_native(ctx, il) :
        cache->materialize(ctx, it->second, il, get_n_kv(), current_sinfo(), false, mat_idxs);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v(ggml_context * ctx, int32_t il) const {
    return get_v_for_attention(ctx, il, uses_native_attention(il));
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v_for_attention(
        ggml_context * ctx, int32_t il, bool native_attention) const {
    const auto it = stored_v.find(cache->mapped_layer_id(il));
    GGML_ASSERT(it != stored_v.end());
    return native_attention ? get_v_native(ctx, il) :
        cache->materialize(ctx, it->second, il, get_n_kv(), current_sinfo(), true, mat_idxs);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_k_tail(ggml_context * ctx, int32_t il) const {
    return cache->get_tail(ctx, il, false);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v_tail(ggml_context * ctx, int32_t il) const {
    return cache->get_tail(ctx, il, true);
}

uint32_t llama_kv_cache_kvarn_context::get_tail_slots() const {
    return base()->get_tail_slots();
}

ggml_type llama_kv_cache_kvarn_context::get_tail_type() const {
    return base()->get_tail_type();
}

uint32_t llama_kv_cache_kvarn_context::get_tail_tokens() const {
    return base()->get_tail_tokens();
}

uint32_t llama_kv_cache_kvarn_context::get_tail_arena_stride() const {
    return base()->get_tail_arena_stride();
}

uint32_t llama_kv_cache_kvarn_context::get_tail_attention_stride(uint32_t n_query_tokens) const {
    return base()->get_tail_attention_stride(n_query_tokens);
}

uint32_t llama_kv_cache_kvarn_context::get_tail_body_execution_stride() const {
    return cache->get_metadata_cache()->get_tail_body_execution_stride();
}

uint32_t llama_kv_cache_kvarn_context::get_tail_body_execution_rows(int32_t il) const {
    return cache->get_metadata_cache()->get_tail_body_execution_rows(il);
}

bool llama_kv_cache_kvarn_context::has_compact_tail() const {
    return base()->has_compact_tail();
}

bool llama_kv_cache_kvarn_context::has_kv_body() const {
    return base()->has_kv_body();
}

bool llama_kv_cache_kvarn_context::has_kv_body(int32_t il) const {
    return cache->get_metadata_cache()->has_kv_body(il);
}

bool llama_kv_cache_kvarn_context::has_tail_current(int32_t il) const {
    return cache->get_metadata_cache()->has_tail_current(il);
}

ggml_backend_dev_t llama_kv_cache_kvarn_context::get_tail_backend(int32_t il) const {
    return cache->get_metadata_cache()->get_tail_backend(il);
}

llama_kv_tail_storage_kind llama_kv_cache_kvarn_context::get_tail_storage_kind() const {
    return base()->get_tail_storage_kind();
}

uint32_t llama_kv_cache_kvarn_context::get_tail_rollback_tokens() const {
    return base()->get_tail_rollback_tokens();
}

llama_kv_tail_route llama_kv_cache_kvarn_context::get_tail_route(int32_t il) const {
    return cache->get_tail_route(il);
}

const llama_kv_tail_layer_route * llama_kv_cache_kvarn_context::get_tail_layer_route(int32_t il) const {
    return cache->get_metadata_cache()->get_tail_layer_route(il);
}

bool llama_kv_cache_kvarn_context::get_tail_explicit_bias(int32_t il) const {
    return cache->get_tail_explicit_bias(il);
}

bool llama_kv_cache_kvarn_context::can_pack_tail_body(const llama_ubatch & ubatch) const {
    GGML_UNUSED(ubatch);
    // Structured persistent records are not row-addressable cache payloads,
    // even when a backend materializes them for attention.
    return false;
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

ggml_tensor * llama_kv_cache_kvarn_context::build_input_kvarn_rot(ggml_context * ctx, int n_rot) const {
    GGML_ASSERT(n_rot == 128 || n_rot == 256 || n_rot == 512);
    ggml_tensor * res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rot, n_rot);
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

ggml_tensor * llama_kv_cache_kvarn_context::cpy_k_with_tail(
        ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *, int32_t) const {
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_v_with_tail(
        ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *, int32_t) const {
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_k_tail(
        ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * tail_idxs,
        int32_t il, ggml_tensor * dependency) const {
    return cache->store_tail(ctx, k_cur, tail_idxs, il, false, dependency);
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_v_tail(
        ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * tail_idxs,
        int32_t il, ggml_tensor * dependency) const {
    return cache->store_tail(ctx, v_cur, tail_idxs, il, true, dependency);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return base()->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return base()->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_tail_idxs(
        ggml_context * ctx, const llama_ubatch & ubatch) const {
    return base()->build_input_tail_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_tail_body_idxs(ggml_context * ctx) const {
    return base()->build_input_tail_body_idxs(ctx);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_k_rot(ggml_context * ctx) const {
    return base()->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_v_rot(ggml_context * ctx) const {
    return base()->build_input_v_rot(ctx);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_kvarn_mat_idxs(ggml_context * ctx) const {
    // SWA ring read: one absolute token position per output cache cell.
    // Sized to the padded n_kv so the graph is reusable across batches; empty
    // window cells are marked with idx < 0 and produce zero output.
    const uint32_t n_kv = get_n_kv();
    ggml_tensor * res = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kvarn_mat_idxs_swa");
    return res;
}

void llama_kv_cache_kvarn_context::set_input_kvarn_mat_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
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

    // During graph compute the metadata cells still hold the previous committed
    // state; the current ubatch is committed only after compute. Mirror the
    // pending SWA ring slot mapping here so native KVarN views use the same
    // absolute positions as the K/V store and mask inputs for this graph.
    if (ubatch != nullptr) {
        const auto & sinfo = current_sinfo();
        if (!sinfo.empty()) {
            GGML_ASSERT(sinfo.n_stream() == 1);
            GGML_ASSERT(ubatch->n_tokens == sinfo.size()*sinfo.n_stream());

            for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
                for (uint32_t i = 0; i < sinfo.size(); ++i) {
                    GGML_ASSERT(sinfo.idxs[s][i] < n_kv);
                    data[sinfo.idxs[s][i]] = (int64_t) ubatch->pos[s*sinfo.size() + i];
                }
            }
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

void llama_kv_cache_kvarn_context::set_input_tail_idxs(
        ggml_tensor * dst, const llama_ubatch * ubatch) const {
    base()->set_input_tail_idxs(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_tail_body_idxs(ggml_tensor * dst) const {
    base()->set_input_tail_body_idxs(dst);
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

void llama_kv_cache_kvarn_context::set_input_kq_mask_tail(
        ggml_tensor * body, ggml_tensor * exact,
        ggml_tensor * read_idxs, ggml_tensor * body_read_idxs, ggml_tensor * bias_read_idxs,
        const llama_ubatch * ubatch, bool causal_attn) const {
    base()->set_input_kq_mask_tail(
            body, exact, read_idxs, body_read_idxs, bias_read_idxs, ubatch, causal_attn);
}

void llama_kv_cache_kvarn_context::set_input_tail_body_plan(
        ggml_tensor * query_order, ggml_tensor * run_desc,
        ggml_tensor * body_mask, const llama_ubatch * ubatch, bool causal_attn) const {
    base()->set_input_tail_body_plan(query_order, run_desc, body_mask, ubatch, causal_attn);
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
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT((dst->ne[0] == 128 || dst->ne[0] == 256 || dst->ne[0] == 512) && dst->ne[1] == dst->ne[0]);

    const auto & data = kvarn_hadamard((int) dst->ne[0]);
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
        uint32_t n_batch,
        uint32_t n_ubatch,
        uint32_t n_pad,
        uint32_t n_swa,
        llama_swa_type swa_type,
        const layer_filter_cb & filter,
        const layer_reuse_cb & reuse,
        uint32_t tail_tokens,
        ggml_type tail_type_requested,
        uint32_t tail_tokens_requested,
        uint32_t tail_rollback_tokens) :
    model(model),
    hparams(hparams),
    params(params),
    n_stream(unified ? 1u : n_seq_max),
    n_seq_max(n_seq_max),
    kv_size(kv_size),
    // Dynamic staging: size the lossless F16 ring from position semantics.
    // Non-SWA keeps a permanent sink slot plus the scheduler-span tail. SWA has
    // no sink, so every stage slot is part of the local tail.
    tail_groups(kvarn_stage_tail_groups(
        n_batch, n_ubatch, n_swa > 0 && swa_type != LLAMA_SWA_TYPE_NONE)),
    stage_groups((n_swa > 0 && swa_type != LLAMA_SWA_TYPE_NONE) ? tail_groups : tail_groups + 1u),
    swa(n_swa > 0 && swa_type != LLAMA_SWA_TYPE_NONE),
    // SWA: the metadata window may span one more 128-token tile than its nominal
    // size, and a batched prefill needs the union of every row's sliding window.
    // The record ring stores only tiles older than the F16 tail; loaders account
    // for the tail offset when deciding whether a ring slot is live.
    n_groups_per_stream(kvarn_record_groups_per_stream(kv_size, n_ubatch, n_swa, swa, tail_groups)),
    exact_tail_tokens(tail_tokens),
    metadata_n_pad(n_pad),
    metadata_n_swa(n_swa),
    metadata_swa_type(swa_type),
    metadata_n_ubatch(n_ubatch),
    exact_tail_tokens_requested(tail_tokens_requested),
    exact_tail_type_requested(tail_type_requested),
    exact_tail_type(tail_type_requested),
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
        nullptr,
        n_ubatch,
        tail_tokens,
        tail_type_requested,
        tail_tokens_requested,
            true,
        tail_rollback_tokens)) {
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(swa || kv_size % KVAR_N_GROUP == 0);
    GGML_ASSERT(stage_groups >= 2 && "KVarN stage depth must be at least 2");
    GGML_ASSERT(tail_groups >= 1 && tail_groups <= stage_groups &&
        "KVarN tail depth must fit within the F16 stage");
    exact_tail_type = metadata->get_tail_type();
    if (swa) {
        GGML_ASSERT(n_stream == 1 && "SWA KVarN ring requires a single-stream cache");
        const uint32_t in_flight_groups = std::max<uint32_t>(1u, (n_ubatch + KVAR_N_GROUP - 1u) / KVAR_N_GROUP);
        // Backstop for the ring-size invariant above: the record ring must have
        // enough slots for the compressed portion of the worst-case visible tile
        // span after subtracting the F16 tail and adding the active ubatch span.
        GGML_ASSERT(n_groups_per_stream + tail_groups >=
                kvarn_swa_visible_groups(kv_size, n_swa) + in_flight_groups - 1u &&
            "SWA KVarN record ring is too small for the deduplicated sliding window");
    }
    // Dynamic staging keeps the F16/compressed mix stable across physical ubatch
    // splits. Log the configured stage depth and its memory cost at cache
    // creation so regressions in the propagation are visible at startup.
    LLAMA_LOG_INFO("KVarN cache: stage_groups=%u tail_groups=%u n_batch=%u n_ubatch=%u%s\n",
            stage_groups, tail_groups, n_batch, n_ubatch, swa ? " (SWA ring)" : "");

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
            /*.mem_size   =*/ size_t((6u + 4u * n_stream) * hparams.n_layer_kv() * ggml_tensor_overhead()),
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

    const size_t k_record_size = kvarn_record_bytes(this->params.key_bits);
    const size_t v_record_size = kvarn_record_bytes(this->params.value_bits);
    const int64_t n_record_groups = int64_t(n_groups_per_stream) * n_stream;
    // Stage depth is a cache property derived from position semantics. Non-SWA
    // caches cover the logical scheduler batch plus one physical ubatch; SWA
    // keeps only the live tail while older visible tiles use the record ring.
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
        if (!llama_kvarn_backend_supports_ops(dev)) {
            throw std::runtime_error(format(
                "KVarN cache layer %u is assigned to backend %s, which cannot store and materialize KVarN records",
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
        const bool explicit_bias = model.self_attention_uses_explicit_bias(il);
        const bool native_tail = exact_tail_tokens == 0 ||
            (!explicit_bias && kvarn_backend_supports_native_tail(
                dev, exact_tail_type, head_dim_k, head_dim_v));
        const bool native_attention =
            llama_kvarn_backend_supports_native_ops(dev) && native_tail;
        const bool mixed_tail_native = native_attention &&
            llama_kvarn_backend_mixed_tail_native_preferred(dev);
        const bool native_original_v = native_attention &&
            llama_kvarn_backend_native_attention_uses_original_v(dev);
        const uint32_t native_rotated_max_query_tokens = native_attention ?
            llama_kvarn_backend_native_rotated_max_query_tokens(dev) : 0;

        const uint32_t n_head_k_sliced = n_head_kv * (uint32_t) k_slices;
        const uint32_t n_head_v_sliced = n_head_kv * (uint32_t) v_slices;
        auto * k_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, k_record_size, n_head_k_sliced, n_record_groups);
        auto * v_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, v_record_size, n_head_v_sliced, n_record_groups);
        auto * k_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, KVAR_N_GROUP, n_head_k_sliced, n_stage_tokens);
        auto * v_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, KVAR_N_GROUP, n_head_v_sliced, n_stage_tokens);
        ggml_tensor * k_tail = nullptr;
        ggml_tensor * v_tail = nullptr;

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
            native_attention,
            mixed_tail_native,
            native_original_v,
            native_rotated_max_query_tokens,
            k_records,
            v_records,
            k_stage,
            v_stage,
            k_tail,
            v_tail,
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

    uint32_t exact_slots = 0;
    std::vector<llama_kv_tail_layer_route> tail_routes;
    if (exact_tail_tokens > 0) {
        const auto tensor_buft = [](const ggml_tensor * tensor) -> ggml_backend_buffer_type_t {
            return tensor && tensor->buffer ? ggml_backend_buffer_get_type(tensor->buffer) : nullptr;
        };
        const auto buft_is_meta = [](ggml_backend_buffer_type_t buft) {
            auto * dev = buft ? ggml_backend_buft_get_device(buft) : nullptr;
            return dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_META;
        };

        // The structured records/stages are the ordinary persistent body. They
        // must exist before overlay placement is accepted, and a meta-split
        // body is rejected before any exact-tail tensor or metadata arena.
        for (const auto & layer : layers) {
            const auto k_buft = tensor_buft(layer.k_records);
            const auto v_buft = tensor_buft(layer.v_records);
            if (!k_buft || !v_buft) {
                throw std::runtime_error(format("KVarN body layer %u has no realized backend buffer", layer.il));
            }
            if (k_buft != v_buft) {
                throw std::runtime_error(format(
                        "KVarN body layer %u places K and V records on different owners", layer.il));
            }
            if (buft_is_meta(k_buft) || buft_is_meta(v_buft)) {
                const auto split = llama_meta_device_get_split_state(
                        layer.k_records, const_cast<llama_meta_device_get_split_state_userdata *>(&model.get_split_state_ud));
                throw std::runtime_error(format(
                        "KVarN exact-tail overlay rejected for layer %u: realized body uses a tensor/meta split buffer "
                        "with split descriptor %s; "
                        "--split-mode layer is supported, while --split-mode tensor currently requires "
                        "--kv-tail-tokens 0 or a full-window native-exact representation",
                        layer.il, ggml_backend_meta_split_axis_name(split.axis)));
            }
        }

        for (const auto & layer_entry : map_layer_ids) {
            // plain locals instead of structured bindings: fail_route below
            // captures logical_il, which C++17 does not allow for structured bindings
            const auto logical_il  = layer_entry.first;
            const auto layer_index = layer_entry.second;
            const auto & layer = layers.at(layer_index);
            auto * buft = tensor_buft(layer.k_records);
            auto * dev = ggml_backend_buft_get_device(buft);
            const char * backend = dev ? ggml_backend_dev_name(dev) : ggml_backend_buft_name(buft);
            const bool device_route = dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
            const bool explicit_bias = model.self_attention_uses_explicit_bias(uint32_t(logical_il));
            const auto fail_route = [&](llama_kv_tail_operation operation) {
                throw std::runtime_error(format(
                        "KVarN exact-tail route is unsupported for architecture %s group %s layer %u backend %s "
                        "body f16/f16 exact %s/%s: missing %s",
                        llm_arch_name(model.arch), swa ? "swa" : "full", uint32_t(logical_il),
                        backend ? backend : "unknown",
                        ggml_type_name(exact_tail_type), ggml_type_name(exact_tail_type),
                        llama_kv_tail_operation_name(operation)));
            };
            llama_kv_tail_route tail_route = LLAMA_KV_TAIL_ROUTE_GENERIC;
            if (device_route) {
                if (!kvarn_backend_supports_tail_write(
                            dev, exact_tail_type, uint64_t(layer.head_dim_k)*layer.n_head_kv)) {
                    fail_route(LLAMA_KV_TAIL_OP_WRITE_K);
                }
                if (!kvarn_backend_supports_tail_write(
                            dev, exact_tail_type, uint64_t(layer.head_dim_v)*layer.n_head_kv)) {
                    fail_route(LLAMA_KV_TAIL_OP_WRITE_V);
                }
            }
            if (layer.native_attention && !explicit_bias) {
                if (!kvarn_backend_supports_native_tail(
                            dev, exact_tail_type, layer.head_dim_k, layer.head_dim_v)) {
                    fail_route(LLAMA_KV_TAIL_OP_NATIVE_ATTENTION);
                }
                tail_route = LLAMA_KV_TAIL_ROUTE_NATIVE;
            }
            tail_routes.push_back({
                    uint32_t(logical_il), backend ? backend : "unknown",
                    GGML_TYPE_F16, GGML_TYPE_F16, exact_tail_type, exact_tail_type,
                    false, hparams.causal_attn, swa, explicit_bias, true,
                    tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE,
                    llama_kv_tail_packed_body_stride(
                            uint64_t(exact_tail_tokens) + metadata->get_tail_rollback_tokens(), 256),
                    dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU),
                    { true, tail_route, LLAMA_KV_TAIL_OP_NONE },
            });
        }

        for (const auto & route : tail_routes) {
            LLAMA_LOG_INFO("KV tail: group=%s layer=%u dev=%s route=%s body=kvarn/%s exact=%s/%s "
                    "presence=%s current=%s execution_rows=%u requested=%u effective=%u\n",
                    swa ? "swa" : "full", route.layer_id, route.backend.c_str(),
                    route.capability.route == LLAMA_KV_TAIL_ROUTE_NATIVE ? "native" : "generic",
                    "kvarn", ggml_type_name(route.exact_type_k), ggml_type_name(route.exact_type_v),
                    route.has_body ? "body" : "bodyless", route.has_current ? "current" : "no-current",
                    route.body_execution_rows, exact_tail_tokens_requested, exact_tail_tokens);
        }
        metadata->set_tail_routes(std::move(tail_routes));
        metadata->finalize_tail_overlay_metadata();
        exact_slots = metadata->get_tail_slots();
        if (exact_slots == 0) {
            throw std::logic_error("KVarN exact-tail metadata finalized without storage slots");
        }

        std::map<ggml_backend_buffer_type_t, ggml_context_ptr, buft_comparator> tail_ctx_map;
        const auto tail_ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
            const auto it = tail_ctx_map.find(buft);
            if (it != tail_ctx_map.end()) {
                return it->second.get();
            }
            ggml_init_params ctx_params = {
                /*.mem_size   =*/ size_t(2u*hparams.n_layer_kv()*ggml_tensor_overhead()),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr ctx { ggml_init(ctx_params) };
            if (!ctx) {
                return nullptr;
            }
            auto * result = ctx.get();
            tail_ctx_map.emplace(buft, std::move(ctx));
            return result;
        };

        for (auto & layer : layers) {
            auto * buft = tensor_buft(layer.k_records);
            auto * ctx = tail_ctx_for_buft(buft);
            if (!ctx) {
                throw std::runtime_error("failed to create KVarN exact-tail tensor context");
            }
            layer.k_tail = ggml_new_tensor_2d(
                    ctx, exact_tail_type, uint64_t(layer.head_dim_k)*layer.n_head_kv, exact_slots);
            layer.v_tail = ggml_new_tensor_2d(
                    ctx, exact_tail_type, uint64_t(layer.head_dim_v)*layer.n_head_kv, exact_slots);
            ggml_format_name(layer.k_tail, "cache_kvarn_k_tail_l%d", layer.il);
            ggml_format_name(layer.v_tail, "cache_kvarn_v_tail_l%d", layer.il);
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
                throw std::runtime_error("failed to allocate KVarN exact-tail buffer");
            }
            ggml_backend_buffer_clear(buf, 0);
            total_bytes += ggml_backend_buffer_get_size(buf);
            LLAMA_LOG_INFO("%s: %10s KVarN tail buffer size = %8.2f MiB\n",
                    __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
            ctxs_bufs.emplace_back(std::move(ctx), buf);
        }

        for (const auto & layer : layers) {
            const uintptr_t owner = reinterpret_cast<uintptr_t>(tensor_buft(layer.k_records));
            auto ownership = llama_kv_tail_plan_layer_ownership(
                    layer.il, owner, owner, true, true);
            ownership.shadow_k_owner = reinterpret_cast<uintptr_t>(tensor_buft(layer.k_tail));
            ownership.shadow_v_owner = reinterpret_cast<uintptr_t>(tensor_buft(layer.v_tail));
            const auto error = llama_kv_tail_validate_layer_ownership(ownership);
            if (error != LLAMA_KV_TAIL_OWNERSHIP_OK) {
                throw std::runtime_error(format("KVarN exact-tail ownership validation failed for layer %u (error %d)",
                        layer.il, int(error)));
            }
        }
    } else {
        metadata->set_tail_routes({});
    }

    LLAMA_LOG_INFO("%s: type = %s, layers = %zu, groups/stream = %u, streams = %u, KVarN = %.2f MiB, equivalent F16 = %.2f MiB\n",
            __func__, llama_kvarn_type_name(this->params.type), layers.size(), n_groups_per_stream, n_stream,
            total_bytes / 1024.0 / 1024.0, raw_bytes / 1024.0 / 1024.0);
}

std::unique_ptr<llama_kv_cache> llama_kv_cache_kvarn::make_metadata_cache() const {
    auto result = std::make_unique<llama_kv_cache>(
            model,
            hparams,
            GGML_TYPE_F16,
            GGML_TYPE_F16,
            false,
            false,
            n_stream == 1,
            kv_size,
            n_seq_max,
            metadata_n_pad,
            metadata_n_swa,
            metadata_swa_type,
            nullptr,
            [](int32_t) { return false; },
            nullptr,
            nullptr,
            metadata_n_ubatch,
            exact_tail_tokens,
            exact_tail_type_requested,
            exact_tail_tokens_requested,
            true,
            metadata->get_tail_rollback_tokens());

    const auto & routes = metadata->get_tail_layer_routes();
    result->set_tail_routes(std::vector<llama_kv_tail_layer_route>(routes.begin(), routes.end()));
    if (!routes.empty()) {
        result->finalize_tail_overlay_metadata();
    }
    return result;
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

llama_memory_i::seq_rm_capability llama_kv_cache_kvarn::get_seq_rm_capability() const {
    return metadata->get_seq_rm_capability();
}

void llama_kv_cache_kvarn::clear(bool data) {
    pending_stream_copies = {};
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
    if (llama_kvarn_can_remove_range(pos_max, p0, p1, KVAR_N_GROUP)) {
        return true;
    }
    if (p0 <= 0 || p1 >= 0) {
        return false;
    }

    llama_pos planned_p0 = -1;
    llama_pos planned_p1 = -1;
    return llama_kvarn_plan_remove_range(
            pos_max, p0, p1, KVAR_N_GROUP,
            stream_is_exclusive_for(seq_id), planned_p0, planned_p1) &&
           planned_p0 == p0 && planned_p1 == p1;
}

bool llama_kv_cache_kvarn::can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    return can_remove(seq_id, p0, p1);
}

bool llama_kv_cache_kvarn::seq_rm_plan(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1,
        llama_pos & planned_p0, llama_pos & planned_p1) const {
    if (seq_id < 0) {
        if (p0 > 0 || p1 >= 0) {
            return false;
        }
        planned_p0 = p0;
        planned_p1 = p1;
        return true;
    }
    if (uint32_t(seq_id) >= n_seq_max) {
        return false;
    }
    if (swa || llama_kvarn_can_remove_range(
            metadata->seq_pos_max(seq_id), p0, p1, KVAR_N_GROUP)) {
        planned_p0 = p0;
        planned_p1 = p1;
        return true;
    }
    if (p0 <= 0 || p1 >= 0) {
        return false;
    }
    return llama_kvarn_plan_remove_range(
            metadata->seq_pos_max(seq_id), p0, p1, KVAR_N_GROUP,
            stream_is_exclusive_for(seq_id), planned_p0, planned_p1);
}

bool llama_kv_cache_kvarn::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    apply_pending_stream_copies(nullptr);
    if (!can_seq_rm(seq_id, p0, p1)) {
        LLAMA_LOG_WARN("%s: KVarN can only remove a complete sequence or the current/previous fp16 tail groups\n", __func__);
        return false;
    }
    return metadata->seq_rm(seq_id, p0, p1);
}

bool llama_kv_cache_kvarn::seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) {
    apply_pending_stream_copies(nullptr);
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
    apply_pending_stream_copies(nullptr);
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
    auto tail_copies = metadata->take_pending_tail_copies();
    pending_stream_copies.tail_src_slots.insert(
            pending_stream_copies.tail_src_slots.end(),
            tail_copies.tail_src_slots.begin(), tail_copies.tail_src_slots.end());
    pending_stream_copies.tail_dst_slots.insert(
            pending_stream_copies.tail_dst_slots.end(),
            tail_copies.tail_dst_slots.begin(), tail_copies.tail_dst_slots.end());
    pending_stream_copies.tail_transaction =
            pending_stream_copies.tail_transaction || tail_copies.tail_transaction;
}

void llama_kv_cache_kvarn::seq_keep(llama_seq_id seq_id) {
    apply_pending_stream_copies(nullptr);
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

bool llama_kv_cache_kvarn::stream_is_exclusive_for(llama_seq_id seq_id) const {
    return llama_kvarn_stream_is_exclusive_for(
            n_stream, n_seq_max, seq_id,
            [&](llama_seq_id other) { return metadata->seq_pos_max(other); });
}

bool llama_kv_cache_kvarn::state_seq_can_save(llama_seq_id seq_id) const {
    return stream_is_exclusive_for(seq_id);
}

bool llama_kv_cache_kvarn::state_seq_can_restore(llama_seq_id seq_id) const {
    return stream_is_exclusive_for(seq_id);
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
    if (lctx) {
        llama_synchronize(lctx);
    }

    const size_t n_copy = pending_stream_copies.ssrc.size();
    for (size_t i = 0; i < n_copy; ++i) {
        copy_kvarn_stream(pending_stream_copies.ssrc[i], pending_stream_copies.sdst[i]);
    }

    const size_t n_tail_copy = pending_stream_copies.tail_src_slots.size();
    GGML_ASSERT(n_tail_copy == pending_stream_copies.tail_dst_slots.size());
    for (auto & layer : layers) {
        for (ggml_tensor * tensor : { layer.k_tail, layer.v_tail }) {
            if (!tensor || n_tail_copy == 0) {
                continue;
            }
            const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
            std::vector<uint8_t> rows(n_tail_copy*row_size);
            for (size_t i = 0; i < n_tail_copy; ++i) {
                const int32_t slot = pending_stream_copies.tail_src_slots[i];
                GGML_ASSERT(slot >= 0 && int64_t(slot) < tensor->ne[1]);
                ggml_backend_tensor_get(tensor, rows.data() + i*row_size, size_t(slot)*row_size, row_size);
            }
            for (size_t i = 0; i < n_tail_copy; ++i) {
                const int32_t slot = pending_stream_copies.tail_dst_slots[i];
                GGML_ASSERT(slot >= 0 && int64_t(slot) < tensor->ne[1]);
                ggml_backend_tensor_set(tensor, rows.data() + i*row_size, size_t(slot)*row_size, row_size);
            }
        }
    }

    if (pending_stream_copies.tail_transaction) {
        metadata->commit_pending_tail_copy();
    }

    pending_stream_copies.ssrc.clear();
    pending_stream_copies.sdst.clear();
    pending_stream_copies.tail_src_slots.clear();
    pending_stream_copies.tail_dst_slots.clear();
    pending_stream_copies.tail_transaction = false;
    return true;
}

llama_kv_memory_stats llama_kv_cache_kvarn::kv_memory_stats() const {
    llama_kv_memory_stats result;
    llama_kv_memory_component_stats & component = swa ? result.swa : result.global;
    for (const auto & route : metadata->get_tail_layer_routes()) {
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

    const uint64_t active_slots = uint64_t(exact_tail_tokens)*n_seq_max;
    const uint64_t rollback_slots = uint64_t(metadata->get_tail_rollback_tokens())*n_seq_max;
    const uint64_t total_tail_slots = active_slots + rollback_slots;
    const auto account_tail = [&](const ggml_tensor * tensor) {
        if (!tensor) {
            return;
        }
        const uint64_t bytes = ggml_nbytes(tensor);
        if (total_tail_slots == 0) {
            component.exact_tail_bytes += bytes;
            return;
        }
        GGML_ASSERT(uint64_t(tensor->ne[1]) >= total_tail_slots);
        const uint64_t row_bytes = tensor->nb[1];
        GGML_ASSERT(row_bytes*uint64_t(tensor->ne[1]) <= bytes);
        component.exact_tail_bytes += row_bytes*active_slots;
        component.rollback_reserve_bytes += row_bytes*rollback_slots;
        component.transient_estimate_bytes += row_bytes*metadata_n_ubatch;
    };
    for (const auto & layer : layers) {
        component.k_payload_bytes += ggml_nbytes(layer.k_records);
        component.v_payload_bytes += ggml_nbytes(layer.v_records);
        component.staging_bytes += ggml_nbytes(layer.k_stage) + ggml_nbytes(layer.v_stage);
        account_tail(layer.k_tail);
        account_tail(layer.v_tail);
    }

    uint64_t allocated = 0;
    for (const auto & [buft, size] : memory_breakdown()) {
        GGML_UNUSED(buft);
        allocated += size;
    }
    const uint64_t accounted = component.k_payload_bytes + component.v_payload_bytes +
            component.exact_tail_bytes + component.rollback_reserve_bytes + component.staging_bytes;
    component.padding_bytes = allocated > accounted ? allocated - accounted : 0;
    component.allocated_capacity_tokens = kv_size;
    return result;
}

bool llama_kv_cache_kvarn::get_kv_tail_coverage(
        uint32_t group_index, llama_seq_id seq_id, llama_kv_tail_coverage_info & out) const {
    return metadata->get_kv_tail_coverage(group_index, seq_id, out);
}

void llama_kv_cache_kvarn::reset_kv_tail_planner_timing() {
    metadata->reset_kv_tail_planner_timing();
}

uint64_t llama_kv_cache_kvarn::get_kv_tail_planner_timing_ns() const {
    return metadata->get_kv_tail_planner_timing_ns();
}

void llama_kv_cache_kvarn::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    metadata->state_write(io, seq_id, flags);
    const bool partial_state = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0 && seq_id >= 0;
    const bool body_only = (flags & LLAMA_STATE_SEQ_FLAGS_BODY_ONLY) != 0;

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
    // Record both stage and workspace depth so SWA no-sink stages and
    // non-SWA sink stages cannot be restored into each other's layout.
    io.write(&stage_groups, sizeof(stage_groups));
    io.write(&tail_groups, sizeof(tail_groups));
    const uint32_t has_exact_tail = exact_tail_tokens > 0 ? 1u : 0u;
    const int32_t state_exact_type = int32_t(exact_tail_type);
    const std::vector<int32_t> exact_payload_slots = has_exact_tail && !body_only ?
        metadata->state_tail_payload_slots(seq_id) : std::vector<int32_t>{};
    if (exact_payload_slots.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("KVarN exact-tail payload count overflows uint32_t");
    }
    const uint32_t n_exact_payloads = uint32_t(exact_payload_slots.size());
    for (const int32_t slot : exact_payload_slots) {
        if (slot < 0) {
            throw std::runtime_error("KVarN exact-tail state contains a negative source slot");
        }
    }
    const auto exact_payload_runs = llama_kv_tail_contiguous_slot_runs(exact_payload_slots);
    const bool on_device = (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) != 0;
    io.write(&has_exact_tail, sizeof(has_exact_tail));
    io.write(&exact_tail_tokens, sizeof(exact_tail_tokens));
    io.write(&state_exact_type, sizeof(state_exact_type));
    io.write(&n_exact_payloads, sizeof(n_exact_payloads));

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
    llama_pos saved_pos_max = -1;
    if (seq_id >= 0) {
        saved_pos_max = metadata->seq_pos_max(seq_id);
    } else {
        for (uint32_t id = 0; id < n_seq_max; ++id) {
            saved_pos_max = std::max(saved_pos_max, metadata->seq_pos_max(id));
        }
    }
    io.write(&saved_pos_max, sizeof(saved_pos_max));

    uint64_t exact_payload_bytes = 0;
    size_t exact_tensor_ops = 0;
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
        const uint64_t k_tail_row = layer.k_tail ? ggml_row_size(layer.k_tail->type, layer.k_tail->ne[0]) : 0;
        const uint64_t v_tail_row = layer.v_tail ? ggml_row_size(layer.v_tail->type, layer.v_tail->ne[0]) : 0;
        io.write(&k_tail_row, sizeof(k_tail_row));
        io.write(&v_tail_row, sizeof(v_tail_row));
        if (layer.k_tail) {
            kvarn_tail_add_bytes(exact_payload_bytes, kvarn_tail_checked_bytes(n_exact_payloads, k_tail_row));
            if (on_device) {
                // Match the layout-independent row topology used by the
                // on-device reader; host checkpoints use the batched runs.
                for (const int32_t slot : exact_payload_slots) {
                    const auto span = kvarn_tail_checked_span(layer.k_tail, slot, 1, k_tail_row);
                    io.write_tensor(layer.k_tail, span.offset, span.size);
                    ++exact_tensor_ops;
                }
            } else {
                for (const auto & run : exact_payload_runs) {
                    const auto span = kvarn_tail_checked_span(layer.k_tail, run.slot_begin, run.length, k_tail_row);
                    io.write_tensor(layer.k_tail, span.offset, span.size);
                    ++exact_tensor_ops;
                }
            }
        }
        if (layer.v_tail) {
            kvarn_tail_add_bytes(exact_payload_bytes, kvarn_tail_checked_bytes(n_exact_payloads, v_tail_row));
            if (on_device) {
                for (const int32_t slot : exact_payload_slots) {
                    const auto span = kvarn_tail_checked_span(layer.v_tail, slot, 1, v_tail_row);
                    io.write_tensor(layer.v_tail, span.offset, span.size);
                    ++exact_tensor_ops;
                }
            } else {
                for (const auto & run : exact_payload_runs) {
                    const auto span = kvarn_tail_checked_span(layer.v_tail, run.slot_begin, run.length, v_tail_row);
                    io.write_tensor(layer.v_tail, span.offset, span.size);
                    ++exact_tensor_ops;
                }
            }
        }
    }

    LLAMA_LOG_DEBUG(
            "%s: KVarN state save: kind=%s version=%u payloads=%u tail_bytes=%llu runs=%zu tensor_ops=%zu device=%s\n",
            __func__, partial_state ? "partial" : "full", KVAR_N_STATE_VERSION, n_exact_payloads,
            (unsigned long long) exact_payload_bytes, exact_payload_runs.size(), exact_tensor_ops,
            on_device ? "true" : "false");
}

bool llama_kv_cache_kvarn_context::uses_native_attention(int32_t il) const {
    return cache->uses_native_attention(il);
}

bool llama_kv_cache_kvarn_context::mixed_tail_native_preferred(int32_t il) const {
    return cache->mixed_tail_native_preferred(il);
}

bool llama_kv_cache_kvarn_context::native_attention_uses_original_v(int32_t il) const {
    return cache->native_attention_uses_original_v(il);
}

uint32_t llama_kv_cache_kvarn_context::native_rotated_max_query_tokens(int32_t il) const {
    return cache->native_rotated_max_query_tokens(il);
}

void llama_kv_cache_kvarn::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (has_pending_stream_copies()) {
        throw std::runtime_error("cannot restore KVarN state while a stream copy is pending");
    }

    // Parse into a restore-only metadata cache. The live metadata remains
    // untouched until every KVarN descriptor and payload has validated and the
    // outer state reader commits its queued tensor writes.
    auto metadata_prepared = make_metadata_cache();
    metadata_prepared->state_read(io, seq_id, flags);
    std::vector<std::vector<int32_t>> exact_destinations = metadata_prepared->take_restored_tail_payload_slots();
    const bool on_device = (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) != 0;

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
    if (version < KVAR_N_STATE_VERSION_MIN) {
        throw std::runtime_error(
            "incompatible KVarN cache state version 11: canonical exact-tail payloads are absent; "
            "re-save the prompt cache with this build");
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

    uint32_t state_kind;
    io.read(&state_kind, sizeof(state_kind));
    if (state_kind != KVAR_N_STATE_RECORDS_FULL && state_kind != KVAR_N_STATE_STAGE_ONLY_PARTIAL) {
        throw std::runtime_error("invalid KVarN cache state kind");
    }

    // The stage contains only the fixed sink/incomplete-group workspace. It is
    // independent of physical ubatch; SWA record-ring depth is remapped below.
    uint32_t saved_stage_groups;
    io.read(&saved_stage_groups, sizeof(saved_stage_groups));
    if (saved_stage_groups < 2) {
        throw std::runtime_error("invalid KVarN cache stage depth");
    }
    if (saved_stage_groups != stage_groups) {
        throw std::runtime_error(format(
            "KVarN cache stage depth mismatch: state has %u stage groups, cache has %u; "
            "re-save the prompt cache with the current --ubatch setting",
            saved_stage_groups, stage_groups));
    }
    uint32_t saved_tail_groups;
    io.read(&saved_tail_groups, sizeof(saved_tail_groups));
    if (saved_tail_groups != tail_groups) {
        throw std::runtime_error(format(
            "KVarN cache tail depth mismatch: state has %u tail groups, cache has %u; "
            "re-save the prompt cache with this build",
            saved_tail_groups, tail_groups));
    }

    uint32_t has_exact_tail;
    uint32_t saved_exact_tail_tokens;
    int32_t saved_exact_type;
    uint32_t n_exact_payloads;
    io.read(&has_exact_tail, sizeof(has_exact_tail));
    io.read(&saved_exact_tail_tokens, sizeof(saved_exact_tail_tokens));
    io.read(&saved_exact_type, sizeof(saved_exact_type));
    io.read(&n_exact_payloads, sizeof(n_exact_payloads));
    if (has_exact_tail != uint32_t(exact_tail_tokens > 0) ||
            saved_exact_tail_tokens != exact_tail_tokens ||
            saved_exact_type != int32_t(exact_tail_type) ||
            n_exact_payloads != exact_destinations.size()) {
        throw std::runtime_error(format(
                "KVarN exact-tail state configuration does not match the context "
                "(present %u/%u, tokens %u/%u, type %d/%d, payloads %u/%zu)",
                has_exact_tail, uint32_t(exact_tail_tokens > 0),
                saved_exact_tail_tokens, exact_tail_tokens,
                saved_exact_type, int32_t(exact_tail_type),
                n_exact_payloads, exact_destinations.size()));
    }
    for (const auto & destinations : exact_destinations) {
        if (destinations.empty()) {
            throw std::runtime_error("KVarN exact-tail state contains an unreferenced payload");
        }
        if (on_device && destinations.size() != 1) {
            throw std::runtime_error("on-device KVarN exact-tail state requires one destination slot per payload");
        }
        for (const int32_t slot : destinations) {
            if (slot < 0) {
                throw std::runtime_error("KVarN exact-tail state contains a negative destination slot");
            }
        }
    }

    uint32_t n_groups_used;
    io.read(&n_groups_used, sizeof(n_groups_used));
    if (n_groups_used == 0 || (!swa && n_groups_used > n_groups_per_stream)) {
        throw std::runtime_error("invalid KVarN cache group count");
    }
    llama_pos saved_pos_max;
    io.read(&saved_pos_max, sizeof(saved_pos_max));

    const uint32_t seq_stream = seq_id == -1 ? 0 : metadata_prepared->get_stream_for_seq(seq_id);
    if (seq_id != -1 && seq_stream >= n_stream) {
        throw std::runtime_error("invalid KVarN sequence stream");
    }
    if (state_kind == KVAR_N_STATE_STAGE_ONLY_PARTIAL) {
        if (seq_id < 0) {
            throw std::runtime_error("KVarN stage-only state requires a destination sequence");
        }
    }

    uint64_t exact_payload_bytes = 0;
    size_t exact_tensor_ops = 0;
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

            const size_t k_records_used = n_groups_used * layer.k_records_stream[stream_dst]->nb[2];
            const size_t v_records_used = n_groups_used * layer.v_records_stream[stream_dst]->nb[2];
            const size_t k_records_total = n_groups_per_stream * layer.k_records_stream[stream_dst]->nb[2];
            const size_t v_records_total = n_groups_per_stream * layer.v_records_stream[stream_dst]->nb[2];

            if (state_kind == KVAR_N_STATE_RECORDS_FULL) {
                if (swa) {
                    read_kvarn_swa_records(io, layer.k_records_stream[stream_dst],
                            n_groups_used, n_groups_per_stream, saved_pos_max, on_device);
                    read_kvarn_swa_records(io, layer.v_records_stream[stream_dst],
                            n_groups_used, n_groups_per_stream, saved_pos_max, on_device);
                } else {
                    read_kvarn_tensor_slice(io, layer.k_records_stream[stream_dst], 0, k_records_used);
                    zero_kvarn_tensor_range(io, layer.k_records_stream[stream_dst], k_records_used, k_records_total - k_records_used);

                    read_kvarn_tensor_slice(io, layer.v_records_stream[stream_dst], 0, v_records_used);
                    zero_kvarn_tensor_range(io, layer.v_records_stream[stream_dst], v_records_used, v_records_total - v_records_used);
                }
            }

            read_kvarn_tensor(io, layer.k_stage_stream[stream_dst]);
            read_kvarn_tensor(io, layer.v_stage_stream[stream_dst]);
        }
        uint64_t k_tail_row;
        uint64_t v_tail_row;
        io.read(&k_tail_row, sizeof(k_tail_row));
        io.read(&v_tail_row, sizeof(v_tail_row));
        const uint64_t expected_k_tail_row = layer.k_tail ? ggml_row_size(layer.k_tail->type, layer.k_tail->ne[0]) : 0;
        const uint64_t expected_v_tail_row = layer.v_tail ? ggml_row_size(layer.v_tail->type, layer.v_tail->ne[0]) : 0;
        if (k_tail_row != expected_k_tail_row || v_tail_row != expected_v_tail_row) {
            throw std::runtime_error("KVarN exact-tail state layer layout mismatch");
        }
        kvarn_tail_add_bytes(exact_payload_bytes, kvarn_tail_checked_bytes(n_exact_payloads, k_tail_row));
        kvarn_tail_add_bytes(exact_payload_bytes, kvarn_tail_checked_bytes(n_exact_payloads, v_tail_row));
        if (version == 12) {
            exact_tensor_ops += read_kvarn_exact_tail_v12_interleaved(
                    io, layer.k_tail, layer.v_tail, k_tail_row, v_tail_row,
                    exact_destinations, on_device);
        } else if (version == 13) {
            exact_tensor_ops += read_kvarn_exact_tail_v13_component(
                    io, layer.k_tail, k_tail_row, exact_destinations, on_device);
            exact_tensor_ops += read_kvarn_exact_tail_v13_component(
                    io, layer.v_tail, v_tail_row, exact_destinations, on_device);
        } else {
            throw std::runtime_error("unsupported KVarN exact-tail state layout");
        }
    }

    auto prepared_owner = std::make_shared<std::unique_ptr<llama_kv_cache>>(std::move(metadata_prepared));
    const size_t exact_destination_runs = kvarn_exact_tail_destination_runs(exact_destinations);
    const char * log_function = __func__;
    io.on_commit([this, prepared_owner, state_kind, version, n_exact_payloads,
                  log_function,
                  exact_payload_bytes, exact_destination_runs, exact_tensor_ops, on_device]() mutable {
        metadata.swap(*prepared_owner);
        LLAMA_LOG_DEBUG(
                "%s: KVarN state restore: kind=%s version=%u payloads=%u tail_bytes=%llu runs=%zu tensor_ops=%zu device=%s\n",
                log_function, state_kind == KVAR_N_STATE_STAGE_ONLY_PARTIAL ? "partial" : "full",
                version, n_exact_payloads, (unsigned long long) exact_payload_bytes,
                exact_destination_runs, exact_tensor_ops, on_device ? "true" : "false");
    });
}

llama_kv_cache * llama_kv_cache_kvarn::get_metadata_cache() const {
    return metadata.get();
}

int32_t llama_kv_cache_kvarn::mapped_layer_id(int32_t il) const {
    return map_layer_ids.at(il);
}

llama_kv_tail_route llama_kv_cache_kvarn::get_tail_route(int32_t il) const {
    return metadata->get_tail_route(il);
}

bool llama_kv_cache_kvarn::get_tail_explicit_bias(int32_t il) const {
    return metadata->get_tail_explicit_bias(il);
}

const llama_kv_cache_kvarn::layer & llama_kv_cache_kvarn::layer_for(int32_t il) const {
    return layers.at(map_layer_ids.at(il));
}

bool llama_kv_cache_kvarn::uses_native_attention(int32_t il) const {
    return layer_for(il).native_attention;
}

bool llama_kv_cache_kvarn::mixed_tail_native_preferred(int32_t il) const {
    return layer_for(il).mixed_tail_native;
}

bool llama_kv_cache_kvarn::native_attention_uses_original_v(int32_t il) const {
    return layer_for(il).native_original_v;
}

uint32_t llama_kv_cache_kvarn::native_rotated_max_query_tokens(int32_t il) const {
    return layer_for(il).native_rotated_max_query_tokens;
}

ggml_tensor * llama_kv_cache_kvarn::get_tail(
        ggml_context * ctx, int32_t il, bool value) const {
    const auto & layer = layer_for(il);
    ggml_tensor * tensor = value ? layer.v_tail : layer.k_tail;
    if (!tensor) {
        return nullptr;
    }
    const uint32_t head_dim = value ? layer.head_dim_v : layer.head_dim_k;
    return ggml_view_4d(ctx, tensor,
            head_dim, layer.n_head_kv, metadata->get_tail_slots(), 1,
            ggml_row_size(tensor->type, head_dim), tensor->nb[1], tensor->nb[2], 0);
}

ggml_tensor * llama_kv_cache_kvarn::store_tail(
        ggml_context * ctx, ggml_tensor * current, ggml_tensor * indices,
        int32_t il, bool value, ggml_tensor * dependency) const {
    const auto & layer = layer_for(il);
    ggml_tensor * dst = value ? layer.v_tail : layer.k_tail;
    if (!dst || !indices) {
        return nullptr;
    }
    const uint32_t head_dim = value ? layer.head_dim_v : layer.head_dim_k;
    const int64_t n_embd = int64_t(head_dim)*layer.n_head_kv;
    const int64_t n_tokens = current->ne[2];
    GGML_ASSERT(current->type == GGML_TYPE_F32 && current->ne[0] == head_dim &&
            current->ne[1] == layer.n_head_kv && dst->ne[0] == n_embd);
    GGML_ASSERT(indices->type == GGML_TYPE_I64 && indices->ne[0] == n_tokens);
    current = ggml_is_contiguous(current)
        ? ggml_reshape_2d(ctx, current, n_embd, n_tokens)
        : ggml_cont_2d(ctx, current, n_embd, n_tokens);
    ggml_tensor * written = dst;
    for (int64_t level = 0; level < indices->ne[1]; ++level) {
        ggml_tensor * level_idxs = indices->ne[1] == 1
            ? indices
            : ggml_view_1d(ctx, indices, n_tokens, level*indices->nb[1]);
        written = ggml_set_rows_ordered(ctx, written, current, level_idxs, dependency);
        dependency = written;
    }
    return ggml_view_4d(ctx, written,
            head_dim, layer.n_head_kv, metadata->get_tail_slots(), 1,
            ggml_row_size(written->type, head_dim), written->nb[1], written->nb[2], 0);
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
    result->op_params[5] = (int32_t) slices; // KVarN head-wide Hadamard slice count
    result->op_params[8] = int32_t(tail_groups);
    result->op_params[9] = 1; // commit every completed record before it leaves the live workspace
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
    result->op_params[8] = int32_t(tail_groups);
    result->op_params[9] = 1;
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

ggml_tensor * llama_kv_cache_kvarn::materialize(
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
    GGML_ASSERT(indices != nullptr);

    ggml_tensor * result = ggml_kvarn_materialize(
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
    // Materialized fallback attention stays in the same rotated domain as the
    // persistent KVarN body. This avoids a full inverse transform per layer.
    result->op_params[4] = 1;
    result->op_params[5] = int32_t(value ? layer.v_slices : layer.k_slices);
    result->op_params[6] = swa ? 1 : 0;
    result->op_params[8] = int32_t(tail_groups);
    result->op_params[9] = 1;
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
