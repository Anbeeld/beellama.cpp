#include "llama-kv-cache-iswa.h"

#include "llama-kv-cache-kvarn.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <string>

//
// llama_kv_cache_iswa
//

llama_kv_cache_iswa::llama_kv_cache_iswa(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                     bool   unified,
                  uint32_t   kv_size,
                  uint32_t   n_seq_max,
                  uint32_t   n_batch,
                  uint32_t   n_ubatch,
                 uint32_t   n_pad,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
          llama_kvarn_params kvarn) : hparams(model.hparams), unified(unified) {

    // chain filters
    const layer_filter_cb filter_base = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return !model.hparams.is_swa(il);
    };

    const layer_filter_cb filter_swa  = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return  model.hparams.is_swa(il);
    };

    const uint32_t size_base = kv_size;

    // note: the SWA cache is always padded to 256 for performance
    //       https://github.com/ggml-org/llama.cpp/issues/17037
    uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);

    const bool use_kvarn = kvarn.type != LLAMA_KVARN_TYPE_DISABLED;
    llama_kvarn_params kvarn_swa = kvarn;
    if (use_kvarn && kvarn.swa_key_bits != 0) {
        const std::string swa_type_name =
            "kvarn_k" + std::to_string(kvarn.swa_key_bits) +
            "v" + std::to_string(kvarn.swa_value_bits) + "_g128";
        const llama_kvarn_type swa_type = llama_kvarn_type_from_name(swa_type_name.c_str());
        GGML_ASSERT(swa_type != LLAMA_KVARN_TYPE_INVALID);

        kvarn_swa = llama_kvarn_params_for_type(swa_type);
        kvarn_swa.sinkhorn_iters      = kvarn.sinkhorn_iters;
        kvarn_swa.sink_tokens         = kvarn.sink_tokens;
        kvarn_swa.fail_if_unsupported = kvarn.fail_if_unsupported;
    }

    // when using full-size SWA cache, we set the SWA cache size to be equal to the base cache size
    if (swa_full) {
        LLAMA_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");

        size_swa = size_base;
    }
    if (use_kvarn) {
        LLAMA_LOG_INFO("%s: KVarN enabled for all layers (non-SWA %s, SWA %s sliding-window ring)\n",
                __func__, llama_kvarn_type_name(kvarn.type), llama_kvarn_type_name(kvarn_swa.type));
    }

    auto make_cache = [&](uint32_t size, uint32_t n_swa, llama_swa_type swa_type, const layer_filter_cb & layer_filter, llama_memory_t cache_mem_other, const llama_kvarn_params & cache_kvarn) -> std::unique_ptr<llama_memory_i> {
        // SWA KVarN ring requires a single stream. Non-unified mode still has one
        // stream when n_seq_max == 1, so only fall back when it actually creates
        // multiple streams.
        const bool kvarn_ok = cache_kvarn.type != LLAMA_KVARN_TYPE_DISABLED &&
            !(n_swa > 0 && swa_type != LLAMA_SWA_TYPE_NONE && n_seq_max > 1 && !unified);
        if (kvarn_ok) {
            // note: structured KVarN caches do not participate in cross-context sharing (mem_other/share)
            return std::make_unique<llama_kv_cache_kvarn>(
                    model,
                    hparams,
                    cache_kvarn,
                    offload,
                    unified,
                    size,
                    n_seq_max,
                    n_batch,
                    n_ubatch,
                    n_pad,
                    n_swa,
                    swa_type,
                    layer_filter,
                    reuse);
        }

        return std::make_unique<llama_kv_cache>(
                model, hparams, type_k, type_v,
                v_trans, offload, unified, size, n_seq_max, n_pad,
                n_swa, swa_type, cache_mem_other, layer_filter, reuse, share);
    };

    LLAMA_LOG_INFO("%s: creating non-SWA KV cache, size = %u cells\n", __func__, size_base);

    llama_memory_t mem_other_base = nullptr;
    if (mem_other) {
        mem_other_base = static_cast<llama_kv_cache_iswa *>(mem_other)->get_base();
    }

    llama_memory_t mem_other_swa = nullptr;
    if (mem_other) {
        mem_other_swa = static_cast<llama_kv_cache_iswa *>(mem_other)->get_swa();
    }

    kv_base = make_cache(size_base, 0, LLAMA_SWA_TYPE_NONE, filter_base, mem_other_base, kvarn);

    LLAMA_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, size_swa);

    kv_swa = make_cache(size_swa, hparams.n_swa, hparams.swa_type, filter_swa, mem_other_swa, kvarn_swa);
}

void llama_kv_cache_iswa::clear(bool data) {
    kv_base->clear(data);
    kv_swa ->clear(data);
}

bool llama_kv_cache_iswa::can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    return kv_base->can_seq_rm(seq_id, p0, p1) &&
           kv_swa ->can_seq_rm(seq_id, p0, p1);
}

bool llama_kv_cache_iswa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!can_seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (!kv_base->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    return kv_swa->seq_rm(seq_id, p0, p1);
}

bool llama_kv_cache_iswa::seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) {
    if (!kv_base->seq_rm_cell(seq_id, cell_idx)) {
        return false;
    }

    return kv_swa->seq_rm_cell(seq_id, cell_idx);
}

int llama_kv_cache_iswa::cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) {
    return kv_base->cells_at_pos(seq_id, pos, cell_indices, n_max);
}

void llama_kv_cache_iswa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_base->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_swa ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_iswa::seq_keep(llama_seq_id seq_id) {
    kv_base->seq_keep(seq_id);
    kv_swa ->seq_keep(seq_id);
}

void llama_kv_cache_iswa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_base->seq_add(seq_id, p0, p1, shift);
    kv_swa ->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_iswa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_base->seq_div(seq_id, p0, p1, d);
    kv_swa ->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_iswa::seq_pos_min(llama_seq_id seq_id) const {
    // the SWA cache defines the oldest position still available to every layer
    return kv_swa->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_iswa::seq_pos_max(llama_seq_id seq_id) const {
    // The accepted suffix must stay current even when some layers source their
    // live sliding-window context outside of kv_swa (for example DFlash draft
    // SWA layers reading the target cross window directly). In the normal ISWA
    // path both caches advance together, so the max remains unchanged there.
    return std::max(kv_base->seq_pos_max(seq_id), kv_swa->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_base->memory_breakdown();
    for (const auto & buft_size : kv_swa->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

bool llama_kv_cache_iswa::requires_state_for_partial_restore() const {
    return kv_base->requires_state_for_partial_restore() ||
           kv_swa->requires_state_for_partial_restore();
}

llama_memory_context_ptr llama_kv_cache_iswa::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    GGML_UNUSED(embd_all);

    // first try simple split
    do {
        if (!unified) {
            // requires equal splits, so we skip the simple split
            break;
        }

        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto ctx_base = kv_base->init_kv_batch(ubatches);
        if (!ctx_base || llama_memory_status_is_fail(ctx_base->get_status())) {
            break;
        }

        auto ctx_swa = kv_swa->init_kv_batch(ubatches);
        if (!ctx_swa || llama_memory_status_is_fail(ctx_swa->get_status())) {
            break;
        }

        return std::make_unique<llama_kv_cache_iswa_context>(
                std::move(ctx_base), std::move(ctx_swa), std::move(ubatches));
    } while (false);

    // if it fails, try equal split
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_equal(n_ubatch, !unified);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto ctx_base = kv_base->init_kv_batch(ubatches);
        if (!ctx_base || llama_memory_status_is_fail(ctx_base->get_status())) {
            break;
        }

        auto ctx_swa = kv_swa->init_kv_batch(ubatches);
        if (!ctx_swa || llama_memory_status_is_fail(ctx_swa->get_status())) {
            break;
        }

        return std::make_unique<llama_kv_cache_iswa_context>(
                std::move(ctx_base), std::move(ctx_swa), std::move(ubatches));
    } while (false);

    // TODO: if we fail again, we should attempt different splitting strategies
    //       but to do that properly, we first have to refactor the batches to be more flexible

    return std::make_unique<llama_kv_cache_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_iswa::init_full() {
    return std::make_unique<llama_kv_cache_iswa_context>(this);
}

llama_memory_context_ptr llama_kv_cache_iswa::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_iswa_context>(this, lctx, optimize);
}

uint32_t llama_kv_cache_iswa::get_kv_n_stream() const {
    return kv_base->get_kv_n_stream();
}

llama_memory_context_ptr llama_kv_cache_iswa::init_kv_batch(const std::vector<llama_ubatch> & ubatches) {
    auto ctx_base = kv_base->init_kv_batch(ubatches);
    if (!ctx_base || llama_memory_status_is_fail(ctx_base->get_status())) {
        return std::make_unique<llama_kv_cache_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    auto ctx_swa = kv_swa->init_kv_batch(ubatches);
    if (!ctx_swa || llama_memory_status_is_fail(ctx_swa->get_status())) {
        return std::make_unique<llama_kv_cache_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    return std::make_unique<llama_kv_cache_iswa_context>(
            std::move(ctx_base), std::move(ctx_swa), ubatches);
}

bool llama_kv_cache_iswa::get_can_shift() const {
    return kv_base->get_can_shift() &&
           kv_swa->get_can_shift() &&
           kv_base->get_kv_size() == kv_swa->get_kv_size();
}

bool llama_kv_cache_iswa::state_seq_restore_requires_exclusive_kv_stream() const {
    return kv_base->state_seq_restore_requires_exclusive_kv_stream() ||
           kv_swa->state_seq_restore_requires_exclusive_kv_stream();
}

void llama_kv_cache_iswa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    const bool include_base = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0 ||
                              kv_base->requires_state_for_partial_restore();
    if (include_base) {
        kv_base->state_write(io, seq_id, flags);
    }

    kv_swa->state_write(io, seq_id, flags);
}

void llama_kv_cache_iswa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    const bool include_base = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0 ||
                              kv_base->requires_state_for_partial_restore();
    if (include_base) {
        kv_base->state_read(io, seq_id, flags);
    }

    kv_swa->state_read(io, seq_id, flags);
}

llama_memory_i * llama_kv_cache_iswa::get_base() const {
    return kv_base.get();
}

llama_memory_i * llama_kv_cache_iswa::get_swa() const {
    return kv_swa.get();
}

//
// llama_kv_cache_iswa_context
//

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(llama_memory_status status) : status(status) {}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv) :
    ctx_base(kv->get_base()->init_full()),
    ctx_swa (kv->get_swa ()->init_full()),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv,
        llama_context * lctx,
        bool optimize) :
    ctx_base(kv->get_base()->init_update(lctx, optimize)),
    ctx_swa (kv->get_swa ()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_memory_context_ptr ctx_base_in,
        llama_memory_context_ptr ctx_swa_in,
        std::vector<llama_ubatch> ubatches) :
    ubatches(std::move(ubatches)),
    ctx_base(std::move(ctx_base_in)),
    ctx_swa (std::move(ctx_swa_in)),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context:: ~llama_kv_cache_iswa_context() = default;

bool llama_kv_cache_iswa_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_base->next();
    ctx_swa ->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_iswa_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_base->apply();
    res = res & ctx_swa ->apply();

    return res;
}

llama_memory_status llama_kv_cache_iswa_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_iswa_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llama_kv_cache_context * llama_kv_cache_iswa_context::get_base() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_context *>(ctx_base.get());
}

const llama_kv_cache_context * llama_kv_cache_iswa_context::get_swa()  const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_context *>(ctx_swa.get());
}
