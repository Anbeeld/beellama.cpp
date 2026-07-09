#pragma once

#include "llama.h"
#include "llama-graph.h"

#include <cstdint>
#include <map>
#include <memory>
#include <functional>
#include <vector>

struct llama_ubatch;

class llama_batch_allocr;

class llama_io_write_i;
class llama_io_read_i;

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;

    llama_context_type ctx_type;

    // fork-specific structured KVarN cache; disabled leaves upstream memory selection unchanged
    llama_kvarn_params kvarn;

    llama_memory_t mem_other;
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_NO_UPDATE,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool llama_memory_status_is_fail(llama_memory_status status);

struct llama_memory_recurrent_copy_profile {
    uint64_t layers_scanned = 0;
    uint64_t tensors_copied = 0;
    uint64_t cuda_d2d_queued = 0;
    uint64_t fallback_copies = 0;
    uint64_t enqueue_us = 0;
    uint64_t sync_us = 0;
};

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_context
//   - llama_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llama_memory_i::apply()
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;

};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llama_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    using layer_share_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llama_memory_context_i::get_status() for the result
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
        GGML_UNUSED(seq_id);
        GGML_UNUSED(p0);
        GGML_UNUSED(p1);
        return true;
    }

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual bool seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) = 0;

    // Return the number of KV cells at a given position for a seq_id.
    // If cell_indices is not NULL and n_max > 0, fill cell_indices with up to n_max cell indices.
    // Returns the total number of cells at the position (may exceed n_max).
    virtual int cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) = 0;

    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;

    // Copy only recurrent state (skip KV/attention). Used by DFlash flat-mode backup
    // where KV backup is unnecessary — flat mode rollback trims rejected positions
    // without needing a full KV restore.
    virtual void seq_cp_recurrent(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual bool seq_rm_recurrent(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
        GGML_UNUSED(seq_id);
        GGML_UNUSED(p0);
        GGML_UNUSED(p1);
        return true;
    }
    virtual void recurrent_copy_profile_reset() {}
    virtual llama_memory_recurrent_copy_profile recurrent_copy_profile() const { return {}; }
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // state write/read
    //

    // Some compact attention memories cannot be recovered from a later live state
    // by trimming arbitrary suffixes, so composite memories must include them
    // even when LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY is requested.
    virtual bool requires_state_for_partial_restore() const { return false; }

    // Some structured shared-stream memories cannot restore a per-sequence state
    // without rewriting physical stream data that may also belong to other sequences.
    virtual bool state_seq_restore_requires_exclusive_kv_stream() const { return false; }

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;

    // DFlash: force per-seq ubatch splits so each ubatch carries exactly one slot's tokens.
    // Default no-op; hybrid memories override.
    virtual void set_force_split_seq(bool /*v*/) {}

    // KV-cache-compatible hooks used by composite memories such as hybrid and iSWA.
    // Non-KV memory types keep the defaults and should not be used as attention memory.
    virtual uint32_t get_kv_n_stream() const { return 0; }
    virtual uint32_t get_kv_size() const { return 0; }
    virtual llama_memory_context_ptr init_kv_batch(const std::vector<llama_ubatch> & /* ubatches */) { return nullptr; }
};

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;
