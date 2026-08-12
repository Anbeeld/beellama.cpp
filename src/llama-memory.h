#pragma once

#include "llama.h"
#include "llama-graph.h"
#include "llama-kv-memory-stats.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <functional>
#include <initializer_list>
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

    uint32_t  kv_tail_tokens;
    uint32_t  kv_tail_tokens_swa;
    uint32_t  kv_tail_tokens_requested;
    uint32_t  kv_tail_tokens_swa_requested;
    bool      kv_tail_native_exact;
    bool      kv_tail_native_exact_swa;
    uint32_t  kv_tail_rollback_tokens;
    ggml_type kv_tail_type;

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

    // Compact cache metadata is prepared by apply(), but is not authoritative
    // until the graph finishes. Wrappers must forward both hooks to their
    // participating child contexts.
    virtual void graph_compute_start() {}
    virtual void graph_compute_finish(ggml_status /* status */) {}

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

    struct seq_rm_capability {
        bool full_clear = true;
        bool arbitrary_ranges = true;
        uint32_t suffix_rollback_tokens = UINT32_MAX;
    };

    virtual seq_rm_capability get_seq_rm_capability() const { return {}; }

    virtual bool seq_rm_plan(
            llama_seq_id seq_id, llama_pos p0, llama_pos p1,
            llama_pos & planned_p0, llama_pos & planned_p1) const {
        if (!can_seq_rm(seq_id, p0, p1)) {
            return false;
        }
        planned_p0 = p0;
        planned_p1 = p1;
        return true;
    }

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual bool seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) = 0;

    // Return the number of KV cells at a given position for a seq_id.
    // If cell_indices is not NULL and n_max > 0, fill cell_indices with up to n_max cell indices.
    // Returns the total number of cells at the position (may exceed n_max).
    virtual int cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) = 0;

    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;

    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;
    virtual llama_kv_memory_stats kv_memory_stats() const { return {}; }
    virtual ggml_type get_kv_tail_type() const { return GGML_TYPE_COUNT; }

    virtual uint32_t get_kv_tail_group_count() const { return 0; }
    virtual bool get_kv_tail_coverage(
            uint32_t /* group_index */,
            llama_seq_id /* seq_id */,
            llama_kv_tail_coverage_info & /* out */) const { return false; }
    virtual void reset_kv_tail_planner_timing() {}
    virtual uint64_t get_kv_tail_planner_timing_ns() const { return 0; }

    //
    // state write/read
    //

    // Some compact attention memories cannot be recovered from a later live state
    // by trimming arbitrary suffixes, so composite memories must include them
    // even when LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY is requested.
    virtual bool requires_state_for_partial_restore() const { return false; }

    // Per-sequence state operations must not snapshot or overwrite physical data
    // owned by another live logical sequence. Composite memories must forward both
    // direction-specific queries to every participating child.
    virtual bool state_seq_can_save(llama_seq_id seq_id) const {
        return seq_id >= 0;
    }
    virtual bool state_seq_can_restore(llama_seq_id seq_id) const {
        return seq_id >= 0;
    }
    virtual bool state_seq_can_save(llama_seq_id seq_id, llama_state_seq_flags flags) const {
        GGML_UNUSED(flags);
        return state_seq_can_save(seq_id);
    }
    virtual bool state_seq_can_restore(llama_seq_id seq_id, llama_state_seq_flags flags) const {
        GGML_UNUSED(flags);
        return state_seq_can_restore(seq_id);
    }

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;

    // KV-cache-compatible hooks used by composite memories such as hybrid and iSWA.
    // Non-KV memory types keep the defaults and should not be used as attention memory.
    virtual uint32_t get_kv_n_stream() const { return 0; }
    virtual uint32_t get_kv_size() const { return 0; }
    virtual llama_memory_context_ptr init_kv_batch(const std::vector<llama_ubatch> & /* ubatches */) { return nullptr; }
};

inline bool llama_memory_seq_rm_plan_all(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1,
        std::initializer_list<const llama_memory_i *> children,
        llama_pos & planned_p0, llama_pos & planned_p1) {
    if (children.size() == 0) {
        return false;
    }

    llama_pos common_p0 = p0;
    llama_pos common_p1 = p1;
    for (const llama_memory_i * child : children) {
        llama_pos child_p0 = p0;
        llama_pos child_p1 = p1;
        if (child == nullptr || !child->seq_rm_plan(seq_id, p0, p1, child_p0, child_p1)) {
            return false;
        }
        if (p1 < 0) {
            if (child_p1 >= 0) {
                return false;
            }
            if (child_p0 < common_p0) {
                common_p0 = child_p0;
            }
        } else if (child_p0 != p0 || child_p1 != p1) {
            return false;
        }
    }

    for (const llama_memory_i * child : children) {
        if (!child->can_seq_rm(seq_id, common_p0, common_p1)) {
            return false;
        }
    }

    planned_p0 = common_p0;
    planned_p1 = common_p1;
    return true;
}

inline llama_memory_i::seq_rm_capability llama_memory_seq_rm_capability_all(
        std::initializer_list<const llama_memory_i *> children) {
    llama_memory_i::seq_rm_capability result;
    for (const llama_memory_i * child : children) {
        if (!child) {
            return { false, false, 0 };
        }
        const auto capability = child->get_seq_rm_capability();
        result.full_clear = result.full_clear && capability.full_clear;
        result.arbitrary_ranges = result.arbitrary_ranges && capability.arbitrary_ranges;
        result.suffix_rollback_tokens = std::min(
                result.suffix_rollback_tokens, capability.suffix_rollback_tokens);
    }
    return result;
}

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;
