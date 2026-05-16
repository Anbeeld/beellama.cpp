#pragma once

#include "llama.h"
#include "common.h"

#include <unordered_map>
#include <vector>

struct common_speculative;

struct common_dflash_ring_write {
    int ring_pos;
    int n_tokens;
    int src_token_offset;
};

// Describes which portion of a prefill sub-batch should be captured into
// the DFlash cross ring.  When should_flush is true, src_offset and n_tokens
// specify the contiguous span within the capture buffer to write.
// When should_flush is false, no capture is needed for this sub-batch.
struct common_dflash_prefill_span {
    bool should_flush = false;

    int32_t capture_begin = 0;
    int32_t capture_end   = 0;

    int  src_offset   = 0;
    int  n_tokens     = 0;
};

common_dflash_ring_write common_dflash_ring_write_plan(int ring_size, int ring_pos, int n_tokens);

// DDTree: tree of likely continuations built from draft logits
struct common_speculative_tree {
    std::vector<llama_token> tokens;   // [n_nodes] tree node tokens (topological order)
    std::vector<int32_t>     parents;  // [n_nodes+1] parent index (-1 for root, 0-based for nodes)
    std::vector<int32_t>     depths;   // [n_nodes] depth (1-based: root's children = 1)
    std::vector<std::unordered_map<llama_token, int>> child_maps; // [n_nodes+1] token → child node index (1-based)
    std::vector<uint8_t>     visibility; // [(n_nodes+1)²] row-major: visibility[i*(n+1)+j] = node i can attend to node j
    std::vector<float>       log_probs;  // [n_nodes] draft log-probability per node (for rejection sampling)
    int n_nodes = 0;
    int main_path_len = 0; // number of main-path nodes (indices 1..main_path_len in batch)
};

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

// Create a drafter context that can be shared across multiple common_speculative
// instances (DFlash multi-slot). The caller owns the returned context and must
// release it with llama_free after all dependent common_speculative instances
// have been freed. Returns nullptr if the speculative params have no draft model.
// topk / sample_temp / other per-ctx_dft config is applied here so the shared
// context is fully configured before it is wired into any common_speculative.
// dflash_n_slots: initial DFlash drafter graph width (1 for non-DFlash or single-slot).
// Sets DFlash graph sizing params before llama_init_from_model so the initial reserve
// allocates compute buffers sized for the target slot count and cross-attention window.
llama_context * common_speculative_create_ctx_dft(const common_params_speculative & params, int dflash_n_slots = 1);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        llama_context             * ctx_dft_shared = nullptr);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// inform the speculative stack which server slot (seq_id on ctx_tgt and the
// shared ctx_dft) this instance services. Safe to call multiple times;
// no-op for non-DFlash impls.
void common_speculative_set_seq_id(common_speculative * spec, llama_seq_id seq_id);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last,
                     std::vector<float> * draft_log_probs = nullptr);

// Batched DFlash draft: all specs prepare cross data, one combined multi-seq
// decode, results distributed back. Each spec must have had set_seq_id called
// and share the same ctx_dft. Specs without a ready DFlash impl get empty
// results. Does not run extension impls (CopySpec etc.) — caller should
// extend per-spec results afterwards if desired.
void common_speculative_draft_batch(
        std::vector<common_speculative *> & specs,
        llama_context                     * ctx_dft,
        const common_params_speculative   & params,
        const std::vector<llama_token>    & id_last_per_spec,
        std::vector<llama_tokens>         & result_per_spec,
        std::vector<std::vector<float>>   * log_probs_per_spec = nullptr);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

// update implementations with logits from the verification decode
void common_speculative_update_logits(common_speculative * spec, llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted);

// tree variant: update ring buffer with specific capture-buffer indices
// (for tree verify where accepted tokens may be non-contiguous in the capture buffer)
void common_speculative_update_logits_by_indices(common_speculative * spec, llama_context * ctx, const std::vector<int> & capture_indices);

// flush hidden states captured during the current prefill sub-batch into
// the DFlash ring buffer. Call after each llama_decode during split prefill
// so checkpoint splits don't lose hidden state context between sub-batches.
// src_offset and n_tokens specify which contiguous span of the captured
// hidden buffer to write into the ring (from the prefill span calculation).
// Pass src_offset=0, n_tokens=0 for the default (write entire buffer).
// Returns the number of tokens actually written to the ring.
int common_speculative_flush_prefill(common_speculative * spec, int src_offset = 0, int n_tokens = 0);

// Enable/disable target hidden capture for DFlash prefill.
// No-op for non-DFlash speculative implementations.
// Used by the server to skip useless hidden capture before the final DFlash
// cross-context suffix during long prompt processing.
void common_speculative_set_prefill_capture_enabled(common_speculative * spec, bool enabled);

// save/restore ring buffer state for checkpoint persistence.
// the ring contains target hidden states needed by the DFlash drafter's
// cross-attention. Without this, checkpoint-restored prefills lose context.
size_t common_speculative_ring_state_size(const common_speculative * spec);
void   common_speculative_ring_state_save(const common_speculative * spec, uint8_t * buf, size_t size);
bool   common_speculative_ring_state_load(common_speculative * spec, const uint8_t * buf, size_t size);

// DDTree: build a tree of likely continuations from draft logits
// tree_budget: internal total tree nodes for one draft call.
// Public configuration uses branch_budget, then runtime adds the main draft path.
common_speculative_tree common_speculative_draft_tree(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last,
                                    int   tree_budget);

int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params);
int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;
