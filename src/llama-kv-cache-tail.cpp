#include "llama-kv-cache-tail.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_set>

size_t llama_kv_tail_store::identity_hash::operator()(const llama_kv_tail_identity & value) const {
    size_t hash = value.stream;
    hash = hash * 0x9e3779b1u + value.cell;
    hash = hash * 0x9e3779b1u + size_t(value.generation ^ (value.generation >> 32));
    return hash;
}

static uint32_t checked_tail_slot_count(uint32_t arena_stride, uint32_t n_seq_max, uint32_t sink_slots) {
    const uint64_t total = uint64_t(arena_stride)*n_seq_max + sink_slots;
    if (total > uint64_t(INT32_MAX)) {
        throw std::overflow_error("KV tail arena slot index overflows int32_t");
    }
    return uint32_t(total);
}

static uint32_t checked_tail_history_limit(uint32_t n_tokens, uint32_t rollback_tokens) {
    const uint64_t total = uint64_t(n_tokens) + rollback_tokens;
    if (total > uint64_t(UINT32_MAX)) {
        throw std::overflow_error("compact KV tail history capacity overflows uint32_t");
    }
    return uint32_t(total);
}

std::vector<llama_kv_tail_slot_run> llama_kv_tail_contiguous_slot_runs(
        const std::vector<int32_t> & slots) {
    std::vector<llama_kv_tail_slot_run> runs;
    if (slots.empty()) {
        return runs;
    }

    runs.push_back({ 0, slots[0], 1 });
    for (size_t payload = 1; payload < slots.size(); ++payload) {
        auto & run = runs.back();
        if (int64_t(slots[payload]) == int64_t(slots[payload - 1]) + 1) {
            ++run.length;
        } else {
            runs.push_back({ uint32_t(payload), slots[payload], 1 });
        }
    }

#ifndef NDEBUG
    uint64_t n_payloads = 0;
    for (const auto & run : runs) {
        assert(run.length > 0);
        assert(run.payload_begin == n_payloads);
        n_payloads += run.length;
    }
    assert(n_payloads == slots.size());
#endif

    return runs;
}

llama_kv_tail_layout llama_kv_tail_layout_for(
        uint32_t n_tokens,
        uint32_t n_seq_max,
        uint32_t n_ubatch) {
    constexpr uint32_t tail_fa_stride = 256;
    const uint64_t arena_need = uint64_t(n_tokens) + n_ubatch;
    const uint64_t arena_rounded = ((arena_need + tail_fa_stride - 1)/tail_fa_stride)*tail_fa_stride;
    if (arena_rounded > uint64_t(INT32_MAX)) {
        throw std::overflow_error("standard KV tail arena stride overflows int32_t");
    }

    const uint32_t sink_slots = n_seq_max > 1 ? n_ubatch : 0;
    return {
        uint32_t(arena_rounded),
        sink_slots,
        checked_tail_slot_count(uint32_t(arena_rounded), n_seq_max, sink_slots),
    };
}

llama_kv_tail_compact_layout llama_kv_tail_compact_layout_for(
        uint32_t n_tokens,
        uint32_t rollback_tokens,
        uint32_t n_seq_max,
        uint32_t n_ubatch) {
    const uint64_t history_stride = uint64_t(n_tokens) + rollback_tokens;
    const uint64_t attention_stride = uint64_t(n_tokens) + n_ubatch;
    if (history_stride > uint64_t(INT32_MAX)) {
        throw std::overflow_error("compact KV tail history stride overflows int32_t");
    }
    if (attention_stride > uint64_t(INT32_MAX)) {
        throw std::overflow_error("compact KV tail attention stride overflows int32_t");
    }
    const uint32_t history_slots =
            checked_tail_slot_count(uint32_t(history_stride), n_seq_max, 0);
    return {
        uint32_t(history_stride),
        history_slots,
        rollback_tokens,
        uint32_t(attention_stride),
    };
}

bool llama_kv_tail_can_remove_suffix(
        llama_pos pos_max,
        llama_pos p0,
        llama_pos p1,
        uint32_t rollback_tokens) {
    if (p0 <= 0 && p1 < 0) {
        return true;
    }
    if (p0 <= 0 || p1 >= 0) {
        return false;
    }
    if (pos_max < p0) {
        return true;
    }

    const llama_pos rollback = pos_max - (p0 - 1);
    return rollback >= 1 && rollback <= llama_pos(rollback_tokens);
}

static uint64_t checked_tail_bytes(uint64_t rows, uint64_t bytes_per_row) {
    if (bytes_per_row != 0 && rows > std::numeric_limits<uint64_t>::max()/bytes_per_row) {
        throw std::overflow_error("standard KV tail byte count overflows uint64_t");
    }
    return rows*bytes_per_row;
}

llama_kv_tail_group_resolution llama_kv_tail_resolve_groups(
        bool automatic,
        bool explicit_complete,
        const std::vector<llama_kv_tail_group_request> & groups) {
    llama_kv_tail_group_resolution result { automatic || explicit_complete, {} };
    result.tokens.reserve(groups.size());
    if (!result.valid) {
        result.tokens.assign(groups.size(), 0);
        return result;
    }

    constexpr uint32_t auto_tokens = 1024;
    for (const auto & group : groups) {
        const uint32_t requested = automatic ? auto_tokens : group.requested_tokens;
        result.tokens.push_back(group.applicable_standard_kv ?
                std::min(requested, group.effective_window) : 0);
    }
    return result;
}

bool llama_kv_tail_sparse_body_capacity_safe(
        uint32_t physical_stream_rows,
        uint32_t arena_stride) {
    return arena_stride > 0 && physical_stream_rows <= arena_stride;
}

uint32_t llama_kv_tail_packed_body_stride(uint64_t logical_rows, uint32_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("packed KV body alignment must be a non-zero power of two");
    }
    if (logical_rows > UINT64_MAX - (alignment - 1)) {
        throw std::overflow_error("packed KV body stride overflows uint64_t");
    }
    const uint64_t aligned = (logical_rows + alignment - 1)/alignment*alignment;
    if (aligned > uint64_t(UINT32_MAX)) {
        throw std::overflow_error("packed KV body stride overflows uint32_t");
    }
    return uint32_t(aligned);
}

llama_kv_tail_route_capability llama_kv_tail_select_route(
        const llama_kv_tail_route_requirements & requirements) {
    const auto missing_write = [&]() -> llama_kv_tail_operation {
        if (!requirements.write_k) return LLAMA_KV_TAIL_OP_WRITE_K;
        if (!requirements.write_v) return LLAMA_KV_TAIL_OP_WRITE_V;
        return LLAMA_KV_TAIL_OP_NONE;
    }();
    if (missing_write != LLAMA_KV_TAIL_OP_NONE) {
        return { false, LLAMA_KV_TAIL_ROUTE_NONE, missing_write };
    }
    if (requirements.native_attention) {
        return { true, LLAMA_KV_TAIL_ROUTE_NATIVE, LLAMA_KV_TAIL_OP_NONE };
    }
    const std::pair<bool, llama_kv_tail_operation> generic[] = {
        { requirements.gather_k,      LLAMA_KV_TAIL_OP_GATHER_K },
        { requirements.gather_v,      LLAMA_KV_TAIL_OP_GATHER_V },
        { requirements.body_score,    LLAMA_KV_TAIL_OP_BODY_SCORE },
        { requirements.body_value,    LLAMA_KV_TAIL_OP_BODY_VALUE },
        { requirements.exact_score,   LLAMA_KV_TAIL_OP_EXACT_SCORE },
        { requirements.exact_value,   LLAMA_KV_TAIL_OP_EXACT_VALUE },
        { requirements.generic_merge, LLAMA_KV_TAIL_OP_GENERIC_MERGE },
    };
    for (const auto & requirement : generic) {
        if (!requirement.first) {
            return { false, LLAMA_KV_TAIL_ROUTE_NONE, requirement.second };
        }
    }
    return { true, LLAMA_KV_TAIL_ROUTE_GENERIC, LLAMA_KV_TAIL_OP_NONE };
}

const char * llama_kv_tail_operation_name(llama_kv_tail_operation operation) {
    switch (operation) {
        case LLAMA_KV_TAIL_OP_NONE:             return "none";
        case LLAMA_KV_TAIL_OP_WRITE_K:          return "K body/exact write";
        case LLAMA_KV_TAIL_OP_WRITE_V:          return "V body/exact write";
        case LLAMA_KV_TAIL_OP_GATHER_K:         return "exact K gather";
        case LLAMA_KV_TAIL_OP_GATHER_V:         return "exact V gather";
        case LLAMA_KV_TAIL_OP_BODY_SCORE:       return "body K score";
        case LLAMA_KV_TAIL_OP_BODY_VALUE:       return "body V reduction";
        case LLAMA_KV_TAIL_OP_EXACT_SCORE:      return "exact K score";
        case LLAMA_KV_TAIL_OP_EXACT_VALUE:      return "exact V reduction";
        case LLAMA_KV_TAIL_OP_GENERIC_MERGE:    return "generic softmax merge";
        case LLAMA_KV_TAIL_OP_NATIVE_ATTENTION: return "native attached-tail attention";
    }
    return "unknown";
}

llama_kv_tail_type_resolution llama_kv_tail_resolve_type(
        ggml_type requested,
        ggml_type family_default,
        const llama_kv_tail_route_capability & bf16,
        const llama_kv_tail_route_capability & f16) {
    if (requested != GGML_TYPE_COUNT && requested != GGML_TYPE_BF16 && requested != GGML_TYPE_F16) {
        throw std::invalid_argument("KV tail type resolution requires F16, BF16, or COUNT");
    }
    if (family_default != GGML_TYPE_BF16 && family_default != GGML_TYPE_F16) {
        throw std::invalid_argument("KV tail family default must be F16 or BF16");
    }

    const bool automatic = requested == GGML_TYPE_COUNT;
    const ggml_type preferred = automatic ? family_default : requested;
    const auto & preferred_capability = preferred == GGML_TYPE_BF16 ? bf16 : f16;
    if (preferred_capability.supported) {
        return { true, preferred, false, LLAMA_KV_TAIL_OP_NONE, LLAMA_KV_TAIL_OP_NONE };
    }
    if (automatic && preferred == GGML_TYPE_BF16 && f16.supported) {
        return { true, GGML_TYPE_F16, true,
                preferred_capability.missing_operation, LLAMA_KV_TAIL_OP_NONE };
    }
    return { false, GGML_TYPE_COUNT, false,
            preferred_capability.missing_operation,
            automatic && preferred == GGML_TYPE_BF16 ? f16.missing_operation : LLAMA_KV_TAIL_OP_NONE };
}

llama_kv_tail_ownership_error llama_kv_tail_validate_layer_ownership(
        const llama_kv_tail_layer_ownership & ownership) {
    if (ownership.body_k_meta_split) {
        return LLAMA_KV_TAIL_OWNERSHIP_META_SPLIT_K;
    }
    if (ownership.body_v_meta_split) {
        return LLAMA_KV_TAIL_OWNERSHIP_META_SPLIT_V;
    }
    if (ownership.shadow_k_owner != 0 && ownership.shadow_k_owner != ownership.body_k_owner) {
        return LLAMA_KV_TAIL_OWNERSHIP_SHADOW_K;
    }
    if (ownership.shadow_v_owner != 0 && ownership.shadow_v_owner != ownership.body_v_owner) {
        return LLAMA_KV_TAIL_OWNERSHIP_SHADOW_V;
    }
    if (ownership.graph_write_owner != ownership.body_k_owner ||
            (ownership.body_v_owner != 0 && ownership.graph_write_owner != ownership.body_v_owner)) {
        return LLAMA_KV_TAIL_OWNERSHIP_GRAPH_WRITE;
    }
    if (ownership.graph_read_owner != ownership.body_k_owner ||
            (ownership.body_v_owner != 0 && ownership.graph_read_owner != ownership.body_v_owner)) {
        return LLAMA_KV_TAIL_OWNERSHIP_GRAPH_READ;
    }
    if (ownership.state_k_owner != ownership.body_k_owner) {
        return LLAMA_KV_TAIL_OWNERSHIP_STATE_K;
    }
    if (ownership.body_v_owner != 0 && ownership.state_v_owner != ownership.body_v_owner) {
        return LLAMA_KV_TAIL_OWNERSHIP_STATE_V;
    }
    return LLAMA_KV_TAIL_OWNERSHIP_OK;
}

llama_kv_tail_layer_ownership llama_kv_tail_plan_layer_ownership(
        uint32_t layer_id,
        uintptr_t body_k_owner,
        uintptr_t body_v_owner,
        bool shadow_k,
        bool shadow_v) {
    return {
        layer_id,
        body_k_owner,
        body_v_owner,
        shadow_k ? body_k_owner : 0,
        shadow_v ? body_v_owner : 0,
        body_k_owner,
        body_k_owner,
        body_k_owner,
        body_v_owner,
        false,
        false,
    };
}

llama_kv_tail_storage_plan llama_kv_tail_storage_plan_for(
        const llama_kv_tail_storage_request & request) {
    llama_kv_tail_storage_plan result {
        LLAMA_KV_TAIL_STORAGE_DISABLED,
        request.requested_tokens,
        request.effective_tokens,
        request.visibility_window,
        request.requested_body_type_k,
        request.requested_body_type_v,
        request.requested_body_type_k,
        request.requested_body_type_v,
        false,
        false,
        GGML_TYPE_COUNT,
        GGML_TYPE_COUNT,
        request.physical_body_rows,
        { 0, 0, 0 },
        checked_tail_bytes(request.physical_body_rows, request.requested_body_bytes_per_row),
        0,
        0,
        0,
        0,
        request.has_owned_body,
        request.has_shared_body,
        false,
        request.graph_consumes_exact_tail,
        {},
    };
    result.actual_body_bytes = result.requested_body_bytes;
    if (request.effective_tokens == 0 || request.physical_body_rows == 0 ||
            (!request.has_owned_body && !request.has_shared_body)) {
        result.effective_tokens = 0;
        return result;
    }
    if (request.exact_type != GGML_TYPE_F16 && request.exact_type != GGML_TYPE_BF16) {
        throw std::invalid_argument("standard KV tail type must be F16 or BF16");
    }

    result.layout = llama_kv_tail_layout_for(
            request.effective_tokens, request.n_seq_max, request.n_ubatch);
    result.compact_layout = llama_kv_tail_compact_layout_for(
            request.effective_tokens, request.rollback_tokens,
            request.n_seq_max, request.n_ubatch);
    result.promotion_increment = checked_tail_bytes(
            request.physical_body_rows, request.promotion_bytes_per_row);
    result.overlay_increment = checked_tail_bytes(
            result.layout.total_slots, request.overlay_bytes_per_row);
    result.compact_history_bytes = checked_tail_bytes(
            result.compact_layout.history_slots, request.overlay_bytes_per_row);
    result.compact_rollback_bytes = checked_tail_bytes(
            uint64_t(request.rollback_tokens)*request.n_seq_max,
            request.overlay_bytes_per_row);

    const ggml_type promoted_k = ggml_is_quantized(request.requested_body_type_k) ?
            request.exact_type : request.requested_body_type_k;
    const ggml_type promoted_v = ggml_is_quantized(request.requested_body_type_v) ?
            request.exact_type : request.requested_body_type_v;
    const bool full_visibility = request.visibility_window > 0 &&
            request.effective_tokens >= request.visibility_window;
    const bool use_compact = request.compact_history_capable &&
            request.compact_current_source_capable &&
            request.compact_ordered_commit_capable;

    if (use_compact) {
        if (full_visibility && request.full_window_body_can_be_omitted) {
            result.kind = LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT;
            result.actual_body_type_k = promoted_k;
            result.actual_body_type_v = promoted_v;
            result.body_promoted = promoted_k != request.requested_body_type_k ||
                    promoted_v != request.requested_body_type_v;
            result.shadow_k = request.shadow_k_capable;
            result.shadow_v = request.shadow_v_capable;
            result.shadow_type_k = result.shadow_k ? promoted_k : GGML_TYPE_COUNT;
            result.shadow_type_v = result.shadow_v ? promoted_v : GGML_TYPE_COUNT;
            result.shadow_bytes = result.compact_history_bytes;
            result.actual_body_bytes = 0;
            result.has_owned_body = false;
            result.has_shared_body = false;
            return result;
        }

        const std::string architecture = request.architecture ? request.architecture : "unknown";
        const std::string group = request.group_id ? request.group_id : "unknown";
        if (!request.graph_consumes_exact_tail) {
            throw std::invalid_argument(
                    "compact standard KV tail is not consumed by architecture " + architecture +
                    " group " + group + " for requested N=" + std::to_string(request.requested_tokens));
        }
        if (!request.overlay_placement_supported) {
            throw std::invalid_argument(
                    "compact standard KV tail for architecture " + architecture + " group " + group +
                    " is unsupported with --split-mode tensor");
        }
        result.kind = LLAMA_KV_TAIL_STORAGE_COMPACT_OVERLAY;
        result.shadow_k = request.shadow_k_capable;
        result.shadow_v = request.shadow_v_capable;
        result.shadow_type_k = result.shadow_k ? promoted_k : GGML_TYPE_COUNT;
        result.shadow_type_v = result.shadow_v ? promoted_v : GGML_TYPE_COUNT;
        result.shadow_bytes = result.compact_history_bytes;
        return result;
    }

    const bool use_native = request.already_exact ||
            (full_visibility && request.native_capable &&
             result.promotion_increment <= result.overlay_increment);

    if (use_native) {
        result.kind = LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT;
        result.actual_body_type_k = promoted_k;
        result.actual_body_type_v = promoted_v;
        result.body_promoted = promoted_k != request.requested_body_type_k ||
                promoted_v != request.requested_body_type_v;
        if (result.promotion_increment > std::numeric_limits<uint64_t>::max() - result.requested_body_bytes) {
            throw std::overflow_error("standard KV tail promoted body byte count overflows uint64_t");
        }
        result.actual_body_bytes = result.requested_body_bytes + result.promotion_increment;
    } else {
        const std::string architecture = request.architecture ? request.architecture : "unknown";
        const std::string group = request.group_id ? request.group_id : "unknown";
        if (!request.graph_consumes_exact_tail) {
            throw std::invalid_argument(
                    "standard KV tail overlay is not consumed by architecture " + architecture +
                    " group " + group + " for requested N=" + std::to_string(request.requested_tokens) +
                    " (unsupported K-only MLA/DSA composition); use --kv-tail-tokens 0 or a "
                    "full-window native-exact cache");
        }
        if (!request.overlay_placement_supported) {
            throw std::invalid_argument(
                    "standard KV tail overlay for architecture " + architecture + " group " + group +
                    " is unsupported with --split-mode tensor; use --split-mode layer, "
                    "--kv-tail-tokens 0, or a full-window native-exact cache");
        }
        result.kind = LLAMA_KV_TAIL_STORAGE_OVERLAY;
        result.shadow_k = request.shadow_k_capable;
        result.shadow_v = request.shadow_v_capable;
        result.shadow_type_k = result.shadow_k ?
                (ggml_is_quantized(request.requested_body_type_k) ? request.exact_type : request.requested_body_type_k) :
                GGML_TYPE_COUNT;
        result.shadow_type_v = result.shadow_v ?
                (ggml_is_quantized(request.requested_body_type_v) ? request.exact_type : request.requested_body_type_v) :
                GGML_TYPE_COUNT;
        result.shadow_bytes = result.overlay_increment;
    }
    return result;
}

void llama_kv_tail_select_masked_entries(
        const std::vector<llama_kv_tail_mask_entry> & entries,
        const std::vector<uint8_t> & finite,
        llama_pos query_position,
        uint32_t retention,
        bool causal_attn,
        std::vector<uint32_t> & selected) {
    if (entries.size() != finite.size()) {
        throw std::invalid_argument("KV tail candidate and mask sizes differ");
    }

    selected.clear();
    selected.reserve(std::min<size_t>(retention, entries.size()));
    if (retention == 0) {
        return;
    }

    for (size_t i = entries.size(); i-- > 0 && selected.size() < retention;) {
        if (!finite[i] || (causal_attn && entries[i].position > query_position)) {
            continue;
        }
        selected.push_back(uint32_t(i));
    }
    std::reverse(selected.begin(), selected.end());
    assert(selected.size() <= retention);
}

llama_kv_tail_store::llama_kv_tail_store(uint32_t n_tokens, uint32_t n_seq_max, uint32_t n_slots) :
        llama_kv_tail_store(
            n_tokens,
            n_seq_max,
            n_seq_max == 0 ? 0 : n_slots/n_seq_max,
            n_seq_max == 0 ? n_slots : n_slots % n_seq_max) {
    if (n_seq_max == 0 || n_slots % n_seq_max != 0) {
        throw std::invalid_argument("KV tail test arena rows must divide evenly across sequences");
    }
}

llama_kv_tail_store::llama_kv_tail_store(
        uint32_t n_tokens,
        uint32_t n_seq_max,
        uint32_t arena_stride,
        uint32_t sink_slots) :
        n_tokens(n_tokens), rollback_tokens(0), history_limit(n_tokens), arena_stride(arena_stride),
        n_slots(checked_tail_slot_count(arena_stride, n_seq_max, sink_slots)), compact_storage(false),
        sequences(n_seq_max),
        entry_by_cell(n_seq_max),
        slot_used(n_seq_max, std::vector<bool>(arena_stride, false)),
        write_cursors(n_seq_max, 0), degradation(n_seq_max, 0),
        recovery_commits(n_seq_max, 0) {
    if (n_tokens > 0 && arena_stride < n_tokens) {
        throw std::invalid_argument("KV tail arena is smaller than one sequence tail");
    }
}

llama_kv_tail_store::llama_kv_tail_store(
        uint32_t n_tokens,
        uint32_t rollback_tokens,
        uint32_t n_seq_max,
        uint32_t history_stride,
        uint32_t sink_slots) :
        n_tokens(n_tokens), rollback_tokens(rollback_tokens),
        history_limit(checked_tail_history_limit(n_tokens, rollback_tokens)),
        arena_stride(history_stride),
        n_slots(checked_tail_slot_count(history_stride, n_seq_max, sink_slots)),
        compact_storage(true), sequences(n_seq_max), entry_by_cell(n_seq_max),
        slot_used(n_seq_max, std::vector<bool>(history_stride, false)),
        write_cursors(n_seq_max, 0), degradation(n_seq_max, 0),
        recovery_commits(n_seq_max, 0) {
    if (history_stride < history_limit) {
        throw std::invalid_argument("compact KV tail arena is smaller than history plus rollback");
    }
}

void llama_kv_tail_store::clear() {
    cancel_seq_cp();
    batch_transaction.reset();
    for (size_t i = 0; i < sequences.size(); ++i) {
        sequences[i].clear();
        entry_by_cell[i].clear();
    }
    std::fill(degradation.begin(), degradation.end(), 0);
    std::fill(recovery_commits.begin(), recovery_commits.end(), 0);
    in_batch = false;
    std::fill(write_cursors.begin(), write_cursors.end(), 0);
    for (auto & used : slot_used) {
        std::fill(used.begin(), used.end(), false);
    }
}

void llama_kv_tail_store::mark_degraded(llama_seq_id seq_id, uint32_t flags) {
    if (seq_id < 0) {
        for (size_t i = 0; i < degradation.size(); ++i) {
            degradation[i] |= flags;
            recovery_commits[i] = 0;
        }
        if (pending_seq_cp) {
            pending_seq_cp->degradation_flags |= flags;
            pending_seq_cp->recovery_commits = 0;
        }
    } else if (valid_seq(seq_id)) {
        if (pending_seq_cp && pending_seq_cp->dst == seq_id) {
            pending_seq_cp->degradation_flags |= flags;
            pending_seq_cp->recovery_commits = 0;
        } else {
            degradation[size_t(seq_id)] |= flags;
            recovery_commits[size_t(seq_id)] = 0;
        }
    }
}

void llama_kv_tail_store::invalidate_slots(const std::vector<int32_t> & slots, uint32_t flags) {
    const std::unordered_set<int32_t> invalid(slots.begin(), slots.end());
    for (llama_seq_id seq_id = 0; size_t(seq_id) < sequences.size(); ++seq_id) {
        auto & entries = sequences[size_t(seq_id)];
        bool removed = false;
        for (auto it = entries.begin(); it != entries.end();) {
            if (invalid.find(it->slot) != invalid.end()) {
                release(seq_id, it->slot);
                it = entries.erase(it);
                removed = true;
            } else {
                ++it;
            }
        }
        if (removed) {
            rebuild_index(seq_id);
            degradation[size_t(seq_id)] |= flags;
            recovery_commits[size_t(seq_id)] = 0;
        }
    }
}

void llama_kv_tail_store::begin_batch() {
    if (compact_storage) {
        if (batch_transaction) {
            throw std::logic_error("a compact KV tail batch transaction is already pending");
        }
        batch_transaction_state transaction;
        transaction.sequences = sequences;
        transaction.slot_used = slot_used;
        transaction.write_cursors = write_cursors;
        transaction.degradation = degradation;
        transaction.recovery_commits = recovery_commits;
        transaction.current.resize(sequences.size());
        transaction.affected.assign(sequences.size(), false);
        batch_transaction = std::move(transaction);
    }
    for (llama_seq_id seq_id = 0; size_t(seq_id) < sequences.size(); ++seq_id) {
        trim(seq_id, compact_storage ? history_limit : n_tokens);
    }
    in_batch = true;
}

void llama_kv_tail_store::finish_batch(bool success, bool payload_may_be_modified) {
    if (!compact_storage) {
        in_batch = false;
        return;
    }
    if (!batch_transaction) {
        return;
    }
    if (!success) {
        auto transaction = std::move(*batch_transaction);
        sequences = std::move(transaction.sequences);
        slot_used = std::move(transaction.slot_used);
        write_cursors = std::move(transaction.write_cursors);
        degradation = std::move(transaction.degradation);
        recovery_commits = std::move(transaction.recovery_commits);
        for (llama_seq_id seq_id = 0; size_t(seq_id) < sequences.size(); ++seq_id) {
            rebuild_index(seq_id);
            if (!payload_may_be_modified || !transaction.affected[size_t(seq_id)]) {
                continue;
            }
            auto & entries = sequences[size_t(seq_id)];
            for (const auto & entry : entries) {
                release(seq_id, entry.slot);
            }
            entries.clear();
            entry_by_cell[size_t(seq_id)].clear();
            degradation[size_t(seq_id)] |= LLAMA_KV_TAIL_DEGRADED_PAYLOAD_INVALID;
            recovery_commits[size_t(seq_id)] = 0;
        }
    }
    batch_transaction.reset();
    in_batch = false;
}

bool llama_kv_tail_store::valid_seq(llama_seq_id seq_id) const {
    return seq_id >= 0 && size_t(seq_id) < sequences.size();
}

uint64_t llama_kv_tail_store::cell_key(uint32_t stream, uint32_t cell) {
    return (uint64_t(stream) << 32) | cell;
}

void llama_kv_tail_store::rebuild_index(llama_seq_id seq_id) {
    assert(valid_seq(seq_id));
    auto & index = entry_by_cell[size_t(seq_id)];
    auto & entries = sequences[size_t(seq_id)];
    index.clear();
    index.reserve(entries.size());
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        const auto & identity = it->identity;
        const bool inserted = index.emplace(cell_key(identity.stream, identity.cell), it).second;
        if (!inserted) {
            throw std::logic_error("duplicate physical cell in KV tail sequence");
        }
    }
}

void llama_kv_tail_store::erase_entry(llama_seq_id seq_id, entry_list::iterator entry, bool release_slot) {
    assert(valid_seq(seq_id));
    auto & entries = sequences[size_t(seq_id)];
    auto & index = entry_by_cell[size_t(seq_id)];
    if (release_slot) {
        release(seq_id, entry->slot);
    }
    index.erase(cell_key(entry->identity.stream, entry->identity.cell));
    entries.erase(entry);
}

int32_t llama_kv_tail_store::acquire(llama_seq_id seq_id) {
    assert(valid_seq(seq_id));
    auto & used = slot_used[size_t(seq_id)];
    auto & cursor = write_cursors[size_t(seq_id)];
    for (uint32_t offset = 0; offset < arena_stride; ++offset) {
        const uint32_t local = (cursor + offset) % arena_stride;
        if (!used[local]) {
            used[local] = true;
            cursor = (local + 1) % arena_stride;
            return int32_t(uint32_t(seq_id)*arena_stride + local);
        }
    }
    throw std::runtime_error(
            "KV tail sequence arena exhausted (sequence=" + std::to_string(seq_id) +
            ", stride=" + std::to_string(arena_stride) + ")");
}

void llama_kv_tail_store::release(llama_seq_id seq_id, int32_t slot) {
    if (!valid_seq(seq_id) || slot < 0) {
        return;
    }
    const uint32_t base = uint32_t(seq_id)*arena_stride;
    assert(uint32_t(slot) >= base && uint32_t(slot) < base + arena_stride);
    slot_used[size_t(seq_id)][uint32_t(slot) - base] = false;
}

void llama_kv_tail_store::trim(llama_seq_id seq_id, uint32_t limit) {
    auto & entries = sequences[size_t(seq_id)];
    auto & index = entry_by_cell[size_t(seq_id)];
    while (entries.size() > limit) {
        index.erase(cell_key(entries.front().identity.stream, entries.front().identity.cell));
        release(seq_id, entries.front().slot);
        entries.pop_front();
    }
}

int32_t llama_kv_tail_store::commit(
        llama_seq_id seq_id,
        llama_kv_tail_identity identity,
        llama_pos position,
        uint64_t insertion_ordinal,
        uint32_t current_row) {
    if (!valid_seq(seq_id) || n_tokens == 0) {
        return LLAMA_KV_TAIL_BODY_SLOT;
    }

    auto & entries = sequences[size_t(seq_id)];
    auto & index = entry_by_cell[size_t(seq_id)];
    const auto duplicate = index.find(cell_key(identity.stream, identity.cell));
    if (duplicate != index.end() && duplicate->second->identity == identity) {
        duplicate->second->position = position;
        duplicate->second->insertion_ordinal = insertion_ordinal;
        entries.sort([](const exact_entry & a, const exact_entry & b) {
            return a.position < b.position ||
                    (a.position == b.position && a.insertion_ordinal < b.insertion_ordinal);
        });
        trim(seq_id, compact_storage ? history_limit : n_tokens);
        const auto retained = index.find(cell_key(identity.stream, identity.cell));
        const int32_t result = retained == index.end() || !(retained->second->identity == identity) ?
                LLAMA_KV_TAIL_BODY_SLOT : retained->second->slot;
        if (batch_transaction && current_row != UINT32_MAX && result >= 0) {
            const uint64_t virtual_slot = uint64_t(n_slots) + current_row;
            if (virtual_slot > uint64_t(INT32_MAX)) {
                throw std::overflow_error("compact KV tail current source index overflows int32_t");
            }
            batch_transaction->current[size_t(seq_id)].push_back(
                    { identity, position, insertion_ordinal, int32_t(virtual_slot) });
            batch_transaction->affected[size_t(seq_id)] = true;
        }
        return result;
    }
    if (duplicate != index.end()) {
        // A generation change normally arrives through recycle().  Keeping
        // commit fail-safe here prevents two shadows from naming one body row.
        erase_entry(seq_id, duplicate->second);
    }

    int32_t slot = LLAMA_KV_TAIL_BODY_SLOT;
    const uint32_t commit_limit = compact_storage ? history_limit : n_tokens;
    if ((compact_storage || !in_batch) && entries.size() >= commit_limit && !entries.empty()) {
        // Entries are kept in (position, insertion_ordinal) order by every
        // insertion and mutation path, so the first entry is the eviction
        // victim. Avoid rescanning the entire retained tail on every token.
        const auto victim = entries.begin();
        slot = victim->slot;
        erase_entry(seq_id, victim, false);
    } else {
        slot = acquire(seq_id);
    }
    exact_entry appended { identity, position, insertion_ordinal, slot };
    auto inserted = entries.end();
    if (entries.empty() || entries.back().position < position ||
            (entries.back().position == position && entries.back().insertion_ordinal <= insertion_ordinal)) {
        entries.push_back(appended);
        inserted = std::prev(entries.end());
    } else {
        inserted = std::find_if(entries.begin(), entries.end(), [&](const exact_entry & entry) {
            return position < entry.position ||
                    (position == entry.position && insertion_ordinal < entry.insertion_ordinal);
        });
        inserted = entries.insert(inserted, appended);
    }
    index[cell_key(identity.stream, identity.cell)] = inserted;
    if (degradation[size_t(seq_id)] != 0 && ++recovery_commits[size_t(seq_id)] >= n_tokens) {
        degradation[size_t(seq_id)] = 0;
        recovery_commits[size_t(seq_id)] = 0;
    }
    if (!in_batch) {
        trim(seq_id, commit_limit);
    }
    const auto retained = index.find(cell_key(identity.stream, identity.cell));
    const int32_t result = retained == index.end() || !(retained->second->identity == identity) ?
            LLAMA_KV_TAIL_BODY_SLOT : retained->second->slot;
    if (batch_transaction && current_row != UINT32_MAX && result >= 0) {
        const uint64_t virtual_slot = uint64_t(n_slots) + current_row;
        if (virtual_slot > uint64_t(INT32_MAX)) {
            throw std::overflow_error("compact KV tail current source index overflows int32_t");
        }
        batch_transaction->current[size_t(seq_id)].push_back(
                { identity, position, insertion_ordinal, int32_t(virtual_slot) });
        batch_transaction->affected[size_t(seq_id)] = true;
    }
    return result;
}

void llama_kv_tail_store::recycle(uint32_t stream, uint32_t cell, uint64_t next_generation) {
    for (llama_seq_id seq_id = 0; size_t(seq_id) < sequences.size(); ++seq_id) {
        auto & index = entry_by_cell[size_t(seq_id)];
        const auto found = index.find(cell_key(stream, cell));
        if (found != index.end() &&
                found->second->identity.generation != next_generation) {
            if (batch_transaction) {
                batch_transaction->affected[size_t(seq_id)] = true;
            }
            erase_entry(seq_id, found->second);
        }
    }
}

void llama_kv_tail_store::seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
    if (!valid_seq(src) || !valid_seq(dst) || src == dst) {
        return;
    }
    const uint32_t src_stream = sequences[size_t(src)].empty() ? 0 : sequences[size_t(src)].front().identity.stream;
    (void) seq_cp_remap(src, dst, src_stream, src_stream, p0, p1);
}

std::vector<llama_kv_tail_slot_copy> llama_kv_tail_store::seq_cp_remap(
        llama_seq_id src,
        llama_seq_id dst,
        uint32_t src_stream,
        uint32_t dst_stream,
        llama_pos p0,
        llama_pos p1) {
    auto result = prepare_seq_cp(src, dst, src_stream, dst_stream, p0, p1);
    commit_seq_cp();
    return result;
}

std::vector<llama_kv_tail_slot_copy> llama_kv_tail_store::prepare_seq_cp(
        llama_seq_id src,
        llama_seq_id dst,
        uint32_t src_stream,
        uint32_t dst_stream,
        llama_pos p0,
        llama_pos p1) {
    std::vector<llama_kv_tail_slot_copy> result;
    if (!valid_seq(src) || !valid_seq(dst) || src == dst) {
        return result;
    }
    if (pending_seq_cp) {
        throw std::logic_error("a KV tail sequence-copy transaction is already pending");
    }

    std::vector<exact_entry> source;
    for (const auto & entry : sequences[size_t(src)]) {
        if (entry.identity.stream != src_stream || entry.position < p0 || (p1 >= 0 && entry.position >= p1)) {
            continue;
        }
        source.push_back(entry);
    }

    struct retained_candidate {
        exact_entry entry;
        int32_t src_slot;
        bool copied;
    };
    std::vector<retained_candidate> candidates;
    candidates.reserve(sequences[size_t(dst)].size() + source.size());
    for (const auto & entry : sequences[size_t(dst)]) {
        if (entry.position < p0 || (p1 >= 0 && entry.position >= p1)) {
            candidates.push_back({ entry, entry.slot, false });
        }
    }
    for (const auto & entry : source) {
        exact_entry copied = entry;
        copied.identity.stream = dst_stream;
        copied.slot = LLAMA_KV_TAIL_BODY_SLOT;
        candidates.push_back({ copied, entry.slot, true });
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const retained_candidate & a, const retained_candidate & b) {
        return a.entry.position < b.entry.position ||
                (a.entry.position == b.entry.position && a.entry.insertion_ordinal < b.entry.insertion_ordinal);
    });
    const uint32_t retained = compact_storage ? history_limit : n_tokens;
    const size_t first = candidates.size() > retained ? candidates.size() - retained : 0;
    std::unordered_set<int32_t> survivor_slots;
    for (size_t i = first; i < candidates.size(); ++i) {
        if (!candidates[i].copied) {
            survivor_slots.insert(candidates[i].entry.slot);
        }
    }
    std::vector<int32_t> reusable_slots;
    for (const auto & old : sequences[size_t(dst)]) {
        if (survivor_slots.find(old.slot) == survivor_slots.end()) {
            reusable_slots.push_back(old.slot);
        }
    }

    pending_seq_cp_state pending;
    pending.src = src;
    pending.dst = dst;
    pending.destination.reserve(candidates.size() - first);
    try {
        for (size_t i = first; i < candidates.size(); ++i) {
            auto retained = candidates[i];
            if (retained.copied) {
                if (!reusable_slots.empty()) {
                    retained.entry.slot = reusable_slots.back();
                    reusable_slots.pop_back();
                } else {
                    retained.entry.slot = acquire(dst);
                    pending.acquired_slots.push_back(retained.entry.slot);
                }
                result.push_back({ retained.src_slot, retained.entry.slot });
            }
            pending.destination.push_back(retained.entry);
        }
    } catch (...) {
        for (int32_t slot : pending.acquired_slots) {
            release(dst, slot);
        }
        throw;
    }
    pending.copies = result;
    pending.degradation_flags = degradation[size_t(dst)] | degradation[size_t(src)];
    pending.recovery_commits = pending.degradation_flags == 0 ?
            std::min(recovery_commits[size_t(dst)], recovery_commits[size_t(src)]) : 0;
    pending_seq_cp = std::move(pending);
    return result;
}

void llama_kv_tail_store::commit_seq_cp() {
    if (!pending_seq_cp) {
        return;
    }
    auto pending = std::move(*pending_seq_cp);
    pending_seq_cp.reset();
    std::unordered_set<int32_t> retained_slots;
    retained_slots.reserve(pending.destination.size());
    for (const auto & entry : pending.destination) {
        retained_slots.insert(entry.slot);
    }
    for (const auto & old : sequences[size_t(pending.dst)]) {
        if (retained_slots.find(old.slot) == retained_slots.end()) {
            release(pending.dst, old.slot);
        }
    }
    auto & destination = sequences[size_t(pending.dst)];
    destination.clear();
    for (const auto & entry : pending.destination) {
        destination.push_back(entry);
    }
    rebuild_index(pending.dst);
    degradation[size_t(pending.dst)] = pending.degradation_flags;
    recovery_commits[size_t(pending.dst)] = pending.recovery_commits;
}

void llama_kv_tail_store::cancel_seq_cp() {
    if (!pending_seq_cp) {
        return;
    }
    const llama_seq_id dst = pending_seq_cp->dst;
    for (int32_t slot : pending_seq_cp->acquired_slots) {
        release(dst, slot);
    }
    pending_seq_cp.reset();
}

llama_seq_id llama_kv_tail_store::pending_seq_cp_destination() const {
    return pending_seq_cp ? pending_seq_cp->dst : -1;
}

void llama_kv_tail_store::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!valid_seq(seq_id)) {
        return;
    }
    auto & entries = sequences[size_t(seq_id)];
    for (auto it = entries.begin(); it != entries.end();) {
        if (it->position >= p0 && (p1 < 0 || it->position < p1)) {
            release(seq_id, it->slot);
            it = entries.erase(it);
        } else {
            ++it;
        }
    }
    rebuild_index(seq_id);
    if (entries.empty() && p0 <= 0 && p1 < 0) {
        degradation[size_t(seq_id)] = 0;
        recovery_commits[size_t(seq_id)] = 0;
    }
}

void llama_kv_tail_store::seq_rm_cell(llama_seq_id seq_id, uint32_t stream, uint32_t cell) {
    if (!valid_seq(seq_id)) {
        return;
    }
    auto & entries = sequences[size_t(seq_id)];
    for (auto it = entries.begin(); it != entries.end();) {
        if (it->identity.stream == stream && it->identity.cell == cell) {
            release(seq_id, it->slot);
            it = entries.erase(it);
        } else {
            ++it;
        }
    }
    rebuild_index(seq_id);
}

void llama_kv_tail_store::seq_keep(llama_seq_id seq_id) {
    for (llama_seq_id current = 0; size_t(current) < sequences.size(); ++current) {
        if (current != seq_id) {
            seq_rm(current, 0, -1);
        }
    }
}

void llama_kv_tail_store::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (!valid_seq(seq_id)) {
        return;
    }
    std::unordered_set<llama_kv_tail_identity, identity_hash> identities;
    for (const auto & entry : sequences[size_t(seq_id)]) {
        if (entry.position >= p0 && (p1 < 0 || entry.position < p1)) {
            identities.insert(entry.identity);
        }
    }
    const bool degrade = shift != 0 && !(p0 <= 0 && p1 < 0);
    for (llama_seq_id current = 0; size_t(current) < sequences.size(); ++current) {
        auto & entries = sequences[size_t(current)];
        bool touched = false;
        for (auto it = entries.begin(); it != entries.end();) {
            if (identities.find(it->identity) == identities.end()) {
                ++it;
                continue;
            }
            touched = true;
            it->position += shift;
            if (it->position < 0) {
                release(current, it->slot);
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
        if (!touched) {
            continue;
        }
        entries.sort([](const exact_entry & a, const exact_entry & b) {
            return a.position < b.position ||
                    (a.position == b.position && a.insertion_ordinal < b.insertion_ordinal);
        });
        trim(current, compact_storage ? history_limit : n_tokens);
        rebuild_index(current);
        if (degrade) {
            degradation[size_t(current)] |= LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP;
            recovery_commits[size_t(current)] = 0;
        }
    }
}

void llama_kv_tail_store::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int divisor) {
    if (!valid_seq(seq_id) || divisor == 0) {
        return;
    }
    std::unordered_set<llama_kv_tail_identity, identity_hash> identities;
    for (const auto & entry : sequences[size_t(seq_id)]) {
        if (entry.position >= p0 && (p1 < 0 || entry.position < p1)) {
            identities.insert(entry.identity);
        }
    }
    const bool degrade = divisor != 1 && !(p0 <= 0 && p1 < 0);
    for (llama_seq_id current = 0; size_t(current) < sequences.size(); ++current) {
        auto & entries = sequences[size_t(current)];
        bool touched = false;
        for (auto & entry : entries) {
            if (identities.find(entry.identity) != identities.end()) {
                entry.position /= divisor;
                touched = true;
            }
        }
        if (!touched) {
            continue;
        }
        entries.sort([](const exact_entry & a, const exact_entry & b) {
            return a.position < b.position ||
                    (a.position == b.position && a.insertion_ordinal < b.insertion_ordinal);
        });
        trim(current, compact_storage ? history_limit : n_tokens);
        rebuild_index(current);
        if (degrade) {
            degradation[size_t(current)] |= LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP;
            recovery_commits[size_t(current)] = 0;
        }
    }
}

std::vector<int32_t> llama_kv_tail_store::build_source_plan(
        llama_seq_id seq_id,
        const std::vector<llama_kv_tail_identity> & visible) const {
    std::vector<int32_t> result(visible.size(), LLAMA_KV_TAIL_BODY_SLOT);
    if (!valid_seq(seq_id) || n_tokens == 0) {
        return result;
    }

    struct candidate {
        size_t visible_index;
        const exact_entry * entry;
    };
    const auto & sequence = sequences[size_t(seq_id)];
    std::unordered_map<llama_kv_tail_identity, const exact_entry *, identity_hash> entries_by_identity;
    entries_by_identity.reserve(sequence.size());
    for (const auto & entry : sequence) {
        entries_by_identity.emplace(entry.identity, &entry);
    }
    std::vector<candidate> candidates;
    candidates.reserve(std::min<size_t>(visible.size(), sequence.size()));
    for (size_t i = 0; i < visible.size(); ++i) {
        const auto entry = entries_by_identity.find(visible[i]);
        if (entry != entries_by_identity.end()) {
            candidates.push_back({ i, entry->second });
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) {
        return a.entry->position < b.entry->position ||
                (a.entry->position == b.entry->position &&
                 a.entry->insertion_ordinal < b.entry->insertion_ordinal);
    });
    const size_t first = candidates.size() > n_tokens ? candidates.size() - n_tokens : 0;
    for (size_t i = first; i < candidates.size(); ++i) {
        result[candidates[i].visible_index] = candidates[i].entry->slot;
    }
    return result;
}

llama_kv_tail_coverage llama_kv_tail_store::coverage(llama_seq_id seq_id, uint32_t available) const {
    const bool pending_destination = pending_seq_cp && pending_seq_cp->dst == seq_id;
    const uint32_t exact = !valid_seq(seq_id) ? 0 : pending_destination ?
            uint32_t(pending_seq_cp->destination.size()) : uint32_t(sequences[size_t(seq_id)].size());
    const uint32_t requested = std::min(n_tokens, available);
    const uint32_t flags = !valid_seq(seq_id) ? 0 : pending_destination ?
            pending_seq_cp->degradation_flags : degradation[size_t(seq_id)];
    const auto state = exact == 0 ? LLAMA_KV_TAIL_COVERAGE_NONE :
            exact >= requested && flags == 0 ? LLAMA_KV_TAIL_COVERAGE_COMPLETE : LLAMA_KV_TAIL_COVERAGE_PARTIAL;
    return { state, requested, std::min(exact, requested), flags };
}

bool llama_kv_tail_store::supports_suffix_rollback(llama_seq_id seq_id, uint32_t n) const {
    return compact_storage && valid_seq(seq_id) && n <= rollback_tokens;
}

std::vector<int32_t> llama_kv_tail_store::body_indices(uint32_t kv_size) const {
    std::vector<int32_t> result(n_slots, 0);
    for (const auto & entries : sequences) {
        for (const auto & entry : entries) {
            const uint64_t index = uint64_t(entry.identity.stream)*kv_size + entry.identity.cell;
            if (index > uint64_t(std::numeric_limits<int32_t>::max())) {
                throw std::overflow_error("KV tail body index overflows int32_t");
            }
            result[size_t(entry.slot)] = int32_t(index);
        }
    }
    return result;
}

std::vector<std::pair<int32_t, llama_kv_tail_identity>> llama_kv_tail_store::active_slots() const {
    std::vector<std::pair<int32_t, llama_kv_tail_identity>> result;
    for (const auto & entries : sequences) {
        result.reserve(result.size() + entries.size());
        for (const auto & entry : entries) {
            result.emplace_back(entry.slot, entry.identity);
        }
    }
    return result;
}

std::vector<llama_kv_tail_snapshot_entry> llama_kv_tail_store::source_candidates(llama_seq_id seq_id) const {
    std::vector<llama_kv_tail_snapshot_entry> result;
    if (!valid_seq(seq_id)) {
        return result;
    }
    const auto & committed = batch_transaction ?
            batch_transaction->sequences[size_t(seq_id)] : sequences[size_t(seq_id)];
    const size_t first = batch_transaction && committed.size() > n_tokens ?
            committed.size() - n_tokens : 0;
    const size_t n_current = batch_transaction ?
            batch_transaction->current[size_t(seq_id)].size() : 0;
    result.reserve(committed.size() - first + n_current);
    auto it = committed.begin();
    std::advance(it, first);
    for (; it != committed.end(); ++it) {
        const auto & entry = *it;
        result.push_back({ seq_id, entry.identity, entry.position,
                entry.insertion_ordinal, entry.slot });
    }
    if (batch_transaction) {
        for (const auto & entry : batch_transaction->current[size_t(seq_id)]) {
            result.push_back({ seq_id, entry.identity, entry.position,
                    entry.insertion_ordinal, entry.slot });
        }
        std::stable_sort(result.begin(), result.end(), [](const auto & a, const auto & b) {
            return a.position < b.position ||
                    (a.position == b.position && a.insertion_ordinal < b.insertion_ordinal);
        });
    }
    return result;
}

std::vector<llama_kv_tail_source_run> llama_kv_tail_store::source_runs(llama_seq_id seq_id) const {
    std::vector<llama_kv_tail_source_run> result;
    if (!valid_seq(seq_id)) {
        return result;
    }
    uint32_t exact_offset = 0;
    for (const auto & entry : sequences[size_t(seq_id)]) {
        if (!result.empty()) {
            auto & last = result.back();
            if (last.exact_offset + last.length == exact_offset &&
                    last.stream == entry.identity.stream &&
                    last.cell + last.length == entry.identity.cell) {
                ++last.length;
                ++exact_offset;
                continue;
            }
        }
        result.push_back({ exact_offset, entry.identity.stream, entry.identity.cell, 1 });
        ++exact_offset;
    }
    return result;
}

std::vector<llama_kv_tail_snapshot_entry> llama_kv_tail_store::snapshot(llama_seq_id seq_id) const {
    std::vector<llama_kv_tail_snapshot_entry> result;
    for (llama_seq_id current = 0; size_t(current) < sequences.size(); ++current) {
        if (seq_id >= 0 && current != seq_id) {
            continue;
        }
        if (pending_seq_cp && pending_seq_cp->dst == current) {
            const auto & ordered = pending_seq_cp->destination;
            const uint32_t retained = compact_storage ? history_limit : n_tokens;
            const size_t first = ordered.size() > retained ? ordered.size() - retained : 0;
            for (size_t i = first; i < ordered.size(); ++i) {
                const auto & entry = ordered[i];
                result.push_back({ current, entry.identity, entry.position,
                        entry.insertion_ordinal, entry.slot });
            }
            continue;
        }
        const auto & ordered = sequences[size_t(current)];
        const uint32_t retained = compact_storage ? history_limit : n_tokens;
        const size_t first = ordered.size() > retained ? ordered.size() - retained : 0;
        auto it = ordered.begin();
        std::advance(it, first);
        for (; it != ordered.end(); ++it) {
            const auto & entry = *it;
            result.push_back({ current, entry.identity, entry.position,
                    entry.insertion_ordinal, entry.slot });
        }
    }
    return result;
}

std::vector<llama_kv_tail_provenance> llama_kv_tail_store::snapshot_provenance(llama_seq_id seq_id) const {
    std::vector<llama_kv_tail_provenance> result;
    if (seq_id >= 0) {
        if (valid_seq(seq_id)) {
            const bool pending_destination = pending_seq_cp && pending_seq_cp->dst == seq_id;
            result.push_back({ seq_id,
                    pending_destination ? pending_seq_cp->degradation_flags : degradation[size_t(seq_id)],
                    pending_destination ? pending_seq_cp->recovery_commits : recovery_commits[size_t(seq_id)] });
        }
        return result;
    }
    result.reserve(sequences.size());
    for (llama_seq_id current = 0; size_t(current) < sequences.size(); ++current) {
        const bool pending_destination = pending_seq_cp && pending_seq_cp->dst == current;
        result.push_back({ current,
                pending_destination ? pending_seq_cp->degradation_flags : degradation[size_t(current)],
                pending_destination ? pending_seq_cp->recovery_commits : recovery_commits[size_t(current)] });
    }
    return result;
}

void llama_kv_tail_store::restore_provenance(
        const std::vector<llama_kv_tail_provenance> & provenance,
        llama_seq_id dest_seq_id) {
    constexpr uint32_t known_flags =
            LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE |
            LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP |
            LLAMA_KV_TAIL_DEGRADED_STATE_RESTORE;
    std::vector<bool> seen(sequences.size(), false);
    for (const auto & saved : provenance) {
        if (!valid_seq(saved.seq_id) || (saved.degradation_flags & ~known_flags) != 0 ||
                saved.recovery_commits > n_tokens ||
                (saved.degradation_flags == 0 && saved.recovery_commits != 0)) {
            throw std::runtime_error("invalid KV tail degradation provenance");
        }
        if (seen[size_t(saved.seq_id)]) {
            throw std::runtime_error("duplicate KV tail degradation provenance");
        }
        seen[size_t(saved.seq_id)] = true;
    }

    if (dest_seq_id >= 0) {
        if (!valid_seq(dest_seq_id) || provenance.size() != 1) {
            throw std::runtime_error("invalid per-sequence KV tail degradation provenance");
        }
        degradation[size_t(dest_seq_id)] = provenance[0].degradation_flags;
        recovery_commits[size_t(dest_seq_id)] = provenance[0].recovery_commits;
        return;
    }

    std::fill(degradation.begin(), degradation.end(), 0);
    std::fill(recovery_commits.begin(), recovery_commits.end(), 0);
    for (const auto & saved : provenance) {
        degradation[size_t(saved.seq_id)] = saved.degradation_flags;
        recovery_commits[size_t(saved.seq_id)] = saved.recovery_commits;
    }
}

std::vector<float> llama_kv_tail_attention_reference(
        const std::vector<float> & query,
        const std::vector<float> & body_k,
        const std::vector<float> & body_v,
        const std::vector<float> & tail_k,
        const std::vector<float> & tail_v,
        const std::vector<int32_t> & source_slots,
        uint32_t key_dim,
        uint32_t value_dim,
        float scale) {
    if (query.size() != key_dim || body_k.size() != source_slots.size()*key_dim ||
            body_v.size() != source_slots.size()*value_dim || tail_k.size() % key_dim != 0 ||
            tail_v.size() % value_dim != 0 || tail_k.size()/key_dim != tail_v.size()/value_dim) {
        throw std::invalid_argument("invalid tiered attention reference shape");
    }

    std::vector<float> logits(source_slots.size());
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t token = 0; token < source_slots.size(); ++token) {
        const int32_t slot = source_slots[token];
        if (slot >= 0 && size_t(slot) >= tail_k.size()/key_dim) {
            throw std::invalid_argument("tiered attention source slot is out of range");
        }
        const float * key = slot >= 0 ? tail_k.data() + size_t(slot)*key_dim : body_k.data() + token*key_dim;
        float logit = 0.0f;
        for (uint32_t i = 0; i < key_dim; ++i) {
            logit += query[i]*key[i];
        }
        logits[token] = logit*scale;
        max_logit = std::max(max_logit, logits[token]);
    }

    std::vector<float> result(value_dim, 0.0f);
    float sum = 0.0f;
    for (size_t token = 0; token < source_slots.size(); ++token) {
        const float weight = std::exp(logits[token] - max_logit);
        sum += weight;
        const int32_t slot = source_slots[token];
        const float * value = slot >= 0 ? tail_v.data() + size_t(slot)*value_dim : body_v.data() + token*value_dim;
        for (uint32_t i = 0; i < value_dim; ++i) {
            result[i] += weight*value[i];
        }
    }
    for (float & value : result) {
        value /= sum;
    }
    return result;
}
