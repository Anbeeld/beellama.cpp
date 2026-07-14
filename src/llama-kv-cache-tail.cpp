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
        n_tokens(n_tokens), arena_stride(arena_stride), sink_slots(sink_slots),
        n_slots(checked_tail_slot_count(arena_stride, n_seq_max, sink_slots)), sequences(n_seq_max),
        entry_by_cell(n_seq_max),
        slot_used(n_seq_max, std::vector<bool>(arena_stride, false)),
        write_cursors(n_seq_max, 0), degradation(n_seq_max, 0),
        recovery_commits(n_seq_max, 0) {
    if (n_tokens > 0 && arena_stride < n_tokens) {
        throw std::invalid_argument("KV tail arena is smaller than one sequence tail");
    }
}

void llama_kv_tail_store::clear() {
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
    } else if (valid_seq(seq_id)) {
        degradation[size_t(seq_id)] |= flags;
        recovery_commits[size_t(seq_id)] = 0;
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
    for (llama_seq_id seq_id = 0; size_t(seq_id) < sequences.size(); ++seq_id) {
        trim(seq_id);
    }
    in_batch = true;
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

void llama_kv_tail_store::trim(llama_seq_id seq_id) {
    auto & entries = sequences[size_t(seq_id)];
    auto & index = entry_by_cell[size_t(seq_id)];
    while (entries.size() > n_tokens) {
        index.erase(cell_key(entries.front().identity.stream, entries.front().identity.cell));
        release(seq_id, entries.front().slot);
        entries.pop_front();
    }
}

int32_t llama_kv_tail_store::commit(
        llama_seq_id seq_id,
        llama_kv_tail_identity identity,
        llama_pos position,
        uint64_t insertion_ordinal) {
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
        trim(seq_id);
        const auto retained = index.find(cell_key(identity.stream, identity.cell));
        return retained == index.end() || !(retained->second->identity == identity) ?
                LLAMA_KV_TAIL_BODY_SLOT : retained->second->slot;
    }
    if (duplicate != index.end()) {
        // A generation change normally arrives through recycle().  Keeping
        // commit fail-safe here prevents two shadows from naming one body row.
        erase_entry(seq_id, duplicate->second);
    }

    int32_t slot = LLAMA_KV_TAIL_BODY_SLOT;
    if (!in_batch && entries.size() >= n_tokens && !entries.empty()) {
        const auto victim = std::min_element(entries.begin(), entries.end(), [](const exact_entry & a, const exact_entry & b) {
            return a.position < b.position ||
                (a.position == b.position && a.insertion_ordinal < b.insertion_ordinal);
        });
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
        trim(seq_id);
    }
    const auto retained = index.find(cell_key(identity.stream, identity.cell));
    return retained == index.end() || !(retained->second->identity == identity) ?
            LLAMA_KV_TAIL_BODY_SLOT : retained->second->slot;
}

void llama_kv_tail_store::recycle(uint32_t stream, uint32_t cell, uint64_t next_generation) {
    for (llama_seq_id seq_id = 0; size_t(seq_id) < sequences.size(); ++seq_id) {
        auto & index = entry_by_cell[size_t(seq_id)];
        const auto found = index.find(cell_key(stream, cell));
        if (found != index.end() &&
                found->second->identity.generation != next_generation) {
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
    std::vector<llama_kv_tail_slot_copy> result;
    if (!valid_seq(src) || !valid_seq(dst) || src == dst) {
        return result;
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
    const size_t first = candidates.size() > n_tokens ? candidates.size() - n_tokens : 0;
    std::unordered_set<int32_t> survivor_slots;
    for (size_t i = first; i < candidates.size(); ++i) {
        if (!candidates[i].copied) {
            survivor_slots.insert(candidates[i].entry.slot);
        }
    }
    for (const auto & old : sequences[size_t(dst)]) {
        if (survivor_slots.find(old.slot) == survivor_slots.end()) {
            release(dst, old.slot);
        }
    }
    auto & destination = sequences[size_t(dst)];
    destination.clear();
    for (size_t i = first; i < candidates.size(); ++i) {
        auto retained = candidates[i];
        if (retained.copied) {
            retained.entry.slot = acquire(dst);
            result.push_back({ retained.src_slot, retained.entry.slot });
        }
        destination.push_back(retained.entry);
    }
    rebuild_index(dst);
    degradation[size_t(dst)] |= degradation[size_t(src)];
    recovery_commits[size_t(dst)] = degradation[size_t(dst)] == 0 ?
            std::min(recovery_commits[size_t(dst)], recovery_commits[size_t(src)]) : 0;
    return result;
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
    auto & entries = sequences[size_t(seq_id)];
    for (auto it = entries.begin(); it != entries.end();) {
        if (it->position >= p0 && (p1 < 0 || it->position < p1)) {
            it->position += shift;
            if (it->position < 0) {
                release(seq_id, it->slot);
                it = entries.erase(it);
                continue;
            }
        }
        ++it;
    }
    entries.sort([](const exact_entry & a, const exact_entry & b) {
        return a.position < b.position ||
                (a.position == b.position && a.insertion_ordinal < b.insertion_ordinal);
    });
    trim(seq_id);
}

void llama_kv_tail_store::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int divisor) {
    if (!valid_seq(seq_id) || divisor == 0) {
        return;
    }
    for (auto & entry : sequences[size_t(seq_id)]) {
        if (entry.position >= p0 && (p1 < 0 || entry.position < p1)) {
            entry.position /= divisor;
        }
    }
    sequences[size_t(seq_id)].sort([](const exact_entry & a, const exact_entry & b) {
        return a.position < b.position ||
                (a.position == b.position && a.insertion_ordinal < b.insertion_ordinal);
    });
    trim(seq_id);
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
    const uint32_t exact = valid_seq(seq_id) ? uint32_t(sequences[size_t(seq_id)].size()) : 0;
    const uint32_t requested = std::min(n_tokens, available);
    const uint32_t flags = valid_seq(seq_id) ? degradation[size_t(seq_id)] : 0;
    const auto state = exact == 0 ? LLAMA_KV_TAIL_COVERAGE_NONE :
            exact >= requested && flags == 0 ? LLAMA_KV_TAIL_COVERAGE_COMPLETE : LLAMA_KV_TAIL_COVERAGE_PARTIAL;
    return { state, requested, std::min(exact, requested), flags };
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
    result.reserve(sequences[size_t(seq_id)].size());
    for (const auto & entry : sequences[size_t(seq_id)]) {
        result.push_back({ seq_id, entry.identity, entry.position,
                entry.insertion_ordinal, entry.slot });
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
        const auto & ordered = sequences[size_t(current)];
        const size_t first = ordered.size() > n_tokens ? ordered.size() - n_tokens : 0;
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
