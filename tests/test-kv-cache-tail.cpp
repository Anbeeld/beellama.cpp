#include "llama-kv-cache-tail.h"
#include "llama-kv-cells.h"

#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

static llama_kv_tail_identity id(uint32_t cell, uint64_t generation = 1) {
    return { 0, cell, generation };
}

int main() {
    // Incremental per-sequence live-cell accounting covers membership,
    // removal, keep, shift eviction, and reset without a cache scan.
    llama_kv_cells cells;
    cells.resize(4);
    CHECK(cells.seq_size(0) == 0);
    cells.pos_set(0, 10);
    cells.seq_add(0, 0);
    cells.pos_set(1, 11);
    cells.seq_add(1, 0);
    cells.seq_add(1, 1);
    CHECK(cells.seq_size(0) == 2);
    CHECK(cells.seq_size(1) == 1);
    CHECK(!cells.seq_rm(1, 0));
    CHECK(cells.seq_size(0) == 1);
    CHECK(cells.seq_size(1) == 1);
    CHECK(!cells.seq_keep(0, 0));
    CHECK(cells.seq_size(0) == 1);
    CHECK(cells.pos_add(0, -20));
    CHECK(cells.seq_size(0) == 0);
    cells.rm(1);
    CHECK(cells.seq_size(1) == 0);
    cells.reset();
    CHECK(cells.seq_size(0) == 0 && cells.seq_size(1) == 0);

    // Arena ownership is per sequence: identical physical-cache identities in
    // two sequence memberships still receive distinct exact payload rows.
    llama_kv_tail_store arenas(2, 2, 8);
    const int32_t arena0_slot = arenas.commit(0, id(40), 0, 0);
    const int32_t arena1_slot = arenas.commit(1, id(40), 0, 1);
    CHECK(arena0_slot >= 0 && arena0_slot < 4);
    CHECK(arena1_slot >= 4 && arena1_slot < 8);
    CHECK(arena0_slot != arena1_slot);

    // Overflow replaces the true recency victim in place. It must not consume
    // another arena row and defer the release until after assignment.
    llama_kv_tail_store victim(2, 1, 4);
    const int32_t victim0 = victim.commit(0, id(41), 10, 0);
    victim.commit(0, id(42), 20, 1);
    const int32_t replacement = victim.commit(0, id(43), 30, 2);
    CHECK(replacement == victim0);

    // Per-sequence arenas require payload copies even when both logical
    // sequences share the same unified-cache stream.
    llama_kv_tail_store same_stream_copy(2, 2, 8);
    same_stream_copy.commit(0, id(44), 10, 0);
    same_stream_copy.commit(0, id(45), 11, 1);
    const auto same_stream_remap = same_stream_copy.seq_cp_remap(0, 1, 0, 0, 0, -1);
    CHECK(same_stream_remap.size() == 2);
    CHECK(same_stream_remap[0].src_slot != same_stream_remap[0].dst_slot);

    // A partial copy into a full destination preselects the newest N records
    // before allocating slots. This remains valid when R < 2N and scales with
    // retained metadata rather than per-layer payload bytes.
    constexpr uint32_t copy_n = 2048;
    llama_kv_tail_store full_copy(copy_n, 2, 4608); // R=2304 < 2N per sequence.
    for (uint32_t i = 0; i < copy_n; ++i) {
        full_copy.commit(0, { 0, i, 1 }, 1024 + int32_t(i), i);
        full_copy.commit(1, { 0, copy_n + i, 1 }, int32_t(i), copy_n + i);
    }
    const auto copy_start = std::chrono::steady_clock::now();
    const auto full_remap = full_copy.seq_cp_remap(0, 1, 0, 0, 1024, 3072);
    const auto copy_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - copy_start).count();
    CHECK(full_copy.coverage(1, copy_n).exact == copy_n);
    CHECK(full_remap.size() == copy_n);
    CHECK(copy_ms < 50.0);

    llama_kv_tail_store tail(3, 2, 12);

    tail.commit(0, id(0), 0, 0);
    tail.commit(0, id(1), 1, 1);
    tail.commit(0, id(2), 2, 2);
    tail.commit(0, id(3), 3, 3);

    const auto plan = tail.build_source_plan(0, { id(0), id(1), id(2), id(3) });
    CHECK(plan.size() == 4);
    CHECK(plan[0] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(plan[1] >= 0 && plan[2] >= 0 && plan[3] >= 0);
    CHECK(tail.coverage(0).state == LLAMA_KV_TAIL_COVERAGE_COMPLETE);
    CHECK(tail.coverage(0).requested == 3);
    CHECK(tail.coverage(0).exact == 3);

    // A branch shares direct exact shadows, but cannot manufacture an evicted one.
    tail.seq_cp(0, 1, 0, 4);
    const auto copied = tail.build_source_plan(1, { id(0), id(1), id(2), id(3) });
    CHECK(copied[0] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(copied[1] >= 0 && copied[2] >= 0 && copied[3] >= 0);

    // Recycling a physical cell invalidates the old generation for every sequence.
    tail.recycle(0, 3, 2);
    const auto stale = tail.build_source_plan(1, { id(1), id(2), id(3) });
    CHECK(stale[2] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(tail.coverage(1).state == LLAMA_KV_TAIL_COVERAGE_PARTIAL);

    tail.commit(0, id(3, 2), 4, 4);
    const auto refreshed = tail.build_source_plan(0, { id(1), id(2), id(3, 2) });
    CHECK(refreshed[0] >= 0 && refreshed[1] >= 0 && refreshed[2] >= 0);

    // Query-local selection never expands merely because more entries are in flight.
    tail.commit(0, id(4), 5, 5);
    tail.commit(0, id(5), 6, 6);
    const auto early = tail.build_source_plan(0, { id(0), id(1), id(2) });
    CHECK(early[0] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(early[1] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(early[2] == LLAMA_KV_TAIL_BODY_SLOT);

    tail.seq_rm(0, 4, 7);
    const auto after_rm = tail.build_source_plan(0, { id(1), id(2) });
    CHECK(after_rm[0] == LLAMA_KV_TAIL_BODY_SLOT && after_rm[1] == LLAMA_KV_TAIL_BODY_SLOT);

    // Cross-stream copies receive new physical identities and payload slots.
    // The returned slot pairs tell the cache which exact rows to copy.
    llama_kv_tail_store cross(2, 2, 8);
    cross.commit(0, { 0, 4, 7 }, 10, 10);
    cross.commit(0, { 0, 5, 8 }, 11, 11);
    const auto remap = cross.seq_cp_remap(0, 1, 0, 1, 0, -1);
    CHECK(remap.size() == 2);
    const auto cross_plan = cross.build_source_plan(1, { { 1, 4, 7 }, { 1, 5, 8 } });
    CHECK(cross_plan[0] >= 0 && cross_plan[1] >= 0);
    CHECK(remap[0].src_slot != remap[0].dst_slot);

    // A failed payload transaction must invalidate every possibly partial
    // destination instead of publishing stale exact rows.
    std::vector<int32_t> failed_dst;
    for (const auto & copy : remap) {
        failed_dst.push_back(copy.dst_slot);
    }
    cross.invalidate_slots(failed_dst, LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
    const auto failed_plan = cross.build_source_plan(1, { { 1, 4, 7 }, { 1, 5, 8 } });
    CHECK(failed_plan[0] == LLAMA_KV_TAIL_BODY_SLOT && failed_plan[1] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(cross.coverage(1, 2).degradation_flags & LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);

    // Rebuild the copied state for the remaining cell-local removal checks.
    (void) cross.seq_cp_remap(0, 1, 0, 1, 0, -1);

    // Cell-local removal must preserve another sequence's shared exact row.
    cross.seq_rm_cell(1, 1, 4);
    CHECK(cross.build_source_plan(1, { { 1, 4, 7 }, { 1, 5, 8 } })[0] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(cross.build_source_plan(0, { { 0, 4, 7 }, { 0, 5, 8 } })[0] >= 0);

    cross.mark_degraded(0, LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
    CHECK(cross.coverage(0).state == LLAMA_KV_TAIL_COVERAGE_PARTIAL);
    CHECK(cross.coverage(0).degradation_flags & LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
    cross.commit(0, { 0, 6, 9 }, 12, 12);
    cross.commit(0, { 0, 7, 10 }, 13, 13);
    CHECK(cross.coverage(0).state == LLAMA_KV_TAIL_COVERAGE_COMPLETE);

    // A partial sequence copy must combine degradation history. It must not
    // clear an existing destination reason merely because the source is clean.
    llama_kv_tail_store degraded_copy(2, 2, 8);
    degraded_copy.commit(0, id(0), 0, 0);
    degraded_copy.commit(0, id(1), 1, 1);
    degraded_copy.commit(1, id(2), 2, 2);
    degraded_copy.mark_degraded(1, LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE);
    degraded_copy.seq_cp(0, 1, 0, 1);
    CHECK(degraded_copy.coverage(1).degradation_flags & LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE);
    degraded_copy.mark_degraded(0, LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);
    degraded_copy.seq_cp(0, 1, 0, 1);
    CHECK(degraded_copy.coverage(1).degradation_flags & LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE);
    CHECK(degraded_copy.coverage(1).degradation_flags & LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP);

    // The rollback reserve keeps every query-local row in the current physical
    // ubatch until the next ubatch boundary.
    llama_kv_tail_store rollback(2, 1, 5);
    rollback.commit(0, id(0), 0, 0);
    rollback.commit(0, id(1), 1, 1);
    rollback.begin_batch();
    rollback.commit(0, id(2), 2, 2);
    rollback.commit(0, id(3), 3, 3);
    rollback.commit(0, id(4), 4, 4);
    const auto rollback_candidates = rollback.source_candidates(0);
    CHECK(rollback_candidates.size() == 5);
    for (size_t i = 0; i < rollback_candidates.size(); ++i) {
        CHECK(rollback_candidates[i].identity == id(uint32_t(i)));
    }
    const auto rollback_runs = rollback.source_runs(0);
    CHECK(rollback_runs.size() == 1);
    CHECK(rollback_runs[0].exact_offset == 0 && rollback_runs[0].stream == 0 &&
            rollback_runs[0].cell == 0 && rollback_runs[0].length == 5);
    const auto early_in_flight = rollback.build_source_plan(0, { id(0), id(1), id(2) });
    CHECK(early_in_flight[0] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(early_in_flight[1] >= 0 && early_in_flight[2] >= 0);
    const auto late_in_flight = rollback.build_source_plan(0, { id(0), id(1), id(2), id(3), id(4) });
    CHECK(late_in_flight[0] == LLAMA_KV_TAIL_BODY_SLOT && late_in_flight[1] == LLAMA_KV_TAIL_BODY_SLOT &&
            late_in_flight[2] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(late_in_flight[3] >= 0 && late_in_flight[4] >= 0);
    rollback.begin_batch();
    CHECK(rollback.active_slots().size() == 2);

    // Recency follows position and insertion ordinal, not physical-cell order.
    llama_kv_tail_store ordered(2, 1, 6);
    ordered.commit(0, id(10), 30, 0);
    ordered.commit(0, id(11), 10, 1);
    ordered.commit(0, id(12), 30, 2);
    auto ordered_plan = ordered.build_source_plan(0, { id(10), id(11), id(12) });
    CHECK(ordered_plan[0] >= 0);
    CHECK(ordered_plan[1] == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(ordered_plan[2] >= 0);
    ordered.seq_add(0, 30, 31, -25);
    ordered_plan = ordered.build_source_plan(0, { id(10), id(11), id(12) });
    CHECK(ordered_plan[0] >= 0 && ordered_plan[2] >= 0);
    CHECK(ordered_plan[1] == LLAMA_KV_TAIL_BODY_SLOT);
    ordered.seq_div(0, 0, -1, 5);
    CHECK(ordered.coverage(0).state == LLAMA_KV_TAIL_COVERAGE_COMPLETE);

    // Exhaustion fails before publishing an identity or sequence membership.
    llama_kv_tail_store bounded(2, 1, 2);
    bounded.begin_batch();
    bounded.commit(0, id(20), 0, 0);
    bounded.commit(0, id(21), 1, 1);
    bool exhausted = false;
    try {
        bounded.commit(0, id(22), 2, 2);
    } catch (const std::runtime_error &) {
        exhausted = true;
    }
    CHECK(exhausted);
    CHECK(bounded.active_slots().size() == 2);
    CHECK(bounded.build_source_plan(0, { id(22) })[0] == LLAMA_KV_TAIL_BODY_SLOT);

    // Updating an existing identity can make it older than the retained
    // window. Returning the body sentinel must not dereference its released
    // shadow after trimming.
    llama_kv_tail_store duplicate(2, 1, 4);
    duplicate.begin_batch();
    duplicate.commit(0, id(30), 10, 0);
    duplicate.commit(0, id(31), 20, 1);
    duplicate.commit(0, id(32), 30, 2);
    CHECK(duplicate.commit(0, id(30), 0, 3) == LLAMA_KV_TAIL_BODY_SLOT);
    CHECK(duplicate.build_source_plan(0, { id(30), id(31), id(32) })[0] == LLAMA_KV_TAIL_BODY_SLOT);

    // The reference attention performs one global normalization while choosing
    // body or exact data independently for every visible entry.
    const std::vector<float> q = { 1.0f, 0.5f };
    const std::vector<float> body_k = { 1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f };
    const std::vector<float> body_v = { 1.0f, 2.0f, 10.0f, 20.0f, 100.0f, 200.0f };
    const std::vector<float> tail_k = { 0.25f, 1.5f };
    const std::vector<float> tail_v = { 30.0f, 60.0f };
    const auto ref = llama_kv_tail_attention_reference(
            q, body_k, body_v, tail_k, tail_v, { -1, 0, -1 }, 2, 2, 1.0f);

    const float e0 = std::exp(1.0f);
    const float e1 = std::exp(1.0f);
    const float e2 = std::exp(0.75f);
    const float z = e0 + e1 + e2;
    CHECK(std::fabs(ref[0] - (e0*1.0f + e1*30.0f + e2*100.0f)/z) < 1e-5f);
    CHECK(std::fabs(ref[1] - (e0*2.0f + e1*60.0f + e2*200.0f)/z) < 1e-5f);

    return 0;
}
