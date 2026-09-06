#pragma once

#include <cstdint>
#include <vector>

// Rank each included physical cell in serialized state order. The predicate is
// evaluated exactly once per cell so callers can reuse the result for every
// compact-tail entry instead of rescanning the cache for each entry.
template <typename Included>
std::vector<int32_t> llama_kv_cache_state_cell_ordinals(
        uint32_t n_cells, Included && included) {
    std::vector<int32_t> ordinals(n_cells, -1);
    int32_t ordinal = 0;
    for (uint32_t cell = 0; cell < n_cells; ++cell) {
        if (included(cell)) {
            ordinals[cell] = ordinal++;
        }
    }
    return ordinals;
}
