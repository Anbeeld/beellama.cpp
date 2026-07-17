#pragma once

#include "llama-memory.h"

#include <utility>

// Executes the fallible portion of an exact-tail copy as one transaction. The
// callbacks are templates so production calls inline directly into the backend
// operations while tests can inject each failure boundary without a runtime
// hook in llama_context or llama_kv_cache.
template <typename Allocate, typename Transfer, typename Compute, typename Rollback>
llama_memory_status llama_kv_tail_copy_transaction(
        Allocate && allocate,
        Transfer && transfer,
        Compute && compute,
        Rollback && rollback) {
    if (!std::forward<Allocate>(allocate)()) {
        std::forward<Rollback>(rollback)();
        return LLAMA_MEMORY_STATUS_FAILED_PREPARE;
    }
    if (!std::forward<Transfer>(transfer)()) {
        std::forward<Rollback>(rollback)();
        return LLAMA_MEMORY_STATUS_FAILED_PREPARE;
    }
    if (!std::forward<Compute>(compute)()) {
        std::forward<Rollback>(rollback)();
        return LLAMA_MEMORY_STATUS_FAILED_COMPUTE;
    }
    return LLAMA_MEMORY_STATUS_SUCCESS;
}
