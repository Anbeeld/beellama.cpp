#include "llama-kv-cache-update.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static void tail_copy_allocation_failure() {
    int transferred = 0;
    int computed = 0;
    int rolled_back = 0;
    const auto status = llama_kv_tail_copy_transaction(
            [] { return false; },
            [&] { ++transferred; return true; },
            [&] { ++computed; return true; },
            [&] { ++rolled_back; });
    require(status == LLAMA_MEMORY_STATUS_FAILED_PREPARE, "allocation failure status");
    require(transferred == 0 && computed == 0 && rolled_back == 1, "allocation failure ordering");
}

static void tail_copy_transfer_failure() {
    int computed = 0;
    int rolled_back = 0;
    const auto status = llama_kv_tail_copy_transaction(
            [] { return true; },
            [] { return false; },
            [&] { ++computed; return true; },
            [&] { ++rolled_back; });
    require(status == LLAMA_MEMORY_STATUS_FAILED_PREPARE, "transfer failure status");
    require(computed == 0 && rolled_back == 1, "transfer failure ordering");
}

static void tail_copy_compute_failure() {
    int rolled_back = 0;
    const auto status = llama_kv_tail_copy_transaction(
            [] { return true; },
            [] { return true; },
            [] { return false; },
            [&] { ++rolled_back; });
    require(status == LLAMA_MEMORY_STATUS_FAILED_COMPUTE, "compute failure status");
    require(rolled_back == 1, "compute failure rollback");
}

static void tail_copy_success() {
    int rolled_back = 0;
    const auto status = llama_kv_tail_copy_transaction(
            [] { return true; },
            [] { return true; },
            [] { return true; },
            [&] { ++rolled_back; });
    require(status == LLAMA_MEMORY_STATUS_SUCCESS, "success status");
    require(rolled_back == 0, "success must not roll back");
}

int main() {
    tail_copy_allocation_failure();
    tail_copy_transfer_failure();
    tail_copy_compute_failure();
    tail_copy_success();
    std::puts("PASS: deferred exact-tail copy transaction fault matrix");
    return 0;
}
