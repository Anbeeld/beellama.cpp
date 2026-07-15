#pragma once

#include <cstdint>

struct llama_kv_memory_component_stats {
    uint64_t k_payload_bytes = 0;
    uint64_t v_payload_bytes = 0;
    uint64_t exact_tail_bytes = 0;
    uint64_t native_exact_bytes = 0;
    uint64_t staging_bytes = 0;
    uint64_t metadata_bytes = 0;
    uint64_t padding_bytes = 0;
    uint64_t allocated_capacity_tokens = 0;

    uint64_t persistent_overhead_bytes() const {
        return staging_bytes + metadata_bytes + padding_bytes;
    }

    uint64_t resident_bytes() const {
        return k_payload_bytes + v_payload_bytes + exact_tail_bytes + native_exact_bytes +
                persistent_overhead_bytes();
    }

    void add(const llama_kv_memory_component_stats & other) {
        k_payload_bytes += other.k_payload_bytes;
        v_payload_bytes += other.v_payload_bytes;
        exact_tail_bytes += other.exact_tail_bytes;
        native_exact_bytes += other.native_exact_bytes;
        staging_bytes += other.staging_bytes;
        metadata_bytes += other.metadata_bytes;
        padding_bytes += other.padding_bytes;
        allocated_capacity_tokens = allocated_capacity_tokens > other.allocated_capacity_tokens ?
                allocated_capacity_tokens : other.allocated_capacity_tokens;
    }
};

struct llama_kv_memory_stats {
    llama_kv_memory_component_stats global;
    llama_kv_memory_component_stats swa;

    void add(const llama_kv_memory_stats & other) {
        global.add(other.global);
        swa.add(other.swa);
    }

    uint64_t k_payload_bytes() const {
        return global.k_payload_bytes + swa.k_payload_bytes;
    }

    uint64_t v_payload_bytes() const {
        return global.v_payload_bytes + swa.v_payload_bytes;
    }

    uint64_t exact_tail_bytes() const {
        return exact_overlay_bytes() + native_exact_bytes();
    }

    uint64_t exact_overlay_bytes() const {
        return global.exact_tail_bytes + swa.exact_tail_bytes;
    }

    uint64_t native_exact_bytes() const {
        return global.native_exact_bytes + swa.native_exact_bytes;
    }

    uint64_t persistent_overhead_bytes() const {
        return global.persistent_overhead_bytes() + swa.persistent_overhead_bytes();
    }

    uint64_t resident_bytes() const {
        return global.resident_bytes() + swa.resident_bytes();
    }

    uint64_t allocated_capacity_tokens() const {
        return global.allocated_capacity_tokens > swa.allocated_capacity_tokens ?
                global.allocated_capacity_tokens : swa.allocated_capacity_tokens;
    }
};
