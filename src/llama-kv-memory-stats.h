#pragma once

#include <cstdint>

struct llama_kv_memory_component_stats {
    uint64_t k_payload_bytes = 0;
    uint64_t v_payload_bytes = 0;
    uint64_t exact_tail_bytes = 0;
    uint64_t native_exact_bytes = 0;
    uint64_t rollback_reserve_bytes = 0;
    uint64_t transient_estimate_bytes = 0;
    uint64_t staging_bytes = 0;
    uint64_t metadata_bytes = 0;
    uint64_t padding_bytes = 0;
    uint64_t allocated_capacity_tokens = 0;
    uint64_t tail_native_bodyless_layers = 0;
    uint64_t tail_native_mixed_layers = 0;
    uint64_t tail_device_fallback_layers = 0;
    uint64_t tail_cpu_layers = 0;

    uint64_t persistent_overhead_bytes() const {
        return staging_bytes + metadata_bytes + padding_bytes;
    }

    uint64_t resident_bytes() const {
        return k_payload_bytes + v_payload_bytes + exact_tail_bytes + native_exact_bytes +
                rollback_reserve_bytes +
                persistent_overhead_bytes();
    }

    void add(const llama_kv_memory_component_stats & other) {
        k_payload_bytes += other.k_payload_bytes;
        v_payload_bytes += other.v_payload_bytes;
        exact_tail_bytes += other.exact_tail_bytes;
        native_exact_bytes += other.native_exact_bytes;
        rollback_reserve_bytes += other.rollback_reserve_bytes;
        transient_estimate_bytes += other.transient_estimate_bytes;
        staging_bytes += other.staging_bytes;
        metadata_bytes += other.metadata_bytes;
        padding_bytes += other.padding_bytes;
        tail_native_bodyless_layers += other.tail_native_bodyless_layers;
        tail_native_mixed_layers += other.tail_native_mixed_layers;
        tail_device_fallback_layers += other.tail_device_fallback_layers;
        tail_cpu_layers += other.tail_cpu_layers;
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
        return exact_history_bytes() + rollback_reserve_bytes();
    }

    uint64_t exact_history_bytes() const {
        return exact_overlay_bytes() + native_exact_bytes();
    }

    uint64_t exact_overlay_bytes() const {
        return global.exact_tail_bytes + swa.exact_tail_bytes;
    }

    uint64_t native_exact_bytes() const {
        return global.native_exact_bytes + swa.native_exact_bytes;
    }

    uint64_t rollback_reserve_bytes() const {
        return global.rollback_reserve_bytes + swa.rollback_reserve_bytes;
    }

    uint64_t transient_estimate_bytes() const {
        return global.transient_estimate_bytes + swa.transient_estimate_bytes;
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

    uint64_t tail_native_bodyless_layers() const {
        return global.tail_native_bodyless_layers + swa.tail_native_bodyless_layers;
    }

    uint64_t tail_native_mixed_layers() const {
        return global.tail_native_mixed_layers + swa.tail_native_mixed_layers;
    }

    uint64_t tail_device_fallback_layers() const {
        return global.tail_device_fallback_layers + swa.tail_device_fallback_layers;
    }

    uint64_t tail_cpu_layers() const {
        return global.tail_cpu_layers + swa.tail_cpu_layers;
    }
};
