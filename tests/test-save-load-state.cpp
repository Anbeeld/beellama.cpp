#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"
#include "../src/llama-memory.h"

#include <clocale>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>

struct llama_batch_ptr {
    llama_batch batch;

    llama_batch_ptr(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
        : batch{llama_batch_init(n_tokens, embd, n_seq_max)} {}

    ~llama_batch_ptr() { llama_batch_free(batch); }

    llama_batch_ptr(const llama_batch_ptr &) = delete;
    llama_batch_ptr & operator=(const llama_batch_ptr &) = delete;
    llama_batch_ptr(llama_batch_ptr &&) = default;
    llama_batch_ptr & operator=(llama_batch_ptr &&) = default;

    llama_batch & get() { return batch; }
    const llama_batch & get() const { return batch; }
};

static llama_tokens generate_tokens(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id) {
    llama_tokens result;
    llama_batch_ptr batch(1, 0, 1);

    for (int i = 0; i < n_predict; i++) {
        auto next_token = llama_sampler_sample(smpl, ctx, -1);

        LOG("%d ", next_token);
        result.push_back(next_token);

        common_batch_clear(batch.get());
        common_batch_add(batch.get(), next_token, n_past, {seq_id}, true);

        if (llama_decode(ctx, batch.get())) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return {};
        }
        n_past++;
    }

    return result;
}

static bool test_tail_state_contract(
        llama_model * model, const common_params & params, const llama_tokens & tokens) {
    if (params.kv_tail_tokens.empty() ||
            !std::all_of(params.kv_tail_tokens.begin(), params.kv_tail_tokens.end(), ::isdigit) ||
            std::stoul(params.kv_tail_tokens) == 0) {
        return true;
    }

    auto source_params = common_context_params_to_llama(params);
    auto source = llama_context_ptr{llama_init_from_model(model, source_params)};
    if (!source) {
        LOG_ERR("%s: failed to create source context\n", __func__);
        return false;
    }
    int n_past = 0;
    if (!common_prompt_batch_decode(source.get(), tokens, int(tokens.size()), n_past,
            params.n_batch, {}, false)) {
        return false;
    }

    const size_t exact_size = llama_state_get_size(source.get());
    std::vector<uint8_t> exact(exact_size);
    if (llama_state_get_data(source.get(), exact.data(), exact.size()) != exact.size()) {
        LOG_ERR("%s: exact full-state size/write mismatch\n", __func__);
        return false;
    }

    llama_kv_tail_coverage_info coverage_before{};
    if (!llama_kv_tail_get_coverage(source.get(), 0, 0, &coverage_before)) {
        LOG_ERR("%s: failed to query initial tail coverage\n", __func__);
        return false;
    }

    const auto body_flag = llama_state_seq_flags(LLAMA_STATE_SEQ_FLAGS_BODY_ONLY);
    const size_t body_size = llama_state_get_size_ext(source.get(), body_flag);
    const bool has_overlay_state = exact_size > body_size;

    if (params.kvarn.type == LLAMA_KVARN_TYPE_DISABLED && has_overlay_state &&
            !llama_get_memory(source.get())->requires_state_for_partial_restore()) {
        LOG_ERR("%s: standard precision tail does not participate in partial checkpoints\n", __func__);
        return false;
    }

    if (params.kvarn.type == LLAMA_KVARN_TYPE_DISABLED && has_overlay_state) {
        const auto partial_flag = llama_state_seq_flags(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        const auto exact_flag = llama_state_seq_flags(LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED);
        const size_t partial_size = llama_state_seq_get_size_ext(source.get(), 0, partial_flag);
        const size_t self_contained_size = llama_state_seq_get_size_ext(source.get(), 0, exact_flag);
        if (partial_size == 0 || partial_size >= self_contained_size) {
            LOG_ERR("%s: standard precision-tail partial checkpoint copied the body (%zu >= %zu)\n",
                    __func__, partial_size, self_contained_size);
            return false;
        }
    }

    if (has_overlay_state) {
        auto mismatch_params = source_params;
        mismatch_params.kv_tail_tokens++;
        auto mismatch = llama_context_ptr{llama_init_from_model(model, mismatch_params)};
        if (!mismatch) {
            LOG_ERR("%s: failed to create mismatch context\n", __func__);
            return false;
        }
        if (llama_state_set_data(mismatch.get(), exact.data(), exact.size()) != 0) {
            LOG_ERR("%s: mismatched overlay tail configuration was accepted\n", __func__);
            return false;
        }
    }

    std::vector<uint8_t> body(body_size);
    if (llama_state_get_data_ext(source.get(), body.data(), body.size(), body_flag) != body.size()) {
        LOG_ERR("%s: body-only full-state size/write mismatch\n", __func__);
        return false;
    }
    llama_memory_clear(llama_get_memory(source.get()), true);
    if (llama_state_set_data(source.get(), body.data(), body.size()) != body.size()) {
        LOG_ERR("%s: body-only full state could not be restored\n", __func__);
        return false;
    }

    llama_kv_tail_coverage_info coverage_after{};
    if (!llama_kv_tail_get_coverage(source.get(), 0, 0, &coverage_after)) {
        LOG_ERR("%s: failed to query restored tail coverage\n", __func__);
        return false;
    }
    llama_kv_tail_coverage_aggregate aggregate{};
    if (!llama_kv_tail_get_coverage_aggregate(source.get(), 0, &aggregate) || aggregate.groups == 0) {
        LOG_ERR("%s: failed to query aggregate restored coverage\n", __func__);
        return false;
    }
    if (has_overlay_state) {
        if (coverage_after.exact != 0 ||
                (coverage_after.degradation_flags & LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE) == 0 ||
                aggregate.exact != 0 || aggregate.none_groups != aggregate.groups ||
                (aggregate.degradation_flags & LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE) == 0) {
            LOG_ERR("%s: overlay body-only restore did not expose degraded coverage\n", __func__);
            return false;
        }
    } else if (coverage_after.state != coverage_before.state ||
            coverage_after.requested != coverage_before.requested ||
            coverage_after.exact != coverage_before.exact || coverage_after.degradation_flags != 0 ||
            aggregate.degradation_flags != 0) {
        LOG_ERR("%s: native-exact body state did not preserve exact coverage\n", __func__);
        return false;
    }
    LOG("\nPASS: representation-specific full and body-only tail state\n");
    return true;
}

static bool test_cross_ubatch_tail_state(
        llama_model * model, const common_params & params, const llama_tokens & tokens,
        uint32_t source_ubatch, uint32_t destination_ubatch) {
    if (params.kv_tail_tokens.empty() ||
            !std::all_of(params.kv_tail_tokens.begin(), params.kv_tail_tokens.end(), ::isdigit) ||
            std::stoul(params.kv_tail_tokens) == 0) {
        return true;
    }

    auto source_params = common_context_params_to_llama(params);
    source_params.n_ubatch = source_ubatch;
    source_params.n_batch = std::max<uint32_t>(source_params.n_batch, source_ubatch);
    auto destination_params = source_params;
    destination_params.n_ubatch = destination_ubatch;
    destination_params.n_batch = std::max<uint32_t>(destination_params.n_batch, destination_ubatch);

    auto source = llama_context_ptr{llama_init_from_model(model, source_params)};
    auto destination = llama_context_ptr{llama_init_from_model(model, destination_params)};
    if (!source || !destination) {
        LOG_ERR("%s: failed to create ubatch %u -> %u contexts\n",
                __func__, source_ubatch, destination_ubatch);
        return false;
    }

    int n_past = 0;
    if (!common_prompt_batch_decode(source.get(), tokens, int(tokens.size()), n_past,
            source_params.n_batch, {}, false)) {
        return false;
    }

    std::vector<uint8_t> state(llama_state_get_size(source.get()));
    if (llama_state_get_data(source.get(), state.data(), state.size()) != state.size() ||
            llama_state_set_data(destination.get(), state.data(), state.size()) != state.size()) {
        LOG_ERR("%s: full-state transfer failed for ubatch %u -> %u\n",
                __func__, source_ubatch, destination_ubatch);
        return false;
    }

    const llama_token probe = tokens.empty() ? 1 : tokens.back();
    llama_batch_ptr source_batch(1, 0, 1);
    llama_batch_ptr destination_batch(1, 0, 1);
    common_batch_add(source_batch.get(), probe, n_past, {0}, true);
    common_batch_add(destination_batch.get(), probe, n_past, {0}, true);
    if (llama_decode(source.get(), source_batch.get()) ||
            llama_decode(destination.get(), destination_batch.get())) {
        LOG_ERR("%s: probe decode failed for ubatch %u -> %u\n",
                __func__, source_ubatch, destination_ubatch);
        return false;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * expected = llama_get_logits_ith(source.get(), -1);
    const float * actual = llama_get_logits_ith(destination.get(), -1);
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double max_abs_error = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        const double diff = double(expected[i]) - double(actual[i]);
        squared_error += diff*diff;
        squared_reference += double(expected[i])*double(expected[i]);
        max_abs_error = std::max(max_abs_error, std::fabs(diff));
    }
    const double nmse = squared_error/std::max(squared_reference, 1e-30);
    if (!std::isfinite(nmse) || nmse > 1e-10 || max_abs_error > 1e-4) {
        LOG_ERR("%s: logits changed for ubatch %u -> %u (nmse=%g max_abs=%g)\n",
                __func__, source_ubatch, destination_ubatch, nmse, max_abs_error);
        return false;
    }

    LOG("\nPASS: logical tail state ubatch %u -> %u\n", source_ubatch, destination_ubatch);
    return true;
}

static bool test_tail_copy_is_immediately_saveable(
        llama_model * model, const common_params & params, const llama_tokens & tokens, bool unified) {
    if (params.kvarn.type != LLAMA_KVARN_TYPE_DISABLED || params.kv_tail_tokens.empty() ||
            !std::all_of(params.kv_tail_tokens.begin(), params.kv_tail_tokens.end(), ::isdigit) ||
            std::stoul(params.kv_tail_tokens) == 0) {
        return true;
    }

    auto context_params = common_context_params_to_llama(params);
    context_params.n_seq_max = 2;
    context_params.kv_unified = unified;
    auto source = llama_context_ptr{llama_init_from_model(model, context_params)};
    auto restored = llama_context_ptr{llama_init_from_model(model, context_params)};
    if (!source || !restored) {
        LOG_ERR("%s: failed to create sequence-copy contexts\n", __func__);
        return false;
    }
    if (!unified && std::stoul(params.kv_tail_tokens) >= llama_n_ctx_seq(source.get())) {
        // A tail that covers the complete rounded per-sequence window is
        // promoted to native exact storage and has no overlay frame to copy.
        return true;
    }

    int n_past = 0;
    if (!common_prompt_batch_decode(source.get(), tokens, int(tokens.size()), n_past,
            context_params.n_batch, {}, false)) {
        return false;
    }
    llama_memory_t memory = llama_get_memory(source.get());
    llama_memory_seq_cp(memory, 0, 1, 0, -1);
    if (llama_memory_seq_pos_max(memory, 1) != llama_memory_seq_pos_max(memory, 0)) {
        LOG_ERR("%s: copied body positions were not immediately observable\n", __func__);
        return false;
    }
    llama_kv_tail_coverage_info source_coverage{};
    llama_kv_tail_coverage_info pending_coverage{};
    if (!llama_kv_tail_get_coverage(source.get(), 0, 0, &source_coverage) ||
            !llama_kv_tail_get_coverage(source.get(), 1, 0, &pending_coverage) ||
            source_coverage.requested != pending_coverage.requested ||
            source_coverage.exact != pending_coverage.exact) {
        LOG_ERR("%s: pending exact coverage did not describe the logical copy "
                "(source requested=%u exact=%u flags=%u, destination requested=%u exact=%u flags=%u)\n",
                __func__, source_coverage.requested, source_coverage.exact, source_coverage.degradation_flags,
                pending_coverage.requested, pending_coverage.exact, pending_coverage.degradation_flags);
        LOG_ERR("%s: body positions source=%d destination=%d\n", __func__,
                llama_memory_seq_pos_max(memory, 0), llama_memory_seq_pos_max(memory, 1));
        return false;
    }

    const size_t state_size = llama_state_seq_get_size(source.get(), 1);
    if (state_size == 0 || state_size != llama_state_seq_get_size(source.get(), 1)) {
        LOG_ERR("%s: pending sequence state sizing was unstable\n", __func__);
        return false;
    }
    std::vector<uint8_t> state(state_size);
    if (llama_state_seq_get_data(source.get(), state.data(), state.size(), 1) != state.size() ||
            llama_state_seq_set_data(restored.get(), state.data(), state.size(), 1) != state.size()) {
        LOG_ERR("%s: immediate copied-sequence save/restore failed\n", __func__);
        return false;
    }

    auto corruption_guard = llama_context_ptr{llama_init_from_model(model, context_params)};
    if (!corruption_guard ||
            llama_state_seq_set_data(corruption_guard.get(), state.data(), state.size(), 1) != state.size()) {
        LOG_ERR("%s: failed to initialize corrupt-state destination guard\n", __func__);
        return false;
    }
    auto read_u32 = [&](const std::vector<uint8_t> & bytes, size_t offset) {
        uint32_t value;
        if (offset > bytes.size() || bytes.size() - offset < sizeof(value)) {
            throw std::runtime_error("state parser exceeded the buffer");
        }
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        return value;
    };
    auto write_u32 = [&](std::vector<uint8_t> & bytes, size_t offset, uint32_t value) {
        if (offset > bytes.size() || bytes.size() - offset < sizeof(value)) {
            throw std::runtime_error("state mutator exceeded the buffer");
        }
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    };
    const uint32_t tail_magic = 0x4c54564b;
    size_t frame = std::string::npos;
    for (size_t i = 0; i + sizeof(tail_magic) <= state.size(); ++i) {
        if (read_u32(state, i) == tail_magic) {
            frame = i;
            break;
        }
    }
    if (frame == std::string::npos || read_u32(state, frame + 4) != 5) {
        LOG_ERR("%s: could not locate standard tail state v5 frame\n", __func__);
        return false;
    }
    const uint32_t group_size = read_u32(state, frame + 28);
    const size_t manifest = frame + 32 + group_size + 3*sizeof(uint64_t);
    const uint32_t stream_count = read_u32(state, manifest);
    const uint32_t n_pos_per_embd = read_u32(state, manifest + 8);
    const uint32_t saved_n_seq_max = read_u32(state, manifest + 12);
    const uint32_t body_layer_count = read_u32(state, manifest + 16);
    size_t cursor = manifest + 20 + size_t(body_layer_count)*36;
    std::vector<uint32_t> body_cell_counts;
    for (uint32_t s = 0; s < stream_count; ++s) {
        const uint32_t cell_count = read_u32(state, cursor);
        const uint32_t run_count = read_u32(state, cursor + 4);
        body_cell_counts.push_back(cell_count);
        cursor += 8 + size_t(run_count)*sizeof(uint32_t);
        for (uint32_t i = 0; i < cell_count; ++i) {
            cursor += sizeof(uint32_t) + sizeof(uint64_t); // source cell + generation
            cursor += sizeof(llama_pos);
            if (n_pos_per_embd > 1) {
                cursor += sizeof(llama_pos)*2;
            }
            const uint32_t seq_count = read_u32(state, cursor);
            cursor += sizeof(uint32_t) + size_t(seq_count)*sizeof(llama_seq_id);
        }
    }
    const size_t tail_header = cursor;
    const uint32_t record_count = read_u32(state, tail_header + 8);
    const uint32_t payload_count = read_u32(state, tail_header + 12);
    const size_t record_begin = tail_header + 24;
    const size_t record_size = 40;
    const size_t tail_layer_begin = record_begin + size_t(record_count)*record_size;
    if (record_count < 2 || payload_count < 2 || body_cell_counts.empty()) {
        LOG_ERR("%s: tail state fixture did not contain enough records for corruption tests\n", __func__);
        return false;
    }

    const llama_pos guard_pos = llama_memory_seq_pos_max(llama_get_memory(corruption_guard.get()), 1);
    auto expect_rejected = [&](std::vector<uint8_t> corrupt, size_t supplied_size, const char * name) {
        if (llama_state_seq_set_data(corruption_guard.get(), corrupt.data(), supplied_size, 1) != 0) {
            LOG_ERR("%s: corrupt state case '%s' was accepted\n", __func__, name);
            return false;
        }
        if (llama_memory_seq_pos_max(llama_get_memory(corruption_guard.get()), 1) != guard_pos) {
            LOG_ERR("%s: corrupt state case '%s' changed the live destination\n", __func__, name);
            return false;
        }
        return true;
    };
    {
        auto corrupt = state;
        write_u32(corrupt, record_begin, saved_n_seq_max);
        if (!expect_rejected(std::move(corrupt), state.size(), "sequence ID")) return false;
    }
    {
        auto corrupt = state;
        write_u32(corrupt, record_begin + 4, stream_count);
        if (!expect_rejected(std::move(corrupt), state.size(), "stream")) return false;
    }
    {
        auto corrupt = state;
        const uint32_t record_stream = read_u32(corrupt, record_begin + 4);
        write_u32(corrupt, record_begin + 8, body_cell_counts.at(record_stream));
        if (!expect_rejected(std::move(corrupt), state.size(), "body ordinal")) return false;
    }
    {
        auto corrupt = state;
        write_u32(corrupt, record_begin + 32, payload_count);
        if (!expect_rejected(std::move(corrupt), state.size(), "payload ordinal")) return false;
    }
    {
        auto corrupt = state;
        write_u32(corrupt, record_begin + 36, UINT32_MAX);
        if (!expect_rejected(std::move(corrupt), state.size(), "tail ring slot")) return false;
    }
    {
        auto corrupt = state;
        std::memcpy(corrupt.data() + record_begin + record_size,
                corrupt.data() + record_begin, 3*sizeof(uint32_t));
        if (!expect_rejected(std::move(corrupt), state.size(), "duplicate identity")) return false;
    }
    {
        auto corrupt = state;
        write_u32(corrupt, record_begin + 32, 1);
        if (!expect_rejected(std::move(corrupt), state.size(), "unreferenced payload")) return false;
    }
    {
        auto corrupt = state;
        write_u32(corrupt, tail_layer_begin, read_u32(corrupt, tail_layer_begin) + 1);
        if (!expect_rejected(std::move(corrupt), state.size(), "layer ID")) return false;
    }
    {
        auto corrupt = state;
        write_u32(corrupt, tail_header + 12, UINT32_MAX);
        if (!expect_rejected(std::move(corrupt), state.size(), "slot exhaustion")) return false;
    }
    if (!expect_rejected(state, state.size() - 1, "truncated row")) {
        return false;
    }

    const llama_token probe = tokens.empty() ? 1 : tokens.back();
    llama_batch_ptr source_batch(1, 0, 1);
    llama_batch_ptr restored_batch(1, 0, 1);
    llama_batch_ptr guard_batch(1, 0, 1);
    common_batch_add(source_batch.get(), probe, n_past, {0}, true);
    common_batch_add(restored_batch.get(), probe, n_past, {1}, true);
    common_batch_add(guard_batch.get(), probe, n_past, {1}, true);
    if (llama_decode(source.get(), source_batch.get()) || llama_decode(restored.get(), restored_batch.get()) ||
            llama_decode(corruption_guard.get(), guard_batch.get())) {
        LOG_ERR("%s: continuation decode failed\n", __func__);
        return false;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * expected = llama_get_logits_ith(source.get(), -1);
    const float * actual = llama_get_logits_ith(restored.get(), -1);
    const float * guarded = llama_get_logits_ith(corruption_guard.get(), -1);
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double max_abs_error = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        const double diff = double(expected[i]) - double(actual[i]);
        const double guard_diff = double(expected[i]) - double(guarded[i]);
        squared_error += diff*diff;
        squared_error += guard_diff*guard_diff;
        squared_reference += double(expected[i])*double(expected[i]);
        max_abs_error = std::max(max_abs_error, std::fabs(diff));
        max_abs_error = std::max(max_abs_error, std::fabs(guard_diff));
    }
    const double nmse = squared_error/std::max(squared_reference, 1e-30);
    if (!std::isfinite(nmse) || nmse > 1e-10 || max_abs_error > 1e-4) {
        LOG_ERR("%s: immediate copied-sequence continuation changed logits (nmse=%g max_abs=%g)\n",
                __func__, nmse, max_abs_error);
        return false;
    }
    LOG("\nPASS: standard tail copy is immediately saveable (%s)\n", unified ? "unified" : "non-unified");
    return true;
}

static bool test_kvarn_full_window_native_exact(
        llama_model * model, const common_params & params, const llama_tokens & tokens) {
    if (params.kvarn.type == LLAMA_KVARN_TYPE_DISABLED ||
            params.kv_tail_tokens.empty() ||
            !std::all_of(params.kv_tail_tokens.begin(), params.kv_tail_tokens.end(), ::isdigit)) {
        return true;
    }

    auto candidate_params = common_context_params_to_llama(params);
    if (candidate_params.n_ctx == 0 || candidate_params.kv_tail_tokens < candidate_params.n_ctx) {
        return true;
    }

    auto oracle_params = candidate_params;
    oracle_params.kvarn = llama_kvarn_default_params();
    oracle_params.type_k = candidate_params.kv_tail_type;
    oracle_params.type_v = candidate_params.kv_tail_type;
    oracle_params.kv_tail_tokens = 0;
    oracle_params.kv_tail_config = nullptr;

    auto candidate = llama_context_ptr{llama_init_from_model(model, candidate_params)};
    auto oracle = llama_context_ptr{llama_init_from_model(model, oracle_params)};
    if (!candidate || !oracle) {
        LOG_ERR("%s: failed to create promoted/oracle contexts\n", __func__);
        return false;
    }

    int candidate_past = 0;
    int oracle_past = 0;
    if (!common_prompt_batch_decode(candidate.get(), tokens, int(tokens.size()), candidate_past,
                candidate_params.n_batch, {}, false) ||
            !common_prompt_batch_decode(oracle.get(), tokens, int(tokens.size()), oracle_past,
                oracle_params.n_batch, {}, false)) {
        LOG_ERR("%s: failed to decode promoted/oracle prompts\n", __func__);
        return false;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * actual = llama_get_logits_ith(candidate.get(), -1);
    const float * expected = llama_get_logits_ith(oracle.get(), -1);
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double max_abs_error = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        const double diff = double(actual[i]) - double(expected[i]);
        squared_error += diff*diff;
        squared_reference += double(expected[i])*double(expected[i]);
        max_abs_error = std::max(max_abs_error, std::fabs(diff));
    }
    const double nmse = squared_error/std::max(squared_reference, 1e-30);
    if (!std::isfinite(nmse) || nmse > 1e-10 || max_abs_error > 1e-4) {
        LOG_ERR("%s: promoted exact cache differs from direct %s cache (nmse=%g max_abs=%g)\n",
                __func__, ggml_type_name(candidate_params.kv_tail_type), nmse, max_abs_error);
        return false;
    }

    LOG("\nPASS: full-window KVarN promotion matches direct %s cache\n",
            ggml_type_name(candidate_params.kv_tail_type));
    return true;
}

static bool test_kvarn_partial_checkpoint_history(
        llama_model * model, const common_params & params, const llama_tokens & tokens) {
    if (params.kvarn.type == LLAMA_KVARN_TYPE_DISABLED || params.kv_tail_tokens.empty() ||
            !std::all_of(params.kv_tail_tokens.begin(), params.kv_tail_tokens.end(), ::isdigit) ||
            std::stoul(params.kv_tail_tokens) == 0) {
        return true;
    }

    auto context_params = common_context_params_to_llama(params);
    context_params.n_seq_max = 1;
    const uint32_t prefix_tokens = context_params.kv_tail_tokens + context_params.n_ubatch + 1;
    context_params.n_ctx = std::max(context_params.n_ctx, prefix_tokens + 16);
    auto context = llama_context_ptr{llama_init_from_model(model, context_params)};
    if (!context) {
        LOG_ERR("%s: failed to create KVarN checkpoint context\n", __func__);
        return false;
    }

    llama_tokens prefix;
    prefix.reserve(prefix_tokens);
    for (uint32_t i = 0; i < prefix_tokens; ++i) {
        prefix.push_back(tokens.empty() ? llama_token(1) : tokens[i % tokens.size()]);
    }
    int n_past = 0;
    for (size_t offset = 0; offset < prefix.size();) {
        const int32_t count = int32_t(std::min<size_t>(context_params.n_batch, prefix.size() - offset));
        llama_batch_ptr batch(count, 0, 1);
        for (int32_t i = 0; i < count; ++i) {
            common_batch_add(batch.get(), prefix[offset + i], n_past + i, { 0 }, false);
        }
        if (llama_decode(context.get(), batch.get())) {
            LOG_ERR("%s: failed to decode wrapped-tail prefix chunk\n", __func__);
            return false;
        }
        offset += count;
        n_past += count;
    }

    {
        auto device_params = context_params;
        device_params.n_seq_max = 2;
        device_params.n_ctx = std::max(device_params.n_ctx, 2*(prefix_tokens + 16));
        auto device_context = llama_context_ptr{llama_init_from_model(model, device_params)};
        if (!device_context) {
            LOG_ERR("%s: failed to create wrapped on-device checkpoint context\n", __func__);
            return false;
        }

        int device_past = 0;
        for (size_t offset = 0; offset < prefix.size();) {
            const int32_t count = int32_t(std::min<size_t>(device_params.n_batch, prefix.size() - offset));
            llama_batch_ptr batch(count, 0, 1);
            for (int32_t i = 0; i < count; ++i) {
                common_batch_add(batch.get(), prefix[offset + i], device_past + i, { 0 }, false);
            }
            if (llama_decode(device_context.get(), batch.get())) {
                LOG_ERR("%s: failed to decode wrapped on-device prefix chunk\n", __func__);
                return false;
            }
            offset += count;
            device_past += count;
        }

        const auto device_flags = llama_state_seq_flags(LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        std::vector<uint8_t> device_state(llama_state_seq_get_size_ext(
                device_context.get(), 0, device_flags));
        if (device_state.empty() || llama_state_seq_get_data_ext(
                device_context.get(), device_state.data(), device_state.size(), 0, device_flags) != device_state.size()) {
            LOG_ERR("%s: failed to save wrapped on-device checkpoint\n", __func__);
            return false;
        }

        const llama_token device_probe = tokens.empty() ? llama_token(2) : tokens.front();
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        llama_batch_ptr source_probe(1, 0, 1);
        common_batch_add(source_probe.get(), device_probe, device_past, { 0 }, true);
        if (llama_decode(device_context.get(), source_probe.get())) {
            LOG_ERR("%s: failed to decode wrapped on-device source probe\n", __func__);
            return false;
        }
        const float * source_values = llama_get_logits_ith(device_context.get(), -1);
        std::vector<float> source_logits(source_values, source_values + n_vocab);

        llama_memory_clear(llama_get_memory(device_context.get()), true);
        if (llama_state_seq_set_data_ext(
                device_context.get(), device_state.data(), device_state.size(), 1, device_flags) != device_state.size()) {
            LOG_ERR("%s: failed to restore wrapped on-device checkpoint\n", __func__);
            return false;
        }
        llama_batch_ptr destination_probe(1, 0, 1);
        common_batch_add(destination_probe.get(), device_probe, device_past, { 1 }, true);
        if (llama_decode(device_context.get(), destination_probe.get())) {
            LOG_ERR("%s: failed to decode wrapped on-device destination probe\n", __func__);
            return false;
        }
        const float * destination_logits = llama_get_logits_ith(device_context.get(), -1);
        double squared_error = 0.0;
        double squared_reference = 0.0;
        double max_abs_error = 0.0;
        for (int32_t i = 0; i < n_vocab; ++i) {
            const double diff = double(source_logits[i]) - double(destination_logits[i]);
            squared_error += diff*diff;
            squared_reference += double(source_logits[i])*double(source_logits[i]);
            max_abs_error = std::max(max_abs_error, std::fabs(diff));
        }
        const double nmse = squared_error/std::max(squared_reference, 1e-30);
        if (!std::isfinite(nmse) || nmse > 1e-10 || max_abs_error > 1e-4) {
            LOG_ERR("%s: wrapped on-device continuation changed (nmse=%g max_abs=%g)\n",
                    __func__, nmse, max_abs_error);
            return false;
        }
    }

    const auto partial_flags = llama_state_seq_flags(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    std::vector<double> save_times_ms;
    std::vector<double> restore_times_ms;
    const auto save_partial = [&](std::vector<uint8_t> & state) {
        const auto start = std::chrono::steady_clock::now();
        state.resize(llama_state_seq_get_size_ext(context.get(), 0, partial_flags));
        const bool result = !state.empty() && llama_state_seq_get_data_ext(
                context.get(), state.data(), state.size(), 0, partial_flags) == state.size();
        save_times_ms.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
        return result;
    };
    const auto restore_partial = [&](const std::vector<uint8_t> & state) {
        const auto start = std::chrono::steady_clock::now();
        const bool result = llama_state_seq_set_data_ext(
                context.get(), state.data(), state.size(), 0, partial_flags) == state.size();
        restore_times_ms.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
        return result;
    };
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto decode_probe = [&](llama_token token, int position, std::vector<float> & logits) {
        llama_batch_ptr batch(1, 0, 1);
        common_batch_add(batch.get(), token, position, { 0 }, true);
        if (llama_decode(context.get(), batch.get())) {
            return false;
        }
        const float * values = llama_get_logits_ith(context.get(), -1);
        logits.assign(values, values + n_vocab);
        return true;
    };
    const auto logits_match = [&](const std::vector<float> & expected, const std::vector<float> & actual) {
        double squared_error = 0.0;
        double squared_reference = 0.0;
        double max_abs_error = 0.0;
        for (int32_t i = 0; i < n_vocab; ++i) {
            const double diff = double(expected[i]) - double(actual[i]);
            squared_error += diff*diff;
            squared_reference += double(expected[i])*double(expected[i]);
            max_abs_error = std::max(max_abs_error, std::fabs(diff));
        }
        const double nmse = squared_error/std::max(squared_reference, 1e-30);
        return std::isfinite(nmse) && nmse <= 1e-10 && max_abs_error <= 1e-4;
    };
    const llama_token probe_a = tokens.empty() ? llama_token(2) : tokens.front();
    const llama_token probe_b = tokens.size() < 2 ? llama_token(3) : tokens[1];
    const llama_token extension = tokens.size() < 3 ? llama_token(4) : tokens[2];

    std::vector<uint8_t> checkpoint_a;
    if (!save_partial(checkpoint_a)) {
        LOG_ERR("%s: failed to save checkpoint A\n", __func__);
        return false;
    }
    std::vector<float> oracle_a;
    if (!decode_probe(probe_a, n_past, oracle_a) || !restore_partial(checkpoint_a)) {
        LOG_ERR("%s: failed to establish checkpoint A oracle\n", __func__);
        return false;
    }

    llama_batch_ptr extension_batch(3, 0, 1);
    for (int i = 0; i < 3; ++i) {
        common_batch_add(extension_batch.get(), extension, n_past + i, { 0 }, i == 2);
    }
    if (llama_decode(context.get(), extension_batch.get())) {
        LOG_ERR("%s: failed to advance from checkpoint A to B\n", __func__);
        return false;
    }
    const int checkpoint_b_past = n_past + 3;
    std::vector<uint8_t> checkpoint_b;
    if (!save_partial(checkpoint_b)) {
        LOG_ERR("%s: failed to save checkpoint B\n", __func__);
        return false;
    }
    std::vector<float> oracle_b;
    if (!decode_probe(probe_b, checkpoint_b_past, oracle_b) || !restore_partial(checkpoint_b)) {
        LOG_ERR("%s: failed to establish checkpoint B oracle\n", __func__);
        return false;
    }

    llama_batch_ptr mutation_batch(2, 0, 1);
    common_batch_add(mutation_batch.get(), extension, checkpoint_b_past, { 0 }, false);
    common_batch_add(mutation_batch.get(), extension, checkpoint_b_past + 1, { 0 }, true);
    if (llama_decode(context.get(), mutation_batch.get())) {
        LOG_ERR("%s: failed to mutate live checkpoint state\n", __func__);
        return false;
    }
    const int live_past = checkpoint_b_past + 2;
    std::vector<uint8_t> live_state;
    if (!save_partial(live_state)) {
        LOG_ERR("%s: failed to save live corruption guard\n", __func__);
        return false;
    }
    std::vector<float> live_oracle;
    if (!decode_probe(probe_a, live_past, live_oracle) || !restore_partial(live_state)) {
        LOG_ERR("%s: failed to establish live corruption oracle\n", __func__);
        return false;
    }

    const auto self_flags = llama_state_seq_flags(LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED);
    std::vector<uint8_t> live_self(llama_state_seq_get_size_ext(context.get(), 0, self_flags));
    if (live_self.empty() || llama_state_seq_get_data_ext(
                context.get(), live_self.data(), live_self.size(), 0, self_flags) != live_self.size()) {
        LOG_ERR("%s: failed to save self-contained restore anchor\n", __func__);
        return false;
    }
    std::unique_ptr<llama_state_seq_restore_plan, decltype(&llama_state_seq_restore_plan_free)> prepared(
            llama_state_seq_prepare_data_ext(
                    context.get(), checkpoint_b.data(), checkpoint_b.size(), 0, partial_flags),
            llama_state_seq_restore_plan_free);
    if (!prepared || llama_memory_seq_pos_max(llama_get_memory(context.get()), 0) != live_past - 1) {
        LOG_ERR("%s: prepared restore mutated the live destination\n", __func__);
        return false;
    }
    if (llama_state_seq_restore_plan_commit(prepared.get()) != checkpoint_b.size() ||
            llama_memory_seq_pos_max(llama_get_memory(context.get()), 0) != checkpoint_b_past - 1 ||
            llama_state_seq_set_data_ext(
                    context.get(), live_self.data(), live_self.size(), 0, self_flags) != live_self.size()) {
        LOG_ERR("%s: prepared restore commit or anchor recovery failed\n", __func__);
        return false;
    }

    const uint32_t kvarn_magic = 0x4e52564b;
    size_t kvarn_frame = std::string::npos;
    for (size_t i = 0; i + 2*sizeof(uint32_t) <= checkpoint_b.size(); ++i) {
        uint32_t value;
        std::memcpy(&value, checkpoint_b.data() + i, sizeof(value));
        if (value == kvarn_magic) {
            kvarn_frame = i;
            break;
        }
    }
    uint32_t version = 0;
    if (kvarn_frame != std::string::npos) {
        std::memcpy(&version, checkpoint_b.data() + kvarn_frame + sizeof(uint32_t), sizeof(version));
    }
    if (kvarn_frame == std::string::npos || version != 15) {
        LOG_ERR("%s: expected KVarN partial checkpoint format v15, found v%u\n", __func__, version);
        return false;
    }

    auto corrupt = checkpoint_b;
    const uint32_t unsupported_version = 16;
    std::memcpy(corrupt.data() + kvarn_frame + sizeof(uint32_t), &unsupported_version, sizeof(unsupported_version));
    std::unique_ptr<llama_state_seq_restore_plan, decltype(&llama_state_seq_restore_plan_free)> corrupt_plan(
            llama_state_seq_prepare_data_ext(
                    context.get(), corrupt.data(), corrupt.size(), 0, partial_flags),
            llama_state_seq_restore_plan_free);
    if (corrupt_plan) {
        LOG_ERR("%s: corrupt KVarN v15 frame produced a restore plan\n", __func__);
        return false;
    }
    if (llama_state_seq_set_data_ext(
            context.get(), corrupt.data(), corrupt.size(), 0, partial_flags) != 0) {
        LOG_ERR("%s: corrupt KVarN v15 frame was accepted\n", __func__);
        return false;
    }
    if (llama_memory_seq_pos_max(llama_get_memory(context.get()), 0) != live_past - 1) {
        LOG_ERR("%s: rejected KVarN frame changed live metadata\n", __func__);
        return false;
    }
    std::vector<float> live_after_reject;
    if (!decode_probe(probe_a, live_past, live_after_reject) ||
            !logits_match(live_oracle, live_after_reject)) {
        LOG_ERR("%s: rejected KVarN frame changed live tensor state\n", __func__);
        return false;
    }

    if (const char * path = std::getenv("KVARN_TEST_V12_STATE")) {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        std::streamsize size = -1;
        if (input) {
            size = static_cast<std::streamsize>(input.tellg());
        }
        if (size <= 0) {
            LOG_ERR("%s: failed to open v12 compatibility fixture '%s'\n", __func__, path);
            return false;
        }
        input.seekg(0);
        std::vector<uint8_t> v12_state(static_cast<size_t>(size));
        input.read(reinterpret_cast<char *>(v12_state.data()), size);
        std::vector<float> restored_v12;
        if (!input || !restore_partial(v12_state) ||
                !decode_probe(probe_b, checkpoint_b_past, restored_v12) ||
                !logits_match(oracle_b, restored_v12)) {
            LOG_ERR("%s: v12 partial checkpoint continuation changed\n", __func__);
            return false;
        }

        auto mismatch_params = context_params;
        ++mismatch_params.kv_tail_tokens;
        auto mismatch = llama_context_ptr{llama_init_from_model(model, mismatch_params)};
        if (!mismatch || llama_state_seq_set_data_ext(
                mismatch.get(), v12_state.data(), v12_state.size(), 0, partial_flags) != 0) {
            LOG_ERR("%s: mismatched KVarN context accepted the v12 checkpoint\n", __func__);
            return false;
        }
        llama_batch_ptr mismatch_probe(1, 0, 1);
        common_batch_add(mismatch_probe.get(), probe_a, 0, { 0 }, true);
        if (llama_decode(mismatch.get(), mismatch_probe.get())) {
            LOG_ERR("%s: rejected v12 checkpoint left its destination unusable\n", __func__);
            return false;
        }
    }

    std::vector<float> restored_a;
    if (!restore_partial(checkpoint_a) || !decode_probe(probe_a, n_past, restored_a) ||
            !logits_match(oracle_a, restored_a)) {
        LOG_ERR("%s: historical checkpoint A continuation changed\n", __func__);
        return false;
    }
    // Partial state references the still-live ordinary-cache body. Restore the
    // self-contained anchor before testing another historical checkpoint;
    // checkpoint A legitimately removed B's later reference cells.
    if (llama_state_seq_set_data_ext(
                context.get(), live_self.data(), live_self.size(), 0, self_flags) != live_self.size()) {
        LOG_ERR("%s: failed to reestablish live anchor before checkpoint B\n", __func__);
        return false;
    }
    std::vector<float> restored_b;
    if (!restore_partial(checkpoint_b) || !decode_probe(probe_b, checkpoint_b_past, restored_b) ||
            !logits_match(oracle_b, restored_b)) {
        LOG_ERR("%s: historical checkpoint B continuation changed\n", __func__);
        return false;
    }

    // Establish the byte baseline immediately before prepare/free so the
    // lifetime contract is isolated from unrelated state captures above.
    std::vector<uint8_t> before_abandon;
    if (!save_partial(before_abandon)) {
        LOG_ERR("%s: failed to save abandoned-plan destination baseline\n", __func__);
        return false;
    }
    {
        const llama_pos pos_before = llama_memory_seq_pos_max(llama_get_memory(context.get()), 0);
        std::unique_ptr<llama_state_seq_restore_plan, decltype(&llama_state_seq_restore_plan_free)> abandoned(
                llama_state_seq_prepare_data_ext(
                        context.get(), checkpoint_b.data(), checkpoint_b.size(), 0, partial_flags),
                llama_state_seq_restore_plan_free);
        if (!abandoned || llama_memory_seq_pos_max(llama_get_memory(context.get()), 0) != pos_before) {
            LOG_ERR("%s: abandoned restore preparation mutated the live destination\n", __func__);
            return false;
        }
        abandoned.reset();
        std::vector<uint8_t> after_abandon;
        if (!save_partial(after_abandon) || after_abandon != before_abandon) {
            LOG_ERR("%s: destroying an uncommitted restore plan changed the destination\n", __func__);
            return false;
        }
    }

    LOG("\nKVarN partial checkpoint bytes: A=%zu B=%zu live=%zu\n",
            checkpoint_a.size(), checkpoint_b.size(), live_state.size());
    LOG("KVarN partial checkpoint save ms:");
    for (const double time_ms : save_times_ms) {
        LOG(" %.3f", time_ms);
    }
    LOG("\nKVarN partial checkpoint restore ms:");
    for (const double time_ms : restore_times_ms) {
        LOG(" %.3f", time_ms);
    }
    LOG("\n");
    LOG("\nPASS: KVarN v15 host partial checkpoints are transactional with a live body anchor\n");
    return true;
}

static bool test_kvarn_unified_capacity(
        llama_model * model, const common_params & params, const llama_tokens & tokens) {
    if (params.kvarn.type == LLAMA_KVARN_TYPE_DISABLED) {
        return true;
    }

    auto context_params = common_context_params_to_llama(params);
    context_params.n_ctx = 512;
    context_params.n_batch = 128;
    context_params.n_ubatch = 128;
    context_params.n_seq_max = 2;
    context_params.kv_unified = true;
    context_params.kv_tail_tokens = 0;
    context_params.kv_tail_config = nullptr;
    auto context = llama_context_ptr{llama_init_from_model(model, context_params)};
    if (!context) {
        LOG_ERR("%s: failed to create unified KVarN capacity context\n", __func__);
        return false;
    }

    constexpr int32_t borrowed_capacity = 384;
    for (int32_t offset = 0; offset < borrowed_capacity; offset += 128) {
        llama_batch_ptr batch(128, 0, 1);
        for (int32_t i = 0; i < 128; ++i) {
            const llama_token token = tokens.empty() ? llama_token(1) : tokens[(offset + i) % tokens.size()];
            common_batch_add(batch.get(), token, offset + i, { 0 }, false);
        }
        if (llama_decode(context.get(), batch.get())) {
            LOG_ERR("%s: sequence 0 could not borrow unused unified KVarN capacity\n", __func__);
            return false;
        }
    }

    llama_batch_ptr other(128, 0, 1);
    for (int32_t i = 0; i < 128; ++i) {
        const llama_token token = tokens.empty() ? llama_token(1) : tokens[i % tokens.size()];
        common_batch_add(other.get(), token, i, { 1 }, false);
    }
    if (llama_decode(context.get(), other.get())) {
        LOG_ERR("%s: borrowing sequence aliased the remaining unified KVarN capacity\n", __func__);
        return false;
    }

    LOG("\nPASS: unified KVarN capacity is borrowed between sequences without cross-sequence aliasing\n");
    return true;
}

static bool test_kvarn_unified_reuses_freed_groups(
        llama_model * model, const common_params & params, const llama_tokens & tokens) {
    if (params.kvarn.type == LLAMA_KVARN_TYPE_DISABLED) {
        return true;
    }

    auto context_params = common_context_params_to_llama(params);
    context_params.n_ctx = 512;
    context_params.n_batch = 128;
    context_params.n_ubatch = 128;
    context_params.n_seq_max = 2;
    context_params.kv_unified = true;
    context_params.kv_tail_tokens = 0;
    context_params.kv_tail_config = nullptr;
    auto context = llama_context_ptr{llama_init_from_model(model, context_params)};
    if (!context) {
        LOG_ERR("%s: failed to create unified KVarN reuse context\n", __func__);
        return false;
    }

    const auto decode = [&](llama_seq_id seq_id, int32_t pos0, int32_t count) {
        for (int32_t offset = 0; offset < count; offset += 128) {
            const int32_t n_tokens = std::min(128, count - offset);
            llama_batch_ptr batch(n_tokens, 0, 1);
            for (int32_t i = 0; i < n_tokens; ++i) {
                const int32_t pos = pos0 + offset + i;
                const llama_token token = tokens.empty() ? llama_token(1) : tokens[pos % tokens.size()];
                common_batch_add(batch.get(), token, pos, { seq_id }, offset + n_tokens == count && i + 1 == n_tokens);
            }
            if (llama_decode(context.get(), batch.get())) {
                return false;
            }
        }
        return true;
    };

    // Sequence 1 temporarily owns physical group 1 while sequence 0 advances
    // through groups 0, 2, and 3. Once sequence 1 is removed, sequence 0 must
    // wrap into the freed group without corrupting either sequence's metadata.
    if (!decode(0, 0, 128) || !decode(1, 0, 128) || !decode(0, 128, 256)) {
        LOG_ERR("%s: failed to exercise freed unified KVarN groups\n", __func__);
        return false;
    }

    auto * memory = llama_get_memory(context.get());
    if (!llama_memory_seq_rm(memory, 1, -1, -1)) {
        LOG_ERR("%s: failed to release the temporary sequence\n", __func__);
        return false;
    }
    if (!decode(0, 384, 1)) {
        LOG_ERR("%s: failed to reuse a freed unified KVarN group\n", __func__);
        return false;
    }
    if (llama_memory_seq_pos_max(memory, 0) != 384 || llama_memory_seq_pos_max(memory, 1) != -1) {
        LOG_ERR("%s: freed-group reuse corrupted sequence positions\n", __func__);
        return false;
    }
    const float * logits = llama_get_logits_ith(context.get(), -1);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (!logits || !std::all_of(logits, logits + n_vocab, [](float value) { return std::isfinite(value); })) {
        LOG_ERR("%s: freed-group reuse produced invalid continuation logits\n", __func__);
        return false;
    }

    LOG("\nPASS: unified KVarN reuses freed groups with valid sequence state and logits\n");
    return true;
}

static bool test_tail_state_v1_compatibility(llama_model * model, const common_params & params) {
    const char * path = std::getenv("KV_TAIL_TEST_V1_STATE");
    if (!path || !*path || params.kv_tail_tokens.empty() || std::stoul(params.kv_tail_tokens) == 0) {
        return true;
    }
    auto context = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!context) {
        LOG_ERR("%s: failed to create v1 compatibility context\n", __func__);
        return false;
    }
    std::vector<llama_token> saved_tokens(std::max<uint32_t>(params.n_ctx, params.n_batch));
    size_t token_count = 0;
    if (!llama_state_load_file(context.get(), path, saved_tokens.data(), saved_tokens.size(), &token_count)) {
        LOG_ERR("%s: failed to load v1 tail state '%s'\n", __func__, path);
        return false;
    }
    llama_kv_tail_coverage_info coverage{};
    if (!llama_kv_tail_get_coverage(context.get(), 0, 0, &coverage) || coverage.exact == 0 ||
            (coverage.degradation_flags & LLAMA_KV_TAIL_DEGRADED_STATE_RESTORE) == 0) {
        LOG_ERR("%s: v1 state upgraded missing provenance (exact=%u flags=%u)\n",
                __func__, coverage.exact, coverage.degradation_flags);
        return false;
    }
    LOG("\nPASS: v1 tail state loads with conservative degradation provenance\n");
    return true;
}

// Test 1: baseline
// - decode all but the last token
// - save state to disk
// - decode the last token
// - generate n_predict tokens
static llama_tokens test_baseline(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!ctx) {
        LOG_ERR("%s: failed to create baseline context\n", __func__);
        return {};
    }

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto n_past = 0;
    if (!common_prompt_batch_decode(ctx.get(), tokens, (int)tokens.size(), n_past, params.n_batch, params.out_file, true)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return {};
    }

    LOG("\n=== Test 1: baseline ===\n");

    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return {};
    }

    LOG("\n");

    return result;
}


// Test 2: sequence removal isolation
// - decode the same prefix into two sequences
// - remove sequence 0
// - verify that sequence 1 remains unchanged
static bool test_seq_rm_isolated(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    if (params.kvarn.type != LLAMA_KVARN_TYPE_DISABLED) {
        return true;
    }
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;

    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return false;
    }

    LOG("\n=== Test 2: sequence removal isolation ===\n");

    const size_t n_tokens = tokens.size() < 128 ? tokens.size() : 128;
    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        llama_batch_ptr batch(n_tokens, 0, 1);
        for (size_t i = 0; i < n_tokens; ++i) {
            common_batch_add(batch.get(), tokens[i], i, { seq_id }, false);
        }

        if (llama_decode(ctx.get(), batch.get())) {
            LOG_ERR("%s: failed to decode prompt for sequence %d\n", __func__, seq_id);
            return false;
        }
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size(ctx.get(), seq_id);
        if (state_size == 0) {
            LOG_ERR("%s: sequence state is empty\n", __func__);
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), state.data(), state.size(), seq_id);
        if (ncopy != state.size()) {
            LOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                    __func__, ncopy, state.size());
            return false;
        }

        return true;
    };

    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, state_before)) {
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return false;
    }

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, state_after)) {
        return false;
    }

    if (state_before != state_after) {
        LOG_ERR("%s: removing sequence 0 changed sequence 1\n", __func__);
        return false;
    }

    LOG("PASS\n");
    return true;
}


// Test 3: state load
// - create a new context
// - load state from file
// - replay the last prompt token
// - generate n_predict tokens and compare against expected result
static bool test_state_load(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
    if (!ctx) {
        LOG_ERR("%s: failed to create state-load context\n", __func__);
        return false;
    }

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 3: state load ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Generate tokens
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 4: seq copy (host)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the CPU path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_host(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create host sequence-copy context\n", __func__);
        return false;
    }

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 4: seq copy (host) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits_before_ptr = llama_get_logits_ith(ctx.get(), -1);
    std::vector<float> logits_before(logits_before_ptr, logits_before_ptr + n_vocab);

    // Migrate KV cache from seq 0 to seq 1 (CPU path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size(ctx.get(), 0));
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), seq_store.data(), seq_store.size(), 0);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data(ctx.get(), seq_store.data(), seq_store.size(), 1);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    const float * logits_after = llama_get_logits_ith(ctx.get(), -1);
    if (!std::equal(logits_before.begin(), logits_before.end(), logits_after)) {
        LOG_ERR("\n%s: state-only sequence migration changed output logits\n", __func__);
        return false;
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 5: seq copy (device)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the on-device path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_device(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create device sequence-copy context\n", __func__);
        return false;
    }

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 5: seq copy (device) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;


    // Migrate KV cache from seq 0 to seq 1 (on-device path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }


    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "";
    params.n_batch = 100;
    params.out_file = "dump_state.bin";
    params.sampling.seed = 1234;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.n_parallel == 1) {
        LOG_TRC("%s: n_parallel == 1, enabling unified kv cache\n", __func__);
        params.kv_unified = true;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    ggml_backend_load_all();

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    GGML_ASSERT(llama_init->context() == nullptr);

    // Tokenize prompt or generate random tokens
    llama_tokens tokens;
    if (params.prompt.empty()) {
        const int n_prompt = params.n_batch;

        // this path is useful for model files that do not have a tokenizer
        LOG_INF("%s: no prompt provided, generating %d (n_batch) random tokens\n", __func__, n_prompt);

        const auto * vocab = llama_model_get_vocab(model);
        const auto n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < n_prompt; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        LOG_INF("%s: tokenizing prompt '%s'\n", __func__, params.prompt.c_str());

        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        if (!ctx) {
            LOG_ERR("%s: failed to create prompt-tokenization context\n", __func__);
            return 1;
        }
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    LOG_INF("%s: the input prompt is %d tokens\n", __func__, (int)tokens.size());

    // Test 1: baseline (saves state to disk)
    auto result_baseline = test_baseline(model, params, tokens);
    if (result_baseline.empty()) {
        return 1;
    }
    if (!test_kvarn_partial_checkpoint_history(model, params, tokens)) {
        return 1;
    }
    if (!test_kvarn_unified_capacity(model, params, tokens)) {
        return 1;
    }
    if (!test_kvarn_unified_reuses_freed_groups(model, params, tokens)) {
        return 1;
    }

    if (!test_tail_state_contract(model, params, tokens)) {
        return 1;
    }
    if (!test_cross_ubatch_tail_state(model, params, tokens, 128, 512) ||
            !test_cross_ubatch_tail_state(model, params, tokens, 512, 128)) {
        return 1;
    }
    if (!test_tail_copy_is_immediately_saveable(model, params, tokens, true) ||
            !test_tail_copy_is_immediately_saveable(model, params, tokens, false)) {
        return 1;
    }
    if (!test_tail_state_v1_compatibility(model, params)) {
        return 1;
    }
    if (!test_kvarn_full_window_native_exact(model, params, tokens)) {
        return 1;
    }
    // Test 2: sequence removal isolation
    if (!test_seq_rm_isolated(model, params, tokens)) {
        return 1;
    }

    // Test 3: state load
    if (!test_state_load(model, params, tokens, result_baseline)) {
        return 1;
    }

    // Test 4: seq copy (host)
    if (!test_seq_cp_host(model, params, tokens, result_baseline)) {
        return 1;
    }

    // Test 5: seq copy (device)
    if (!test_seq_cp_device(model, params, tokens, result_baseline)) {
        return 1;
    }

    LOG("\nAll tests passed.\n");

    return 0;
}
