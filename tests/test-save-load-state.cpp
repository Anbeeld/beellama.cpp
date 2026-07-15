#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <clocale>
#include <algorithm>
#include <cctype>
#include <cmath>
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

// Test 1: baseline
// - decode all but the last token
// - save state to disk
// - decode the last token
// - generate n_predict tokens
static llama_tokens test_baseline(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

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


// Test 2: state load
// - create a new context
// - load state from file
// - replay the last prompt token
// - generate n_predict tokens and compare against expected result
static bool test_state_load(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 2: state load ===\n");

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


// Test 3: seq copy (host)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the CPU path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_host(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 3: seq copy (host) ===\n");

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


// Test 4: seq copy (device)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the on-device path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_device(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 4: seq copy (device) ===\n");

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
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    LOG_INF("%s: the input prompt is %d tokens\n", __func__, (int)tokens.size());

    // Test 1: baseline (saves state to disk)
    auto result_baseline = test_baseline(model, params, tokens);
    if (result_baseline.empty()) {
        return 1;
    }

    if (!test_tail_state_contract(model, params, tokens)) {
        return 1;
    }
    if (!test_cross_ubatch_tail_state(model, params, tokens, 128, 512) ||
            !test_cross_ubatch_tail_state(model, params, tokens, 512, 128)) {
        return 1;
    }
    if (!test_kvarn_full_window_native_exact(model, params, tokens)) {
        return 1;
    }

    // Test 2: state load
    if (!test_state_load(model, params, tokens, result_baseline)) {
        return 1;
    }

    // Test 3: seq copy (host)
    if (!test_seq_cp_host(model, params, tokens, result_baseline)) {
        return 1;
    }

    // Test 4: seq copy (device)
    if (!test_seq_cp_device(model, params, tokens, result_baseline)) {
        return 1;
    }

    LOG("\nAll tests passed.\n");

    return 0;
}
