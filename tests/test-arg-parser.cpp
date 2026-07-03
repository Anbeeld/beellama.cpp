#include "arg.h"
#include "common.h"
#include "download.h"

#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

int main(void) {
    common_params params;

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    params = common_params();
    argv = {"binary_name", "--kv-kvarn", "kvarn_k4v2_g128"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    params = common_params();
    argv = {"binary_name", "--cache-type-k", "kvarn7"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    params = common_params();
    argv = {"binary_name", "--spec-draft-type-k", "kvarn8"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-k", "kvarn4", "--cache-type-v", "kvarn2", "--kv-kvarn-sink-tokens", "256"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-k", "kvarn4", "--kv-kvarn-sinkhorn-iters", "12"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-k", "kvarn4", "--kv-kvarn-fallback"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-k", "kvarn4", "--grp-attn-n", "2"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMPLETION));

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-k", "kvarn4", "--kv-kvarn-pool-mem-frac", "0.15"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // removed legacy speculative aliases
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {"binary_name", "--draft-n", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {"binary_name", "--draft-max", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {"binary_name", "--draft-min", "1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {"binary_name", "--draft-n-min", "1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {"binary_name", "--tree-budget", "20"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {"binary_name", "--spec-dflash-default"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {"binary_name", "--dflash-max-slots", "1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {"binary_name", "--draft-topk", "4"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {"binary_name", "--draft-model", "draft.gguf"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {"binary_name", "--spec-replace", "TARGET", "DRAFT"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {"binary_name", "--spec-draft-replace", "TARGET", "DRAFT"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    // negated arg
    argv = {"binary_name", "--no-mmap"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));


    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    params = common_params();
    argv = {
        "binary_name", "-m", "model_file.gguf",
        "--cache-type-k", "kvarn4",
        "--cache-type-v", "kvarn2",
    };
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kvarn.type == LLAMA_KVARN_K4V2_G128);
    assert(params.kvarn.key_bits == 4);
    assert(params.kvarn.value_bits == 2);
    assert(params.kvarn.swa_key_bits == 0);
    assert(params.kvarn.swa_value_bits == 0);
    assert(params.kvarn.sink_tokens == 128);
    assert(params.kvarn.sinkhorn_iters == 16);
    assert(params.kvarn.fail_if_unsupported);
    assert(params.cache_kvarn_bits_k == 4);
    assert(params.cache_kvarn_bits_v == 2);
    assert(params.cache_type_k == GGML_TYPE_Q4_0);
    assert(params.cache_type_v == GGML_TYPE_Q2_0);
    assert(!params.kv_unified);
    assert(common_context_params_to_llama(params).kvarn.type == LLAMA_KVARN_K4V2_G128);
    assert(!common_context_params_to_llama(params).kv_unified);

    params = common_params();
    argv = {
        "binary_name", "-m", "model_file.gguf",
        "--cache-type-k", "kvarn4",
        "--cache-type-v", "kvarn4",
        "--cache-type-k-swa", "kvarn8",
        "--cache-type-v-swa", "kvarn6",
    };
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kvarn.type == LLAMA_KVARN_K4V4_G128);
    assert(params.kvarn.swa_key_bits == 8);
    assert(params.kvarn.swa_value_bits == 6);
    assert(common_context_params_to_llama(params).kvarn.swa_key_bits == 8);
    assert(common_context_params_to_llama(params).kvarn.swa_value_bits == 6);

    params = common_params();
    argv = {
        "binary_name", "-m", "model_file.gguf",
        "--cache-type-k", "kvarn4",
        "--cache-type-v", "kvarn4",
        "--cache-type-k-swa", "kvarn8",
    };
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--kv-unified", "--cache-type-k", "kvarn4"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.kvarn.type == LLAMA_KVARN_K4V4_G128);
    assert(params.kv_unified);
    assert(common_context_params_to_llama(params).kv_unified);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-k", "kvarn3"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kvarn.type == LLAMA_KVARN_K3V3_G128);
    assert(params.kvarn.key_bits == 3);
    assert(params.kvarn.value_bits == 3);
    assert(params.cache_kvarn_bits_k == 3);
    assert(params.cache_kvarn_bits_v == 3);
    assert(params.cache_type_k == GGML_TYPE_Q3_0);
    assert(params.cache_type_v == GGML_TYPE_Q3_0);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-v", "kvarn2"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kvarn.type == LLAMA_KVARN_K2V2_G128);
    assert(params.kvarn.key_bits == 2);
    assert(params.kvarn.value_bits == 2);
    assert(params.cache_kvarn_bits_k == 2);
    assert(params.cache_kvarn_bits_v == 2);
    assert(params.cache_type_k == GGML_TYPE_Q2_0);
    assert(params.cache_type_v == GGML_TYPE_Q2_0);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-k", "kvarn4", "--cache-type-v", "f16"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kvarn.type == LLAMA_KVARN_K4V4_G128);
    assert(params.kvarn.key_bits == 4);
    assert(params.kvarn.value_bits == 4);
    assert(params.cache_kvarn_bits_k == 4);
    assert(params.cache_kvarn_bits_v == 4);
    assert(params.cache_type_k == GGML_TYPE_Q4_0);
    assert(params.cache_type_v == GGML_TYPE_Q4_0);

    params = common_params();
    argv = {
        "binary_name", "-m", "model_file.gguf",
        "--cache-type-k", "kvarn5",
        "--cache-type-v", "kvarn2",
    };
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kvarn.type == LLAMA_KVARN_K5V2_G128);
    assert(params.kvarn.key_bits == 5);
    assert(params.kvarn.value_bits == 2);
    assert(params.cache_kvarn_bits_k == 5);
    assert(params.cache_kvarn_bits_v == 2);
    assert(params.cache_type_k == GGML_TYPE_Q5_0);
    assert(params.cache_type_v == GGML_TYPE_Q2_0);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-k", "kvarn6"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kvarn.type == LLAMA_KVARN_K6V6_G128);
    assert(params.kvarn.key_bits == 6);
    assert(params.kvarn.value_bits == 6);
    assert(params.cache_kvarn_bits_k == 6);
    assert(params.cache_kvarn_bits_v == 6);
    assert(params.cache_type_k == GGML_TYPE_Q6_0);
    assert(params.cache_type_v == GGML_TYPE_Q6_0);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-v", "kvarn8"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kvarn.type == LLAMA_KVARN_K8V8_G128);
    assert(params.kvarn.key_bits == 8);
    assert(params.kvarn.value_bits == 8);
    assert(params.cache_kvarn_bits_k == 8);
    assert(params.cache_kvarn_bits_v == 8);
    assert(params.cache_type_k == GGML_TYPE_Q8_0);
    assert(params.cache_type_v == GGML_TYPE_Q8_0);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-k", "q6_1", "--cache-type-v", "turbo4_tcq"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cache_type_k == GGML_TYPE_Q6_1);
    assert(params.cache_type_v == GGML_TYPE_TURBO4_TCQ);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--spec-draft-type-k", "q2_1", "--spec-draft-type-v", "q3_0"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.cache_type_k == GGML_TYPE_Q2_1);
    assert(params.speculative.draft.cache_type_v == GGML_TYPE_Q3_0);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);
    assert(params.speculative.n_max == 123);

    params = common_params();
    argv = {"binary_name", "--spec-type", "mtp"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.draft.n_max == 3);
    assert(params.speculative.n_max == 3);
    assert(params.n_batch == 2048);
    assert(params.n_ubatch == 512);

    params = common_params();
    argv = {"binary_name", "--spec-type", "dflash"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.n_max == 16);
    assert(params.speculative.draft.n_max == 16);
    assert(params.speculative.draft.n_ctx == 256);
    assert(params.n_batch == 2048);
    assert(params.n_ubatch == 512);

    params = common_params();
    argv = {"binary_name", "--spec-type", "dflash", "--spec-draft-n-max", "7"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.n_max == 7);
    assert(params.speculative.draft.n_max == 7);
    assert(params.speculative.draft.n_ctx == 256);

    params = common_params();
    argv = {"binary_name", "--spec-type", "dflash", "--spec-dflash-cross-ctx", "1024"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.dflash_cross_ctx == 1024);
    assert(params.speculative.draft.n_ctx == 256);

    params = common_params();
    argv = {"binary_name", "--spec-type", "dflash", "--spec-dflash-cross-ctx", "1024", "-cd", "1040"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.dflash_cross_ctx == 1024);
    assert(params.speculative.draft.n_ctx == 1040);

    params = common_params();
    const common_params_speculative defaults_speculative = params.speculative;
    argv = {"binary_name", "--spec-dm-min-reach", "6"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.dm_min_reach == defaults_speculative.dm_min_reach);

    params = common_params();
    argv = {"binary_name", "--spec-type", "mtp", "--spec-dm-min-reach", "6", "--spec-draft-temp", "0.5", "--spec-dflash-cross-ctx", "1024", "--spec-branch-budget", "4"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.draft.n_max == 3);
    assert(params.speculative.n_max == 3);
    assert(params.speculative.dm_min_reach == defaults_speculative.dm_min_reach);
    assert(params.speculative.sample_temp == defaults_speculative.sample_temp);
    assert(params.speculative.dflash_cross_ctx == defaults_speculative.dflash_cross_ctx);
    assert(params.speculative.branch_budget == defaults_speculative.branch_budget);

    params = common_params();
    argv = {"binary_name", "--spec-draft-model", "dflash-draft.gguf", "--spec-dflash-cross-ctx", "1024", "--spec-dm-min-reach", "6"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.type() == COMMON_SPECULATIVE_TYPE_NONE);
    assert(params.speculative.dflash_cross_ctx == 1024);
    assert(params.speculative.dm_min_reach == 6);
    assert(params.speculative.dflash_only_args_explicit);

    params = common_params();
    assert(params.speculative.draft.p_min == 0.0f);
    assert(params.speculative.p_min == 0.0f);
    assert(params.speculative.dm_profit_min == 0.05f);
    assert(params.speculative.dm_profit_raise_margin == 0.05f);
    assert(params.speculative.dm_profit_lower_margin == 0.05f);
    assert(params.speculative.dm_controller == COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT);
    assert(params.speculative.dm_profit_min_samples == 3);
    assert(params.speculative.dm_profit_warmup == 0);
    assert(params.speculative.dm_profit_baseline_interval == 1024);
    assert(!params.fit_params_target.empty());

    argv = {"binary_name", "-m", "model_file.gguf", "--fit-target", "256"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    for (size_t target : params.fit_params_target) {
        assert(target == 256ull * 1024ull * 1024ull);
    }

    argv = {"binary_name", "--spec-draft-p-min", "0"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.draft.p_min == 0.0f);
    assert(params.speculative.p_min == 0.0f);

    argv = {"binary_name", "--draft-p-min", "0.25"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.draft.p_min == 0.25f);
    assert(params.speculative.p_min == 0.25f);

    params = common_params();
    argv = {"binary_name", "--spec-type", "dflash", "--spec-draft-temp", "0.5"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.sample_temp == 0.5f);

    params = common_params();
    argv = {"binary_name", "--spec-type", "dflash", "--spec-draft-temp", "auto"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.sample_temp == -1.0f);

    argv = {
        "binary_name",
        "--spec-type", "dflash",
        "--spec-dm-controller", "fringe",
    };
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.dm_controller == COMMON_SPECULATIVE_DM_CONTROLLER_FRINGE);

    argv = {
        "binary_name",
        "--spec-type", "dflash",
        "--spec-dm-controller", "profit",
        "--spec-dm-profit-min", "0.03",
        "--spec-dm-profit-raise-margin", "0.06",
        "--spec-dm-profit-lower-margin", "0.02",
        "--spec-dm-profit-ewma-alpha", "0.15",
        "--spec-dm-profit-min-samples", "6",
        "--spec-dm-profit-warmup", "4",
        "--spec-dm-profit-baseline-interval", "256",
    };
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.dm_controller == COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT);
    assert(params.speculative.dm_profit_min == 0.03f);
    assert(params.speculative.dm_profit_raise_margin == 0.06f);
    assert(params.speculative.dm_profit_lower_margin == 0.02f);
    assert(params.speculative.dm_profit_ewma_alpha == 0.15f);
    assert(params.speculative.dm_profit_min_samples == 6);
    assert(params.speculative.dm_profit_warmup == 4);
    assert(params.speculative.dm_profit_baseline_interval == 256);

    argv = {"binary_name", "--spec-dm-controller", std::string("profit-") + "shadow"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {"binary_name", "--spec-dm-controller", "invalid"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {
        "binary_name",
        "--reasoning-loop-guard", "force-close",
        "--reasoning-loop-min-tokens", "1024",
        "--reasoning-loop-window", "2048",
        "--reasoning-loop-max-period", "512",
        "--reasoning-loop-min-coverage", "768",
        "--reasoning-loop-check-interval", "32",
        "--reasoning-loop-interventions", "1",
    };
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.reasoning_loop_guard.mode == COMMON_REASONING_LOOP_GUARD_FORCE_CLOSE);
    assert(params.reasoning_loop_guard.min_reasoning_tokens == 1024);
    assert(params.reasoning_loop_guard.window_tokens == 2048);
    assert(params.reasoning_loop_guard.max_period == 512);
    assert(params.reasoning_loop_guard.min_repeated_coverage == 768);
    assert(params.reasoning_loop_guard.check_interval == 32);
    assert(params.reasoning_loop_guard.interventions_max == 1);

    argv = {"binary_name", "--reasoning-loop-guard", "invalid"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    argv = {"binary_name", "--reasoning-loop-window", "64", "--reasoning-loop-min-coverage", "128"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    // multi-value args (CSV)
    params = common_params();
    params.model.path = "model_file.gguf";
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_MMAP", "0", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.use_mmap == false);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    printf("test-arg-parser: test download functions\n\n");
    common_download_model_result download_result_fields;
    download_result_fields.dflash_draft_path = "dflash-draft.gguf";
    assert(download_result_fields.dflash_draft_path == "dflash-draft.gguf");

    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    std::pair<long, std::vector<char>> good_url_res;
    try {
        good_url_res = common_remote_get_content(GOOD_URL, {});
    } catch (std::exception & e) {
        fprintf(stderr, "SKIP: could not fetch %s (%s)\n", GOOD_URL, e.what());
        printf("test-arg-parser: all tests OK\n\n");
        return 0;
    }

    {
        printf("test-arg-parser: test good URL\n\n");
        assert(good_url_res.first == 200);
        assert(good_url_res.second.size() > 0);
        std::string str(good_url_res.second.data(), good_url_res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}
