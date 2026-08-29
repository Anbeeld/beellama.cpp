#include "arg.h"
#include "common.h"
#include "download.h"
#include "gguf.h"
#include "llama.h"
#include "preset.h"
#include "speculative.h"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#undef NDEBUG
#include <cassert>

static std::string capture_stderr(const std::function<void()> & fn) {
    fflush(stderr);
    FILE * capture = tmpfile();
    assert(capture != nullptr);

#ifdef _WIN32
    const int stderr_fd = _fileno(stderr);
    const int saved_fd  = _dup(stderr_fd);
    assert(saved_fd >= 0);
    assert(_dup2(_fileno(capture), stderr_fd) == 0);
#else
    const int stderr_fd = fileno(stderr);
    const int saved_fd  = dup(stderr_fd);
    assert(saved_fd >= 0);
    assert(dup2(fileno(capture), stderr_fd) == 0);
#endif

    fn();
    fflush(stderr);
    assert(fseek(capture, 0, SEEK_SET) == 0);

    std::string result;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), capture) != nullptr) {
        result += buffer;
    }

#ifdef _WIN32
    assert(_dup2(saved_fd, stderr_fd) == 0);
    _close(saved_fd);
#else
    assert(dup2(saved_fd, stderr_fd) == 0);
    close(saved_fd);
#endif
    fclose(capture);
    return result;
}

static void set_test_env(const char * name, const char * value) {
#ifdef _WIN32
    assert(_putenv_s(name, value) == 0);
#else
    assert(setenv(name, value, true) == 0);
#endif
}

static void unset_test_env(const char * name) {
#ifdef _WIN32
    assert(_putenv_s(name, "") == 0);
#else
    assert(unsetenv(name) == 0);
#endif
}

static void test(void) {
    common_params params;

    auto assert_output_limits = [](int32_t n_batch, int32_t n_parallel, int32_t n_draft,
                                   int32_t total, int32_t per_seq) {
        const auto limits = common_speculative_get_output_limits(n_batch, n_parallel, n_draft);
        assert(limits.total == total);
        assert(limits.per_seq == per_seq);
    };

    assert_output_limits(16, 2,  3, 8, 4);
    assert_output_limits(16, 2, -1, 2, 1);
    assert_output_limits( 6, 2,  3, 6, 4);
    assert_output_limits( 2, 1,  3, 2, 2);
    assert_output_limits(
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max());

    {
        common_params_speculative spec;
        spec.synth_len = 3.4;

        auto assert_invalid = [](const common_params_speculative & value, int32_t n_max) {
            try {
                common_speculative_synth_rates_resolve(&value, n_max);
                assert(false);
            } catch (const std::invalid_argument &) {
            }
        };

        const auto rates = common_speculative_synth_rates_resolve(&spec, 4);
        assert(rates.size() == 4);
        assert(std::abs(rates[0] - 0.80581) < 1e-5);
        assert(std::abs(rates[1] - 0.64933) < 1e-5);
        assert(std::abs(rates[2] - 0.52323) < 1e-5);
        assert(std::abs(rates[3] - 0.42163) < 1e-5);
        assert(std::abs(1.0 + rates[0] + rates[1] + rates[2] + rates[3] - 3.4) < 1e-8);

        spec.synth_len = 1.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({0.0, 0.0, 0.0, 0.0}));

        spec.synth_len = 5.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({1.0, 1.0, 1.0, 1.0}));

        spec.synth_len = 5.1;
        assert_invalid(spec, 4);

        spec.synth_len = std::numeric_limits<double>::quiet_NaN();
        assert_invalid(spec, 4);

        spec.synth_len = 0.0;
        assert_invalid(spec, 4);

        spec.synth_len = -1.0;
        spec.synth_rates = {0.8, 0.6, 0.4};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        assert(common_speculative_synth_rates_resolve(&spec, 4) == spec.synth_rates);

        spec.synth_rates = {0.8, 0.9, 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, std::numeric_limits<double>::quiet_NaN(), 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, -0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        spec.synth_len = 3.0;
        assert_invalid(spec, 4);
    }

    {
        common_params base;
        base.n_parallel = 4;
        base.n_outputs_max_per_seq = 8;

        const auto draft = common_base_params_to_speculative(base);
        assert(draft.n_outputs_max == 4);
        assert(draft.n_outputs_max_per_seq == 1);
    }

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

    {
        const std::string canary = "hf_PRODUCTION_READINESS_SECRET_CANARY_123456789";
        std::vector<std::string> secret_argv = {"llama-server", "--hf-token", canary};
        common_preset_context preset_ctx(LLAMA_EXAMPLE_SERVER);
        common_preset preset = preset_ctx.load_from_args(
            (int) secret_argv.size(), list_str_to_char(secret_argv).data());

        const std::vector<std::string> rendered = preset.to_args("llama-server");
        assert(std::find(rendered.begin(), rendered.end(), canary) == rendered.end());
        assert(preset.to_ini().find(canary) == std::string::npos);
    }

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
    argv = {"binary_name", "--kv-tail-tokens", "auto,128"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    params = common_params();
    argv = {"binary_name", "--kv-tail-tokens", "full=128,full=256"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    params = common_params();
    argv = {"binary_name", "--kv-tail-type", "f32"};
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

    {
        common_params penalty_params;
        assert(penalty_params.sampling.penalty_last_n == 64);
        assert(penalty_params.sampling.dry_penalty_last_n == 64);

        argv = {"binary_name", "--repeat-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--dry-penalty-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "0"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "nan"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        const char * penalty_options[] = {"--frequency-penalty", "--presence-penalty"};
        const char * nonfinite_values[] = {"nan", "inf", "-inf"};
        for (const char * option : penalty_options) {
            for (const char * value : nonfinite_values) {
                argv = {"binary_name", option, value};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));
            }
        }
    }

    // Removed legacy speculative aliases, including --draft outside llama-speculative.
    params = common_params();
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

    argv = {"binary_name", "-lm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");
    assert(params.kv_tail_tokens == "0");
    assert(params.kv_tail_type == GGML_TYPE_COUNT);
    assert(common_context_params_to_llama(params).kv_tail_tokens == 0);
    assert(common_context_params_to_llama(params).kv_tail_type == GGML_TYPE_COUNT);

    const llama_context_params context_defaults = llama_context_default_params();
    assert(context_defaults.kv_tail_type == GGML_TYPE_COUNT);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
            "--kv-tail-tokens", "1024", "--kv-tail-type", "f16"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kvarn.type == LLAMA_KVARN_TYPE_DISABLED);
    assert(params.kv_tail_type == GGML_TYPE_F16);
    assert(common_context_params_to_llama(params).kv_tail_type == GGML_TYPE_F16);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-k", "kvarn4", "--cache-type-v", "kvarn4",
            "--kv-tail-tokens", "1024"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kvarn.type == LLAMA_KVARN_K4V4_G128);
    assert(params.kv_tail_type == GGML_TYPE_COUNT);
    assert(common_context_params_to_llama(params).kv_tail_type == GGML_TYPE_COUNT);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-k", "kvarn4", "--cache-type-v", "kvarn4",
            "--kv-tail-tokens", "1024", "--kv-tail-type", "bf16"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kvarn.type == LLAMA_KVARN_K4V4_G128);
    assert(params.kv_tail_type == GGML_TYPE_BF16);
    assert(common_context_params_to_llama(params).kv_tail_type == GGML_TYPE_BF16);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--kv-tail-tokens", "2048", "--kv-tail-type", "bf16"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kv_tail_tokens == "2048");
    assert(params.kv_tail_type == GGML_TYPE_BF16);
    assert(common_context_params_to_llama(params).kv_tail_tokens == 2048);
    assert(common_context_params_to_llama(params).kv_tail_type == GGML_TYPE_BF16);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--kv-tail-tokens", "auto"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kv_tail_tokens == "auto");

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--kv-tail-tokens", "128,512"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kv_tail_tokens == "128,512");

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--kv-tail-tokens", "full@l0=128,swa@l1=512"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kv_tail_tokens == "full@l0=128,swa@l1=512");

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--batch-layout", "round-robin", "--logits-out", "tail-logits.bin"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_BENCH));
    assert(params.batched_bench_batch_layout == "round-robin");
    assert(params.batched_bench_logits_out == "tail-logits.bin");

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--batch-layout", "invalid"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_BENCH));

    argv = {"binary_name", "-m", "model_file.gguf", "-t", "1234"};
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
    assert(params.cache_type_v == GGML_TYPE_Q2_0S);
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
    assert(params.cache_type_k == GGML_TYPE_Q2_0S);
    assert(params.cache_type_v == GGML_TYPE_Q2_0S);

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
    assert(params.cache_type_v == GGML_TYPE_Q2_0S);

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

    // Removed Turbo cache spellings redirect to the equivalent target KVarN types.
    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--cache-type-k", "turbo3", "--cache-type-v", "turbo4_tcq"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kvarn.key_bits == 3);
    assert(params.kvarn.value_bits == 4);
    assert(params.cache_kvarn_bits_k == 3);
    assert(params.cache_kvarn_bits_v == 4);
    assert(params.cache_type_k == GGML_TYPE_Q3_0);
    assert(params.cache_type_v == GGML_TYPE_Q4_0);

    // Draft contexts cannot use KVarN records, so the same aliases use matching q fallbacks.
    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--spec-draft-type-k", "turbo2_tcq", "--spec-draft-type-v", "turbo4"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.cache_type_k == GGML_TYPE_Q2_0S);
    assert(params.speculative.draft.cache_type_v == GGML_TYPE_Q4_0);

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

    unset_test_env("LLAMA_ARG_SPEC_DRAFT_N_MAX");
    params = common_params();
    argv = {"binary_name", "--spec-type", "draft-dflash"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.draft.n_max == 3);
    assert(!params.speculative.draft_n_max_explicit);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);
    assert(params.speculative.draft_n_max_explicit);

    set_test_env("LLAMA_ARG_SPEC_DRAFT_N_MAX", "7");
    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 7);
    assert(params.speculative.draft_n_max_explicit);
    unset_test_env("LLAMA_ARG_SPEC_DRAFT_N_MAX");

    const std::filesystem::path dflash_fixture =
            std::filesystem::temp_directory_path() / "beellama-dflash-depth-policy.gguf";
    gguf_context * fixture = gguf_init_empty();
    assert(fixture != nullptr);
    gguf_set_val_str(fixture, "general.architecture", "dflash");
    gguf_set_val_u32(fixture, "dflash.block_size", 16);
    assert(gguf_write_to_file(fixture, dflash_fixture.string().c_str(), true));
    gguf_free(fixture);

    params = common_params();
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
    assert(common_speculative_resolve_dflash_draft_n_max(params.speculative, dflash_fixture.string()));
    assert(params.speculative.draft.n_max == 15);

    params = common_params();
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
    params.speculative.draft.n_max = 20;
    params.speculative.draft_n_max_explicit = true;
    assert(common_speculative_resolve_dflash_draft_n_max(params.speculative, dflash_fixture.string()));
    assert(params.speculative.draft.n_max == 20);

    params = common_params();
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
    assert(common_speculative_resolve_dflash_draft_n_max(params.speculative, dflash_fixture.string()));
    assert(params.speculative.draft.n_max == 3);
    assert(std::filesystem::remove(dflash_fixture));

    params = common_params();
    argv = {"binary_name", "--spec-type", "dflash"};
    bool dflash_parsed = true;
    const std::string dflash_error = capture_stderr([&]() {
        dflash_parsed = common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER);
    });
    assert(false == dflash_parsed);
    assert(dflash_error.find("unknown speculative type: dflash") != std::string::npos);

    for (const std::string removed : {"copyspec", "suffix", "recycle"}) {
        params = common_params();
        argv = {"binary_name", "--spec-type", removed};
        bool parsed = true;
        const std::string error = capture_stderr([&]() {
            parsed = common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER);
        });
        assert(false == parsed);
        assert(error.find("speculative type '" + removed +
                          "' was removed in v0.4.0; use draft-dflash or upstream's ngram modes") != std::string::npos);
    }

    params = common_params();
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

    argv = {
        "binary_name",
        "--spec-dm-controller", "fringe",
    };
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    params = common_params();
    argv = {
        "binary_name",
        "--spec-type", "draft-dflash",
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

    params = common_params();
    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_len == 3.4);
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-rates", "0.8,0.6,0.2"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_rates == std::vector<double>({0.8, 0.6, 0.2}));
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4x"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
    }

    argv = {"binary_name", "-m", "model_file.gguf", "-lm", "none"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "-lm", "mmap"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "-lm", "mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "-lm", "mmap+mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "-lm", "dio"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

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

    setenv("LLAMA_ARG_LOAD_MODE", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_LOAD_MODE", "mmap", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    setenv("LLAMA_ARG_LOAD_MODE", "mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "mmap+mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "dio", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_LOAD_MODE", "none", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);
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
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    std::pair<long, std::vector<char>> good_url_res;
    try {
        good_url_res = common_remote_get_content(GOOD_URL, {});
    } catch (std::exception & e) {
        fprintf(stderr, "SKIP: could not fetch %s (%s)\n", GOOD_URL, e.what());
        printf("test-arg-parser: all tests OK\n\n");
        return;
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

static void test_single_device_draft_does_not_inherit_target_tensor_split() {
    common_params params;
    params.split_mode      = LLAMA_SPLIT_MODE_TENSOR;
    params.tensor_split[0] = 3.0f;
    params.tensor_split[1] = 1.0f;
    params.speculative.draft.mparams.path = "draft.gguf";
    params.speculative.draft.devices = {
        reinterpret_cast<ggml_backend_dev_t>(uintptr_t{1}),
        nullptr,
    };

    const common_params draft = common_base_params_to_speculative(params);

    assert(draft.split_mode == LLAMA_SPLIT_MODE_NONE);
    for (float value : draft.tensor_split) {
        assert(value == 0.0f);
    }
}

int main(void) {
    try {
        test();
        test_single_device_draft_does_not_inherit_target_tensor_split();
    } catch (std::exception & e) {
        fprintf(stderr, "test-arg-parser: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
