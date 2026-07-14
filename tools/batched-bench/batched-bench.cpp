#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -c 2048 -b 2048 -ub 512 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32 [-pps]\n", argv[0]);
    LOG("\n");
}

// satisfies -Wmissing-declarations
int llama_batched_bench(int argc, char ** argv);

int llama_batched_bench(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BENCH, print_usage)) {
        return 1;
    }

    int is_pp_shared   = params.is_pp_shared;
    int is_tg_separate = params.is_tg_separate;

    std::vector<int> n_pp = params.n_pp;
    std::vector<int> n_tg = params.n_tg;
    std::vector<int> n_pl = params.n_pl;

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model

    llama_model_params model_params = common_model_params_to_llama(params);

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);

    // ensure enough sequences are available
    ctx_params.n_seq_max = n_pl.empty() ? 1 : *std::max_element(n_pl.begin(), n_pl.end());

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    const uint64_t validation_seed = params.sampling.seed == LLAMA_DEFAULT_SEED ? 1 : params.sampling.seed;
    const auto get_token = [n_vocab, validation_seed](int32_t seq_id, int32_t pos) -> llama_token {
        uint64_t x = validation_seed ^ (uint64_t(uint32_t(seq_id)) << 32) ^ uint32_t(pos);
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return llama_token(x % uint64_t(n_vocab));
    };

    auto * mem = llama_get_memory(ctx);

    const int32_t n_kv_max = llama_n_ctx(ctx);

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

    std::unique_ptr<std::ofstream> logits_out;
    if (!params.batched_bench_logits_out.empty()) {
        logits_out = std::make_unique<std::ofstream>(
                params.batched_bench_logits_out, std::ios::binary | std::ios::trunc);
        if (!*logits_out) {
            LOG_ERR("failed to open logits output: %s\n", params.batched_bench_logits_out.c_str());
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        const char magic[8] = { 'S', 'T', 'T', 'A', 'I', 'L', 'L', 'G' };
        const uint32_t version = 1;
        const uint32_t vocab_size = uint32_t(n_vocab);
        logits_out->write(magic, sizeof(magic));
        logits_out->write(reinterpret_cast<const char *>(&version), sizeof(version));
        logits_out->write(reinterpret_cast<const char *>(&vocab_size), sizeof(vocab_size));
    }

    int32_t current_pp = -1;
    int32_t current_tg = -1;
    int32_t current_pl = -1;

    const auto write_logits = [&](llama_context * decode_ctx, const llama_batch & decoded) {
        if (!logits_out || current_pl < 0) {
            return;
        }
        for (int32_t i = 0; i < decoded.n_tokens; ++i) {
            if (!decoded.logits[i]) {
                continue;
            }
            const float * row = llama_get_logits_ith(decode_ctx, i);
            bool finite = row != nullptr;
            int32_t top = -1;
            float top_value = -std::numeric_limits<float>::infinity();
            for (int32_t token = 0; row && token < n_vocab; ++token) {
                finite = finite && std::isfinite(row[token]);
                if (row[token] > top_value) {
                    top_value = row[token];
                    top = token;
                }
            }
            const int32_t seq_id = decoded.n_seq_id[i] > 0 ? decoded.seq_id[i][0] : -1;
            const int32_t position = decoded.pos[i];
            const int32_t finite_i = finite ? 1 : 0;
            const int32_t vocab_i = n_vocab;
            for (const int32_t value : { current_pp, current_tg, current_pl, seq_id, position, finite_i, top, vocab_i }) {
                logits_out->write(reinterpret_cast<const char *>(&value), sizeof(value));
            }
            if (row) {
                logits_out->write(reinterpret_cast<const char *>(row), size_t(n_vocab)*sizeof(float));
            } else {
                std::vector<float> missing(size_t(n_vocab), std::numeric_limits<float>::quiet_NaN());
                logits_out->write(reinterpret_cast<const char *>(missing.data()), missing.size()*sizeof(float));
            }
        }
    };

    // decode in batches of ctx_params.n_batch tokens
    auto decode_helper = [&](llama_context * ctx, llama_batch & batch, int32_t n_batch, bool synchronize) {
        for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                LOG_ERR("failed to decode the batch, n_batch = %d, ret = %d\n", n_batch, ret);
                return false;
            }

            if (synchronize || logits_out) {
                llama_synchronize(ctx);
            }
            write_logits(ctx, batch_view);
        }

        return true;
    };

    // warm up
    {
        for (int i = 0; i < 16; ++i) {
            common_batch_add(batch, get_token(0, i), i, { 0 }, false);
        }

        if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }

    if (!params.batched_bench_output_jsonl) {
        LOG("\n");
        LOG("%s: n_kv_max = %d, n_batch = %d, n_ubatch = %d, flash_attn = %d, is_pp_shared = %d, is_tg_separate = %d, n_gpu_layers = %d, n_threads = %u, n_threads_batch = %u\n", __func__, n_kv_max, params.n_batch, params.n_ubatch, int(params.flash_attn_type), is_pp_shared, is_tg_separate, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch);
        LOG("\n");
        LOG("|%6s | %6s | %4s | %6s | %8s | %8s | %8s | %8s | %8s | %8s |\n", "PP", "TG", "B", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "T s", "S t/s");
        LOG("|%6s-|-%6s-|-%4s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|\n", "------", "------", "----", "------", "--------", "--------", "--------", "--------", "--------", "--------");
    }

    for (        int i_pp = 0; i_pp < (int) n_pp.size(); ++i_pp) {
        for (    int i_tg = 0; i_tg < (int) n_tg.size(); ++i_tg) {
            for (int i_pl = 0; i_pl < (int) n_pl.size(); ++i_pl) {
                const int pp = n_pp[i_pp];
                const int tg = n_tg[i_tg];
                const int pl = n_pl[i_pl];
                current_pp = pp;
                current_tg = tg;
                current_pl = pl;

                const int n_ctx_req = is_pp_shared ? (params.kv_unified ? pp : pl*pp) + pl*tg : pl*(pp + tg);

                if (n_ctx_req > n_kv_max) {
                    continue;
                }

                common_batch_clear(batch);

                const int prompt_sequences = is_pp_shared ? 1 : pl;
                const auto add_prompt_token = [&](int j, int i) {
                    common_batch_add(batch, get_token(j, i), i, { j }, i == pp - 1);
                };
                if (params.batched_bench_batch_layout == "seq-major") {
                    for (int j = 0; j < prompt_sequences; ++j) {
                        for (int i = 0; i < pp; ++i) {
                            add_prompt_token(j, i);
                        }
                    }
                } else {
                    for (int i = 0; i < pp; ++i) {
                        for (int j = 0; j < prompt_sequences; ++j) {
                            add_prompt_token(j, i);
                        }
                    }
                }

                llama_memory_clear(mem, false);

                const auto t_pp_start = ggml_time_us();

                if (!decode_helper(ctx, batch, ctx_params.n_batch, false)) {
                    LOG_ERR("%s: llama_decode() failed\n", __func__);
                    llama_free(ctx);
                    llama_model_free(model);
                    return 1;
                }

                llama_synchronize(ctx);

                const auto t_pp_end = ggml_time_us();

                if (is_pp_shared) {
                    for (int32_t i = 1; i < pl; ++i) {
                        llama_memory_seq_cp(mem, 0, i, -1, -1);
                    }

                    if (!params.kv_unified) {
                        // run one dummy token to apply the memory copy
                        common_batch_clear(batch);
                        common_batch_add(batch, get_token(0, pp), pp + 0, { 0 }, true);
                        if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
                            LOG_ERR("%s: llama_decode() failed\n", __func__);
                            llama_free(ctx);
                            llama_model_free(model);
                            return 1;
                        }
                        llama_memory_seq_rm(mem, 0, pp, -1);
                    }
                }

                const auto t_tg_start = ggml_time_us();

                if (is_tg_separate) {
                    // decode pattern:
                    // 0 0 0 ... 1 1 1 ... 2 2 2 ... 3 3 3 ...
                    for (int j = 0; j < pl; ++j) {
                        for (int i = 0; i < tg; ++i) {
                            common_batch_clear(batch);

                            common_batch_add(batch, get_token(j, pp + i), pp + i, { j }, true);

                            if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
                                LOG_ERR("%s: llama_decode() failed\n", __func__);
                                llama_free(ctx);
                                llama_model_free(model);
                                return 1;
                            }
                        }
                    }
                } else {
                    // decode pattern:
                    // 0123 0123 0123 ...
                    for (int i = 0; i < tg; ++i) {
                        common_batch_clear(batch);

                        for (int j = 0; j < pl; ++j) {
                            common_batch_add(batch, get_token(j, pp + i), pp + i, { j }, true);
                        }

                        if (!decode_helper(ctx, batch, ctx_params.n_batch, true)) {
                            LOG_ERR("%s: llama_decode() failed\n", __func__);
                            llama_free(ctx);
                            llama_model_free(model);
                            return 1;
                        }
                    }
                }

                const auto t_tg_end = ggml_time_us();

                const int32_t n_kv = n_ctx_req;

                const float t_pp = (t_pp_end - t_pp_start) / 1000000.0f;
                const float t_tg = (t_tg_end - t_tg_start) / 1000000.0f;
                const float t    = t_pp + t_tg;

                const float speed_pp = is_pp_shared ? pp / t_pp : pl*pp / t_pp;
                const float speed_tg = pl*tg / t_tg;
                const float speed    = ((is_pp_shared ? pp : pl*pp) + pl*tg) / t;

                if(params.batched_bench_output_jsonl) {
                    LOG(
                        "{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, \"is_pp_shared\": %d, \"n_gpu_layers\": %d, \"n_threads\": %u, \"n_threads_batch\": %u, "
                        "\"pp\": %d, \"tg\": %d, \"pl\": %d, \"n_kv\": %d, \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f, \"t\": %f, \"speed\": %f}\n",
                        n_kv_max, params.n_batch, params.n_ubatch, int(params.flash_attn_type), params.is_pp_shared, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch,
                        pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed
                    );
                } else {
                    LOG("|%6d | %6d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f |\n", pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed);
                }
            }
        }
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_batch_free(batch);

    llama_free(ctx);
    llama_model_free(model);

    llama_backend_free();

    return 0;
}
