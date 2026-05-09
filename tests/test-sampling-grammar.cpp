#include "common.h"
#include "json-schema-to-grammar.h"
#include "sampling.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <exception>
#include <memory>

static void test_output_format_thinking_prefill(const char * vocab_file) {
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;

    std::unique_ptr<llama_model, decltype(&llama_model_free)> model {
        llama_model_load_from_file(vocab_file, mparams),
        llama_model_free,
    };

    if (!model) {
        fprintf(stderr, "%s: failed to load vocab '%s'\n", __func__, vocab_file);
        GGML_ABORT("failed to load vocab");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const auto schema = nlohmann::ordered_json::parse(R"({
        "type": "object",
        "properties": {
            "answer": {
                "type": "string"
            }
        },
        "required": ["answer"]
    })");

    common_params_sampling params;
    params.grammar = {
        COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,
        json_schema_to_grammar(schema),
    };
    params.generation_prompt = "<think>\n";
    params.reasoning_budget_start = common_tokenize(vocab, "<think>\n", false, true);
    params.reasoning_budget_end = common_tokenize(vocab, "\n</think>\n", false, true);
    params.reasoning_budget_forced = params.reasoning_budget_end;
    params.reasoning_budget_tokens = -1;
    params.reasoning_budget_tracking = false;

    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler {
        common_sampler_init(model.get(), params),
        common_sampler_free,
    };

    if (!sampler) {
        GGML_ABORT("failed to initialize sampler");
    }

    GGML_ASSERT(common_sampler_get_reasoning_budget_state(sampler.get()) == REASONING_BUDGET_COUNTING);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <vocab-file>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    try {
        test_output_format_thinking_prefill(argv[1]);
    } catch (const std::exception & e) {
        fprintf(stderr, "unexpected exception: %s\n", e.what());
        llama_backend_free();
        return 1;
    }

    llama_backend_free();
    return 0;
}
