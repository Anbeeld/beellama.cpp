#pragma once
#include "sinq-search.h"
#include "../llama-sinq.h"
#include "../../vendor/nlohmann/json.hpp"
namespace sinq {
// Offline bridge request: no public llama API fields and no runtime metadata.
struct bridge_request {
    std::string operation, weights, activations, validation, importance, payload, scales, output;
    ggml_type codec=GGML_TYPE_COUNT, source_type=GGML_TYPE_F32;
    int64_t rows=0, columns=0, first_row=0;
    int samples=0, validation_samples=0, threads=24;
    size_t workspace=64*1024*1024;
    size_t max_output_bytes=64*1024*1024;
    llama_sinq_options normalization;
    search_options search;
};
bridge_request parse_bridge_request(const nlohmann::json &);
}
