#pragma once
#include "sinq-candidates.h"
#include "../../vendor/nlohmann/json.hpp"
namespace sinq {
struct bridge_request {
    std::string operation,weights,activations,validation,importance,payload,scales,output;
    ggml_type codec=GGML_TYPE_COUNT,source_type=GGML_TYPE_F32;
    int64_t rows=0,columns=0,first_row=0;
    int samples=0,validation_samples=0,threads=int(std::max(1u,std::thread::hardware_concurrency()));
    size_t workspace=64*1024*1024,max_output_bytes=64*1024*1024;
    search_options search;
};
nlohmann::json parse_bridge_json(std::istream&);
bridge_request parse_bridge_request(const nlohmann::json&);
candidate_spec parse_candidate_spec(const nlohmann::json&);
nlohmann::json candidate_spec_json(const candidate_spec&);
}
