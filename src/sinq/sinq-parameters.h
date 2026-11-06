#pragma once

#include "llama-sinq.h"
#include "json.hpp"
#include <string>
#include <vector>

namespace sinq {
struct parameters {
    std::string generator = "sinq";
    int strip_width = 64;
    int iterations = 16;
    double strength = 1.0;
    std::string correction;
    double alpha = 0.0;
    double lambda = 0.0;
    llama_sinq_options normalization() const;
};
// All input JSON must enter through this duplicate-key/nonfinite rejecting parser.
nlohmann::json parse_json_strict(const std::string & text);
parameters parse_parameters(const nlohmann::json & value, bool complete = true);
nlohmann::json parameters_json(const parameters & value);
parameters validate_recipe(const std::string & canonical_json);
struct tensor_descriptor {
    std::string name;
    std::string role;
    int layer = -1;
    int64_t columns = 0;
    int64_t rows = 0;
    ggml_type qtype = GGML_TYPE_F32;
    bool eligible = false;
};
struct resolved_tensor {
    tensor_descriptor tensor;
    parameters settings;
    bool locked = true;
};
std::vector<resolved_tensor> resolve_direct(const nlohmann::json & config,
                                           const std::vector<tensor_descriptor> & tensors);
std::vector<resolved_tensor> resolve_direct_file(const std::string & path,
                                                const std::vector<tensor_descriptor> & tensors);
// Shared runtime/bridge writer validation. Names and recipes must be aligned.
std::vector<parameters> validate_metadata(const std::vector<std::string> & names,
                                         const std::vector<std::string> & recipes,
                                         const std::string & producer);
}
