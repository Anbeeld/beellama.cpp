#pragma once
#include "ggml.h"
#include <string>
#include <vector>
namespace sinq {
constexpr int schema_version=1;
enum class scale_family { ordinary, weight_activation, activation_only };
enum class local_objective { output_l1, output_mse };
struct candidate_spec {
    scale_family family=scale_family::ordinary;
    double alpha=0;
    double lambda=0;
};
struct candidate_result {
    candidate_spec spec;
    std::vector<ggml_fp16_t> scales;
    double train_score=0, validation_score=0;
    bool has_validation=false;
    bool encoded=false;
    size_t duplicate_of=0;
};
}
