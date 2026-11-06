#pragma once
#include "ggml.h"
#include <functional>
#include <string>
#include <vector>
namespace sinq {
constexpr int schema_version=2;
enum class generator_kind { sinq, awq };
enum class correction_kind { weight_activation, activation_only };
enum class local_objective { output_l1, output_mse };
struct candidate_spec {
    generator_kind generator=generator_kind::sinq;
    int strip_width=64;
    int iterations=16;
    double strength=1;
    correction_kind correction=correction_kind::weight_activation;
    double alpha=0;
    double lambda=0;
};
using weight_reader=std::function<void(int64_t,int64_t,float*)>;
using native_encoder=std::function<size_t(const float*,void*,int64_t,int64_t,const float*)>;
struct candidate_result {
    candidate_spec spec;
    std::vector<ggml_fp16_t> scales;
    double train_score=0, validation_score=0;
    bool has_score=false, has_validation=false;
    bool encoded=false;
    size_t duplicate_of=0;
};
}
