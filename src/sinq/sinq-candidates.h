#pragma once
#include "../llama-sinq.h"
#include "sinq-types.h"
#include <algorithm>
#include <thread>
namespace sinq {
struct search_options {
    int threads=int(std::max(1u,std::thread::hardware_concurrency()));
    std::vector<candidate_spec> candidates;
    local_objective objective=local_objective::output_l1;
    bool held_out=false;
    size_t max_candidates=128;
};
struct search_result { std::vector<candidate_result> candidates; size_t encode_count=0; size_t workspace_bytes=0; };
using candidate_writer=std::function<void(size_t,const std::vector<ggml_fp16_t>&,int64_t,int64_t,const uint8_t*,size_t)>;
// Normalization uses the canonical native implementation, one complete spec at a time.
using scale_computer=std::function<std::vector<float>(const candidate_spec&)>;
search_result materialize_candidates(ggml_type,int64_t,int64_t,
    const std::vector<float>&,int,const std::vector<float>&,int,const float*,size_t,
    const weight_reader&,const native_encoder&,const candidate_writer&,const scale_computer&,const search_options&);
}
