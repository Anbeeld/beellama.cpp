#pragma once
#include "../llama-sinq-calibration.h"
#include "sinq-types.h"
#include <algorithm>
#include <thread>
namespace sinq {
struct search_options {
    int threads=int(std::max(1u, std::thread::hardware_concurrency()));
    std::vector<candidate_spec> candidates;
    local_objective objective=local_objective::output_l1;
    bool held_out=false;
    double tolerance=0;
    size_t max_candidates=128;
    size_t max_shortlist=0; // zero: small grid; nonzero: refine top coarse hypotheses
    double refine_step=0.05;
};
struct skipped_candidate { candidate_spec spec; std::string reason; };
struct search_result { std::vector<candidate_result> candidates; std::vector<skipped_candidate> skipped; size_t selected=0; size_t encode_count=0; size_t workspace_bytes=0; };
// Immutable packed slabs are emitted in candidate/row order; all duplicate
// candidates refer to their first encoded representative in the result.
using candidate_writer=std::function<void(size_t,const std::vector<ggml_fp16_t>&,int64_t,int64_t,const uint8_t*,size_t)>;
search_result research_search(ggml_type, int64_t, int64_t, const std::vector<float> &,
    const std::vector<float> &, int, const std::vector<float> &, int, const float *, size_t,
    const llama_sinq_weight_reader &, const llama_sinq_native_encoder &, const candidate_writer &, const search_options &);
llama_sinq_calibration_result legacy_search(ggml_type, int64_t, int64_t,
    const std::vector<float> &, const std::vector<ggml_fp16_t> &, const std::vector<float> &, int,
    const float *, size_t, const llama_sinq_weight_reader &, const llama_sinq_native_encoder &, int);
}
