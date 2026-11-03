// Activation-aware SINQ adaptation from huawei-csl/SINQ at
// 42ca83edf8fe1afc48c40fca6acc973c5d2bb59f, Apache-2.0.
// Native GGML encoding replaces the reference uniform row quantizer.
// Frozen legacy compatibility facade. Research policy lives in private modules.
#include "llama-sinq-calibration.h"
#include "sinq/sinq-awq.h"
#include "sinq/sinq-local-score.h"
#include "sinq/sinq-search.h"
std::vector<double> llama_sinq_awq_factors(const std::vector<double> & x, const std::vector<double> & w, int i) {
    return sinq::legacy_awq_factors(x,w,i);
}
double llama_sinq_output_l1(const float * w,const float * dq,const ggml_fp16_t * u,int64_t n,int64_t k,const float * x,int samples) {
    return sinq::output_l1(w,dq,u,n,k,x,samples);
}
llama_sinq_calibration_result llama_sinq_calibrate_native(ggml_type codec,int64_t n,int64_t k,
        const std::vector<float> & raw,const std::vector<ggml_fp16_t> & ordinary,const std::vector<float> & x,
        int samples,const float * importance,size_t limit,const llama_sinq_weight_reader & read,const llama_sinq_native_encoder & encode,int nthread) {
    return sinq::legacy_search(codec,n,k,raw,ordinary,x,samples,importance,limit,read,encode,nthread);
}
