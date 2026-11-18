// Activation-aware SINQ adaptation from huawei-csl/SINQ at
// 42ca83edf8fe1afc48c40fca6acc973c5d2bb59f, Apache-2.0.
#include "sinq-awq.h"
#include "sinq-numeric.h"
#include <algorithm>
#include <cmath>
#include <limits>
namespace sinq {
std::vector<ggml_fp16_t> compensation(const std::vector<float>&raw,const std::vector<double>&mx,
        const std::vector<double>&mw,const candidate_spec&spec) {
    require(!raw.empty()&&std::isfinite(spec.alpha)&&spec.alpha>=0&&spec.alpha<=1&&std::isfinite(spec.lambda)&&spec.lambda>=0&&spec.lambda<=1,"invalid alpha/lambda");
    require(spec.generator==generator_kind::sinq||(spec.generator==generator_kind::awq&&(spec.correction==correction_kind::weight_activation||spec.correction==correction_kind::activation_only)),"invalid generator/correction");
    std::vector<double>factors(raw.size(),1.);
    const bool corrected=spec.generator==generator_kind::awq&&spec.lambda!=0&&!(spec.correction==correction_kind::activation_only&&spec.alpha==0);
    if(corrected) {
        require(mx.size()==raw.size()&&(spec.correction==correction_kind::activation_only||mw.size()==raw.size()),"channel statistics mismatch");
        double lo=std::numeric_limits<double>::infinity(),hi=0;
        for(size_t j=0;j<raw.size();++j) {
            require(std::isfinite(mx[j])&&mx[j]>=0,"invalid activation mean");
            factors[j]=std::max(1.e-4,std::pow(mx[j],spec.alpha));
            if(spec.correction==correction_kind::weight_activation) {
                require(std::isfinite(mw[j])&&mw[j]>=0,"invalid weight mean");
                factors[j]=std::max(1.e-4,factors[j]/(std::pow(mw[j],1.-spec.alpha)+1.e-4));
            }
            lo=std::min(lo,factors[j]);hi=std::max(hi,factors[j]);
        }
        const double normalization=std::sqrt(lo*hi);
        require(normalization>0&&std::isfinite(normalization),"invalid geometric normalization");
        for(auto&a:factors){a/=normalization;require(a>0&&std::isfinite(a),"invalid normalized factor");}
    }
    std::vector<ggml_fp16_t>u(raw.size());
    for(size_t j=0;j<raw.size();++j) {
        require(raw[j]>0&&std::isfinite(raw[j]),"invalid Sinkhorn factor");
        const double a=spec.lambda==1?factors[j]:std::pow(factors[j],spec.lambda);
        u[j]=ggml_fp32_to_fp16(corrected?float(double(raw[j])/a):raw[j]);
        const float v=ggml_fp16_to_fp32(u[j]);require(v>0&&std::isfinite(v),"compensation underflow/overflow in F16");
    }
    return u;
}
}
