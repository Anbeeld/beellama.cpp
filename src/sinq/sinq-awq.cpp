// AWQ adaptation attribution retained in ../llama-sinq-calibration.cpp.
#include "sinq-awq.h"
#include "sinq-numeric.h"
#include <algorithm>
#include <cmath>
#include <limits>
namespace sinq {
std::vector<double> legacy_awq_factors(const std::vector<double> & mx,const std::vector<double> & mw,int index) {
    require(!mx.empty() && mx.size()==mw.size() && index>=0 && index<20,"A-SINQ invalid AWQ statistics or alpha index");
    const double alpha=double(index)/20.;std::vector<double> a(mx.size());double lo=std::numeric_limits<double>::infinity(),hi=0;
    for (size_t j=0;j<a.size();++j) {
        require(std::isfinite(mx[j]) && mx[j]>=0 && std::isfinite(mw[j]) && mw[j]>=0,"A-SINQ invalid channel mean");
        // awq.py: beta=0, use_weightscale=True, 20 alphas in [0,1).
        a[j]=std::max(1.e-4,std::pow(mx[j],alpha));
        a[j]=std::max(1.e-4,a[j]/(std::pow(mw[j],1.-alpha)+1.e-4));
        lo=std::min(lo,a[j]);hi=std::max(hi,a[j]);
    }
    const double normalization=std::sqrt(lo*hi);require(normalization>0 && std::isfinite(normalization),"A-SINQ invalid AWQ normalization");
    for (double & v:a) { v/=normalization;require(v>0 && std::isfinite(v),"A-SINQ nonpositive AWQ factor"); }
    return a;
}

}

namespace sinq {
std::vector<ggml_fp16_t> compensation(const std::vector<float> & raw,
        const std::vector<double> & mx,const std::vector<double> & mw,const candidate_spec & spec) {
    require(!raw.empty() && std::isfinite(spec.alpha) && spec.alpha>=0 && spec.alpha<=1 &&
        std::isfinite(spec.lambda) && spec.lambda>=0 && spec.lambda<=1,"research invalid alpha/lambda");
    require(spec.family==scale_family::ordinary || spec.family==scale_family::weight_activation ||
        spec.family==scale_family::activation_only,"research invalid scale family");
    std::vector<double> factors(raw.size(),1.);
    // Endpoints take the ordinary path, not a pow()/division approximation.
    const bool corrected=spec.family!=scale_family::ordinary && spec.lambda!=0 &&
        !(spec.family==scale_family::activation_only && spec.alpha==0);
    if(corrected) {
        require(mx.size()==raw.size() && (spec.family==scale_family::activation_only || mw.size()==raw.size()),"research channel statistics mismatch");
        double lo=std::numeric_limits<double>::infinity(),hi=0;
        for(size_t j=0;j<raw.size();++j) {
            require(std::isfinite(mx[j]) && mx[j]>=0,"research invalid activation mean");
            factors[j]=std::max(1.e-4,std::pow(mx[j],spec.alpha));
            if(spec.family==scale_family::weight_activation) {
                require(std::isfinite(mw[j]) && mw[j]>=0,"research invalid weight mean");
                factors[j]=std::max(1.e-4,factors[j]/(std::pow(mw[j],1.-spec.alpha)+1.e-4));
            }
            lo=std::min(lo,factors[j]);hi=std::max(hi,factors[j]);
        }
        const double normalization=std::sqrt(lo*hi);
        require(normalization>0 && std::isfinite(normalization),"research invalid geometric normalization");
        for(auto & a:factors) { a/=normalization;require(a>0 && std::isfinite(a),"research invalid normalized factor"); }
    }
    std::vector<ggml_fp16_t> u(raw.size());
    for(size_t j=0;j<raw.size();++j) {
        require(raw[j]>0 && std::isfinite(raw[j]),"research invalid Sinkhorn factor");
        // Avoid pow(a,1) to preserve the precise legacy arithmetic path.
        const double a=spec.lambda==1 ? factors[j] : std::pow(factors[j],spec.lambda);
        u[j]=ggml_fp32_to_fp16(corrected ? float(double(raw[j])/a) : raw[j]);
        const float v=ggml_fp16_to_fp32(u[j]);
        require(v>0 && std::isfinite(v),"research compensation underflow/overflow in F16");
    }
    return u;
}
}
