#include "sinq-local-score.h"
#include "sinq-numeric.h"
#include <cmath>
namespace sinq {
double row_loss(const float * w,const float * dq,const ggml_fp16_t * u,int64_t k,const float * x,int samples,std::vector<double> & error) {
    for (int64_t j=0;j<k;++j) {
        require(std::isfinite(w[j]) && std::isfinite(dq[j]),"A-SINQ nonfinite weight or native reconstruction");
        const double scale=ggml_fp16_to_fp32(u[j]);require(scale>0 && std::isfinite(scale),"A-SINQ invalid serialized scale");
        error[j]=double(dq[j])*scale-double(w[j]);
    }
    sum absolute;
    for (int s=0;s<samples;++s) {
        sum dot;
        for (int64_t j=0;j<k;++j) { require(std::isfinite(x[size_t(s)*k+j]),"A-SINQ nonfinite activation");dot.add(double(x[size_t(s)*k+j])*error[j]); }
        absolute.add(std::abs(dot.value));
    }
    return absolute.value/double(samples);
}
double output_l1(const float * w,const float * dq,const ggml_fp16_t * u,int64_t n,int64_t k,const float * x,int samples) {
    require(w && dq && u && x && n>0 && k>0 && samples>0,"A-SINQ invalid output score dimensions");
    std::vector<double> error(size_t(k),0.);sum result;
    for (int64_t r=0;r<n;++r)result.add(row_loss(w+r*k,dq+r*k,u,k,x,samples,error));
    return result.value/double(n);
}

}

namespace sinq {
double output_score(const float * w,const float * dq,const ggml_fp16_t * u,int64_t n,int64_t k,
        const float * x,int samples,local_objective objective) {
    if(objective==local_objective::output_l1) return output_l1(w,dq,u,n,k,x,samples);
    require(objective==local_objective::output_mse,"research invalid local objective");
    require(w && dq && u && x && n>0 && k>0 && samples>0,"research invalid score dimensions");
    sum result;
    for(int64_t r=0;r<n;++r) {
        sum row;
        for(int t=0;t<samples;++t) {
            sum dot;
            for(int64_t j=0;j<k;++j) {
                const double scale=ggml_fp16_to_fp32(u[j]);
                require(scale>0 && std::isfinite(scale) && std::isfinite(w[r*k+j]) &&
                    std::isfinite(dq[r*k+j]) && std::isfinite(x[size_t(t)*k+j]),"research nonfinite score input");
                dot.add(double(x[size_t(t)*k+j])*(double(dq[r*k+j])*scale-double(w[r*k+j])));
            }
            row.add(dot.value*dot.value);
        }
        result.add(row.value/double(samples));
    }
    const double value=result.value/double(n);require(std::isfinite(value),"research score overflow");return value;
}
}
