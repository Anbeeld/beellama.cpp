// Activation-aware SINQ adaptation from huawei-csl/SINQ at
// 42ca83edf8fe1afc48c40fca6acc973c5d2bb59f, Apache-2.0.
// Native GGML encoding replaces the reference uniform row quantizer.
#include "llama-sinq-calibration.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
void require(bool ok, const char * message) { if (!ok) { throw std::runtime_error(message); } }
struct sum {
    double value=0, correction=0;
    void add(double x) { const double y=x-correction,next=value+y;correction=(next-value)-y;value=next; }
};
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
}

std::vector<double> llama_sinq_awq_factors(const std::vector<double> & mx,const std::vector<double> & mw,int index) {
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

double llama_sinq_output_l1(const float * w,const float * dq,const ggml_fp16_t * u,int64_t n,int64_t k,const float * x,int samples) {
    require(w && dq && u && x && n>0 && k>0 && samples>0,"A-SINQ invalid output score dimensions");
    std::vector<double> error(size_t(k),0.);sum result;
    for (int64_t r=0;r<n;++r)result.add(row_loss(w+r*k,dq+r*k,u,k,x,samples,error));
    return result.value/double(n);
}

llama_sinq_calibration_result llama_sinq_calibrate_native(ggml_type codec,int64_t n,int64_t k,
        const std::vector<float> & raw,const std::vector<ggml_fp16_t> & ordinary,const std::vector<float> & x,
        int samples,const float * importance,size_t limit,const llama_sinq_weight_reader & read,const llama_sinq_native_encoder & encode) {
    require(llama_sinq_supported_type(codec) && n>=2 && k>0 && k%ggml_blck_size(codec)==0 && samples>0 && samples<=256 && read && encode,"A-SINQ unsupported native codec, dimensions or sample count");
    require(uint64_t(k)<=std::numeric_limits<size_t>::max()/256/4 && uint64_t(n)<=std::numeric_limits<size_t>::max()/uint64_t(k)/4 && raw.size()==size_t(k) && ordinary.size()==size_t(k) && x.size()==size_t(k)*samples,"A-SINQ dimension/activation cardinality mismatch");
    for (float v:x)require(std::isfinite(v),"A-SINQ nonfinite activation");
    for (int64_t j=0;j<k;++j) {
        require(raw[j]>0 && std::isfinite(raw[j]) && ggml_fp16_to_fp32(ordinary[j])>0 && std::isfinite(ggml_fp16_to_fp32(ordinary[j])) && ordinary[j]==ggml_fp32_to_fp16(raw[j]),"A-SINQ baseline scale/rounding mismatch");
        if (importance)require(importance[j]>=0 && std::isfinite(importance[j]),"A-SINQ invalid importance");
    }
    require(importance || !ggml_quantize_requires_imatrix(codec),"A-SINQ native codec requires the final quantizer's imatrix");
    const size_t row_bytes=ggml_row_size(codec,k),fixed=size_t(k)*(64+size_t(samples)*4)+4096;
    const size_t per_row=size_t(k)*12+row_bytes+sizeof(double);
    require(limit>fixed && limit-fixed>=per_row,"A-SINQ workspace limit cannot fit one native row");
    const int64_t slab=std::min<int64_t>(n,(limit-fixed)/per_row);
    const size_t cells=size_t(slab)*k;
    std::vector<float> w(cells),pre(cells),dq(cells),transformed_importance(importance ? size_t(k) : 0);
    std::vector<uint8_t> packed(size_t(slab)*row_bytes);
    std::vector<double> mx(size_t(k),0.),mw(size_t(k),0.),error(size_t(k),0.);
    for (int64_t j=0;j<k;++j) { sum v;for(int s=0;s<samples;++s)v.add(std::abs(double(x[size_t(s)*k+j])));mx[j]=v.value/samples; }
    // Column means use a stable row order independent of slab boundaries.
    std::vector<sum> weight_sums(size_t(k),sum{});
    for (int64_t first=0;first<n;first+=slab) {
        const int64_t rows=std::min(slab,n-first);read(first,rows,w.data());
        for (int64_t r=0;r<rows;++r)for(int64_t j=0;j<k;++j) {
            const float v=w[r*k+j];require(std::isfinite(v),"A-SINQ nonfinite original weight");weight_sums[j].add(std::abs(double(v)));
        }
    }
    for (int64_t j=0;j<k;++j)mw[j]=weight_sums[j].value/double(n);
    llama_sinq_calibration_result result;result.scales=ordinary;result.workspace_bytes=fixed+size_t(slab)*per_row;
    for (int candidate=-1;candidate<20;++candidate) {
        std::vector<ggml_fp16_t> u=ordinary;
        if (candidate>=0) {
            const auto a=llama_sinq_awq_factors(mx,mw,candidate);
            for(int64_t j=0;j<k;++j) {
                // dual_shift.py matrix*=a; mu1/=a. Round exactly once.
                u[j]=ggml_fp32_to_fp16(float(double(raw[j])/a[j]));
                const float v=ggml_fp16_to_fp32(u[j]);require(v>0 && std::isfinite(v),"A-SINQ candidate underflowed/overflowed its F16 companion");
            }
        }
        if (importance)for(int64_t j=0;j<k;++j) {
            const float v=ggml_fp16_to_fp32(u[j]);transformed_importance[j]=importance[j]*v*v;
            require(std::isfinite(transformed_importance[j]) && transformed_importance[j]>=0,"A-SINQ transformed imatrix overflow");
        }
        sum loss;
        for(int64_t first=0;first<n;first+=slab) {
            const int64_t rows=std::min(slab,n-first);read(first,rows,w.data());
            for(int64_t i=0;i<rows*k;++i)pre[i]=w[i]/ggml_fp16_to_fp32(u[i%k]);
            const size_t bytes=encode(pre.data(),packed.data(),first,rows,importance ? transformed_importance.data() : nullptr);
            require(bytes==size_t(rows)*row_bytes && ggml_validate_row_data(codec,packed.data(),bytes),"A-SINQ native encoder output mismatch");
            for(int64_t r=0;r<rows;++r)ggml_get_type_traits(codec)->to_float(packed.data()+r*row_bytes,dq.data()+r*k,k);
            for(int64_t r=0;r<rows;++r)loss.add(row_loss(w.data()+r*k,dq.data()+r*k,u.data(),k,x.data(),samples,error));
        }
        const double score=loss.value/double(n);require(std::isfinite(score),"A-SINQ nonfinite layer-output L1");result.candidate_losses.push_back(score);
        if(candidate<0) { result.baseline_loss=result.selected_loss=score; }
        else {
            const double tolerance=64*std::numeric_limits<double>::epsilon()*std::max({1.,score,result.selected_loss});
            if(score<result.selected_loss-tolerance) { result.selected_loss=score;result.alpha_index=candidate;result.scales=u; }
        }
    }
    return result;
}
