#include "llama-sinq-calibration.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <limits>

static void check(bool ok, const char * message) { if (!ok) { throw std::runtime_error(message); } }
int main() try {
    auto * context=ggml_init({1024,nullptr,true});ggml_free(context);
    const auto factors=llama_sinq_awq_factors({1,16},{4,1},10);
    const double a=.5/(1.+.00005),b=4./1.0001,norm=std::sqrt(a*b);
    check(std::abs(factors[0]-a/norm)<1.e-12 && std::abs(factors[1]-b/norm)<1.e-12,"upstream AWQ factor formula");
    check(1/factors[0]>1 && 1/factors[1]<1,"AWQ divides compensation rather than multiplying it");
    const float original[]={0,0},decoded[]={1,1},plus[]={1,1,-1,-1},minus[]={1,-1,-1,1};
    const ggml_fp16_t one[]={ggml_fp32_to_fp16(1),ggml_fp32_to_fp16(1)};
    check(llama_sinq_output_l1(original,decoded,one,1,2,plus,2)==2,"correlated activation output loss");
    check(llama_sinq_output_l1(original,decoded,one,1,2,minus,2)==0,"same diagonal moments must not substitute for actual activations");
    for (ggml_type codec : {GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S,
         GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_NL,GGML_TYPE_IQ4_XS,GGML_TYPE_Q3_K,GGML_TYPE_Q4_K,
         GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q6_K,GGML_TYPE_Q8_0}) {
        constexpr int n=3,k=256,samples=4;
        std::vector<float> w(n*k),x(samples*k),importance(k),raw(k);
        std::vector<ggml_fp16_t> ordinary(k);
        for (int j=0;j<k;++j) { raw[j]=.75f+float(j%11)*.07f;ordinary[j]=ggml_fp32_to_fp16(raw[j]);importance[j]=1.f+float(j%7)*.2f; }
        for (int i=0;i<n*k;++i) { w[i]=float(std::sin(i*.37)*(.1+(i%17)*.03)); }
        for (int i=0;i<samples*k;++i) { x[i]=float(std::cos(i*.19)*(1+(i%13)*.3)); }
        int calls=0;
        auto read=[&](int64_t first,int64_t rows,float * out) { std::copy_n(w.data()+first*k,rows*k,out); };
        auto encode=[&](const float * values,void * out,int64_t,int64_t rows,const float * imatrix) {
            const size_t bytes=ggml_row_size(codec,k)*rows;
            const size_t actual=ggml_quantize_chunk(codec,values,out,0,rows,k,imatrix);
            std::vector<uint8_t> reference(bytes);
            check(ggml_quantize_chunk(codec,values,reference.data(),0,rows,k,imatrix)==bytes,"native reference encode size");
            check(actual==bytes && std::memcmp(out,reference.data(),bytes)==0,"candidate bytes differ from native codec");
            ++calls;return actual;
        };
        const auto result=llama_sinq_calibrate_native(codec,n,k,raw,ordinary,x,samples,importance.data(),1024*1024,read,encode);
        check(result.scales.size()==k && result.candidate_losses.size()==21 && calls>=21,"bounded complete candidate search");
        check(result.selected_loss<=result.baseline_loss,"calibration worse than uncalibrated candidate");
        std::vector<float> pre(n*k),imatrix(k),dq(n*k);
        for (int j=0;j<k;++j) { float u=ggml_fp16_to_fp32(result.scales[j]);imatrix[j]=importance[j]*u*u; }
        for (int i=0;i<n*k;++i)pre[i]=w[i]/ggml_fp16_to_fp32(result.scales[i%k]);
        std::vector<uint8_t> packed(ggml_row_size(codec,k)*n);
        ggml_quantize_chunk(codec,pre.data(),packed.data(),0,n,k,imatrix.data());
        ggml_get_type_traits(codec)->to_float(packed.data(),dq.data(),n*k);
        const double independent=llama_sinq_output_l1(w.data(),dq.data(),result.scales.data(),n,k,x.data(),samples);
        check(std::abs(independent-result.selected_loss)<1.e-12,"selected actual native reconstruction score");
        const auto again=llama_sinq_calibrate_native(codec,n,k,raw,ordinary,x,samples,importance.data(),32*1024,read,encode);
        check(again.scales==result.scales && again.alpha_index==result.alpha_index && again.selected_loss==result.selected_loss,"workspace-independent deterministic calibration");
        std::printf("native calibration %s parity PASS\n",ggml_type_name(codec));
    }
    {
        constexpr int n=2,k=256;
        std::vector<float> raw(k,1),x(2*k,1);
        std::vector<ggml_fp16_t> ordinary(k,ggml_fp32_to_fp16(1));
        auto read=[=](int64_t,int64_t rows,float * out) { std::fill_n(out,rows*k,0.f); };
        auto encode=[=](const float * in,void * out,int64_t,int64_t rows,const float * importance) {
            return ggml_quantize_chunk(GGML_TYPE_Q3_K,in,out,0,rows,k,importance);
        };
        auto run=[&]() { return llama_sinq_calibrate_native(GGML_TYPE_Q3_K,n,k,raw,ordinary,x,2,nullptr,64*1024,read,encode); };
        bool budget_rejected=false;
        try { llama_sinq_calibrate_native(GGML_TYPE_Q3_K,n,k,raw,ordinary,x,2,nullptr,64,read,encode); }
        catch(const std::exception &) { budget_rejected=true; }
        check(budget_rejected,"insufficient calibration workspace accepted");
        const auto tie=run();
        check(tie.alpha_index==-1 && tie.selected_loss==0 && tie.scales==ordinary,"identity native control/tie must retain uncalibrated candidate");
        const auto reject=[&](const char * message) { bool failed=false;try { run(); } catch(const std::exception &) { failed=true; }check(failed,message); };
        raw[0]=0;reject("zero raw compensation accepted");raw[0]=1;
        x[0]=std::numeric_limits<float>::quiet_NaN();reject("nonfinite samples accepted");
        x.assign(2*k,0);x[0]=x[k]=std::numeric_limits<float>::max();reject("underflowed F16 AWQ compensation accepted");
    }
    ggml_quantize_free();std::puts("test-sinq-calibration: PASS");return 0;
} catch (const std::exception & e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
