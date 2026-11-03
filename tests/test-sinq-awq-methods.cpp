#include "sinq/sinq-awq.h"
#include "sinq/sinq-local-score.h"
#include <cmath>
#include <limits>
#include <cstdio>
#include <stdexcept>
static void check(bool v,const char * s) { if(!v) throw std::runtime_error(s); }
int main() try {
    auto * ctx=ggml_init({1024,nullptr,true});ggml_free(ctx);
    std::vector<float> raw={.75f,1.12f};
    const std::vector<double> mx={1,16}, mw={4,1};
    const auto ordinary=sinq::compensation(raw,mx,mw,{sinq::scale_family::ordinary,0,0});
    for(int i=0;i<20;++i) {
        const auto a=sinq::legacy_awq_factors(mx,mw,i);
        const auto u=sinq::compensation(raw,mx,mw,{sinq::scale_family::weight_activation,double(i)/20,1});
        for(size_t j=0;j<raw.size();++j) check(u[j]==ggml_fp32_to_fp16(float(double(raw[j])/a[j])),"lambda=1 legacy rounding");
        check(sinq::compensation(raw,mx,mw,{sinq::scale_family::weight_activation,double(i)/20,0})==ordinary,"lambda=0 exact ordinary");
    }
    check(sinq::compensation(raw,mx,mw,{sinq::scale_family::activation_only,0,1})==ordinary,"activation alpha=0 ordinary");
    const auto a=sinq::compensation(raw,mx,mw,{sinq::scale_family::activation_only,.5,.5});
    const auto b=sinq::compensation(raw,mx,{100,300},{sinq::scale_family::activation_only,.5,.5});
    check(a==b && a!=ordinary,"activation-only must ignore weight mean");
    for(auto h:a)check(std::isfinite(ggml_fp16_to_fp32(h)) && ggml_fp16_to_fp32(h)>0,"positive finite F16");
    const float w[]={0,0},dq[]={1,1},x[]={1,1,-1,-1};
    const ggml_fp16_t u[]={ggml_fp32_to_fp16(1),ggml_fp32_to_fp16(1)};
    check(sinq::output_score(w,dq,u,1,2,x,2,sinq::local_objective::output_l1)==2,"analytic L1");
    check(sinq::output_score(w,dq,u,1,2,x,2,sinq::local_objective::output_mse)==4,"analytic MSE");
    for(double alpha:{-1.,1.1,std::numeric_limits<double>::infinity()}) {
        bool failed=false;try{sinq::compensation(raw,mx,mw,{sinq::scale_family::activation_only,alpha,1});}catch(...){failed=true;}check(failed,"invalid alpha rejected");
    }
    for(double lambda:{-1.,1.1,std::numeric_limits<double>::quiet_NaN()}) {
        bool failed=false;try{sinq::compensation(raw,mx,mw,{sinq::scale_family::weight_activation,.5,lambda});}catch(...){failed=true;}check(failed,"invalid lambda rejected");
    }
    bool failed=false;try{sinq::compensation({std::numeric_limits<float>::max()}, {1},{1},{sinq::scale_family::ordinary,0,0});}catch(...){failed=true;}check(failed,"F16 overflow rejected");
    puts("test-sinq-awq-methods: PASS");return 0;
} catch(const std::exception &e){fprintf(stderr,"%s\n",e.what());return 1;}
