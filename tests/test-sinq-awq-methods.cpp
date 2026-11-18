#include "sinq/sinq-awq.h"
#include "sinq/sinq-local-score.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <stdexcept>
static void check(bool v,const char*s){if(!v)throw std::runtime_error(s);}
static sinq::candidate_spec awq(sinq::correction_kind correction,double alpha,double lambda){sinq::candidate_spec s;s.generator=sinq::generator_kind::awq;s.correction=correction;s.alpha=alpha;s.lambda=lambda;return s;}
int main()try {
    auto*ctx=ggml_init({1024,nullptr,true});ggml_free(ctx);
    std::vector<float>raw={.75f,1.12f};const std::vector<double>mx={1,16},mw={4,1};
    const auto ordinary=sinq::compensation(raw,mx,mw,{});
    for(int i=0;i<20;++i){
        // Independent scalar reference for preserved weight+activation correction.
        const double alpha=double(i)/20.;std::vector<double>a(2);
        for(size_t j=0;j<2;++j)a[j]=std::max(1.e-4,std::max(1.e-4,std::pow(mx[j],alpha))/(std::pow(mw[j],1.-alpha)+1.e-4));
        const double norm=std::sqrt(std::min(a[0],a[1])*std::max(a[0],a[1]));for(double&v:a)v/=norm;
        auto u=sinq::compensation(raw,mx,mw,awq(sinq::correction_kind::weight_activation,alpha,1));
        for(size_t j=0;j<2;++j)check(u[j]==ggml_fp32_to_fp16(float(double(raw[j])/a[j])),"correction numeric oracle");
        check(sinq::compensation(raw,mx,mw,awq(sinq::correction_kind::weight_activation,alpha,0))==ordinary,"lambda zero exact ordinary");
    }
    check(sinq::compensation(raw,mx,mw,awq(sinq::correction_kind::activation_only,0,1))==ordinary,"activation alpha zero ordinary");
    auto a=sinq::compensation(raw,mx,mw,awq(sinq::correction_kind::activation_only,.5,.5));
    check(a==sinq::compensation(raw,mx,{100,300},awq(sinq::correction_kind::activation_only,.5,.5))&&a!=ordinary,"activation only ignores weights");
    const float w[]={0,0},dq[]={1,1},x[]={1,1,-1,-1};const ggml_fp16_t u[]={ggml_fp32_to_fp16(1),ggml_fp32_to_fp16(1)};
    check(sinq::output_score(w,dq,u,1,2,x,2,sinq::local_objective::output_l1)==2,"analytic L1");
    check(sinq::output_score(w,dq,u,1,2,x,2,sinq::local_objective::output_mse)==4,"analytic MSE");
    for(double bad:{-1.,1.1,std::numeric_limits<double>::infinity()}){bool failed=false;try{sinq::compensation(raw,mx,mw,awq(sinq::correction_kind::activation_only,bad,1));}catch(...){failed=true;}check(failed,"bad alpha");}
    for(double bad:{-1.,1.1,std::numeric_limits<double>::quiet_NaN()}){bool failed=false;try{sinq::compensation(raw,mx,mw,awq(sinq::correction_kind::weight_activation,.5,bad));}catch(...){failed=true;}check(failed,"bad lambda");}
    bool failed=false;try{sinq::compensation({std::numeric_limits<float>::max()},{1},{1},{});}catch(...){failed=true;}check(failed,"F16 overflow");
    puts("test-sinq-awq-methods: PASS");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}
