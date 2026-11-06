#include "sinq/sinq-candidates.h"
#include "sinq/sinq-native-codec.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <cmath>
static void check(bool x,const char*s){if(!x)throw std::runtime_error(s);}
int main()try {
    auto*ctx=ggml_init({1024,nullptr,true});ggml_free(ctx);
    {
        const ggml_fp16_t u[] = {ggml_fp32_to_fp16(.5f), ggml_fp32_to_fp16(2.f)};
        const float weights[] = {1.f, 4.f, -2.f, 8.f}, importance[] = {3.f, 5.f};
        float conditioned[4], transformed[2];
        sinq::precondition_weights(weights, conditioned, 2, 2, u, 2);
        sinq::transform_importance(importance, transformed, 2, u);
        check(conditioned[0]==2.f && conditioned[1]==2.f && conditioned[2]==-4.f && conditioned[3]==4.f, "serialized weight transform");
        check(transformed[0]==.75f && transformed[1]==20.f, "serialized importance transform");
        const float invalid[] = {-1.f, 0.f};
        bool rejected=false;try{sinq::transform_importance(invalid, transformed, 2, u);}catch(...){rejected=true;}
        check(rejected, "negative importance rejected");
    }
    const int n=3,k=512,samples=3;
    std::vector<float>w(n*k),x(samples*k),v(samples*k),importance(k,1);
    for(int i=0;i<n*k;++i)w[i]=float(sin(i*.17));
    for(int i=0;i<samples*k;++i){x[i]=float(cos(i*.31));v[i]=float(sin(i*.71));}
    auto read=[&](int64_t first,int64_t rows,float*out){std::copy_n(w.data()+first*k,rows*k,out);};
    auto encode=[&](const float*in,void*out,int64_t,int64_t rows,const float*im){return ggml_quantize_chunk(GGML_TYPE_Q5_0,in,out,0,rows,k,im);};
    auto compute=[&](const sinq::candidate_spec&s){llama_sinq_options o;o.strip_width=s.strip_width;o.iterations=s.iterations;o.strength=float(s.strength);std::vector<ggml_fp16_t>u;std::vector<float>raw;llama_sinq_statistics stats;check(llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,o,1,u,stats,&raw),"canonical scales");return raw;};
    sinq::search_options opt;opt.threads=1;opt.max_candidates=128;
    for(int width:{32,64,128,256,512})for(int it:{4,8,16})for(double strength:{0.,.5,1.}){sinq::candidate_spec s;s.strip_width=width;s.iterations=it;s.strength=strength;opt.candidates.push_back(s);}
    std::vector<std::vector<uint8_t>>bytes(opt.candidates.size());
    auto emit=[&](size_t id,const std::vector<ggml_fp16_t>&,int64_t,int64_t rows,const uint8_t*p,size_t count){check(count==size_t(rows)*ggml_row_size(GGML_TYPE_Q5_0,k),"native bytes");bytes[id].insert(bytes[id].end(),p,p+count);};
    auto a=sinq::materialize_candidates(GGML_TYPE_Q5_0,n,k,{},0,{},0,importance.data(),4<<20,read,encode,emit,compute,opt);
    check(a.candidates.size()==45&&a.encode_count==45,"exact explicit parameter grid");
    for(size_t i=0;i<a.candidates.size();++i){check(!a.candidates[i].has_score,"ordinary without captures");const auto raw=compute(opt.candidates[i]);for(int j=0;j<k;++j)check(a.candidates[i].scales[j]==ggml_fp32_to_fp16(raw[j]),"native normalization oracle");}
    opt.candidates.resize(2);opt.candidates[1]=opt.candidates[0];
    auto b=sinq::materialize_candidates(GGML_TYPE_Q5_0,n,k,{},0,{},0,nullptr,4<<20,read,encode,emit,compute,opt);
    check(b.candidates.size()==2&&b.encode_count==2,"equal payload provenance preserved");
    opt.candidates[1].generator=sinq::generator_kind::awq;opt.candidates[1].alpha=.5;opt.candidates[1].lambda=1;
    for(auto objective:{sinq::local_objective::output_l1,sinq::local_objective::output_mse})for(bool held:{false,true}){
        opt.objective=objective;opt.held_out=held;opt.threads=1;
        auto serial=sinq::materialize_candidates(GGML_TYPE_Q5_0,n,k,x,samples,held?v:std::vector<float>{},held?samples:0,nullptr,4<<20,read,encode,emit,compute,opt);
        opt.threads=4;auto parallel=sinq::materialize_candidates(GGML_TYPE_Q5_0,n,k,x,samples,held?v:std::vector<float>{},held?samples:0,nullptr,4<<20,read,encode,emit,compute,opt);
        for(size_t i=0;i<serial.candidates.size();++i)check(serial.candidates[i].train_score==parallel.candidates[i].train_score&&serial.candidates[i].validation_score==parallel.candidates[i].validation_score,"thread independent scores");
    }
    bool failed=false;opt.held_out=false;try{sinq::materialize_candidates(GGML_TYPE_Q5_0,n,k,{},0,{},0,nullptr,4<<20,read,encode,emit,compute,opt);}catch(...){failed=true;}check(failed,"AWQ requires capture");
    ggml_quantize_free();puts("test-sinq-search: PASS");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}
