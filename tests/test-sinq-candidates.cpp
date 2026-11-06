#include "sinq/sinq-candidates.h"
#include "sinq/sinq-native-codec.h"
#include "sinq/sinq-awq.h"
#include "sinq/sinq-local-score.h"
#include "sinq/sinq-numeric.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <cmath>
#include <limits>
static void check(bool x,const char*s){if(!x)throw std::runtime_error(s);}

// Characterize the original all-statistics path while exercising grids that
// now omit one or both statistics. A one-row budget also makes validation of
// the final source slab observably precede every callback that can emit data.
static void test_effective_correction(const std::vector<float>&w,const std::vector<float>&x,
        const std::vector<float>&v,int n,int k,int samples) {
    sinq::candidate_spec ordinary,wa,activation;
    wa.generator=sinq::generator_kind::awq;wa.correction=sinq::correction_kind::weight_activation;wa.alpha=.5;wa.lambda=.75;
    activation=wa;activation.correction=sinq::correction_kind::activation_only;
    auto wa0=wa,activation0=activation,identity=activation,wa_alpha0=wa,wa_alpha1=wa;
    wa0.lambda=0;activation0.lambda=0;identity.alpha=0;wa_alpha0.alpha=0;wa_alpha1.alpha=1;
    const std::vector<std::vector<sinq::candidate_spec>>grids={
        {ordinary},{wa},{activation},{ordinary,activation,wa},
        {wa0},{activation0},{wa0,activation0},{identity},{wa_alpha0},{wa_alpha1}};
    std::vector<double>mx(k),mw(k);
    for(int j=0;j<k;++j){
        sinq::sum sx,sw;
        for(int t=0;t<samples;++t)sx.add(std::abs(double(x[size_t(t)*k+j])));
        for(int r=0;r<n;++r)sw.add(std::abs(double(w[size_t(r)*k+j])));
        mx[j]=sx.value/samples;mw[j]=sw.value/n;
    }
    auto raw=[&](const sinq::candidate_spec&s){
        std::vector<float>u(k);for(int j=0;j<k;++j)u[j]=float(.75+(j%13)*.03125+s.strength*.125);return u;
    };
    const size_t row_bytes=ggml_row_size(GGML_TYPE_Q5_0,k);
    for(const auto&grid:grids)for(auto objective:{sinq::local_objective::output_l1,sinq::local_objective::output_mse})for(bool held:{false,true}) {
        // Ordinary SINQ must work both with and without captured activations.
        for(int captured:{0,samples}) {
            if(!captured&&(held||grid[0].generator!=sinq::generator_kind::sinq||grid.size()!=1))continue;
            sinq::search_options opt;opt.candidates=grid;opt.threads=4;opt.objective=objective;opt.held_out=held;
            const size_t fixed=size_t(k)*(96+size_t(captured+(held?samples:0))*4+grid.size()*2)+grid.size()*256+4096;
            const size_t limit=fixed+size_t(k)*12+row_bytes+2*sizeof(double)+size_t(k)*sizeof(double);
            std::vector<std::vector<ggml_fp16_t>>scales;
            std::vector<std::vector<uint8_t>>expected;
            std::vector<double>train_scores,validation_scores;
            for(const auto&s:grid) {
                scales.push_back(sinq::compensation(raw(s),mx,mw,s));
                expected.emplace_back(size_t(n)*row_bytes);
                std::vector<float>pre(k),dq(k);sinq::sum train,val;
                for(int r=0;r<n;++r) {
                    sinq::precondition_weights(w.data()+size_t(r)*k,pre.data(),1,k,scales.back().data(),1);
                    auto*p=expected.back().data()+size_t(r)*row_bytes;
                    check(ggml_quantize_chunk(GGML_TYPE_Q5_0,pre.data(),p,0,1,k,nullptr)==row_bytes,"reference byte count");
                    if(captured) {
                        ggml_get_type_traits(GGML_TYPE_Q5_0)->to_float(p,dq.data(),k);
                        double score;
                        sinq::output_scores(w.data()+size_t(r)*k,dq.data(),scales.back().data(),1,k,x.data(),captured,objective,&score,1);train.add(score);
                        if(held){sinq::output_scores(w.data()+size_t(r)*k,dq.data(),scales.back().data(),1,k,v.data(),samples,objective,&score,1);val.add(score);}
                    }
                }
                train_scores.push_back(train.value/n);validation_scores.push_back(val.value/n);
            }
            size_t emissions=0;
            auto read=[&](int64_t first,int64_t rows,float*out){std::copy_n(w.data()+first*k,rows*k,out);};
            auto encode=[&](const float*in,void*out,int64_t,int64_t rows,const float*im){return ggml_quantize_chunk(GGML_TYPE_Q5_0,in,out,0,rows,k,im);};
            auto emit=[&](size_t id,const std::vector<ggml_fp16_t>&u,int64_t first,int64_t rows,const uint8_t*p,size_t bytes){
                check(id==emissions/size_t(n)&&first==int64_t(emissions%size_t(n))&&rows==1,"candidate and slab order");
                check(u==scales[id]&&bytes==row_bytes&&std::equal(p,p+bytes,expected[id].data()+size_t(first)*row_bytes),"all-statistics scales and bytes");++emissions;
            };
            auto result=sinq::materialize_candidates(GGML_TYPE_Q5_0,n,k,captured?x:std::vector<float>{},captured,held?v:std::vector<float>{},held?samples:0,nullptr,limit,read,encode,emit,raw,opt);
            check(result.workspace_bytes==limit&&result.encode_count==grid.size()&&emissions==grid.size()*size_t(n),"unchanged workspace and candidate budgets");
            for(size_t i=0;i<grid.size();++i)check(result.candidates[i].train_score==train_scores[i]&&result.candidates[i].validation_score==validation_scores[i],"all-statistics score equivalence");
            for(float bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
                size_t reads=0,computes=0,encodes=0,emits=0;
                auto invalid_read=[&](int64_t first,int64_t rows,float*out){read(first,rows,out);++reads;if(first+rows==n)out[rows*k-1]=bad;};
                auto count_compute=[&](const sinq::candidate_spec&s){++computes;return raw(s);};
                auto count_encode=[&](const float*in,void*out,int64_t first,int64_t rows,const float*im){++encodes;return encode(in,out,first,rows,im);};
                auto count_emit=[&](size_t,const std::vector<ggml_fp16_t>&,int64_t,int64_t,const uint8_t*,size_t){++emits;};
                bool rejected=false;
                try{sinq::materialize_candidates(GGML_TYPE_Q5_0,n,k,captured?x:std::vector<float>{},captured,held?v:std::vector<float>{},held?samples:0,nullptr,limit,invalid_read,count_encode,count_emit,count_compute,opt);}catch(const std::runtime_error&){rejected=true;}
                check(rejected&&reads==size_t(n)&&!computes&&!encodes&&!emits,"complete finite validation before any candidate work");
            }
        }
    }
}
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
    test_effective_correction(w,x,v,n,k,samples);
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
