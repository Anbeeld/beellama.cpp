#include "sinq/sinq-search.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <cmath>
static void check(bool x,const char*s){if(!x)throw std::runtime_error(s);}
int main() try {
    auto *ctx=ggml_init({1024,nullptr,true});ggml_free(ctx);
    const int n=3,k=256,samples=3;
    std::vector<float>w(n*k),raw(k,1),x(samples*k),v(samples*k),importance(k,1);
    for(int i=0;i<n*k;++i)w[i]=float(sin(i*.17));
    for(int i=0;i<samples*k;++i){x[i]=float(cos(i*.31));v[i]=float(sin(i*.71));}
    auto read=[&](int64_t first,int64_t rows,float*out){std::copy_n(w.data()+first*k,rows*k,out);};
    auto encode=[&](const float*in,void*out,int64_t,int64_t rows,const float*im){return ggml_quantize_chunk(GGML_TYPE_Q5_0,in,out,0,rows,k,im);};
    sinq::search_options opt;opt.max_candidates=32;opt.max_shortlist=2;opt.refine_step=.1;opt.held_out=true;
    opt.candidates={{sinq::scale_family::weight_activation,.5,1},{sinq::scale_family::activation_only,0,1},{sinq::scale_family::weight_activation,.5,0},{sinq::scale_family::activation_only,.5,.5}};
    size_t calls=0; auto emit=[&](size_t,const std::vector<ggml_fp16_t>&,int64_t,int64_t rows,const uint8_t*,size_t bytes){check(bytes==size_t(rows)*ggml_row_size(GGML_TYPE_Q5_0,k),"actual packed slab");++calls;};
    const auto a=sinq::research_search(GGML_TYPE_Q5_0,n,k,raw,x,samples,v,samples,importance.data(),65536,read,encode,emit,opt);
    check(!a.candidates.empty() && a.candidates[0].spec.family==sinq::scale_family::ordinary,"ordinary first");
    check(a.candidates.size()<=32 && calls>0,"bounded native search");
    check(a.candidates[a.selected].has_validation,"heldout ranked frozen candidates");
    check(a.candidates[1].scales==a.candidates[0].scales && !a.candidates[1].encoded,"canonical identity dedupe");
    const auto b=sinq::research_search(GGML_TYPE_Q5_0,n,k,raw,x,samples,v,samples,importance.data(),128*1024,read,encode,emit,opt);
    check(a.selected==b.selected && a.candidates.size()==b.candidates.size(),"slab choice invariant");
    for(size_t i=0;i<a.candidates.size();++i)check(a.candidates[i].train_score==b.candidates[i].train_score && a.candidates[i].validation_score==b.candidates[i].validation_score,"score reduction invariant");
    bool failed=false;opt.max_candidates=1;try{sinq::research_search(GGML_TYPE_Q5_0,n,k,raw,x,samples,v,samples,nullptr,65536,read,encode,emit,opt);}catch(...){failed=true;}check(failed,"candidate budget rejected");
    ggml_quantize_free();puts("test-sinq-search: PASS");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}
