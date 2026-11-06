#include "llama-sinq.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)
static float half(float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }
static bool close(float a, float b, float tol = 1.e-4f) {
    return std::isfinite(a) && std::isfinite(b) && std::abs(a-b) <= tol*std::max(1.e-6f,std::abs(b));
}
static float clip(float x, float a, float b) { return std::max(a,std::min(b,x)); }

// Independent transcription of hash-pinned sinkhorn.py with its FP32 CUDA
// reduction tree. Double two-pass variance is not an oracle here: tiny changes
// in near-zero uniform rescaling can reverse the best-iterate comparison.
// Locked binaries expose T, but do not contain row factors.
static void reference(const std::vector<float> & w, int n, int g, int iterations,
                      std::vector<float> & bc, std::vector<float> & br, llama_sinq_tile_statistics * diagnostic = nullptr) {
    std::vector<float> lr(n,0), lc(g,0), rs(n), cs(g), cur(w.size());
    br.assign(n,1); bc.assign(g,1);
    struct accumulator { float avg=0, sum=0, size=0; };
    auto merge = [](accumulator a, accumulator b) {
        if (a.size==0) { return b; } if (b.size==0) { return a; }
        float delta=b.avg-a.avg, total=a.size+b.size, fraction=b.size/total;
        return accumulator{std::fma(delta,fraction,a.avg),std::fma(delta*delta*a.size,fraction,a.sum+b.sum),total};
    };
    auto deviation = [&](int base,int stride,int length,int lanes,bool row) {
        std::vector<accumulator> values(lanes);
        for (int lane=0;lane<lanes;++lane) {
            accumulator pair[2];
            for(int index=lane,step=0;index<length;index+=lanes,++step) {
                auto & a=pair[step%2]; float x=cur[base+index*stride];
                a.size+=1; float delta=x-a.avg; a.avg+=delta/a.size;
                a.sum=std::fma(delta,x-a.avg,a.sum);
            }
            values[lane]=merge(pair[0],pair[1]);
        }
        if(row) {
            int dim=lanes;
            while(dim>32) {
                for(int lane=0;lane<dim/2;++lane) { values[lane]=merge(values[lane],values[lane+dim/2]); }
                dim/=2;
            }
            for(int offset=1;offset<dim;offset*=2) {
                for(int lane=0;lane+offset<dim;++lane) { values[lane]=merge(values[lane],values[lane+offset]); }
            }
        } else {
            for(int offset=lanes/2;offset>0;offset/=2) {
                for(int lane=0;lane<offset;++lane) { values[lane]=merge(values[lane],values[lane+offset]); }
            }
        }
        return std::sqrt(values[0].sum/float(length-1));
    };
    auto axes = [&]() {
        int h=1; while(h*2<=n && h<16) { h*=2; }
        int ch=1; while(ch*2<=n && ch*2<=128/(g/4)) { ch*=2; }
        int cl=n>=std::min(ch*16,256)?ch:1;
        for(int i=0;i<n;++i) { rs[i]=deviation(i*g,1,g,std::min(g,512/h),true); }
        for(int j=0;j<g;++j) { cs[j]=deviation(j,g,n,cl,false); }
    };
    auto imbalance = [&]() {
        float lo=std::numeric_limits<float>::infinity(), hi=0;
        for (float x:rs) { lo=std::min(lo,x); hi=std::max(hi,x); }
        for (float x:cs) { lo=std::min(lo,x); hi=std::max(hi,x); }
        return hi/std::max(1.e-12f,lo);
    };
    cur=w; axes(); float target=1.e3f;
    for (float x:rs) { target=std::min(target,clip(x,1.e-3f,1.e3f)); }
    for (float x:cs) { target=std::min(target,clip(x,1.e-3f,1.e3f)); }
    target+=1.e-6f; float best=imbalance(); bool stopped=false;
    if (diagnostic) { diagnostic->initial_imbalance=best; }
    for (int it=0; it<iterations; ++it) {
        for (int i=0; i<n; ++i) { for (int j=0; j<g; ++j) { cur[i*g+j]=w[i*g+j]/std::exp(lc[j])/std::exp(lr[i]); } }
        axes(); float ib=imbalance();
        if (ib<=best) {
            if (diagnostic) { diagnostic->chosen_iteration=it; }
            best=ib;
            for (int i=0; i<n; ++i) { br[i]=std::exp(lr[i]); }
            for (int j=0; j<g; ++j) { bc[j]=std::exp(lc[j]); }
        }
        stopped=stopped || ib>best;
        if (!stopped) {
            for (int i=0; i<n; ++i) { lr[i]=clip(lr[i]+std::log(clip(clip(rs[i],1.e-3f,1.e3f)/target,.7f,2.f)),-.3f,10.f); }
            for (int j=0; j<g; ++j) { lc[j]=clip(lc[j]+std::log(clip(clip(cs[j],1.e-3f,1.e3f)/target,.7f,2.f)),-.3f,10.f); }
        }
    }
    if (diagnostic) {
        diagnostic->best_imbalance=best;
        for (int i=0;i<n;++i) { for (int j=0;j<g;++j) { cur[i*g+j]=w[i*g+j]/std::exp(lc[j])/std::exp(lr[i]); } }
        axes();diagnostic->terminal_imbalance=imbalance();
    }
}
struct fixture {
    uint32_t h[6];
    std::vector<float> w,t;
};
static fixture read_fixture(const std::string & path) {
    std::ifstream f(path,std::ios::binary); CHECK(f.good()); fixture a;
    f.read(reinterpret_cast<char *>(a.h),sizeof(a.h));
    CHECK(a.h[0]==0x514e4953 && a.h[1]==2 && a.h[2]>=2 && a.h[3]%a.h[4]==0);
    const size_t nk=size_t(a.h[2])*a.h[3], groups=nk/a.h[4];
    a.w.resize(nk); a.t.resize(a.h[3]);
    f.read(reinterpret_cast<char *>(a.w.data()),nk*sizeof(float));
    f.seekg((nk+2*groups)*sizeof(float),std::ios::cur);
    f.read(reinterpret_cast<char *>(a.t.data()),a.t.size()*sizeof(float)); CHECK(f.good());
    CHECK(f.peek()==std::char_traits<char>::eof()); return a;
}
static void fixtures(int argc, char ** argv) {
    CHECK(argc>1);
    for (int arg=1; arg<argc; ++arg) {
        const std::string path=argv[arg]; fixture a=read_fixture(path);
        CHECK(path.size()>8 && path.substr(path.size()-8)==".pre.bin");
        fixture post=read_fixture(path.substr(0,path.size()-8)+".bin");
        CHECK(a.w==post.w && std::memcmp(a.h,post.h,sizeof(a.h))==0);
        const int n=a.h[2], k=a.h[3], g=a.h[4];
        llama_sinq_options opt; opt.strip_width=g;
        std::vector<ggml_fp16_t> t, parallel; llama_sinq_statistics stats, ps;
        CHECK(llama_sinq_compute_scales(a.w.data(),GGML_TYPE_F32,n,k,opt,1,t,stats));
        CHECK(llama_sinq_compute_scales(a.w.data(),GGML_TYPE_F32,n,k,opt,4,parallel,ps));
        CHECK(t==parallel && !stats.identity && !ps.identity);
        CHECK(stats.strips.size()==size_t(k/g));
        float lo=std::numeric_limits<float>::infinity(), hi=0;
        for (int j=0; j<k; ++j) {
            float actual=ggml_fp16_to_fp32(t[j]); CHECK(actual>0 && std::isfinite(actual));
            CHECK(t[j] == ggml_fp32_to_fp16(post.t[j]));
            CHECK(ggml_fp32_to_fp16(a.t[j]) == ggml_fp32_to_fp16(post.t[j]));
            lo=std::min(lo,actual); hi=std::max(hi,actual);
        }
        CHECK(stats.min_scale==lo && stats.max_scale==hi);
        for (int start=0; start<k; start+=g) {
            std::vector<float> tile(n*g), columns(g), rows(n), ec,er;
            for (int i=0; i<n; ++i) { for (int j=0; j<g; ++j) { tile[i*g+j]=a.w[i*k+start+j]; } }
            for (int it:{4,8,16}) {
                CHECK(llama_sinq_balance_tile(tile.data(),n,g,it,columns.data(),rows.data()));
                reference(tile,n,g,it,ec,er);
                for (int j=0; j<g; ++j) {
                    if (!close(columns[j],ec[j])) { std::fprintf(stderr,"reference mismatch %s strip=%d iterations=%d column=%d actual=%.9g expected=%.9g\n",path.c_str(),start,it,j,columns[j],ec[j]); }
                    CHECK(close(columns[j],ec[j]));
                }
                for (int i=0; i<n; ++i) { CHECK(close(rows[i],er[i])); }
                if (it==16) { for (int j=0; j<g; ++j) { CHECK(close(columns[j],a.t[start+j])); } }
            }
        }
    }
    std::printf("locked fixtures: %d, independent row/column factors at 4/8/16 steps\n",argc-1);
}
static void invalid() {
    std::vector<float> w(512,1), c(128),r(4); std::vector<ggml_fp16_t> t;
    llama_sinq_options o; llama_sinq_statistics s;
    CHECK(!llama_sinq_balance_tile(nullptr,4,64,16,c.data(),r.data()));
    CHECK(!llama_sinq_balance_tile(w.data(),1,64,16,c.data(),r.data()));
    CHECK(!llama_sinq_balance_tile(w.data(),4,48,16,c.data(),r.data()));
    CHECK(!llama_sinq_balance_tile(w.data(),4,64,15,c.data(),r.data()));
    CHECK(!llama_sinq_balance_tile(w.data(),4,64,16,nullptr,r.data()));
    CHECK(!llama_sinq_balance_tile(w.data(),4,64,16,c.data(),nullptr));
    CHECK(!llama_sinq_compute_scales(nullptr,GGML_TYPE_F32,4,64,o,1,t,s));
    CHECK(!llama_sinq_compute_scales(w.data(),GGML_TYPE_Q4_K,4,64,o,1,t,s));
    CHECK(!llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,1,64,o,1,t,s));
    CHECK(!llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,4,0,o,1,t,s));
    CHECK(!llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,4,96,o,1,t,s));
    CHECK(!llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,4,64,o,0,t,s));
    CHECK(!llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,4,64,o,-1,t,s));
    CHECK(!llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,std::numeric_limits<int64_t>::max(),128,o,1,t,s));
    o.iterations=0; CHECK(!llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,4,64,o,1,t,s));
    o.iterations=16; o.strip_width=48; CHECK(!llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,4,64,o,1,t,s));
    o.strip_width=64;
    for (float bad:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        w[17]=bad;
        CHECK(!llama_sinq_balance_tile(w.data(),4,64,16,c.data(),r.data()));
        for (bool identity:{false,true}) {
            o.identity=identity;
            CHECK(!llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,4,64,o,1,t,s));
            std::vector<ggml_fp16_t> h(w.size()); std::vector<ggml_bf16_t> b(w.size());
            for (size_t i=0;i<w.size();++i) { h[i]=ggml_fp32_to_fp16(w[i]); b[i]=ggml_fp32_to_bf16(w[i]); }
            CHECK(!llama_sinq_compute_scales(h.data(),GGML_TYPE_F16,4,64,o,1,t,s));
            CHECK(!llama_sinq_compute_scales(b.data(),GGML_TYPE_BF16,4,64,o,1,t,s));
        }
    }
    w[17]=70000; CHECK(!llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,4,64,o,1,t,s));
}
static void width_edges() {
    llama_sinq_options o; llama_sinq_statistics s;
    std::vector<ggml_fp16_t> scales;
    float dummy=1;
    for(int bad:{-512,-1,0,16,31,48,96,1024,std::numeric_limits<int>::max()}) {
        CHECK(!llama_sinq_valid_strip_width(bad)); o.strip_width=bad;
        CHECK(!llama_sinq_compute_scales(&dummy,GGML_TYPE_F32,2,512,o,1,scales,s));
    }
    for(int width:{32,64,128,256,512}) {
        CHECK(llama_sinq_valid_strip_width(width)); o.strip_width=width;
        CHECK(!llama_sinq_compute_scales(&dummy,GGML_TYPE_F32,0,width,o,1,scales,s));
        CHECK(!llama_sinq_compute_scales(&dummy,GGML_TYPE_F32,-2,width,o,1,scales,s));
        CHECK(!llama_sinq_compute_scales(&dummy,GGML_TYPE_F32,2,-width,o,1,scales,s));
        CHECK(!llama_sinq_compute_scales(&dummy,GGML_TYPE_F32,2,width+width/2,o,1,scales,s));
        CHECK(!llama_sinq_compute_scales(&dummy,GGML_TYPE_F32,std::numeric_limits<int64_t>::max(),width,o,1,scales,s));
        // A legal tensor byte count can still exceed the indivisible strip
        // budget. Reject before touching the deliberately one-element source.
        const int64_t too_tall=int64_t((128ull*1024*1024-uint64_t(width)*32-8192)/(uint64_t(width)*8+24))+1;
        size_t bytes=0;
        CHECK(llama_sinq_workspace_size(too_tall-1,width,bytes));
        CHECK(bytes==size_t(uint64_t(too_tall-1)*(uint64_t(width)*8+24)+uint64_t(width)*32+8192));
        CHECK(bytes<=128ull*1024*1024);
        CHECK(!llama_sinq_workspace_size(too_tall,width,bytes));
        CHECK(!llama_sinq_workspace_size(std::numeric_limits<int64_t>::max(),width,bytes));
        CHECK(!llama_sinq_compute_scales(&dummy,GGML_TYPE_F32,too_tall,width,o,4,scales,s));
        CHECK(!llama_sinq_balance_tile(&dummy,too_tall,width,16,&dummy,&dummy));
        const int n=3,k=4*width;
        std::vector<float> w(n*k),rounded(n*k),col(width),row(n),ec,er;
        std::vector<ggml_fp16_t> h(n*k); std::vector<ggml_bf16_t> b(n*k);
        for(float value:{0.f,.125f,1.e-7f}) { for(bool outlier:{false,true}) {
            std::fill(w.begin(),w.end(),value); if(outlier) { w[width+3]=30.f; }
            for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16}) {
                for(int i=0;i<n*k;++i) {
                    h[i]=ggml_fp32_to_fp16(w[i]); b[i]=ggml_fp32_to_bf16(w[i]);
                    rounded[i]=half(type==GGML_TYPE_BF16?ggml_bf16_to_fp32(b[i]):w[i]);
                }
                const void * input=type==GGML_TYPE_F32?static_cast<const void *>(w.data()):
                    type==GGML_TYPE_F16?static_cast<const void *>(h.data()):static_cast<const void *>(b.data());
                std::vector<ggml_fp16_t> parallel;
                CHECK(llama_sinq_compute_scales(input,type,n,k,o,1,scales,s));
                CHECK(llama_sinq_compute_scales(input,type,n,k,o,4,parallel,s)); CHECK(scales==parallel);
                for(int start=0;start<k;start+=width) {
                    std::vector<float> tile(n*width);
                    for(int i=0;i<n;++i) { for(int j=0;j<width;++j) { tile[i*width+j]=rounded[i*k+start+j]; } }
                    reference(tile,n,width,16,ec,er);
                    for(int j=0;j<width;++j) {
                        CHECK(scales[start+j]==ggml_fp32_to_fp16(ec[j]));
                        CHECK(std::isfinite(ggml_fp16_to_fp32(scales[start+j])) && ggml_fp16_to_fp32(scales[start+j])>0);
                    }
                }
            }
        } }
        for(float bad:{INFINITY,-INFINITY,NAN,70000.f}) {
            std::fill(w.begin(),w.end(),1.f); w[width-1]=bad;
            for(int i=0;i<n*k;++i) { h[i]=ggml_fp32_to_fp16(w[i]); b[i]=ggml_fp32_to_bf16(w[i]); }
            for(bool identity:{false,true}) {
                o.identity=identity;
                CHECK(!llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,o,1,scales,s));
                CHECK(!llama_sinq_compute_scales(h.data(),GGML_TYPE_F16,n,k,o,4,scales,s));
                CHECK(!llama_sinq_compute_scales(b.data(),GGML_TYPE_BF16,n,k,o,4,scales,s));
            }
            o.identity=false;
        }
        for(float bad:{INFINITY,-INFINITY,NAN}) {
            w[width-1]=bad;
            CHECK(!llama_sinq_balance_tile(w.data(),n,width,16,col.data(),row.data()));
        }
    }
    std::puts("width edges: unsupported/misaligned/overflow/resource shapes, all source types, zero/constant/near-zero/outlier/nonfinite");
}
static void synthetic() {
    const int n=9,k=512;
    std::vector<float> w(n*k), src(n*k), rounded(n*k);
    for (int i=0;i<n;++i) { for (int j=0;j<k;++j) { w[i*k+j]=float(std::sin(i*3.17+j*.73)*(1+(j%17)*.19)*(1+i*.11)+.031); } }
    for (ggml_type type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16}) {
        std::vector<ggml_fp16_t> h(n*k); std::vector<ggml_bf16_t> b(n*k);
        for (int i=0;i<n*k;++i) {
            h[i]=ggml_fp32_to_fp16(w[i]); b[i]=ggml_fp32_to_bf16(w[i]);
            src[i]=type==GGML_TYPE_F32?w[i]:type==GGML_TYPE_F16?ggml_fp16_to_fp32(h[i]):ggml_bf16_to_fp32(b[i]);
            rounded[i]=half(src[i]);
        }
        const void * input=type==GGML_TYPE_F32?static_cast<const void *>(w.data()):type==GGML_TYPE_F16?static_cast<const void *>(h.data()):static_cast<const void *>(b.data());
        for (int width:{32,64,128,256,512}) { for (int it:{4,8,16}) {
            llama_sinq_options o; o.strip_width=width; o.iterations=it;
            llama_sinq_statistics s,rs; std::vector<ggml_fp16_t> t,t4,t32,rt;
            std::vector<float> raw,raw32;
            CHECK(llama_sinq_compute_scales(input,type,n,k,o,1,t,s,&raw));
            CHECK(llama_sinq_compute_scales(input,type,n,k,o,4,t4,rs)); CHECK(t==t4);
            CHECK(llama_sinq_compute_scales(input,type,n,k,o,32,t32,rs,&raw32));
            CHECK(t==t32 && raw==raw32);
            CHECK(llama_sinq_compute_scales(rounded.data(),GGML_TYPE_F32,n,k,o,1,rt,rs)); CHECK(t==rt);
            std::vector<float> scaled(n*k), x(k), compensated(k);
            bool rounded_factor=false;
            for (int start=0;start<k;start+=width) {
                std::vector<float> tile(n*width),col(width),row(n);
                for (int i=0;i<n;++i) { for(int j=0;j<width;++j) { tile[i*width+j]=rounded[i*k+start+j]; } }
                CHECK(llama_sinq_balance_tile(tile.data(),n,width,it,col.data(),row.data()));
                for (int j=0;j<width;++j) {
                    CHECK(t[start+j]==ggml_fp32_to_fp16(col[j]));
                    CHECK(raw[start+j]==col[j]);
                    rounded_factor=rounded_factor || col[j]!=half(col[j]);
                }
            }
            CHECK(rounded_factor);
            for(int j=0;j<k;++j) { x[j]=float(std::cos(j*.37)); compensated[j]=x[j]*ggml_fp16_to_fp32(t[j]); }
            double wrong_delta=0;
            for(int i=0;i<n;++i) {
                double expected=0,actual=0;
                for(int j=0;j<k;++j) {
                    scaled[i*k+j]=src[i*k+j]/ggml_fp16_to_fp32(t[j]);
                    expected+=double(src[i*k+j])*x[j]; actual+=double(scaled[i*k+j])*compensated[j];
                    wrong_delta+=std::abs(double(scaled[i*k+j])-double(rounded[i*k+j])/ggml_fp16_to_fp32(t[j]));
                }
                CHECK(std::abs(actual-expected)<2.e-5*(1+std::abs(expected)));
            }
            if(type!=GGML_TYPE_F16) { CHECK(wrong_delta>0); }
            // A three-column manual matmul, independently accumulated in FP64.
            for(int rhs=0;rhs<3;++rhs) { for(int i=0;i<n;++i) {
                double expected=0,actual=0;
                for(int j=0;j<k;++j) {
                    const float activation=float(std::sin(j*.21+rhs*1.31));
                    const float runtime_input=activation*ggml_fp16_to_fp32(t[j]);
                    expected+=double(src[i*k+j])*activation;
                    actual+=double(scaled[i*k+j])*runtime_input;
                }
                CHECK(std::abs(actual-expected)<2.e-5*(1+std::abs(expected)));
            } }
            for(ggml_type qtype:{GGML_TYPE_Q3_K,GGML_TYPE_Q4_K}) {
                const size_t bytes=ggml_row_size(qtype,k);
                std::vector<uint8_t> encoded(n*bytes),ref(bytes); std::vector<float> decoded(n*k);
                CHECK(ggml_quantize_chunk(qtype,scaled.data(),encoded.data(),0,n,k,nullptr)==encoded.size());
                double err=0,energy=0;
                for(int i=0;i<n;++i) {
                    ggml_get_type_traits(qtype)->from_float_ref(scaled.data()+i*k,ref.data(),k);
                    CHECK(std::memcmp(ref.data(),encoded.data()+i*bytes,bytes)==0);
                    ggml_get_type_traits(qtype)->to_float(encoded.data()+i*bytes,decoded.data()+i*k,k);
                    double left=0,right=0;
                    for(int j=0;j<k;++j) {
                        double reconstructed=double(decoded[i*k+j])*ggml_fp16_to_fp32(t[j]);
                        double d=reconstructed-src[i*k+j]; err+=d*d; energy+=double(src[i*k+j])*src[i*k+j];
                        left+=double(decoded[i*k+j])*compensated[j]; right+=reconstructed*x[j];
                    }
                    CHECK(std::abs(left-right)<2.e-5*(1+std::abs(right)));
                }
                CHECK(std::sqrt(err/energy)<(qtype==GGML_TYPE_Q3_K?.20:.12));
            }
            o.identity=true; CHECK(llama_sinq_compute_scales(input,type,n,k,o,4,t,s)); CHECK(s.identity);
            for(auto v:t) { CHECK(ggml_fp16_to_fp32(v)==1); }
        } }
    }
    for(int width:{32,64,128,256,512}) {
        llama_sinq_options o; o.strip_width=width; llama_sinq_statistics s; std::vector<ggml_fp16_t> t;
        std::fill(w.begin(),w.end(),.125f);
        CHECK(llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,o,4,t,s)); CHECK(!s.identity);
        std::vector<float> constant(n*width,.125f),ec,er;
        reference(constant,n,width,16,ec,er);
        for(int j=0;j<k;++j) { CHECK(t[j]==ggml_fp32_to_fp16(ec[j%width])); }
        w[k+width+3]=30;
        CHECK(llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,o,4,t,s));
        CHECK(!s.identity && s.strips.size()==size_t(k/width));
        for (int start=0; start<k; start+=width) {
            std::vector<float> tile(n*width),columns,rows;
            for (int i=0;i<n;++i) { for (int j=0;j<width;++j) { tile[i*width+j]=w[i*k+start+j]; } }
            reference(tile,n,width,16,columns,rows);
            for (int j=0;j<width;++j) { CHECK(t[start+j]==ggml_fp32_to_fp16(columns[j])); }
        }
    }
    std::puts("synthetic: source F32/F16/BF16, widths 32/64/128/256/512, iterations 4/8/16, threads 1/4/32, exact raw/scales, native Q3_K/Q4_K, exact serialized compensation");
}
// Independent scalar two-pass FP64 sample moments audit the initial Welford
// reductions without repeating their lane/tree algorithm. Best-iterate parity
// still uses the FP32 reference because near ties depend on operation order.
static double scalar_imbalance(const std::vector<float> & tile, int n, int g) {
    double lo=std::numeric_limits<double>::infinity(),hi=0;
    auto deviation=[&](int base,int stride,int length) {
        double mean=0,m2=0;
        for(int i=0;i<length;++i) { mean+=tile[base+i*stride]; }
        mean/=length;
        for(int i=0;i<length;++i) { const double d=tile[base+i*stride]-mean; m2+=d*d; }
        const double sd=std::sqrt(m2/(length-1)); lo=std::min(lo,sd); hi=std::max(hi,sd);
    };
    for(int i=0;i<n;++i) { deviation(i*g,1,g); }
    for(int j=0;j<g;++j) { deviation(j,g,n); }
    return hi/std::max(1.e-12,lo);
}
static void row_shapes() {
    for(int n:{2,3,7,16,17,33,128,257,1025}) { for(int g:{32,64,128,256,512}) {
        const int k=4*g;
        std::vector<float> w(n*k),tile(n*g),columns(g),rows(n),ec,er;
        for(int i=0;i<n;++i) { for(int j=0;j<k;++j) {
            w[i*k+j]=half(float(std::sin(i*.77+j*.39)*(1+(j%19)*.1)+(i%5)*.02));
            if(j<g) { tile[i*g+j]=w[i*k+j]; }
        } }
        for(int it:{4,8,16}) {
            llama_sinq_tile_statistics diagnostic,expected;
            CHECK(llama_sinq_balance_tile(tile.data(),n,g,it,columns.data(),rows.data(),&diagnostic));
            reference(tile,n,g,it,ec,er,&expected);
            CHECK(close(diagnostic.initial_imbalance,float(scalar_imbalance(tile,n,g)),3.e-4f));
            CHECK(diagnostic.chosen_iteration==expected.chosen_iteration);
            CHECK(diagnostic.best_imbalance==expected.best_imbalance);
            CHECK(diagnostic.terminal_imbalance==expected.terminal_imbalance);
            for(int j=0;j<g;++j) { CHECK(close(columns[j],ec[j])); }
            for(int i=0;i<n;++i) { CHECK(close(rows[i],er[i])); }
            llama_sinq_options o; o.strip_width=g; o.iterations=it;
            llama_sinq_statistics s1,s4; std::vector<ggml_fp16_t> t1,t4;
            CHECK(llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,o,1,t1,s1));
            CHECK(llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,o,4,t4,s4));
            CHECK(t1==t4 && s1.identity==s4.identity);
            for(size_t group=0;group<s1.strips.size();++group) {
                CHECK(s1.strips[group].chosen_iteration==s4.strips[group].chosen_iteration);
                CHECK(s1.strips[group].best_imbalance==s4.strips[group].best_imbalance);
            }
            for(int j=0;j<g;++j) { CHECK(t1[j]==ggml_fp32_to_fp16(columns[j])); }
        }
    } }
    std::puts("row shapes: 2/3/7/16/17/33/128/257/1025, all widths, full-height reduction branches");
}
static void strength_and_diagnostics() {
    constexpr int n=33,k=512;
    std::vector<float> w(n*k);
    for (int i=0;i<n*k;++i) { w[i]=half(float(std::sin(i*.73)*(1+(i%23)*.2))); }
    for (int width:{32,64,128,256,512}) { for (int iterations:{4,8,16}) {
        llama_sinq_options o;o.strip_width=width;o.iterations=iterations;
        std::vector<ggml_fp16_t> legacy,actual,identity;
        llama_sinq_statistics ls,s;
        std::vector<float> cached_factors;
        CHECK(llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,o,1,legacy,ls,&cached_factors));
        o.identity=true;CHECK(llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,o,1,identity,s));o.identity=false;
        for(float alpha:{0.f,.5f,.75f,1.f}) {
            o.strength=alpha;
            CHECK(llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,o,4,actual,s));
            std::vector<ggml_fp16_t> one,many;std::vector<float> raw_one,raw_many;
            llama_sinq_statistics one_stats,many_stats;
            CHECK(llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,o,1,one,one_stats,&raw_one));
            CHECK(llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,o,32,many,many_stats,&raw_many));
            CHECK(one==actual && many==actual && raw_one==raw_many);
            std::vector<ggml_fp16_t> cached;
            std::vector<float> cached_raw;
            auto cached_stats=ls;
            CHECK(llama_sinq_apply_strength(cached_factors,alpha,cached,cached_stats,&cached_raw));
            CHECK(cached==actual && cached_raw==raw_one);
            CHECK(cached_stats.min_scale==s.min_scale && cached_stats.max_scale==s.max_scale && cached_stats.median_scale==s.median_scale);
            if(alpha==0) {
                CHECK(actual==identity && s.identity && s.strips.empty());
                for(float value:raw_one)CHECK(value==1.f);
                continue;
            }
            if(alpha==1)CHECK(actual==legacy);
            CHECK(s.strips.size()==size_t(k/width));
            size_t selected_identity=0;
            for(int group=0;group<k/width;++group) {
                std::vector<float> tile(n*width),ec,er;
                for(int i=0;i<n;++i)for(int j=0;j<width;++j)tile[i*width+j]=w[i*k+group*width+j];
                llama_sinq_tile_statistics expected;
                reference(tile,n,width,iterations,ec,er,&expected);
                const auto & d=s.strips[group];
                CHECK(d.chosen_iteration==expected.chosen_iteration);
                CHECK(d.chosen_iteration>=0 && d.chosen_iteration<iterations);
                CHECK(close(d.initial_imbalance,expected.initial_imbalance));
                CHECK(close(d.best_imbalance,expected.best_imbalance));
                CHECK(close(d.terminal_imbalance,expected.terminal_imbalance));
                CHECK(d.best_imbalance<=d.initial_imbalance);
                selected_identity+=d.chosen_iteration==0;
                for(int j=0;j<width;++j) {
                    const float value=alpha==1 ? ec[j] : std::exp(alpha*std::log(ec[j]));
                    CHECK(actual[group*width+j]==ggml_fp32_to_fp16(value));
                    CHECK(close(raw_one[group*width+j],value));
                    const double t=ggml_fp16_to_fp32(actual[group*width+j]);
                    const double x=.375,I=1.25,W=w[group*width+j];
                    CHECK(std::abs((W/t)*(x*t)-W*x)<=1.e-12*(1+std::abs(W*x)));
                    CHECK(std::abs((x*t)*(x*t)*I-x*x*(t*t*I))<=1.e-12);
                }
            }
            CHECK(s.selected_identity_count==selected_identity);
        }
        for(float alpha:{-.01f,1.01f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
            o.strength=alpha;CHECK(!llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,o,1,actual,s));
        }
    } }
    std::fill(w.begin(),w.end(),.125f);
    llama_sinq_options o;llama_sinq_statistics s;std::vector<ggml_fp16_t> t;
    CHECK(llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,o,1,t,s));CHECK(!s.identity && s.strips.size()==size_t(k/o.strip_width));
    std::fill(w.begin(),w.end(),0.f);
    CHECK(llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,o,1,t,s));CHECK(!s.identity && s.strips.size()==size_t(k/o.strip_width));
    std::puts("strength/diagnostics: independent preupdate oracle, serialized compensation and constant normalization passed");
}

static void parallel_source_validation() {
    // 32 coarse scan tasks, with bad values in both the caller's range and
    // later worker ranges. Identity isolates validation from strip work.
    constexpr int n=64,k=32768;
    std::vector<float> w(n*k,.125f);
    std::vector<ggml_fp16_t> h(n*k,ggml_fp32_to_fp16(.125f));
    std::vector<ggml_bf16_t> b(n*k,ggml_fp32_to_bf16(.125f));
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16}) {
        const void * input=type==GGML_TYPE_F32?static_cast<const void *>(w.data()):
            type==GGML_TYPE_F16?static_cast<const void *>(h.data()):static_cast<const void *>(b.data());
        llama_sinq_options o;o.identity=true;
        llama_sinq_statistics s1,s32;
        std::vector<ggml_fp16_t> one,many;std::vector<float> raw_one,raw_many;
        CHECK(llama_sinq_compute_scales(input,type,n,k,o,1,one,s1,&raw_one));
        CHECK(llama_sinq_compute_scales(input,type,n,k,o,32,many,s32,&raw_many));
        CHECK(one==many && raw_one==raw_many && s1.identity && s32.identity);
        CHECK(one.size()==size_t(k) && raw_one.size()==size_t(k));
        for(int j=0;j<k;++j)CHECK(one[j]==ggml_fp32_to_fp16(1.f) && raw_one[j]==1.f);
        for(int index:{17,n*k/2+3,n*k-1})for(float bad:{INFINITY,-INFINITY,NAN,70000.f}) {
            w[index]=bad;h[index]=ggml_fp32_to_fp16(bad);b[index]=ggml_fp32_to_bf16(bad);
            for(int mode=0;mode<3;++mode) {
                o.identity=mode==1;o.strength=mode==2?0.f:1.f;
                for(int threads:{1,32}) {
                    std::vector<ggml_fp16_t> output{ggml_fp32_to_fp16(2.f)};
                    std::vector<float> raw{3.f};llama_sinq_statistics stats;stats.min_scale=7.f;
                    CHECK(!llama_sinq_compute_scales(input,type,n,k,o,threads,output,stats,&raw));
                    CHECK(output.size()==1 && output[0]==ggml_fp32_to_fp16(2.f));
                    CHECK(raw.size()==1 && raw[0]==3.f && stats.min_scale==7.f);
                }
            }
            w[index]=.125f;h[index]=ggml_fp32_to_fp16(.125f);b[index]=ggml_fp32_to_bf16(.125f);
        }
    }
    std::puts("parallel source validation: exact identity outputs, F32/F16/BF16 nonfinite and FP16 overflow rejected at 1/32 threads without publishing outputs");
}

int main(int argc,char ** argv) {
    // Initialize conversion tables on platforms that use them.
    ggml_context * ctx=ggml_init({1024,nullptr,true}); CHECK(ctx); ggml_free(ctx);
    parallel_source_validation(); strength_and_diagnostics(); fixtures(argc,argv); invalid(); width_edges(); synthetic(); row_shapes(); std::puts("test-qk-sinq-numeric: PASS");
}
