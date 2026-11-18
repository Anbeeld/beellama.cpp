#include "sinq/sinq-parallel.h"
#include "sinq/sinq-local-score.h"
#include "sinq/sinq-bridge-config.h"
#include "sinq/sinq-candidates.h"
#include "sinq/sinq-native-codec.h"
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <thread>

static void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static void candidate_threads() {
    constexpr int64_t n=97,k=64;
    constexpr int samples=7;
    constexpr size_t limit=65536;
    std::vector<float> w(n*k),x(samples*k),validation(samples*k),raw(k);
    for(int64_t i=0;i<n;++i)for(int64_t j=0;j<k;++j)
        w[i*k+j]=float((i*13+j*7)%37-18)*.0625f*float(1+j%5);
    for(int t=0;t<samples;++t)for(int64_t j=0;j<k;++j) {
        x[t*k+j]=float((t*11+j*3)%29-14)*.125f;
        validation[t*k+j]=float((t*7+j*5)%31-15)*.0625f;
    }
    for(int64_t j=0;j<k;++j)raw[j]=1.f+float(j%7)*.125f;
    sinq::search_options options;options.held_out=true;
    options.candidates.resize(3);
    for(auto & spec:options.candidates)spec.strip_width=32;
    for(size_t i=1;i<options.candidates.size();++i) {
        auto & spec=options.candidates[i];spec.generator=sinq::generator_kind::awq;
        spec.alpha=.5;spec.lambda=1;
        spec.correction=i==1?sinq::correction_kind::weight_activation:sinq::correction_kind::activation_only;
    }
    // Scalar means on dyadic inputs are exact in FP64, independently of the
    // production compensated accumulators and the slab partition.
    std::vector<std::vector<ggml_fp16_t>> expected(3,std::vector<ggml_fp16_t>(k));
    for(size_t id=0;id<expected.size();++id) {
        std::vector<double> factors(k,1.);
        double lo=std::numeric_limits<double>::infinity(),hi=0;
        for(int64_t j=0;j<k;++j) {
            double activation=0,weight=0;
            for(int t=0;t<samples;++t)activation+=std::abs(double(x[t*k+j]));
            for(int64_t i=0;i<n;++i)weight+=std::abs(double(w[i*k+j]));
            if(id)factors[j]=std::max(1.e-4,std::pow(activation/samples,.5));
            if(id==1)factors[j]=std::max(1.e-4,factors[j]/(std::pow(weight/n,.5)+1.e-4));
            lo=std::min(lo,factors[j]);hi=std::max(hi,factors[j]);
        }
        const double normalization=std::sqrt(lo*hi);
        for(int64_t j=0;j<k;++j)expected[id][j]=ggml_fp32_to_fp16(id?float(double(raw[j])/(factors[j]/normalization)):raw[j]);
    }
    const auto caller=std::this_thread::get_id();
    int64_t read_end=0;
    auto read=[&](int64_t first,int64_t rows,float * out) {
        check(std::this_thread::get_id()==caller,"slab reads stay on caller thread");
        check(first==0||first==read_end,"slab reads stay in ascending order");read_end=first+rows;
        std::copy_n(w.data()+first*k,rows*k,out);
    };
    auto encode=[&](const float * in,void * out,int64_t,int64_t rows,const float * importance) {
        check(std::this_thread::get_id()==caller,"encoder stays on caller thread");
        return ggml_quantize_chunk(GGML_TYPE_Q5_0,in,out,0,rows,k,importance);
    };
    auto compute=[&](const sinq::candidate_spec &) {return raw;};
    for(auto objective:{sinq::local_objective::output_l1,sinq::local_objective::output_mse}) {
        options.objective=objective;
        std::vector<std::vector<uint8_t>> one_bytes(3),many_bytes(3);
        auto run=[&](int threads,std::vector<std::vector<uint8_t>> & bytes) {
            options.threads=threads;
            auto emit=[&](size_t id,const std::vector<ggml_fp16_t> &,int64_t first,int64_t,const uint8_t * data,size_t count) {
                check(std::this_thread::get_id()==caller,"emitter stays on caller thread");
                check(bytes[id].size()==size_t(first)*ggml_row_size(GGML_TYPE_Q5_0,k),"emission stays in row order");
                bytes[id].insert(bytes[id].end(),data,data+count);
            };
            return sinq::materialize_candidates(GGML_TYPE_Q5_0,n,k,x,samples,validation,samples,nullptr,limit,read,encode,emit,compute,options);
        };
        const auto one=run(1,one_bytes),many=run(32,many_bytes);
        check(one.workspace_bytes<=limit&&many.workspace_bytes<=limit,"threaded candidates retain workspace budget");
        check(one.encode_count==3&&many.encode_count==3,"all generator descriptors encoded");
        check(one_bytes==many_bytes,"1/32 threads preserve exact packed outputs across slab sizes");
        for(size_t id=0;id<3;++id) {
            const auto & a=one.candidates[id];const auto & b=many.candidates[id];
            check(a.scales==expected[id]&&b.scales==expected[id],"generator scales match independent scalar means");
            check(a.train_score==b.train_score&&a.validation_score==b.validation_score,"1/32 threads preserve exact generator scores");
        }
    }
    w.back()=std::numeric_limits<float>::quiet_NaN();
    for(int threads:{1,32}) {
        options.threads=threads;bool rejected=false;int emissions=0;
        try {
            sinq::materialize_candidates(GGML_TYPE_Q5_0,n,k,x,samples,validation,samples,nullptr,limit,read,encode,
                [&](size_t,const std::vector<ggml_fp16_t>&,int64_t,int64_t,const uint8_t*,size_t){++emissions;},compute,options);
        } catch(const std::runtime_error &) {rejected=true;}
        check(rejected&&emissions==0,"invalid source propagates from column workers before emission");
    }
}

static std::thread::id encode_caller;
static std::atomic<bool> worker_threw{false};
static size_t throwing_quantize(ggml_type type, const float * input, void * output,
        int64_t start, int64_t rows, int64_t columns, const float * importance) {
    if (std::this_thread::get_id()!=encode_caller) {
        worker_threw=true;
        throw std::runtime_error("injected worker codec failure");
    }
    while (!worker_threw.load()) std::this_thread::yield();
    return ggml_quantize_chunk(type,input,output,start,rows,columns,importance);
}

static size_t invalid_quantize(ggml_type type, const float *, void * output,
        int64_t, int64_t rows, int64_t columns, const float *) {
    const auto bytes=rows*ggml_row_size(type,columns);
    std::memset(output,0xff,bytes);return bytes;
}

static size_t throw_any_quantize(ggml_type, const float *, void *,
        int64_t, int64_t, int64_t, const float *) {
    throw std::runtime_error("injected serial codec failure");
}

static void native_failures_and_parity() {
    constexpr int64_t columns=256, rows=37, experts=11;
    std::vector<float> input(rows*columns),importance(5*columns,1.f);
    for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i)*.031f);
    std::vector<std::thread> workers;
    for(auto type:{GGML_TYPE_Q3_K,GGML_TYPE_Q4_K,GGML_TYPE_Q5_0,GGML_TYPE_IQ4_XS}) {
        std::vector<uint8_t> expected(rows*ggml_row_size(type,columns)),actual(expected.size());
        check(sinq::native_encode(type,input.data(),expected.data(),3*columns,2,rows,experts,columns,importance.data(),workers,1)==expected.size(),"serial byte count");
        for(int threads:{2,4,8,16,32}) {
            check(sinq::native_encode(type,input.data(),actual.data(),3*columns,2,rows,experts,columns,importance.data(),workers,threads)==actual.size(),"parallel byte count");
            check(actual==expected&&workers.empty(),"thread/expert-boundary packed parity and cleanup");
        }
    }
    std::vector<uint8_t> output(rows*ggml_row_size(GGML_TYPE_Q4_K,columns));
    sinq::native_encode_hooks hooks;
    hooks.quantize=throwing_quantize;encode_caller=std::this_thread::get_id();worker_threw=false;
    bool caught=false;
    try {sinq::native_encode(GGML_TYPE_Q4_K,input.data(),output.data(),columns,0,rows,experts,columns,importance.data(),workers,4,&hooks);}
    catch(const std::runtime_error & error) {
        caught=true;
        const std::string message=error.what();
        check(message.find(ggml_type_name(GGML_TYPE_Q4_K))!=std::string::npos&&message.find("rows [")!=std::string::npos,
            "worker failure reports codec and global row range");
        bool nested=false;
        try {std::rethrow_if_nested(error);} catch(const std::runtime_error & cause) {
            nested=std::string(cause.what())=="injected worker codec failure";
        }
        check(nested,"worker failure preserves original codec exception");
    }
    check(caught&&worker_threw&&workers.empty(),"non-main worker exception propagates and joins");
    hooks=sinq::native_encode_hooks{};
    int launches=0;std::atomic<int> active{0};
    hooks.launch=[&](std::function<void()> work) {
        if(++launches==2)throw std::runtime_error("injected thread launch failure");
        return std::thread([&,work] {++active;work();--active;});
    };
    caught=false;
    try {sinq::native_encode(GGML_TYPE_Q4_K,input.data(),output.data(),columns,0,rows,experts,columns,importance.data(),workers,4,&hooks);}
    catch(const std::runtime_error &) {caught=true;}
    check(caught&&launches==2&&active==0&&workers.empty(),"partial launch failure joins and clears");
    check(sinq::native_encode(GGML_TYPE_Q4_K,input.data(),output.data(),columns,0,rows,experts,columns,importance.data(),workers,32)==output.size(),"encode succeeds after injected failures");
    for(int threads:{1,32}) {
        hooks=sinq::native_encode_hooks{};hooks.quantize=invalid_quantize;caught=false;
        try {sinq::native_encode(GGML_TYPE_Q4_K,input.data(),output.data(),columns,0,rows,experts,columns,importance.data(),workers,threads,&hooks);}
        catch(const std::runtime_error &) {caught=true;}
        check(caught&&workers.empty(),"invalid packed data fails with all workers joined");
    }
    hooks=sinq::native_encode_hooks{};hooks.quantize=throw_any_quantize;caught=false;
    try {sinq::native_encode(GGML_TYPE_Q4_K,input.data(),output.data(),columns,0,rows,experts,columns,importance.data(),workers,1,&hooks);}
    catch(const std::runtime_error &) {caught=true;}
    check(caught&&workers.empty(),"serial callback exception remains controlled");
    hooks=sinq::native_encode_hooks{};launches=0;
    hooks.launch=[&](std::function<void()> work) {++launches;return std::thread(std::move(work));};
    check(sinq::native_encode(GGML_TYPE_Q4_K,input.data(),output.data(),columns*rows,0,rows,rows,columns,importance.data(),workers,32,&hooks)==output.size(),"single-task size");
    check(launches==0&&workers.empty(),"small slabs never launch idle workers");
}

int main() try {
    auto * context=ggml_init({1024,nullptr,true});check(context!=nullptr,"ggml initialization");ggml_free(context);
    native_failures_and_parity();
    candidate_threads();
    std::vector<int> visits(10000, 0);
    std::mutex mutex;
    std::set<std::thread::id> workers;
    sinq::parallel_for(int64_t(visits.size()), 4, [&](int64_t begin, int64_t end) {
        {
            std::lock_guard<std::mutex> guard(mutex);
            workers.insert(std::this_thread::get_id());
        }
        for (int64_t i = begin; i < end; ++i) ++visits[size_t(i)];
    });
    check(workers.size() > 1 && workers.size() <= 4, "parallel range used multiple bounded workers");
    for (int value : visits) check(value == 1, "parallel ranges cover every element exactly once");

    bool propagated = false;
    try {
        sinq::parallel_for(100, 4, [](int64_t begin, int64_t) {
            if (begin != 0) throw std::runtime_error("worker failure");
        });
    } catch (const std::runtime_error &) { propagated = true; }
    check(propagated, "worker exception propagated after join");

    const int64_t rows = 23, columns = 64;
    std::vector<float> weights(size_t(rows * columns)), decoded(size_t(rows * columns));
    std::vector<float> activation(5 * columns);
    std::vector<ggml_fp16_t> scales(static_cast<size_t>(columns));
    for (int64_t i = 0; i < rows * columns; ++i) {
        weights[size_t(i)] = std::sin(float(i) * .013f);
        decoded[size_t(i)] = std::cos(float(i) * .019f);
    }
    for (int64_t i = 0; i < 5 * columns; ++i) activation[size_t(i)] = std::sin(float(i) * .071f);
    for (auto & scale : scales) scale = ggml_fp32_to_fp16(1.0f);
    for (auto objective : {sinq::local_objective::output_l1, sinq::local_objective::output_mse}) {
        std::vector<double> one(static_cast<size_t>(rows));
        std::vector<double> many(static_cast<size_t>(rows));
        sinq::output_scores(weights.data(), decoded.data(), scales.data(), rows, columns,
            activation.data(), 5, objective, one.data(), 1);
        sinq::output_scores(weights.data(), decoded.data(), scales.data(), rows, columns,
            activation.data(), 5, objective, many.data(), 4);
        check(one == many, "parallel per-row scoring preserves exact values");
        sinq::output_scores(weights.data(), decoded.data(), scales.data(), rows, columns,
            activation.data(), 5, objective, many.data(), 32);
        check(one == many, "32-thread per-row scoring preserves exact values");
    }

    const auto default_threads = std::max(1u, std::thread::hardware_concurrency());
    check(sinq::search_options{}.threads == int(default_threads), "search defaults to all logical CPU threads");
    check(sinq::bridge_request{}.threads == int(default_threads), "bridge defaults to all logical CPU threads");
    std::puts("test-sinq-threading: PASS");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "test-sinq-threading: %s\n", error.what());
    return 1;
}
