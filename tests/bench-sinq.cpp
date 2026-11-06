// Standalone scaling evidence, deliberately not a CTest performance gate.
#include "llama-sinq.h"
#include "sinq/sinq-native-codec.h"
#include "sinq/sinq-local-score.h"
#include "ggml-impl.h"
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

int main() try {
    auto * context=ggml_init({1024,nullptr,true});check(context!=nullptr,"ggml initialization");ggml_free(context);
    nlohmann::json records=nlohmann::json::array();
    for (const auto shape : {std::pair<int64_t,int64_t>{32,256},{512,1024},{4096,1024}}) {
        const auto n=shape.first,k=shape.second;
        constexpr int samples=4,repetitions=3;
        std::vector<float> w(size_t(n*k)),x(size_t(samples*k)),dq(w.size()),importance(size_t(k),1.f);
        for (int64_t i=0;i<n*k;++i) w[size_t(i)]=std::sin(float(i)*.017f)*(.25f+float(i%k)/float(k));
        for (int64_t i=0;i<samples*k;++i) x[size_t(i)]=std::cos(float(i)*.031f);
        const size_t bytes=size_t(n)*ggml_row_size(GGML_TYPE_Q4_K,k);
        std::vector<uint8_t> packed(bytes),reference_bytes;
        std::vector<ggml_fp16_t> scales,reference_scales;
        std::vector<double> losses(size_t(n),0.),reference_l1,reference_mse;
        std::vector<std::thread> workers;
        const int64_t rows_per_chunk=std::max<int64_t>(1,16384/k);
        const int64_t tasks=(n+rows_per_chunk-1)/rows_per_chunk;
        for (int threads : {1,2,4,8,16,32}) {
            for (int repeat=0;repeat<repetitions;++repeat) {
                const auto measure=[&](const char * phase,int effective,const auto & work) {
                    const auto begin=std::chrono::steady_clock::now();work();
                    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                    records.push_back({{"phase",phase},{"rows",n},{"columns",k},{"samples",samples},
                        {"threads",threads},{"effective_threads",effective},{"repeat",repeat},{"seconds",seconds},
                        {"elements",n*k},{"bytes_packed",bytes}});
                };
                llama_sinq_options options;options.strip_width=64;options.iterations=8;options.strength=.5f;
                llama_sinq_statistics stats;size_t workspace=0;
                check(llama_sinq_workspace_size(n,64,workspace),"normalization workspace");
                const int normalization_workers=int(std::min<int64_t>({threads,k/64,int64_t((128ull<<20)/workspace)}));
                measure("normalization",normalization_workers,[&] {
                    check(llama_sinq_compute_scales(w.data(),GGML_TYPE_F32,n,k,options,threads,scales,stats),"normalization failed");
                });
                if (reference_scales.empty()) reference_scales=scales;
                check(scales==reference_scales,"normalization thread/repeat parity");
                measure("native_pack",int(std::min<int64_t>(threads,tasks)),[&] {
                    check(sinq::native_encode(GGML_TYPE_Q4_K,w.data(),packed.data(),rows_per_chunk*k,0,n,n,k,
                        importance.data(),workers,threads)==bytes,"native byte count");
                });
                if (reference_bytes.empty()) reference_bytes=packed;
                check(packed==reference_bytes&&workers.empty(),"packing thread/repeat parity and cleanup");
                for (int64_t row=0;row<n;++row) ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(
                    packed.data()+row*ggml_row_size(GGML_TYPE_Q4_K,k),dq.data()+row*k,k);
                for (const auto objective : {sinq::local_objective::output_l1,sinq::local_objective::output_mse}) {
                    const bool l1=objective==sinq::local_objective::output_l1;
                    measure(l1?"local_l1":"local_mse",int(std::min<int64_t>(threads,n)),[&] {
                        sinq::output_scores(w.data(),dq.data(),scales.data(),n,k,x.data(),samples,objective,losses.data(),threads);
                    });
                    auto & reference=l1?reference_l1:reference_mse;
                    if (reference.empty()) reference=losses;
                    check(losses==reference,"local score thread/repeat parity");
                }
            }
        }
    }
    const nlohmann::json result={{"schema_version",1},{"status","PASS"},{"repetitions",3},
        {"scope","Standalone native phase scaling; exact F16, packed-byte and per-row score parity"},
        {"normalization",{{"strip_width",64},{"iterations",8},{"strength",.5}}},
        {"codec","q4_k"},{"records",records}};
    std::puts(result.dump(2).c_str());ggml_quantize_free();return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr,"bench-sinq: %s\n",error.what());return 1;
}
