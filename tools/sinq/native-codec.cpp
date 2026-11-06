// Offline optimizer filesystem bridge; numerical algorithms remain in src/sinq.
#include "sinq/sinq-bridge-config.h"
#include "sinq/sinq-native-codec.h"
#include "sinq/sinq-numeric.h"
#include "sinq/sinq-local-score.h"
#include "sinq/sinq-parallel.h"
#include "sinq/sinq-parameters.h"
#include <regex>
#include "llama-sinq-activation.h"
#include <filesystem>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>
namespace fs=std::filesystem;
using json=nlohmann::json;
static std::vector<uint8_t> load(const std::string & path,size_t count,size_t limit) {
    sinq::require(count<=limit && fs::file_size(path)==count,"SINQ file cardinality/budget mismatch");
    std::vector<uint8_t> v(count);std::ifstream f(path,std::ios::binary);f.read((char*)v.data(),count);
    sinq::require(bool(f),"SINQ input read failure");return v;
}
static std::vector<float> floats(const std::string&p,size_t count,size_t limit) {
    auto v=load(p,count*sizeof(float),limit);std::vector<float> out(count);std::memcpy(out.data(),v.data(),v.size());return out;
}
static void write(const fs::path&p,const void*data,size_t count) {
    sinq::require(!fs::exists(p),"SINQ refusing output overwrite");
    std::ofstream f(p,std::ios::binary);f.write((const char*)data,count);f.flush();sinq::require(bool(f),"SINQ output write failure");
}
int main(int argc,char**argv) try {
    if(argc==4 && std::string(argv[1])=="--plan-config") {
        sinq::require(fs::file_size(argv[2])<=1024*1024 && fs::file_size(argv[3])<=16*1024*1024,"plan input exceeds byte budget");
        std::ifstream config(argv[2]),inventory_file(argv[3]);
        const auto config_json=sinq::parse_bridge_json(config),inventory=sinq::parse_bridge_json(inventory_file);
        sinq::require(inventory.is_array() && inventory.size()<=100000,"plan inventory must be a bounded array");
        const auto type=[](const std::string&name){for(int t=0;t<GGML_TYPE_COUNT;++t){const char*n=ggml_type_name(ggml_type(t));if(!n)continue;std::string canonical(n);std::transform(canonical.begin(),canonical.end(),canonical.begin(),[](unsigned char c){return char(std::tolower(c));});if(name==canonical)return ggml_type(t);}throw std::runtime_error("unknown inventory type");};
        std::vector<sinq::tensor_descriptor>tensors;
        const std::regex pattern("^blk\\.([0-9]+)\\.(attn_q|attn_k|attn_v|attn_qkv|attn_gate|attn_output|ssm_beta|ssm_alpha|ssm_out|ffn_gate|ffn_up|ffn_down)\\.weight$");
        for(const auto&item:inventory) {
            sinq::require(item.is_object() && item.size()==5 && item.contains("name") && item.contains("rows") && item.contains("columns") && item.contains("qtype") && item.contains("source_type"),"invalid inventory fields");
            sinq::require(item.at("rows").is_number_integer() && item.at("columns").is_number_integer(),"inventory dimensions must be integers");
            sinq::tensor_descriptor d;d.name=item.at("name").get<std::string>();d.rows=item.at("rows");d.columns=item.at("columns");d.qtype=type(item.at("qtype"));
            sinq::require(!d.name.empty() && d.rows>0 && d.columns>0,"invalid inventory name/dimensions");
            const auto source=type(item.at("source_type"));std::smatch match;
            const bool projection=std::regex_match(d.name,match,pattern);
            if(projection){d.layer=std::stoi(match[1]);d.role=match[2];}
            d.eligible=projection && d.rows>=2 && llama_sinq_supported_type(d.qtype) && (source==GGML_TYPE_F32 || source==GGML_TYPE_F16 || source==GGML_TYPE_BF16);
            tensors.push_back(std::move(d));
        }
        json entries=json::array();
        for(const auto&entry:sinq::resolve_direct(config_json,tensors))entries.push_back({{"name",entry.tensor.name},{"parameters",sinq::parameters_json(entry.settings)},{"shape",{entry.tensor.columns,entry.tensor.rows}},{"qtype",ggml_type_name(entry.tensor.qtype)},{"locked",entry.locked}});
        std::cout<<json({{"tensors",entries}}).dump()<<"\n";return 0;
    }
    if((argc==4 || argc==5) && std::string(argv[1])=="--validate-activations") {
        int threads=int(std::max(1u,std::thread::hardware_concurrency()));
        if(argc==5) {
            size_t used=0;threads=std::stoi(argv[4],&used);
            sinq::require(used==std::strlen(argv[4]) && threads>=1 && threads<=1024,"invalid activation validation thread count");
        }
        const auto manifest=llama_sinq_load_activations(argv[2],argv[3],threads);
        json entries=json::object();
        for(const auto & item:manifest.tensors) { const auto & e=item.second;
            entries[item.first]={{"input_dim",e.input_dim},{"output_dim",e.output_dim},{"count",e.count},{"data_file",e.data_file},{"sha256",e.sha256},{"site",e.site},{"reference_module",e.reference_module}};
        }
        std::cout<<json({{"source_sha256",manifest.source_sha256},{"manifest_sha256",manifest.manifest_sha256},{"source_revision",manifest.source_revision},{"tensors",entries}}).dump()<<"\n";
        return 0;
    }
    sinq::require(argc==2,"usage: llama-sinq-native request.json | --validate-activations manifest source.gguf");
    sinq::require(fs::file_size(argv[1])<=1024*1024,"SINQ bridge config too large");
    std::ifstream config(argv[1]);const auto j=sinq::parse_bridge_json(config);const auto r=sinq::parse_bridge_request(j);
    auto *ctx=ggml_init({1024,nullptr,true});ggml_free(ctx);
    const size_t row_bytes=ggml_row_size(r.codec,r.columns);
    const size_t cells=size_t(r.rows)*r.columns;
    const size_t packed_size=size_t(r.rows)*row_bytes;
    const auto start=std::chrono::steady_clock::now();
    if(r.operation=="decode") {
        const size_t offset=size_t(r.first_row)*row_bytes;
        sinq::require(fs::file_size(r.payload)>=offset && fs::file_size(r.payload)-offset>=packed_size &&
            packed_size+cells*4+size_t(r.columns)*2<=r.workspace && cells*4<=r.max_output_bytes,"SINQ decode bounds");
        sinq::require(!fs::exists(r.output),"SINQ refusing output overwrite");
        std::ifstream f(r.payload,std::ios::binary);f.seekg(offset);std::vector<uint8_t>packed(packed_size);f.read((char*)packed.data(),packed.size());
        sinq::require(bool(f) && ggml_validate_row_data(r.codec,packed.data(),packed.size()),"SINQ malformed native payload");
        std::vector<ggml_fp16_t>u(size_t(r.columns),ggml_fp32_to_fp16(1));
        if(!r.scales.empty()){auto b=load(r.scales,u.size()*2,r.workspace);std::memcpy(u.data(),b.data(),b.size());}
        std::vector<float>out(cells);
        sinq::parallel_for(r.rows,r.threads,[&](int64_t begin,int64_t end){
            for(int64_t row=begin;row<end;++row) {
                ggml_get_type_traits(r.codec)->to_float(packed.data()+row*row_bytes,out.data()+row*r.columns,r.columns);
                for(int64_t column=0;column<r.columns;++column) {
                    const size_t i=size_t(row)*r.columns+column;
                    const float scale=ggml_fp16_to_fp32(u[size_t(column)]);
                    sinq::require(scale>0 && std::isfinite(scale),"SINQ invalid F16 scale");
                    out[i]*=scale;sinq::require(std::isfinite(out[i]),"SINQ nonfinite reconstruction");
                }
            }
        });
        write(r.output,out.data(),out.size()*4);std::cout<<"{\"status\":\"PASS\",\"operation\":\"decode\"}\n";return 0;
    }
    const size_t source_bytes=ggml_row_size(r.source_type,r.columns)*size_t(r.rows);
    // The source is bounded independently of candidate slabs. No hidden unbounded
    // tensor allocation: large tensors require an explicitly larger budget.
    sinq::require(source_bytes+cells*4<r.workspace/2,"SINQ source exceeds resident budget");
    sinq::require(!fs::exists(r.output),"SINQ output directory already exists");
    const size_t candidate_bound=r.operation=="stock"?1:r.search.candidates.size();
    if(r.operation!="score")sinq::require(candidate_bound<=r.max_output_bytes/(packed_size+size_t(r.columns)*2),"SINQ output byte budget exceeded");
    const auto source=load(r.weights,source_bytes,r.workspace);
    std::vector<float>w(cells);
    const int conversion_threads=int(std::min<size_t>(r.threads,(cells+16383)/16384));
    sinq::parallel_for(r.rows,conversion_threads,[&](int64_t begin,int64_t end) {
        const size_t first=size_t(begin)*r.columns,count=size_t(end-begin)*r.columns;
        if(r.source_type==GGML_TYPE_F32)std::memcpy(w.data()+first,source.data()+first*sizeof(float),count*sizeof(float));
        else if(r.source_type==GGML_TYPE_F16)ggml_fp16_to_fp32_row((const ggml_fp16_t*)source.data()+first,w.data()+first,count);
        else ggml_bf16_to_fp32_row((const ggml_bf16_t*)source.data()+first,w.data()+first,count);
        for(size_t i=first;i<first+count;++i)sinq::require(std::isfinite(w[i]),"optimizer nonfinite source weight");
    });
    std::vector<float>importance;
    if(!r.importance.empty())importance=floats(r.importance,r.columns,r.workspace);
    sinq::require(r.operation=="score" || !importance.empty() || !ggml_quantize_requires_imatrix(r.codec),"SINQ required imatrix missing");
    for(float v:importance)sinq::require(std::isfinite(v)&&v>=0,"SINQ invalid importance");
    std::vector<float>x,validation;
    if(r.operation=="generate" || r.operation=="score") {
        if(r.samples)x=floats(r.activations,size_t(r.samples)*r.columns,r.workspace);
        if(r.search.held_out)validation=floats(r.validation,size_t(r.validation_samples)*r.columns,r.workspace);
    }
    if(r.operation=="score") {
        const size_t per=row_bytes+size_t(r.columns)*4+2*sizeof(double);
        const size_t base_resident=source_bytes+w.size()*4+(x.size()+validation.size())*4+size_t(r.columns)*32;
        const size_t scratch=size_t(r.columns)*sizeof(double);
        sinq::require(r.workspace>base_resident && r.workspace-base_resident>=per+scratch,"SINQ rescore workspace exhausted");
        const int scoring_threads=int(std::min<size_t>(size_t(std::min<int64_t>(r.threads,r.rows)),
            (r.workspace-base_resident-per)/scratch));
        const size_t resident=base_resident+size_t(scoring_threads)*scratch;
        sinq::require(r.workspace>resident && r.workspace-resident>=per,"SINQ rescore workspace exhausted");
        sinq::require(fs::file_size(r.payload)==packed_size,"SINQ score payload size mismatch");
        const int64_t slab=std::min<int64_t>(r.rows,(r.workspace-resident)/per);
        std::vector<uint8_t>packed(size_t(slab)*row_bytes);std::vector<float>dq(size_t(slab)*r.columns);
        std::vector<double>train_rows(size_t(slab),0.),validation_rows(size_t(slab),0.);
        std::vector<ggml_fp16_t>u(size_t(r.columns),ggml_fp32_to_fp16(1));
        if(!r.scales.empty()){auto b=load(r.scales,u.size()*2,r.workspace);std::memcpy(u.data(),b.data(),b.size());}
        std::ifstream f(r.payload,std::ios::binary);sinq::sum train,val;
        for(int64_t first=0;first<r.rows;first+=slab) {
            const auto rows=std::min(slab,r.rows-first);const auto count=size_t(rows)*row_bytes;
            f.read((char*)packed.data(),count);sinq::require(bool(f)&&ggml_validate_row_data(r.codec,packed.data(),count),"SINQ score invalid native payload");
            sinq::parallel_for(rows,scoring_threads,[&](int64_t begin,int64_t end){
                for(int64_t row=begin;row<end;++row)
                    ggml_get_type_traits(r.codec)->to_float(packed.data()+row*row_bytes,dq.data()+row*r.columns,r.columns);
            });
            sinq::output_scores(w.data()+first*r.columns,dq.data(),u.data(),rows,r.columns,x.data(),r.samples,
                r.search.objective,train_rows.data(),scoring_threads);
            if(r.search.held_out)sinq::output_scores(w.data()+first*r.columns,dq.data(),u.data(),rows,r.columns,
                validation.data(),r.validation_samples,r.search.objective,validation_rows.data(),scoring_threads);
            for(int64_t row=0;row<rows;++row) {
                train.add(train_rows[size_t(row)]);
                if(r.search.held_out)val.add(validation_rows[size_t(row)]);
            }
        }
        const auto text=json({{"schema_version",2},{"objective",r.search.objective==sinq::local_objective::output_l1?"output_l1":"output_mse"},{"train_score",train.value/double(r.rows)},{"validation_score",r.search.held_out?json(val.value/double(r.rows)):json(nullptr)},{"encode_count",0}}).dump(2)+"\n";
        write(r.output,text.data(),text.size());std::cout<<text;return 0;
    }
    const auto temporary=fs::path(r.output+".partial");
    sinq::require(!fs::exists(temporary),"SINQ stale partial output; use a fresh directory");
    fs::create_directories(temporary);
    std::vector<std::thread>workers;
    const int64_t chunk=r.columns*std::max<int64_t>(1,(16384+r.columns-1)/r.columns);
    double encode_seconds=0,normalization_seconds=0,derive_seconds=0,statistics_seconds=0,score_seconds=0,io_seconds=0;
    int effective_encode_threads_peak=0;
    size_t normalization_calls=0,normalization_cache_hits=0;
    auto encode=[&](const float*in,void*out,int64_t first,int64_t rows,const float*im){
        const auto begin=std::chrono::steady_clock::now();
        const auto rows_per_chunk=chunk/r.columns;
        effective_encode_threads_peak=std::max(effective_encode_threads_peak,int(std::min<int64_t>(r.threads,(rows+rows_per_chunk-1)/rows_per_chunk)));
        const auto bytes=sinq::native_encode(r.codec,in,out,chunk,first,rows,r.rows,r.columns,im,workers,r.threads);
        encode_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        return bytes;
    };
    json result={{"schema_version",2},{"operation",r.operation},{"codec",ggml_type_name(r.codec)},{"rows",r.rows},{"columns",r.columns}};
    if(r.operation=="stock") {
        std::vector<uint8_t>packed(packed_size);
        sinq::require(encode(w.data(),packed.data(),0,r.rows,importance.empty()?nullptr:importance.data())==packed.size(),"SINQ native size mismatch");
        write(temporary/"stock.bin",packed.data(),packed.size());result["payload"]="stock.bin";
    } else {
        auto read=[&](int64_t first,int64_t rows,float*out){std::copy_n(w.data()+first*r.columns,rows*r.columns,out);};
        auto emit=[&](size_t id,const std::vector<ggml_fp16_t>&u,int64_t first,int64_t,const uint8_t*bytes,size_t count){
            const std::string name="candidate-"+std::to_string(id);
            if(first==0)write(temporary/(name+".f16"),u.data(),u.size()*2);
            std::ofstream out(temporary/(name+".bin"),std::ios::binary|std::ios::app);out.write((const char*)bytes,count);out.flush();sinq::require(bool(out),"SINQ slab write failed");
        };
        // Four FIFO entries, including raw FP32 factors and strip diagnostics,
        // remain within 32 bytes/column. Charge this in addition to temporaries.
        constexpr size_t cache_entries=4;
        const size_t resident=source_bytes+w.size()*4+size_t(r.columns)*64;
        size_t normalization_reserve=0;
        for(const auto&spec:r.search.candidates)if(spec.strength!=0) {
            size_t bytes=0;sinq::require(llama_sinq_workspace_size(r.rows,spec.strip_width,bytes),"invalid normalization workspace");
            const size_t workers=std::min<size_t>(size_t(r.columns/spec.strip_width),std::min<size_t>(r.threads,(128ull*1024*1024)/bytes));
            normalization_reserve=std::max(normalization_reserve,bytes*workers);
        }
        sinq::require(resident<r.workspace && normalization_reserve<r.workspace-resident,"SINQ resident workspace exhausted");
        struct normalization_entry {
            int width,iterations;
            std::vector<float> factors;
            llama_sinq_statistics statistics;
        };
        std::vector<normalization_entry> normalization_cache;
        std::vector<json>diagnostics,normalization_execution;
        auto compute=[&](const sinq::candidate_spec&spec) {
            llama_sinq_options normalization;normalization.strip_width=spec.strip_width;normalization.iterations=spec.iterations;normalization.strength=float(spec.strength);
            size_t per_worker=0;
            sinq::require(llama_sinq_workspace_size(r.rows,spec.strip_width,per_worker),"normalization shape exceeds canonical limit");
            const size_t workers=std::min<size_t>(size_t(r.columns/spec.strip_width),std::min<size_t>(r.threads,(128ull*1024*1024)/per_worker));
            sinq::require(spec.strength==0 || per_worker*workers<=normalization_reserve,"normalization exceeds aggregate workspace budget");
            std::vector<ggml_fp16_t>ordinary;std::vector<float>raw;llama_sinq_statistics stats;
            if (spec.strength==0) {
                const auto begin=std::chrono::steady_clock::now();
                sinq::require(llama_sinq_compute_scales(source.data(),r.source_type,r.rows,r.columns,normalization,r.threads,ordinary,stats,&raw),"Sinkhorn failed");
                derive_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            } else {
                auto found=std::find_if(normalization_cache.begin(),normalization_cache.end(),[&](const normalization_entry & entry) {
                    return entry.width==spec.strip_width && entry.iterations==spec.iterations;
                });
                if (found==normalization_cache.end()) {
                    const auto begin=std::chrono::steady_clock::now();
                    normalization.strength=1.f;
                    normalization_entry entry{spec.strip_width,spec.iterations,{},{}};
                    sinq::require(llama_sinq_compute_scales(source.data(),r.source_type,r.rows,r.columns,normalization,r.threads,ordinary,entry.statistics,&entry.factors),"Sinkhorn failed");
                    ++normalization_calls;
                    normalization_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                    if (normalization_cache.size()==cache_entries) normalization_cache.erase(normalization_cache.begin());
                    normalization_cache.push_back(std::move(entry));
                    found=normalization_cache.end()-1;
                } else ++normalization_cache_hits;
                const auto begin=std::chrono::steady_clock::now();
                stats=found->statistics;
                sinq::require(llama_sinq_apply_strength(found->factors,float(spec.strength),ordinary,stats,&raw),"Sinkhorn strength failed");
                derive_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            }
            normalization_execution.push_back({{"workers",spec.strength==0?0:workers},{"workspace_bytes",spec.strength==0?0:per_worker*workers}});
            diagnostics.push_back({{"min_scale",stats.min_scale},{"max_scale",stats.max_scale},{"median_scale",stats.median_scale},{"identity_strips",stats.selected_identity_count},{"packed_bytes",packed_size}});
            return raw;
        };
        const auto found=sinq::materialize_candidates(r.codec,r.rows,r.columns,x,r.samples,validation,r.validation_samples,
            importance.empty()?nullptr:importance.data(),r.workspace-resident-normalization_reserve,read,encode,emit,compute,r.search);
        derive_seconds+=found.compensation_seconds;statistics_seconds=found.statistics_seconds;
        score_seconds=found.score_seconds;io_seconds=found.emit_seconds;
        result["normalization_execution"]=normalization_execution;
        result["effective_materialization_threads"]=found.effective_threads;result["slab_rows"]=found.slab_rows;
        result["activation_statistics"]=found.activation_statistics;result["weight_statistics"]=found.weight_statistics;
        result["encode_count"]=found.encode_count;result["workspace_bytes"]=found.workspace_bytes+resident+normalization_reserve;result["candidates"]=json::array();
        for(size_t id=0;id<found.candidates.size();++id){const auto&c=found.candidates[id];const auto name="candidate-"+std::to_string(id);
            result["candidates"].push_back({{"index",id},{"parameters",sinq::candidate_spec_json(c.spec)},{"diagnostics",diagnostics.at(id)},{"encoded",c.encoded},{"payload",name+".bin"},{"scales",name+".f16"},{"train_score",c.has_score?json(c.train_score):json(nullptr)},{"validation_score",c.has_validation?json(c.validation_score):json(nullptr)}});
        }
    }
    result["requested_threads"]=r.threads;
    result["effective_encode_threads_peak"]=effective_encode_threads_peak;
    result["normalization_calls"]=normalization_calls;
    result["normalization_cache_hits"]=normalization_cache_hits;
    result["phase_seconds"]={{"normalize",normalization_seconds},{"derive",derive_seconds},{"encode",encode_seconds},
        {"statistics",statistics_seconds},{"score",score_seconds},{"candidate_write",io_seconds}};
    result["elapsed_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const auto text=result.dump(2)+"\n";write(temporary/"result.json",text.data(),text.size());fs::rename(temporary,r.output);
    ggml_quantize_free();std::cout<<result.dump()<<"\n";return 0;
} catch(const std::exception&e){std::cerr<<"SINQ optimizer native: "<<e.what()<<"\n";return 1;}
