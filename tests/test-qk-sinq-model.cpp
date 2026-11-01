#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "../src/llama-sinq.h"
#include "../vendor/nlohmann/json.hpp"


#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) { if (!ok) { throw std::runtime_error(message); } }
static std::string logs;
static void logger(ggml_log_level, const char * text, void *) { logs += text; }
static void make_model(const std::string & path, const std::string & mode, bool incompatible_shape = false) {
    ggml_context * ctx = ggml_init({64*ggml_tensor_overhead(),nullptr,true});
    gguf_context * file = gguf_init_empty();
    gguf_set_val_str(file,"general.architecture","qwen35");
    gguf_set_val_str(file,"general.name","SINQ integration fixture");
    const std::pair<const char *,uint32_t> ints[] = {
        {"context_length",256},{"embedding_length",256},{"block_count",1},{"feed_forward_length",512},
        {"attention.head_count",2},{"attention.head_count_kv",1},{"attention.key_length",128},{"attention.value_length",128},
        {"full_attention_interval",1},{"ssm.conv_kernel",4},{"ssm.inner_size",128},{"ssm.state_size",128},{"ssm.time_step_rank",1},{"ssm.group_count",1},
    };
    for (const auto & kv : ints) { gguf_set_val_u32(file,(std::string("qwen35.")+kv.first).c_str(),kv.second); }
    gguf_set_val_f32(file,"qwen35.attention.layer_norm_rms_epsilon",1.e-6f);
    uint32_t sections[] = {16,16,16,16};
    gguf_set_arr_data(file,"qwen35.rope.dimension_sections",GGUF_TYPE_UINT32,sections,4);
    const char * tokens[] = {"<unk>","<s>","</s>","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"," ",".","!"};
    float scores[32] = {}; int32_t token_types[32];
    for (int i = 0; i < 32; ++i) { token_types[i] = i == 0 ? 2 : i < 3 ? 3 : 1; }
    gguf_set_val_str(file,"tokenizer.ggml.model","llama");
    gguf_set_arr_str(file,"tokenizer.ggml.tokens",tokens,32);
    gguf_set_arr_data(file,"tokenizer.ggml.scores",GGUF_TYPE_FLOAT32,scores,32);
    gguf_set_arr_data(file,"tokenizer.ggml.token_type",GGUF_TYPE_INT32,token_types,32);
    gguf_set_val_u32(file,"tokenizer.ggml.unknown_token_id",0);
    gguf_set_val_u32(file,"tokenizer.ggml.bos_token_id",1);
    gguf_set_val_u32(file,"tokenizer.ggml.eos_token_id",2);
    const bool sinq = mode != "source" && mode != "metadata_only" && mode != "orphan";
    if ((sinq || mode == "orphan") && mode != "missing_version") {
        if (mode == "version_dtype") { gguf_set_val_str(file,"beellama.qk_sinq.version","1"); }
        else { gguf_set_val_u32(file,"beellama.qk_sinq.version",mode == "version_value" ? 2 : 1); }
    }
    if (mode != "source" && mode != "metadata_only") {
        gguf_set_val_str(file,"beellama.qk_sinq.algorithm","sinkhorn_log");
        gguf_set_val_u32(file,"beellama.qk_sinq.strip_width",mode == "bad_width" ? 32 : 64);
        gguf_set_val_u32(file,"beellama.qk_sinq.iterations",mode == "bad_iterations" ? 3 : 16);
        const char * target = mode == "unsupported" ? "output.weight" : "blk.0.ffn_up.weight";
        const char * targets[] = {target,target};
        if (mode != "missing_manifest") {
            gguf_set_arr_str(file,"beellama.qk_sinq.tensors",targets,mode == "duplicate_manifest" ? 2 : 1);
        }
    }
    if (mode == "legacy_enabled" || mode == "legacy_disabled") {
        gguf_set_val_bool(file,"beellama.qk_sinq.kurtosis_gate",mode == "legacy_enabled");
    }
    if (mode == "metadata_only") { gguf_set_val_bool(file,"beellama.qk_sinq.kurtosis_gate",false); }
    if (mode == "legacy_mistyped") { gguf_set_val_str(file,"beellama.qk_sinq.kurtosis_gate","off"); }
    if (mode == "unknown_metadata") { gguf_set_val_bool(file,"beellama.qk_sinq.unknown_control",false); }
    if (mode == "hadamard") { gguf_set_val_bool(file,"prism.hadamard.test",true); }
    std::vector<std::vector<uint8_t>> payloads;
    auto add = [&](const char * name, int k, int n, bool norm = false) {
        const bool selected = (mode == "unsupported" ? std::strcmp(name,"output.weight") == 0 : std::strcmp(name,"blk.0.ffn_up.weight") == 0);
        const ggml_type type = sinq && selected ? GGML_TYPE_Q3_K : GGML_TYPE_F32;
        ggml_tensor * w = n == 1 ? ggml_new_tensor_1d(ctx,type,k) : ggml_new_tensor_2d(ctx,type,k,n);
        ggml_set_name(w,name);
        payloads.emplace_back(ggml_nbytes(w)); auto & data = payloads.back();
        std::vector<float> f(k*n);
        for (size_t i = 0; i < f.size(); ++i) { f[i] = norm ? 1.f : .02f*std::sin(float(i)*.71f); }
        std::vector<ggml_fp16_t> scales(k,ggml_fp32_to_fp16(1.f));
        if (mode == "valid" && selected) { scales[0]=ggml_fp32_to_fp16(2.f); }
        if (type == GGML_TYPE_F32) { std::memcpy(data.data(),f.data(),data.size()); }
        else { check(ggml_quantize_chunk(type,f.data(),data.data(),0,n,k,nullptr)==data.size(),"ordinary Q3_K fixture encode"); }
        w->data = data.data(); gguf_add_tensor(file,w);
        if (selected && (sinq || mode == "orphan") && mode != "missing_scale") {
            const int size = mode == "scale_shape" ? k-1 : k;
            const auto scale_type = mode == "scale_dtype" ? GGML_TYPE_BF16 : GGML_TYPE_F16;
            auto * t = ggml_new_tensor_1d(ctx,scale_type,size);
            ggml_set_name(t,(std::string(name)+".sinq_scale2").c_str());
            if (mode == "scale_nan") { scales[0] = ggml_fp32_to_fp16(NAN); }
            if (mode == "scale_zero") { scales[0] = ggml_fp32_to_fp16(0); }
            if (mode == "scale_negative") { scales[0] = ggml_fp32_to_fp16(-1); }
            payloads.emplace_back(ggml_nbytes(t)); auto & aux = payloads.back();
            std::memcpy(aux.data(),scales.data(),aux.size()); t->data = aux.data(); gguf_add_tensor(file,t);
        }
    };
    add("token_embd.weight",256,32); add("output_norm.weight",256,1,true); add("output.weight",256,32);
    add("blk.0.attn_norm.weight",256,1,true); add("blk.0.post_attention_norm.weight",256,1,true);
    add("blk.0.attn_q.weight",256,512); add("blk.0.attn_k.weight",256,128); add("blk.0.attn_v.weight",256,128); add("blk.0.attn_output.weight",256,256);
    add("blk.0.attn_q_norm.weight",128,1,true); add("blk.0.attn_k_norm.weight",128,1,true);
    add("blk.0.ffn_gate.weight",256,512); add("blk.0.ffn_up.weight",incompatible_shape ? 96 : 256,512); add("blk.0.ffn_down.weight",512,256);
    check(gguf_write_to_file(file,path.c_str(),false),"fixture write");
    gguf_free(file); ggml_free(ctx);
}
static llama_model * load(const std::string & path, llama_split_mode split = LLAMA_SPLIT_MODE_NONE, bool check_tensors = true) {
    auto p = llama_model_default_params(); p.n_gpu_layers = 0; p.split_mode = split; p.check_tensors = check_tensors;
    logs.clear(); return llama_model_load_from_file(path.c_str(),p);
}
static std::string split_source(const std::string & source, const std::filesystem::path & dir, bool misplace_scale = false, bool unsupported_metadata = false) {
    ggml_context * data = nullptr;
    gguf_init_params init = {false,&data};
    auto * input = gguf_init_from_file(source.c_str(),init);
    check(input != nullptr && data != nullptr,"split source read");
    const int count = int(gguf_get_n_tensors(input)), parts = (count+3)/4;
    std::string first;
    for (int part = 0; part < parts; ++part) {
        auto * output = gguf_init_empty(); gguf_set_kv(output,input);
        gguf_set_val_u16(output,"split.no",uint16_t(part));
        gguf_set_val_u16(output,"split.count",uint16_t(parts));
        gguf_set_val_i32(output,"split.tensors.count",count);
        if (unsupported_metadata && part == 1) { gguf_set_val_bool(output,"beellama.qk_sinq.kurtosis_gate",false); }
        for (int i = 0; i < count; ++i) {
            const char * name = gguf_get_tensor_name(input,i);
            const int destination = misplace_scale && std::strstr(name,".sinq_scale2") ? 0 : i/4;
            if (destination == part) { gguf_add_tensor(output,ggml_get_tensor(data,name)); }
        }
        char filename[80]; std::snprintf(filename,sizeof(filename),"%s-%05d-of-%05d.gguf",unsupported_metadata ? "unsupported-source" : misplace_scale ? "misplaced-source" : "split-source",part+1,parts);
        const auto path = (dir/filename).string();
        check(gguf_write_to_file(output,path.c_str(),false),"split fixture write");
        if (part == 0) { first = path; }
        gguf_free(output);
    }
    gguf_free(input); ggml_free(data); return first;
}
static void full_coverage_pairs(const std::string & source, const std::filesystem::path & dir) {
    for (auto format : {LLAMA_FTYPE_MOSTLY_Q3_K_M,LLAMA_FTYPE_MOSTLY_Q4_K_M}) {
        const auto type = format==LLAMA_FTYPE_MOSTLY_Q3_K_M ? GGML_TYPE_Q3_K : GGML_TYPE_Q4_K;
        auto p = llama_model_quantize_default_params();
        p.ftype = format; p.pure = true; p.nthread = 2;
        p.output_tensor_type = type; p.token_embedding_type = type;
        const auto stock = (dir/("full-stock-"+std::to_string(format)+".gguf")).string();
        check(llama_model_quantize(source.c_str(),stock.c_str(),&p)==0,"full pure stock quantization");
        ggml_context * base_data = nullptr;
        auto * base = gguf_init_from_file(stock.c_str(),{false,&base_data});
        check(base && base_data,"full pure stock metadata/data read");
        const auto identity = stock+".identity.gguf";
        const auto treatment = stock+".conditioned.gguf";
        for (auto mode : {LLAMA_SINQ_IDENTITY,LLAMA_SINQ_SIDECAR}) {
            p.sinq_precondition = mode;
            const auto & path = mode==LLAMA_SINQ_IDENTITY ? identity : treatment;
            check(llama_model_quantize(source.c_str(),path.c_str(),&p)==0,"full pure sidecar quantization");
            ggml_context * data = nullptr;
            auto * metadata = gguf_init_from_file(path.c_str(),{false,&data});
            check(metadata && data,"full pure sidecar metadata/data read");
            int matrices = 0, scales = 0;
            for (int64_t i = 0; i < gguf_get_n_tensors(base); ++i) {
                const char * name = gguf_get_tensor_name(base,i);
                const auto j = gguf_find_tensor(metadata,name);
                check(j>=0,"full pure sidecar lost base tensor");
                const bool matrix = gguf_get_tensor_ne(base,i)[1]>1;
                const bool conditioned = matrix && std::strncmp(name,"blk.",4)==0;
                if (matrix) {
                    ++matrices;
                    if (gguf_get_tensor_type(metadata,j)!=type) {
                        std::fprintf(stderr,"full coverage %s requested=%s actual=%s\n",name,ggml_type_name(type),ggml_type_name(gguf_get_tensor_type(metadata,j)));
                    }
                    check(gguf_get_tensor_type(base,i)==type && gguf_get_tensor_type(metadata,j)==type,
                          "full pure mode retained/promoted an embedding, output or projection");
                }
                const bool has_scale = gguf_find_tensor(metadata,(std::string(name)+".sinq_scale2").c_str())>=0;
                check(has_scale==conditioned,"full quantization must not extend conditioning to embedding/output");
                scales += has_scale;
                if (mode==LLAMA_SINQ_IDENTITY || !conditioned) {
                    auto * a = ggml_get_tensor(base_data,name); auto * b = ggml_get_tensor(data,name);
                    check(a && b && ggml_nbytes(a)==ggml_nbytes(b) &&
                          std::memcmp(a->data,b->data,ggml_nbytes(a))==0,"full pure unconditioned/identity payload changed");
                }
            }
            check(matrices==9 && scales==7,"full fixture must quantize both heads and seven projections");
            gguf_free(metadata); ggml_free(data);
            auto * model = load(path); check(model!=nullptr,"full pure sidecar reload"); llama_model_free(model);
        }
        check(std::filesystem::file_size(identity)==std::filesystem::file_size(treatment),"full identity/treatment byte budget differs");
        gguf_free(base); ggml_free(base_data);
    }
}

static void native_codec_pairs(const std::string & source, const std::filesystem::path & dir) {
    const std::pair<llama_ftype,ggml_type> codecs[] = {
        {LLAMA_FTYPE_MOSTLY_IQ4_XS,GGML_TYPE_IQ4_XS}, {LLAMA_FTYPE_MOSTLY_IQ1_S,GGML_TYPE_IQ1_S},
        {LLAMA_FTYPE_MOSTLY_IQ1_M,GGML_TYPE_IQ1_M}, {LLAMA_FTYPE_MOSTLY_IQ2_XXS,GGML_TYPE_IQ2_XXS},
        {LLAMA_FTYPE_MOSTLY_IQ2_XS,GGML_TYPE_IQ2_XS}, {LLAMA_FTYPE_MOSTLY_IQ2_S,GGML_TYPE_IQ2_S},
        {LLAMA_FTYPE_MOSTLY_IQ3_XXS,GGML_TYPE_IQ3_XXS}, {LLAMA_FTYPE_MOSTLY_IQ3_S,GGML_TYPE_IQ3_S},
        {LLAMA_FTYPE_MOSTLY_IQ4_NL,GGML_TYPE_IQ4_NL}, {LLAMA_FTYPE_MOSTLY_Q5_0,GGML_TYPE_Q5_0},
        {LLAMA_FTYPE_MOSTLY_Q5_1,GGML_TYPE_Q5_1}, {LLAMA_FTYPE_MOSTLY_Q6_K,GGML_TYPE_Q6_K},
        {LLAMA_FTYPE_MOSTLY_Q8_0,GGML_TYPE_Q8_0},
    };
    const char * names[] = {"token_embd.weight","output.weight","blk.0.attn_q.weight","blk.0.attn_k.weight",
        "blk.0.attn_v.weight","blk.0.attn_output.weight","blk.0.ffn_gate.weight","blk.0.ffn_up.weight","blk.0.ffn_down.weight"};
    ggml_context * original_data=nullptr;auto * original_meta=gguf_init_from_file(source.c_str(),{false,&original_data});
    check(original_meta && original_data,"native codec original data read");
    std::vector<std::vector<float>> importance(9);
    std::vector<llama_model_imatrix_data> entries;
    for (int i=0;i<9;++i) {
        importance[i].resize(i==8 ? 512 : 256);
        for (size_t j=0;j<importance[i].size();++j) { importance[i][j]=1.f+.01f*float(j%13); }
        entries.push_back({names[i],importance[i].data(),importance[i].size()});
    }
    entries.push_back({nullptr,nullptr,0});
    for (const auto & codec : codecs) {
        auto p=llama_model_quantize_default_params();p.ftype=codec.first;p.pure=true;p.nthread=2;
        const llama_model_tensor_override overrides[]={{".*",codec.second},{nullptr,GGML_TYPE_COUNT}};
        p.tt_overrides=overrides;
        p.output_tensor_type=codec.second;p.token_embedding_type=codec.second;p.imatrix=entries.data();
        const auto stock=(dir/("codec-"+std::to_string(codec.second)+"-stock.gguf")).string();
        check(llama_model_quantize(source.c_str(),stock.c_str(),&p)==0,"native codec stock production");
        ggml_context * stock_data=nullptr;auto * stock_meta=gguf_init_from_file(stock.c_str(),{false,&stock_data});
        check(stock_meta && stock_data,"native codec stock read");
        std::string identity,treatment;
        for (auto mode : {LLAMA_SINQ_IDENTITY,LLAMA_SINQ_SIDECAR}) {
            p.sinq_precondition=mode;
            const auto path=stock+(mode==LLAMA_SINQ_IDENTITY ? ".identity.gguf" : ".sinq.gguf");
            if(mode==LLAMA_SINQ_IDENTITY)identity=path;else treatment=path;
            if(llama_model_quantize(source.c_str(),path.c_str(),&p)!=0) {
                std::fprintf(stderr,"native codec companion failed: %s\n",ggml_type_name(codec.second));
                check(false,"native codec identity/treatment producer unsupported");
            }
            ggml_context * data=nullptr;auto * meta=gguf_init_from_file(path.c_str(),{false,&data});
            check(meta && data,"native codec companion read");int scales=0;
            for(int64_t i=0;i<gguf_get_n_tensors(stock_meta);++i) {
                const char * name=gguf_get_tensor_name(stock_meta,i);auto * a=ggml_get_tensor(stock_data,name);auto * b=ggml_get_tensor(data,name);
                check(a && b && a->type==b->type,"native codec changed base type map");
                const bool matrix=a->ne[1]>1;const bool conditioned=matrix && std::strncmp(name,"blk.",4)==0;
                if(matrix && a->type!=codec.second) {
                    std::fprintf(stderr,"native codec requested=%s tensor=%s actual=%s\n",ggml_type_name(codec.second),name,ggml_type_name(a->type));
                    check(false,"native codec type promotion");
                }
                const bool companion=gguf_find_tensor(meta,(std::string(name)+".sinq_scale2").c_str())>=0;
                check(companion==conditioned,"native codec projection association mismatch");scales+=companion;
                if(mode==LLAMA_SINQ_IDENTITY || !conditioned)check(ggml_nbytes(a)==ggml_nbytes(b) && std::memcmp(a->data,b->data,ggml_nbytes(a))==0,"native codec identity/unconditioned payload changed");
                if(mode==LLAMA_SINQ_SIDECAR && conditioned) {
                    auto * original=ggml_get_tensor(original_data,name);
                    auto * diagonal=ggml_get_tensor(data,(std::string(name)+".sinq_scale2").c_str());
                    check(original && original->type==GGML_TYPE_F32 && diagonal,"native codec independent input source");
                    const int64_t k=original->ne[0],n=original->ne[1];size_t index=0;
                    while(index<9 && std::strcmp(names[index],name)!=0)++index;
                    check(index<9,"native codec independent importance mapping");
                    const auto * t=static_cast<const ggml_fp16_t *>(diagonal->data);
                    const auto * w=static_cast<const float *>(original->data);
                    std::vector<float> divided(size_t(k*n)),transformed(size_t(k),0.f);
                    for(int64_t j=0;j<k;++j) {
                        const float factor=ggml_fp16_to_fp32(t[j]);
                        transformed[j]=importance[index][j]*factor*factor;
                        for(int64_t i=0;i<n;++i)divided[size_t(i*k+j)]=w[i*k+j]/factor;
                    }
                    std::vector<uint8_t> expected(ggml_nbytes(b));
                    check(ggml_quantize_chunk(codec.second,divided.data(),expected.data(),0,n,k,transformed.data())==expected.size(),"independent native encoder size");
                    check(std::memcmp(expected.data(),b->data,expected.size())==0,"producer differs from native codec on original W/t and transformed t^2 I");
                }
            }
            check(scales==7,"native codec missing projection scales");gguf_free(meta);ggml_free(data);
            auto * model=load(path);check(model!=nullptr,"native codec companion loader unsupported");llama_model_free(model);
        }
        check(std::filesystem::file_size(identity)==std::filesystem::file_size(treatment),"native codec identity/treatment byte budget differs");
        gguf_free(stock_meta);ggml_free(stock_data);
        if(codec.second==GGML_TYPE_IQ2_XS) {
            // A lookup has no linear activation-channel moment. Its required
            // native weighting adapter must equal explicit uniform weights.
            std::vector<float> uniform(256,1.f);
            auto explicit_entries=entries;explicit_entries[0].data=uniform.data();
            p.sinq_precondition=LLAMA_SINQ_NONE;p.imatrix=explicit_entries.data();
            const auto expected=stock+".lookup-uniform.gguf";
            const auto observed=stock+".lookup-no-linear-moment.gguf";
            check(llama_model_quantize(source.c_str(),expected.c_str(),&p)==0,"explicit lookup weights");
            p.imatrix=entries.data()+1;
            check(llama_model_quantize(source.c_str(),observed.c_str(),&p)==0,"native lookup requires codec-neutral uniform importance fallback");
            std::ifstream a(expected,std::ios::binary),b(observed,std::ios::binary);
            const std::vector<char> first((std::istreambuf_iterator<char>(a)),{}),second((std::istreambuf_iterator<char>(b)),{});
            check(first==second,"lookup importance fallback differs from explicit uniform native encoding");
        }
    }
    gguf_free(original_meta);ggml_free(original_data);
}

static std::vector<char> file_bytes(const std::string & path) {
    std::ifstream file(path,std::ios::binary);
    check(bool(file),"open output for byte comparison");
    return std::vector<char>((std::istreambuf_iterator<char>(file)),{});
}

static void settings_controls(const std::string & source, const std::filesystem::path & dir) {
    const std::vector<std::string> ffn = {"blk.0.ffn_down.weight","blk.0.ffn_gate.weight","blk.0.ffn_up.weight"};
    const std::vector<std::string> attention = {"blk.0.attn_k.weight","blk.0.attn_output.weight","blk.0.attn_q.weight","blk.0.attn_v.weight"};
    auto all = ffn; all.insert(all.end(),attention.begin(),attention.end());
    auto write_list = [&](const std::string & name, const std::vector<std::string> & names) {
        const auto path=(dir/name).string(); std::ofstream out(path,std::ios::binary);
        for (const auto & tensor : names) { out << tensor << '\n'; }
        out.close(); check(bool(out),"tensor-list fixture write"); return path;
    };
    const auto ffn_path=write_list("ffn.txt",ffn), attention_path=write_list("attention.txt",attention), all_path=write_list("all.txt",all);
    auto p=llama_model_quantize_default_params(); p.ftype=LLAMA_FTYPE_MOSTLY_Q3_K_M; p.pure=true; p.nthread=2;
    check(p.sinq_strength==1.f && !p.sinq_tensor_list && !p.sinq_diagnostics_json,"new API defaults");
    const auto stock=(dir/"settings-stock.gguf").string();
    const auto stock_json=(dir/"settings-stock.json").string(); p.sinq_diagnostics_json=stock_json.c_str();
    check(llama_model_quantize(source.c_str(),stock.c_str(),&p)==0,"stock diagnostics quantization");
    check(!file_bytes(stock_json).empty(),"stock diagnostics missing"); p.sinq_diagnostics_json=nullptr;
    const auto stock_plain=stock+".plain.gguf";
    check(llama_model_quantize(source.c_str(),stock_plain.c_str(),&p)==0 && file_bytes(stock)==file_bytes(stock_plain),"diagnostics changed stock GGUF");
    ggml_context * stock_data=nullptr; auto * stock_meta=gguf_init_from_file(stock.c_str(),{false,&stock_data});
    check(stock_meta && stock_data,"settings stock read");
    check(gguf_find_key(stock_meta,"beellama.qk_sinq.version")<0,"stock gained SINQ metadata");
    p.sinq_precondition=LLAMA_SINQ_SIDECAR;
    const auto legacy=(dir/"settings-legacy.gguf").string();
    check(llama_model_quantize(source.c_str(),legacy.c_str(),&p)==0,"legacy defaults production");
    const auto explicit_full=legacy+".full.gguf"; p.sinq_tensor_list=all_path.c_str(); p.sinq_strength=1.f;
    check(llama_model_quantize(source.c_str(),explicit_full.c_str(),&p)==0,"explicit full mask strength one");
    check(file_bytes(legacy)==file_bytes(explicit_full),"full exact mask strength one changed legacy bytes");
    for (int partition=0;partition<2;++partition) {
        const auto & names=partition==0 ? ffn : attention;
        p.sinq_tensor_list=partition==0 ? ffn_path.c_str() : attention_path.c_str();
        const auto output=(dir/("settings-mask-"+std::to_string(partition)+".gguf")).string();
        const auto report=output+".json"; p.sinq_diagnostics_json=report.c_str();
        check(llama_model_quantize(source.c_str(),output.c_str(),&p)==0,"exact-name mask quantization");
        const auto first_report=file_bytes(report);
        const auto diagnostic=nlohmann::json::parse(first_report.begin(),first_report.end());
        check(diagnostic.at("totals").at("eligible_tensors")==7 && diagnostic.at("totals").at("included_tensors")==names.size(),"diagnostic mask totals");
        check(diagnostic.at("tensor_list_sha256").get<std::string>().size()==64,"normalized mask digest missing");
        std::string previous;
        for (const auto & record : diagnostic.at("tensors")) {
            const auto name=record.at("name").get<std::string>();
            check(previous.empty() || previous<name,"diagnostics not sorted by exact name"); previous=name;
            const bool selected=std::find(names.begin(),names.end(),name)!=names.end();
            check(record.at("included")==selected && record.at("qtype")==ggml_type_name(GGML_TYPE_Q3_K),"diagnostics selection or actual codec");
            if (selected) {
                check(record.at("statistics_status")=="computed" && record.at("strips").is_array() &&
                      record.at("scales_sha256").get<std::string>().size()==64 && record.at("companion_bytes").get<size_t>()>0,"included diagnostics incomplete");
            } else {
                check(record.at("reason")=="not_in_tensor_list" && record.at("statistics_status")=="not_computed" &&
                      record.at("strips").is_null() && record.at("scales_sha256").is_null() && record.at("companion_bytes")==0,"excluded diagnostics invented statistics");
            }
        }
        auto reversed=names; std::reverse(reversed.begin(),reversed.end());
        const auto reversed_path=write_list("reversed-mask.txt",reversed); p.sinq_tensor_list=reversed_path.c_str();
        check(llama_model_quantize(source.c_str(),output.c_str(),&p)==0 && first_report==file_bytes(report),"diagnostics not deterministic");
        ggml_context * data=nullptr; auto * meta=gguf_init_from_file(output.c_str(),{false,&data});
        check(meta && data,"mask output read"); int companions=0;
        for (int64_t i=0;i<gguf_get_n_tensors(stock_meta);++i) {
            const char * name=gguf_get_tensor_name(stock_meta,i);
            auto * a=ggml_get_tensor(stock_data,name); auto * b=ggml_get_tensor(data,name);
            check(a && b && a->type==b->type,"mask changed stock allocation");
            const bool selected=std::find(names.begin(),names.end(),name)!=names.end();
            const bool has_scale=gguf_find_tensor(meta,(std::string(name)+".sinq_scale2").c_str())>=0;
            check(has_scale==selected,"mask selection not exact"); companions+=has_scale;
            if (!selected) { check(ggml_nbytes(a)==ggml_nbytes(b) && std::memcmp(a->data,b->data,ggml_nbytes(a))==0,"mask excluded payload differs from stock"); }
        }
        check(companions==int(names.size()),"mask complement companion count");
        gguf_free(meta); ggml_free(data);
    }
    p.sinq_tensor_list=nullptr; p.sinq_diagnostics_json=nullptr; p.sinq_strength=0.f;
    const auto zero=legacy+".zero.gguf", identity=legacy+".identity.gguf";
    check(llama_model_quantize(source.c_str(),zero.c_str(),&p)==0,"strength zero production");
    p.sinq_precondition=LLAMA_SINQ_IDENTITY; p.sinq_strength=1.f;
    check(llama_model_quantize(source.c_str(),identity.c_str(),&p)==0,"legacy identity production");
    ggml_context * zero_data=nullptr, * identity_data=nullptr;
    auto * zero_meta=gguf_init_from_file(zero.c_str(),{false,&zero_data});
    auto * identity_meta=gguf_init_from_file(identity.c_str(),{false,&identity_data});
    check(zero_meta && identity_meta && zero_data && identity_data,"identity output read");
    for (int64_t i=0;i<gguf_get_n_tensors(zero_meta);++i) {
        const char * name=gguf_get_tensor_name(zero_meta,i);
        auto * a=ggml_get_tensor(zero_data,name); auto * b=ggml_get_tensor(identity_data,name);
        check(a && b && a->type==b->type && ggml_nbytes(a)==ggml_nbytes(b) && std::memcmp(a->data,b->data,ggml_nbytes(a))==0,"strength zero differs from identity payload");
        if (auto * stock_tensor=ggml_get_tensor(stock_data,name)) {
            check(ggml_nbytes(a)==ggml_nbytes(stock_tensor) && std::memcmp(a->data,stock_tensor->data,ggml_nbytes(a))==0,"strength zero differs from stock payload");
        }
    }
    gguf_free(zero_meta); gguf_free(identity_meta); ggml_free(zero_data); ggml_free(identity_data);
    gguf_free(stock_meta); ggml_free(stock_data);
    p.sinq_precondition=LLAMA_SINQ_SIDECAR;
    const auto rejected=(dir/"settings-rejected.gguf").string();
    auto reject=[&](const char * message) { check(llama_model_quantize(source.c_str(),rejected.c_str(),&p)!=0,message); };
    for (float value : {-1.f,1.01f,NAN,INFINITY}) { p.sinq_strength=value; reject("bad API strength accepted"); }
    p.sinq_strength=1.f;
    const std::vector<std::vector<std::string>> invalid_lists={ {}, {""}, {ffn[0],ffn[0]}, {"blk.99.ffn_up.weight"},
        {"output.weight"}, {"token_embd.weight"}, {"blk.0.attn_norm.weight"}, {"blk.*.ffn_up.weight"}, {" blk.0.ffn_up.weight"}, {"\xff"} };
    for (size_t i=0;i<invalid_lists.size();++i) {
        const auto path=write_list("bad-mask-"+std::to_string(i)+".txt",invalid_lists[i]); p.sinq_tensor_list=path.c_str(); reject("bad mask accepted");
    }
    const auto missing=(dir/"nonexistent-mask.txt").string(); p.sinq_tensor_list=missing.c_str(); reject("missing mask accepted");
    p.sinq_tensor_list=ffn_path.c_str();
    const llama_model_tensor_override unsupported[]={{".*",GGML_TYPE_F32},{nullptr,GGML_TYPE_COUNT}};
    p.tt_overrides=unsupported; reject("explicit unsupported codec accepted"); p.tt_overrides=nullptr;
    p.sinq_coverage=true; reject("mask with historical allocation accepted"); p.sinq_coverage=false;
    p.sinq_precondition=LLAMA_SINQ_NONE; reject("mask without conditioning accepted");
    p.sinq_tensor_list=nullptr; p.sinq_strength=.5f; reject("strength without conditioning accepted");
    p.sinq_precondition=LLAMA_SINQ_IDENTITY; reject("nondefault identity strength accepted");
    p.sinq_precondition=LLAMA_SINQ_SIDECAR; p.sinq_strength=1.f;
    p.sinq_strip_width=32; reject("bad strip width accepted"); p.sinq_strip_width=64;
    p.sinq_iterations=3; reject("bad iterations accepted"); p.sinq_iterations=16;
    p.only_copy=true; reject("conditioning copy accepted"); p.only_copy=false;
    const auto bad_report=(dir/"missing-report-directory"/"report.json").string();
    p.sinq_diagnostics_json=bad_report.c_str(); reject("diagnostics write error ignored");
    p.sinq_diagnostics_json=source.c_str(); reject("diagnostics overwrite of source accepted");
    p.sinq_diagnostics_json=stock_json.c_str(); p.dry_run=true; reject("dry-run diagnostics accepted");
    p.dry_run=false; p.sinq_diagnostics_json=nullptr;
    const auto malformed=(dir/"settings-incompatible-shape.gguf").string(); make_model(malformed,"source",true);
    const auto up_only=write_list("up-only.txt",{"blk.0.ffn_up.weight"}); p.sinq_tensor_list=up_only.c_str();
    check(llama_model_quantize(malformed.c_str(),rejected.c_str(),&p)!=0,"explicit selection silently changed codec for incompatible shape");
    p.ftype=LLAMA_FTYPE_MOSTLY_Q8_0;
    check(llama_model_quantize(malformed.c_str(),rejected.c_str(),&p)!=0,"explicit strip-incompatible shape accepted");
}

int main(int argc, char ** argv) try {
    if (argc == 3 && std::strcmp(argv[1],"--load-only") == 0) {
        llama_log_set(logger,nullptr); llama_backend_init();
        auto * model = load(argv[2]); if (!model) { std::fprintf(stderr,"%s",logs.c_str()); }
        check(model != nullptr,"external fixture load"); llama_model_free(model); llama_backend_free();
        std::puts("SINQ external fixture load passed"); return 0;
    }
    check(argc == 2,"provide a dedicated temporary directory");
    const auto dir = std::filesystem::path(argv[1]); std::filesystem::create_directories(dir);
    llama_log_set(logger,nullptr); llama_backend_init();
    const auto source = (dir/"source.gguf").string(); make_model(source,"source");
    auto * ordinary = load(source); if (!ordinary) { std::fprintf(stderr,"%s",logs.c_str()); }
    check(ordinary != nullptr,"ordinary fixture load"); llama_model_free(ordinary);
    const llama_ftype formats[] = {LLAMA_FTYPE_MOSTLY_Q3_K_M,LLAMA_FTYPE_MOSTLY_Q4_K_M};
    for (auto format : formats) {
        const auto output = (dir/("quantized-"+std::to_string(format)+".gguf")).string();
        auto p = llama_model_quantize_default_params(); p.ftype = format; p.pure = true; p.nthread = 2;
        p.sinq_precondition = LLAMA_SINQ_SIDECAR; p.sinq_strip_width = 64; p.sinq_iterations = 16;
        check(llama_model_quantize(source.c_str(),output.c_str(),&p) == 0,"standard QK model precondition/quantize");
        auto * metadata = gguf_init_from_file(output.c_str(),{true,nullptr});
        check(metadata != nullptr,"read quantized metadata");
        const auto scale = gguf_find_tensor(metadata,"blk.0.ffn_up.weight.sinq_scale2");
        check(scale >= 0,"quantizer did not produce required QK companion");
        check(gguf_get_tensor_type(metadata,gguf_find_tensor(metadata,"blk.0.ffn_up.weight")) == (format==LLAMA_FTYPE_MOSTLY_Q3_K_M ? GGML_TYPE_Q3_K : GGML_TYPE_Q4_K),"not an ordinary QK wire type");
        check(gguf_find_key(metadata,"beellama.qk_sinq.version")>=0,"new version namespace missing");
        gguf_free(metadata);
        auto * model = load(output); if (!model) { std::fprintf(stderr,"%s",logs.c_str()); }
        check(model != nullptr,"QK companion fixture load"); llama_model_free(model);
        const auto chunked = output+".chunked.gguf";
        p.max_buf_size = 4096;
        check(llama_model_quantize(source.c_str(),chunked.c_str(),&p) == 0,"small-slab quantization");
        std::ifstream full(output,std::ios::binary), slabs(chunked,std::ios::binary);
        const std::vector<char> full_bytes((std::istreambuf_iterator<char>(full)),{});
        const std::vector<char> slab_bytes((std::istreambuf_iterator<char>(slabs)),{});
        check(full_bytes == slab_bytes,"slab boundaries changed normalization, native encoding or GGUF bytes");
        std::printf("standard QK quantize/reload/slab identity format=%d OK\n",format);
    }
    settings_controls(source,dir);
    full_coverage_pairs(source,dir);
    native_codec_pairs(source,dir);
    // Conditioning must preserve the existing S/M/L allocator, including
    // conventional types that do not receive a channel companion.
    for (auto format : {LLAMA_FTYPE_MOSTLY_Q3_K_S,LLAMA_FTYPE_MOSTLY_Q3_K_M,
                        LLAMA_FTYPE_MOSTLY_Q3_K_L,LLAMA_FTYPE_MOSTLY_Q4_K_S,LLAMA_FTYPE_MOSTLY_Q4_K_M}) {
        const auto standard = (dir/("preset-standard-"+std::to_string(format)+".gguf")).string();
        const auto normalized = (dir/("preset-normalized-"+std::to_string(format)+".gguf")).string();
        auto p = llama_model_quantize_default_params(); p.ftype = format; p.nthread = 2;
        // Historical reproduction only: this option changes stock allocation
        // by preserving noneligible source tensors. It is not a SINQ-only mask.
        p.sinq_coverage = true;
        check(!p.pure,"mixed preset unexpectedly defaults to pure");
        check(llama_model_quantize(source.c_str(),standard.c_str(),&p)==0,"ordinary mixed preset quantize");
        p.sinq_precondition = LLAMA_SINQ_SIDECAR;
        check(llama_model_quantize(source.c_str(),normalized.c_str(),&p)==0,"conditioned mixed preset quantize");
        ggml_context * base_data = nullptr, * conditioned_data = nullptr;
        auto * base = gguf_init_from_file(standard.c_str(),{false,&base_data});
        auto * conditioned = gguf_init_from_file(normalized.c_str(),{false,&conditioned_data});
        check(base && conditioned && base_data && conditioned_data,"mixed preset metadata/data read");
        check(gguf_get_val_u32(base,gguf_find_key(base,"general.file_type")) ==
              gguf_get_val_u32(conditioned,gguf_find_key(conditioned,"general.file_type")),"conditioning changed conventional ftype");
        int companions = 0;
        for (int64_t i = 0; i < gguf_get_n_tensors(base); ++i) {
            const char * name = gguf_get_tensor_name(base,i);
            const int64_t j = gguf_find_tensor(conditioned,name);
            check(j>=0 && gguf_get_tensor_type(base,i)==gguf_get_tensor_type(conditioned,j),"conditioning changed mixed preset type allocation");
            const auto type = gguf_get_tensor_type(base,i);
            const bool eligible = std::strncmp(name,"blk.",4)==0 && (type==GGML_TYPE_Q3_K || type==GGML_TYPE_Q4_K || type==GGML_TYPE_Q6_K);
            const bool has_scale = gguf_find_tensor(conditioned,(std::string(name)+".sinq_scale2").c_str())>=0;
            check(has_scale==eligible,"mixed preset companion coverage does not match audited Q3/Q4/Q6 allocation");
            if (!eligible) {
                auto * a = ggml_get_tensor(base_data,name);
                auto * b = ggml_get_tensor(conditioned_data,name);
                check(a && b && ggml_nbytes(a)==ggml_nbytes(b) &&
                      std::memcmp(a->data,b->data,ggml_nbytes(a))==0,"conditioning changed an unassociated mixed-preset payload");
            }
            companions += has_scale;
        }
        check(companions>0,"mixed preset fixture did not exercise conditioning");
        gguf_free(base); gguf_free(conditioned);
        ggml_free(base_data); ggml_free(conditioned_data);
        auto * model = load(normalized);
        check(model!=nullptr,"conditioned mixed preset reload");
        llama_model_free(model);
        std::printf("mixed preset type allocation/companion/reload format=%d OK\n",format);
    }
    const auto mixed = (dir/"mixed-standard-types.gguf").string();
    auto mixed_params = llama_model_quantize_default_params();
    mixed_params.ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;
    mixed_params.pure = true;
    mixed_params.sinq_precondition = LLAMA_SINQ_SIDECAR;
    mixed_params.nthread = 2;
    const llama_model_tensor_override overrides[] = {
        {"blk\\.0\\.ffn_up\\.weight",GGML_TYPE_F32},
        {"blk\\.0\\.ffn_gate\\.weight",GGML_TYPE_Q4_K},
        {nullptr,GGML_TYPE_COUNT},
    };
    mixed_params.tt_overrides = overrides;
    check(llama_model_quantize(source.c_str(),mixed.c_str(),&mixed_params)==0,"mixed QK and floating tensor selection");
    auto * mixed_metadata=gguf_init_from_file(mixed.c_str(),{true,nullptr});
    check(mixed_metadata!=nullptr,"mixed metadata read");
    check(gguf_get_tensor_type(mixed_metadata,gguf_find_tensor(mixed_metadata,"blk.0.ffn_up.weight"))==GGML_TYPE_F32 &&
          gguf_find_tensor(mixed_metadata,"blk.0.ffn_up.weight.sinq_scale2")<0,"floating override gained a companion");
    check(gguf_get_tensor_type(mixed_metadata,gguf_find_tensor(mixed_metadata,"blk.0.ffn_gate.weight"))==GGML_TYPE_Q4_K &&
          gguf_find_tensor(mixed_metadata,"blk.0.ffn_gate.weight.sinq_scale2")>=0,"Q4 override lost its companion");
    gguf_free(mixed_metadata);
    auto * mixed_model=load(mixed);
    check(mixed_model!=nullptr,"mixed standard-type model reload");
    llama_model_free(mixed_model);
    const auto first_split = split_source(source,dir);
    const auto split_prefix = (dir/"split-quantized").string();
    auto split_params = llama_model_quantize_default_params();
    split_params.ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M; split_params.sinq_precondition=LLAMA_SINQ_SIDECAR; split_params.pure = true; split_params.keep_split = true; split_params.nthread = 2;
    check(llama_model_quantize(first_split.c_str(),split_prefix.c_str(),&split_params) == 0,"keep-split quantization");
    const auto split_output = split_prefix+"-00001-of-00004.gguf";
    auto * split_model = load(split_output); if (!split_model) { std::fprintf(stderr,"%s",logs.c_str()); }
    check(split_model != nullptr,"keep-split model reload"); llama_model_free(split_model);
    std::puts("four-shard quantize/companion/reload OK");
    const std::pair<const char *,const char *> bad[] = {
        {"missing_version","qk_sinq"},{"version_dtype","qk_sinq"},{"version_value","qk_sinq"},
        {"missing_manifest","qk_sinq"},{"duplicate_manifest","duplicate"},
        {"bad_width","normalization recipe"},{"bad_iterations","normalization recipe"},
        {"missing_scale","companion"},{"orphan","companion"},{"scale_shape","FP16"},{"scale_dtype","FP16"},
        {"scale_nan","non-finite or non-positive"},{"scale_zero","non-finite or non-positive"},{"scale_negative","non-finite or non-positive"},
        {"unsupported","unsupported tensor"},{"hadamard","Hadamard"},
        {"legacy_enabled","unsupported SINQ metadata"},{"legacy_disabled","unsupported SINQ metadata"},
        {"legacy_mistyped","unsupported SINQ metadata"},{"unknown_metadata","unsupported SINQ metadata"},
        {"metadata_only","unsupported SINQ metadata"},
    };
    for (const auto & entry : bad) {
        const auto path = (dir/(std::string(entry.first)+".gguf")).string(); make_model(path,entry.first);
        auto * model = load(path); if (model) { llama_model_free(model); }
        if (model || logs.find(entry.second) == std::string::npos) { std::fprintf(stderr,"%s",logs.c_str()); }
        check(!model && logs.find(entry.second) != std::string::npos,"malformed fixture accepted or wrong error");
        std::printf("malformed %s rejected OK\n",entry.first);
    }
    // Unknown transform metadata must not leak through stock COPY or quantization.
    for (bool copy : {false,true}) {
        auto p=llama_model_quantize_default_params(); p.only_copy=copy;
        check(llama_model_quantize((dir/"metadata_only.gguf").string().c_str(),
            (dir/"metadata-rejected.gguf").string().c_str(),&p)!=0,"unsupported metadata accepted without version");
    }
    // Required companion validation also runs without the optional general
    // tensor checker, which catches NaN earlier when it is enabled above.
    auto * nan_without_check = load((dir/"scale_nan.gguf").string(),LLAMA_SPLIT_MODE_NONE,false);
    if (nan_without_check) { llama_model_free(nan_without_check); }
    check(!nan_without_check && logs.find("non-finite or non-positive") != std::string::npos,"NaN side scale must fail without optional tensor checks");
    const auto valid = (dir/"valid.gguf").string(); make_model(valid,"valid");
    const auto unsupported_split = split_source(valid,dir,false,true);
    auto * unsupported_model = load(unsupported_split);
    if (unsupported_model) { llama_model_free(unsupported_model); }
    check(!unsupported_model && logs.find("unsupported SINQ metadata") != std::string::npos,"unsupported metadata accepted in secondary shard");
    auto reject_copy=llama_model_quantize_default_params(); reject_copy.only_copy=true;
    check(llama_model_quantize(unsupported_split.c_str(),(dir/"rejected-split.gguf").string().c_str(),&reject_copy)!=0,
          "COPY accepted unsupported metadata from secondary shard");
    llama_model_kv_override obsolete_override[2] = {};
    std::strcpy(obsolete_override[0].key,"beellama.qk_sinq.kurtosis_gate");
    obsolete_override[0].tag=LLAMA_KV_OVERRIDE_TYPE_BOOL; obsolete_override[0].val_bool=false;
    auto override_source=llama_model_default_params(); override_source.n_gpu_layers=0; override_source.kv_overrides=obsolete_override;
    auto * overridden=llama_model_load_from_file(source.c_str(),override_source);
    if (overridden) { llama_model_free(overridden); }
    check(!overridden,"unsupported metadata override accepted by loader");
    reject_copy.kv_overrides=obsolete_override;
    check(llama_model_quantize(source.c_str(),(dir/"rejected-override.gguf").string().c_str(),&reject_copy)!=0,
          "COPY accepted unsupported metadata override");
    const auto misplaced = split_source(valid,dir,true);
    auto * misplaced_model = load(misplaced);
    if (misplaced_model) { llama_model_free(misplaced_model); }
    check(!misplaced_model && logs.find("same GGUF shard") != std::string::npos,"cross-shard companion must fail closed");
    auto * split = load(valid,LLAMA_SPLIT_MODE_ROW);
    if (split) { llama_model_free(split); }
    check(!split && logs.find("QK SINQ does not support row/tensor split") != std::string::npos,"row split must fail closed");
    if (auto * cuda = ggml_backend_dev_by_name("CUDA0")) {
        auto override_params = llama_model_default_params();
        override_params.n_gpu_layers = 99;
        override_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        ggml_backend_dev_t devices[] = {cuda,nullptr};
        override_params.devices = devices;
        const llama_model_tensor_buft_override placements[] = {
            {"^blk\\.0\\.ffn_up\\.weight$",ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU))},
            {nullptr,nullptr},
        };
        override_params.tensor_buft_overrides = placements;
        logs.clear();
        auto * conflict = llama_model_load_from_file(valid.c_str(),override_params);
        if (conflict) { llama_model_free(conflict); }
        check(!conflict && logs.find("scale placement must match") != std::string::npos,
              "companion/base placement mismatch was not rejected");
        std::puts("CUDA companion/base placement mismatch rejected");
    }
    auto no_alloc_params = llama_model_default_params();
    no_alloc_params.n_gpu_layers = 0;
    no_alloc_params.no_alloc = true;
    no_alloc_params.load_mode = LLAMA_LOAD_MODE_NONE;
    auto * probe = llama_model_load_from_file(valid.c_str(),no_alloc_params);
    check(probe != nullptr,"no-allocation fit probe with validated companions");
    llama_model_free(probe);
    llama_backend_free(); std::puts("SINQ model tests passed"); return 0;
} catch (const std::exception & e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
