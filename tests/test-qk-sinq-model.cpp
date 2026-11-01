#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "../src/llama-sinq.h"


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
static void make_model(const std::string & path, const std::string & mode) {
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
    const bool sinq = mode != "source" && mode != "orphan";
    if ((sinq || mode == "orphan") && mode != "missing_version") {
        if (mode == "version_dtype") { gguf_set_val_str(file,"beellama.qk_sinq.version","1"); }
        else { gguf_set_val_u32(file,"beellama.qk_sinq.version",mode == "version_value" ? 2 : 1); }
    }
    if (mode != "source") {
        gguf_set_val_str(file,"beellama.qk_sinq.algorithm","sinkhorn_log");
        gguf_set_val_u32(file,"beellama.qk_sinq.strip_width",mode == "bad_width" ? 32 : 64);
        gguf_set_val_u32(file,"beellama.qk_sinq.iterations",mode == "bad_iterations" ? 3 : 16);
        const char * target = mode == "unsupported" ? "output.weight" : "blk.0.ffn_up.weight";
        const char * targets[] = {target,target};
        if (mode != "missing_manifest") {
            gguf_set_arr_str(file,"beellama.qk_sinq.tensors",targets,mode == "duplicate_manifest" ? 2 : 1);
        }
    }
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
    add("blk.0.ffn_gate.weight",256,512); add("blk.0.ffn_up.weight",256,512); add("blk.0.ffn_down.weight",512,256);
    check(gguf_write_to_file(file,path.c_str(),false),"fixture write");
    gguf_free(file); ggml_free(ctx);
}
static llama_model * load(const std::string & path, llama_split_mode split = LLAMA_SPLIT_MODE_NONE, bool check_tensors = true) {
    auto p = llama_model_default_params(); p.n_gpu_layers = 0; p.split_mode = split; p.check_tensors = check_tensors;
    logs.clear(); return llama_model_load_from_file(path.c_str(),p);
}
static std::string split_source(const std::string & source, const std::filesystem::path & dir, bool misplace_scale = false) {
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
        for (int i = 0; i < count; ++i) {
            const char * name = gguf_get_tensor_name(input,i);
            const int destination = misplace_scale && std::strstr(name,".sinq_scale2") ? 0 : i/4;
            if (destination == part) { gguf_add_tensor(output,ggml_get_tensor(data,name)); }
        }
        char filename[80]; std::snprintf(filename,sizeof(filename),"%s-%05d-of-%05d.gguf",misplace_scale ? "misplaced-source" : "split-source",part+1,parts);
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
            p.sinq_precondition = mode; p.sinq_gate = false;
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
        p.output_tensor_type=codec.second;p.token_embedding_type=codec.second;p.imatrix=entries.data();p.sinq_gate=false;
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
        p.sinq_precondition = LLAMA_SINQ_SIDECAR; p.sinq_strip_width = 64; p.sinq_iterations = 16; p.sinq_gate = false;
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
    full_coverage_pairs(source,dir);
    native_codec_pairs(source,dir);
    // Conditioning must preserve the existing S/M/L allocator, including
    // conventional types that do not receive a channel companion.
    for (auto format : {LLAMA_FTYPE_MOSTLY_Q3_K_S,LLAMA_FTYPE_MOSTLY_Q3_K_M,
                        LLAMA_FTYPE_MOSTLY_Q3_K_L,LLAMA_FTYPE_MOSTLY_Q4_K_S,LLAMA_FTYPE_MOSTLY_Q4_K_M}) {
        const auto standard = (dir/("preset-standard-"+std::to_string(format)+".gguf")).string();
        const auto normalized = (dir/("preset-normalized-"+std::to_string(format)+".gguf")).string();
        auto p = llama_model_quantize_default_params(); p.ftype = format; p.nthread = 2;
        // The production sidecar mode always preserves excluded source tensors;
        // the ordinary control must use that same coverage, not quantize heads.
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
    };
    for (const auto & entry : bad) {
        const auto path = (dir/(std::string(entry.first)+".gguf")).string(); make_model(path,entry.first);
        auto * model = load(path); if (model) { llama_model_free(model); }
        if (model || logs.find(entry.second) == std::string::npos) { std::fprintf(stderr,"%s",logs.c_str()); }
        check(!model && logs.find(entry.second) != std::string::npos,"malformed fixture accepted or wrong error");
        std::printf("malformed %s rejected OK\n",entry.first);
    }
    // Required companion validation also runs without the optional general
    // tensor checker, which catches NaN earlier when it is enabled above.
    auto * nan_without_check = load((dir/"scale_nan.gguf").string(),LLAMA_SPLIT_MODE_NONE,false);
    if (nan_without_check) { llama_model_free(nan_without_check); }
    check(!nan_without_check && logs.find("non-finite or non-positive") != std::string::npos,"NaN side scale must fail without optional tensor checks");
    const auto valid = (dir/"valid.gguf").string(); make_model(valid,"valid");
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
