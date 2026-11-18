// Optional optimizer validation uses only public model/context/backend APIs.
#include "llama.h"
#include "ggml-backend.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
int main(int argc,char**argv) try {
    if(argc<3 || (std::string(argv[1])!="--load-only" && std::string(argv[1])!="--forward"))throw std::runtime_error("usage: llama-sinq-model --load-only MODEL | --forward MODEL [--gpu-layers N] [--threads N] TOKEN...");
    const bool forward=std::string(argv[1])=="--forward";
    std::vector<llama_token>tokens;
    int gpu_layers=0,threads=0;
    auto integer=[](const char * text){size_t used=0;const int value=std::stoi(text,&used);if(used!=std::string(text).size())throw std::runtime_error("invalid integer argument");return value;};
    for(int i=3;i<argc;++i){
        if(std::string(argv[i])=="--gpu-layers"){
            if(!forward || gpu_layers || ++i>=argc)throw std::runtime_error("invalid GPU layer option");
            gpu_layers=integer(argv[i]);if(gpu_layers<=0)throw std::runtime_error("GPU layers must be positive");
        }else if(std::string(argv[i])=="--threads"){
            if(!forward || threads || ++i>=argc)throw std::runtime_error("invalid thread option");
            threads=integer(argv[i]);if(threads<=0 || threads>1024)throw std::runtime_error("threads must be in 1..1024");
        }else{if(!forward)throw std::runtime_error("unexpected load-only argument");tokens.push_back(integer(argv[i]));}
    }
    llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=gpu_layers;
    if(gpu_layers && !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU))throw std::runtime_error("requested GPU offload but no GPU backend is available");
    auto*model=llama_model_load_from_file(argv[2],mp);if(!model)throw std::runtime_error("native model loader rejected hybrid");
    if(!forward){llama_model_free(model);llama_backend_free();std::cout<<"{\"status\":\"PASS\",\"load\":true}\n";return 0;}
    const int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
    for(const int t:tokens)if(t<0 || t>=vocab)throw std::runtime_error("token outside vocabulary");
    if(tokens.empty() || tokens.size()>32)throw std::runtime_error("forward token budget 1..32");
    auto cp=llama_context_default_params();cp.n_ctx=64;cp.n_batch=32;cp.n_ubatch=24;
    if(threads)cp.n_threads=cp.n_threads_batch=threads;
    auto*ctx=llama_init_from_model(model,cp);if(!ctx)throw std::runtime_error("native context rejected model");
    if(llama_decode(ctx,llama_batch_get_one(tokens.data(),int(tokens.size())))!=0)throw std::runtime_error("native forward failed");
    const auto*logits=llama_get_logits_ith(ctx,-1);if(!logits)throw std::runtime_error("native logits missing");
    std::cout<<"{\"status\":\"PASS\",\"tokens\":"<<tokens.size()<<",\"logits\":[";
    for(int i=0;i<vocab;++i){if(!std::isfinite(logits[i]))throw std::runtime_error("nonfinite native logits");if(i)std::cout<<',';std::cout.precision(9);std::cout<<logits[i];}
    std::cout<<"]}\n";llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}catch(const std::exception&e){std::cerr<<"optimizer model: "<<e.what()<<'\n';return 1;}
