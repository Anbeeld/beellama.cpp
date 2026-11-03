// Optional research validation uses only the existing public model/context APIs.
#include "llama.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
int main(int argc,char**argv) try {
    if(argc<3 || (std::string(argv[1])!="--load-only" && std::string(argv[1])!="--forward"))throw std::runtime_error("usage: llama-sinq-model --load-only MODEL | --forward MODEL TOKEN...");
    llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;
    auto*model=llama_model_load_from_file(argv[2],mp);if(!model)throw std::runtime_error("native model loader rejected hybrid");
    if(std::string(argv[1])=="--load-only"){llama_model_free(model);llama_backend_free();std::cout<<"{\"status\":\"PASS\",\"load\":true}\n";return 0;}
    std::vector<llama_token>tokens;
    const int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
    for(int i=3;i<argc;++i){const int t=std::stoi(argv[i]);if(t<0 || t>=vocab)throw std::runtime_error("token outside vocabulary");tokens.push_back(t);}
    if(tokens.empty() || tokens.size()>32)throw std::runtime_error("forward token budget 1..32");
    auto cp=llama_context_default_params();cp.n_ctx=64;cp.n_batch=32;cp.n_ubatch=24;cp.n_threads=24;cp.n_threads_batch=24;
    auto*ctx=llama_init_from_model(model,cp);if(!ctx)throw std::runtime_error("native context rejected model");
    if(llama_decode(ctx,llama_batch_get_one(tokens.data(),int(tokens.size())))!=0)throw std::runtime_error("native forward failed");
    const auto*logits=llama_get_logits_ith(ctx,-1);if(!logits)throw std::runtime_error("native logits missing");
    std::cout<<"{\"status\":\"PASS\",\"tokens\":"<<tokens.size()<<",\"logits\":[";
    for(int i=0;i<vocab;++i){if(!std::isfinite(logits[i]))throw std::runtime_error("nonfinite native logits");if(i)std::cout<<',';std::cout.precision(9);std::cout<<logits[i];}
    std::cout<<"]}\n";llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}catch(const std::exception&e){std::cerr<<"research model: "<<e.what()<<'\n';return 1;}
