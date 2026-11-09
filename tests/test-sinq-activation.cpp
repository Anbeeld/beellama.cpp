#include "../src/llama-sinq-activation.h"
#include "../vendor/nlohmann/json.hpp"
extern "C" {
#include "../vendor/hash/sha256/sha256.h"
}
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;
namespace {
void require(bool ok, const std::string & message) { if (!ok) { throw std::runtime_error(message); } }
std::string digest(const std::string & bytes) {
    unsigned char out[32]; sha256_hash(out, reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size());
    std::string result; const char * hex = "0123456789abcdef";
    for (auto x : out) { result += hex[x >> 4]; result += hex[x & 15]; } return result;
}
void put(const fs::path & path, const std::string & bytes) {
    std::ofstream f(path, std::ios::binary); f.write(bytes.data(), bytes.size());
    require(bool(f), "fixture write failed");
}
struct fixture {
    fs::path dir;
    json root;
    const std::string bytes = std::string("\0\0\x80\x3f\0\0\0\xc0", 8);
    fixture() {
        dir = fs::temp_directory_path() / ("sinq-activation-" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
        require(fs::create_directory(dir), "fixture directory creation failed");
        put(dir / "source.gguf", "abc"); put(dir / "samples.bin", bytes);
        const std::string abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        root = {{"schema_version",1},{"architecture","qwen3_5"},{"source_sha256",abc},{"source_revision","fixture"},
            {"sampling",{{"contract","valid-token-prefill-reservoir-v1"},{"seed",0},{"max_rows",2},{"corpus_sha256",abc},{"token_sha256",abc}}},
            {"tensors",json::array()}};
        for (const auto * role : {"attn_q", "ffn_down"}) {
            root["tensors"].push_back({{"name",std::string("blk.0.")+role+".weight"},
                {"site",std::string("qwen3_5.layers.0.")+role+".input"},{"reference_module",std::string("model.layers.0.")+role},
                {"channel_layout","gguf-input"},{"input_dim",2},{"output_dim",3},{"count",1},
                {"dtype","<f4"},{"order","row-major"},{"data_file","samples.bin"},{"sha256",digest(bytes)}});
        }
    }
    ~fixture() { std::error_code ec; fs::remove_all(dir,ec); }
    llama_sinq_activation_manifest load(const json & value, int threads = 1) {
        put(dir / "activations.json",value.dump());
        return llama_sinq_load_activations(dir.string(),(dir / "source.gguf").string(),threads);
    }
    void reject(const json & value, const std::string & label) {
        bool threw = false; try { load(value); } catch (const std::exception &) { threw = true; }
        require(threw,"accepted invalid fixture: " + label);
    }
};
}
int main() {
    try {
        fixture f;
        auto m = f.load(f.root);
        require(m.tensors.size()==2,"valid table missing entries");
        auto parallel=f.load(f.root,32);
        require(parallel.manifest_sha256==m.manifest_sha256 && parallel.source_sha256==m.source_sha256 && parallel.tensors.size()==m.tensors.size(),"parallel manifest validation changed identities");
        for (int threads : {0,-1,1025}) {
            bool rejected=false;try { f.load(f.root,threads); } catch(const std::exception &) { rejected=true; }
            require(rejected,"invalid validation thread limit accepted");
        }
        for (size_t size : {size_t(0),size_t(64),size_t(65536),size_t(65537),size_t(2*1024*1024+17)}) {
            const std::string bytes(size,'x');put(f.dir / "source.gguf",bytes);auto j=f.root;j["source_sha256"]=digest(bytes);
            require(f.load(j,32).source_sha256==digest(bytes),"source hashing differs from independent SHA256 oracle");
        }
        put(f.dir / "source.gguf","abc");
        auto many=f.root;many["tensors"]=json::array();
        for (int layer=0;layer<32;++layer) for (const auto & entry:f.root["tensors"]) {
            auto copy=entry;const std::string role=entry["name"].get<std::string>().substr(6);
            copy["name"]="blk."+std::to_string(layer)+"."+role;
            copy["site"]="qwen3_5.layers."+std::to_string(layer)+"."+role.substr(0,role.size()-7)+".input";
            many["tensors"].push_back(copy);
        }
        require(f.load(many,32).tensors.size()==64,"parallel manifest lost entries");
        auto bad=many;bad["tensors"][37]["sha256"]=std::string(64,'0');bool rejected=false;
        try { f.load(bad,32); } catch(const std::exception &) { rejected=true; }
        require(rejected,"parallel sample checksum failure not propagated");
        require(m.manifest_sha256==digest(f.root.dump()),"manifest digest mismatch");
        auto x = llama_sinq_load_activation_samples(m.tensors.at("blk.0.attn_q.weight"));
        require(x.size()==2 && x[0]==1.0f && x[1]==-2.0f,"little-endian decode mismatch");
        require(fs::path(m.tensors.begin()->second.data_file).is_absolute(),"data path not absolute");
        const auto reject_root = [&](const char * key, const json & value) { auto j=f.root; j[key]=value; f.reject(j,key); };
        reject_root("source_sha256",std::string(64,'0')); reject_root("architecture","wrong");
        reject_root("schema_version",2); reject_root("source_revision",""); reject_root("extra",0);
        reject_root("tensors",json::array()); reject_root("provenance",0);
        for (const auto & item : f.root.items()) {
            if (item.key()=="provenance") { continue; }
            auto j=f.root; j.erase(item.key()); f.reject(j,"missing root " + item.key());
        }
        for (const auto & item : f.root["sampling"].items()) {
            auto j=f.root; j["sampling"].erase(item.key()); f.reject(j,"missing sampling " + item.key());
        }
        for (const auto & item : f.root["tensors"][0].items()) {
            auto j=f.root; j["tensors"][0].erase(item.key()); f.reject(j,"missing tensor " + item.key());
        }
        const auto reject_tensor = [&](const char * key, const json & value) { auto j=f.root; j["tensors"][0][key]=value; f.reject(j,key); };
        reject_tensor("name","blk.0.unsupported.weight"); reject_tensor("name","xblk.0.attn_q.weight");
        reject_tensor("site","qwen3_5.layers.1.attn_q.input"); reject_tensor("channel_layout","other");
        reject_tensor("dtype","f4"); reject_tensor("order","column-major"); reject_tensor("reference_module","");
        reject_tensor("input_dim",0); reject_tensor("input_dim",-1); reject_tensor("input_dim",1.5);
        reject_tensor("input_dim",uint64_t(-1)); reject_tensor("input_dim",int64_t(1)<<62);
        reject_tensor("output_dim",0); reject_tensor("output_dim",uint64_t(-1));
        reject_tensor("count",0); reject_tensor("count",3); reject_tensor("count",1.0);
        reject_tensor("data_file","../outside.bin"); reject_tensor("data_file",(f.dir / "samples.bin").string());
        reject_tensor("sha256",std::string(64,'0')); reject_tensor("sha256",std::string(64,'A')); reject_tensor("extra",0);
        auto duplicate=f.root; duplicate["tensors"].push_back(duplicate["tensors"][0]); f.reject(duplicate,"duplicate tensor name");
        for (const auto & pair : {std::make_pair("max_rows",json(257)),std::make_pair("seed",json(-1)),
             std::make_pair("seed",json(uint64_t(1)<<32)),std::make_pair("contract",json("other")),std::make_pair("extra",json(0))}) {
            auto j=f.root; j["sampling"][pair.first]=pair.second; f.reject(j,"sampling " + std::string(pair.first));
        }
        for (const auto & bytes : {f.bytes.substr(0,7),f.bytes+"x",std::string("\0\0\xc0\x7f\0\0\0\0",8),
                                  std::string("\0\0\x80\x7f\0\0\0\0",8)}) {
            put(f.dir / "samples.bin",bytes); auto j=f.root;
            for (auto & t : j["tensors"]) { t["sha256"]=digest(bytes); }
            f.reject(j,"bad binary length or finite check");
        }
        put(f.dir / "samples.bin",f.bytes);
        auto too_many=f.root;
        while (too_many["tensors"].size()<=4096) { too_many["tensors"].push_back(f.root["tensors"][0]); }
        f.reject(too_many,"tensor count limit");
        auto provenance=f.root; provenance["provenance"]={{"tool","fixture"}}; f.load(provenance);
        f.load(f.root);
        require(llama_sinq_load_activations((f.dir / "activations.json").string(),
            (f.dir / "source.gguf").string()).tensors.size()==2,"explicit manifest path rejected");
        put(f.dir / "samples.bin",f.bytes+"x");
        bool reload_threw=false;
        try { llama_sinq_load_activation_samples(m.tensors.begin()->second); }
        catch (const std::exception &) { reload_threw=true; }
        require(reload_threw,"sample reload failed to revalidate length");
        put(f.dir / "samples.bin",f.bytes);
        put(f.dir / "activations.json","{\"schema_version\":1,"+f.root.dump().substr(1));
        bool threw=false; try { llama_sinq_load_activations((f.dir / "activations.json").string(),(f.dir / "source.gguf").string()); }
        catch (const std::exception &) { threw=true; } require(threw,"duplicate JSON key accepted");
        put(f.dir / "activations.json",std::string(8*1024*1024+1,' '));
        threw=false;
        try { llama_sinq_load_activations(f.dir.string(),(f.dir / "source.gguf").string()); }
        catch (const std::exception &) { threw=true; } require(threw,"oversize manifest accepted");
        fixture outside;
        std::error_code ec;
        fs::create_directory_symlink(outside.dir,f.dir / "escape",ec);
        if (!ec) { reject_tensor("data_file","escape/samples.bin"); }
        else { std::cout << "symlink test unavailable: " << ec.message() << '\n'; }
        std::cout << "activation manifest tests passed\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}
