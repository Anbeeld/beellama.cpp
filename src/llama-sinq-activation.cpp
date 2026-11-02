#include "llama-sinq-activation.h"
#include "../vendor/nlohmann/json.hpp"
extern "C" {
#include "../vendor/hash/sha256/sha256.h"
}
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
using json = nlohmann::ordered_json;
constexpr uint64_t manifest_limit = 8ull*1024*1024;
constexpr uint64_t sample_limit = 256ull*1024*1024;
[[noreturn]] void invalid(const std::string & why) { throw std::runtime_error("SINQ activation manifest: " + why); }
std::string finish_hash(sha256_t & hash) {
    unsigned char bytes[32]; sha256_final(&hash,bytes);
    const char * hex = "0123456789abcdef";
    std::string out; out.reserve(64);
    for (unsigned char b : bytes) { out += hex[b >> 4]; out += hex[b & 15]; }
    return out;
}
std::string hash_file(const fs::path & path) {
    std::ifstream file(path,std::ios::binary);
    if (!file) { invalid("cannot open source GGUF"); }
    sha256_t hash; sha256_init(&hash);
    std::array<char,65536> buffer;
    while (file) {
        file.read(buffer.data(),buffer.size());
        sha256_update(&hash,reinterpret_cast<const unsigned char *>(buffer.data()),size_t(file.gcount()));
    }
    if (!file.eof()) { invalid("source GGUF read failed"); }
    return finish_hash(hash);
}
void fields(const json & j, std::initializer_list<const char *> required,
            std::initializer_list<const char *> optional = {}) {
    if (!j.is_object()) { invalid("expected object"); }
    std::set<std::string> allowed;
    for (const auto * key : required) {
        allowed.insert(key);
        if (!j.contains(key)) { invalid(std::string("missing field ") + key); }
    }
    for (const auto * key : optional) { allowed.insert(key); }
    for (const auto & item : j.items()) {
        if (!allowed.count(item.key())) { invalid("unknown field " + item.key()); }
    }
}
std::string text(const json & j, const char * key) {
    const auto & value = j.at(key);
    if (!value.is_string()) { invalid(std::string("expected string: ") + key); }
    std::string result = value.get<std::string>();
    if (result.empty() || result.find('\0') != std::string::npos) { invalid(std::string("empty or NUL string: ") + key); }
    return result;
}
void equal(const json & j, const char * key, const char * expected) {
    if (text(j,key) != expected) { invalid(std::string("unsupported ") + key); }
}
uint64_t integer(const json & j, const char * key, uint64_t low, uint64_t high) {
    const auto & v = j.at(key);
    if (!v.is_number_integer() || (!v.is_number_unsigned() && v.get<int64_t>() < 0)) {
        invalid(std::string("expected nonnegative integer: ") + key);
    }
    const uint64_t n = v.get<uint64_t>();
    if (n < low || n > high) { invalid(std::string("out of range: ") + key); }
    return n;
}
std::string hash_text(const json & j, const char * key) {
    const std::string value = text(j,key);
    if (value.size() != 64 || value.find_first_not_of("0123456789abcdef") != std::string::npos) {
        invalid(std::string("invalid SHA256: ") + key);
    }
    return value;
}
uint64_t sample_bytes(const llama_sinq_activation_entry & entry) {
    if (entry.input_dim <= 0 || entry.output_dim <= 0 || entry.count <= 0 || entry.count > 256 ||
        uint64_t(entry.input_dim) > sample_limit/4/uint64_t(entry.count)) { invalid("sample dimensions exceed bounds"); }
    return uint64_t(entry.input_dim)*uint64_t(entry.count)*4;
}
void read_samples(const llama_sinq_activation_entry & entry, std::vector<float> * samples) {
    static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,"requires IEEE binary32 float");
    const uint64_t bytes = sample_bytes(entry);
    const fs::path path = fs::u8path(entry.data_file);
    if (!fs::is_regular_file(path) || fs::file_size(path) != bytes) { invalid("sample file size mismatch for " + entry.name); }
    std::ifstream file(path,std::ios::binary);
    if (!file) { invalid("cannot open samples for " + entry.name); }
    if (samples) { samples->resize(size_t(bytes/4)); }
    sha256_t hash; sha256_init(&hash);
    std::array<unsigned char,65536> buffer;
    uint64_t offset = 0;
    while (offset < bytes) {
        const size_t n = size_t(std::min<uint64_t>(buffer.size(),bytes-offset));
        file.read(reinterpret_cast<char *>(buffer.data()),std::streamsize(n));
        if (file.gcount() != std::streamsize(n)) { invalid("truncated samples for " + entry.name); }
        sha256_update(&hash,buffer.data(),n);
        for (size_t i=0; i<n; i+=4) {
            const uint32_t bits = uint32_t(buffer[i]) | (uint32_t(buffer[i+1])<<8) |
                (uint32_t(buffer[i+2])<<16) | (uint32_t(buffer[i+3])<<24);
            float value; std::memcpy(&value,&bits,4);
            if (!std::isfinite(value)) { invalid("nonfinite sample for " + entry.name); }
            if (samples) { (*samples)[size_t(offset/4)+i/4] = value; }
        }
        offset += n;
    }
    char trailing;
    if (file.get(trailing) || !file.eof()) { invalid("sample trailer or read failure for " + entry.name); }
    if (finish_hash(hash) != entry.sha256) { invalid("sample SHA256 mismatch for " + entry.name); }
}
fs::path contained_file(const fs::path & dir, const std::string & name) {
    const fs::path relative = fs::u8path(name);
    if (relative.is_absolute() || relative.has_root_name() || relative.has_root_directory() ||
        name.find('\\') != std::string::npos || name.find(':') != std::string::npos) { invalid("data_file must be relative"); }
    for (const auto & part : relative) { if (part == "..") { invalid("data_file traversal"); } }
    const fs::path resolved = fs::canonical(dir/relative);
    auto child = resolved.begin();
    for (auto parent = dir.begin(); parent != dir.end(); ++parent,++child) {
        if (child == resolved.end() || *child != *parent) { invalid("data_file escapes manifest directory"); }
    }
    if (!fs::is_regular_file(resolved)) { invalid("data_file is not regular"); }
    return resolved;
}
}

llama_sinq_activation_manifest llama_sinq_load_activations(const std::string & path,
                                                          const std::string & original_gguf) {
    fs::path manifest_path = fs::u8path(path);
    if (fs::is_directory(manifest_path)) { manifest_path /= "activations.json"; }
    manifest_path = fs::canonical(manifest_path);
    if (!fs::is_regular_file(manifest_path) || fs::file_size(manifest_path) > manifest_limit) { invalid("manifest exceeds 8 MiB or is not regular"); }
    std::ifstream file(manifest_path,std::ios::binary);
    if (!file) { invalid("cannot open manifest"); }
    std::string bytes;
    std::array<char,65536> buffer;
    while (file) {
        file.read(buffer.data(),buffer.size());
        if (uint64_t(bytes.size())+uint64_t(file.gcount()) > manifest_limit) { invalid("manifest exceeds 8 MiB"); }
        bytes.append(buffer.data(),size_t(file.gcount()));
    }
    if (!file.eof()) { invalid("manifest read failure"); }
    std::vector<std::set<std::string>> keys;
    const auto callback = [&](int depth, json::parse_event_t event, json & value) {
        if (depth > 64) { invalid("JSON nesting too deep"); }
        if (event == json::parse_event_t::object_start) { keys.emplace_back(); }
        if (event == json::parse_event_t::key && !keys.back().insert(value.get<std::string>()).second) { invalid("duplicate JSON key"); }
        if (event == json::parse_event_t::object_end) { keys.pop_back(); }
        return true;
    };
    const json root = json::parse(bytes,callback);
    fields(root,{"schema_version","architecture","source_sha256","source_revision","sampling","tensors"},{"provenance"});
    integer(root,"schema_version",1,1); equal(root,"architecture","qwen3_5");
    if (root.contains("provenance") && !root.at("provenance").is_object()) { invalid("provenance must be an object"); }
    llama_sinq_activation_manifest out;
    out.source_sha256 = hash_text(root,"source_sha256");
    out.source_revision = text(root,"source_revision");
    sha256_t manifest_hash; sha256_init(&manifest_hash);
    sha256_update(&manifest_hash,reinterpret_cast<const unsigned char *>(bytes.data()),bytes.size());
    out.manifest_sha256 = finish_hash(manifest_hash);
    if (hash_file(fs::u8path(original_gguf)) != out.source_sha256) { invalid("source GGUF SHA256 mismatch"); }
    const auto & sampling = root.at("sampling");
    fields(sampling,{"contract","seed","max_rows","corpus_sha256","token_sha256"});
    equal(sampling,"contract","valid-token-prefill-reservoir-v1");
    integer(sampling,"seed",0,std::numeric_limits<uint32_t>::max());
    const uint64_t max_rows = integer(sampling,"max_rows",1,256);
    hash_text(sampling,"corpus_sha256"); hash_text(sampling,"token_sha256");
    const auto & tensors = root.at("tensors");
    if (!tensors.is_array() || tensors.empty() || tensors.size() > 4096) { invalid("expected 1..4096 tensors"); }
    static const std::set<std::string> roles = {"attn_q","attn_k","attn_v","attn_qkv","attn_gate","attn_output",
        "ssm_beta","ssm_alpha","ssm_out","ffn_gate","ffn_up","ffn_down"};
    for (const auto & tensor : tensors) {
        fields(tensor,{"name","site","reference_module","channel_layout","input_dim","output_dim","count","dtype","order","data_file","sha256"});
        llama_sinq_activation_entry entry;
        entry.name = text(tensor,"name");
        // Equivalent to the anchored name regex, without recursive regex parsing on untrusted input.
        const size_t split = entry.name.find('.',4);
        if (entry.name.compare(0,4,"blk.") != 0 || split == std::string::npos || split == 4 ||
            entry.name.size() < split+8 || entry.name.compare(entry.name.size()-7,7,".weight") != 0) {
            invalid("unsupported tensor name " + entry.name);
        }
        const std::string layer = entry.name.substr(4,split-4);
        const std::string role = entry.name.substr(split+1,entry.name.size()-split-8);
        if (layer.find_first_not_of("0123456789") != std::string::npos || !roles.count(role)) {
            invalid("unsupported tensor name " + entry.name);
        }
        entry.site = text(tensor,"site");
        if (entry.site != "qwen3_5.layers." + layer + "." + role + ".input") { invalid("incorrect site for " + entry.name); }
        entry.reference_module = text(tensor,"reference_module");
        equal(tensor,"channel_layout","gguf-input"); equal(tensor,"dtype","<f4"); equal(tensor,"order","row-major");
        entry.input_dim = int64_t(integer(tensor,"input_dim",1,std::numeric_limits<int64_t>::max()));
        entry.output_dim = int64_t(integer(tensor,"output_dim",1,std::numeric_limits<int64_t>::max()));
        entry.count = int(integer(tensor,"count",1,max_rows));
        sample_bytes(entry);
        entry.sha256 = hash_text(tensor,"sha256");
        entry.data_file = contained_file(manifest_path.parent_path(),text(tensor,"data_file")).u8string();
        if (out.tensors.count(entry.name)) { invalid("duplicate tensor " + entry.name); }
        read_samples(entry,nullptr);
        out.tensors.emplace(entry.name,std::move(entry));
    }
    return out;
}

std::vector<float> llama_sinq_load_activation_samples(const llama_sinq_activation_entry & entry) {
    std::vector<float> out;
    read_samples(entry,&out);
    return out;
}
