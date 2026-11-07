// Offline, run-owned CUDA model. No public inference API or quantization math.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

bool perplexity_full(llama_context *, const common_params &);
using json = nlohmann::ordered_json;

static std::vector<char> read_exact(const std::string & path, size_t bytes) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != std::streamoff(bytes)) { throw std::runtime_error("candidate byte length mismatch"); }
    f.seekg(0);
    std::vector<char> data(bytes);
    f.read(data.data(), std::streamsize(bytes));
    if (!f) { throw std::runtime_error("candidate payload read failed"); }
    return data;
}

int main(int argc, char ** argv) try {
    common_init();
    common_params params;
    params.escape = false;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) { return 1; }
    if (!params.ppl_full || !params.kl_divergence || params.n_ctx <= 0 || params.n_gpu_layers < 999 ||
            params.n_parallel != 1 || params.split_mode != LLAMA_SPLIT_MODE_NONE ||
            params.cache_type_k != GGML_TYPE_BF16 || params.cache_type_v != GGML_TYPE_BF16 ||
            params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED || params.warmup || params.fit_params) {
        throw std::runtime_error("prepared evaluator requires explicit full-offload CUDA/BF16/FA/no-warmup/no-fit policy");
    }
    if (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) { throw std::runtime_error("CUDA device required"); }
    llama_backend_init();
    // model_only avoids binding any graph before private scale vectors exist.
    auto loaded = common_init_from_params(params, true);
    auto * model = loaded->model();
    if (!model) { throw std::runtime_error("starting model load failed"); }
    std::unique_ptr<ggml_context, decltype(&ggml_free)> scales(nullptr, ggml_free);
    std::vector<std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>> buffers;
    std::map<std::string, ggml_tensor *> weights, private_scales;
    for (const auto & entry : llama_internal_get_tensor_map(model)) { weights.emplace(entry.first, entry.second); }
    // A bounded setup message names the complete mutable tensor allocation. Each
    // gets a PRIVATE F32 companion, never the loader's deduplicated runtime cache.
    // Stock and all-one conditioned diagonals are elided exactly as in the
    // canonical loader. A fresh context below rebuilds graphs after transitions.
    std::string line;
    if (!std::getline(std::cin, line) || line.size() > (1u << 20)) { throw std::runtime_error("setup message missing/oversized"); }
    const auto setup = json::parse(line);
    const auto names = setup.at("tensors").get<std::vector<std::string>>();
    if (names.empty() || names.size() > 10000) { throw std::runtime_error("invalid mutable tensor count"); }
    scales.reset(ggml_init({ggml_tensor_overhead()*names.size(), nullptr, true}));
    if (!scales) { throw std::runtime_error("scale context allocation failed"); }
    for (const auto & name : names) {
        auto found = weights.find(name);
        if (found == weights.end() || private_scales.count(name)) { throw std::runtime_error("unknown/duplicate mutable tensor"); }
        auto * weight = found->second;
        auto * device = ggml_backend_buffer_get_type(weight->buffer);
        auto * gpu = ggml_backend_buft_get_device(device);
        if (!gpu || ggml_backend_dev_type(gpu) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                std::string(ggml_backend_dev_name(gpu)) != "CUDA0") { throw std::runtime_error("mutable weight not on requested CUDA device"); }
        auto * scale = ggml_new_tensor_1d(scales.get(), GGML_TYPE_F32, weight->ne[0]);
        ggml_format_name(scale, "%s.prepared_scale", name.c_str());
        buffers.emplace_back(ggml_backend_buft_alloc_buffer(device, ggml_nbytes(scale) + ggml_backend_buft_get_alignment(device)), ggml_backend_buffer_free);
        if (!buffers.back() || ggml_backend_tensor_alloc(buffers.back().get(), scale, ggml_backend_buffer_get_base(buffers.back().get())) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("private CUDA scale allocation failed");
        }
        ggml_backend_buffer_set_usage(buffers.back().get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> values(size_t(weight->ne[0]), 1.0f);
        auto previous = model->sinq_scales.find(weight);
        if (previous != model->sinq_scales.end()) {
            if (previous->second->type != GGML_TYPE_F32 || ggml_nbytes(previous->second) != values.size()*sizeof(float)) {
                throw std::runtime_error("unexpected loader runtime scale representation");
            }
            ggml_backend_tensor_get(previous->second, values.data(), 0, values.size()*sizeof(float));
        }
        ggml_backend_tensor_set(scale, values.data(), 0, values.size()*sizeof(float));
        if (std::all_of(values.begin(), values.end(), [](float value) { return value == 1.0f; })) {
            model->sinq_scales.erase(weight);
        } else {
            model->sinq_scales[weight] = scale;
        }
        private_scales.emplace(name, scale);
    }
    std::cout << "SINQ_RESPONSE " << json({{"status","READY"},{"mutable_tensors",names.size()},{"topology","private-f32-diagonals-canonical-identity-elision-context-rebuilt-per-score"}}).dump() << std::endl;
    while (std::getline(std::cin, line)) {
        if (line.size() > (1u << 20)) { throw std::runtime_error("request budget exceeded"); }
        const auto request = json::parse(line);
        if (request.value("operation", "") == "close") { break; }
        const auto & updates = request.at("updates");
        if (!updates.is_array() || updates.size() > names.size()) { throw std::runtime_error("invalid update count"); }
        std::set<std::string> changed;
        for (const auto & update : updates) {
            const auto name = update.at("name").get<std::string>();
            if (!private_scales.count(name) || !changed.insert(name).second) { throw std::runtime_error("unregistered/duplicate update"); }
            auto * weight = weights.at(name);
            if (update.at("qtype").get<int>() != int(weight->type)) { throw std::runtime_error("candidate qtype changes require canonical reload"); }
            const auto payload = read_exact(update.at("payload"), ggml_nbytes(weight));
            std::vector<float> values(size_t(weight->ne[0]), 1.0f);
            if (!update.at("scales").is_null()) {
                const auto encoded = read_exact(update.at("scales"), values.size()*sizeof(ggml_fp16_t));
                for (size_t i=0; i<values.size(); ++i) {
                    ggml_fp16_t value;
                    std::memcpy(&value, encoded.data()+i*sizeof(value), sizeof(value));
                    values[i] = ggml_fp16_to_fp32(value);
                    if (!std::isfinite(values[i]) || values[i] <= 0) { throw std::runtime_error("invalid candidate diagonal"); }
                }
            }
            ggml_backend_tensor_set(weight, payload.data(), 0, payload.size());
            ggml_backend_tensor_set(private_scales.at(name), values.data(), 0, values.size()*sizeof(float));
            if (std::all_of(values.begin(), values.end(), [](float value) { return value == 1.0f; })) {
                model->sinq_scales.erase(weight);
            } else {
                model->sinq_scales[weight] = private_scales.at(name);
            }
        }
        // Rebuilding the CONTEXT, not the model, invalidates all graphs, KV,
        // recurrent/SSM state and logits without reloading unchanged GPU weights.
        auto cp = common_context_params_to_llama(params);
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, cp), llama_free);
        if (!ctx) { throw std::runtime_error("fresh candidate context failed"); }
        auto scoring = params;
        scoring.kld_token_output = request.at("token_output").get<std::string>();
        if (scoring.kld_token_output.empty() || !perplexity_full(ctx.get(), scoring)) { throw std::runtime_error("candidate scoring failed"); }
        llama_synchronize(ctx.get());
        std::cout << "SINQ_RESPONSE " << json({{"status","PASS"},{"request_id",request.at("request_id")},{"updates",updates.size()}}).dump() << std::endl;
    }
    // Contexts are already destroyed. Drop map pointers before releasing buffers.
    model->sinq_scales.clear();
    loaded.reset();
    llama_backend_free();
    return 0;
} catch (const std::exception & error) {
    std::cerr << "prepared groupwise evaluator: " << error.what() << '\n';
    return 1;
}
