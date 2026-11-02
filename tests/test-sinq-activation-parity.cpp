// Test-only CPU projection-input capture. No inference API or product hooks.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct capture_state {
    std::filesystem::path directory;
    std::map<std::string, int> counts;
    std::string error;
};

static bool eligible(const char * name) {
    static const std::array<const char *, 12> roles = {{
        "attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out",
        "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"
    }};
    for (const char * role : roles) {
        for (int layer = 0; layer < 2; ++layer) {
            if (std::string(name) == "blk." + std::to_string(layer) + "." + role + ".weight") {
                return true;
            }
        }
    }
    return false;
}

static bool capture(ggml_tensor * node, bool ask, void * opaque) {
    auto & state = *static_cast<capture_state *>(opaque);
    const bool selected = node->op == GGML_OP_MUL_MAT && node->src[0] && eligible(node->src[0]->name);
    if (ask || !selected) {
        return selected;
    }
    try {
        const ggml_tensor * input = node->src[1];
        if (input->type != GGML_TYPE_F32 || input->ne[2] != 1 || input->ne[3] != 1) {
            throw std::runtime_error("expected two-dimensional F32 projection input");
        }
        const std::string name = node->src[0]->name;
        if (++state.counts[name] != 1) {
            throw std::runtime_error("duplicate projection: " + name);
        }
        // Backend reads use physical byte strides, including noncontiguous views.
        std::vector<float> values(size_t(input->ne[0] * input->ne[1]));
        for (int64_t row = 0; row < input->ne[1]; ++row) {
            for (int64_t col = 0; col < input->ne[0]; ++col) {
                ggml_backend_tensor_get(input, &values[size_t(row * input->ne[0] + col)],
                    size_t(row) * input->nb[1] + size_t(col) * input->nb[0], sizeof(float));
            }
        }
        std::ofstream data(state.directory / (name + ".f32"), std::ios::binary);
        data.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
        std::ofstream shape(state.directory / (name + ".shape"));
        shape << input->ne[1] << ' ' << input->ne[0] << '\n';
        if (!data || !shape) {
            throw std::runtime_error("cannot write capture: " + name);
        }
    } catch (const std::exception & e) {
        state.error = e.what();
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s model.gguf output-dir token token [token ...]\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    try {
        ggml_backend_dev_t devices[] = {nullptr};
        llama_model_tensor_buft_override overrides[] = {{".*", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.devices = devices;
        mp.tensor_buft_overrides = overrides;
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        model = llama_model_load_from_file(argv[1], mp);
        if (!model) {
            throw std::runtime_error("model load failed");
        }
        capture_state state;
        auto cp = llama_context_default_params();
        cp.n_ctx = 64;
        cp.n_batch = cp.n_ubatch = 32;
        cp.n_threads = cp.n_threads_batch = 1;
        cp.offload_kqv = cp.op_offload = false;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.cb_eval = capture;
        cp.cb_eval_user_data = &state;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            throw std::runtime_error("context creation failed");
        }
        const int n = argc - 3;
        if (n > 32) {
            throw std::runtime_error("fixture exceeds batch size");
        }
        // Fresh prefill, then clear both recurrent and KV state and repeat.
        for (int repeat = 0; repeat < 2; ++repeat) {
            state.directory = std::filesystem::path(argv[2]) / std::to_string(repeat);
            std::filesystem::create_directories(state.directory);
            state.counts.clear();
            state.error.clear();
            llama_memory_clear(llama_get_memory(ctx), true);
            auto batch = llama_batch_init(n, 0, 1);
            batch.n_tokens = n;
            for (int i = 0; i < n; ++i) {
                batch.token[i] = std::stoi(argv[i + 3]);
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 1; // Preserve every last-layer row, not only the last token.
            }
            const int result = llama_decode(ctx, batch);
            llama_synchronize(ctx);
            llama_batch_free(batch);
            if (result != 0 || !state.error.empty() || state.counts.size() != 15) {
                throw std::runtime_error("capture failed: decode=" + std::to_string(result) +
                    " sites=" + std::to_string(state.counts.size()) + " " + state.error);
            }
        }
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "activation parity: %s\n", e.what());
        if (ctx) { llama_free(ctx); }
        if (model) { llama_model_free(model); }
        llama_backend_free();
        return 1;
    }
}
