#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-cpp.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

int main() {
    llama_model_ptr model(llama_model_create(LLM_ARCH_QWEN4EXP, llama_model_default_params()));
    auto * base = static_cast<llama_model_base *>(model.get());
    model->hparams.n_layer_all = 3;
    model->hparams.n_layer_nextn = 1;
    for (bool trunk : { false, true }) {
        for (bool mtp : { false, true }) {
            for (bool load_mtp : { false, true }) {
                gguf_context_ptr metadata(gguf_init_empty());
                gguf_set_val_str(metadata.get(), "general.architecture", "qwen4exp");
                ggml_init_params init = { 2*ggml_tensor_overhead() + 64, nullptr, false };
                ggml_context_ptr tensors(ggml_init(init));
                for (const auto & entry : std::vector<std::pair<bool, const char *>> {
                        { trunk, "blk.0.hc_attn_norm.weight" }, { mtp, "blk.2.nextn.eh_proj.weight" } }) {
                    if (entry.first) {
                        auto * tensor = ggml_new_tensor_1d(tensors.get(), GGML_TYPE_F32, 1);
                        ggml_set_name(tensor, entry.second);
                        *static_cast<float *>(tensor->data) = 1;
                        gguf_add_tensor(metadata.get(), tensor);
                    }
                }
                FILE * file = std::tmpfile();
                require(file != nullptr, "could not create the NextN fixture");
                require(gguf_write_to_file_ptr(metadata.get(), file, false), "could not write the NextN fixture");
                std::rewind(file);
                {
                    std::vector<std::string> splits;
                    llama_model_loader loader(nullptr, nullptr, nullptr, "", splits, file,
                            LLAMA_LOAD_MODE_NONE, false, false, load_mtp, nullptr, nullptr);
                    const auto flags = base->nextn_flags(loader, LLM_TENSOR_HC_ATTN_NORM);
                    require(flags.trunk == (trunk ? 0 : llama_model_loader::TENSOR_NOT_REQUIRED),
                            "standalone MTP must not require trunk weights");
                    require(flags.mtp == ((mtp ? 0 : llama_model_loader::TENSOR_NOT_REQUIRED) |
                            (load_mtp ? 0 : llama_model_loader::TENSOR_SKIP)),
                            "NextN tensor admission must distinguish missing and explicitly skipped heads");
                    model->hparams.n_layer_nextn = 0;
                    const auto ordinary = base->nextn_flags(loader, LLM_TENSOR_HC_ATTN_NORM);
                    require(ordinary.trunk == 0 && ordinary.mtp == (load_mtp ? 0 : llama_model_loader::TENSOR_SKIP),
                            "an ordinary model must retain its required trunk");
                    model->hparams.n_layer_nextn = 1;
                }
                std::fclose(file);
            }
        }
    }
}
