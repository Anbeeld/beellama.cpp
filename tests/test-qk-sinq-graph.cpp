#include "llama-adapter.h"
#include "llama-graph.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "QK SINQ graph: %s\n", message);
        std::abort();
    }
}

// Metadata-only characterization: scaling belongs to the base matmul, not LoRA.
int main() {
    for (const auto type : { GGML_TYPE_Q3_K, GGML_TYPE_Q4_K }) {
      for (const int k : { 256, 512 }) {
        llm_graph_result result(128);
        llm_graph_params params = {};
        params.hparams.n_layer_all = 1;
        params.hparams.n_embd = k;
        params.hparams.n_head_arr[0] = 1;
        params.hparams.n_head_kv_arr[0] = 1;
        params.hparams.n_embd_head_k_full = k;
        params.hparams.n_embd_head_v_full = k;
        params.res = &result;

        auto * ctx = result.get_ctx();
        auto * weight = ggml_new_tensor_2d(ctx, type, k, 16);
        ggml_set_name(weight, "blk.0.ffn_up.weight");
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 2);
        auto * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, k);
        std::unordered_map<const ggml_tensor *, ggml_tensor *> scales = { { weight, scale } };
        params.sinq_scales = &scales;

        llama_adapter_loras loras;
        params.loras = &loras;
        llm_graph_context graph(params);
        auto * base = graph.build_lora_mm(weight, input);
        require(base->op == GGML_OP_MUL_MAT && base->src[0] == weight, "base matmul lost its ordinary QK weight");
        require(base->src[1]->op == GGML_OP_MUL, "base input is not scaled");
        require(base->src[1]->src[0] == input, "scale must multiply the original input");
        auto * scale_f32 = base->src[1]->src[1];
        require(scale_f32->op == GGML_OP_CPY && scale_f32->type == GGML_TYPE_F32 && scale_f32->src[0] == scale,
                "FP16 companion must be converted for portable F32 activation multiplication");

        require(graph.build_lora_mm(weight,input)->src[1] == base->src[1],
                "same activation and scale must reuse their multiplication");
        auto * cached_scale = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,k);
        auto * cached_weight = ggml_new_tensor_2d(ctx,type,k,16);
        scales.emplace(cached_weight,cached_scale);
        auto * cached_mm = graph.build_lora_mm(cached_weight,input);
        require(cached_mm->src[1]->src[1] == cached_scale,
                "load-time F32 companion cache must not add a per-step cast/copy");

        llama_adapter_lora adapter(nullptr);
        adapter.alpha = 0.0f;
        auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 4);
        auto * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 16);
        adapter.ab_map.emplace(weight->name, llama_adapter_lora_weight(a, b));
        loras.emplace(&adapter, 1.0f);
        auto * combined = graph.build_lora_mm(weight, input);
        require(combined->op == GGML_OP_ADD, "LoRA delta is not added");
        auto * delta = combined->src[1];
        require(delta->op == GGML_OP_SCALE, "LoRA adapter scaling is missing");
        auto * b_mm = delta->src[0];
        require(b_mm->op == GGML_OP_MUL_MAT && b_mm->src[0] == b, "LoRA B matmul is missing");
        auto * a_mm = b_mm->src[1];
        require(a_mm->op == GGML_OP_MUL_MAT && a_mm->src[0] == a && a_mm->src[1] == input,
                "LoRA A must read the original, unscaled input");
        require(combined->src[0]->src[0] == weight && combined->src[0]->src[1]->op == GGML_OP_MUL,
                "LoRA changed base weight or input scaling");

        for (const auto ordinary_type : { type, GGML_TYPE_F32 }) {
            auto * ordinary = ggml_new_tensor_2d(ctx, ordinary_type, k, 16);
            auto * ordinary_mm = graph.build_lora_mm(ordinary, input);
            require(ordinary_mm->op == GGML_OP_MUL_MAT && ordinary_mm->src[0] == ordinary && ordinary_mm->src[1] == input,
                    "unassociated weights must keep the original input");
        }
        require(weight->type == type && scale->type == GGML_TYPE_F16 && input->op == GGML_OP_NONE,
                "graph construction mutated a source tensor");
        std::printf("%s K=%d: base-only scale, original LoRA input, unassociated QK/F32 unchanged\n", ggml_type_name(type), k);
      }
    }
    std::puts("QK SINQ graph transport: OK");
    return 0;
}
