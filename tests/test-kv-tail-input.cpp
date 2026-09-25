#include "llama-kv-cache.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

int main() {
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    check(backend != nullptr, "CPU backend unavailable");
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    // No model tensors: exercise the same metadata cache used by structured caches.
    llama_kv_cache cache(*model, model->hparams, GGML_TYPE_F16, GGML_TYPE_F16,
            false, false, true, 32, 2, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, {}, {}, {}, 4, 2, GGML_TYPE_F16, 2, true, 0);
    check(cache.has_compact_tail(), "fixture must use a compact tail");
    llama_kv_tail_layer_route route = {};
    route.capability.supported = true;
    route.capability.route = LLAMA_KV_TAIL_ROUTE_NATIVE;
    cache.set_tail_routes({route});
    cache.finalize_tail_overlay_metadata();

    llama_pos positions[] = {0, 1, 2, 3};
    int32_t counts[] = {2, 1, 1, 1};
    llama_seq_id owners[] = {1, 0};
    llama_seq_id * seqs[] = {owners, owners + 1, owners + 1, owners + 1};
    llama_ubatch batch = {};
    batch.n_tokens = 4;
    batch.n_pos = 1;
    batch.pos = positions;
    batch.n_seq_id = counts;
    batch.seq_id = seqs;
    llama_kv_cache::slot_info slots = {};
    slots.strm = {0};
    slots.idxs = {{0, 1, 2, 3}};
    cache.apply_ubatch(slots, batch);

    auto ctx = ggml_init({1024*1024, nullptr, true});
    auto indices = cache.build_input_tail_idxs(ctx, batch);
    check(indices && indices->ne[1] == 2, "fixture must produce two owner levels");
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "input allocation failed");
    cache.set_input_tail_idxs(indices, &batch);
    std::vector<int64_t> got(8);
    ggml_backend_tensor_get(indices, got.data(), 0, got.size()*sizeof(int64_t));
    // Level 1/row 0 and level 0/row 2 reuse slot 0. Only row 2 persists.
    check(got[2] == 0 && got[3] == 1 && got[4] == -1,
            "production input did not retain the globally last slot write");
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] < 0) continue;
        for (size_t j = i + 1; j < got.size(); ++j) {
            check(got[i] != got[j], "production input retained duplicate destinations");
        }
    }
    cache.finish_tail_batch(true, false);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}
