#include "llama-context.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

static std::string read_file(const std::string & path) {
    std::ifstream file(path);
    if (!file.good()) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(1);
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static bool expect(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
    }
    return ok;
}

int main(int argc, char ** argv) {
    bool ok = true;

    ok &= expect(llama_dflash_gpu_tape_supported_arch(LLM_ARCH_QWEN35), "Qwen3.5 must support GPU tape");
    ok &= expect(llama_dflash_gpu_tape_supported_arch(LLM_ARCH_QWEN35MOE), "Qwen3.5-MoE must support GPU tape");
    ok &= expect(!llama_dflash_gpu_tape_supported_arch(LLM_ARCH_QWEN3NEXT), "Qwen3Next must stay on fallback");
    ok &= expect(!llama_dflash_gpu_tape_supported_arch(LLM_ARCH_KIMI_LINEAR), "Kimi-linear must stay on fallback");
    ok &= expect(!llama_dflash_gpu_tape_supported_arch(LLM_ARCH_UNKNOWN), "unknown arch must stay on fallback");

    ok &= expect(argc == 2, "expected repo root argument");
    if (!ok) {
        return 1;
    }

    const std::string root = argv[1];
    const std::string context_h = read_file(root + "/src/llama-context.h");
    const std::string context_cpp = read_file(root + "/src/llama-context.cpp");
    const std::string cparams_h = read_file(root + "/src/llama-cparams.h");
    const std::string graph_cpp = read_file(root + "/src/llama-graph.cpp");
    const std::string llama_h = read_file(root + "/include/llama.h");
    const std::string sampling_h = read_file(root + "/common/sampling.h");
    const std::string sampling_cpp = read_file(root + "/common/sampling.cpp");
    const std::string server_context = read_file(root + "/tools/server/server-context.cpp");
    const std::string speculative = read_file(root + "/common/speculative.cpp");
    const std::string dflash_draft = read_file(root + "/src/models/dflash_draft.cpp");
    const std::string memory_recurrent = read_file(root + "/src/llama-memory-recurrent.cpp");
    const std::string qwen35 = read_file(root + "/src/models/qwen35.cpp");
    const std::string qwen35moe = read_file(root + "/src/models/qwen35moe.cpp");
    const std::string graph_h = read_file(root + "/src/llama-graph.h");
    const std::string model_cpp = read_file(root + "/src/llama-model.cpp");
    const std::string cuda_ring = read_file(root + "/ggml/src/ggml-cuda/cross-ring-interleave.cu");
    const std::string cuda_cpp = read_file(root + "/ggml/src/ggml-cuda/ggml-cuda.cu");
    const std::string cuda_argmax = read_file(root + "/ggml/src/ggml-cuda/argmax.cu");
    const std::string cuda_gdn = read_file(root + "/ggml/src/ggml-cuda/gated_delta_net.cu");
    const std::string cuda_reg = read_file(root + "/ggml/src/ggml-cuda/ggml-cuda.cu");

    const size_t pretranspose = qwen35moe.find("\"qkv_mixed_pretranspose\"");
    const size_t transpose    = qwen35moe.find("qkv_mixed = ggml_transpose(ctx0, qkv_mixed)");
    const size_t transposed   = qwen35moe.find("cb(qkv_mixed, \"qkv_mixed_transposed\", il)");

    ok &= expect(pretranspose != std::string::npos, "Qwen3.5-MoE must name pre-transpose QKV");
    ok &= expect(transpose    != std::string::npos, "Qwen3.5-MoE transpose must exist");
    ok &= expect(transposed   != std::string::npos, "Qwen3.5-MoE transposed QKV name must remain");
    ok &= expect(pretranspose < transpose, "Qwen3.5-MoE pre-transpose QKV name must come before transpose");
    ok &= expect(transpose    < transposed, "Qwen3.5-MoE transposed QKV name must come after transpose");

    ok &= expect(context_h.find("ggml_tensor * qkv") != std::string::npos, "GPU tape layer must own QKV tensor");
    ok &= expect(context_cpp.find("tl.qkv  = ggml_new_tensor_2d") != std::string::npos, "GPU tape allocator must allocate QKV tensor");
    ok &= expect(context_cpp.find("n_accepted <= gpu_tape->n_tokens") != std::string::npos, "GPU tape replay must require populated token count");
    ok &= expect(context_cpp.find("std::max((int) layer_hiddens.size(), (int) dflash_capture->tapes.size())") != std::string::npos, "active DFlash slot must work without GPU tape");
    ok &= expect(context_cpp.find("ggml_backend_tensor_get_async(gpu_backend, gpu_layer->qkv") != std::string::npos, "conv rebuild must read QKV from GPU tape");
    ok &= expect(graph_cpp.find("t_logits_argmax = nullptr;") != std::string::npos, "graph reset must clear reduced logits output pointer");
    ok &= expect(context_cpp.find("logits_argmax_buf.clear();") != std::string::npos, "decode must clear stale reduced logits ids");
    ok &= expect(context_cpp.find("logits_argmax_prob_buf.clear();") != std::string::npos, "decode must clear stale reduced logits probabilities");
    ok &= expect(context_cpp.find("const int64_t step = 128;") != std::string::npos, "DFlash cross buckets must avoid the 513-to-1024 latency cliff");
    ok &= expect(dflash_draft.find("bool can_reuse(const llm_graph_params & params) override") != std::string::npos, "DFlash drafter graph input must opt into graph reuse");
    ok &= expect(dflash_draft.find("dflash_draft_ctx_len(params.cross, params.cparams) != ctx_len") != std::string::npos, "DFlash drafter reuse must invalidate on cross bucket shape changes");
    ok &= expect(dflash_draft.find("dflash_kv_cache_ready_for_window(params.cross, ctx_len) != use_kv_cache") != std::string::npos, "DFlash drafter reuse must invalidate when K/V cache topology changes");
    ok &= expect(dflash_draft.find("Kcur_ctx_f16") == std::string::npos, "DFlash drafter must not cast K before ggml_concat; CUDA concat requires F32 sources");
    ok &= expect(dflash_draft.find("Vcur_ctx_f16") == std::string::npos, "DFlash drafter must not cast V before ggml_concat; CUDA concat requires F32 sources");
    ok &= expect(dflash_draft.find("ggml_tensor * inp_out_ids = n_outputs < n_tokens ? build_inp_out_ids() : nullptr;") != std::string::npos, "DFlash drafter must avoid unused seed-token output rows");
    ok &= expect(dflash_draft.find("inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);") != std::string::npos, "DFlash drafter must gather only requested output rows before lm_head");
    ok &= expect(dflash_draft.find("const bool need_swa_mask = have_swa && hparams.n_swa > 0 && (int64_t) hparams.n_swa < n_kv_total;") != std::string::npos, "DFlash drafter must skip redundant SWA masks when the cross window is within n_swa");
    ok &= expect(graph_h.find("LLM_GRAPH_TYPE_DFLASH_KV_UPDATE") != std::string::npos, "DFlash K/V cache update must have a distinct graph type");
    ok &= expect(graph_h.find("llama_dflash_kv_cache_view * dflash_kv_cache") != std::string::npos, "DFlash cross data must carry the drafter K/V cache view");
    ok &= expect(graph_h.find("std::vector<ggml_tensor *> k_ring") != std::string::npos, "DFlash K/V cache must use ring-only persistent storage");
    ok &= expect(model_cpp.find("llm_build_dflash_kv_update") != std::string::npos, "DFlash draft model must dispatch the K/V cache update graph");
    ok &= expect(dflash_draft.find("Kcur_ctx_cache_pre_rope") != std::string::npos, "DFlash draft graph must consume cached pre-RoPE K and still apply current window RoPE");
    ok &= expect(dflash_draft.find("Vcur_ctx_cache") != std::string::npos, "DFlash draft graph must consume cached V");
    ok &= expect(speculative.find("GGML_DFLASH_DISABLE_KV_CACHE") != std::string::npos, "DFlash K/V cache must have a diagnostic kill switch");
    ok &= expect(dflash_draft.find("GGML_DFLASH_KV_CACHE_MODE") != std::string::npos, "DFlash K/V cache must support K-only/V-only isolation");
    ok &= expect(dflash_draft.find("llm_build_dflash_kv_update::llm_build_dflash_kv_update") != std::string::npos, "DFlash K/V update graph builder must exist");
    ok &= expect(context_cpp.find("dflash_kv_cache_prepare((int) cross.n_enc)") != std::string::npos, "DFlash set_cross_data_gpu must expose cached K/V for the active window");
    ok &= expect(context_cpp.find("n_layers * 2 + 8") != std::string::npos, "DFlash K/V cache must not allocate duplicate staging tensors");
    ok &= expect(speculative.find("llama_dflash_kv_cache_update_from_ring(ctx_dft, gpu_ring_handle") != std::string::npos, "DFlash accepted hiddens must update K/V cache from the GPU ring without mutating cross state");
    ok &= expect(context_cpp.find("dflash_kv_update_gpu") != std::string::npos, "DFlash K/V update must use a temporary input source separate from the main cross state");
    ok &= expect(cuda_cpp.find("dflash_kv_update") != std::string::npos, "DFlash K/V update graph must be excluded from CUDA graph capture");
    ok &= expect(cuda_ring.find("dflash_kv_cache_write_d2d") != std::string::npos, "CUDA backend must provide D2D K/V cache ring writes");
    ok &= expect(cuda_ring.find("dflash_kv_cache_append_d2d") != std::string::npos, "CUDA backend must provide chronological D2D K/V cache appends");
    ok &= expect(context_cpp.find("fn_append_d2d") != std::string::npos, "DFlash K/V cache updates must append chronologically instead of exposing wrapped rings to the draft graph");
    ok &= expect(cuda_ring.find("dflash_kv_cache_interleave") != std::string::npos, "CUDA backend must stage ordered DFlash K/V cache windows");
    ok &= expect(cuda_cpp.find("dflash_kv_cache_interleave") != std::string::npos, "CUDA backend registry must expose DFlash K/V cache helpers");
    ok &= expect(context_h.find("fn_append_d2d_no_check") != std::string::npos, "DFlash K/V cache must track hot-loop append helper");
    ok &= expect(context_h.find("fn_sync_ptr") != std::string::npos, "DFlash K/V cache no-sync appends must have a batched stream-sync helper");
    ok &= expect(context_cpp.find("fn_append_d2d_no_check") != std::string::npos, "DFlash K/V cache update must prefer no-sync hot-loop append helper");
    ok &= expect(context_cpp.find("fast_append && !dflash_kv_cache->fn_sync_ptr") != std::string::npos, "DFlash K/V cache no-sync appends must synchronize before the drafter graph reads cached K/V");
    ok &= expect(cuda_ring.find("dflash_kv_cache_append_d2d_no_check") != std::string::npos, "CUDA backend must provide no-sync DFlash K/V cache appends");
    ok &= expect(cuda_reg.find("\"dflash_kv_cache_append_d2d_no_check\"") != std::string::npos, "CUDA backend registry must publish no-sync DFlash K/V cache appends");
    ok &= expect(speculative.find("common_batch_add(batch_dft, id_last, cross_len, { seq_id }, true);") != std::string::npos, "DFlash flat/tree drafts must preserve seed-token logits for row alignment");
    ok &= expect(speculative.find("float * logits = llama_get_logits_ith(ctx_dft, i);") != std::string::npos, "DFlash flat draft fallback rows must preserve seed-token offset");
    ok &= expect(speculative.find("const int offset = r * batch_len;") != std::string::npos, "DFlash batched draft argmax row offsets must preserve seed-token output rows");
    ok &= expect(speculative.find("graph_reuse=%d") != std::string::npos, "DFlash draft profile must report drafter graph reuse");
    ok &= expect(context_cpp.find("output_reorder();\n    if (logits_argmax_buf.empty())") != std::string::npos, "argmax row access must honor output reordering");
    ok &= expect(context_cpp.find("std::swap(logits_argmax_buf") != std::string::npos, "output reordering must include reduced logits ids");
    ok &= expect(context_cpp.find("std::swap(logits_argmax_prob_buf") != std::string::npos, "output reordering must include reduced logits probabilities");
    ok &= expect(qwen35.find("ggml_build_forward_expand(gf, ggml_cpy(ctx0, qkv_cont, qkv_dst))") != std::string::npos, "Qwen3.5 must graph-copy QKV into GPU tape");
    ok &= expect(qwen35moe.find("ggml_build_forward_expand(gf, ggml_cpy(ctx0, qkv_cont, qkv_dst))") != std::string::npos, "Qwen3.5-MoE must graph-copy QKV into GPU tape");
    ok &= expect(speculative.find("GGML_DFLASH_PROFILE") != std::string::npos, "DFlash speculative path must honor profiling flag");
    ok &= expect(speculative.find("gpu_sync=%.3f ms") != std::string::npos, "DFlash ring profiling must report GPU sync time");
    ok &= expect(speculative.find("kv_cache_update requested=%d update=%d") != std::string::npos, "DFlash accept profiling must report drafter K/V cache update time");
    ok &= expect(context_h.find("struct dflash_hidden_gpu") != std::string::npos, "DFlash must define GPU hidden capture storage");
    ok &= expect(cparams_h.find("hidden_gpu_seqs") != std::string::npos, "graph params must expose per-seq GPU hidden targets");
    ok &= expect(context_cpp.find("allocate_hidden_gpu(n_slots, max_tokens)") != std::string::npos, "GPU tape allocation must allocate hidden capture buffers too");
    ok &= expect(context_cpp.find("dflash_graph_hidden_ready ? nullptr : dflash_eval_callback") != std::string::npos, "eligible DFlash verifier graph must disable eval callback");
    ok &= expect(context_cpp.find("const bool dflash_graph_tape_ready") != std::string::npos, "DFlash decode must gate GPU tape copies separately from hidden capture");
    ok &= expect(context_cpp.find("dflash_graph_hidden_ready =\n                    !dflash_capture->hidden_gpu.empty()") != std::string::npos, "GPU hidden graph capture must not depend on active tape recording");
    ok &= expect(context_cpp.find("dflash_tape_gpu * graph_tp = dflash_graph_tape_ready ? tp : nullptr") != std::string::npos, "GPU tape graph pointers must be disabled when tape recording is inactive");
    ok &= expect(context_cpp.find("buf.n_tokens <= 0 || buf.data.empty()") != std::string::npos, "empty CPU hidden buffers must not mask GPU hidden buffers");
    ok &= expect(context_cpp.find("hidden->n_tokens = 0") != std::string::npos, "GPU hidden token counts must reset between decodes");
    ok &= expect(context_cpp.find("ggml_backend_tensor_get(tensor, staging.data()") != std::string::npos, "GPU hidden D2D fallback must preserve correctness via readback");
    ok &= expect(qwen35.find("cparams.hidden_gpu_n_seqs > 0") != std::string::npos, "Qwen3.5 must graph-copy l_out into GPU hidden buffers");
    ok &= expect(qwen35moe.find("cparams.hidden_gpu_n_seqs > 0") != std::string::npos, "Qwen3.5-MoE must graph-copy l_out into GPU hidden buffers");
    ok &= expect(cuda_ring.find("dflash_cross_ring_gpu_write_d2d") != std::string::npos, "CUDA ring must expose D2D hidden writes");
    ok &= expect(cuda_reg.find("\"dflash_cross_ring_gpu_write_d2d\"") != std::string::npos, "CUDA backend registry must publish D2D ring writes");
    ok &= expect(llama_h.find("llama_dflash_cross_ring_gpu_write_hidden") != std::string::npos, "public DFlash API must expose hidden GPU ring writes");
    ok &= expect(speculative.find("llama_dflash_cross_ring_gpu_write_hidden") != std::string::npos, "DFlash ring update must use GPU hidden writes when CPU capture is absent");
    ok &= expect(speculative.find("gpu_upload_queued = true;") != std::string::npos, "DFlash GPU hidden D2D ring writes must synchronize before the drafter graph reads the ring");
    const size_t set_tensor_fn = cuda_ring.find("dflash_cross_ring_gpu_set_tensor");
    ok &= expect(set_tensor_fn != std::string::npos, "cross-ring source must provide set_tensor helper");
    const size_t set_tensor_end = cuda_ring.find("dflash_kv_cache_write_d2d", set_tensor_fn);
    const size_t sync_in_set_tensor = cuda_ring.find("cudaStreamSynchronize", set_tensor_fn);
    ok &= expect(sync_in_set_tensor != std::string::npos && sync_in_set_tensor < set_tensor_end,
        "dflash_cross_ring_gpu_set_tensor must synchronize before GGML backend-stream graph reads");
    ok &= expect(context_h.find("tape_replay_conv_gpu") != std::string::npos, "DFlash must declare GPU conv rebuild fast path");
    ok &= expect(context_cpp.find("tape_replay_conv_gpu(mem_recurrent, cell_idx, n_accepted)") != std::string::npos, "conv replay must try GPU rebuild before CPU fallback");
    ok &= expect(context_cpp.find("conv_gpu_enqueue=%.3f ms") != std::string::npos, "DFlash profile must report GPU conv enqueue time");
    ok &= expect(cuda_ring.find("k_dflash_rebuild_conv_state") != std::string::npos, "CUDA ring source must provide conv rebuild kernel");
    ok &= expect(cuda_reg.find("\"dflash_rebuild_conv_state\"") != std::string::npos, "CUDA backend registry must publish conv rebuild kernel");
    ok &= expect(context_h.find("replay_direct_gpu") != std::string::npos, "DFlash capture state must track direct GPU replay");
    ok &= expect(context_h.find("tape_replay_gdn_direct_gpu") != std::string::npos, "DFlash must declare direct GPU GDN replay fast path");
    ok &= expect(context_cpp.find("tape_replay_gdn_direct_gpu(mem_recurrent, cell_idx, n_accepted)") != std::string::npos, "tape replay must try direct GPU GDN replay before ggml graph replay");
    ok &= expect(context_cpp.find("dflash_replay_gdn_state_no_check") != std::string::npos, "DFlash direct replay must call CUDA GDN replay helper");
    ok &= expect(context_cpp.find("tl.gate->ne[0] != 1 || tl.beta->ne[0] != 1") != std::string::npos, "direct GDN replay must reject KDA gate/beta layouts");
    ok &= expect(context_cpp.find("DFlash direct GPU GDN replay launch failed after validation") != std::string::npos, "direct GDN replay must not fall back after partial GPU state mutation");
    ok &= expect(cuda_gdn.find("dflash_gdn_state_replay_cuda") != std::string::npos, "CUDA GDN source must provide state-only replay kernel");
    ok &= expect(cuda_gdn.find("dflash_replay_gdn_state_no_check") != std::string::npos, "CUDA GDN source must export state-only replay helper");
    ok &= expect(cuda_reg.find("\"dflash_replay_gdn_state_no_check\"") != std::string::npos, "CUDA backend registry must publish state-only GDN replay helper");
    ok &= expect(cuda_ring.find("dflash_cuda_copy_d2d") != std::string::npos, "CUDA ring source must provide unsynchronized DFlash D2D copy helper");
    ok &= expect(cuda_reg.find("\"dflash_cuda_copy_d2d\"") != std::string::npos, "CUDA backend registry must publish DFlash D2D copy helper");
    ok &= expect(cuda_ring.find("dflash_cuda_copy_d2d_no_check") != std::string::npos, "CUDA ring source must provide hot-loop D2D copy helper without per-copy pointer validation");
    ok &= expect(cuda_reg.find("\"dflash_cuda_copy_d2d_no_check\"") != std::string::npos, "CUDA backend registry must publish hot-loop D2D copy helper");
    ok &= expect(memory_recurrent.find("dflash_cuda_copy_d2d_no_check") != std::string::npos, "recurrent rollback copy must use the batched D2D helper");
    ok &= expect(memory_recurrent.find("seq_cp_recurrent_no_sync") != std::string::npos, "recurrent memory must expose DFlash no-sync restore");
    ok &= expect(context_cpp.find("seq_cp_recurrent_no_sync(seq_backup, seq_id") != std::string::npos, "DFlash rollback must avoid synchronous recurrent restore before replay");
    ok &= expect(context_cpp.find("recurrent_restore=%.3f ms") != std::string::npos, "DFlash rollback profiling must expose recurrent restore cost");
    ok &= expect(cparams_h.find("dflash_verify_logits") != std::string::npos, "target verifier reduction must be explicitly gated in cparams");
    ok &= expect(cparams_h.find("dflash_reduced_consumer_active") != std::string::npos, "reduced verifier raw-logit skip must be controlled separately from graph topology");
    ok &= expect(llama_h.find("llama_set_dflash_verify_logits") != std::string::npos, "public API must gate target verifier reduction");
    ok &= expect(llama_h.find("llama_set_dflash_consume_reduced") != std::string::npos, "public API must expose reduced verifier consumption without topology changes");
    ok &= expect(llama_h.find("llama_get_logits_argmax_ith") != std::string::npos, "public API must expose reduced logits by batch row");
    const size_t set_verify_pos = context_cpp.find("void llama_context::set_dflash_verify_logits");
    const size_t set_slots_pos = context_cpp.find("void llama_context::set_dflash_n_slots", set_verify_pos);
    ok &= expect(set_verify_pos != std::string::npos && set_slots_pos != std::string::npos &&
            context_cpp.substr(set_verify_pos, set_slots_pos - set_verify_pos).find("gf_res_prev->reset()") == std::string::npos,
            "DFlash verifier-logit toggles must not reset graph reuse on every verify cycle");
    ok &= expect(qwen35.find("cparams.dflash_verify_logits") != std::string::npos, "Qwen3.5 target graph must emit reduced verifier logits only when gated");
    ok &= expect(qwen35moe.find("cparams.dflash_verify_logits") != std::string::npos, "Qwen3.5-MoE target graph must emit reduced verifier logits only when gated");
    ok &= expect(qwen35.find("ggml_topk_ext(ctx0, cur, topk, 0.0f, 0)") != std::string::npos, "Qwen3.5 target verifier top-K must emit raw logit candidates");
    ok &= expect(qwen35moe.find("ggml_topk_ext(ctx0, cur, topk, 0.0f, 0)") != std::string::npos, "Qwen3.5-MoE target verifier top-K must emit raw logit candidates");
    ok &= expect(cuda_argmax.find("temp > 0.0f && seed != 0") != std::string::npos, "CUDA argmax/top-K must skip logsumexp for deterministic verifier top-K");
    ok &= expect(cuda_argmax.find("const float raw_logit = heap_idx[i] >= 0 ? rowx[heap_idx[i]] : -FLT_MAX;") != std::string::npos, "CUDA deterministic top-K must return raw logits, not zero scores");
    ok &= expect(cuda_argmax.find("cub::DeviceTopK::MaxPairs") != std::string::npos, "CUDA deterministic top-K must use CUB fast path when available");
    ok &= expect(sampling_h.find("common_sampler_sample_reduced_and_accept_n") != std::string::npos, "common sampler must expose reduced-candidate verifier sampling");
    ok &= expect(sampling_cpp.find("llama_sampler_apply(gsmpl->chain, &gsmpl->cur_p)") != std::string::npos, "reduced verifier must still run the sampler chain");
    ok &= expect(sampling_cpp.find("gsmpl->cur_p = { gsmpl->cur.data(), gsmpl->cur.size(), -1, false }") != std::string::npos, "reduced verifier sampler must tolerate unsorted GPU top-K candidates");
    ok &= expect(sampling_cpp.find("common_reasoning_budget_get_state(gsmpl->rbudget) != REASONING_BUDGET_FORCING") != std::string::npos, "reduced verifier must allow passthrough reasoning-budget tracking");
    ok &= expect(sampling_cpp.find("llama_sampler_apply(gsmpl->rbudget, &gsmpl->cur_p)") != std::string::npos, "reduced verifier must preserve reasoning-budget sampler state");
    ok &= expect(server_context.find("dflash_select_reduced_verify_plan") != std::string::npos, "server must explicitly choose reduced verifier eligibility");
    ok &= expect(server_context.find("sampling.has_logit_bias() || sampling.ignore_eos") != std::string::npos, "server must not treat inactive precomputed EOG biases as active logit bias");
    ok &= expect(server_context.find("finite-reasoning-budget") != std::string::npos, "server must disable reduced verifier only for finite reasoning budgets");
    ok &= expect(server_context.find("llama_set_dflash_verify_logits(ctx, dflash_verify_graph_enabled") != std::string::npos, "server must enable reduced verifier graph once per eligible batch");
    ok &= expect(server_context.find("llama_set_dflash_verify_logits(ctx, dflash_reduce_this_view") == std::string::npos, "server must not toggle reduced verifier graph around each ubatch");
    ok &= expect(server_context.find("llama_set_dflash_verify_logits(ctx, false, 1)") == std::string::npos, "server must not disable reduced verifier graph after each ubatch");
    ok &= expect(server_context.find("llama_set_dflash_consume_reduced(ctx, dflash_reduce_this_view)") != std::string::npos, "server must toggle only the raw-logit consumption flag per ubatch");
    ok &= expect(server_context.find("shrunk recurrent state to %d cells before draft load") != std::string::npos, "server must shrink recurrent backup cells before draft model load");
    ok &= expect(server_context.find("expanded recurrent state to %d cells before speculative GPU buffers") != std::string::npos, "server must expand recurrent backup cells before DFlash slot/GPU buffer init");
    ok &= expect(llama_h.find("llama_context_recurrent_expand") != std::string::npos, "public API must expose context-level recurrent expansion with graph invalidation");
    ok &= expect(server_context.find("llama_context_recurrent_shrink(ctx, n_parallel_user)") != std::string::npos, "server recurrent shrink must invalidate the context scheduler graph cache");
    ok &= expect(server_context.find("llama_context_recurrent_expand(ctx, n_seq_max_full)") != std::string::npos, "server recurrent expansion must invalidate the context scheduler graph cache");
    ok &= expect(context_cpp.find("bool llama_context::resize_recurrent_memory") != std::string::npos, "context recurrent resize must be implemented at context level");
    ok &= expect(context_cpp.find("sched_need_reserve = true") != std::string::npos, "context recurrent resize must reserve a fresh scheduler graph after tensor reallocation");
    ok &= expect(server_context.find("dflash_profit_controller") == std::string::npos, "DFlash profit controller must use the normal adaptive depth probe path");
    ok &= expect(server_context.find("dflash_fixed_verify_shape") == std::string::npos, "DFlash profit controller must not override adaptive depth decisions");
    ok &= expect(server_context.find("continuing would corrupt recurrent replay") != std::string::npos, "server must abort instead of continuing after recurrent backup expansion failure");
    ok &= expect(graph_h.find("cparams.cb_eval") != std::string::npos, "graph reuse must key on eval callback topology");
    ok &= expect(graph_h.find("cparams.cb_eval_user_data") != std::string::npos, "graph reuse must key on eval callback user data");
    ok &= expect(graph_h.find("cparams.hidden_gpu_n_seqs") != std::string::npos, "graph reuse must key on GPU hidden capture topology");
    ok &= expect(graph_h.find("cparams.tape_gpu != nullptr") != std::string::npos, "graph reuse must key on GPU tape topology");
    ok &= expect(context_cpp.find("!dflash_reduced_consumed && needs_raw_logits") != std::string::npos, "raw logits must still be copied for fallback views when stable top-K is present");
    ok &= expect(server_context.find("dflash profile: accept n_draft=%zu ids=%zu") != std::string::npos, "server must profile DFlash accept subphases");
    ok &= expect(server_context.find("dflash_sample_reduced_verify") != std::string::npos, "server must consume reduced verifier logits");
    ok &= expect(server_context.find("falling back is unsafe because raw logits were not copied") != std::string::npos, "server must not silently fall back after reduced raw-logit skip");
    ok &= expect(server_context.find("spec_pad_i_batch") != std::string::npos, "server must track DFlash verifier padding rows separately from real draft rows");
    ok &= expect(server_context.find("padded DFlash verifier batch") != std::string::npos, "server must pad short flat DFlash verify batches to stabilize target graph shape");
    ok &= expect(server_context.find("active_verify_draft_max") != std::string::npos, "server must pad reduced verifier batches only to the active adaptive draft depth");
    ok &= expect(server_context.find("std::max(n_draft_max, original_n_max)") == std::string::npos, "DFlash verifier padding must not force adaptive depth to pay for configured max depth");
    ok &= expect(server_context.find("rows_available") != std::string::npos, "server DFlash verifier padding must respect batch and ubatch capacity");
    ok &= expect(server_context.find("for (int idx : slot.spec_pad_i_batch)") != std::string::npos, "reduced verifier coverage must account for explicit padding rows");
    ok &= expect(server_context.find("const bool had_dflash_padding = !slot.spec_pad_i_batch.empty()") != std::string::npos, "server must remember verifier padding through accept bookkeeping");
    ok &= expect(server_context.find("const bool all_accepted_flat = (ids.size() == n_draft + 1) && !had_dflash_padding") != std::string::npos, "DFlash verifier padding must force rollback even when all real draft tokens were accepted");

    return ok ? 0 : 1;
}
