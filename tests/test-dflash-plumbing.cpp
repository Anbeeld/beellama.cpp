#include "llama-context.h"
#include "speculative.h"

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
    const std::string server_task = read_file(root + "/tools/server/server-task.cpp");
    const std::string chat_auto_parser_generator = read_file(root + "/common/chat-auto-parser-generator.cpp");
    const std::string speculative = read_file(root + "/common/speculative.cpp");
    const std::string speculative_h = read_file(root + "/common/speculative.h");
    const std::string dflash_draft = read_file(root + "/src/models/dflash_draft.cpp");
    const std::string memory_recurrent = read_file(root + "/src/llama-memory-recurrent.cpp");
    const std::string qwen35 = read_file(root + "/src/models/qwen35.cpp");
    const std::string qwen35moe = read_file(root + "/src/models/qwen35moe.cpp");
    const std::string gemma4_iswa = read_file(root + "/src/models/gemma4-iswa.cpp");
    const std::string convert_py = read_file(root + "/convert_hf_to_gguf.py");
    const std::string graph_h = read_file(root + "/src/llama-graph.h");
    const std::string model_cpp = read_file(root + "/src/llama-model.cpp");
    const std::string cuda_ring = read_file(root + "/ggml/src/ggml-cuda/cross-ring-interleave.cu");
    const std::string cuda_cpp = read_file(root + "/ggml/src/ggml-cuda/ggml-cuda.cu");
    const std::string cuda_argmax = read_file(root + "/ggml/src/ggml-cuda/argmax.cu");
    const std::string cuda_gdn = read_file(root + "/ggml/src/ggml-cuda/gated_delta_net.cu");
    const std::string cuda_reg = read_file(root + "/ggml/src/ggml-cuda/ggml-cuda.cu");
    const std::string cuda_fattn = read_file(root + "/ggml/src/ggml-cuda/fattn.cu");
    const std::string cuda_fattn_vec_q4_0_q4_0 = read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu");
    const std::string cuda_fattn_vec_q5_0_q5_1 = read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q5_1.cu");
    const std::string cuda_template_generator = read_file(root + "/ggml/src/ggml-cuda/template-instances/generate_cu_files.py");

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
    ok &= expect(context_cpp.find("dflash_capture->hidden_gpu.size()") != std::string::npos &&
                 context_cpp.find("dflash_capture->prefill_gpu.size()") != std::string::npos,
        "active DFlash slot must work without GPU tape and account for hidden-only/prefill-only contexts");
    ok &= expect(context_cpp.find("get_tensor_data(gpu_layer->qkv") != std::string::npos, "conv rebuild must read QKV from GPU tape through placement-safe tensor access");
    ok &= expect(graph_cpp.find("t_logits_argmax = nullptr;") != std::string::npos, "graph reset must clear reduced logits output pointer");
    ok &= expect(context_cpp.find("logits_argmax_buf.clear();") != std::string::npos, "decode must clear stale reduced logits ids");
    ok &= expect(context_cpp.find("logits_argmax_prob_buf.clear();") != std::string::npos, "decode must clear stale reduced logits probabilities");
    ok &= expect(context_cpp.find("const int64_t step = 128;") != std::string::npos, "DFlash cross buckets must avoid the 513-to-1024 latency cliff");
    ok &= expect(cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)") != std::string::npos,
        "CUDA FlashAttention must dispatch q4_0/q4_0 D=512 for Gemma4 non-SWA KV cache");
    ok &= expect(cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)") != std::string::npos,
        "CUDA FlashAttention all-quant dispatch must include D=512 q5 K/V cache pairs");
    ok &= expect(cuda_fattn_vec_q4_0_q4_0.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);") != std::string::npos,
        "q4_0/q4_0 FlashAttention template instance must include D=512");
    ok &= expect(cuda_fattn_vec_q5_0_q5_1.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1);") != std::string::npos,
        "q5_0/q5_1 FlashAttention template instance must include D=512");
    ok &= expect(cuda_template_generator.find("DECL_FATTN_VEC_CASE(512, {type_k}, {type_v});") != std::string::npos,
        "CUDA template generator must preserve D=512 quantized FlashAttention vector instances");
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
    const size_t kv_cache_init_fn = context_cpp.find("bool llama_context::dflash_kv_cache_init");
    const size_t kv_cache_update_fn = context_cpp.find("bool llama_context::dflash_kv_cache_update");
    const size_t kv_cache_update_graph = context_cpp.find("ggml_backend_graph_compute_async(gpu_backend, gf)", kv_cache_update_fn);
    ok &= expect(context_h.find("dflash_kv_cache_multi_gpu_fallback_logged") != std::string::npos, "DFlash K/V cache multi-GPU fallback logging must be per context");
    ok &= expect(
        kv_cache_init_fn != std::string::npos &&
        kv_cache_update_fn != std::string::npos &&
        context_cpp.find("multi-GPU drafter detected", kv_cache_init_fn) < kv_cache_update_fn,
        "DFlash drafter K/V cache must fall back before initializing on multi-GPU draft placement");
    ok &= expect(
        kv_cache_update_fn != std::string::npos &&
        kv_cache_update_graph != std::string::npos &&
        context_cpp.find("model.n_devices() > 1", kv_cache_update_fn) < kv_cache_update_graph,
        "DFlash K/V cache update must not compute a split drafter graph on one CUDA backend");
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
    ok &= expect(context_h.find("bool gpu_capture_enabled = true") != std::string::npos, "DFlash capture must be able to force CPU hidden callbacks");
    ok &= expect(context_h.find("multi_gpu_replay_fallback_logged") != std::string::npos, "DFlash capture must log multi-GPU recurrent replay fallback once");
    ok &= expect(llama_h.find("llama_set_dflash_gpu_capture") != std::string::npos, "public DFlash API must expose GPU capture gating");
    ok &= expect(context_cpp.find("allocate_hidden_gpu(n_slots, max_tokens)") != std::string::npos, "GPU tape allocation must allocate hidden capture buffers too");
    ok &= expect(context_cpp.find("dflash_skip_eval_callback ? nullptr : dflash_eval_callback") != std::string::npos, "eligible DFlash verifier graph must disable eval callback, including suppressed no-intersection ubatches");
    ok &= expect(context_cpp.find("const bool dflash_graph_tape_ready") != std::string::npos, "DFlash decode must gate GPU tape copies separately from hidden capture");
    ok &= expect(context_cpp.find("dflash_graph_hidden_ready =\n                            !dflash_capture->hidden_gpu.empty()") != std::string::npos, "GPU hidden graph capture must not depend on active tape recording");
    ok &= expect(context_cpp.find("dflash_tape_gpu * graph_tp = dflash_graph_tape_ready ? tp : nullptr") != std::string::npos, "GPU tape graph pointers must be disabled when tape recording is inactive");
    ok &= expect(context_cpp.find("multi-GPU target detected") != std::string::npos, "multi-GPU target must fall back from graph-embedded DFlash GPU capture");
    ok &= expect(context_cpp.find("const bool dflash_gpu_capture_ready = model.n_devices() <= 1 && dflash_capture->gpu_capture_enabled") != std::string::npos, "DFlash graph capture must be gated to single-GPU target placement and explicit GPU capture policy");
    ok &= expect(context_cpp.find("const bool multi_gpu_target = model.n_devices() > 1;") != std::string::npos, "DFlash replay must detect multi-GPU target placement before choosing GPU replay");
    ok &= expect(context_cpp.find("exact CUDA DFlash replay unavailable, using CPU recurrent replay fallback") != std::string::npos, "DFlash replay must log only the multi-GPU CPU replay fallback");
    ok &= expect(context_cpp.find("tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);\n        tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);") != std::string::npos,
        "DFlash multi-GPU replay fallback must update both S-state and conv state");
    ok &= expect(speculative.find("common_dflash_gpu_ring_allowed") != std::string::npos, "DFlash GPU cross ring must have an explicit multi-GPU policy");
    ok &= expect(speculative.find("llama_model_n_devices(llama_get_model(ctx_tgt))") != std::string::npos, "DFlash GPU cross ring policy must check target placement");
    ok &= expect(speculative.find("llama_model_n_devices(llama_get_model(ctx_dft))") != std::string::npos, "DFlash GPU cross ring policy must check drafter placement");
    ok &= expect(
        speculative.find("common_dflash_gpu_ring_allowed(ctx_tgt, ctx_dft)") != std::string::npos &&
        speculative.find("common_dflash_gpu_ring_allowed(ctx_tgt, ctx_dft)") < speculative.find("llama_dflash_cross_ring_gpu_init(ctx_dft"),
        "DFlash GPU cross ring must be disabled before allocation on multi-GPU target or drafter placement");
    ok &= expect(speculative.find("llama_set_dflash_gpu_capture(ctx_tgt, gpu_ring_requested)") != std::string::npos,
        "DFlash must force CPU hidden capture whenever the GPU cross ring is unavailable");
    ok &= expect(context_cpp.find("buf.n_tokens <= 0 || buf.data.empty()") != std::string::npos, "empty CPU hidden buffers must not mask GPU hidden buffers");
    ok &= expect(context_cpp.find("hidden->n_tokens = 0") != std::string::npos, "GPU hidden token counts must reset between decodes");
    ok &= expect(context_cpp.find("ggml_backend_tensor_get(tensor, staging.data()") != std::string::npos, "GPU hidden D2D fallback must preserve correctness via readback");
    ok &= expect(qwen35.find("cparams.hidden_gpu_n_seqs > 0") != std::string::npos, "Qwen3.5 must graph-copy l_out into GPU hidden buffers");
    ok &= expect(qwen35moe.find("cparams.hidden_gpu_n_seqs > 0") != std::string::npos, "Qwen3.5-MoE must graph-copy l_out into GPU hidden buffers");
    ok &= expect(gemma4_iswa.find("cparams.hidden_gpu_n_seqs > 0") != std::string::npos, "Gemma4-ISWA must graph-copy l_out into GPU hidden buffers");
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
    ok &= expect(
        set_tensor_fn != std::string::npos &&
        set_tensor_end != std::string::npos &&
        cuda_ring.find("cudaPointerGetAttributes(&dst_attr", set_tensor_fn) < set_tensor_end &&
        cuda_ring.find("cudaMemcpyPeerAsync", set_tensor_fn) < set_tensor_end,
        "dflash_cross_ring_gpu_set_tensor must handle cross-device D2D copies explicitly");
    const size_t cuda_graph_check = cuda_cpp.find("static bool ggml_cuda_graph_check_compability");
    const size_t cuda_graph_check_end = cuda_cpp.find("static const void * ggml_cuda_graph_get_key", cuda_graph_check);
    ok &= expect(
        cuda_graph_check != std::string::npos &&
        cuda_graph_check_end != std::string::npos &&
        cuda_cpp.find("ggml_backend_cuda_context * cuda_ctx", cuda_graph_check) < cuda_graph_check_end &&
        cuda_cpp.find("for (int j = 0; j < GGML_MAX_SRC; ++j)", cuda_graph_check) < cuda_graph_check_end &&
        cuda_cpp.find("ggml_cuda_buffer_visible_to_backend(cuda_ctx", cuda_graph_check) < cuda_graph_check_end &&
        cuda_cpp.find("ggml_backend_cuda_buffer_type(cuda_ctx->device)") != std::string::npos,
        "CUDA graph compatibility must reject non-local source buffers before capture");
    ok &= expect(cuda_cpp.find("source buffer is not visible to CUDA backend device") != std::string::npos,
        "CUDA backend assert diagnostics must report the non-local source tensor");
    ok &= expect(context_h.find("tape_replay_conv_gpu") != std::string::npos, "DFlash must declare GPU conv rebuild fast path");
    ok &= expect(context_cpp.find("model.n_devices() <= 1 && tape_replay_conv_gpu(mem_recurrent, cell_idx, n_accepted)") != std::string::npos, "conv replay must try GPU rebuild only for single-GPU target placement");
    ok &= expect(context_cpp.find("const bool use_async_backend = gpu_backend && model.n_devices() <= 1;") != std::string::npos, "conv replay CPU fallback must use async backend copies only for single-GPU target placement");
    ok &= expect(context_cpp.find("ggml_backend_tensor_get_async(gpu_backend, r_tensor") == std::string::npos, "conv replay must not read split recurrent state through the first GPU backend");
    ok &= expect(context_cpp.find("ggml_backend_tensor_set_async(gpu_backend, d.r_tensor") == std::string::npos, "conv replay must not write split recurrent state through the first GPU backend");
    ok &= expect(context_cpp.find("conv_gpu_enqueue=%.3f ms") != std::string::npos, "DFlash profile must report GPU conv enqueue time");
    ok &= expect(cuda_ring.find("k_dflash_rebuild_conv_state") != std::string::npos, "CUDA ring source must provide conv rebuild kernel");
    ok &= expect(cuda_reg.find("\"dflash_rebuild_conv_state\"") != std::string::npos, "CUDA backend registry must publish conv rebuild kernel");
    ok &= expect(context_h.find("replay_direct_gpu") != std::string::npos, "DFlash capture state must track direct GPU replay");
    ok &= expect(context_h.find("std::vector<const void *> replay_sync_ptrs") != std::string::npos, "direct DFlash replay must track every CUDA device that received replay work");
    ok &= expect(context_h.find("tape_replay_gdn_direct_gpu") != std::string::npos, "DFlash must declare direct GPU GDN replay fast path");
    ok &= expect(context_h.find("tape_replay_gdn_direct_from_cpu_tape") != std::string::npos, "DFlash must expose exact CUDA replay from CPU tape for split-device targets");
    ok &= expect(context_h.find("tape_replay_conv_gpu_from_cpu_tape") != std::string::npos, "DFlash must expose CUDA conv rebuild from CPU tape for split-device targets");
    ok &= expect(context_cpp.find("tape_replay_gdn_direct_gpu(mem_recurrent, cell_idx, n_accepted)") != std::string::npos, "tape replay must try direct GPU GDN replay before ggml graph replay");
    ok &= expect(context_cpp.find("tape_replay_gdn_direct_from_cpu_tape(mem_recurrent, cell_idx, n_accepted)") != std::string::npos, "split-device DFlash replay must try exact CUDA replay from CPU tape before CPU fallback");
    ok &= expect(context_cpp.find("tape_replay_conv_gpu_from_cpu_tape(mem_recurrent, cell_idx, n_accepted, seq_id)") != std::string::npos, "split-device DFlash conv rebuild must try CUDA rebuild from CPU tape before CPU fallback");
    ok &= expect(context_cpp.find("const int64_t hk = hv % H_k;") != std::string::npos, "CPU DFlash GDN replay fallback must match CUDA head mapping");
    ok &= expect(context_cpp.find("hv / head_ratio") == std::string::npos, "CPU DFlash GDN replay must not use grouped value-to-key head mapping");
    ok &= expect(context_cpp.find("fn_prepare(launch.state)") != std::string::npos, "direct DFlash GPU replay must set the current CUDA device per recurrent layer");
    ok &= expect(context_cpp.find("for (const void * ptr : dflash_capture->replay_sync_ptrs)") != std::string::npos, "direct DFlash GPU replay sync must wait on every touched CUDA device");
    ok &= expect(context_cpp.find("const ggml_status replay_status = ggml_backend_graph_compute_async(gpu_backend, graph);") != std::string::npos, "DFlash GPU replay graph launch status must be checked");
    ok &= expect(context_cpp.find("GPU DFlash recurrent replay graph failed") != std::string::npos, "DFlash GPU replay graph failure must fall back instead of leaving pending replay state");
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
    ok &= expect(memory_recurrent.find("fn_prepare(dst)") != std::string::npos, "split recurrent rollback copy must prepare the CUDA device for each destination tensor");
    ok &= expect(memory_recurrent.find("sync_ptrs.push_back(dst)") != std::string::npos, "split recurrent rollback copy must remember every device stream that may need synchronization");
    ok &= expect(memory_recurrent.find("seq_cp_recurrent_no_sync") != std::string::npos, "recurrent memory must expose DFlash no-sync restore");
    ok &= expect(context_cpp.find("seq_cp_recurrent_no_sync(seq_backup, seq_id") != std::string::npos, "DFlash rollback must avoid synchronous recurrent restore before replay");
    ok &= expect(context_cpp.find("recurrent_restore=%.3f ms") != std::string::npos, "DFlash rollback profiling must expose recurrent restore cost");
    ok &= expect(cuda_cpp.find("ggml_cuda_buffer_visible_to_backend(ggml_backend_cuda_context * cuda_ctx") != std::string::npos, "CUDA backend must centralize backend buffer visibility checks");
    ok &= expect(cuda_cpp.find("ggml_backend_cuda_buffer_matches_backend") != std::string::npos, "CUDA async tensor access must share backend/buffer ownership checks");
    ok &= expect(cuda_cpp.find("ggml_backend_tensor_get(tensor, data, offset, size);") != std::string::npos, "CUDA async tensor reads must fall back to the tensor's owning buffer instead of asserting on wrong-device buffers");
    ok &= expect(cuda_cpp.find("ggml_backend_tensor_set(tensor, data, offset, size);") != std::string::npos, "CUDA async tensor writes must fall back to the tensor's owning buffer instead of asserting on wrong-device buffers");
    ok &= expect(cuda_cpp.find("unsupported buffer type") == std::string::npos, "CUDA backend must not assert on wrong-device async tensor access");
    ok &= expect(cuda_cpp.find("ggml_cuda_graph_check_compability(ggml_backend_cuda_context * cuda_ctx") != std::string::npos, "CUDA graph compatibility must know the active CUDA backend");
    ok &= expect(cuda_cpp.find("for (int j = 0; j < GGML_MAX_SRC; ++j)") != std::string::npos, "CUDA graph compatibility must inspect every source buffer");
    ok &= expect(cuda_cpp.find("ggml_cuda_log_nonlocal_src_buffer") != std::string::npos, "CUDA backend must log non-local source tensor details");
    ok &= expect(cuda_cpp.find("source buffer is not visible to CUDA backend device") != std::string::npos, "CUDA backend diagnostics must report non-local source tensors");
    ok &= expect(cuda_cpp.find("assert(src_visible)") == std::string::npos, "CUDA backend must fail closed instead of asserting on non-local source buffers");
    ok &= expect(cuda_cpp.find("return GGML_STATUS_FAILED;") != std::string::npos, "CUDA backend must propagate non-local source buffer failures");
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
    ok &= expect(gemma4_iswa.find("cparams.dflash_verify_logits") != std::string::npos, "Gemma4-ISWA target graph must emit reduced verifier logits only when gated");
    ok &= expect(gemma4_iswa.find("ggml_topk_ext(ctx0, cur, topk, 0.0f, 0)") != std::string::npos, "Gemma4-ISWA target verifier top-K must emit raw logit candidates");
    ok &= expect(gemma4_iswa.find("#include <algorithm>") != std::string::npos, "Gemma4-ISWA must include <algorithm> for std::max/std::min in verifier path");
    ok &= expect(speculative.find("n_target_features != n_embd * n_target_layers") != std::string::npos,
        "DFlash must validate n_target_features against n_embd * n_target_layers");
    ok &= expect(speculative.find("target_layer_ids contain duplicates") != std::string::npos,
        "DFlash must reject duplicate target_layer_ids");
    ok &= expect(speculative.find("target_layer_id") != std::string::npos &&
                 speculative.find("outside target layer range") != std::string::npos,
        "DFlash must validate target_layer_ids against target model layer range");
    ok &= expect(speculative.find("dflash: contract ok:") != std::string::npos,
        "DFlash must log validated drafter/target contract");
    ok &= expect(dflash_draft.find("n_feat != target_hidden->ne[0]") != std::string::npos,
        "DFlash drafter input must reject cross feature-size mismatches");
    ok &= expect(dflash_draft.find("ggml_backend_tensor_memset(target_hidden, 0, 0, tensor_bytes)") != std::string::npos,
        "DFlash drafter input must zero target_hidden on invalid or missing cross data");
    ok &= expect(convert_py.find("DFlashDraftModel: missing") != std::string::npos,
        "DFlash converter must warn when using default metadata");
    ok &= expect(convert_py.find("DFlashDraftModel metadata:") != std::string::npos,
        "DFlash converter must print DFlash metadata summary");
    // --- PR8 checks: DFlash runtime diagnostics, shape validation, env toggles, CUDA debug ---
    ok &= expect(speculative.find("GGML_DFLASH_FORCE_CPU_CROSS") != std::string::npos,
        "DFlash must support GGML_DFLASH_FORCE_CPU_CROSS env toggle");
    ok &= expect(speculative.find("GGML_DFLASH_VERBOSE_CONTRACT") != std::string::npos,
        "DFlash must support GGML_DFLASH_VERBOSE_CONTRACT env toggle");
    ok &= expect(speculative.find("forced_cpu") != std::string::npos &&
                 speculative.find("allowed=") != std::string::npos &&
                 speculative.find("requested=") != std::string::npos,
        "DFlash GPU hidden capture policy must log allowed/forced_cpu/requested");
    ok &= expect(speculative.find("target_vocab") != std::string::npos &&
                 speculative.find("drafter_vocab") != std::string::npos &&
                 speculative.find("vocab_match") != std::string::npos &&
                 speculative.find("capture_min") != std::string::npos &&
                 speculative.find("capture_max") != std::string::npos,
        "DFlash must log target/drafter vocab sizes and capture layer range");
    ok &= expect(speculative.find("validate_target_hiddens") != std::string::npos,
        "DFlash must have validate_target_hiddens method");
    ok &= expect(speculative.find("hidden slot count mismatch") != std::string::npos,
        "DFlash must warn on hidden slot count mismatch");
    ok &= expect(speculative.find("hidden[%d] shape mismatch") != std::string::npos ||
                 speculative.find("shape mismatch: embd=") != std::string::npos,
        "DFlash must warn on hidden shape mismatch");
    ok &= expect(speculative.find("GGML_DFLASH_FORCE_CPU_CROSS") != std::string::npos,
        "DFlash must log when GPU cross ring is forced off by env var");
    ok &= expect(dflash_draft.find("GGML_DFLASH_INPUT_DEBUG") != std::string::npos,
        "DFlash drafter input must support GGML_DFLASH_INPUT_DEBUG env toggle");
    ok &= expect(dflash_draft.find("feature mismatch single-slot") != std::string::npos,
        "DFlash drafter input must log single-slot feature mismatch under debug");
    ok &= expect(dflash_draft.find("feature mismatch multi-slot") != std::string::npos,
        "DFlash drafter input must log multi-slot feature mismatch under debug");
    ok &= expect(cuda_ring.find("GGML_DFLASH_CUDA_DEBUG") != std::string::npos,
        "CUDA cross ring must support GGML_DFLASH_CUDA_DEBUG env toggle");
    ok &= expect(cuda_ring.find("dflash cuda:") != std::string::npos,
        "CUDA cross ring must log diagnostics under debug env");
    ok &= expect(gemma4_iswa.find("inp_per_layer_reshape") != std::string::npos,
        "Gemma4-ISWA must profile inp_per_layer reshape");
    ok &= expect(gemma4_iswa.find("inp_per_layer_scaled") != std::string::npos,
        "Gemma4-ISWA must profile inp_per_layer scale");
    ok &= expect(gemma4_iswa.find("compact post-KV attention input/mask path") != std::string::npos,
        "Gemma4-ISWA must document why early trim is unsafe");
    ok &= expect(qwen35.find("ggml_topk_ext(ctx0, cur, topk, 0.0f, 0)") != std::string::npos, "Qwen3.5 target verifier top-K must emit raw logit candidates");
    ok &= expect(qwen35moe.find("ggml_topk_ext(ctx0, cur, topk, 0.0f, 0)") != std::string::npos, "Qwen3.5-MoE target verifier top-K must emit raw logit candidates");
    ok &= expect(cuda_argmax.find("temp > 0.0f && seed != 0") != std::string::npos, "CUDA argmax/top-K must skip logsumexp for deterministic verifier top-K");
    ok &= expect(cuda_argmax.find("const float raw_logit = heap_idx[i] >= 0 ? rowx[heap_idx[i]] : -FLT_MAX;") != std::string::npos, "CUDA deterministic top-K must return raw logits, not zero scores");
    ok &= expect(cuda_argmax.find("cub::DeviceTopK::MaxPairs") != std::string::npos, "CUDA deterministic top-K must use CUB fast path when available");
    ok &= expect(sampling_h.find("common_sampler_sample_reduced_and_accept_n") != std::string::npos, "common sampler must expose reduced-candidate verifier sampling");
    ok &= expect(sampling_h.find("common_sampler_blocks_speculative") != std::string::npos, "common sampler must expose grammar/reasoning guard for speculative decoding");
    ok &= expect(sampling_cpp.find("Lazy grammars are safe to speculate while still awaiting their trigger") != std::string::npos, "speculative guard must keep DFlash available before lazy grammar triggers");
    ok &= expect(sampling_cpp.find("if (common_sampler_blocks_speculative(gsmpl))") != std::string::npos, "speculative accept must stop when a token activates grammar/reasoning boundaries");
    ok &= expect(sampling_cpp.find("llama_sampler_apply(gsmpl->chain, &gsmpl->cur_p)") != std::string::npos, "reduced verifier must still run the sampler chain");
    ok &= expect(sampling_cpp.find("gsmpl->cur_p = { gsmpl->cur.data(), gsmpl->cur.size(), -1, false }") != std::string::npos, "reduced verifier sampler must tolerate unsorted GPU top-K candidates");
    ok &= expect(sampling_cpp.find("common_reasoning_budget_get_state(gsmpl->rbudget) != REASONING_BUDGET_FORCING") != std::string::npos, "reduced verifier must allow passthrough reasoning-budget tracking");
    ok &= expect(sampling_cpp.find("llama_sampler_apply(gsmpl->rbudget, &gsmpl->cur_p)") != std::string::npos, "reduced verifier must preserve reasoning-budget sampler state");
    ok &= expect(server_context.find("dflash_select_reduced_verify_plan") != std::string::npos, "server must explicitly choose reduced verifier eligibility");
    ok &= expect(server_context.find("common_sampler_blocks_speculative(slot.smpl.get())") != std::string::npos, "DFlash server path must skip drafting when grammar/reasoning guard requires full sampling");
    ok &= expect(server_context.find("common_sampler_blocks_speculative(smpl)") != std::string::npos, "DFlash rejection sampling must stop at grammar/reasoning boundaries");
    ok &= expect(server_context.find("speculative_flat_result_has_bonus") != std::string::npos, "server must distinguish grammar-boundary stops from bonus-token accepts");
    ok &= expect(server_context.find("n_hidden_keep = ids.empty() ? 0 : n_accepted_draft + 1") != std::string::npos, "DFlash ring/tape keep count must include root plus accepted draft tokens");
    ok &= expect(server_context.find("common_speculative_accept(slot.spec.get(), n_accepted_draft)") != std::string::npos, "speculative stats must count accepted draft tokens, not bonus-token-shaped results");
    ok &= expect(server_context.find("llama_dflash_rollback(ctx, slot.id, seq_backup, slot.n_pos_before_draft, n_hidden_keep)") != std::string::npos, "DFlash rollback must use the hidden-state keep count at grammar boundaries");
    ok &= expect(server_context.find("dflash_suppressed_for_reasoning_tool_marker") != std::string::npos, "server must disable DFlash after raw tool markers inside hidden reasoning without steering generation");
    ok &= expect(server_task.find("state.update_chat_msg(content, true, oaicompat_msg_diffs, true)") != std::string::npos, "streaming responses must filter partial tool-call deltas");
    ok &= expect(server_task.find("task_result_has_complete_partial_tool_calls") != std::string::npos, "streaming responses must allow complete tool-call deltas before final EOS");
    ok &= expect(server_task.find("task_result_filter_incomplete_partial_tool_calls") != std::string::npos, "streaming responses must expose stable tool-call headers without partial arguments");
    ok &= expect(server_task.find("A partial stream may expose the stable tool name/id for UX") != std::string::npos, "partial tool-call streaming must document the header-only reliability boundary");
    ok &= expect(server_task.find("task_result_quarantine_raw_tool_text") != std::string::npos, "streaming responses must quarantine malformed raw tool markers in tool-parsing mode");
    ok &= expect(server_task.find("task_result_pos_is_in_code_fence") != std::string::npos, "raw marker quarantine must avoid code fence content");
    ok &= expect(server_task.find("task_result_starts_with_raw_tool_marker") != std::string::npos, "streaming responses must suppress parser fallback text for wrapperless raw tool calls");
    ok &= expect(server_task.find("task_result_freeze_text_fields") != std::string::npos, "incomplete parsed tool calls must not leak fallback text/reasoning deltas");
    ok &= expect(server_context.find("raw tool marker observed while lazy grammar is enabled") != std::string::npos, "DFlash must suppress after raw tool markers even outside parsed reasoning");
    ok &= expect(server_context.find("server_tail_pos_is_in_code_fence") != std::string::npos, "DFlash raw-marker suppression must avoid fenced code content");
    ok &= expect(server_context.find("server_tail_tool_marker_has_boundary") != std::string::npos, "DFlash raw-marker suppression must avoid embedded string false positives");
    ok &= expect(chat_auto_parser_generator.find("allow_direct_func_start") != std::string::npos, "tag-style parsers must accept valid direct function starts without outer wrappers");
    ok &= expect(chat_auto_parser_generator.find("autoparser.tools.function.name_prefix") != std::string::npos, "lazy grammar triggers must include structural function markers");
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
    ok &= expect(server_context.find("const bool all_accepted_flat = (n_accepted_draft == (int) n_draft) && !had_dflash_padding") != std::string::npos, "DFlash verifier padding must force rollback even when all real draft tokens were accepted");

    // DFlashDraftModel converter vocab path checks
    ok &= expect(convert_py.find("_is_gemma4_dflash") != std::string::npos, "DFlash converter must have Gemma4 detection helper");
    ok &= expect(convert_py.find("_set_vocab_gemma4_hf_bpe") != std::string::npos, "DFlash converter must have Gemma4 HF/BPE vocab helper");
    ok &= expect(convert_py.find("visible_tokens = {") != std::string::npos, "DFlash Gemma4 vocab helper must define visible tokens set");
    ok &= expect(convert_py.find("errors=\"replace\"") != std::string::npos, "DFlash Gemma4 vocab helper must decode tokens with errors=replace");
    ok &= expect(convert_py.find("self._set_vocab_gpt2()") != std::string::npos, "DFlash converter must fall back to GPT-2 vocab for non-Gemma drafters");

    ok &= expect(server_context.find("should_flush_dflash_prefill") != std::string::npos,
        "server must gate DFlash prefill flushes to the useful suffix");
    ok &= expect(server_context.find("dflash prefill: skip flush") != std::string::npos,
        "server must log skipped DFlash prefill flushes under DFlash profiling");
    ok &= expect(server_context.find("dflash prefill: suffix flush") != std::string::npos,
        "server must log suffix DFlash prefill flushes under DFlash profiling");
    ok &= expect(server_context.find("params_base.speculative.dflash_cross_ctx") != std::string::npos,
        "DFlash prefill gate must use --spec-dflash-cross-ctx");
    ok &= expect(server_context.find("batch_end <= capture_from") != std::string::npos,
        "DFlash prefill gate must skip batches entirely before the useful suffix");

    ok &= expect(speculative_h.find("common_speculative_set_prefill_capture_enabled") != std::string::npos,
        "DFlash must expose a prefill hidden-capture toggle");

    ok &= expect(speculative.find("set_prefill_capture_enabled(bool") != std::string::npos,
        "DFlash state must implement hidden-capture enable/disable");

    ok &= expect(speculative.find("llama_set_dflash_capture_active") != std::string::npos,
        "DFlash prefill toggle must use the non-destructive capture-active API");

    ok &= expect(llama_h.find("llama_set_dflash_capture_active") != std::string::npos,
        "public API must expose llama_set_dflash_capture_active");

    ok &= expect(context_h.find("capture_active = true") != std::string::npos,
        "dflash_capture_data must track logical capture activity separately from layer config");

    ok &= expect(context_cpp.find("set_dflash_capture_active") != std::string::npos &&
                 context_cpp.find("dflash_capture->capture_active") != std::string::npos,
        "llama_context must implement logical capture-active toggle");

    ok &= expect(speculative_h.find("common_speculative_set_prefill_capture_enabled") != std::string::npos,
        "DFlash must expose a prefill hidden-capture toggle");

    ok &= expect(speculative.find("dflash prefill capture: disabled hidden capture") != std::string::npos,
        "DFlash capture disable path must be profile-loggable");

    ok &= expect(speculative.find("dflash prefill capture: enabled hidden capture") != std::string::npos,
        "DFlash capture re-enable path must be profile-loggable");

    ok &= expect(server_context.find("dflash_capture_needed_for_view") != std::string::npos,
        "server must decide whether DFlash hidden capture is needed before llama_decode");

    ok &= expect(server_context.find("common_speculative_set_prefill_capture_enabled(slot.spec.get(), dflash_capture_needed_for_view)") != std::string::npos,
        "server must toggle DFlash hidden capture before target decode");

    ok &= expect(server_context.find("SLOT_STATE_GENERATING") != std::string::npos &&
                 server_context.find("dflash_capture_needed_for_view = true") != std::string::npos,
        "server must keep DFlash hidden capture enabled for generation/verification");

    ok &= expect(context_cpp.find("capture_active = false") != std::string::npos,
        "set_dflash_capture(nullptr) must mark capture as inactive");

    ok &= expect(context_cpp.find("!dflash_capture->capture_active") != std::string::npos,
        "decode loop must check capture_active before setting eval callback");

    ok &= expect(context_cpp.find("capture_active && !dflash_capture->layer_ids.empty()") != std::string::npos,
        "capture re-enable must check both active flag and layer config");

    // Prefill GPU staging buffer
    ok &= expect(context_h.find("prefill_gpu") != std::string::npos,
        "dflash_capture_data must declare prefill GPU staging buffers");
    ok &= expect(context_h.find("allocate_prefill_gpu") != std::string::npos,
        "llama_context must declare lazy prefill GPU allocation");
    ok &= expect(context_cpp.find("allocate_prefill_gpu") != std::string::npos,
        "decode loop must lazily allocate prefill GPU staging on first suffix batch");
    ok &= expect(context_h.find("prefill_gpu_active()") != std::string::npos,
        "llama_context must expose prefill GPU active check");
    ok &= expect(context_h.find("prefill_gpu_n_tokens") != std::string::npos,
        "llama_context must expose prefill GPU token count query");
    ok &= expect(context_h.find("prefill_gpu_write_hidden") != std::string::npos,
        "llama_context must expose prefill GPU hidden write method");
    ok &= expect(llama_h.find("llama_dflash_prefill_gpu_write_hidden") != std::string::npos,
        "public API must expose prefill GPU hidden D2D ring writes");
    ok &= expect(llama_h.find("llama_dflash_prefill_gpu_active") != std::string::npos,
        "public API must expose prefill GPU active check");
    ok &= expect(llama_h.find("llama_dflash_prefill_gpu_n_tokens") != std::string::npos,
        "public API must expose prefill GPU token count query");
    ok &= expect(cparams_h.find("prefill_gpu_seqs") != std::string::npos,
        "cparams must declare prefill GPU seq pointers for graph builder");
    ok &= expect(cparams_h.find("prefill_gpu_n_seqs") != std::string::npos,
        "cparams must declare prefill GPU seq count for graph builder");
    ok &= expect(graph_h.find("cparams.prefill_gpu_n_seqs") != std::string::npos,
        "graph reuse must key on prefill GPU staging topology");
    ok &= expect(context_cpp.find("use_prefill_staging") != std::string::npos,
        "decode loop must choose between prefill staging and verify hidden_gpu");
    ok &= expect(context_cpp.find("prefill_plan_needs_staging") != std::string::npos,
        "prefill staging must use the planned suffix span, not the current ubatch token count");

    // Prefill span math and flush
    ok &= expect(speculative_h.find("common_dflash_prefill_span") != std::string::npos,
        "DFlash must declare prefill span struct");
    ok &= expect(speculative_h.find("common_speculative_flush_prefill") != std::string::npos,
        "DFlash must expose prefill flush API");
    ok &= expect(speculative.find("flush_prefill(int src_offset") != std::string::npos,
        "speculative state must implement flush_prefill with span parameters");
    ok &= expect(speculative.find("common_speculative_flush_prefill") != std::string::npos,
        "speculative module must implement prefill flush dispatch");

    // Explicit capture source for ring_write
    ok &= expect(speculative.find("dflash_capture_source") != std::string::npos,
        "DFlash ring write must use explicit capture source enum");
    ok &= expect(speculative.find("dflash_capture_source::prefill_gpu_hidden") != std::string::npos,
        "DFlash capture source must distinguish prefill GPU staging");

    // Prefill suffix tracking
    ok &= expect(speculative.find("prefill_suffix_seen") != std::string::npos,
        "DFlash state must track whether suffix prefill was seen");
    ok &= expect(speculative.find("prefill_flush_called") != std::string::npos,
        "DFlash state must track whether flush_prefill was called at all");
    ok &= expect(speculative.find("prefill_flush_requested") != std::string::npos,
        "DFlash state must track total tokens requested across flush_prefill calls");
    ok &= expect(speculative.find("prefill_flush_written") != std::string::npos,
        "DFlash state must track total tokens written across flush_prefill calls");
    ok &= expect(speculative.find("prefill suffix was scheduled but flush_prefill() was never called") != std::string::npos,
        "invariant must distinguish scheduled-but-never-called from called-with-zero-writes");
    ok &= expect(speculative.find("flush was called but wrote 0") != std::string::npos,
        "invariant must distinguish called-with-zero-writes from ring-empty");

    // Source-first flush_prefill: GPU staging path must not gate on CPU hidden count
    ok &= expect(speculative.find("n_src_layers = use_prefill_gpu") != std::string::npos,
        "flush_prefill must determine source layer count independently for GPU staging");
    ok &= expect(speculative.find("n_src_layers = n_target_layers") != std::string::npos,
        "flush_prefill must use n_target_layers for prefill GPU staging source");
    ok &= expect(speculative.find("reason=no-layer-slots") != std::string::npos,
        "flush_prefill must log early-return reason when CPU hidden has no layer slots");
    ok &= expect(speculative.find("reason=no-captured-tokens") != std::string::npos,
        "flush_prefill must log early-return reason when no tokens were captured");
    ok &= expect(speculative.find("reason=empty-clamped-span") != std::string::npos,
        "flush_prefill must log early-return reason when clamped span is empty");
    ok &= expect(speculative.find("n_src_layers") != std::string::npos && speculative.find("dflash prefill flush:") != std::string::npos,
        "flush_prefill must log n_src_layers for diagnostics");

    // Source-first ring_write: GPU staging must use n_target_layers for D2D loop
    ok &= expect(speculative.find("n_src_layers = use_prefill_gpu") != std::string::npos,
        "ring_write must determine source layer count independently for GPU staging");
    ok &= expect(speculative.find("n_src_layers") != std::string::npos,
        "ring_write must use source-aware layer count in both sizing and write loops");

    // Persistent pre-decode flush decisions
    ok &= expect(server_context.find("pending_prefill_flushes") != std::string::npos,
        "server must persist pre-decode flush decisions through decode");
    ok &= expect(server_context.find("common_speculative_flush_prefill") != std::string::npos,
        "server must call prefill flush after decode regardless of slot state");

    // Prefill capture plan: struct, API, and span
    ok &= expect(context_h.find("dflash_prefill_capture_plan") != std::string::npos,
        "dflash_capture_data must declare prefill capture plan struct");
    ok &= expect(context_h.find("prefill_plan") != std::string::npos,
        "dflash_capture_data must include prefill_plan field");
    ok &= expect(context_h.find("std::vector<dflash_prefill_capture_plan> prefill_plans") != std::string::npos,
        "dflash_capture_data must keep prefill capture plans per slot/seq");
    ok &= expect(context_h.find("dflash_prefill_capture_plan prefill_plan;") == std::string::npos,
        "dflash_capture_data must not use one global prefill capture plan");
    ok &= expect(context_h.find("dflash_prefill_capture_begin") != std::string::npos,
        "llama_context must declare prefill capture begin method");
    ok &= expect(context_h.find("dflash_prefill_capture_end") != std::string::npos,
        "llama_context must declare prefill capture end method");
    ok &= expect(context_h.find("dflash_prefill_capture_info") != std::string::npos,
        "llama_context must declare prefill capture info method");
    ok &= expect(llama_h.find("llama_dflash_prefill_capture_begin") != std::string::npos,
        "public API must expose prefill capture begin");
    ok &= expect(llama_h.find("llama_dflash_prefill_capture_end") != std::string::npos,
        "public API must expose prefill capture end");
    ok &= expect(llama_h.find("llama_dflash_prefill_capture_info") != std::string::npos,
        "public API must expose prefill capture info");
    ok &= expect(context_cpp.find("dflash_prefill_capture_begin") != std::string::npos,
        "llama_context must implement prefill capture begin");
    ok &= expect(context_cpp.find("dflash_prefill_capture_end") != std::string::npos,
        "llama_context must implement prefill capture end");
    ok &= expect(context_cpp.find("dflash_prefill_capture_info") != std::string::npos,
        "llama_context must implement prefill capture info query");
    ok &= expect(speculative_h.find("capture_begin") != std::string::npos,
        "common_dflash_prefill_span must include capture_begin");
    ok &= expect(speculative_h.find("capture_end") != std::string::npos,
        "common_dflash_prefill_span must include capture_end");

    // Prefill capture plan: n_written tracking and plan reset
    ok &= expect(context_h.find("n_written") != std::string::npos,
        "prefill capture plan must track n_written");
    ok &= expect(context_cpp.find("plan.n_written = 0") != std::string::npos,
        "capture begin must reset n_written to 0");
    ok &= expect(context_cpp.find("plan.n_written") != std::string::npos,
        "decode loop must advance n_written after graph copy");

    // CParams intersection offsets for graph builder
    ok &= expect(cparams_h.find("dflash_prefill_capture_active") != std::string::npos,
        "cparams must declare prefill capture active flag");
    ok &= expect(cparams_h.find("dflash_prefill_src_offset") != std::string::npos,
        "cparams must declare prefill src_offset");
    ok &= expect(cparams_h.find("dflash_prefill_dst_offset") != std::string::npos,
        "cparams must declare prefill dst_offset");
    ok &= expect(cparams_h.find("dflash_prefill_n_tokens") != std::string::npos,
        "cparams must declare prefill n_tokens");
    ok &= expect(cparams_h.find("dflash_prefill_src_offsets") != std::string::npos,
        "cparams must declare per-seq prefill src offsets for multi-slot capture");
    ok &= expect(cparams_h.find("dflash_prefill_dst_offsets") != std::string::npos,
        "cparams must declare per-seq prefill dst offsets for multi-slot capture");
    ok &= expect(cparams_h.find("dflash_prefill_n_tokens_seqs") != std::string::npos,
        "cparams must declare per-seq prefill token counts for multi-slot capture");
    ok &= expect(graph_h.find("dflash_prefill_capture_active") != std::string::npos,
        "graph reuse must key on prefill capture active flag");
    ok &= expect(graph_h.find("dflash_prefill_n_tokens") != std::string::npos,
        "graph reuse must key on prefill n_tokens");

    // Graph builders must use intersection offsets
    ok &= expect(qwen35.find("dflash_prefill_capture_active") != std::string::npos,
        "Qwen3.5 graph builder must check prefill capture active");
    ok &= expect(qwen35.find("dflash_prefill_src_offsets[s]") != std::string::npos,
        "Qwen3.5 graph builder must use per-seq prefill src offsets");
    ok &= expect(qwen35.find("dflash_prefill_dst_offsets[s]") != std::string::npos,
        "Qwen3.5 graph builder must use per-seq prefill dst offsets");
    ok &= expect(qwen35.find("dflash_prefill_n_tokens_seqs[s]") != std::string::npos,
        "Qwen3.5 graph builder must use per-seq prefill token counts");
    ok &= expect(qwen35.find(": cparams.dflash_prefill_n_tokens") == std::string::npos,
        "Qwen3.5 graph builder must not fall back to scalar prefill token count for zero-copy seqs");
    ok &= expect(qwen35moe.find("dflash_prefill_capture_active") != std::string::npos,
        "Qwen3.5-MoE graph builder must check prefill capture active");
    ok &= expect(qwen35moe.find("dflash_prefill_src_offsets[s]") != std::string::npos,
        "Qwen3.5-MoE graph builder must use per-seq prefill src offsets");
    ok &= expect(qwen35moe.find("dflash_prefill_dst_offsets[s]") != std::string::npos,
        "Qwen3.5-MoE graph builder must use per-seq prefill dst offsets");
    ok &= expect(qwen35moe.find("dflash_prefill_n_tokens_seqs[s]") != std::string::npos,
        "Qwen3.5-MoE graph builder must use per-seq prefill token counts");
    ok &= expect(qwen35moe.find(": cparams.dflash_prefill_n_tokens") == std::string::npos,
        "Qwen3.5-MoE graph builder must not fall back to scalar prefill token count for zero-copy seqs");

    // Server must call capture_begin and capture_end
    ok &= expect(server_context.find("llama_dflash_prefill_capture_begin") != std::string::npos,
        "server must call prefill capture begin before decode");
    ok &= expect(server_context.find("llama_dflash_prefill_capture_end") != std::string::npos,
        "server must call prefill capture end after decode");

    // flush_prefill must use capture_info for GPU path
    ok &= expect(speculative.find("llama_dflash_prefill_capture_info") != std::string::npos,
        "flush_prefill must query capture plan info for GPU staging");
    ok &= expect(speculative.find("capture_info") != std::string::npos,
        "flush_prefill must reference capture_info for n_written");

    // Server should pass the planned source offset. GPU staging normalizes to
    // window-relative offset 0 inside flush_prefill(), while CPU fallback needs
    // the original sub-batch offset.
    ok &= expect(server_context.find("common_speculative_flush_prefill(pf.spec, pf.span.src_offset,") != std::string::npos,
        "server must pass planned source offset to prefill flush");

    // Decode loop must compute intersection offsets
    ok &= expect(context_cpp.find("inter_begin") != std::string::npos,
        "decode loop must compute intersection begin");
    ok &= expect(context_cpp.find("inter_end") != std::string::npos,
        "decode loop must compute intersection end");
    ok &= expect(context_cpp.find("dflash_prefill_src_offset") != std::string::npos,
        "decode loop must set cparams.dflash_prefill_src_offset");
    ok &= expect(context_cpp.find("dflash_prefill_dst_offset") != std::string::npos,
        "decode loop must set cparams.dflash_prefill_dst_offset");

    // Prefill GPU allocation must use plan size, not ubatch size
    ok &= expect(context_cpp.find("max_prefill_plan_tokens") != std::string::npos,
        "decode loop must allocate prefill GPU based on plan n_tokens");

    // Graph reuse must key on prefill intersection offsets
    ok &= expect(graph_h.find("dflash_prefill_src_offset") != std::string::npos,
        "graph reuse must key on DFlash prefill src_offset because ggml_view offsets are topology");
    ok &= expect(graph_h.find("dflash_prefill_dst_offset") != std::string::npos,
        "graph reuse must key on DFlash prefill dst_offset because staging write offset is topology");

    // Per-view capture plan: span must use span_begin/span_end, not global window
    ok &= expect(server_context.find("span.capture_begin = span_begin") != std::string::npos,
        "server must set capture_begin to per-view span_begin, not global capture_from");
    ok &= expect(server_context.find("span.capture_end") != std::string::npos && server_context.find("span_end") != std::string::npos,
        "server must set capture_end to per-view span_end, not global prompt_total");

    // Capture toggle must clear prefill cparams
    ok &= expect(context_cpp.find("cparams.dflash_prefill_capture_active = false") != std::string::npos,
        "set_dflash_capture_active(false) must clear dflash_prefill_capture_active");
    ok &= expect(context_cpp.find("prefill_gpu_seqs[s] = nullptr") != std::string::npos,
        "set_dflash_capture_active(false) must clear prefill_gpu_seqs");

    // Gemma4 is hidden-supported but not tape-supported
    ok &= expect(llama_dflash_gpu_hidden_supported_arch(LLM_ARCH_GEMMA4),
        "Gemma4 must support GPU hidden capture");
    ok &= expect(!llama_dflash_gpu_tape_supported_arch(LLM_ARCH_GEMMA4),
        "Gemma4 must NOT support GPU tape replay");

    // Gemma4 graph builder must have prefill_gpu staging copy (like Qwen35)
    ok &= expect(gemma4_iswa.find("prefill_gpu_n_seqs") != std::string::npos,
        "Gemma4 graph builder must check prefill_gpu_n_seqs");
    ok &= expect(gemma4_iswa.find("dflash_prefill_capture_active") != std::string::npos,
        "Gemma4 graph builder must check dflash_prefill_capture_active");
    ok &= expect(gemma4_iswa.find("dflash_prefill_src_offsets[s]") != std::string::npos,
        "Gemma4 graph builder must use per-seq prefill src offsets");
    ok &= expect(gemma4_iswa.find("dflash_prefill_dst_offsets[s]") != std::string::npos,
        "Gemma4 graph builder must use per-seq prefill dst offsets");
    ok &= expect(gemma4_iswa.find("dflash_prefill_n_tokens_seqs[s]") != std::string::npos,
        "Gemma4 graph builder must use per-seq prefill token counts");
    ok &= expect(gemma4_iswa.find(": cparams.dflash_prefill_n_tokens") == std::string::npos,
        "Gemma4 graph builder must not fall back to scalar prefill token count for zero-copy seqs");

    // flush_prefill must use source-first (prefill_gpu check before CPU validation)
    ok &= expect(speculative.find("use_prefill_gpu && !validate_target_hiddens") != std::string::npos
              || (speculative.find("use_prefill_gpu = llama_dflash_prefill_gpu_active") != std::string::npos
                  && speculative.find("!use_prefill_gpu && !validate_target_hiddens") != std::string::npos),
        "flush_prefill must check prefill GPU before CPU hidden validation");

    // Partial prefill GPU capture must refuse ring write
    ok &= expect(speculative.find("refusing partial ring write") != std::string::npos,
        "flush_prefill must refuse partial GPU capture instead of clamping");

    // Prefill flush log must show capture_source and ring_dst
    ok &= expect(speculative.find("capture_source=") != std::string::npos,
        "prefill flush log must show capture_source");
    ok &= expect(speculative.find("ring_dst=") != std::string::npos,
        "prefill flush log must show ring_dst");

    // Multi-slot DFlash drafter must handle GPU per-seq cross data
    ok &= expect(dflash_draft.find("slot_info[s].gpu") != std::string::npos,
        "multi-slot dflash input must read v_embd_gpu from per-seq cross data");
    ok &= expect(dflash_draft.find("fn_set_tensor_d2d") != std::string::npos
                  && dflash_draft.find("gpu_src") != std::string::npos,
        "multi-slot dflash input must use D2D path for GPU cross data");

    // Invalid mask token must fail at DFlash startup
    ok &= expect(speculative.find("mask_token_id must match target vocab mask token") != std::string::npos,
        "DFlash constructor must validate mask_token_id against target vocab mask");

    // CPU ring validity must gate ring state saving
    ok &= expect(speculative.find("cpu_ring_valid") != std::string::npos,
        "ring state must track cpu_ring_valid to prevent saving stale GPU-only prefill data");

    // Prefill capture complete guard (test helper)
    ok &= expect(common_dflash_prefill_capture_complete_for_test(287, 287),
        "capture complete: captured==requested must be complete");
    ok &= expect(!common_dflash_prefill_capture_complete_for_test(286, 287),
        "capture incomplete: captured<requested must not be complete");
    ok &= expect(common_dflash_prefill_capture_complete_for_test(0, 0),
        "capture complete: zero requested must be complete");

    // Mismatch in server must disable DFlash drafting
    ok &= expect(server_context.find("common_speculative_set_prefill_capture_enabled(pf.spec, false)") != std::string::npos,
        "prefill flush mismatch must disable DFlash capture for the affected slot");

    ok &= expect(speculative_h.find("common_speculative_note_prefill_suffix_scheduled") != std::string::npos,
        "DFlash must expose a separate prefill-suffix scheduled marker");
    ok &= expect(speculative.find("note_prefill_suffix_scheduled") != std::string::npos,
        "DFlash state must implement a separate prefill-suffix scheduled marker");
    ok &= expect(server_context.find("common_speculative_note_prefill_suffix_scheduled(slot.spec.get())") != std::string::npos,
        "server must mark only slots with a scheduled suffix prefill");
    {
        const size_t fn = speculative.find("void set_prefill_capture_enabled(bool enabled) override");
        const size_t end = fn == std::string::npos ? std::string::npos : speculative.find("void note_prefill_suffix_scheduled() override", fn);
        const std::string body = (fn == std::string::npos || end == std::string::npos) ? "" : speculative.substr(fn, end - fn);
        ok &= expect(body.find("prefill_suffix_seen = true") == std::string::npos,
            "generic capture enable must not mark every DFlash slot as having scheduled suffix prefill");
    }

    // CPU ring validity state machine
    ok &= expect(!common_dflash_cpu_ring_valid_after_write_for_test(true,  false, true,  false),
        "GPU-only write must invalidate CPU mirror");
    ok &= expect(common_dflash_cpu_ring_valid_after_write_for_test(false, true,  true,  true),
        "Forced CPU-ring write with all layers copied must restore validity");
    ok &= expect(common_dflash_cpu_ring_valid_after_write_for_test(false, false, false, true),
        "CPU fallback without GPU ring must restore validity when complete");
    ok &= expect(!common_dflash_cpu_ring_valid_after_write_for_test(true,  true,  true,  false),
        "CPU-tracked path with incomplete layer copy must remain invalid");

    // ring_state_save must guard against stale CPU ring
    ok &= expect(speculative.find("ring_state_save") != std::string::npos
                  && speculative.find("if (!cpu_ring_valid)") != std::string::npos,
        "ring_state_save must guard against stale CPU ring data");

    // build_cross_data must refuse stale CPU ring
    ok &= expect(speculative.find("CPU cross ring is stale") != std::string::npos,
        "build_cross_data must refuse to build from stale CPU ring");

    // draft/draft_tree must guard against empty build_cross_data
    ok &= expect(speculative.find("cross_len <= 0") != std::string::npos,
        "draft and draft_tree must bail out when build_cross_data returns <= 0");

    // Duplicate comment cleanup in multi-slot dflash_draft
    {
        size_t pos = 0;
        int count = 0;
        std::string needle = "collect per-slot cross data";
        while ((pos = dflash_draft.find(needle, pos)) != std::string::npos) {
            ++count;
            pos += needle.size();
        }
        ok &= expect(count == 1,
            "multi-slot dflash_draft must have only one collect-per-slot-cross-data comment");
    }

    // DFlash capture token count for single unique-seq ubatch
    ok &= expect(llama_dflash_capture_tokens_per_seq(11, 1, 1) == 11,
        "single unique-seq ubatch must capture n_tokens, not n_seq_tokens");
    ok &= expect(llama_dflash_capture_tokens_per_seq(1939, 1, 1) == 1939,
        "single unique-seq large ubatch must capture n_tokens");
    ok &= expect(llama_dflash_capture_tokens_per_seq(22, 11, 2) == 11,
        "multi-seq ubatch must capture n_seq_tokens per slot");

    // Decode must use dflash_capture_n_tokens for staging/verify routing
    ok &= expect(context_cpp.find("dflash_capture_n_tokens") != std::string::npos,
        "decode must compute dflash_capture_n_tokens for capture routing");
    ok &= expect(context_cpp.find("dflash_capture_n_seqs") != std::string::npos,
        "decode must compute dflash_capture_n_seqs for capture routing");

    // Source-aware ring_write must gate CPU ring tracking on source_has_cpu_hidden
    ok &= expect(speculative.find("source_has_cpu_hidden") != std::string::npos,
        "ring_write must check source_has_cpu_hidden for CPU ring tracking");
    ok &= expect(speculative.find("force_cpu_ring_for_flush") != std::string::npos,
        "flush_prefill must pass force_cpu_ring_for_flush, not literal true");

    // verify_gpu_hidden must appear in flush_prefill source detection
    ok &= expect(speculative.find("verify_gpu_hidden") != std::string::npos,
        "flush_prefill must detect verify_gpu_hidden source when CPU data is absent");

    // Non-overlapping prefill ubatches must suppress eval callback
    ok &= expect(context_cpp.find("dflash_suppress_callback_for_view") != std::string::npos,
        "decode must suppress eval callback for no-intersection prefill ubatches");

    // dflash_diagnostic_debug_enabled must gate per-ubatch route logs
    ok &= expect(context_cpp.find("dflash_diagnostic_debug_enabled") != std::string::npos,
        "per-ubatch route logs must be gated by GGML_DFLASH_DEBUG, not GGML_DFLASH_PROFILE");

    // Graph builders must use dflash_capture_n_tokens/n_seqs
    ok &= expect(gemma4_iswa.find("dflash_capture_n_tokens") != std::string::npos,
        "Gemma4 graph builder must use dflash_capture_n_tokens");
    ok &= expect(qwen35.find("dflash_capture_n_tokens") != std::string::npos,
        "Qwen35 graph builder must use dflash_capture_n_tokens");
    ok &= expect(qwen35moe.find("dflash_capture_n_tokens") != std::string::npos,
        "Qwen35-MoE graph builder must use dflash_capture_n_tokens");

    // capture_layers must not be shadowed by local variable
    ok &= expect(speculative.find("capture_layers.assign") != std::string::npos,
        "constructor must assign member capture_layers, not declare a local that shadows it");

    // Large prefill cannot fallback to 1-token verify capture
    ok &= expect(common_dflash_should_refuse_large_prefill_fallback_for_test(764, false, true),
        "large CPU-only prefill fallback with GPU ring must be refused");
    ok &= expect(!common_dflash_should_refuse_large_prefill_fallback_for_test(4, false, true),
        "small verify-sized CPU fallback must be allowed");
    ok &= expect(!common_dflash_should_refuse_large_prefill_fallback_for_test(764, true, true),
        "prefill GPU capture must not be refused even for large spans");

    // Reduced verifier must not be disabled by reasoning state alone
    ok &= expect(server_context.find("reasoning-active") == std::string::npos
                  || server_context.find("reasoning-active") > server_context.rfind("sampler"),
        "reduced verifier must not reject on reasoning-active; only sampler capability matters");

    // Debug-gated ring mismatch logs
    ok &= expect(speculative.find("common_dflash_debug_logs_enabled") != std::string::npos,
        "ring_write MISMATCH log must be gated by GGML_DFLASH_DEBUG env var");

    // GPU-only writes must invalidate CPU ring even when caller mistakenly asks for CPU tracking
    // source: 0 = cpu_hidden, 1 = verify_gpu_hidden, 2 = prefill_gpu_hidden
    ok &= expect(common_dflash_cpu_ring_valid_after_source_write_for_test(
        false, 0, true, true, true),
        "CPU callback with all layers copied can validate CPU ring");
    ok &= expect(!common_dflash_cpu_ring_valid_after_source_write_for_test(
        true, 0, true, true, false),
        "CPU callback with missing layer data must not validate CPU ring");
    ok &= expect(!common_dflash_cpu_ring_valid_after_source_write_for_test(
        true, 1, true, true, false),
        "Verify GPU is GPU-only even if force_cpu_ring was accidentally true");
    ok &= expect(!common_dflash_cpu_ring_valid_after_source_write_for_test(
        true, 2, true, true, false),
        "Prefill GPU is GPU-only even if force_cpu_ring was accidentally true");
    ok &= expect(common_dflash_cpu_ring_valid_after_source_write_for_test(
        false, 0, false, false, true),
        "CPU fallback without GPU ring validates only when CPU data exists");

    // Non-overlapping prefill ubatches suppress callback
    ok &= expect(llama_dflash_suppress_callback_for_prefill_ubatch_for_test(
        true, true, false),
        "no-intersection prefill ubatch must suppress callback");
    ok &= expect(!llama_dflash_suppress_callback_for_prefill_ubatch_for_test(
        true, true, true),
        "intersecting prefill ubatch must not suppress callback");
    ok &= expect(!llama_dflash_suppress_callback_for_prefill_ubatch_for_test(
        true, false, false),
        "non-staging ubatch must not suppress callback");
    ok &= expect(!llama_dflash_suppress_callback_for_prefill_ubatch_for_test(
        false, true, false),
        "inactive plan must not suppress callback");

    // Planned-span prefill staging must use the planned suffix size, not the
    // current internal ubatch token count.  A 280-token suffix split into
    // 256+24 internal ubatches must keep using prefill staging for both parts.
    ok &= expect(llama_dflash_prefill_plan_needs_staging_for_test(280, 256),
        "280-token plan needs staging even when current ubatch is 256");
    ok &= expect(llama_dflash_prefill_plan_needs_staging_for_test(280, 24),
        "280-token plan needs staging even when current ubatch tail is 24");
    ok &= expect(!llama_dflash_prefill_plan_needs_staging_for_test(4, 4),
        "4-token verify-sized plan does not need staging");
    ok &= expect(!llama_dflash_prefill_plan_needs_staging_for_test(
        LLAMA_DFLASH_MAX_VERIFY_TOKENS, LLAMA_DFLASH_MAX_VERIFY_TOKENS),
        "verify-max plan does not need staging");

    // Tree/indexed update must fail closed when only GPU hidden capture is
    // available (no CPU hidden data). GPU ring + no CPU hidden means the
    // indexed writer cannot access CPU hidden data.
    ok &= expect(common_dflash_tree_update_requires_cpu_hidden_for_test(false, true),
        "GPU ring without CPU hidden must fail closed");
    ok &= expect(!common_dflash_tree_update_requires_cpu_hidden_for_test(true, true),
        "CPU hidden available: indexed writer can proceed");
    ok &= expect(!common_dflash_tree_update_requires_cpu_hidden_for_test(false, false),
        "No GPU ring: CPU fallback world, not the GPU-only hazard");

    return ok ? 0 : 1;
}
