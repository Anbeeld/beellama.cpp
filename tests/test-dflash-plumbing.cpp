#include "llama-context.h"
#include "dflash-profile.h"
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

static std::string read_file_optional(const std::string & path) {
    std::ifstream file(path);
    if (!file.good()) {
        return "";
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

static int count_occurrences(const std::string & text, const std::string & needle) {
    int count = 0;
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

// whitespace-insensitive containment check: collapses every run of whitespace
// (spaces, tabs, newlines) to a single space in both haystack and needle, so
// upstream reformatting (comment-field alignment, line wrapping) does not break
// code invariants that only care about token sequence
static std::string normalize_ws(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    bool in_ws = false;
    for (char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            in_ws = true;
            continue;
        }
        if (in_ws && !out.empty()) {
            out += ' ';
        }
        in_ws = false;
        out += c;
    }
    return out;
}

static bool contains_ws(const std::string & text, const std::string & needle) {
    return normalize_ws(text).find(normalize_ws(needle)) != std::string::npos;
}

static std::string slice_between(const std::string & text, const std::string & begin, const std::string & end) {
    const size_t b = text.find(begin);
    if (b == std::string::npos) {
        return "";
    }
    const size_t e = text.find(end, b + begin.size());
    if (e == std::string::npos) {
        return text.substr(b);
    }
    return text.substr(b, e - b);
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
    const std::string cmake_root = read_file(root + "/CMakeLists.txt");
    const std::string context_cpp = read_file(root + "/src/llama-context.cpp");
    const std::string kv_cache_h = read_file(root + "/src/llama-kv-cache.h");
    const std::string kv_cache_cpp = read_file(root + "/src/llama-kv-cache.cpp");
    const std::string kv_cache_kvarn_h = read_file(root + "/src/llama-kv-cache-kvarn.h");
    const std::string kv_cache_kvarn_cpp = read_file(root + "/src/llama-kv-cache-kvarn.cpp");
    const std::string kv_cache_iswa_cpp = read_file(root + "/src/llama-kv-cache-iswa.cpp");
    const std::string kv_cache_dsa_cpp = read_file(root + "/src/llama-kv-cache-dsa.cpp");
    const std::string llama_kvarn_h = read_file(root + "/src/llama-kvarn.h");
    const std::string dflash_profile_h = read_file(root + "/src/dflash-profile.h");
    const std::string cparams_h = read_file(root + "/src/llama-cparams.h");
    const std::string graph_cpp = read_file(root + "/src/llama-graph.cpp");
    const std::string ggml_backend_cpp = read_file(root + "/ggml/src/ggml-backend.cpp");
    const std::string ggml_backend_meta = read_file(root + "/ggml/src/ggml-backend-meta.cpp");
    const std::string memory_recurrent_cpp = read_file(root + "/src/llama-memory-recurrent.cpp");
    const std::string llama_h = read_file(root + "/include/llama.h");
    const std::string llama_ext_h = read_file(root + "/src/llama-ext.h");
    const std::string sampling_h = read_file(root + "/common/sampling.h");
    const std::string sampling_cpp = read_file(root + "/common/sampling.cpp");
    const std::string ggml_h = read_file(root + "/ggml/include/ggml.h");
    const std::string ggml_backend_h = read_file(root + "/ggml/include/ggml-backend.h");
    const std::string ggml_common_h = read_file(root + "/ggml/src/ggml-common.h");
    const std::string ggml_c = read_file(root + "/ggml/src/ggml.c");
    const std::string ggml_quants_h = read_file(root + "/ggml/src/ggml-quants.h");
    const std::string ggml_quants_c = read_file(root + "/ggml/src/ggml-quants.c");
    const std::string ggml_turbo_quant_c = read_file(root + "/ggml/src/ggml-turbo-quant.c");
    const std::string ggml_cpu_c = read_file(root + "/ggml/src/ggml-cpu/ggml-cpu.c");
    const std::string ggml_cpu_quants_h = read_file(root + "/ggml/src/ggml-cpu/quants.h");
    const std::string ggml_cpu_quants_c = read_file(root + "/ggml/src/ggml-cpu/quants.c");
    const std::string server_cpp = read_file(root + "/tools/server/server.cpp");
    const std::string server_context = read_file(root + "/tools/server/server-context.cpp");
    const std::string server_adaptive_dm_h = read_file(root + "/tools/server/server-adaptive-dm.h");
    const std::string server_task = read_file(root + "/tools/server/server-task.cpp");
    const std::string server_task_h = read_file(root + "/tools/server/server-task.h");
    const std::string chat_auto_parser_generator = read_file(root + "/common/chat-auto-parser-generator.cpp");
    const std::string speculative = read_file(root + "/common/speculative.cpp");
    const std::string speculative_h = read_file(root + "/common/speculative.h");
    const std::string download_h = read_file(root + "/common/download.h");
    const std::string download_cpp = read_file(root + "/common/download.cpp");
    const std::string arg_cpp = read_file(root + "/common/arg.cpp");
    const std::string common_h = read_file(root + "/common/common.h");
    const std::string common_cpp = read_file(root + "/common/common.cpp");
    const std::string llama_bench = read_file(root + "/tools/llama-bench/llama-bench.cpp");
    const std::string dflash_draft = read_file(root + "/src/models/dflash_draft.cpp");
    const std::string delta_net_base = read_file(root + "/src/models/delta-net-base.cpp");
    const std::string arch_cpp = read_file(root + "/src/llama-arch.cpp");
    const std::string memory_h = read_file(root + "/src/llama-memory.h");
    const std::string memory_hybrid_h = read_file(root + "/src/llama-memory-hybrid.h");
    const std::string memory_hybrid = read_file(root + "/src/llama-memory-hybrid.cpp");
    const std::string memory_hybrid_iswa_h = read_file(root + "/src/llama-memory-hybrid-iswa.h");
    const std::string memory_hybrid_iswa = read_file(root + "/src/llama-memory-hybrid-iswa.cpp");
    const std::string memory_recurrent = read_file(root + "/src/llama-memory-recurrent.cpp");
    const std::string qwen35 = read_file(root + "/src/models/qwen35.cpp");
    const std::string qwen35moe = read_file(root + "/src/models/qwen35moe.cpp");
    const std::string gemma4_iswa = read_file(root + "/src/models/gemma4.cpp");
    const std::string convert_py = read_file(root + "/convert_hf_to_gguf.py");
    const std::string dflash_conversion = read_file(root + "/conversion/dflash.py");
    const std::string graph_h = read_file(root + "/src/llama-graph.h");
    const std::string model_cpp = read_file(root + "/src/llama-model.cpp");
    const std::string cuda_ring = read_file(root + "/ggml/src/ggml-cuda/cross-ring-interleave.cu");
    const std::string cuda_cpp = read_file(root + "/ggml/src/ggml-cuda/ggml-cuda.cu");
    const std::string ggml_cuda_kvarn = read_file(root + "/ggml/src/ggml-cuda/kvarn.cu");
    const std::string ggml_cuda_kvarn_h = read_file(root + "/ggml/src/ggml-cuda/kvarn.cuh");
    const std::string cuda_argmax = read_file(root + "/ggml/src/ggml-cuda/argmax.cu");
    const std::string cuda_gdn = read_file(root + "/ggml/src/ggml-cuda/gated_delta_net.cu");
    const std::string cuda_reg = read_file(root + "/ggml/src/ggml-cuda/ggml-cuda.cu");
    const std::string cuda_fattn = read_file(root + "/ggml/src/ggml-cuda/fattn.cu");
    const std::string cuda_common = read_file(root + "/ggml/src/ggml-cuda/common.cuh");
    const std::string cuda_dequantize = read_file(root + "/ggml/src/ggml-cuda/dequantize.cuh");
    const std::string cuda_cpy = read_file(root + "/ggml/src/ggml-cuda/cpy.cu");
    const std::string cuda_cpy_utils = read_file(root + "/ggml/src/ggml-cuda/cpy-utils.cuh");
    const std::string cuda_getrows = read_file(root + "/ggml/src/ggml-cuda/getrows.cu");
    const std::string cuda_set_rows = read_file(root + "/ggml/src/ggml-cuda/set-rows.cu");
    const std::string cuda_fattn_common = read_file(root + "/ggml/src/ggml-cuda/fattn-common.cuh");
    const std::string cuda_fattn_vec = read_file(root + "/ggml/src/ggml-cuda/fattn-vec.cuh");
    const std::string cuda_fattn_kvarn = read_file_optional(root + "/ggml/src/ggml-cuda/fattn-kvarn.cuh");
    const std::string cuda_fattn_mma_kvarn = read_file_optional(root + "/ggml/src/ggml-cuda/fattn-mma-kvarn.cuh");
    const std::string cuda_fattn_mma_kvarn_case = read_file(root + "/ggml/src/ggml-cuda/fattn-mma-kvarn-case.cuh");
    const std::string cuda_fattn_mma_kvarn_case_decl = read_file(root + "/ggml/src/ggml-cuda/fattn-mma-kvarn-case-decl.cuh");
    const std::string cuda_fattn_mma_f16 = read_file(root + "/ggml/src/ggml-cuda/fattn-mma-f16.cuh");
    const std::string cuda_fattn_mma_kvarn_impl = read_file_optional(root + "/ggml/src/ggml-cuda/fattn-mma-kvarn-impl.cuh");
    const std::string cuda_fattn_mma_kvarn_load = read_file(root + "/ggml/src/ggml-cuda/fattn-mma-kvarn-load.cuh");
    const std::string cuda_fattn_mma_kvarn_decode_decl = read_file(root + "/ggml/src/ggml-cuda/fattn-mma-kvarn-decode-decl.cuh");
    const std::string cuda_fattn_mma_kvarn_decode = read_file(root + "/ggml/src/ggml-cuda/fattn-mma-kvarn-decode.cuh");
    const std::string cuda_fattn_kvarn_vec_decl = read_file_optional(root + "/ggml/src/ggml-cuda/fattn-kvarn-vec-decl.cuh");
    const std::string cuda_fattn_kvarn_vec = read_file_optional(root + "/ggml/src/ggml-cuda/fattn-kvarn-vec.cuh");
    const std::string cuda_fattn_mma_kvarn_decode_k4v3 =
        read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-mma-kvarn-decode-instance-k4-v3.cu");
    const std::string cuda_fattn_mma_kvarn_decode_k4v4 =
        read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-mma-kvarn-decode-instance-k4-v4.cu");
    const std::string cuda_fattn_mma_kvarn_generic_d512 =
        read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-mma-kvarn-instance-ncols1_16-ncols2_4.cu");
    const std::string cuda_turbo_quant = read_file(root + "/ggml/src/ggml-cuda/turbo-quant-cuda.cuh");
    const std::string cuda_cmake = read_file(root + "/ggml/src/ggml-cuda/CMakeLists.txt");
    const std::string cuda_fattn_h = read_file(root + "/ggml/src/ggml-cuda/fattn.cuh");
    const std::string cuda_fattn_vec_q4_0_q4_0 = read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu");
    const std::string cuda_fattn_vec_q4_1_q4_1 = read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_1-q4_1.cu");
    const std::string cuda_fattn_vec_q5_0_q5_0 = read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q5_0.cu");
    const std::string cuda_fattn_vec_q5_0_q5_1 = read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_0-q5_1.cu");
    const std::string cuda_fattn_vec_q5_1_q5_1 = read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q5_1-q5_1.cu");
    const std::string cuda_fattn_vec_q8_0_q8_0 = read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu");
    const std::string cuda_fattn_vec_bf16_bf16 = read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu");
    const std::string cuda_fattn_vec_q8_0_turbo3_tcq = read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-turbo3_tcq.cu");
    const std::string cuda_fattn_vec_turbo3_tcq_q8_0 = read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo3_tcq-q8_0.cu");
    const std::string cuda_fattn_vec_turbo3_tcq_turbo3_tcq = read_file(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo3_tcq-turbo3_tcq.cu");
    const std::string cuda_fattn_vec_q6_0_q6_0 = read_file_optional(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q6_0-q6_0.cu");
    const std::string cuda_fattn_vec_q6_0_q8_0 = read_file_optional(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q6_0-q8_0.cu");
    const std::string cuda_fattn_vec_q8_0_q6_0 = read_file_optional(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q6_0.cu");
    const std::string cuda_fattn_vec_q6_1_q6_1 = read_file_optional(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q6_1-q6_1.cu");
    const std::string cuda_fattn_vec_q3_0_q3_0 = read_file_optional(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q3_0-q3_0.cu");
    const std::string cuda_fattn_vec_q3_1_q3_1 = read_file_optional(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q3_1-q3_1.cu");
    const std::string cuda_fattn_vec_q2_0_q2_0 = read_file_optional(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q2_0-q2_0.cu");
    const std::string cuda_fattn_vec_q2_1_q2_1 = read_file_optional(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q2_1-q2_1.cu");
    const std::string cuda_fattn_vec_turbo4_tcq_turbo4_tcq = read_file_optional(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo4_tcq-turbo4_tcq.cu");
    const std::string cuda_fattn_vec_q8_0_turbo4_tcq = read_file_optional(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-turbo4_tcq.cu");
    const std::string cuda_fattn_vec_turbo4_tcq_q8_0 = read_file_optional(root + "/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo4_tcq-q8_0.cu");
    const std::string cuda_template_generator = read_file(root + "/ggml/src/ggml-cuda/template-instances/generate_cu_files.py");
    const std::string cuda_dispatch_generator = read_file(root + "/scripts/gen-fattn-vec-dispatch.py");
    const std::string metal = read_file(root + "/ggml/src/ggml-metal/ggml-metal.metal");
    const std::string vulkan_cpp = read_file(root + "/ggml/src/ggml-vulkan/ggml-vulkan.cpp");
    const std::string vulkan_shader_gen = read_file(root + "/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp");
    const std::string vulkan_kvarn_store = read_file(root + "/ggml/src/ggml-vulkan/vulkan-shaders/kvarn_store.comp");
    const std::string vulkan_kvarn_materialize = read_file_optional(root + "/ggml/src/ggml-vulkan/vulkan-shaders/kvarn_materialize.comp");
    const std::string readme_md = read_file(root + "/README.md");
    const std::string agents_md = read_file(root + "/AGENTS.md");
    const std::string claude_md = read_file(root + "/CLAUDE.md");
    const std::string docs_build_md = read_file(root + "/docs/build.md");
    const std::string quickstart_qwen36_md = read_file(root + "/docs/quickstart-qwen36-dflash.md");
    const std::string quickstart_gemma4_md = read_file(root + "/docs/quickstart-gemma-4-31b-dflash.md");
    const std::string release_workflow = read_file(root + "/.github/workflows/release.yml");
    const std::string release_dispatch_workflow = read_file(root + "/.github/workflows/release-dispatch.yml");

    ok &= expect(dflash_profile_parse_env(nullptr) == 0, "DFlash profile parser must treat missing env as disabled");
    ok &= expect(dflash_profile_parse_env("0") == 0, "DFlash profile parser must treat 0 as disabled");
    ok &= expect(dflash_profile_parse_env("off") == 0, "DFlash profile parser must treat off as disabled");
    ok &= expect((dflash_profile_parse_env("1") & DFLASH_PROFILE_DEFAULT) == DFLASH_PROFILE_DEFAULT,
        "DFlash profile parser must keep GGML_DFLASH_PROFILE=1 as the default useful profile set");
    ok &= expect((dflash_profile_parse_env("1") & DFLASH_PROFILE_TRACE) == 0,
        "DFlash profile parser must not enable trace firehose for GGML_DFLASH_PROFILE=1");
    ok &= expect(dflash_profile_parse_env("replay,copy,prefill") ==
            (DFLASH_PROFILE_REPLAY | DFLASH_PROFILE_COPY | DFLASH_PROFILE_PREFILL),
        "DFlash profile parser must accept comma-separated profile categories");
    ok &= expect(dflash_profile_parse_env("summary replay verify") ==
            (DFLASH_PROFILE_SUMMARY | DFLASH_PROFILE_REPLAY | DFLASH_PROFILE_VERIFY),
        "DFlash profile parser must accept whitespace-separated profile categories");
    ok &= expect(dflash_profile_parse_env("all") == DFLASH_PROFILE_ALL,
        "DFlash profile parser must support all profile categories");
    ok &= expect(dflash_profile_h.find("DFLASH_PROFILE_PREFILL") != std::string::npos &&
                 dflash_profile_h.find("DFLASH_PROFILE_TRACE") != std::string::npos,
        "DFlash profile categories must be centralized in dflash-profile.h");
    ok &= expect(common_cpp.find("return data_tgt.size() + data_dft.size() + ring_data.size();") != std::string::npos,
        "prompt checkpoints must account for DFlash ring state bytes when reporting/cache-sizing checkpoint size");
    ok &= expect(common_cpp.find("ring_data.clear();") != std::string::npos,
        "prompt checkpoints must clear DFlash ring state with target/draft context state");
    ok &= expect(cmake_root.find("GGML_CUDA_ARCH is not a supported CMake option") != std::string::npos &&
                 cmake_root.find("CMAKE_CUDA_ARCHITECTURES=120") != std::string::npos,
        "CMake must fail loudly when users pass obsolete GGML_CUDA_ARCH instead of CMAKE_CUDA_ARCHITECTURES");

    {
        const size_t zero_reuse_reset = server_context.find("n_past == 0");
        const size_t stale_ring_reset = server_context.find("common_speculative_discard_dflash_state(slot.get_spec(), nullptr)");
        const size_t keep_first       = server_context.find("slot.prompt.tokens.keep_first(n_past);");

        ok &= expect(zero_reuse_reset != std::string::npos &&
                     stale_ring_reset != std::string::npos &&
                     keep_first       != std::string::npos &&
                     zero_reuse_reset < stale_ring_reset &&
                     stale_ring_reset < keep_first,
            "server must discard stale DFlash ring state before processing a prompt with zero reusable prefix tokens");
    }
    {
        const size_t checkpoint_start = server_context.find("bool do_checkpoint = effective_checkpoint_limit > 0;");
        const size_t cache_prompt_gate = server_context.find("do_checkpoint = do_checkpoint && slot.task->params.cache_prompt;");
        const size_t checkpoint_memory_gate = server_context.find("// make a checkpoint of the parts of the memory that cannot be rolled back.");

        ok &= expect(checkpoint_start != std::string::npos &&
                     cache_prompt_gate != std::string::npos &&
                     checkpoint_memory_gate != std::string::npos &&
                     checkpoint_start < cache_prompt_gate &&
                     cache_prompt_gate < checkpoint_memory_gate,
            "server must not create context checkpoints for requests that explicitly disable prompt caching");
    }
    ok &= expect(server_context.find("prompt_cache_boundary_checkpoints_enabled") != std::string::npos &&
                 server_context.find("server_prompt_effective_checkpoint_limit(") != std::string::npos &&
                 server_context.find("server_prompt_checkpoint_creation_allowed(") != std::string::npos &&
                 server_context.find("do_regular_context_checkpoint && do_checkpoint") != std::string::npos,
        "hybrid prompt cache must keep prompt-boundary checkpoints separate from regular context checkpoints");
    {
        const size_t host_size_helper = server_context.find("size_t prompt_host_cache_size() const");
        const size_t checkpoint_budget = server_context.find("prompt_host_cache_would_exceed(new_size)");
        const size_t checkpoint_alloc  = server_context.find("cur.data_tgt.resize(cur_size_tgt)");

        ok &= expect(host_size_helper != std::string::npos &&
                     checkpoint_budget != std::string::npos &&
                     checkpoint_alloc  != std::string::npos &&
                     checkpoint_budget < checkpoint_alloc,
            "server context checkpoints must enforce the host prompt-cache RAM budget before allocating checkpoint state");
    }
    ok &= expect(server_context.find("mark_mtp_draft_context_seq_rm_supported") == std::string::npos,
        "server must not keep the stale MTP seq_rm helper now that upstream probes ctx_dft directly");

    {
        const size_t shared_gate = server_context.find("bool dflash_shared_drafter_batch_allowed");
        const size_t opt_out_env = server_context.find("GGML_DFLASH_SHARED_DRAFT_BATCH");
        const size_t opt_out_check = server_context.find("dflash_shared_drafter_batch_disabled()", shared_gate);
        const size_t draft_batch_call = server_context.find("common_speculative_draft_batch", shared_gate);

        ok &= expect(shared_gate != std::string::npos &&
                     opt_out_env != std::string::npos &&
                     opt_out_check != std::string::npos &&
                     draft_batch_call != std::string::npos &&
                     shared_gate < opt_out_check &&
                     opt_out_check < draft_batch_call,
            "flat DFlash shared drafter batching must be default-enabled with GGML_DFLASH_SHARED_DRAFT_BATCH=0 as the fallback kill switch");
    }

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
    ok &= expect(context_cpp.find("dflash_gpu_hidden_span_in_bounds") != std::string::npos &&
                 count_occurrences(context_cpp, "dflash_gpu_hidden_span_in_bounds(") >= 3 &&
                 context_cpp.find("src_offset_bytes + n_bytes <= ggml_nbytes(tensor)") != std::string::npos,
        "DFlash GPU hidden ring writes must byte-check tensor spans before D2D/readback to avoid backend read OOB asserts");
    ok &= expect(llama_dflash_replay_gdn_supported_s_for_test(16) &&
                 llama_dflash_replay_gdn_supported_s_for_test(128) &&
                 !llama_dflash_replay_gdn_supported_s_for_test(256),
        "DFlash replay must know which CUDA GDN state sizes are supported before building replay graphs");
    ok &= expect(llama_dflash_replay_state_shape_valid_for_test(128, 8, 128 * 128 * 8) &&
                 !llama_dflash_replay_state_shape_valid_for_test(128, 8, 128 * 128 * 8 + 1),
        "DFlash replay must verify S*S*H_v matches recurrent state size before creating state views");
    ok &= expect(llama_dflash_view_span_in_bounds_for_test(1024, 512, 512) &&
                 !llama_dflash_view_span_in_bounds_for_test(1024, 513, 512),
        "DFlash replay must validate tensor byte spans before creating ggml views");
    ok &= expect(context_cpp.find("DFlash recurrent replay view out of bounds") != std::string::npos &&
                 context_cpp.find("llama_dflash_view_span_in_bounds_for_test") != std::string::npos,
        "DFlash recurrent replay must guard result/state ggml views instead of relying on ggml assertions");
    ok &= expect(cuda_argmax.find("argmax < 0") != std::string::npos &&
                 cuda_argmax.find("dst[nrows + row]") != std::string::npos,
        "CUDA reduced-logits argmax must not read rowx[-1] when every row candidate is invalid");
    ok &= expect(context_cpp.find("GGML_ASSERT(batch_inp.token || batch_inp.embd);") != std::string::npos,
        "decode must allow MTP draft batches to carry both token ids and target hidden embeddings");
    ok &= expect(contains_ws(context_cpp, "/*.ctx_type =*/ cparams.ctx_type"),
        "context creation must propagate ctx_type into memory creation so MTP uses its MTP-only KV cache");
    ok &= expect(context_cpp.find("graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type))") != std::string::npos,
        "graph reservation must use the active context graph type so MTP reserves the MTP graph");
    ok &= expect(context_cpp.find("auto * t_h_nextn") != std::string::npos &&
                 context_cpp.find("n_tokens_prev") != std::string::npos &&
                 context_cpp.find("ggml_backend_tensor_get_async(backend_h, t_h_nextn") != std::string::npos &&
                 context_cpp.find("embd_nextn.size > 0") != std::string::npos,
        "decode must copy nextn hidden graph outputs for MTP target and draft contexts");
    ok &= expect(context_cpp.find("embd_nextn.size > 0 && cparams.embeddings_nextn_masked") != std::string::npos,
        "output reordering must only swap masked nextn rows; unmasked nextn rows are indexed by raw token position");
    ok &= expect(speculative.find("target nextn embeddings are not available") != std::string::npos &&
                 speculative.find("draft nextn embedding row") != std::string::npos,
        "MTP speculation must fail cleanly if required nextn hidden rows are unavailable");
    ok &= expect(context_cpp.find("get_tensor_data(gpu_layer->qkv") != std::string::npos, "conv rebuild must read QKV from GPU tape through placement-safe tensor access");
    ok &= expect(graph_cpp.find("t_logits_argmax = nullptr;") != std::string::npos, "graph reset must clear reduced logits output pointer");
    ok &= expect(graph_cpp.find("if (t_h_nextn != nullptr)") != std::string::npos,
        "nextn hidden graph output must be exposed when model graphs create it");
    ok &= expect(context_cpp.find("logits_argmax_buf.clear();") != std::string::npos, "decode must clear stale reduced logits ids");
    ok &= expect(context_cpp.find("logits_argmax_prob_buf.clear();") != std::string::npos, "decode must clear stale reduced logits probabilities");
    ok &= expect(graph_h.find("cparams.cb_eval              == other.cparams.cb_eval") == std::string::npos,
        "graph reuse must not treat eval callback changes as topology changes");
    ok &= expect(context_cpp.find("ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);\n\n    // set the input data") != std::string::npos,
        "process_ubatch must refresh the scheduler eval callback before every compute, including graph reuse");
    ok &= expect(ggml_backend_cpp.find("const bool async_ok = split_backend->iface.cpy_tensor_async") != std::string::npos,
        "backend scheduler copies must avoid re-running async-copy expressions when deciding fallback");
    ok &= expect(ggml_backend_cpp.find("if (need) {\n                    // TODO: pass backend to the callback") != std::string::npos,
        "backend scheduler must only synchronize split backends for eval-callback tensors that are actually needed");
    ok &= expect(ggml_backend_meta.find("static void ggml_backend_meta_get_tensor_async") != std::string::npos &&
                 ggml_backend_meta.find("GGML_ASSERT(offset == 0);") == std::string::npos &&
                 ggml_backend_meta.find("const size_t simple_offset = i_start * chunk_size_j;") != std::string::npos &&
                 ggml_backend_meta.find("ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor") != std::string::npos,
        "Meta backend tensor reads must allow nonzero offsets for split compact outputs");
    ok &= expect(ggml_backend_meta.find("handle_turbo_wht") != std::string::npos &&
                 ggml_backend_meta.find("turbo_wht_split_axis_ne") != std::string::npos &&
                 ggml_backend_meta.find("case GGML_OP_TURBO_WHT:") != std::string::npos,
        "Meta backend split-state propagation must support TurboQuant WHT for tensor-split Turbo/TCQ cache graphs");
    ok &= expect(ggml_backend_h.find("ggml_backend_dev_is_meta") != std::string::npos &&
                 ggml_backend_h.find("ggml_backend_meta_dev_n_devs") != std::string::npos &&
                 ggml_backend_h.find("ggml_backend_meta_dev_simple_dev") != std::string::npos &&
                 ggml_backend_meta.find("bool ggml_backend_dev_is_meta") != std::string::npos &&
                 ggml_backend_meta.find("ggml_backend_dev_t ggml_backend_meta_dev_simple_dev") != std::string::npos &&
                 context_cpp.find("dflash_capture_backend_for_layer") != std::string::npos &&
                 context_cpp.find("allow_meta && ggml_backend_dev_is_meta(dev)") != std::string::npos &&
                 context_cpp.find("ggml_backend_meta_dev_simple_dev(layer_dev") != std::string::npos &&
                 context_cpp.find("capture_backend.backend") != std::string::npos,
        "DFlash tensor-split capture must allocate through the active Meta backend and retain simple-device fallback introspection");
    ok &= expect(context_cpp.find("dflash_tensor_buffer_is_meta") != std::string::npos &&
                 context_cpp.find("if (!tensor_is_meta)") != std::string::npos &&
                 context_cpp.find("ggml_backend_tensor_get(tensor, staging.data(), src_offset_bytes, n_bytes);") != std::string::npos,
        "DFlash GPU ring writes must avoid pointer-based D2D from Meta capture tensors and use backend readback fallback");
    ok &= expect(memory_recurrent_cpp.find("llama_recurrent_tensor_is_meta") != std::string::npos &&
                 memory_recurrent_cpp.find("ggml_backend_dev_is_meta(dev)") != std::string::npos &&
                 memory_recurrent_cpp.find("ggml_backend_tensor_get(tensor, row_data.data(), src_offset, row_bytes);") != std::string::npos &&
                 memory_recurrent_cpp.find("ggml_backend_tensor_set(tensor, row_data.data(), dst_offset, row_bytes);") != std::string::npos,
        "DFlash recurrent backup must copy Meta-backed recurrent rows through backend get/set instead of view pointer copies");
    ok &= expect(cuda_cpp.find("dflash_cuda_backend_wait_for_stream") != std::string::npos,
        "CUDA backend must expose an event wait from GGML compute stream to DFlash per-thread stream");
    ok &= expect(cuda_cpp.find("thread_local cudaEvent_t dflash_wait_events") != std::string::npos,
        "DFlash CUDA stream wait helper must reuse per-thread events instead of allocating on every decode");
    ok &= expect(context_h.find("fn_sync_backend_to_stream") != std::string::npos,
        "DFlash capture state must cache the CUDA stream ordering helper");
    ok &= expect(context_cpp.find("dflash_wait_for_gpu_capture_stream()") != std::string::npos,
        "decode must use fine-grained CUDA stream ordering for graph-copied DFlash hidden tensors");
    ok &= expect(context_cpp.find("const bool dflash_gpu_capture_stream_ready") != std::string::npos &&
                 context_cpp.find("if (dflash_capture && !dflash_gpu_capture_stream_ready)") != std::string::npos,
        "decode must only fall back to full scheduler synchronization when CUDA stream ordering is unavailable");
    ok &= expect(graph_h.find("cparams.hidden_gpu_seqs[i] != other.cparams.hidden_gpu_seqs[i]") != std::string::npos,
        "graph reuse must invalidate when DFlash hidden GPU graph-copy destinations change");
    ok &= expect(graph_h.find("cparams.prefill_gpu_seqs[i] != other.cparams.prefill_gpu_seqs[i]") != std::string::npos,
        "graph reuse must invalidate when DFlash prefill GPU graph-copy destinations change");
    ok &= expect(graph_h.find("cparams.tape_gpu_seqs[i] != other.cparams.tape_gpu_seqs[i]") != std::string::npos,
        "graph reuse must invalidate when DFlash tape graph-copy destinations change");
    ok &= expect(context_cpp.find("const int64_t step = 128;") != std::string::npos, "DFlash cross buckets must avoid the 513-to-1024 latency cliff");
    ok &= expect(cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)") != std::string::npos,
        "CUDA FlashAttention must dispatch q4_0/q4_0 D=512 for Gemma4 non-SWA KV cache");
    ok &= expect(cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)") != std::string::npos,
        "CUDA FlashAttention all-quant dispatch must include D=512 q4_1 K/V cache pairs");
    ok &= expect(cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)") != std::string::npos,
        "CUDA FlashAttention all-quant dispatch must include D=512 q5_0 K/V cache pairs");
    ok &= expect(cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)") != std::string::npos,
        "CUDA FlashAttention all-quant dispatch must include D=512 q5 K/V cache pairs");
    ok &= expect(cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)") != std::string::npos,
        "CUDA FlashAttention all-quant dispatch must include D=512 q5_1 K/V cache pairs");
    ok &= expect(cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)") != std::string::npos,
        "CUDA FlashAttention all-quant dispatch must include D=512 q8_0 K/V cache pairs");
    ok &= expect(cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_BF16, GGML_TYPE_BF16)") != std::string::npos,
        "CUDA FlashAttention all-quant dispatch must include D=512 bf16 K/V cache pairs");
    ok &= expect(cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_TCQ)") != std::string::npos &&
                 cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q8_0)") != std::string::npos,
        "CUDA FlashAttention all-quant dispatch must include D=512 TCQ mixed q8/turbo3 pairs");
    ok &= expect(cuda_fattn.find("hip_native_tcq_decode") != std::string::npos &&
                 cuda_fattn.find("#if defined(GGML_USE_HIP)") != std::string::npos &&
                 contains_ws(cuda_fattn, "!hip_native_tcq_decode && !turbo_decode_native && !prefer_native_vec && turbo_kv"),
        "HIP TCQ decode must stay on the native VEC path instead of dequantizing into generic tile/MMA FlashAttention");
    ok &= expect(cuda_fattn.find("turbo_mma_fused && turbo_mma_supported && Q->ne[1] <= 4") != std::string::npos &&
                 cuda_fattn.find("K->type == GGML_TYPE_TURBO4_0 ||") != std::string::npos &&
                 cuda_fattn.find("K->type == GGML_TYPE_TURBO3_0 ||") != std::string::npos &&
                 cuda_fattn.find("K->type == GGML_TYPE_TURBO2_0") != std::string::npos,
        "fused Turbo MMA must stay limited to decode-sized batches and straight turbo K/V pairs");
    ok &= expect(ggml_h.find("GGML_OP_KVARN_VIEW") != std::string::npos &&
                 ggml_h.find("ggml_kvarn_view") != std::string::npos &&
                 ggml_h.find("GGML_OP_KVARN_MATERIALIZE") == std::string::npos &&
                 ggml_h.find("ggml_kvarn_materialize") == std::string::npos &&
                 ggml_c.find("GGML_OP_KVARN_VIEW") != std::string::npos &&
                 ggml_c.find("ggml_kvarn_view") != std::string::npos &&
                 ggml_c.find("GGML_OP_KVARN_MATERIALIZE") == std::string::npos &&
                 ggml_c.find("ggml_kvarn_materialize") == std::string::npos &&
                 ggml_c.find("ggml_flash_attn_ext_add_kvarn_deps") != std::string::npos &&
                 ggml_c.find("result->src[8] = v_view->src[2];") != std::string::npos,
        "GGML must expose only the no-materialize KVarN view/proxy op and keep raw KVarN FA dependencies alive");
    ok &= expect(kv_cache_kvarn_cpp.find("return cache->materialize") == std::string::npos &&
                 kv_cache_kvarn_cpp.find("llama_kv_cache_kvarn::materialize") == std::string::npos &&
                 kv_cache_kvarn_cpp.find("ggml_kvarn_materialize") == std::string::npos,
        "KVarN cache get_k/get_v must not retain a materialize graph path");
    ok &= expect(kv_cache_kvarn_h.find("ceil((n_batch + n_ubatch) / KVAR_N_GROUP)") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("((n_batch + n_ubatch + KVAR_N_GROUP - 1u) / KVAR_N_GROUP)") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("kvarn_stage_tail_groups(") != std::string::npos &&
                 contains_ws(model_cpp, "cparams.n_ctx_seq, cparams.n_seq_max, cparams.n_batch, cparams.n_ubatch") &&
                 contains_ws(kv_cache_iswa_cpp, "size, n_seq_max, n_batch, n_ubatch") &&
                 contains_ws(memory_hybrid_iswa, "kv_size, n_seq_max, n_batch, n_ubatch"),
        "KVarN non-SWA stage depth must size from n_batch plus one n_ubatch and all construction paths must pass n_batch");
    ok &= expect(kv_cache_kvarn_h.find("SWA tail_groups     = max(2, ceil(min(kv_size, n_swa + n_ubatch) / KVAR_N_GROUP) + 1)") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("const uint32_t swa_stage_cells = std::min(kv_size, n_swa + n_ubatch);") != std::string::npos &&
                 contains_ws(kv_cache_kvarn_cpp, "is_swa ? ((swa_stage_cells + KVAR_N_GROUP - 1u) / KVAR_N_GROUP) + 1u") &&
                 kv_cache_kvarn_cpp.find("is_swa ? 2u : 4u") != std::string::npos,
        "KVarN SWA stage depth must cover the full sliding-window tile span, not only the physical ubatch tail");
    ok &= expect(contains_ws(graph_cpp, "static enum ggml_flash_attn_ext_kvarn_domain llm_kvarn_attn_domain( const llama_cparams & cparams, const ggml_tensor * q, bool is_swa)") &&
                 graph_cpp.find("if (n_q == 1)") != std::string::npos &&
                 graph_cpp.find("GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V") != std::string::npos &&
                 graph_cpp.find("GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_KV_OUTPUT_ORIGINAL") == std::string::npos &&
                 graph_cpp.find("const bool use_kvarn_mixed_domain") != std::string::npos &&
                 graph_cpp.find("const bool use_kvarn_rotated_output_original_domain") == std::string::npos &&
                 ggml_cuda_kvarn.find("K stage is already rotated-domain; V stage is original-domain.") != std::string::npos &&
                 ggml_cuda_kvarn.find("if (value)") != std::string::npos &&
                 ggml_cuda_kvarn.find("kvarn_wht_stage_tile(tile, value);") != std::string::npos &&
                 cuda_fattn_mma_kvarn_impl.find("bool stage;") != std::string::npos &&
                 cuda_fattn_mma_kvarn_impl.find("const bool fast_stage_direct") != std::string::npos &&
                 cuda_fattn_mma_kvarn_impl.find("tile.stage_pos_begin") != std::string::npos &&
                 cuda_fattn_mma_kvarn_impl.find("const bool stage_original = loaded_from_stage && desc.value;") != std::string::npos &&
                 cuda_fattn_mma_kvarn_case_decl.find("ggml_cuda_fattn_kvarn_output_original_domain") == std::string::npos &&
                 cuda_fattn_mma_kvarn_case.find("nbytes_shared_kvarn_output") == std::string::npos &&
                 cuda_fattn_mma_f16.find("flash_attn_ext_kvarn_inverse_wht_output_tile") == std::string::npos &&
                 vulkan_kvarn_store.find("K stage is stored rotated-domain") != std::string::npos &&
                 vulkan_kvarn_store.find("return acc * KVAR_N_WHT_SCALE;") != std::string::npos,
        "KVarN prefill must use mixed K-rotated/V-original attention with per-side live-stage domains and no failed output-WHT route");
    ok &= expect(ggml_backend_cpp.find("pending_new_split_inputs") != std::string::npos &&
                 ggml_backend_cpp.find("split->n_inputs + pending_new_split_inputs > GGML_SCHED_MAX_SPLIT_INPUTS") != std::string::npos,
        "backend scheduler must pre-count all new cross-backend inputs for a node before deciding whether to start a new split");
    ok &= expect(ggml_backend_cpp.find("ggml_backend_sched_allows_bufferless_kvarn_src") != std::string::npos &&
                 ggml_backend_cpp.find("node->op == GGML_OP_FLASH_ATTN_EXT") != std::string::npos &&
                 ggml_backend_cpp.find("src_index == 1 || src_index == 2") != std::string::npos,
        "backend scheduler must not copy bufferless KVarN native K/V proxy sources as split inputs");
    ok &= expect(cuda_fattn.find("#include \"fattn-mma-kvarn.cuh\"") != std::string::npos &&
                 cuda_fattn.find("#include \"fattn-mma-kvarn-case-decl.cuh\"") != std::string::npos &&
                 cuda_fattn.find("#include \"fattn-mma-f16.cuh\"") == std::string::npos &&
                 cuda_fattn_mma_kvarn_case.find("#include \"fattn-mma-f16.cuh\"") != std::string::npos &&
                 cuda_fattn.find("ggml_cuda_flash_attn_ext_mma_kvarn_case") != std::string::npos &&
                 cuda_fattn_mma_kvarn_generic_d512.find("DECL_FATTN_MMA_KVARN_CASE(512, 512, 16, 4);") != std::string::npos &&
                 cuda_fattn.find("ggml_cuda_fattn_kvarn_supported") != std::string::npos &&
                 cuda_fattn_mma_kvarn.find("ggml_cuda_fattn_kvarn_unwrap_view") != std::string::npos &&
                 cuda_fattn_mma_kvarn.find("GGML_OP_KVARN_MATERIALIZE") == std::string::npos,
        "CUDA FlashAttention must recognize KVarN view inputs directly, keep heavy KVarN MMA instantiations sharded, and avoid materialize ancestry in the native route");
    ok &= expect(cuda_fattn_mma_kvarn.find("GGML_CUDA_FATTN_KVARN_NATIVE_MAX_Q") == std::string::npos &&
                 cuda_fattn_mma_kvarn.find("return ggml_cuda_fattn_kvarn_view_supported(device, dst, out);") != std::string::npos &&
                 cuda_fattn.find("GGML_CUDA_FATTN_KVARN_DECODE_MAX_Q") == std::string::npos &&
                 cuda_fattn.find("Q->ne[1] > GGML_CUDA_FATTN_KVARN_DECODE_MAX_Q") == std::string::npos &&
                 cuda_fattn.find("materialize_kvarn_f16") == std::string::npos,
        "KVarN descriptors must stay native for DFlash/prefill query widths without a static split-decode row cap or F16 materialize fallback");
    ok &= expect(cuda_fattn.find("ggml_cuda_flash_attn_ext_mma_kvarn_case") != std::string::npos &&
                 cuda_fattn.find("#include \"fattn-mma-kvarn-decode-decl.cuh\"") != std::string::npos &&
                 cuda_fattn.find("ggml_cuda_fattn_kvarn_decode_mma_kernel") == std::string::npos &&
                 cuda_fattn_mma_kvarn_decode.find(
                     "template<int D, int MAX_GQA, int SPLIT_TOKENS, int NWARPS, int K_BITS, int V_BITS>") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode.find("ggml_cuda_fattn_kvarn_decode_mma_kernel") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode_decl.find(
                     "template<int D, int K_BITS, int V_BITS>") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode.find("#include \"fattn-mma-kvarn-load.cuh\"") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode.find("#include \"fattn-mma-kvarn-impl.cuh\"") == std::string::npos &&
                 cuda_fattn_mma_kvarn_load.find("ggml_cuda_fattn_kvarn_load_rotated") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode_k4v3.find(
                     "DECL_FATTN_KVARN_DECODE_CASE(128, 4, 3);") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode_k4v3.find(
                     "DECL_FATTN_KVARN_DECODE_CASE(256, 4, 3);") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode_k4v3.find(
                     "DECL_FATTN_KVARN_DECODE_CASE(512, 4, 3);") != std::string::npos &&
                 cuda_fattn.find("Q->ne[1] > GGML_CUDA_FATTN_KVARN_DECODE_MAX_Q") == std::string::npos &&
                 cuda_fattn_mma_kvarn_case.find("need_f16_K=false") != std::string::npos &&
                 cuda_fattn_mma_kvarn_case.find("need_f16_V=false") != std::string::npos &&
                 cuda_fattn_mma_kvarn_impl.find("flash_attn_ext_kvarn_load_tile") != std::string::npos &&
                 cuda_fattn_mma_kvarn.find("GGML_KVARN_FORCE_MATERIALIZE") == std::string::npos &&
                 cuda_fattn_mma_kvarn.find("ggml_cuda_fattn_kvarn_force_materialize_enabled") == std::string::npos,
        "native KVarN FlashAttention must keep the SWA-aware tensor-core decode in per-pair instance TUs plus generic shallow fallback without K/V F16 materialize buffers or fallback env gates");
    ok &= expect(cuda_fattn_mma_kvarn.find("record_cache") == std::string::npos &&
                 cuda_fattn.find("nbytes_shared_record =") == std::string::npos &&
                 cuda_fattn.find("nbytes_shared_KV_mask_record") == std::string::npos,
        "native KVarN MMA must stream record payloads instead of copying full records into shared memory");
    ok &= expect(cuda_fattn_mma_kvarn_impl.find("const bool stream_record = fast_record;") != std::string::npos &&
                 cuda_fattn_mma_kvarn_impl.find("tile.pos_begin + row") != std::string::npos &&
                 cuda_fattn_mma_kvarn_impl.find("fast_record && !desc.swa") == std::string::npos,
        "native KVarN MMA must use the batched record loader for SWA tiles and derive record positions from the SWA-aware tile plan");
    ok &= expect(cuda_fattn.find("ggml_cuda_fattn_kvarn_decode_select") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode_decl.find("ggml_cuda_fattn_kvarn_decode_geometry") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode_decl.find("ggml_cuda_fattn_kvarn_decode_select") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode.find("cudaOccupancyMaxActiveBlocksPerMultiprocessor") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode.find("ggml_cuda_fattn_kvarn_decode_consider") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode.find("ggml_cuda_fattn_kvarn_decode_split_tokens") == std::string::npos &&
                 cuda_fattn_mma_kvarn_decode.find("constexpr int WARPS_PER_CHUNK = D / GGML_CUDA_FATTN_KVARN_DIM;") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode.find("constexpr int CHUNKS_PER_PASS = NWARPS / WARPS_PER_CHUNK;") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode.find("score_partial_sh") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode.find("args.split_tokens") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode.find("args.nwarps") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode.find("args.gqa_per_block") != std::string::npos,
        "native KVarN split decode must select compiled geometries from occupancy/wave evidence instead of hardcoded D-specific split tokens");
    ok &= expect(cuda_fattn_kvarn_vec.find("ggml_cuda_fattn_kvarn_vec_kernel") != std::string::npos &&
                 cuda_fattn_kvarn_vec.find("template<int D, int TOKENS_PER_SPLIT, int MAX_GQA, int K_BITS, int V_BITS>") != std::string::npos &&
                 cuda_fattn_kvarn_vec_decl.find("DECL_FATTN_KVARN_VEC_CASE") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode_k4v4.find("DECL_FATTN_KVARN_VEC_CASE(256, 4, 4);") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode_k4v4.find("DECL_FATTN_KVARN_VEC_CASE(512, 4, 4);") == std::string::npos &&
                 cuda_fattn_mma_kvarn_decode_k4v3.find("DECL_FATTN_KVARN_VEC_CASE(256, 4, 3);") != std::string::npos &&
                 cuda_fattn_mma_kvarn_decode_k4v3.find("DECL_FATTN_KVARN_VEC_CASE(512, 4, 3);") == std::string::npos &&
                 cuda_template_generator.find("head_size == 256") != std::string::npos &&
                 cuda_fattn.find("ggml_cuda_flash_attn_ext_kvarn_vec_supported") != std::string::npos &&
                 cuda_fattn.find("ggml_cuda_fattn_kvarn_vec_launch<D, 4, 4>") == std::string::npos &&
                 cuda_fattn.find("plan.k.bits != 4 || plan.v.bits != 4") == std::string::npos &&
                 cuda_fattn.find("GGML_CUDA_FATTN_KVARN_VEC_DISPATCH_K") != std::string::npos &&
                 cuda_fattn.find("GGML_KVARN_VEC_D512") == std::string::npos &&
                 cuda_fattn.find("CUDA_FA_ROUTE_EXEC_DISPATCH kernel=KVARN_DECODE_VEC") != std::string::npos,
        "KVarN low-parallelism vec decode must cover every KVarN bit pair at D256 via a bit-generic dispatch (not a k4v4 hardcode) and never compile a D512 vec instance");
    ok &= expect(cuda_fattn.find("D=512: MMA/TILE templates don't support this head_dim, use VEC unconditionally") == std::string::npos &&
                 cuda_fattn.find("if (Q->ne[0] == 512) {\n        return BEST_FATTN_KERNEL_VEC;") == std::string::npos,
        "CUDA FlashAttention must not force all D=512 non-turbo attention onto the vector kernel; Gemma4 global layers need the MMA selector path");
    ok &= expect(cuda_fattn_vec_q4_0_q4_0.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);") != std::string::npos,
        "q4_0/q4_0 FlashAttention template instance must include D=512");
    ok &= expect(cuda_fattn_vec_q4_1_q4_1.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1);") != std::string::npos,
        "q4_1/q4_1 FlashAttention template instance must include D=512");
    ok &= expect(cuda_fattn_vec_q5_0_q5_0.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0);") != std::string::npos,
        "q5_0/q5_0 FlashAttention template instance must include D=512");
    ok &= expect(cuda_fattn_vec_q5_0_q5_1.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1);") != std::string::npos,
        "q5_0/q5_1 FlashAttention template instance must include D=512");
    ok &= expect(cuda_fattn_vec_q5_1_q5_1.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1);") != std::string::npos,
        "q5_1/q5_1 FlashAttention template instance must include D=512");
    ok &= expect(cuda_fattn_vec_q8_0_q8_0.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);") != std::string::npos,
        "q8_0/q8_0 FlashAttention template instance must include D=512");
    ok &= expect(cuda_fattn_vec_bf16_bf16.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_BF16, GGML_TYPE_BF16);") != std::string::npos,
        "bf16/bf16 FlashAttention template instance must include D=512");
    ok &= expect(cuda_fattn_vec_q8_0_turbo3_tcq.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_TCQ);") != std::string::npos,
        "q8_0/turbo3_tcq FlashAttention template instance must include D=512");
    ok &= expect(cuda_fattn_vec_turbo3_tcq_q8_0.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q8_0);") != std::string::npos,
        "turbo3_tcq/q8_0 FlashAttention template instance must include D=512");
    ok &= expect(cuda_fattn_vec_turbo3_tcq_turbo3_tcq.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ);") != std::string::npos,
        "turbo3_tcq/turbo3_tcq FlashAttention template instance must include D=512");
    ok &= expect(cuda_template_generator.find("DECL_FATTN_VEC_CASE(512, {type_k}, {type_v});") != std::string::npos,
        "CUDA template generator must preserve D=512 quantized FlashAttention vector instances");
    ok &= expect(ggml_h.find("GGML_TYPE_Q6_0") != std::string::npos &&
                 ggml_common_h.find("block_q6_0") != std::string::npos &&
                 ggml_c.find(".type_name                = \"q6_0\"") != std::string::npos,
        "GGML core type metadata must register q6_0");
    ok &= expect(ggml_quants_h.find("quantize_row_q6_0_ref") != std::string::npos &&
                 ggml_quants_h.find("dequantize_row_q6_0") != std::string::npos &&
                 ggml_quants_c.find("quantize_row_q6_0_ref") != std::string::npos &&
                 ggml_quants_c.find("dequantize_row_q6_0") != std::string::npos,
        "GGML quantization must provide q6_0 reference quantize/dequantize functions");
    ok &= expect(ggml_cpu_quants_h.find("ggml_vec_dot_q6_0_q8_0") != std::string::npos &&
                 ggml_cpu_quants_c.find("ggml_vec_dot_q6_0_q8_0_generic") != std::string::npos &&
                 ggml_cpu_c.find("[GGML_TYPE_Q6_0]") != std::string::npos,
        "CPU backend traits must support q6_0 quantize and dot-product paths");
    ok &= expect(arg_cpp.find("GGML_TYPE_Q6_0") != std::string::npos &&
                 llama_bench.find("GGML_TYPE_Q6_0") != std::string::npos,
        "CLI and benchmark cache-type parsers must expose q6_0");
    ok &= expect(arg_cpp.find("case 2:  return GGML_TYPE_Q2_0;") != std::string::npos &&
                 arg_cpp.find("case 3:  return GGML_TYPE_Q3_0;") != std::string::npos &&
                 llama_bench.find("case 2:  return GGML_TYPE_Q2_0;") != std::string::npos &&
                 llama_bench.find("case 3:  return GGML_TYPE_Q3_0;") != std::string::npos,
        "KVarN 2/3-bit fallback cache types must use q2_0/q3_0 in CLI and benchmark");
    ok &= expect(cuda_common.find("ggml_cuda_type_traits<GGML_TYPE_Q6_0>") != std::string::npos &&
                 cuda_dequantize.find("dequantize_q6_0") != std::string::npos &&
                 cuda_cpy_utils.find("quantize_f32_q6_0_block") != std::string::npos &&
                 cuda_getrows.find("GGML_TYPE_Q6_0") != std::string::npos &&
                 cuda_set_rows.find("GGML_TYPE_Q6_0") != std::string::npos &&
                 cuda_cpy.find("GGML_TYPE_Q6_0") != std::string::npos &&
                 cuda_cpp.find("GGML_TYPE_Q6_0") != std::string::npos,
        "CUDA backend must support q6_0 SET_ROWS, GET_ROWS, CPY, and capability checks");
    ok &= expect(cuda_fattn_common.find("vec_dot_fattn_vec_KQ_q6_0") != std::string::npos &&
                 cuda_fattn_common.find("dequantize_V_q6_0") != std::string::npos &&
                 cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q6_0, GGML_TYPE_Q6_0)") != std::string::npos &&
                 cuda_fattn_vec_q6_0_q6_0.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q6_0, GGML_TYPE_Q6_0);") != std::string::npos &&
                 cuda_fattn_vec_q6_0_q8_0.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q6_0, GGML_TYPE_Q8_0);") != std::string::npos &&
                 cuda_fattn_vec_q8_0_q6_0.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q8_0, GGML_TYPE_Q6_0);") != std::string::npos,
        "CUDA FlashAttention must dispatch and instantiate q6_0 KV cache pairs through D=512");
    {
        struct classic_kv_type {
            const char * enum_name;
            const char * lower_name;
            const char * block_name;
            const char * quantize_ref;
            const char * dequantize;
            const char * vec_dot;
            const char * cuda_traits;
            const char * cuda_quantize;
            const std::string * fattn_instance;
        };

        const classic_kv_type classic_types[] = {
            { "GGML_TYPE_Q6_1", "q6_1", "block_q6_1", "quantize_row_q6_1_ref", "dequantize_row_q6_1", "ggml_vec_dot_q6_1_q8_1", "ggml_cuda_type_traits<GGML_TYPE_Q6_1>", "quantize_f32_q6_1_block", &cuda_fattn_vec_q6_1_q6_1 },
            { "GGML_TYPE_Q3_0", "q3_0", "block_q3_0", "quantize_row_q3_0_ref", "dequantize_row_q3_0", "ggml_vec_dot_q3_0_q8_0", "ggml_cuda_type_traits<GGML_TYPE_Q3_0>", "quantize_f32_q3_0_block", &cuda_fattn_vec_q3_0_q3_0 },
            { "GGML_TYPE_Q3_1", "q3_1", "block_q3_1", "quantize_row_q3_1_ref", "dequantize_row_q3_1", "ggml_vec_dot_q3_1_q8_1", "ggml_cuda_type_traits<GGML_TYPE_Q3_1>", "quantize_f32_q3_1_block", &cuda_fattn_vec_q3_1_q3_1 },
            { "GGML_TYPE_Q2_0", "q2_0", "block_q2_0", "quantize_row_q2_0_ref", "dequantize_row_q2_0", "ggml_vec_dot_q2_0_q8_0", "ggml_cuda_type_traits<GGML_TYPE_Q2_0>", "quantize_f32_q2_0_block", &cuda_fattn_vec_q2_0_q2_0 },
            { "GGML_TYPE_Q2_1", "q2_1", "block_q2_1", "quantize_row_q2_1_ref", "dequantize_row_q2_1", "ggml_vec_dot_q2_1_q8_1", "ggml_cuda_type_traits<GGML_TYPE_Q2_1>", "quantize_f32_q2_1_block", &cuda_fattn_vec_q2_1_q2_1 },
        };

        for (const classic_kv_type & type : classic_types) {
            const std::string type_name(type.lower_name);
            const std::string enum_name(type.enum_name);

            ok &= expect(ggml_h.find(type.enum_name) != std::string::npos &&
                         ggml_common_h.find(type.block_name) != std::string::npos &&
                         ggml_c.find(".type_name                = \"" + type_name + "\"") != std::string::npos,
                ("GGML core type metadata must register " + type_name).c_str());
            ok &= expect(ggml_quants_h.find(type.quantize_ref) != std::string::npos &&
                         ggml_quants_h.find(type.dequantize) != std::string::npos &&
                         ggml_quants_c.find(type.quantize_ref) != std::string::npos &&
                         ggml_quants_c.find(type.dequantize) != std::string::npos,
                ("GGML quantization must provide reference quantize/dequantize for " + type_name).c_str());
            ok &= expect(ggml_cpu_quants_h.find(type.vec_dot) != std::string::npos &&
                         ggml_cpu_quants_c.find(std::string(type.vec_dot) + "_generic") != std::string::npos &&
                         ggml_cpu_c.find("[" + enum_name + "]") != std::string::npos,
                ("CPU backend traits must support " + type_name).c_str());
            ok &= expect(arg_cpp.find(type.enum_name) != std::string::npos &&
                         llama_bench.find(type.enum_name) != std::string::npos,
                ("CLI and benchmark cache-type parsers must expose " + type_name).c_str());
            ok &= expect(cuda_common.find(type.cuda_traits) != std::string::npos &&
                         cuda_dequantize.find(std::string("dequantize_") + type.lower_name) != std::string::npos &&
                         cuda_cpy_utils.find(type.cuda_quantize) != std::string::npos &&
                         cuda_getrows.find(type.enum_name) != std::string::npos &&
                         cuda_set_rows.find(type.enum_name) != std::string::npos &&
                         cuda_cpy.find(type.enum_name) != std::string::npos &&
                         cuda_cpp.find(type.enum_name) != std::string::npos,
                ("CUDA backend must support SET_ROWS, GET_ROWS, CPY, and capability checks for " + type_name).c_str());
            ok &= expect(cuda_fattn_common.find(std::string("vec_dot_fattn_vec_KQ_") + type.lower_name) != std::string::npos &&
                         cuda_fattn_common.find(std::string("dequantize_V_") + type.lower_name) != std::string::npos &&
                         cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(" + enum_name + ", " + enum_name + ")") != std::string::npos &&
                         type.fattn_instance->find("DECL_FATTN_VEC_CASE(512, " + enum_name + ", " + enum_name + ");") != std::string::npos,
                ("CUDA FlashAttention must dispatch and instantiate matched " + type_name + " KV cache pairs through D=512").c_str());
        }
    }
    ok &= expect(ggml_h.find("GGML_TYPE_TURBO4_TCQ") != std::string::npos &&
                 ggml_common_h.find("block_turbo4_tcq") != std::string::npos &&
                 ggml_c.find(".type_name                = \"turbo4_tcq\"") != std::string::npos,
        "GGML core type metadata must register turbo4_tcq");
    ok &= expect(GGML_TYPE_TURBO4_TCQ == 55 &&
                 ggml_type_size(GGML_TYPE_TURBO4_TCQ) == 68,
        "GGML runtime type metadata must preserve turbo4_tcq enum value and block size");
    ok &= expect(ggml_quants_h.find("quantize_row_turbo4_tcq_ref") != std::string::npos &&
                 ggml_quants_h.find("dequantize_row_turbo4_tcq") != std::string::npos &&
                 ggml_turbo_quant_c.find("quantize_row_turbo4_tcq_ref") != std::string::npos &&
                 ggml_turbo_quant_c.find("dequantize_row_turbo4_tcq") != std::string::npos &&
                 ggml_c.find("GGML_TYPE_TURBO4_TCQ: result = quantize_turbo4_tcq") != std::string::npos,
        "GGML quantization must provide turbo4_tcq reference metadata and chunk dispatch");
    ok &= expect(arg_cpp.find("GGML_TYPE_TURBO4_TCQ") != std::string::npos &&
                 llama_bench.find("GGML_TYPE_TURBO4_TCQ") != std::string::npos &&
                 kv_cache_cpp.find("GGML_TYPE_TURBO4_TCQ") != std::string::npos &&
                 context_cpp.find("GGML_TYPE_TURBO4_TCQ") != std::string::npos &&
                 graph_cpp.find("GGML_TYPE_TURBO4_TCQ") != std::string::npos,
        "CLI, benchmark, and llama KV cache logic must expose turbo4_tcq");
    ok &= expect(cuda_turbo_quant.find("d_turbo4_tcq_codebook") != std::string::npos &&
                 cuda_turbo_quant.find("dequantize_turbo4_tcq") != std::string::npos &&
                 cuda_turbo_quant.find("k_set_rows_turbo4_tcq") != std::string::npos &&
                 cuda_getrows.find("GGML_TYPE_TURBO4_TCQ") != std::string::npos &&
                 cuda_set_rows.find("GGML_TYPE_TURBO4_TCQ") != std::string::npos &&
                 cuda_cpp.find("GGML_TYPE_TURBO4_TCQ") != std::string::npos,
        "CUDA backend must support turbo4_tcq SET_ROWS, GET_ROWS, and capability checks");
    ok &= expect(cuda_fattn_common.find("vec_dot_fattn_vec_KQ_turbo4_tcq") != std::string::npos &&
                 cuda_fattn_common.find("dequantize_V_turbo4_tcq") != std::string::npos &&
                 cuda_fattn.find("k_turbo4_tcq_dequant_f16") != std::string::npos &&
                 cuda_fattn_vec.find("GGML_TYPE_TURBO4_TCQ") != std::string::npos &&
                 cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO4_TCQ)") != std::string::npos &&
                 cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_TCQ)") != std::string::npos &&
                 cuda_fattn.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q8_0)") != std::string::npos &&
                 cuda_fattn_vec_turbo4_tcq_turbo4_tcq.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO4_TCQ);") != std::string::npos &&
                 cuda_fattn_vec_q8_0_turbo4_tcq.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_TCQ);") != std::string::npos &&
                 cuda_fattn_vec_turbo4_tcq_q8_0.find("DECL_FATTN_VEC_CASE(512, GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q8_0);") != std::string::npos,
        "CUDA FlashAttention must dispatch and instantiate turbo4_tcq matched and q8 mixed pairs through D=512");
    ok &= expect(cuda_template_generator.find("GGML_TYPE_TURBO4_TCQ") != std::string::npos &&
                 cuda_template_generator.find("GGML_TYPE_Q6_1") != std::string::npos &&
                 cuda_template_generator.find("GGML_TYPE_Q3_0") != std::string::npos &&
                 cuda_template_generator.find("GGML_TYPE_Q2_1") != std::string::npos &&
                 cuda_cmake.find("turbo4_tcq") != std::string::npos &&
                 cuda_cmake.find("GGML_CUDA_FA_HALF_PAIR_COUNT EQUAL 217") != std::string::npos &&
                 cuda_dispatch_generator.find("GGML_TYPE_TURBO4_TCQ") != std::string::npos &&
                 cuda_dispatch_generator.find("assert len(TYPES) == 19") != std::string::npos &&
                 cuda_dispatch_generator.find("assert all_count == 361") != std::string::npos &&
                 cuda_dispatch_generator.find("assert half_count == 217") != std::string::npos,
        "CUDA FlashAttention generators and CMake must include turbo4_tcq and the new q KV types in ALL/HALF matrices");
    {
        const std::string half_order =
            "q5_0\n"
            "            q4_1\n"
            "            turbo4_tcq\n"
            "            turbo4_0\n"
            "            q4_0\n"
            "            q3_1\n"
            "            turbo3_tcq\n"
            "            turbo3_0\n"
            "            q3_0\n"
            "            q2_1\n"
            "            turbo2_tcq\n"
            "            turbo2_0\n"
            "            q2_0";
        const std::string generator_half_order =
            "GGML_TYPE_Q5_0\",       \"q5_0\"),\n"
            "    (\"GGML_TYPE_Q4_1\",       \"q4_1\"),\n"
            "    (\"GGML_TYPE_TURBO4_TCQ\", \"turbo4_tcq\"),\n"
            "    (\"GGML_TYPE_TURBO4_0\",   \"turbo4_0\"),\n"
            "    (\"GGML_TYPE_Q4_0\",       \"q4_0\"),\n"
            "    (\"GGML_TYPE_Q3_1\",       \"q3_1\"),\n"
            "    (\"GGML_TYPE_TURBO3_TCQ\", \"turbo3_tcq\"),\n"
            "    (\"GGML_TYPE_TURBO3_0\",   \"turbo3_0\"),\n"
            "    (\"GGML_TYPE_Q3_0\",       \"q3_0\"),\n"
            "    (\"GGML_TYPE_Q2_1\",       \"q2_1\"),\n"
            "    (\"GGML_TYPE_TURBO2_TCQ\", \"turbo2_tcq\"),\n"
            "    (\"GGML_TYPE_TURBO2_0\",   \"turbo2_0\"),\n"
            "    (\"GGML_TYPE_Q2_0\",       \"q2_0\")";
        const std::string template_generator_half_order =
            "\"GGML_TYPE_Q5_0\",\n"
            "    \"GGML_TYPE_Q4_1\",\n"
            "    \"GGML_TYPE_TURBO4_TCQ\",\n"
            "    \"GGML_TYPE_TURBO4_0\",\n"
            "    \"GGML_TYPE_Q4_0\",\n"
            "    \"GGML_TYPE_Q3_1\",\n"
            "    \"GGML_TYPE_TURBO3_TCQ\",\n"
            "    \"GGML_TYPE_TURBO3_0\",\n"
            "    \"GGML_TYPE_Q3_0\",\n"
            "    \"GGML_TYPE_Q2_1\",\n"
            "    \"GGML_TYPE_TURBO2_TCQ\",\n"
            "    \"GGML_TYPE_TURBO2_0\",\n"
            "    \"GGML_TYPE_Q2_0\"";
        const std::string cuda_fattn_half_dispatch = slice_between(
            cuda_fattn,
            "#elif defined(GGML_CUDA_FA_HALF_QUANTS)\n    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_F16)",
            "#else\n    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_F16)");

        ok &= expect(cuda_cmake.find(half_order) != std::string::npos &&
                     cuda_cmake.find("function(ggml_cuda_fa_kv_rank OUT TYPE)") != std::string::npos &&
                     cuda_cmake.find("TYPE STREQUAL \"q4_1\" OR TYPE STREQUAL \"turbo4_tcq\" OR TYPE STREQUAL \"turbo4_0\"") != std::string::npos &&
                     cuda_cmake.find("TYPE STREQUAL \"q3_1\" OR TYPE STREQUAL \"turbo3_tcq\" OR TYPE STREQUAL \"turbo3_0\"") != std::string::npos &&
                     cuda_cmake.find("TYPE STREQUAL \"q2_1\" OR TYPE STREQUAL \"turbo2_tcq\" OR TYPE STREQUAL \"turbo2_0\"") != std::string::npos &&
                     cuda_dispatch_generator.find("HALF_RANKS = {") != std::string::npos &&
                     cuda_dispatch_generator.find("\"GGML_TYPE_Q4_1\": 7") != std::string::npos &&
                     cuda_dispatch_generator.find("\"GGML_TYPE_TURBO4_TCQ\": 7") != std::string::npos &&
                     cuda_dispatch_generator.find("\"GGML_TYPE_TURBO4_0\": 7") != std::string::npos &&
                     cuda_dispatch_generator.find(generator_half_order) != std::string::npos &&
                     cuda_template_generator.find(template_generator_half_order) != std::string::npos,
            "CUDA FlashAttention HALF ranking must treat Turbo/TurboTCQ as pseudo-equal to the _1 member of each bit tier");
        ok &= expect(cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_TURBO4_TCQ)") != std::string::npos &&
                     cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q4_1)") != std::string::npos &&
                     cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_1, GGML_TYPE_TURBO4_0)") != std::string::npos &&
                     cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q4_1)") != std::string::npos &&
                     cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_TCQ)") != std::string::npos &&
                     cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_TURBO4_0)") != std::string::npos &&
                     cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_TURBO4_TCQ)") == std::string::npos &&
                     cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_TURBO4_0)") == std::string::npos &&
                     cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_TCQ, GGML_TYPE_Q4_0)") != std::string::npos &&
                     cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_1, GGML_TYPE_TURBO3_TCQ)") != std::string::npos &&
                     cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q3_1)") != std::string::npos &&
                     cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q3_0, GGML_TYPE_TURBO3_TCQ)") == std::string::npos &&
                     cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_1, GGML_TYPE_TURBO2_TCQ)") != std::string::npos &&
                     cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q2_1)") != std::string::npos &&
                     cuda_fattn_half_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q2_0, GGML_TYPE_TURBO2_TCQ)") == std::string::npos,
            "CUDA FlashAttention HALF dispatch must compile pseudo-equal qX_1/Turbo pairs both ways and exclude qX_0-to-Turbo K<V pairs");
    }
    {
        const std::string cuda_cmake_default_pairs = slice_between(
            cuda_cmake,
            "set(GGML_CUDA_FA_DEFAULT_KV_PAIRS",
            "foreach(PAIR ${GGML_CUDA_FA_DEFAULT_KV_PAIRS})");
        const std::string cuda_fattn_default_dispatch = slice_between(
            cuda_fattn,
            "#else\n    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16, GGML_TYPE_F16)",
            "#endif\n\n    fprintf(stderr,");
        const std::string generator_default_pairs = slice_between(
            cuda_dispatch_generator,
            "DEFAULT_PAIRS = [",
            "]\n\ndef emit_pair");
        const auto cmake_has_default_pair = [&](const char * pair) {
            return cuda_cmake_default_pairs.find(std::string("\n            ") + pair + "\n") != std::string::npos;
        };

        ok &= expect(cuda_cmake.find("GGML_CUDA_FA_DEFAULT_PAIR_COUNT EQUAL 62") != std::string::npos &&
                     cuda_dispatch_generator.find("assert len(DEFAULT_PAIRS) == 62") != std::string::npos,
            "CUDA FlashAttention default build must compile the fork q/KVarN fallback 62-pair policy");
        ok &= expect(cuda_fattn.find("ggml_cuda_fattn_default_kv_tier") != std::string::npos &&
                     cuda_fattn.find("tier_V - tier_K <= 2") != std::string::npos,
            "CUDA FlashAttention runtime support checks must derive default pairs from the same bit-tier policy");
        ok &= expect(cuda_fattn.find("standard q/KVarN fallback cache types") != std::string::npos &&
                     cuda_fattn.find("GGML_CUDA_FA_HALF_QUANTS or GGML_CUDA_FA_ALL_QUANTS") != std::string::npos,
            "CUDA FlashAttention missing-pair diagnostics must suggest q/KVarN or larger FA build modes");
        ok &= expect(cmake_has_default_pair("f16:f16") &&
                     cmake_has_default_pair("bf16:bf16") &&
                     cmake_has_default_pair("q8_0:q4_0") &&
                     cmake_has_default_pair("q6_0:q4_0") &&
                     cmake_has_default_pair("q5_0:q3_0") &&
                     cmake_has_default_pair("q4_0:q2_0"),
            "CUDA FlashAttention default CMake pairs must include q/KVarN fallback tiers through q2");
        ok &= expect(generator_default_pairs.find("GGML_TYPE_Q8_0\", \"GGML_TYPE_Q4_0") != std::string::npos &&
                     generator_default_pairs.find("GGML_TYPE_Q5_0\", \"GGML_TYPE_Q3_0") != std::string::npos &&
                     generator_default_pairs.find("GGML_TYPE_Q4_0\", \"GGML_TYPE_Q2_0") != std::string::npos,
            "CUDA FlashAttention default dispatch generator must include the q/KVarN fallback pair tiers");
        ok &= expect(cuda_fattn_default_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)") != std::string::npos &&
                     cuda_fattn_default_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q5_0, GGML_TYPE_Q3_0)") != std::string::npos &&
                     cuda_fattn_default_dispatch.find("FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q4_0, GGML_TYPE_Q2_0)") != std::string::npos,
            "CUDA FlashAttention default dispatch must include q/KVarN fallback pair tiers");
        ok &= expect(cuda_cmake_default_pairs.find("turbo") == std::string::npos &&
                     generator_default_pairs.find("GGML_TYPE_TURBO") == std::string::npos &&
                     cuda_fattn_default_dispatch.find("GGML_TYPE_TURBO") == std::string::npos,
            "CUDA FlashAttention default build must leave Turbo/TurboTCQ pairs to HALF or ALL builds");
        ok &= expect(!cmake_has_default_pair("f16:bf16") &&
                     !cmake_has_default_pair("bf16:f16") &&
                     !cmake_has_default_pair("f16:q4_0") &&
                     !cmake_has_default_pair("bf16:q4_0") &&
                     !cmake_has_default_pair("q8_0:q3_0") &&
                     !cmake_has_default_pair("q6_0:q3_0"),
            "CUDA FlashAttention default build must enforce same-or-higher K precision and the two-tier-group V limit without f16/bf16 asymmetry");

        ok &= expect(cuda_fattn_h.find("ggml_cuda_fa_build_policy") != std::string::npos &&
                     cuda_fattn_h.find("ggml_cuda_fa_pair_compiled") != std::string::npos &&
                     cuda_fattn.find("ggml_cuda_fa_build_policy") != std::string::npos &&
                     cuda_fattn.find("ggml_cuda_fa_pair_compiled") != std::string::npos &&
                     cuda_cpp.find("ggml_cuda_fa_build_policy") != std::string::npos &&
                     cuda_cpp.find("ggml_cuda_fa_pair_compiled") != std::string::npos,
            "CUDA backend must expose FA build policy and pair-compiled diagnostics through proc_address");
        ok &= expect(cparams_h.find("flash_attn_required") != std::string::npos &&
                     context_cpp.find("llama_cuda_fa_missing_pair_message") != std::string::npos &&
                     context_cpp.find("Default builds compile standard q/KVarN-fallback pairs only (no TurboQuant/TCQ)") != std::string::npos &&
                     context_cpp.find("HALF builds compile same-or-higher ranked K than V, with TurboQuant/TCQ is treated as pseudo-equal to qX_1") != std::string::npos &&
                     context_cpp.find("GGML_CUDA_FA_IGNORE_UNCOMPILED_PAIRS") != std::string::npos &&
                     context_cpp.find("failed to reserve graph for required Flash Attention check") != std::string::npos &&
                     context_cpp.find("This is not a quant-pair build-policy miss") != std::string::npos &&
                     context_cpp.find("Flash Attention was auto, set to disabled") != std::string::npos,
            "llama-context must fail required CUDA FA missing pairs with build-specific text and only warn/disable optional auto FA");

        const std::string user_docs = readme_md + "\n" + agents_md + "\n" + claude_md + "\n" +
            docs_build_md + "\n" + quickstart_qwen36_md + "\n" + quickstart_gemma4_md;
        ok &= expect(release_workflow.find("-DGGML_CUDA_FA_HALF_QUANTS=ON") == std::string::npos &&
                     release_workflow.find("-DGGML_CUDA_FA_ALL_QUANTS=ON") == std::string::npos &&
                     release_dispatch_workflow.find("-DGGML_CUDA_FA_HALF_QUANTS=ON") == std::string::npos &&
                     release_dispatch_workflow.find("-DGGML_CUDA_FA_ALL_QUANTS=ON") == std::string::npos,
            "Release workflows must use default CUDA FA quant compilation");
        ok &= expect(quickstart_qwen36_md.find("-DGGML_CUDA_FA_HALF_QUANTS=ON") == std::string::npos &&
                     quickstart_gemma4_md.find("-DGGML_CUDA_FA_HALF_QUANTS=ON") == std::string::npos,
            "DFlash quickstart build commands must use default CUDA FA quant compilation");
        ok &= expect(user_docs.find("standard q cache types or KVarN") != std::string::npos &&
                     user_docs.find("TurboQuant/TCQ has not shown a benchmark advantage") != std::string::npos &&
                     user_docs.find("same or higher precision than V") != std::string::npos &&
                     user_docs.find("two tier groups") != std::string::npos &&
                     user_docs.find("GGML_CUDA_FA_HALF_QUANTS=ON") != std::string::npos &&
                     user_docs.find("GGML_CUDA_FA_ALL_QUANTS=ON") != std::string::npos,
            "User-facing docs must recommend default CUDA FA q/KVarN compilation while documenting HALF/ALL escape hatches");
    }
    ok &= expect(arch_cpp.find("{ LLM_ARCH_DFLASH,") != std::string::npos,
        "upstream dflash architecture must be registered separately");
    ok &= expect(convert_py.find("from conversion import") != std::string::npos,
        "converter wrapper must keep using the upstream modular conversion package");
    ok &= expect(arch_cpp.find("{ LLM_ARCH_DFLASH,           \"dflash\"") != std::string::npos &&
                 arch_cpp.find("{ LLM_ARCH_DFLASH_DRAFT,     \"dflash-draft\"") != std::string::npos,
        "upstream and Bee DFlash architecture names must both be recognized");
    ok &= expect(arch_cpp.find("bool llm_arch_is_dflash_drafter") != std::string::npos &&
                 arch_cpp.find("arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_DFLASH_DRAFT") != std::string::npos,
        "both DFlash schemas must be treated as DFlash drafters");
    ok &= expect(arch_cpp.find("LLM_TENSOR_DFLASH_UPSTREAM_FC") != std::string::npos &&
                 arch_cpp.find("\"fc\"") != std::string::npos &&
                 arch_cpp.find("\"hidden_norm\"") != std::string::npos,
        "upstream DFlash tensor names must be registered separately");
    ok &= expect(dflash_draft.find("\"dflash.block_size\"") != std::string::npos &&
                 dflash_draft.find("\"dflash.target_layer_ids\"") != std::string::npos,
        "upstream DFlash metadata keys must be read literally");
    ok &= expect(dflash_draft.find("LLM_TENSOR_DFLASH_UPSTREAM_FC") != std::string::npos,
        "upstream DFlash must use upstream fusion tensor names");
    ok &= expect(model_cpp.find("case LLM_ARCH_DFLASH:\n            {\n                res = nullptr;\n            } break;") != std::string::npos,
        "upstream DFlash must remain stateless");
    ok &= expect(model_cpp.find("case LLM_ARCH_DFLASH_DRAFT:\n            {\n                res = nullptr;") == std::string::npos,
        "Bee DFlash draft must allocate KV memory so full-attention layers can track the accepted suffix");
    ok &= expect(dflash_draft.find("arch == LLM_ARCH_DFLASH ? LLM_TENSOR_FFN_NORM : LLM_TENSOR_ATTN_POST_NORM") != std::string::npos,
        "upstream DFlash must use ffn_norm where Bee DFlash uses post_attention_norm");
    ok &= expect(dflash_draft.find("bool can_reuse(const llm_graph_params & params) override") != std::string::npos, "DFlash drafter graph input must opt into graph reuse");
    ok &= expect(dflash_draft.find("dflash_draft_ctx_len(params.cross, params.cparams) != ctx_len") != std::string::npos, "DFlash drafter reuse must invalidate on cross bucket shape changes");
    ok &= expect(dflash_draft.find("dflash_kv_cache_ready_for_window(params.cross, ctx_len) != use_kv_cache") != std::string::npos, "DFlash drafter reuse must invalidate when K/V cache topology changes");
    ok &= expect(dflash_draft.find("Kcur_ctx_f16") == std::string::npos, "DFlash drafter must not cast K before ggml_concat; CUDA concat requires F32 sources");
    ok &= expect(dflash_draft.find("Vcur_ctx_f16") == std::string::npos, "DFlash drafter must not cast V before ggml_concat; CUDA concat requires F32 sources");
    ok &= expect(dflash_draft.find("ggml_tensor * inp_out_ids = n_outputs < n_tokens ? build_inp_out_ids() : nullptr;") != std::string::npos, "DFlash drafter must avoid unused seed-token output rows");
    ok &= expect(dflash_draft.find("inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);") != std::string::npos, "DFlash drafter must gather only requested output rows before lm_head");
    ok &= expect(dflash_draft.find("const bool need_swa_mask = have_swa && hparams.n_swa > 0 && (int64_t) hparams.n_swa < n_kv_total;") != std::string::npos,
        "DFlash drafter must skip redundant SWA masks when the cross window is within n_swa");
    ok &= expect(dflash_draft.find("const int32_t ctx_pos_base = q_pos_base - (int32_t) n_real;") != std::string::npos &&
                 dflash_draft.find("data[i] = (i < n_real) ? (ctx_pos_base + (int32_t) i) : 0;") != std::string::npos &&
                 dflash_draft.find("data[q] = have_pos ? (int32_t) ubatch->pos[q]") != std::string::npos &&
                 dflash_draft.find("const int32_t k_pos = ctx_pos_base + (int32_t) k;") != std::string::npos,
        "single-slot DFlash draft must preserve absolute context/query positions when the cross window slides");
    ok &= expect(dflash_draft.find("slot_ctx_pos_base[s] = q_pos_base - (int32_t) nc;") != std::string::npos &&
                 dflash_draft.find("data[off + i] = (i < nc) ? (slot_ctx_pos_base[s] + (int32_t) i) : 0;") != std::string::npos &&
                 dflash_draft.find("const int32_t k_pos = slot_ctx_pos_base[ks] + kl;") != std::string::npos,
        "multi-slot DFlash draft must preserve absolute per-slot positions when the cross window slides");
    ok &= expect(dflash_draft.find("q_pos - k_pos >= window") != std::string::npos &&
                 dflash_draft.find("q_pos - k_pos < window") != std::string::npos,
        "DFlash sliding masks must use absolute key/query positions when enforcing the SWA window");
    ok &= expect(dflash_draft.find("dflash_build_base_attn_input") != std::string::npos &&
                 dflash_draft.find("static_cast<const llama_kv_cache_iswa_context *>(mctx)->get_base()") != std::string::npos,
        "DFlash full-attention layers must read accepted-prefix KV from the drafter base cache, not the cross window");
    ok &= expect(speculative.find("common_dflash_align_drafter_seq_or_clear") != std::string::npos &&
                 speculative.find("common_dflash_reset_drafter_seq_and_kv_cache") != std::string::npos,
        "DFlash must have explicit helpers for clearing stale drafter sequence memory and projection cache");
    ok &= expect(count_occurrences(speculative, "common_dflash_align_drafter_seq_or_clear(ctx_dft") >= 3,
        "DFlash draft paths must clear stale drafter memory before decoding at absolute committed positions");
    ok &= expect(speculative.find("common_dflash_reset_drafter_seq_and_kv_cache(ctx_dft, seq_id, \"first prefill flush\")") != std::string::npos &&
                 speculative.find("common_dflash_reset_drafter_seq_and_kv_cache(ctx_dft, seq_id, \"capture target hiddens\")") != std::string::npos &&
                 speculative.find("common_dflash_reset_drafter_seq_and_kv_cache(ctx_dft, seq_id, \"ring state load\")") != std::string::npos,
        "DFlash ring reset/restore paths must clear ctx_dft sequence memory so the next draft batch is consecutive");
    ok &= expect(dflash_draft.find("class llm_graph_input_attn_kv_backend final : public llm_graph_input_attn_kv") != std::string::npos &&
                 dflash_draft.find("mctx->set_input_k_idxs_backend(self_k_idxs, ubatch);") != std::string::npos &&
                 dflash_draft.find("mctx->set_input_v_idxs_backend(self_v_idxs, ubatch);") != std::string::npos &&
                 dflash_draft.find("static_cast<const llama_kv_cache_iswa_context *>(params.mctx)->get_base()") != std::string::npos &&
                 dflash_draft.find("const bool rebind_base_from_iswa = false;") != std::string::npos,
        "DFlash base-KV draft graphs must upload KV indices directly into GPU input tensors and rebind reused ISWA graphs to the base cache context");
    ok &= expect(dflash_draft.find("class llm_graph_input_attn_kv_commit_backend final : public llm_graph_input_attn_kv") != std::string::npos &&
                 dflash_draft.find("kv(mctx ? mctx->get_kv() : nullptr)") != std::string::npos &&
                 dflash_draft.find("sinfo(mctx ? mctx->current_sinfo() : llama_kv_cache::slot_info())") != std::string::npos &&
                 dflash_draft.find("dflash_build_commit_attn_input") != std::string::npos,
        "DFlash full-KV commit graphs must snapshot the resolved base-KV slot mapping instead of chasing a mutable batch-context vector at set_input time");
    ok &= expect(graph_h.find("LLM_GRAPH_TYPE_DFLASH_KV_UPDATE") != std::string::npos, "DFlash K/V cache update must have a distinct graph type");
    ok &= expect(graph_h.find("llama_dflash_kv_cache_view * dflash_kv_cache") != std::string::npos, "DFlash cross data must carry the drafter K/V cache view");
    ok &= expect(graph_h.find("std::vector<ggml_tensor *> k_ring") != std::string::npos, "DFlash K/V cache must use ring-only persistent storage");
    ok &= expect(dflash_draft.find("llm_build_dflash_kv_update") != std::string::npos, "DFlash draft model must dispatch the K/V cache update graph");
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
    ok &= expect(context_h.find("dflash_kv_cache_multi_gpu_fallback_logged") != std::string::npos,
        "DFlash K/V cache multi-GPU fallback logging must be tracked per context");
    ok &= expect(
        kv_cache_init_fn != std::string::npos &&
        kv_cache_update_fn != std::string::npos &&
        context_cpp.find("multi-device DFlash drafter detected", kv_cache_init_fn) < kv_cache_update_fn,
        "DFlash drafter K/V cache must fall back before initializing on multi-GPU draft placement");
    ok &= expect(
        kv_cache_init_fn != std::string::npos &&
        kv_cache_update_fn != std::string::npos &&
        context_cpp.find("cparams.n_seq_max > 1", kv_cache_init_fn) == std::string::npos &&
        context_cpp.find("cparams.dflash_n_slots > 1", context_cpp.find("bool llama_context::dflash_kv_cache_prepare")) < kv_cache_update_fn &&
        context_cpp.find("active multi-slot DFlash draft batch", kv_cache_update_fn) == std::string::npos &&
        context_cpp.find("cparams.dflash_n_slots > 1", kv_cache_update_fn) == std::string::npos,
        "DFlash drafter K/V cache updates must remain per-slot in an -np > 1 server instead of being disabled by active multi-slot drafting");
    ok &= expect(
        kv_cache_update_fn != std::string::npos &&
        kv_cache_update_graph != std::string::npos &&
        context_cpp.find("model.n_devices() > 1", kv_cache_update_fn) < kv_cache_update_graph,
        "DFlash K/V cache update must not compute a split drafter graph on one CUDA backend");
    ok &= expect(speculative.find("llama_dflash_kv_cache_update_from_ring_seq(ctx_dft, gpu_ring_handle") != std::string::npos &&
                 speculative.find("llama_dflash_kv_cache_set_active_seq(ctx_dft, seq_id);") != std::string::npos &&
                 llama_h.find("llama_dflash_kv_cache_update_from_ring_seq") != std::string::npos,
        "DFlash accepted hiddens must update the slot-owned K/V projection cache from the GPU ring without mutating cross state");
    ok &= expect(context_h.find("std::map<llama_seq_id, std::unique_ptr<dflash_kv_cache_data>> dflash_kv_caches") != std::string::npos &&
                 context_cpp.find("void llama_context::dflash_kv_cache_set_active_seq") != std::string::npos &&
                 context_cpp.find("dflash_kv_cache_active_seq = seq_id;") != std::string::npos,
        "DFlash drafter K/V projection caches must be keyed by logical seq so concurrent slots cannot share one cache");
    {
        const size_t prepare_batch_fn = context_cpp.find("bool llama_context::dflash_kv_cache_prepare_batch");
        ok &= expect(prepare_batch_fn != std::string::npos &&
                     context_h.find("dflash_kv_cache_batch") != std::string::npos &&
                     context_h.find("bool dflash_kv_cache_prepare_batch") != std::string::npos &&
                     llama_h.find("llama_dflash_kv_cache_prepare_batch") != std::string::npos &&
                     context_cpp.find("cross.dflash_kv_cache = &dflash_kv_cache_batch.view", prepare_batch_fn) != std::string::npos,
            "DFlash shared multi-slot drafts must expose a composite slot-owned K/V projection cache, not the active single-slot cache");
    }
    {
        const size_t batch_fn = speculative.find("void common_speculative_draft_batch");
        const size_t prewidth = speculative.find("llama_set_dflash_n_slots(ctx_dft, std::max", batch_fn);
        const size_t prepare = speculative.find("prepare_batch_draft(ctx_dft)", batch_fn);
        const size_t readywidth = speculative.find("llama_set_dflash_n_slots(ctx_dft, n_ready)", batch_fn);
        const size_t cache_prepare = speculative.find("llama_dflash_kv_cache_prepare_batch(ctx_dft", readywidth);
        const size_t decode = speculative.find("llama_decode(ctx_dft, batch)", batch_fn);
        ok &= expect(batch_fn != std::string::npos &&
                     prewidth != std::string::npos &&
                     prepare != std::string::npos &&
                     readywidth != std::string::npos &&
                     cache_prepare != std::string::npos &&
                     decode != std::string::npos &&
                     prewidth < prepare &&
                     prepare < readywidth &&
                     readywidth < cache_prepare &&
                     cache_prepare < decode,
            "DFlash shared drafter must enter multi-slot mode before preparing cross data, then expose a batched K/V cache before decode");
    }
    ok &= expect(context_h.find("dflash_target_kv_cache_update_gpu") != std::string::npos &&
                 llama_h.find("llama_dflash_target_kv_cache_update_from_ring") != std::string::npos &&
                 context_cpp.find("dflash_target_kv_cache_update_gpu") != std::string::npos &&
                 speculative.find("llama_dflash_target_kv_cache_update_from_ring(") != std::string::npos,
        "DFlash accepted target hiddens must also refresh the drafter base KV cache for full-attention layers");
    ok &= expect(speculative.find("const int n_full_kv_update = std::min(n_update, drafter_prefix_window());") != std::string::npos,
        "DFlash accepted-prefix full-KV commits must clamp to the drafter prefix window so small -cd stays safe");
    ok &= expect(context_cpp.find("base_kv->seq_rm(seq_id, start_pos, -1);") != std::string::npos,
        "DFlash accepted-prefix full-KV commits must replace only the tail suffix instead of overwriting occupied base-KV cells in place");
    ok &= expect(speculative.find("if (has_dflash) {\n            if (!has_copyspec)") == std::string::npos,
        "DFlash must not implicitly stack CopySpec when --spec-type dflash is requested");
    ok &= expect(context_cpp.find("reuse_update_buf &&\n            dflash_kv_cache &&\n            dflash_kv_cache->fn_wait_backend_stream &&\n            dflash_kv_cache->fn_wait_backend_stream(gpu_backend)") != std::string::npos,
        "DFlash accepted-prefix full-KV commits must stream-order reused GPU scratch before the next commit reuses update_buf");
    ok &= expect(kv_cache_iswa_cpp.find("return std::max(kv_base->seq_pos_max(seq_id), kv_swa->seq_pos_max(seq_id));") != std::string::npos,
        "ISWA seq_pos_max must follow the newest accepted suffix even when SWA layers source live context outside kv_swa");
    ok &= expect(speculative.find("int drafter_prefix_window() const") != std::string::npos &&
                 speculative.find("trim_drafter_prefix_window();") != std::string::npos &&
                 speculative.find("llama_memory_seq_rm(mem_dft, seq_id, 0, trim_before);") != std::string::npos,
        "DFlash drafter KV must slide forward with the accepted suffix instead of staying anchored at position 0");
    ok &= expect(kv_cache_h.find("set_input_k_idxs_backend") != std::string::npos &&
                 kv_cache_h.find("set_input_v_idxs_backend") != std::string::npos &&
                 kv_cache_cpp.find("set_input_k_idxs_backend") != std::string::npos &&
                 kv_cache_cpp.find("set_input_v_idxs_backend") != std::string::npos,
        "DFlash manual GPU-only commit graphs must have backend-safe KV index upload helpers");
    ok &= expect(kv_cache_h.find("llama_kv_cache * get_kv() const;") != std::string::npos &&
                 kv_cache_h.find("const llama_kv_cache::slot_info & current_sinfo() const;") != std::string::npos &&
                 kv_cache_cpp.find("llama_kv_cache_context::current_sinfo() const") != std::string::npos,
        "DFlash commit graphs must expose the resolved KV slot mapping so graph inputs can snapshot it safely");
    ok &= expect(context_cpp.find("dflash_kv_update_gpu") != std::string::npos, "DFlash K/V update must use a temporary input source separate from the main cross state");
    ok &= expect(cuda_cpp.find("dflash_kv_update") != std::string::npos, "DFlash K/V update graph must be excluded from CUDA graph capture");
    ok &= expect(cuda_ring.find("dflash_kv_cache_write_d2d") != std::string::npos, "CUDA backend must provide D2D K/V cache ring writes");
    ok &= expect(cuda_ring.find("dflash_kv_cache_write_d2d_no_check") != std::string::npos, "CUDA backend must provide batched no-sync D2D K/V cache ring writes");
    ok &= expect(context_cpp.find("const bool needs_shift = total > dflash_kv_cache->ring_size;") != std::string::npos &&
                 context_cpp.find("shift_buf") != std::string::npos &&
                 context_cpp.find("shift_ptr") != std::string::npos &&
                 context_cpp.find("fn_copy(cache, dflash_kv_cache->shift_ptr, keep_bytes)") != std::string::npos,
        "DFlash K/V cache overflow must move the live suffix through a scratch buffer instead of shifting the cache in place");
    ok &= expect(cuda_ring.find("dflash_kv_cache_interleave") != std::string::npos, "CUDA backend must stage ordered DFlash K/V cache windows");
    ok &= expect(cuda_cpp.find("dflash_kv_cache_interleave") != std::string::npos, "CUDA backend registry must expose DFlash K/V cache helpers");
    ok &= expect(context_h.find("fn_copy_d2d_no_check") != std::string::npos, "DFlash K/V cache must track hot-loop D2D copy helper for safe overflow compaction");
    ok &= expect(context_h.find("fn_sync_ptr") != std::string::npos, "DFlash K/V cache no-sync writes must have a batched stream-sync helper");
    ok &= expect(context_cpp.find("fn_copy_d2d_no_check") != std::string::npos, "DFlash K/V cache update must prefer no-sync hot-loop D2D copies");
    ok &= expect(context_cpp.find("if (used_async_copy || (!needs_shift && fast_append))") != std::string::npos &&
                 context_cpp.find("dflash_kv_cache->fn_wait_dflash_stream(gpu_backend)") != std::string::npos &&
                 context_cpp.find("dflash_kv_cache->fn_sync_ptr(dflash_kv_cache->k_ring[0]->data)") != std::string::npos,
        "DFlash K/V cache no-sync writes must synchronize or stream-order before the drafter graph reads cached K/V");
    ok &= expect(cuda_reg.find("\"dflash_kv_cache_write_d2d_no_check\"") != std::string::npos, "CUDA backend registry must publish no-sync DFlash K/V cache ring writes");
    ok &= expect(speculative.find("const int draft_pos_base = committed_len;") != std::string::npos &&
                 speculative.find("common_batch_add(batch_dft, id_last, draft_pos_base, { seq_id }, true);") != std::string::npos &&
                 speculative.find("common_batch_add(batch, id_last_per_spec[rs.spec_idx], rs.draft_pos_base, { rs.seq_id }, true);") != std::string::npos,
        "DFlash flat/tree/batched drafts must keep absolute target positions for seed-token row alignment");
    ok &= expect(speculative.find("llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, committed_len, -1);") != std::string::npos &&
                 speculative.find("llama_memory_seq_rm(llama_get_memory(ctx_dft), rs.seq_id, rs.draft_pos_base, -1);") != std::string::npos,
        "DFlash drafter must trim each slot back to its accepted prefix before the next block draft so stale proposal state cannot survive into the next cycle");
    ok &= expect(arg_cpp.find("params.speculative.draft.n_ctx = 256;") != std::string::npos &&
                 arg_cpp.find("drafter doesn't need the full main ctx") != std::string::npos,
        "DFlash default -cd must stay at the production 256-token drafter context unless the user overrides it");
    ok &= expect(speculative.find("float * logits = llama_get_logits_ith(ctx_dft, i);") != std::string::npos, "DFlash flat draft fallback rows must preserve seed-token offset");
    ok &= expect(speculative.find("const int offset = r * batch_len;") != std::string::npos, "DFlash batched draft argmax row offsets must preserve seed-token output rows");
    ok &= expect(speculative.find("graph_reuse=%d") != std::string::npos, "DFlash draft profile must report drafter graph reuse");
    ok &= expect(context_cpp.find("output_reorder();\n    if (logits_argmax_buf.empty())") != std::string::npos, "argmax row access must honor output reordering");
    ok &= expect(context_cpp.find("std::swap(logits_argmax_buf") != std::string::npos, "output reordering must include reduced logits ids");
    ok &= expect(context_cpp.find("std::swap(logits_argmax_prob_buf") != std::string::npos, "output reordering must include reduced logits probabilities");
    ok &= expect(qwen35.find("ggml_build_forward_expand(gf, ggml_cpy(ctx0, qkv_cont, qkv_dst))") != std::string::npos, "Qwen3.5 must graph-copy QKV into GPU tape");
    ok &= expect(qwen35moe.find("ggml_build_forward_expand(gf, ggml_cpy(ctx0, qkv_cont, qkv_dst))") != std::string::npos, "Qwen3.5-MoE must graph-copy QKV into GPU tape");
    ok &= expect(qwen35.find("const bool need_full_h_nextn = cparams.embeddings_nextn && !cparams.embeddings_nextn_masked;") != std::string::npos &&
                 qwen35.find("inp_out_ids && !need_full_h_nextn") != std::string::npos &&
                 qwen35.find("if (need_full_h_nextn && inp_out_ids)") != std::string::npos &&
                 qwen35.find("res->t_h_nextn = cur;") != std::string::npos,
        "Qwen3.5 prefill must gather final-layer output rows early unless unmasked nextn output is requested");
    ok &= expect(qwen35moe.find("const bool need_full_h_nextn = cparams.embeddings_nextn && !cparams.embeddings_nextn_masked;") != std::string::npos &&
                 qwen35moe.find("inp_out_ids && !need_full_h_nextn") != std::string::npos &&
                 qwen35moe.find("if (need_full_h_nextn && inp_out_ids)") != std::string::npos &&
                 qwen35moe.find("res->t_h_nextn = cur;") != std::string::npos,
        "Qwen3.5-MoE prefill must gather final-layer output rows early unless unmasked nextn output is requested");
    ok &= expect(qwen35.find("const int64_t n_head_kv_il = hparams.n_head_kv(il);") != std::string::npos &&
                 qwen35.find("Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv_il, n_tokens);") != std::string::npos &&
                 qwen35.find("Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv_il, n_tokens);") != std::string::npos,
        "Qwen3.5 full-attention layers must use per-layer n_head_kv for RYS-compatible KV shapes");
    ok &= expect(qwen35moe.find("const int64_t n_head_kv_il = hparams.n_head_kv(il);") != std::string::npos &&
                 qwen35moe.find("Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv_il, n_tokens);") != std::string::npos &&
                 qwen35moe.find("Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv_il, n_tokens);") != std::string::npos,
        "Qwen3.5-MoE full-attention layers must use per-layer n_head_kv for RYS-compatible KV shapes");
    ok &= expect(delta_net_base.find("ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s);") != std::string::npos &&
                 delta_net_base.find("ggml_reshape_3d(ctx0, s, S_v * S_v * H_v, 1, n_seqs)") == std::string::npos,
        "DeltaNet fused K=1 path must pass the existing 4D recurrent state directly instead of adding a per-layer reshape");
    ok &= expect(delta_net_base.find("tree_parent_ids && n_seq_tokens > 1 && n_seqs_in == 1 && tree_ssm_intermediates") != std::string::npos &&
                 delta_net_base.find("n_seq_tokens <= ggml_nelements(tree_parent_ids)") != std::string::npos &&
                 delta_net_base.find("ggml_gated_delta_net_tree(ctx0, q, k, v, g, b, s, tree_parent_ids, persist_inter)") != std::string::npos &&
                 delta_net_base.find("cb(result, \"fgdn_tree\", il)") != std::string::npos,
        "DeltaNet must preserve the DDTree tree-aware GDN path for single-sequence multi-token verification batches");
    {
        const std::string fgdn_resolve = slice_between(
            context_cpp,
            "if (cparams.auto_fgdn) {\n        LLAMA_LOG_INFO(\"%s: resolving fused Gated Delta Net support:",
            "        cparams.auto_fgdn = false;");
        ok &= expect(fgdn_resolve.find("graph_reserve(1, n_seqs, n_outputs, mctx.get(), true)") != std::string::npos &&
                     fgdn_resolve.find("device_gdn != device_kv") != std::string::npos,
            "fused GDN auto mode must rely on scheduler capability/device-placement probing");
        ok &= expect(context_cpp.find("fused Gated Delta Net disabled (non-CUDA backend)") == std::string::npos &&
                     context_cpp.find("have_cuda_gpu") == std::string::npos,
            "fused GDN must not be disabled before probing just because the backend is ROCm");
        ok &= expect(cuda_cpp.find("case GGML_OP_GATED_DELTA_NET:") != std::string::npos &&
                     cuda_cpp.find("#ifdef GGML_USE_MUSA\n            return false;\n#else\n            return true;\n#endif // GGML_USE_MUSA") != std::string::npos,
            "CUDA/HIP backend support must keep GATED_DELTA_NET enabled for ROCm while excluding MUSA");
    }
    ok &= expect(qwen35.find("const bool tree_mode = (tree_parent_ids != nullptr && n_seq_tokens > 1 && n_seqs == 1") != std::string::npos &&
                 qwen35.find("ggml_ssm_conv_tree(ctx0, conv_input, conv_kernel, tree_parent_ids)") != std::string::npos &&
                 qwen35.find("conv_output_silu = conv_output_proper;") != std::string::npos,
        "Qwen3.5 linear attention must preserve DDTree tree-mode SSM convolution without an extra SiLU");
    ok &= expect(qwen35moe.find("const bool tree_mode = (tree_parent_ids != nullptr && n_seq_tokens > 1 && n_seqs == 1") != std::string::npos &&
                 qwen35moe.find("ggml_ssm_conv_tree(ctx0, conv_input, conv_kernel, tree_parent_ids)") != std::string::npos &&
                 qwen35moe.find("conv_output_silu = conv_output_proper;") != std::string::npos,
        "Qwen3.5-MoE linear attention must preserve DDTree tree-mode SSM convolution without an extra SiLU");
    ok &= expect(dflash_profile_h.find("GGML_DFLASH_PROFILE") != std::string::npos, "DFlash profile helper must honor profiling flag");
    ok &= expect(speculative.find("gpu_sync=%.3f ms") != std::string::npos, "DFlash ring profiling must report GPU sync time");
    ok &= expect(speculative.find("kv_cache_update requested=%d update=%d") != std::string::npos, "DFlash accept profiling must report drafter K/V cache update time");
    ok &= expect(context_h.find("struct dflash_hidden_gpu") != std::string::npos, "DFlash must define GPU hidden capture storage");
    ok &= expect(cparams_h.find("hidden_gpu_seqs") != std::string::npos, "graph params must expose per-seq GPU hidden targets");
    ok &= expect(context_h.find("bool gpu_capture_enabled = true") != std::string::npos, "DFlash capture must be able to force CPU hidden callbacks");
    ok &= expect(context_h.find("multi_gpu_replay_fallback_logged") != std::string::npos, "DFlash capture must log multi-GPU recurrent replay fallback once");
    ok &= expect(llama_h.find("llama_set_dflash_gpu_capture") != std::string::npos, "public DFlash API must expose GPU capture gating");
    ok &= expect(llama_h.find("llama_dflash_allow_multi_gpu_tape") != std::string::npos,
        "public DFlash API must expose the multi-GPU tape policy");
    ok &= expect(context_cpp.find("GGML_DFLASH_MULTI_GPU_TAPE") != std::string::npos &&
                 context_cpp.find("GGML_DFLASH_ALLOW_MULTI_GPU_TAPE") != std::string::npos,
        "DFlash multi-GPU tape must be default-enabled with explicit env kill switches");
    const std::string dflash_wait_gpu_capture = slice_between(
        context_cpp,
        "bool llama_context::dflash_wait_for_gpu_capture_stream()",
        "bool llama_context::dflash_memory_seq_cp_recurrent_ordered");
    ok &= expect(dflash_wait_gpu_capture.find("model.n_devices() > 1") != std::string::npos &&
                 dflash_wait_gpu_capture.find("scheduler sync") != std::string::npos &&
                 dflash_wait_gpu_capture.find("peer D2D") != std::string::npos,
        "multi-GPU DFlash hidden capture must use scheduler sync before peer D2D ring writes");
    ok &= expect(context_cpp.find("allocate_hidden_gpu(n_slots, max_tokens)") != std::string::npos, "GPU tape allocation must allocate hidden capture buffers too");
    ok &= expect(context_cpp.find("dflash_skip_eval_callback ? nullptr : dflash_eval_callback") != std::string::npos, "eligible DFlash verifier graph must disable eval callback, including suppressed no-intersection ubatches");
    ok &= expect(context_cpp.find("bool dflash_graph_tape_ready") != std::string::npos, "DFlash decode must gate GPU tape copies separately from hidden capture");
    ok &= expect(context_cpp.find("dflash_graph_hidden_ready =\n                            !dflash_capture->hidden_gpu.empty()") != std::string::npos ||
                 context_cpp.find("!dflash_capture->hidden_gpu.empty() &&\n                            dflash_gpu_capture_ready") != std::string::npos,
        "GPU hidden graph capture must not depend on active tape recording");
    ok &= expect(context_cpp.find("dflash_tape_gpu * graph_tp = dflash_graph_tape_ready ? tp : nullptr") != std::string::npos, "GPU tape graph pointers must be disabled when tape recording is inactive");
    ok &= expect(context_cpp.find("multi-GPU target detected") != std::string::npos, "multi-GPU target capture/tape decisions must be logged");
    ok &= expect(context_cpp.find("dflash_gpu_capture_ready = (model.n_devices() <= 1 || llama_dflash_allow_multi_gpu_tape()) && dflash_capture->gpu_capture_enabled") != std::string::npos,
        "DFlash graph capture must be enabled for default-on multi-GPU tape when GPU capture is otherwise enabled");
    ok &= expect(context_cpp.find("model.n_devices() > 1 && !llama_dflash_allow_multi_gpu_tape()") != std::string::npos,
        "DFlash multi-GPU fallback must be controlled by the explicit kill switch instead of a hard device-count gate");
    ok &= expect(context_cpp.find("const bool multi_gpu_target = model.n_devices() > 1;") != std::string::npos, "DFlash replay must detect multi-GPU target placement before choosing GPU replay");
    ok &= expect(context_cpp.find("exact CUDA DFlash replay unavailable, using CPU recurrent replay fallback") != std::string::npos, "DFlash replay must log only the multi-GPU CPU replay fallback");
    {
        const size_t direct_replay = context_cpp.find("if (use_gpu_tape && tape_replay_gdn_direct_gpu(mem_recurrent, cell_idx, n_accepted))");
        const size_t multi_gpu_fallback = context_cpp.find("if (multi_gpu_target)", direct_replay);
        ok &= expect(direct_replay != std::string::npos &&
                     multi_gpu_fallback != std::string::npos &&
                     direct_replay < multi_gpu_fallback,
            "DFlash multi-GPU replay must try device-aware GPU tape replay before CPU fallback");
    }
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
    ok &= expect(context_h.find("std::vector<ggml_backend_buffer_t> bufs") != std::string::npos &&
                 context_h.find("std::vector<ggml_context *> ctxs") != std::string::npos,
        "DFlash hidden/prefill GPU capture must own per-layer buffers for split-device placement");
    ok &= expect(context_h.find("ggml_backend_buffer_t buf = nullptr") != std::string::npos &&
                 context_h.find("ggml_backend_dev_t dev = nullptr") != std::string::npos,
        "DFlash GPU tape layers must own per-layer backend buffers and device placement");
    ok &= expect(context_cpp.find("model.dev_layer(il)") != std::string::npos &&
                 context_cpp.find("dflash tape placement") != std::string::npos,
        "DFlash GPU tape allocation must follow each recurrent layer device placement");
    ok &= expect(context_h.find("capture_wait_backends") != std::string::npos &&
                 context_cpp.find("capture_wait_backends") != std::string::npos,
        "DFlash graph capture stream waits must cover all GPU backends touched by split-device capture");
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
    ok &= expect(
        cuda_ring.find("dflash_cuda_enable_peer_access") != std::string::npos &&
        cuda_ring.find("cudaDeviceCanAccessPeer") != std::string::npos &&
        cuda_ring.find("cudaDeviceEnablePeerAccess") != std::string::npos &&
        cuda_ring.find("cudaErrorPeerAccessAlreadyEnabled") != std::string::npos &&
        cuda_ring.find("GGML_CUDA_MAX_DEVICES") != std::string::npos,
        "DFlash D2D helpers must lazily enable CUDA/ROCm peer access using GGML_CUDA_MAX_DEVICES bounds");
    const size_t write_d2d_fn = cuda_ring.find("dflash_cross_ring_gpu_write_d2d");
    const size_t write_d2d_end = cuda_ring.find("k_dflash_rebuild_conv_state", write_d2d_fn);
    ok &= expect(
        write_d2d_fn != std::string::npos &&
        write_d2d_end != std::string::npos &&
        cuda_ring.find("dflash_cuda_enable_peer_access(ring->device, attr.device)", write_d2d_fn) < write_d2d_end &&
        cuda_ring.find("cudaMemcpyPeerAsync", write_d2d_fn) < write_d2d_end,
        "DFlash hidden ring D2D writes must allow peer device pointers after enabling peer access");
    ok &= expect(
        write_d2d_fn != std::string::npos &&
        write_d2d_end != std::string::npos &&
        cuda_ring.find("cudaGetDevice(&prev_device)", write_d2d_fn) < write_d2d_end &&
        cuda_ring.find("cudaSetDevice(prev_device)", write_d2d_fn) < write_d2d_end,
        "DFlash hidden ring D2D writes must restore the caller CUDA device after peer copies");
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
    ok &= expect(context_cpp.find("if (tape_replay_conv_gpu(mem_recurrent, cell_idx, n_accepted))") != std::string::npos,
        "conv replay must try the self-gated GPU rebuild path for split-device placement too");
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
    ok &= expect(context_cpp.find("fn_ptr_device(launch.state, &device)") != std::string::npos,
        "direct DFlash GPU replay must resolve the CUDA device while validating recurrent state pointers");
    ok &= expect(context_cpp.find("fn_ptr_device(launch.k, &k_device)") != std::string::npos &&
                 context_cpp.find("k_device != device || v_device != device || gate_device != device || beta_device != device") != std::string::npos,
        "direct DFlash GPU replay must ensure every per-layer input is local to that layer's recurrent state device");
    ok &= expect(context_cpp.find("replay_sync_ptrs.push_back(launch.state)") != std::string::npos,
        "direct DFlash GPU replay must sync every touched device when launches span devices");
    ok &= expect(context_cpp.find("fn_sync_device(dflash_capture->replay_sync_device)") != std::string::npos ||
                 context_cpp.find("for (const void * ptr : dflash_capture->replay_sync_ptrs)") != std::string::npos,
        "direct DFlash GPU replay sync must wait on all validated replay CUDA devices");
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
    ok &= expect(cuda_ring.find("dflash_cuda_set_device") != std::string::npos &&
                 cuda_reg.find("\"dflash_cuda_set_device\"") != std::string::npos,
        "CUDA backend must expose a DFlash set-device helper for prevalidated hot-loop copies");
    ok &= expect(memory_recurrent.find("dflash_cuda_ptr_device") != std::string::npos,
        "recurrent copy plan must validate CUDA device pointers when the plan is built");
    ok &= expect(memory_recurrent.find("dflash_cuda_set_device") != std::string::npos,
        "recurrent copy hot path must set the cached CUDA device without per-tensor pointer validation");
    ok &= expect(memory_recurrent.find("dflash_cuda_synchronize_device") != std::string::npos,
        "recurrent backup copy must synchronize once on the cached CUDA device");
    ok &= expect(memory_recurrent.find("copy_plan_device") == std::string::npos,
        "recurrent backup copy plan must not assume all recurrent tensors are on one CUDA device");
    ok &= expect(memory_recurrent.find("copy_plan_touched_devices") != std::string::npos &&
                 memory_recurrent.find("entry.device") != std::string::npos,
        "recurrent backup copy plan must track and synchronize all touched CUDA/ROCm devices");
    ok &= expect(memory_recurrent.find("fn_prepare(dst)") == std::string::npos,
        "recurrent copy hot path must not prepare every destination tensor");
    ok &= expect(memory_recurrent.find("sync_ptrs.push_back(dst)") == std::string::npos,
        "recurrent backup copy must not retain one sync pointer per recurrent tensor");
    ok &= expect(memory_recurrent.find("build_recurrent_copy_plan") != std::string::npos &&
                 memory_recurrent.find("copy_plan_valid = false") != std::string::npos,
        "recurrent memory must cache and invalidate its CUDA copy plan");
    ok &= expect(memory_recurrent.find("seq_cp_recurrent_no_sync") != std::string::npos, "recurrent memory must expose DFlash no-sync restore");
    ok &= expect(context_cpp.find("seq_cp_recurrent_no_sync(seq_backup, seq_id") != std::string::npos, "DFlash rollback must avoid synchronous recurrent restore before replay");
    ok &= expect(context_cpp.find("recurrent_restore=%.3f ms") != std::string::npos, "DFlash rollback profiling must expose recurrent restore cost");
    ok &= expect(context_cpp.find("rollback_restore_enqueue=") != std::string::npos &&
                 context_cpp.find("rollback_restore_sync=") != std::string::npos &&
                 context_cpp.find("rollback_restore_fallback=") != std::string::npos,
        "DFlash rollback profiling must expose recurrent restore enqueue/sync/fallback counters");
    ok &= expect(memory_h.find("llama_memory_recurrent_copy_profile") != std::string::npos &&
                 memory_h.find("layers_scanned") != std::string::npos &&
                 memory_h.find("cuda_d2d_queued") != std::string::npos &&
                 memory_h.find("fallback_copies") != std::string::npos,
        "recurrent memory must expose copy profile counters for DFlash CPU-bubble diagnostics");
    ok &= expect(memory_hybrid_h.find("recurrent_copy_profile_reset() override") != std::string::npos &&
                 memory_hybrid.find("mem_recr->recurrent_copy_profile_reset()") != std::string::npos &&
                 memory_hybrid.find("return mem_recr->recurrent_copy_profile()") != std::string::npos,
        "hybrid memory must forward recurrent copy profile counters from the recurrent cache");
    ok &= expect(memory_hybrid_iswa_h.find("recurrent_copy_profile_reset() override") != std::string::npos &&
                 memory_hybrid_iswa.find("mem_recr->recurrent_copy_profile_reset()") != std::string::npos &&
                 memory_hybrid_iswa.find("return mem_recr->recurrent_copy_profile()") != std::string::npos,
        "hybrid-ISWA memory must forward recurrent copy profile counters from the recurrent cache");
    ok &= expect(server_context.find("backup_enqueue=") != std::string::npos &&
                 server_context.find("backup_sync=") != std::string::npos &&
                 server_context.find("backup_fallback=") != std::string::npos,
        "server DFlash cycle log must surface recurrent backup enqueue/sync/fallback counters");
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
    ok &= expect(graph_h.find("cparams.dflash_reduced_consumer_active == other.cparams.dflash_reduced_consumer_active") != std::string::npos,
        "graph reuse must invalidate when reduced verifier raw-logit output lifetime changes");
    ok &= expect(graph_cpp.find("skip_logits_output") != std::string::npos &&
                 graph_cpp.find("dflash_reduced_consumer_active && t_logits_argmax != nullptr") != std::string::npos,
        "reduced verifier graphs must not keep full raw logits as graph outputs when compact logits are consumed");
    ok &= expect(dflash_draft.find("if (cparams.dflash_reduced_consumer_active)") != std::string::npos &&
                 dflash_draft.find("ggml_build_forward_expand(gf, cur);") != std::string::npos &&
                 speculative.find("llama_model_dev_output(params.model_dft)") != std::string::npos &&
                 speculative.find("draft backend sampling disabled on Meta output placement") != std::string::npos &&
                 speculative.find("llama_set_dflash_consume_reduced(ctx_dft, draft_reduced_logits)") != std::string::npos,
        "DFlash drafter compact argmax/top-k must fall back to full logits on Meta output placement");
    ok &= expect(context_cpp.find("dflash_reduced_logits_only") != std::string::npos &&
                 context_cpp.find("has_logits = !dflash_reduced_logits_only") != std::string::npos,
        "reduced verifier decode must avoid reserving a raw logits CPU output buffer");
    const size_t set_verify_pos = context_cpp.find("void llama_context::set_dflash_verify_logits");
    const size_t set_slots_pos = context_cpp.find("void llama_context::set_dflash_n_slots", set_verify_pos);
    ok &= expect(set_verify_pos != std::string::npos && set_slots_pos != std::string::npos &&
            context_cpp.substr(set_verify_pos, set_slots_pos - set_verify_pos).find("gf_res_prev->reset()") == std::string::npos,
            "DFlash verifier-logit toggles must not reset graph reuse on every verify cycle");
    ok &= expect(qwen35.find("cparams.dflash_verify_logits") != std::string::npos, "Qwen3.5 target graph must emit reduced verifier logits only when gated");
    ok &= expect(qwen35moe.find("cparams.dflash_verify_logits") != std::string::npos, "Qwen3.5-MoE target graph must emit reduced verifier logits only when gated");
    ok &= expect(gemma4_iswa.find("cparams.dflash_verify_logits") != std::string::npos, "Gemma4-ISWA target graph must emit reduced verifier logits only when gated");
    ok &= expect(qwen35.find("const bool dflash_compact_verifier_only") != std::string::npos &&
                 qwen35.find("if (!dflash_compact_verifier_only)") != std::string::npos,
        "Qwen3.5 reduced verifier graph must use the named compact-verifier-only guard");
    ok &= expect(qwen35moe.find("const bool dflash_compact_verifier_only") != std::string::npos &&
                 qwen35moe.find("if (!dflash_compact_verifier_only)") != std::string::npos,
        "Qwen3.5-MoE reduced verifier graph must use the named compact-verifier-only guard");
    ok &= expect(gemma4_iswa.find("const bool dflash_compact_verifier_only") != std::string::npos &&
                 gemma4_iswa.find("if (!dflash_compact_verifier_only)") != std::string::npos,
        "Gemma4-ISWA reduced verifier graph must use the named compact-verifier-only guard");
    ok &= expect(gemma4_iswa.find("ggml_topk_ext(ctx0, cur, topk, 0.0f, 0)") != std::string::npos, "Gemma4-ISWA target verifier top-K must emit raw logit candidates");
    ok &= expect(gemma4_iswa.find("#include <algorithm>") != std::string::npos, "Gemma4-ISWA must include <algorithm> for std::max/std::min in verifier path");
    ok &= expect(context_h.find("profile_output_extract_us") != std::string::npos,
        "DFlash profile must track verifier output extraction time");
    ok &= expect(context_h.find("profile_raw_logits_skipped") != std::string::npos,
        "DFlash profile must count compact verifier raw-logit skips");
    ok &= expect(context_cpp.find("profile_output_extract_us") != std::string::npos &&
                 context_cpp.find("profile_raw_logits_skipped += n_outputs") != std::string::npos,
        "DFlash decode must account for output extraction time and skipped raw-logit rows");
    ok &= expect(context_cpp.find("raw_logits_skipped=") != std::string::npos &&
                 context_cpp.find("raw_logits_skipped_bytes_est=") != std::string::npos,
        "DFlash profile log must print skipped raw-logit rows and estimated bytes");
    ok &= expect(context_h.find("profile_replay_gdn_enqueue_us") != std::string::npos &&
                 context_h.find("profile_replay_gdn_wait_us") != std::string::npos &&
                 context_h.find("profile_replay_conv_enqueue_us") != std::string::npos &&
                 context_h.find("profile_replay_conv_wait_us") != std::string::npos,
        "DFlash profile must split Qwen replay into GDN enqueue/wait and conv enqueue/wait substages");
    ok &= expect(context_h.find("profile_replay_layers") != std::string::npos &&
                 context_h.find("profile_replay_sync_calls") != std::string::npos,
        "DFlash profile must count replayed recurrent layers and CUDA sync calls");
    ok &= expect(context_cpp.find("replay_path=direct-gpu") != std::string::npos &&
                 context_cpp.find("replay_path=ggml-gpu") != std::string::npos &&
                 context_cpp.find("replay_path=cpu-fallback") != std::string::npos,
        "DFlash profile log must name the replay backend path");
    ok &= expect(cuda_ring.find("dflash_cuda_synchronize_device") != std::string::npos &&
                 cuda_cpp.find("dflash_cuda_synchronize_device") != std::string::npos,
        "CUDA backend must expose single-device DFlash stream synchronization");
    ok &= expect(cuda_cpp.find("dflash_cuda_backend_wait_for_dflash_stream") != std::string::npos &&
                 cuda_cpp.find("cudaEventRecord(event, cudaStreamPerThread)") != std::string::npos &&
                 cuda_cpp.find("cudaStreamWaitEvent(cuda_ctx->stream(), event, 0)") != std::string::npos,
        "CUDA backend must expose an event wait from the DFlash per-thread stream to the GGML compute stream");
    ok &= expect(context_cpp.find("dflash_cuda_synchronize_device") != std::string::npos &&
                 context_cpp.find("replay_sync_device") != std::string::npos,
        "direct GPU tape replay must synchronize the relevant device once instead of per recurrent layer");
    ok &= expect(context_h.find("dflash_memory_seq_cp_recurrent_ordered") != std::string::npos &&
                 context_cpp.find("seq_cp_recurrent_no_sync(seq_id_src, seq_id_dst") != std::string::npos &&
                 context_cpp.find("dflash_cuda_backend_wait_for_dflash_stream") != std::string::npos,
        "DFlash recurrent backup must copy without host sync and order the verifier backend stream after the DFlash stream");
    ok &= expect(llama_h.find("llama_dflash_memory_seq_cp_recurrent_ordered") != std::string::npos &&
                 context_cpp.find("bool llama_dflash_memory_seq_cp_recurrent_ordered(") != std::string::npos,
        "DFlash ordered recurrent backup must be exported through the C API for Windows DLL consumers");
    ok &= expect(context_h.find("fn_wait_backend_stream") != std::string::npos &&
                 context_h.find("fn_wait_dflash_stream") != std::string::npos &&
                 context_cpp.find("dflash_cuda_backend_wait_for_stream") != std::string::npos &&
                 context_cpp.find("dflash_kv_cache->fn_wait_backend_stream(gpu_backend)") != std::string::npos &&
                 context_cpp.find("dflash_kv_cache->fn_wait_dflash_stream(gpu_backend)") != std::string::npos,
        "DFlash drafter K/V cache update must use stream ordering around async append with host-sync fallback");
    ok &= expect(server_context.find("dflash_backup_recurrent_state") != std::string::npos &&
                 server_context.find("llama_dflash_memory_seq_cp_recurrent_ordered") != std::string::npos,
        "server DFlash recurrent backup must use the ordered async backup path when available");
    ok &= expect(server_context.find("t_replay_sync_total") != std::string::npos &&
                 server_context.find("t_recurrent_backup_total") != std::string::npos &&
                 server_context.find("t_tape_record_total") != std::string::npos,
        "server DFlash profile must account for replay sync, recurrent backup, and tape-recording subcomponents");
    ok &= expect(server_context.find("replay_sync=") != std::string::npos &&
                 server_context.find("recurrent_backup=") != std::string::npos &&
                 server_context.find("tape_record=") != std::string::npos,
        "server speculative cycle log must print DFlash CPU-bubble subcomponents");
    ok &= expect(server_context.find("dflash_log_reduced_verify_decision") != std::string::npos &&
                 server_context.find("dflash reduced verifier decision:") != std::string::npos,
        "server must log one reduced-verifier batch decision behind GGML_DFLASH_PROFILE");
    ok &= expect(server_context.find("dflash compact-output mismatch:") != std::string::npos,
        "server must log reduced verifier compact-output mismatches behind GGML_DFLASH_PROFILE");
    ok &= expect(server_context.find("DFLASH_PROFILE_VERIFY") != std::string::npos &&
                 server_context.find("DFLASH_PROFILE_PREFILL") != std::string::npos &&
                 speculative.find("DFLASH_PROFILE_COPY") != std::string::npos &&
                 context_cpp.find("DFLASH_PROFILE_TRACE") != std::string::npos,
        "DFlash profile logging must use profile categories instead of one broad boolean gate");
    ok &= expect(server_context.find("dflash_slot_in_view") != std::string::npos,
        "server must share DFlash slot-in-view detection between per-view scans");
    {
        const size_t helper_pos = server_context.find("static bool dflash_view_has_unexpected_prompt_logits");
        const size_t helper_end = helper_pos == std::string::npos ? std::string::npos :
            server_context.find("static ", helper_pos + 1);
        const std::string helper = helper_pos == std::string::npos || helper_end == std::string::npos
            ? std::string()
            : server_context.substr(helper_pos, helper_end - helper_pos);
        ok &= expect(!helper.empty() &&
                     helper.find("need_sampling()") != std::string::npos &&
                     helper.find("need_embd()") == std::string::npos,
            "DFlash prompt/prefill raw-logit diagnostic must be based on sampling prompt state, not need_embd()");
    }
    {
        const size_t dflash_pos = speculative.find("struct common_speculative_impl_dflash");
        const size_t dflash_end = dflash_pos == std::string::npos ? std::string::npos :
            speculative.find("private:", dflash_pos);
        const std::string dflash_impl = dflash_pos == std::string::npos || dflash_end == std::string::npos
            ? std::string()
            : speculative.substr(dflash_pos, dflash_end - dflash_pos);
        ok &= expect(!dflash_impl.empty() &&
                     dflash_impl.find("bool need_embd() const override {\n        return false;\n    }") != std::string::npos,
            "DFlash must not request post-norm embeddings because that forces prompt prefill to output full raw logits");
    }
    ok &= expect(speculative.find("n_target_features != n_embd * n_target_layers") != std::string::npos,
        "DFlash must validate n_target_features against n_embd * n_target_layers");
    ok &= expect(speculative.find("target_layer_ids contain duplicates") != std::string::npos,
        "DFlash must reject duplicate target_layer_ids");
    ok &= expect(speculative.find("target_layer_id") != std::string::npos &&
                 speculative.find("outside target layer range") != std::string::npos,
        "DFlash must validate target_layer_ids against target model layer range");
    ok &= expect(speculative.find("dflash: contract ok:") != std::string::npos,
        "DFlash must log validated drafter/target contract");
    ok &= expect(speculative.find("!common_speculative_are_compatible(model_tgt, model_dft_)") != std::string::npos &&
                 speculative.find("dflash: target and drafter vocab are incompatible") != std::string::npos,
        "DFlash must fail closed on incompatible target/drafter vocabs");
    ok &= expect(count_occurrences(speculative, "result.size() >= params.n_min") >= 2,
        "DFlash GPU argmax p_min discard must respect n_min on both single and batched draft paths");
    ok &= expect(download_h.find("std::string dflash_draft_path;") != std::string::npos &&
                 download_h.find("bool download_dflash = false") != std::string::npos,
        "download result/API must expose optional DFlash draft discovery");
    ok &= expect(download_cpp.find("find_best_dflash") != std::string::npos &&
                 download_cpp.find("\"dflash-\"") != std::string::npos &&
                 download_cpp.find("\"draft-dflash-\"") != std::string::npos,
        "download code must discover dflash- and draft-dflash- sibling GGUFs");
    ok &= expect(download_cpp.find("model_common != model_dir || dflash_common != dflash_dir") != std::string::npos,
        "DFlash sibling discovery must require the full parent directory to match");
    ok &= expect(download_cpp.find("filename.rfind(\"dflash-\", 0) != 0") != std::string::npos &&
                 download_cpp.find("filename.rfind(\"draft-dflash-\", 0) != 0") != std::string::npos,
        "download code must exclude DFlash drafts from primary model selection");
    ok &= expect(arg_cpp.find("found_dflash") != std::string::npos &&
                 arg_cpp.find("params.speculative.draft.mparams.path = res.dflash.path") != std::string::npos,
        "arg handling must auto-fill DFlash draft path only after primary model handling");
    ok &= expect(dflash_draft.find("n_feat != target_hidden->ne[0]") != std::string::npos,
        "DFlash drafter input must reject cross feature-size mismatches");
    ok &= expect(dflash_draft.find("ggml_backend_tensor_memset(target_hidden, 0, 0, tensor_bytes)") != std::string::npos,
        "DFlash drafter input must zero target_hidden on invalid or missing cross data");
    ok &= expect(dflash_conversion.find("DFlashDraftModel: missing") != std::string::npos,
        "DFlash converter must warn when using default metadata");
    ok &= expect(dflash_conversion.find("DFlashDraftModel metadata:") != std::string::npos,
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
    ok &= expect(sampling_h.find("common_sampler_has_active_grammar") != std::string::npos, "common sampler must expose active grammar state separately from hard speculative stops");
    ok &= expect(sampling_h.find("common_sampler_reasoning_is_forcing") != std::string::npos, "common sampler must expose reasoning-forcing state as the hard DFlash draft stop");
    ok &= expect(sampling_h.find("common_sampler_stops_speculative_accept") != std::string::npos, "common sampler must expose per-cycle grammar transition stops for speculative accept");
    ok &= expect(sampling_cpp.find("Lazy grammars are safe to speculate while still awaiting their trigger") != std::string::npos, "speculative guard must keep DFlash available before lazy grammar triggers");
    ok &= expect(sampling_cpp.find("const bool grammar_active_at_start = common_sampler_has_active_grammar(gsmpl);") != std::string::npos,
        "speculative accept must snapshot whether grammar was already active at cycle start");
    ok &= expect(sampling_cpp.find("common_sampler_stops_speculative_accept(gsmpl, grammar_active_at_start)") != std::string::npos,
        "speculative accept must stop only when a token newly activates grammar or reasoning-forcing boundaries");
    ok &= expect(sampling_cpp.find("llama_sampler_apply(gsmpl->chain, &gsmpl->cur_p)") != std::string::npos, "reduced verifier must still run the sampler chain");
    ok &= expect(sampling_cpp.find("gsmpl->cur_p = { gsmpl->cur.data(), gsmpl->cur.size(), -1, false }") != std::string::npos, "reduced verifier sampler must tolerate unsorted GPU top-K candidates");
    ok &= expect(sampling_cpp.find("!common_sampler_reasoning_is_forcing(gsmpl)") != std::string::npos, "reduced verifier must allow passthrough reasoning-budget tracking");
    ok &= expect(sampling_cpp.find("llama_sampler_apply(gsmpl->rbudget, &gsmpl->cur_p)") != std::string::npos, "reduced verifier must preserve reasoning-budget sampler state");
    ok &= expect(sampling_cpp.find("common_reasoning_budget_next_forced_token(gsmpl->rbudget)") != std::string::npos,
        "reduced verifier must emit the deterministic forced reasoning-end token when EOG starts forcing");
    ok &= expect(server_context.find("dflash_select_reduced_verify_plan") != std::string::npos, "server must explicitly choose reduced verifier eligibility");
    ok &= expect(server_context.find("dflash_reduced_sampler_chain_supported") != std::string::npos &&
                 server_context.find("sampler-order") != std::string::npos,
        "server reduced verifier eligibility must reject sampler chains that can select outside compact candidates");
    ok &= expect(server_context.find("dflash_select_batch_reduced_verify_plan") != std::string::npos &&
                 server_context.find("slot.task->params.sampling") != std::string::npos,
        "server reduced verifier eligibility must use the active slot/request sampling params");
    ok &= expect(server_context.find("top-k-mismatch") != std::string::npos,
        "batched reduced verifier must reject mixed top-K requirements under one target graph");
    ok &= expect(server_context.find("\n        if (common_sampler_blocks_speculative(smpl.get())) {\n            return 0;\n        }\n\n        const int n_draft_min") == std::string::npos,
        "DFlash draft horizon must not globally drop to zero just because tool-call grammar is active");
    ok &= expect(server_context.find("common_sampler_reasoning_is_forcing(smpl.get())") != std::string::npos,
        "DFlash draft horizon must still hard-stop during deterministic reasoning-budget forcing");
    ok &= expect(server_context.find("dflash_active_grammar") != std::string::npos,
        "DFlash server path must track active grammar so flat drafting can continue while tree/rejection shortcuts stay disabled");
    ok &= expect(server_context.find("const bool grammar_active_for_accept = common_sampler_has_active_grammar(slot.smpl.get());") != std::string::npos,
        "DFlash flat accept must check active grammar at accept time");
    ok &= expect(server_context.find("&& !grammar_active_for_accept") != std::string::npos,
        "DFlash rejection sampling must stay disabled while grammar is active");
    ok &= expect(server_context.find("!dflash_active_grammar") != std::string::npos,
        "DFlash tree verification must stay disabled while grammar is active");
    ok &= expect(server_context.find("!sampling.grammar.empty() && !sampling.grammar_lazy") != std::string::npos,
        "reduced verifier planning must allow inactive lazy tool grammars before their trigger");
    ok &= expect(server_context.find("sampling.grammar_lazy || !sampling.grammar_triggers.empty()") == std::string::npos,
        "reduced verifier planning must not reject every lazy tool-call request before grammar activation");
    ok &= expect(server_context.find("common_sampler_blocks_speculative(smpl)") != std::string::npos, "DFlash rejection sampling must stop at newly activated grammar/reasoning boundaries");
    ok &= expect(server_context.find("speculative_flat_result_has_bonus") != std::string::npos, "server must distinguish grammar-boundary stops from bonus-token accepts");
    ok &= expect(server_context.find("n_hidden_keep = ids.empty() ? 0 : n_accepted_draft + 1") != std::string::npos, "DFlash ring/tape keep count must include root plus accepted draft tokens");
    ok &= expect(server_context.find("common_speculative_accept(slot.get_spec(), n_accepted_draft)") != std::string::npos, "speculative stats must count accepted draft tokens, not bonus-token-shaped results");
    ok &= expect(server_context.find("llama_dflash_rollback(ctx_tgt, slot.id, seq_backup, slot.n_pos_before_draft, n_hidden_keep)") != std::string::npos, "DFlash rollback must use the hidden-state keep count at grammar boundaries");
    ok &= expect(server_context.find("dflash_suppressed_for_reasoning_tool_marker") == std::string::npos,
        "server must not globally disable DFlash for the rest of a response after a raw tool marker; lazy grammar boundaries must be precise");
    ok &= expect(server_task.find("state.update_chat_msg(content, true, oaicompat_msg_diffs, true)") != std::string::npos, "streaming responses must filter partial tool-call deltas");
    ok &= expect(server_task.find("state.update_chat_msg(content, false, oaicompat_msg_diffs, true)") != std::string::npos ||
                 server_task_h.find("state.update_chat_msg(content, false, oaicompat_msg_diffs, true)") != std::string::npos,
        "final tool-parsing responses must filter malformed raw tool-call text too");
    ok &= expect(server_task.find("task_result_has_complete_partial_tool_calls") != std::string::npos, "streaming responses must allow complete tool-call deltas before final EOS");
    ok &= expect(server_task.find("task_result_filter_incomplete_partial_tool_calls") != std::string::npos, "streaming responses must expose stable tool-call headers without partial arguments");
    ok &= expect(server_task.find("A partial stream may expose the stable tool name/id for UX") != std::string::npos, "partial tool-call streaming must document the header-only reliability boundary");
    ok &= expect(server_task.find("task_result_quarantine_raw_tool_text") != std::string::npos, "streaming responses must quarantine malformed raw tool markers in tool-parsing mode");
    ok &= expect(server_task.find("task_result_pos_is_in_code_fence") != std::string::npos, "raw marker quarantine must avoid code fence content");
    ok &= expect(server_task.find("task_result_starts_with_raw_tool_marker") != std::string::npos, "streaming responses must suppress parser fallback text for wrapperless raw tool calls");
    ok &= expect(server_task.find("task_result_freeze_text_fields") != std::string::npos, "incomplete parsed tool calls must not leak fallback text/reasoning deltas");
    ok &= expect(server_context.find("raw tool marker observed while lazy grammar is enabled") != std::string::npos &&
                 server_context.find("suppressing DFlash for this response") == std::string::npos,
        "server may log raw tool markers, but must keep DFlash available outside active lazy-grammar boundaries");
    ok &= expect(server_context.find("server_tail_pos_is_in_code_fence") != std::string::npos, "DFlash raw-marker suppression must avoid fenced code content");
    ok &= expect(server_context.find("server_tail_tool_marker_has_boundary") != std::string::npos, "DFlash raw-marker suppression must avoid embedded string false positives");
    ok &= expect(chat_auto_parser_generator.find("allow_direct_func_start") != std::string::npos, "tag-style parsers must accept valid direct function starts without outer wrappers");
    ok &= expect(chat_auto_parser_generator.find("autoparser.tools.function.name_prefix") != std::string::npos, "lazy grammar triggers must include structural function markers");
    ok &= expect(server_context.find("sampling.has_logit_bias() || sampling.ignore_eos") != std::string::npos, "server must not treat inactive precomputed EOG biases as active logit bias");
    ok &= expect(server_context.find("finite-reasoning-budget") != std::string::npos, "server must disable reduced verifier only for finite reasoning budgets");
    ok &= expect(server_context.find("llama_set_dflash_verify_logits(ctx_tgt, dflash_verify_graph_enabled") != std::string::npos, "server must enable reduced verifier graph once per eligible batch");
    ok &= expect(server_context.find("llama_set_dflash_verify_logits(ctx_tgt, dflash_reduce_this_view") == std::string::npos, "server must not toggle reduced verifier graph around each ubatch");
    ok &= expect(server_context.find("llama_set_dflash_verify_logits(ctx_tgt, false, 1)") == std::string::npos, "server must not disable reduced verifier graph after each ubatch");
    ok &= expect(server_context.find("llama_set_dflash_consume_reduced(ctx_tgt, dflash_reduce_this_view)") != std::string::npos, "server must toggle only the raw-logit consumption flag per ubatch");
    ok &= expect(server_context.find("dflash_flat_effective_draft_max") != std::string::npos &&
                 server_context.find("block_size - 1") != std::string::npos,
        "flat DFlash verifier must cap draft max to the drafter's effective block_size-1 horizon");
    ok &= expect(server_context.find("n_draft_max = dflash_flat_effective_draft_max(ctx_dft_shared.get(), n_draft_max)") != std::string::npos,
        "server must avoid padding flat DFlash verifier batches to unreachable draft rows");
    ok &= expect(dflash_draft.find("skip_target_hidden_upload") != std::string::npos &&
                 dflash_draft.find("use_kv_cache && dflash_kv_cache_mode_env() == DFLASH_KV_CACHE_BOTH") != std::string::npos,
        "DFlash drafter must skip unused cross-hidden uploads when the full K/V projection cache is active");
    ok &= expect(metal.find("constant float turbo_centroids_4bit[16]") != std::string::npos &&
                 metal.find("constant float turbo_mid_4bit[15]") != std::string::npos &&
                 metal.find("turbo_find_nearest_4bit") != std::string::npos,
        "Metal Turbo4 must use the same pure 4-bit codebook as CUDA");
    ok &= expect(metal.find("QK_TURBO4 * 3 / 8") == std::string::npos &&
                 metal.find("dst.rnorm") == std::string::npos &&
                 metal.find("xb->rnorm") == std::string::npos &&
                 metal.find("xb->signs[j / 8]") == std::string::npos,
        "Metal Turbo4 must not use the removed 3-bit+QJL rnorm/signs layout");
    ok &= expect(metal.find("dst.qs[j / 2] = (idx1 << 4) | idx0") != std::string::npos &&
                 metal.find("turbo_centroids_4bit[idx] * norm") != std::string::npos,
        "Metal Turbo4 must pack two 4-bit indices per byte and dequantize through 4-bit centroids");
    ok &= expect(metal.find("kernel_set_rows_turbo<int64_t, block_turbo4_0") == std::string::npos &&
                 metal.find("kernel_set_rows_turbo<int32_t, block_turbo4_0") == std::string::npos,
        "Metal Turbo4 set_rows must not instantiate the generic signs-based Turbo3 kernel");
    ok &= expect(metal.find("kernel void kernel_set_rows_turbo4(") != std::string::npos &&
                 metal.find("typedef decltype(kernel_set_rows_turbo4<int64_t>) set_rows_turbo4_t") != std::string::npos,
        "Metal Turbo4 set_rows must use a dedicated pure 4-bit kernel");
    ok &= expect(speculative.find("if (common_dflash_debug_logs_enabled())") != std::string::npos &&
                 speculative.find("DFLASH_DBG append_target_hiddens") != std::string::npos,
        "DFlash append-target diagnostics must stay behind the debug logging gate");
    ok &= expect(speculative.find("if (common_dflash_debug_logs_enabled()) {\n            LOG_DBG(\"DFLASH_DBG build_cross_data") != std::string::npos,
        "DFlash build-cross diagnostics must stay behind the debug logging gate");
    ok &= expect(common_cpp.find("MTP-TRACE") == std::string::npos &&
                 speculative.find("MTP-TRACE") == std::string::npos &&
                 server_context.find("MTP-TRACE") == std::string::npos &&
                 delta_net_base.find("MTP-TRACE") == std::string::npos,
        "MTP issue-tracing logs must not stay in production paths");
    ok &= expect(sampling_cpp.find("MTP-ACCEPT-STOP") == std::string::npos &&
                 sampling_cpp.find("EOG-FORCE") == std::string::npos,
        "sampling issue-tracing logs must not stay in production paths");
    const std::string mtp_impl = slice_between(
        speculative,
        "struct common_speculative_impl_draft_mtp : public common_speculative_impl",
        "struct common_speculative_impl_ngram_simple : public common_speculative_impl");
    ok &= expect(!mtp_impl.empty(), "MTP speculative implementation must be present");
    ok &= expect(mtp_impl.find("verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd") != std::string::npos,
        "MTP must align pending hidden state from the last verify row in process() (upstream pattern)");
    ok &= expect(mtp_impl.find("truncate ctx_dft to the start of this batch") == std::string::npos &&
                 mtp_impl.find("Truncate stale draft positions") == std::string::npos,
        "MTP must not reintroduce blind draft-context truncation without hidden-state alignment");
    ok &= expect(speculative.find("void common_speculative_accept(common_speculative * spec, uint16_t n_accepted)") != std::string::npos &&
                 speculative.find("if (n_accepted == 0 || spec == nullptr)") == std::string::npos,
        "single-slot speculative accept must propagate accept(0), which MTP uses to advance pending target hidden state after rejection");
    ok &= expect(server_context.find("shrunk recurrent state to %d cells before draft load") != std::string::npos, "server must shrink recurrent backup cells before draft model load");
    ok &= expect(server_context.find("expanded recurrent state to %d cells before speculative GPU buffers") != std::string::npos, "server must expand recurrent backup cells before DFlash slot/GPU buffer init");
    ok &= expect(server_context.find("server_dflash_recurrent_rollback_plan speculative_recurrent_rollback_plan() const") != std::string::npos &&
                 server_context.find("params_base.speculative.dflash_selected_or_pending()") != std::string::npos &&
                 server_context.find("flat DFlash target architecture %s will use recurrent-only backup cells; keeping attention streams at n_parallel=%d") != std::string::npos &&
                 server_context.find("llama_memory_seq_rm_recurrent(mem, seq_backup, -1, -1)") != std::string::npos,
        "server must use recurrent-only backup cells for flat DFlash instead of RS snapshots or attention backup streams");
    ok &= expect(common_h.find("return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP;") != std::string::npos &&
                 common_h.find("t == COMMON_SPECULATIVE_TYPE_DFLASH && branch_budget == 0") == std::string::npos &&
                 common_cpp.find("const uint32_t n_rs_batch = cparams.n_rs_seq + 1") != std::string::npos,
        "MTP target contexts must enable bounded recurrent snapshots without applying that memory multiplier to flat DFlash");
    ok &= expect(contains_ws(server_context, "cparams.n_rs_seq = 0"),
        "MTP draft context creation must force n_rs_seq = 0 (upstream invariant: MTP heads have no delta-net layers)");
    ok &= expect(server_context.find("const bool slot_uses_fork_spec") != std::string::npos &&
                 server_context.find("server_speculative_uses_fork_slot_impls(params_base.speculative)") != std::string::npos,
        "server must keep fork-only per-slot speculative state separate from upstream MTP shared speculative state");
    {
        const std::string generic_horizon = slice_between(
                server_context,
                "int get_n_draft_max(",
                "int get_dflash_n_draft_max(");
        ok &= expect(!generic_horizon.empty() &&
                     generic_horizon.find("COMMON_SPECULATIVE_TYPE_DFLASH") == std::string::npos &&
                     generic_horizon.find("common_sampler_blocks_speculative") == std::string::npos &&
                     generic_horizon.find("dm_adaptive") == std::string::npos,
            "upstream MTP draft horizon must not inherit DFlash grammar or adaptive-depth policy");
    }
    ok &= expect(server_context.find("int get_dflash_n_draft_max(") != std::string::npos &&
                 server_context.find("common_sampler_reasoning_is_forcing") > server_context.find("int get_dflash_n_draft_max("),
        "DFlash draft-depth policy must live in an explicit DFlash helper");
    ok &= expect(server_context.find("common_context_can_seq_rm(ctx_dft.get())") != std::string::npos,
        "MTP draft context must probe seq_rm capability with common_context_can_seq_rm, not guess from n_rs_seq");
    ok &= expect(server_context.find("failed to create MTP draft-model context") != std::string::npos,
        "server must fail cleanly if a user-provided MTP draft-model context cannot be created");
    ok &= expect(server_context.find("common_context_seq_rm(ctx_dft.get(), slot.id, ckpt.pos_max + 1, -1)") != std::string::npos,
        "server must roll MTP draft context back using upstream checkpoint position before target verification");
    ok &= expect(server_context.find("const llama_pos draft_n_past = use_mtp_spec ? slot.prompt.n_tokens() : -1") != std::string::npos &&
                 server_context.find("common_speculative_get_draft_params(slot.get_spec(), slot.id)") != std::string::npos &&
                 server_context.find("/* .n_past   = */ draft_n_past") != std::string::npos,
        "server must seed upstream MTP drafting through the shared upstream dparams path using slot.prompt.n_tokens()");
    {
        const size_t mtp_draft_phase = server_context.find("// Phase 2: single draft call for all collected MTP slots");
        const size_t mtp_timer = server_context.find("const int64_t t_mtp_draft_start = ggml_time_us();", mtp_draft_phase);
        const size_t mtp_draft = server_context.find("common_speculative_draft(spec.get());", mtp_timer);
        const size_t mtp_account = server_context.find("t_draft_total += ggml_time_us() - t_mtp_draft_start;", mtp_draft);
        ok &= expect(mtp_draft_phase != std::string::npos &&
                     mtp_timer != std::string::npos &&
                     mtp_draft != std::string::npos &&
                     mtp_account != std::string::npos &&
                     mtp_timer < mtp_draft &&
                     mtp_draft < mtp_account,
            "server must account post-loop MTP draft time as draft_ms instead of hiding it in other_ms");
    }
    ok &= expect(server_context.find("common_speculative_accept(slot.get_spec(), slot.id, n_accepted_draft)") != std::string::npos,
        "server must accept upstream MTP drafts through the per-sequence shared speculative accept path");
    ok &= expect(server_context.find("slot.spec_ckpt.update_pos(") != std::string::npos,
        "server must initialize upstream-style checkpoint position before MTP draft");
    ok &= expect(server_context.find("slot.update_batch(batch);") != std::string::npos,
        "server must batch upstream MTP drafts through slot.update_batch instead of duplicating batch construction");
    ok &= expect(server_context.find("ckpt.load_tgt(slot.ctx_tgt, slot.id,") != std::string::npos,
        "server must load target checkpoint for FULL/RS rollback");
    ok &= expect(server_context.find("ckpt.load_dft(slot.ctx_dft, slot.id,") != std::string::npos,
        "server must load draft checkpoint for FULL rollback");
    ok &= expect(server_context.find("n_rollback > 0") != std::string::npos &&
                 server_context.find("slot.smpl_save") == std::string::npos,
        "server must keep sampler rollback local to the upstream MTP accept path, not as stale slot state");
    {
        const std::string mtp_accept = slice_between(
                server_context,
                "if (use_mtp_spec_accept) {",
                "llama_tokens ids;");
        ok &= expect(!mtp_accept.empty() &&
                     mtp_accept.find("dm_adaptive") == std::string::npos &&
                     mtp_accept.find("profit_pending") == std::string::npos,
            "upstream MTP accept path must stay free of DFlash adaptive-depth telemetry");
    }
    ok &= expect(server_context.find("raw tool marker observed while lazy grammar is enabled; keeping DFlash") == std::string::npos,
        "non-DFlash lazy grammar/tool marker path must not emit DFlash-specific policy text");
    ok &= expect(llama_h.find("llama_context_recurrent_expand") != std::string::npos, "public API must expose context-level recurrent expansion with graph invalidation");
    ok &= expect(server_context.find("llama_context_recurrent_shrink(ctx_tgt, n_parallel_user)") != std::string::npos, "server recurrent shrink must invalidate the context scheduler graph cache");
    ok &= expect(server_context.find("llama_context_recurrent_expand(ctx_tgt, n_seq_max_full)") != std::string::npos, "server recurrent expansion must invalidate the context scheduler graph cache");
    ok &= expect(server_context.find("bool recurrent_shrink_for_prompt_cache(const char * reason)") != std::string::npos,
        "server must shrink expanded recurrent backup cells around prompt-cache restore");
    ok &= expect(server_context.find("recurrent_shrink_for_prompt_cache(\"before prompt cache save/load\")") != std::string::npos &&
                 server_context.find("if (update_cache)") < server_context.find("recurrent_shrink_for_prompt_cache(\"before prompt cache save/load\")"),
        "server must only shrink recurrent state for an actual prompt-cache save/load");
    ok &= expect(server_context.find("void recurrent_expand_after_prompt_cache(const char * reason)") != std::string::npos &&
                 server_context.find("recurrent_expand_after_prompt_cache(\"after prompt cache save/load\")") != std::string::npos,
        "server must expand recurrent backup cells again before normal prefill graph reservation");
    ok &= expect(server_context.find("llama_seq_id seq_id_backup = -1") != std::string::npos &&
                 server_context.find("slot.seq_id_backup = seq_backup") != std::string::npos &&
                 server_context.find("const llama_seq_id seq_backup = slot.seq_id_backup") != std::string::npos,
        "server speculative backup cleanup must track the actual backup seq_id");
    ok &= expect(server_context.find("if (has_draft_backup && seq_id_backup >= 0)") != std::string::npos,
        "server slot release must clean up leaked speculative backup sequences");
    ok &= expect(server_context.find("uses_shared_kv_pool() && task.n_tokens() > 0") != std::string::npos &&
                 server_context.find("cells_committed += std::max((int64_t) s.prompt.n_tokens(), (int64_t) s.task->n_tokens())") != std::string::npos &&
                 server_context.find("cells_available < (int64_t) task.n_tokens()") != std::string::npos,
        "unified KV scheduling must account for active slot cell commitments before launch");
    ok &= expect(server_context.find("task.n_tokens() < slot->n_ctx") != std::string::npos,
        "unified KV deferral must let oversized tasks reach existing context-size validation instead of starving");
    ok &= expect(context_cpp.find("bool llama_context::resize_recurrent_memory") != std::string::npos, "context recurrent resize must be implemented at context level");
    ok &= expect(context_cpp.find("sched_need_reserve = true") != std::string::npos, "context recurrent resize must reserve a fresh scheduler graph after tensor reallocation");
    ok &= expect(server_context.find("dflash_profit_controller") == std::string::npos, "DFlash profit controller must use the normal adaptive depth probe path");
    ok &= expect(server_context.find("dflash_fixed_verify_shape") == std::string::npos, "DFlash profit controller must not override adaptive depth decisions");
    ok &= expect(server_context.find("continuing would corrupt recurrent replay") != std::string::npos, "server must abort instead of continuing after recurrent backup expansion failure");
    ok &= expect(graph_h.find("cparams.cb_eval              == other.cparams.cb_eval") == std::string::npos,
        "graph reuse must not key on eval callback runtime state");
    ok &= expect(graph_h.find("cparams.cb_eval_user_data    == other.cparams.cb_eval_user_data") == std::string::npos,
        "graph reuse must not key on eval callback user data runtime state");
    ok &= expect(graph_h.find("cparams.hidden_gpu_n_seqs") != std::string::npos, "graph reuse must key on GPU hidden capture topology");
    ok &= expect(graph_h.find("cparams.tape_gpu             == other.cparams.tape_gpu") != std::string::npos, "graph reuse must key on GPU tape topology");
    ok &= expect(context_cpp.find("raw_logits_needed") != std::string::npos &&
                 context_cpp.find("needs_raw_logits(ubatch, sampling.samplers)") != std::string::npos &&
                 context_cpp.find("raw_logits_needed && !dflash_reduced_consumed") != std::string::npos,
        "raw logits must still be copied for fallback views when stable top-K is present");
    ok &= expect(server_context.find("dflash profile: accept n_draft=%zu ids=%zu") != std::string::npos, "server must profile DFlash accept subphases");
    ok &= expect(server_context.find("dflash_sample_reduced_verify") != std::string::npos, "server must consume reduced verifier logits");
    ok &= expect(server_context.find("falling back is unsafe because raw logits were not copied") != std::string::npos, "server must not silently fall back after reduced raw-logit skip");
    ok &= expect(server_context.find("spec_pad_i_batch") != std::string::npos, "server must track DFlash verifier padding rows separately from real draft rows");
    ok &= expect(server_context.find("GGML_DFLASH_VERIFY_PAD") != std::string::npos &&
                 server_context.find("dflash_verify_padding_enabled()") != std::string::npos,
        "server must keep DFlash verifier padding behind an explicit diagnostic env toggle");
    ok &= expect(server_context.find("padded DFlash verifier batch") != std::string::npos, "server must still log diagnostic DFlash verifier padding when explicitly enabled");
    ok &= expect(server_context.find("active_verify_draft_max") != std::string::npos, "server diagnostic padding must only pad to the active adaptive draft depth");
    ok &= expect(server_context.find("std::max(n_draft_max, original_n_max)") == std::string::npos, "DFlash verifier padding must not force adaptive depth to pay for configured max depth");
    ok &= expect(server_context.find("rows_available") != std::string::npos, "server diagnostic DFlash verifier padding must respect batch and ubatch capacity");
    ok &= expect(server_context.find("for (int idx : slot.spec_pad_i_batch)") != std::string::npos, "reduced verifier coverage must account for explicit padding rows");
    ok &= expect(server_context.find("const bool had_dflash_padding = !slot.spec_pad_i_batch.empty()") != std::string::npos, "server must remember verifier padding through accept bookkeeping");
    ok &= expect(server_context.find("const bool all_accepted_flat = (n_accepted_draft == (int) n_draft) && !had_dflash_padding") != std::string::npos, "DFlash verifier padding must force rollback even when all real draft tokens were accepted");

    // DFlashDraftModel converter vocab path checks
    ok &= expect(dflash_conversion.find("_is_gemma4_dflash") != std::string::npos, "DFlash converter must have Gemma4 detection helper");
    ok &= expect(dflash_conversion.find("_set_vocab_gemma4_hf_bpe") != std::string::npos, "DFlash converter must have Gemma4 HF/BPE vocab helper");
    ok &= expect(dflash_conversion.find("visible_tokens = {") != std::string::npos, "DFlash Gemma4 vocab helper must define visible tokens set");
    ok &= expect(dflash_conversion.find("errors=\"replace\"") != std::string::npos, "DFlash Gemma4 vocab helper must decode tokens with errors=replace");
    ok &= expect(dflash_conversion.find("self._set_vocab_gpt2()") != std::string::npos, "DFlash converter must fall back to GPT-2 vocab for non-Gemma drafters");

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

    {
        const size_t fn = speculative.find("void set_prefill_capture_enabled(bool enabled) override");
        const size_t early_return = fn == std::string::npos ? std::string::npos :
            speculative.find("if (target_capture_enabled == enabled)", fn);
        const size_t active_call = fn == std::string::npos ? std::string::npos :
            speculative.find("llama_set_dflash_capture_active(ctx_tgt", fn);
        ok &= expect(fn != std::string::npos &&
                     active_call != std::string::npos &&
                     (early_return == std::string::npos || early_return > active_call),
            "DFlash prefill capture must re-apply shared target capture state even when the per-slot cache already matches");
    }

    ok &= expect(server_context.find("dflash_capture_needed_for_view") != std::string::npos,
        "server must decide whether DFlash hidden capture is needed before llama_decode");

    ok &= expect(server_context.find("common_speculative_set_prefill_capture_enabled(slot.get_spec(), dflash_capture_needed_for_view)") != std::string::npos,
        "server must toggle DFlash hidden capture before target decode");
    ok &= expect(server_context.find("mixed prompt/generation prefill view") != std::string::npos &&
                 server_context.find("common_speculative_discard_dflash_state(slot.get_spec(), \"mixed prompt/generation prefill view\")") != std::string::npos,
        "server must fail closed for prompt DFlash prefill when the same view contains generation");
    ok &= expect(server_context.find("dflash_mark_skip_begin(slot.id)") != std::string::npos &&
                 server_context.find("dflash_should_skip_begin(slot.id)") != std::string::npos,
        "server must not rebuild DFlash state from the same unsafe mixed prefill view");

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
    const size_t flush_call = server_context.find("common_speculative_flush_prefill(");
    ok &= expect(flush_call != std::string::npos &&
                 server_context.find("pf.span.src_offset", flush_call) != std::string::npos &&
                 server_context.find("pf.span.n_tokens", flush_call) != std::string::npos &&
                 server_context.find("pf.span.capture_end", flush_call) != std::string::npos,
        "server must pass planned source offset and absolute capture end to prefill flush");
    ok &= expect(speculative_h.find("common_dflash_prefill_committed_after_flush") != std::string::npos &&
                 speculative.find("common_dflash_prefill_committed_after_flush(") != std::string::npos,
        "DFlash prefill flush must preserve absolute committed positions for long prompts");

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

    // Mismatch in server must discard DFlash state and disable capture for the affected slot
    ok &= expect(speculative_h.find("common_speculative_discard_dflash_state") != std::string::npos,
        "DFlash must expose explicit state discard for unsafe flush/capture paths");
    ok &= expect(speculative.find("void discard_cross_ring(const char * reason)") != std::string::npos &&
                 speculative.find("ring_write_pos = 0") != std::string::npos &&
                 speculative.find("committed_len = 0") != std::string::npos &&
                 speculative.find("llama_dflash_kv_cache_reset(ctx_dft)") != std::string::npos,
        "DFlash state discard must clear ring counters and K/V projection cache");
    ok &= expect(speculative.find("discard_cross_ring(\"GPU hidden D2D ring write failed\")") != std::string::npos,
        "DFlash GPU D2D write failure must discard cross-ring state");
    ok &= expect(speculative.find("ring_write_discarded") != std::string::npos &&
                 count_occurrences(speculative, "if (ring_write_discarded)") >= 2,
        "DFlash ring-write callers must not re-commit state after an internal discard");
    ok &= expect(speculative.find("actual_written != to_write") != std::string::npos &&
                 speculative.find("discard_cross_ring(\"incomplete prefill flush\")") != std::string::npos,
        "incomplete DFlash prefill flush must discard cross-ring state");
    ok &= expect(server_context.find("common_speculative_discard_dflash_state(pf.spec, \"prefill flush mismatch\")") != std::string::npos &&
                 server_context.find("dflash_mark_skip_begin(pf.slot_id)") != std::string::npos &&
                 server_context.find("common_speculative_set_prefill_capture_enabled(pf.spec, false)") != std::string::npos,
        "prefill flush mismatch must discard DFlash state, skip begin, and disable capture for the affected slot");

    ok &= expect(speculative_h.find("common_speculative_note_prefill_suffix_scheduled") != std::string::npos,
        "DFlash must expose a separate prefill-suffix scheduled marker");
    ok &= expect(speculative.find("note_prefill_suffix_scheduled") != std::string::npos,
        "DFlash state must implement a separate prefill-suffix scheduled marker");
    ok &= expect(server_context.find("common_speculative_note_prefill_suffix_scheduled(slot.get_spec())") != std::string::npos,
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

    // ring_state_save must not serialize stale CPU ring data; GPU-only rings
    // must be checkpointed through a D2H snapshot instead.
    ok &= expect(speculative.find("llama_dflash_cross_ring_gpu_snapshot") != std::string::npos &&
                 speculative.find("compact_gpu_snapshot") != std::string::npos,
        "GPU-only DFlash ring checkpoints must snapshot the GPU ring instead of saving stale CPU data");
    ok &= expect(llama_h.find("llama_dflash_cross_ring_gpu_snapshot") != std::string::npos &&
                 context_cpp.find("fn_snapshot") != std::string::npos &&
                 cuda_ring.find("dflash_cross_ring_gpu_snapshot") != std::string::npos &&
                 cuda_reg.find("\"dflash_cross_ring_gpu_snapshot\"") != std::string::npos,
        "GPU ring snapshot API must be wired through llama.h, context, CUDA ring, and CUDA proc lookup");

    ok &= expect(speculative_h.find("bool   common_speculative_ring_state_save") != std::string::npos &&
                 server_context.find("ring_saved = common_speculative_ring_state_save") != std::string::npos,
        "DFlash checkpoint save must report snapshot failure instead of logging non-empty bytes as saved");

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
    ok &= expect(context_cpp.find("memory->set_force_split_seq(false)") != std::string::npos,
        "DFlash capture disable must clear force_split_seq so adaptive no-spec fallback can batch like normal target decoding");
    ok &= expect(context_cpp.find("dflash_mixed_capture_supported") != std::string::npos &&
                 context_cpp.find("dflash mixed capture suppressed") != std::string::npos,
        "decode must suppress DFlash capture for mixed ubatches that include non-DFlash slots");
    ok &= expect(server_context.find("dflash_tg_batch_all_spec_slots") != std::string::npos &&
                 server_context.find("dflash multiseq target batching disabled") != std::string::npos,
        "server must not multi-seq batch pure TG DFlash views unless every sequence owns DFlash state");
    ok &= expect(server_context.find("dflash_multiseq_rows") != std::string::npos &&
                 server_context.find("uneven-rows") != std::string::npos,
        "server must not multi-seq batch DFlash target views with uneven per-slot rows because GPU hidden capture is per-ubatch");
    ok &= expect(server_context.find("dflash_batch_has_tg") != std::string::npos &&
                 server_context.find("!dflash_batch_has_tg && (params_base.cont_batching || batch.n_tokens == 0)") != std::string::npos,
        "server must defer prompt batching when the current DFlash batch already contains TG verifier rows");
    ok &= expect(server_context.find("dflash_try_reserve_tg_rows") != std::string::npos &&
                 server_context.find("dflash_skip_tg_slot_for_next_cycle") != std::string::npos &&
                 server_context.find("dflash uneven TG rows deferred") != std::string::npos,
        "server must defer uneven DFlash TG slots instead of decoding a combined view with partial hidden capture");
    ok &= expect(server_context.find("dflash_multi_seq_tg_rows_safe") != std::string::npos &&
                 server_context.find("rows == 1") != std::string::npos,
        "server must only multi-seq batch DFlash target verification when each slot contributes one row");
    ok &= expect(server_context.find("dflash_next_tg_slot") != std::string::npos &&
                 server_context.find("dflash_tg_slot_start") != std::string::npos,
        "server must fairly rotate one-slot DFlash multi-row verifier cycles instead of always starting at slot 0");
    ok &= expect(server_context.find("dflash_has_pending_prompt") != std::string::npos &&
                 server_context.find("dflash_has_pending_prompt && slot.can_speculate()") != std::string::npos &&
                 server_context.find("params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&\n            needs_reeval &&\n            std::any_of(slots.begin(), slots.end(), [](const server_slot & slot) {\n                return slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT;\n            })") == std::string::npos,
        "DFlash must prioritize unfinished prompt prefill before TG rows on all target types, not only recurrent/hybrid targets");
    ok &= expect(server_context.find("dflash_has_profit_baseline_slot") != std::string::npos &&
                 server_context.find("dflash_force_profit_baseline_cycle") != std::string::npos &&
                 server_context.find("dflash_force_profit_baseline_cycle && n_draft_max > 0") != std::string::npos,
        "DFlash adaptive profit baseline slots must get one-row target cycles instead of starving behind multi-row verifier batches");
    ok &= expect(common_h.find("int32_t dflash_max_slots = 0") != std::string::npos &&
                 server_context.find("params_base.speculative.dflash_max_slots > 0") != std::string::npos &&
                 server_context.find("DFlash enabled for all %d slots") != std::string::npos,
        "DFlash must default to all -np slots and use --spec-dflash-max-slots only as an explicit cap");
    ok &= expect(speculative.find("llama_n_ctx_seq(ctx_dft)") != std::string::npos &&
                 server_context.find("dflash_draft_ctx_per_slot * dflash_draft_slots_clamped") != std::string::npos &&
                 server_context.find("params_dft.kv_unified = false") != std::string::npos,
        "DFlash drafter KV retention must use partitioned per-slot context, not a unified shared pool");
    ok &= expect(server_context.find("params_dft.kvarn") != std::string::npos &&
                 server_context.find("llama_kvarn_default_params();") != std::string::npos,
        "DFlash draft contexts must not inherit target-only KVarN settings");
    ok &= expect(server_context.find("reduced-row-count-mismatch") != std::string::npos,
        "DFlash reduced verifier must reject uneven multi-slot rows so compact output is not overwritten by internal ubatch splits");
    ok &= expect(server_context.find("dflash_shared_drafter_batch_allowed") != std::string::npos &&
                 server_context.find("const bool target_non_recurrent = !needs_reeval;") == std::string::npos &&
                 server_context.find("DFlash shared drafter batching disabled: reason=target-recurrent") == std::string::npos,
        "DFlash shared drafter batching must stay enabled for recurrent/hybrid targets once cross-slot acceptance is rollback-isolated");
    ok &= expect(server_context.find("target-recurrent-tree") != std::string::npos &&
                 server_context.find("needs_reeval && dflash_multiseq_n_unique > 1 && ddtree_batch_active") != std::string::npos,
        "DFlash recurrent multi-seq verification must still fall back to per-slot mode for tree batches, but stay enabled for flat multi-slot DFlash");
    ok &= expect(server_context.find("dflash_multislot_flat_accept_barrier") != std::string::npos &&
                 server_context.find("struct dflash_flat_accept_prefetch") != std::string::npos &&
                 server_context.find("common_speculative_update_logits_deferred_dflash_kv(slot.get_spec(), ctx_tgt, batch_tokens, prefetched.n_hidden_keep);") != std::string::npos &&
                 server_context.find("dflash_has_pending_prompt") != std::string::npos &&
                 server_context.find("dflash_recurrent_draft_rr") == std::string::npos &&
                 server_context.find("dflash_recurrent_cycle_slot_id") == std::string::npos,
        "DFlash recurrent/hybrid multi-slot scheduling must isolate prompt prefill, pre-sample every flat speculative slot before rollback, and remove the old one-slot recurrent round-robin");
    ok &= expect(server_context.find("profit_acceptance_collapse_disabled") == std::string::npos &&
                 server_context.find("dflash_slot_runtime_enabled") == std::string::npos &&
                 server_context.find("dflash_request_disabled") == std::string::npos &&
                 server_adaptive_dm_h.find("profit_acceptance_collapse_disabled") == std::string::npos &&
                 server_adaptive_dm_h.find("disabling DFlash for request after sustained low acceptance") == std::string::npos,
        "DFlash adaptive DM must not hard-disable request-local DFlash capture/update state from low acceptance alone");
    {
        const size_t multi_profit = server_context.find("dflash_profit_shared_scale");
        const size_t observe = server_context.find("slot.observe_profit_timing", multi_profit);
        const size_t decide = server_context.find("apply_profit_decision(slot)", observe);
        ok &= expect(multi_profit != std::string::npos &&
                     server_context.find("n_slots_drafted > 1", multi_profit) < observe &&
                     observe != std::string::npos &&
                     decide != std::string::npos &&
                     observe < decide,
            "DFlash profit telemetry must observe multi-slot cycles instead of clearing pending samples when -np > 1");
    }
    ok &= expect(context_cpp.find("const size_t total_ids = (size_t) K * (size_t) n_outputs_all") != std::string::npos &&
                 context_cpp.find("const size_t offset_ids = (size_t) K * (size_t) n_outputs_prev") != std::string::npos &&
                 context_cpp.find("logits_argmax_count = (int32_t) (n_outputs_prev + n_outputs)") != std::string::npos,
        "DFlash compact reduced-verifier logits must accumulate across split target ubatches instead of keeping only the last split");
    ok &= expect(speculative.find("actual_written != n_accepted") != std::string::npos &&
                 speculative.find("incomplete target hidden capture") != std::string::npos &&
                 speculative.find("refusing to advance DFlash ring") != std::string::npos,
        "DFlash must discard partial hidden captures instead of advancing the cross-ring with missing accepted rows");
    ok &= expect(speculative_h.find("common_speculative_update_logits_deferred_dflash_kv") != std::string::npos &&
                 speculative.find("deferred_drafter_kv_tokens") != std::string::npos &&
                 speculative.find("flush_deferred_drafter_kv_cache(\"flat draft\")") != std::string::npos &&
                 speculative.find("flush_deferred_drafter_kv_cache(\"tree draft\")") != std::string::npos &&
                 speculative.find("flush_deferred_drafter_kv_cache(\"batched draft\")") != std::string::npos,
        "DFlash adaptive-off fallback must defer drafter KV maintenance but flush it before every probe/draft path");
    ok &= expect(server_context.find("dflash_defer_kv_cache_update") != std::string::npos &&
                 server_context.find("slot.dm_adaptive && slot.adaptive_n_max == 0") != std::string::npos &&
                 server_context.find("common_speculative_update_logits_deferred_dflash_kv(slot.get_spec(), ctx_tgt, batch_tokens, 1)") != std::string::npos,
        "server adaptive-off DFlash fallback must keep the target ring current while skipping per-token drafter KV updates");
    ok &= expect(server_context.find("common_speculative_update_logits_deferred_dflash_kv(slot.get_spec(), ctx_tgt, batch_tokens, prefetched.n_hidden_keep);") != std::string::npos &&
                 server_context.find("common_speculative_update_logits_deferred_dflash_kv(slot.get_spec(), ctx_tgt, batch_tokens, n_hidden_keep);") != std::string::npos,
        "DFlash flat speculative accept must defer drafter KV maintenance until the next draft while still updating the target hidden ring");
    {
        const size_t sync_comment = server_context.find("DFlash: a previous async rollback replay");
        const size_t sync_call    = server_context.find("llama_tape_replay_sync(ctx_tgt);", sync_comment);
        const size_t target_decode = server_context.find("const int ret = llama_decode(ctx_tgt, batch_view);");
        ok &= expect(sync_comment != std::string::npos &&
                     sync_call    != std::string::npos &&
                     target_decode != std::string::npos &&
                     sync_comment < sync_call &&
                     sync_call < target_decode,
            "DFlash must synchronize pending recurrent tape replay immediately before every target decode, including non-spec fallback after a rejected draft");
    }
    {
        const size_t rollback_call = server_context.find("llama_dflash_rollback(ctx_tgt, slot.id, seq_backup, slot.n_pos_before_draft, n_hidden_keep);");
        const size_t rollback_guard = server_context.find("if (n_slots_drafted > 1)", rollback_call);
        const size_t rollback_sync = server_context.find("llama_tape_replay_sync(ctx_tgt);", rollback_guard);
        const size_t next_accept_lap = server_context.find("profile_accept_lap(profile_accept_rollback_us);", rollback_call);
        ok &= expect(rollback_call != std::string::npos &&
                     rollback_guard != std::string::npos &&
                     rollback_sync != std::string::npos &&
                     next_accept_lap != std::string::npos &&
                     rollback_call < rollback_sync &&
                     rollback_call < rollback_guard &&
                     rollback_guard < rollback_sync &&
                     rollback_sync < next_accept_lap,
            "DFlash recurrent accept must defer single-slot flat rollback replay sync while still synchronizing multi-slot accept before the next slot mutates recurrent state");
    }
    {
        const size_t draft_added = server_context.find("slot.n_draft_total += slot.spec_draft.size();");
        const size_t draft_sync = server_context.find("llama_tape_replay_sync(ctx_tgt);", draft_added);
        const size_t backup_call = server_context.find("dflash_backup_recurrent_state(slot.id, seq_backup);", draft_sync);
        ok &= expect(draft_added != std::string::npos &&
                     draft_sync != std::string::npos &&
                     backup_call != std::string::npos &&
                     draft_added < draft_sync &&
                     draft_sync < backup_call,
            "DFlash flat draft setup must synchronize any deferred rollback replay before recurrent backup");
    }

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

    // Large prefill cannot fallback to incomplete verify capture, but a complete
    // callback capture is valid and should keep the DFlash ring alive.
    ok &= expect(common_dflash_should_refuse_large_prefill_fallback_for_test(764, 16, false, true),
        "large incomplete CPU-only prefill fallback with GPU ring must be refused");
    ok &= expect(!common_dflash_should_refuse_large_prefill_fallback_for_test(764, 764, false, true),
        "large complete CPU fallback must be allowed when GPU prefill staging is unavailable");
    ok &= expect(!common_dflash_should_refuse_large_prefill_fallback_for_test(4, 4, false, true),
        "small verify-sized CPU fallback must be allowed");
    ok &= expect(!common_dflash_should_refuse_large_prefill_fallback_for_test(764, 0, true, true),
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
    ok &= expect(common_dflash_invalid_reduced_logits_next_streak_for_test(0, false) == 1 &&
                 common_dflash_invalid_reduced_logits_next_streak_for_test(3, false) == 4 &&
                 common_dflash_invalid_reduced_logits_next_streak_for_test(3, true) == 0,
        "invalid reduced-logits streak must count consecutive failures and reset on a valid draft");
    ok &= expect(!common_dflash_invalid_reduced_logits_fail_closed_for_test(3, 4) &&
                 common_dflash_invalid_reduced_logits_fail_closed_for_test(4, 4),
        "invalid reduced-logits streak must fail closed once the threshold is reached");
    ok &= expect(speculative.find("invalid_reduced_logits_streak") != std::string::npos &&
                 speculative.find("fail-closed") != std::string::npos &&
                 speculative.find("invalid reduced-logits token") != std::string::npos &&
                 speculative.find("ring_filled") != std::string::npos,
        "DFlash invalid reduced-logits logs must carry enough ring context and then fail closed");

    // DFlash draft depth must have one source of truth: server adaptive DM.
    ok &= expect(arg_cpp.find("--spec-dflash-fixed-depth") == std::string::npos &&
                 arg_cpp.find("--spec-dflash-disable-accept-shrink") == std::string::npos &&
                 common_h.find("dflash_fixed_depth") == std::string::npos &&
                 common_h.find("dflash_disable_accept_shrink") == std::string::npos,
        "DFlash must not expose extra fixed-depth or accept-shrink controls");
    ok &= expect(speculative.find("adaptive_n_draft") == std::string::npos &&
                 speculative.find("n_low_accept") == std::string::npos &&
                 speculative.find("GGML_DFLASH_DISABLE_ACCEPT_SHRINK") == std::string::npos &&
                 server_context.find("int n_draft_max = (dm_adaptive && adaptive_n_max >= 0) ? std::min((int) adaptive_n_max, base_n_max) : base_n_max") != std::string::npos &&
                 server_context.find("else if (dm_adaptive && adaptive_n_max == 0)") != std::string::npos,
        "DFlash-local accept shrink must be removed and non-adaptive mode must ignore adaptive state");
    ok &= expect(server_context.find("server_adaptive_dm_should_preserve_for_continuation(sim_best, f_keep)") != std::string::npos &&
                 server_context.find("adaptive dm: preserving state for continuation") != std::string::npos &&
                 server_context.find("adaptive dm: reset state for prompt change") != std::string::npos,
        "adaptive DFlash state must reset on prompt changes while preserving true continuations");
    ok &= expect(server_context.find("adaptive dm: reset state for LRU slot selection") != std::string::npos &&
                 server_context.find("adaptive dm: reset state for canceled task") != std::string::npos,
        "adaptive DFlash request state must reset on fresh LRU slots and canceled tasks");
    ok &= expect(server_context.find("slot.reset_profit_if_config_changed(task.params.speculative, base_n_max") != std::string::npos &&
                 server_context.find("&task.params.sampling") != std::string::npos,
        "adaptive DFlash profit controller must refresh its config key with task sampling when a task is launched");
    ok &= expect(server_context.find("dflash_effective_adaptive_base_n_max") != std::string::npos &&
                 server_context.find("const int base_n_max = dflash_effective_adaptive_base_n_max(") != std::string::npos,
        "adaptive DFlash profit decisions must use the effective flat DFlash draft cap, not only the configured nominal max");
    ok &= expect(context_h.find("profile_reduced_logits_ids_us") != std::string::npos &&
                 context_cpp.find("GGML_DFLASH_PROFILE_SYNC_SPLIT") != std::string::npos &&
                 context_cpp.find("reduced_logits_ids=") != std::string::npos &&
                 context_cpp.find("verify_sync_split=") != std::string::npos,
        "DFlash verifier profiling must split decode sync and compact logits copies");
    ok &= expect(cuda_argmax.find("GGML_CUDA_DFLASH_CUB_TOP_K") != std::string::npos &&
                 cuda_argmax.find("GGML_DFLASH_ARGMAX_PROFILE") != std::string::npos &&
                 cuda_argmax.find("path=%s") != std::string::npos,
        "CUDA top-k verifier path must be selectable and loggable");
    ok &= expect(server_context.find("dflash_target_reduced_verify_supported") != std::string::npos &&
                 server_context.find("ggml_backend_reg_name") != std::string::npos &&
                 server_context.find("\"MTL\"") != std::string::npos &&
                 server_context.find("metal-argmax-ext") != std::string::npos,
        "DFlash compact verifier must be disabled on Metal until Metal ARGMAX honors extended top-k/log-prob params");
    ok &= expect(server_context.find("dflash acceptance histogram") != std::string::npos &&
                 server_context.find("dflash_accept_hist_by_ctx") != std::string::npos &&
                 server_context.find("note_dflash_cycle") != std::string::npos &&
                 server_context.find("dflash_cycle_count > 0 && dflash_server_profile_enabled(DFLASH_PROFILE_SUMMARY)") != std::string::npos,
        "DFlash cycle logs must include profile-gated acceptance histograms by context bucket");
    ok &= expect(count_occurrences(server_context, "SRV_INF(\"spec cycle") == 1 &&
                 server_context.find("if (dflash_server_profile_enabled(DFLASH_PROFILE_VERIFY)) {\n                SRV_INF(\"  verify ubatch:") != std::string::npos,
        "DFlash per-cycle and verify-ubatch timing logs must be opt-in profile logs");
    ok &= expect(speculative.find("target capture layers mix SWA and FULL") != std::string::npos &&
                 speculative.find("target/draft SWA window mismatch") != std::string::npos &&
                 speculative.find("common_dflash_capture_neighborhood_to_string") != std::string::npos &&
                 speculative.find("common_dflash_log_contract_verbose()") != std::string::npos,
        "DFlash startup must warn about Gemma SWA/FULL capture-contract risks while gating verbose contract detail");
    ok &= expect(speculative.find("dflash checkpoint: ring restore") != std::string::npos &&
                 speculative.find("DFLASH_DBG ring_state_load") == std::string::npos,
        "DFlash checkpoint restore detail must be profile-gated and must not use always-on debug warnings");
    ok &= expect(server_context.find("server_model_is_dflash_drafter(model_dft.get())") != std::string::npos &&
                 server_context.find("llama_model_dflash_n_target_layers(model)") != std::string::npos &&
                 server_context.find("llama_model_dflash_n_target_features(model)") != std::string::npos,
        "server DFlash auto-detection must require complete DFlash metadata, not the default block size");
    ok &= expect(server_context.find("const bool draft_devices_explicit = !params_spec.devices.empty();") != std::string::npos &&
                 server_context.find("#include \"src/llama-ext.h\"") != std::string::npos &&
                 server_context.find("params_dft.split_mode = LLAMA_SPLIT_MODE_NONE;") != std::string::npos &&
                 server_context.find("llama_model_n_devices(model_dft.get()) > 1") != std::string::npos &&
                 server_context.find("SRV_INF(\"%s\", \"DFlash draft model will use a single device by default") != std::string::npos &&
                 server_context.find("SRV_INF(\"%s\", \"reloading auto-detected DFlash draft model on a single device") != std::string::npos,
        "DFlash draft model loading must default to one device unless --spec-draft-device is explicit");
    ok &= expect(llama_ext_h.find("llama_model_dev_output") != std::string::npos &&
                 model_cpp.find("llama_model_dev_output") != std::string::npos &&
                 server_context.find("ggml_backend_dev_t target_output_dev = llama_model_dev_output(model_tgt);") != std::string::npos &&
                 server_context.find("params_dft.devices = { target_output_dev, nullptr };") != std::string::npos &&
                 server_context.find("params_dft.main_gpu = 0;") != std::string::npos &&
                 server_context.find("const bool dflash_auto_device_mismatch") != std::string::npos &&
                 server_context.find("const bool dflash_auto_single_gpu_reload") != std::string::npos &&
                 server_context.find("target_output_is_gpu &&") != std::string::npos &&
                 server_context.find("dflash_auto_device_mismatch") != std::string::npos &&
                 server_context.find("DFlash draft model uses shared target output tensor on device") != std::string::npos,
        "DFlash draft auto-placement must include the target output device used by shared tensors");
    ok &= expect(server_context.find("server_backend_dev_is_dflash_shared_output_compatible") != std::string::npos &&
                 server_context.find("GGML_BACKEND_DEVICE_TYPE_META") != std::string::npos &&
                 server_context.find("const bool target_output_is_meta") != std::string::npos &&
                 server_context.find("DFlash draft model will use tensor-split placement with target Meta output") != std::string::npos &&
                 server_context.find("params_dft.split_mode = LLAMA_SPLIT_MODE_TENSOR;") != std::string::npos &&
                 server_context.find("params_dft.devices = params_base.devices;") != std::string::npos &&
                 server_context.find("server_model_supports_device_buffer(model_dft.get(), target_output_dev)") != std::string::npos &&
                 server_context.find("target_output_is_meta &&\n                    !server_model_supports_device_buffer(model_dft.get(), target_output_dev)") != std::string::npos &&
                 server_context.find("DFlash disabled for tensor-split target") == std::string::npos &&
                 server_context.find("GGML_DFLASH_ALLOW_TENSOR_SPLIT") == std::string::npos &&
                 server_context.find("DFlash on tensor-split target is EXPERIMENTAL") == std::string::npos &&
                 server_context.find("draft_output_is_meta ||") == std::string::npos,
        "DFlash draft auto-placement must keep a tensor-split drafter when shared target tensors live in a Meta buffer");
    ok &= expect(llama_h.find("ggml_backend_dev_t output_device;") != std::string::npos &&
                 common_h.find("ggml_backend_dev_t output_device = nullptr;") != std::string::npos &&
                 common_cpp.find("mparams.output_device = params.output_device;") != std::string::npos &&
                 contains_ws(model_cpp, "/*.output_device =*/ nullptr") &&
                 model_cpp.find("pimpl->dev_output = params.output_device") != std::string::npos,
        "model loading must allow DFlash to pin the shared output tensor before target load");
    ok &= expect(server_context.find("server_dflash_single_explicit_draft_device(params_base") != std::string::npos &&
                 server_context.find("params_base.output_device = dflash_single_draft_dev;") != std::string::npos &&
                 server_context.find("DFlash target output tensor will use explicit draft device") != std::string::npos &&
                 server_context.find("const bool dflash_explicit_single_draft_device") != std::string::npos &&
                 server_context.find("params_dft.split_mode = LLAMA_SPLIT_MODE_NONE;") != std::string::npos,
        "DFlash single --spec-draft-device must move the target output tensor before loading the target model");
    ok &= expect(model_cpp.find("!llm_arch_is_dflash_drafter(model->arch)") != std::string::npos,
        "public DFlash hparam accessors must return zero for non-DFlash model architectures");
    ok &= expect(server_context.find("failed to initialize slot speculative decoding context") != std::string::npos,
        "slot speculative initialization must convert DFlash contract failures into load errors instead of process termination");
    ok &= expect(speculative.find("draft-simple requires ctx_tgt and ctx_dft") != std::string::npos,
        "ordinary draft-simple speculation must reject missing target/draft contexts before dereferencing them");
    ok &= expect(server_context.find("params_dft.n_parallel   = params_base.n_parallel") != std::string::npos &&
                 server_context.find("ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));") != std::string::npos &&
                 server_context.find("params_base.speculative.draft.ctx_tgt = ctx_tgt") != std::string::npos &&
                 server_context.find("params_base.speculative.draft.ctx_dft = ctx_dft.get()") != std::string::npos,
        "ordinary non-DFlash draft models must create and wire a draft context for upstream draft-simple speculation");
    ok &= expect(speculative.find("llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pos_min_by_seq[seq_id], -1);") != std::string::npos,
        "ordinary draft-simple process must roll draft KV back before replaying overlapping verifier batches");
    ok &= expect(server_cpp.find("n_parallel is set to auto, using n_parallel = 4 and kv_unified = true") != std::string::npos &&
                 server_cpp.find("KVarN requires non-unified KV") == std::string::npos,
        "server auto n_parallel must allow unified KV for KVarN");
    ok &= expect(server_context.find("server_probe_model_arch") != std::string::npos &&
                 server_context.find("gguf_find_key(ctx_gguf, \"general.architecture\")") != std::string::npos &&
                 server_context.find("params_base.speculative.dflash_selected_or_pending()") != std::string::npos &&
                 server_context.find("server_arch_name_is_recurrent_or_hybrid") != std::string::npos &&
                 server_context.find("\"qwen35\"") != std::string::npos &&
                 server_context.find("\"gemma4\"") == std::string::npos &&
                 server_context.find("does not need recurrent rollback") != std::string::npos &&
                 server_context.find("recurrent-only backup cells") != std::string::npos,
        "DFlash recurrent rollback allocation must be gated by probed recurrent/hybrid target architecture");
    ok &= expect(server_context.find("DFlash target requires recurrent rollback, but neither recurrent snapshots nor backup cells are available") != std::string::npos,
        "DFlash startup must fail closed if a recurrent/hybrid target has no rollback mechanism");
    ok &= expect(server_context.find("auto-enabled kv-unified: DDTree backup uses internal attention streams") != std::string::npos &&
                 server_context.find("KVarN requires non-unified KV") == std::string::npos,
        "DFlash DDTree backup auto-unification must allow unified KV for KVarN");
    ok &= expect(server_context.find("server_context_n_ctx_for_internal_seqs(") == std::string::npos &&
                 server_context.find("expanded target n_ctx from %d to %d") == std::string::npos &&
                 server_context.find("recurrent_backup_attention_streams") != std::string::npos &&
                 llama_h.find("llama_memory_seq_rm_recurrent") != std::string::npos &&
                 memory_h.find("virtual bool seq_rm_recurrent") != std::string::npos &&
                 context_cpp.find("Flat mode: no duplicate entries at same position, safe to keep accepted KV\n        int kv_keep_pos = n_past_before + n_accepted;\n        mem_attn->seq_rm(seq_id, kv_keep_pos, -1);") != std::string::npos,
        "DFlash flat recurrent rollback must not preserve visible context by allocating dead backup KV streams");
    ok &= expect(context_cpp.find("llama_memory_seq_add") != std::string::npos &&
                 context_cpp.find("llama_memory_can_shift(mem)") != std::string::npos &&
                 context_cpp.find("cannot add/divide sequence positions because the memory implementation does not support shifting") != std::string::npos,
        "public sequence position mutation wrappers must fail fast when memory cannot shift");
    ok &= expect(llama_h.find("llama_memory_can_seq_rm") != std::string::npos &&
                 memory_h.find("virtual bool can_seq_rm") != std::string::npos &&
                 context_cpp.find("return mem->can_seq_rm(seq_id, p0, p1);") != std::string::npos,
        "public memory API must expose a side-effect-free seq_rm capability check");
    ok &= expect(kv_cache_kvarn_cpp.find("bool llama_kv_cache_kvarn::can_seq_rm") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("return can_remove(seq_id, p0, p1);") != std::string::npos,
        "KVarN must report range-specific seq_rm capability without mutating cache metadata");
    ok &= expect(kv_cache_iswa_cpp.find("bool llama_kv_cache_iswa::can_seq_rm") != std::string::npos &&
                 kv_cache_iswa_cpp.find("kv_base->can_seq_rm(seq_id, p0, p1)") != std::string::npos &&
                 kv_cache_iswa_cpp.find("if (!can_seq_rm(seq_id, p0, p1))") != std::string::npos,
        "ISWA seq_rm must check both child caches before mutating either child");
    ok &= expect(kv_cache_dsa_cpp.find("bool llama_kv_cache_dsa::can_seq_rm") != std::string::npos &&
                 kv_cache_dsa_cpp.find("kv_mla->can_seq_rm(seq_id, p0, p1)") != std::string::npos &&
                 kv_cache_dsa_cpp.find("if (!can_seq_rm(seq_id, p0, p1))") != std::string::npos,
        "DSA seq_rm must check both child caches before mutating either child");
    ok &= expect(kv_cache_iswa_cpp.find("res = res & kv_base->seq_rm_cell") == std::string::npos &&
                 kv_cache_iswa_cpp.find("if (!kv_base->seq_rm_cell(seq_id, cell_idx))") != std::string::npos &&
                 kv_cache_dsa_cpp.find("res = res & kv_mla->seq_rm_cell") == std::string::npos &&
                 kv_cache_dsa_cpp.find("if (!kv_mla->seq_rm_cell(seq_id, cell_idx))") != std::string::npos,
        "composite seq_rm_cell must short-circuit so a failed first child does not mutate later children");
    ok &= expect(memory_hybrid.find("bool llama_memory_hybrid::can_seq_rm") != std::string::npos &&
                 memory_hybrid.find("mem_recr->can_seq_rm(seq_id, p0, p1)") != std::string::npos &&
                 memory_hybrid.find("if (!can_seq_rm(seq_id, p0, p1))") != std::string::npos &&
                 memory_hybrid_iswa.find("bool llama_memory_hybrid_iswa::can_seq_rm") != std::string::npos &&
                 memory_hybrid_iswa.find("mem_attn->can_seq_rm(seq_id, p0, p1)") != std::string::npos,
        "hybrid seq_rm must check recurrent and attention children before mutating either child");
    const std::string kvarn_partial_restore = slice_between(
            kv_cache_kvarn_cpp,
            "bool llama_kv_cache_kvarn::requires_state_for_partial_restore() const",
            "\n}\n");
    ok &= expect(memory_h.find("virtual bool requires_state_for_partial_restore() const") != std::string::npos &&
                 kvarn_partial_restore.find("return true;") != std::string::npos &&
                 memory_hybrid.find("mem_attn->requires_state_for_partial_restore()") != std::string::npos &&
                 kv_cache_iswa_cpp.find("kv_base->requires_state_for_partial_restore()") != std::string::npos,
        "KVarN attention memory must be included in partial sequence state restores");
    ok &= expect(llama_h.find("llama_context_kv_n_stream") != std::string::npos &&
                 llama_h.find("llama_memory_state_seq_restore_requires_exclusive_kv_stream") != std::string::npos &&
                 context_cpp.find("return mem ? mem->get_kv_n_stream() : 0;") != std::string::npos &&
                 memory_h.find("virtual bool state_seq_restore_requires_exclusive_kv_stream() const") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("bool llama_kv_cache_kvarn::state_seq_restore_requires_exclusive_kv_stream() const") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("return get_kv_n_stream() == 1;") != std::string::npos &&
                 memory_hybrid.find("mem_attn->state_seq_restore_requires_exclusive_kv_stream()") != std::string::npos &&
                 kv_cache_iswa_cpp.find("kv_base->state_seq_restore_requires_exclusive_kv_stream()") != std::string::npos,
        "KVarN unified state restore must be exposed as a memory capability and forwarded through composite memories");
    ok &= expect(server_context.find("bool uses_shared_kv_pool() const") != std::string::npos &&
                 server_context.find("return llama_context_kv_n_stream(ctx_tgt) == 1;") != std::string::npos &&
                 server_context.find("bool prompt_cache_state_restore_allowed(const server_slot & slot) const") != std::string::npos &&
                 server_context.find("llama_memory_state_seq_restore_requires_exclusive_kv_stream(llama_get_memory(ctx_tgt))") != std::string::npos &&
                 server_context.find("other.prompt.n_tokens() > 0") != std::string::npos &&
                 server_task.find("allow_state_restore") != std::string::npos &&
                 server_task.find("skipped state restore because the KV stream is shared with another live slot") != std::string::npos,
        "server prompt-cache restore must be gated by the actual memory restore capability");
    ok &= expect(memory_recurrent_cpp.find("bool llama_memory_recurrent::can_seq_rm") != std::string::npos &&
                 memory_recurrent_cpp.find("rollback <= (llama_pos) n_rs_seq") != std::string::npos,
        "recurrent memory must expose the same bounded rollback capability used by seq_rm");
    ok &= expect(server_context.find("const llama_pos checkpoint_trim_p0") == std::string::npos &&
                 server_context.find("target memory cannot trim from") == std::string::npos &&
                 server_context.find("draft memory cannot trim from") == std::string::npos &&
                 server_context.find("server_prompt_checkpoint_matches_restore_window") != std::string::npos,
        "server prompt-cache restore must not preflight checkpoints against the live memory state");
    ok &= expect(server_context.find("const bool has_new_tokens = (n_past < slot.task->n_tokens());") != std::string::npos &&
                 server_context.find("pos_next - n_swa - (has_new_tokens ? 0 : 1)") != std::string::npos,
        "server prompt-cache checkpoint threshold must account for newly appended suffix tokens");
    ok &= expect(server_context.find("const llama_pos prompt_trim_p0 = slot.prompt.tokens.pos_next(n_past);") != std::string::npos &&
                 server_context.find("memory cannot trim cached suffix") != std::string::npos &&
                 server_context.find("slot.prompt.checkpoints.clear();") != std::string::npos,
        "server prompt-cache reuse must fall back to full prefill when the final cached-suffix trim is unsupported");
    ok &= expect(kv_cache_kvarn_cpp.find("GGML_ABORT(\"KVarN does not support position shifts\")") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("GGML_ABORT(\"KVarN does not support position division\")") != std::string::npos,
        "KVarN seq_add/seq_div must fail fast instead of logging and continuing");
    ok &= expect(kv_cache_kvarn_cpp.find("constexpr uint32_t KVAR_N_STATE_VERSION = 7") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("if (version < 7)") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("if (version == 1)") == std::string::npos &&
                 kv_cache_kvarn_cpp.find("KVAR_N_STATE_RECORDS_FULL") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("KVAR_N_STATE_STAGE_ONLY_PARTIAL") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("const uint32_t state_kind = partial_state ? KVAR_N_STATE_STAGE_ONLY_PARTIAL : KVAR_N_STATE_RECORDS_FULL;") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("saved_streams") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("metadata->get_stream_for_seq(seq_id)") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("layer.k_records_stream[stream]") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("layer.v_stage_stream[stream]") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("n_groups_used") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("write_kvarn_tensor_slice") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("read_kvarn_tensor_slice") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("if (state_kind == KVAR_N_STATE_RECORDS_FULL)") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("if (state_kind == KVAR_N_STATE_STAGE_ONLY_PARTIAL)") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("ggml_backend_tensor_memset") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("const uint64_t size64 = size") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("io.write(&stage_groups, sizeof(stage_groups))") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("io.read(&saved_stage_groups, sizeof(saved_stage_groups))") != std::string::npos,
        "KVarN sequence state must serialize full records compactly and partial checkpoints as stage-only overlays");
    ok &= expect(kv_cache_kvarn_cpp.find("pending_stream_copies") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("llama_synchronize(lctx)") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("copy_kvarn_stream") != std::string::npos,
        "KVarN cross-stream seq_cp must be queued and copied during synchronized update");
    ok &= expect(kv_cache_kvarn_cpp.find("ggml_backend_kvarn_native_ops") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("ggml_backend_reg_get_proc_address") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("std::strstr(name, \"CUDA\")") == std::string::npos &&
                 kv_cache_kvarn_cpp.find("std::strstr(name, \"ROCm\")") == std::string::npos &&
                 kv_cache_kvarn_cpp.find("std::strstr(name, \"HIP\")") == std::string::npos &&
                 kv_cache_kvarn_cpp.find("std::strstr(name, \"Vulkan\")") == std::string::npos,
        "KVarN backend support must use a backend capability proc-address hook, not registry-name substring matching");
    ok &= expect(cuda_reg.find("ggml_backend_kvarn_native_ops") != std::string::npos &&
                 cuda_reg.find("ggml_backend_cuda_kvarn_native_ops") != std::string::npos &&
                 cuda_reg.find("ggml_cuda_kvarn_required_shared_bytes()") != std::string::npos &&
                 cuda_reg.find("GGML_USE_MUSA") != std::string::npos,
        "CUDA/HIP backend registry must expose a KVarN native-op capability hook and keep MUSA disabled");
    ok &= expect(vulkan_cpp.find("pipeline_kvarn_store") != std::string::npos &&
                 vulkan_cpp.find("pipeline_kvarn_materialize") == std::string::npos &&
                 vulkan_cpp.find("ggml_backend_vk_kvarn_native_ops") != std::string::npos &&
                 vulkan_cpp.find("return false;") != std::string::npos &&
                 vulkan_cpp.find("ggml_backend_kvarn_native_ops") != std::string::npos &&
                 vulkan_cpp.find("GGML_OP_KVARN_STORE") != std::string::npos &&
                 vulkan_cpp.find("case GGML_OP_KVARN_VIEW") == std::string::npos &&
                 vulkan_cpp.find("GGML_OP_KVARN_MATERIALIZE") == std::string::npos &&
                 vulkan_cpp.find("ggml_vk_kvarn_store") != std::string::npos &&
                 vulkan_cpp.find("ggml_vk_kvarn_materialize") == std::string::npos &&
                 vulkan_cpp.find("get_proc_address = */ ggml_backend_vk_reg_get_proc_address") != std::string::npos,
        "Vulkan backend must keep KVarN store support but not advertise full native view support without a view consumer");
    ok &= expect(vulkan_shader_gen.find("kvarn_store.comp") != std::string::npos &&
                 vulkan_shader_gen.find("kvarn_materialize.comp") == std::string::npos &&
                 vulkan_kvarn_store.find("layout(local_size_x = 128") != std::string::npos &&
                 vulkan_kvarn_store.find("packHalf2x16") != std::string::npos &&
                 vulkan_kvarn_store.find("data_records") != std::string::npos &&
                 vulkan_kvarn_materialize.empty(),
        "Vulkan KVarN shaders must not generate a materialize shader");
    ok &= expect(llama_kvarn_h.find("LLAMA_API") == std::string::npos,
        "internal C++ KVarN helper declarations must not be exported from the public DLL ABI");
    ok &= expect(kv_cache_iswa_cpp.find("KVarN enabled for all layers (non-SWA full-context and SWA sliding-window ring)") != std::string::npos &&
                 kv_cache_iswa_cpp.find("make_cache(size_base, 0, LLAMA_SWA_TYPE_NONE, filter_base, mem_other_base, use_kvarn)") != std::string::npos &&
                 kv_cache_iswa_cpp.find("make_cache(size_swa, hparams.n_swa, hparams.swa_type, filter_swa, mem_other_swa, use_kvarn)") != std::string::npos &&
                 kv_cache_iswa_cpp.find("n_seq_max > 1 && !unified") != std::string::npos &&
                 kv_cache_iswa_cpp.find("SWA KVarN ring requires a single stream") != std::string::npos,
        "ISWA with KVarN must keep KVarN on single-stream SWA rings and fall back only when non-unified mode creates multiple streams");
    ok &= expect(kv_cache_kvarn_h.find("void set_input_kvarn_mat_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;") != std::string::npos &&
                 graph_cpp.find("kvarn_swa->set_input_kvarn_mat_idxs(self_kvarn_mat_idxs_swa, ubatch);") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("const auto & sinfo = current_sinfo();") != std::string::npos &&
                 contains_ws(kv_cache_kvarn_cpp, "data[sinfo.idxs[s][i]] = (int64_t) ubatch->pos[s*sinfo.size() + i];"),
        "SWA KVarN native mat indexes must include the pending ubatch overlay before KV metadata is committed");
    ok &= expect(context_cpp.find("is experimental; only kvarn_k4v2_g128 is reference-aligned") == std::string::npos,
        "KVarN presets must not emit experimental/reference-aligned startup warnings");
    ok &= expect(llama_bench.find("bench_cache_type_from_name") != std::string::npos &&
                 llama_bench.find("kvarn_bits_from_cache_type") != std::string::npos &&
                 llama_bench.find("normalize_kvarn_cache_pair") != std::string::npos &&
                 llama_bench.find("cache_type_list_has_kvarn(params.type_k)") != std::string::npos &&
                 llama_bench.find("cparams.kvarn") != std::string::npos &&
                 llama_bench.find("bench_cache_type_name(type_k)") != std::string::npos,
        "llama-bench must accept cache-type KVarN pseudo names and pass KVarN params to llama_context_params");
    ok &= expect(cuda_fattn.find("ggml_cuda_fattn_kvarn_init_descs_kernel") != std::string::npos &&
                 cuda_fattn.find("ggml_cuda_fattn_kvarn_live_group_for_thread") != std::string::npos &&
                 cuda_fattn.find("k_desc.live_group = k_partial[0]") != std::string::npos &&
                 cuda_fattn.find("v_desc.live_group = v_partial[0]") != std::string::npos &&
                 cuda_fattn.find("cudaMemcpyAsync(k_desc") == std::string::npos,
        "CUDA native KVarN FlashAttention must build graph-capture-safe descriptors and live groups once per stream on device");
    ok &= expect(ggml_cuda_kvarn_h.find("ggml_cuda_kvarn_required_shared_bytes") != std::string::npos &&
                 ggml_cuda_kvarn.find("KVAR_N_LOWSHMEM_BYTES") != std::string::npos &&
                 ggml_cuda_kvarn.find("kvarn_store_kernel_hishmem") != std::string::npos &&
                 ggml_cuda_kvarn.find("kvarn_store_kernel_lowshmem") != std::string::npos &&
                 ggml_cuda_kvarn.find("GGML_KVARN_") == std::string::npos,
        "CUDA/HIP KVarN store must keep high/low-shared-memory paths without KVarN runtime env knobs");
    ok &= expect(cuda_fattn_kvarn.empty() &&
                  cuda_fattn.find("fattn-kvarn.cuh") == std::string::npos &&
                  cuda_fattn.find("ggml_cuda_flash_attn_ext_kvarn(") == std::string::npos &&
                  cuda_fattn_h.find("ggml_cuda_flash_attn_ext_kvarn") == std::string::npos &&
                  cuda_cpp.find("ggml_cuda_try_fuse_kvarn_fattn") == std::string::npos &&
                  cuda_cpp.find("ggml_cuda_flash_attn_ext_kvarn_supported") == std::string::npos &&
                  cuda_cpp.find("GGML_KVARN_FUSED") == std::string::npos &&
                  cuda_cpp.find("GGML_KVARN_NATIVE_FA") == std::string::npos,
        "legacy CUDA KVarN fused FlashAttention path and route envs must stay removed");
    ok &= expect(graph_cpp.find("GGML_KVARN_FORCE_MATERIALIZE") == std::string::npos &&
                 graph_cpp.find("kvarn_force_materialize_enabled") == std::string::npos &&
                 graph_cpp.find("GGML_KVARN_ROTATED_FA") == std::string::npos &&
                 graph_cpp.find("get_k_rotated") == std::string::npos &&
                 graph_cpp.find("get_v_rotated") == std::string::npos &&
                 graph_cpp.find("ggml_mul_mat_aux(ctx0, q, inp->self_kvarn_rot)") != std::string::npos &&
                 graph_cpp.find("ggml_mul_mat_aux(ctx0, cur, inp->self_kvarn_rot)") != std::string::npos &&
                 graph_cpp.find("GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_KV_OUTPUT_ORIGINAL") == std::string::npos &&
                 cuda_fattn_mma_f16.find("kvarn_output_original_domain") == std::string::npos &&
                 cuda_fattn_mma_f16.find("flash_attn_ext_kvarn_inverse_wht_output_tile") == std::string::npos &&
                 kv_cache_kvarn_h.find("get_k_native") != std::string::npos &&
                 kv_cache_kvarn_h.find("get_v_native") != std::string::npos &&
                 kv_cache_kvarn_h.find("get_k_rotated") == std::string::npos &&
                 kv_cache_kvarn_h.find("get_v_rotated") == std::string::npos &&
                 kv_cache_kvarn_h.find("build_input_kvarn_rot") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("ggml_kvarn_view") != std::string::npos &&
                 kv_cache_kvarn_cpp.find("result->op_params[5] = emit_rotated ? 1 : 0") == std::string::npos &&
                 cuda_fattn_mma_kvarn.find("for (int stride = 1; stride < GGML_CUDA_FATTN_KVARN_DIM; stride <<= 1)") == std::string::npos,
        "KVarN attention graph must keep native K/V views, rotate only Q/output once, and avoid per-tile inverse-WHT in native FA");

    return ok ? 0 : 1;
}
