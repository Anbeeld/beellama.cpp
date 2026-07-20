#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "../ggml/src/ggml-cuda/fattn-kvarn-route-policy.h"

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

static std::string slice_between(const std::string & text, const std::string & begin, const std::string & end) {
    const size_t b = text.find(begin);
    if (b == std::string::npos) {
        return {};
    }
    const size_t e = text.find(end, b);
    if (e == std::string::npos) {
        return text.substr(b);
    }
    return text.substr(b, e - b);
}

static size_t count_occurrences(const std::string & text, const std::string & needle) {
    size_t count = 0;
    for (size_t pos = 0; (pos = text.find(needle, pos)) != std::string::npos; pos += needle.size()) {
        ++count;
    }
    return count;
}

int main(int argc, char ** argv) {
    bool ok = true;

    ok &= expect(argc == 2, "expected repo root argument");
    ok &= expect(GGML_CUDA_FATTN_KVARN_DECODE_MAX_Q == 16,
        "CUDA KVarN split decode must cover a complete DFlash verification block");
    if (!ok) {
        return 1;
    }

    const std::string root = argv[1];
    const std::string fattn = read_file(root + "/ggml/src/ggml-cuda/fattn.cu");
    const std::string kvarn = read_file(root + "/ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu");
    const std::string cmake = read_file(root + "/ggml/CMakeLists.txt");
    const std::string cuda_cmake = read_file(root + "/ggml/src/ggml-cuda/CMakeLists.txt");
    const std::string release = read_file(root + "/.github/workflows/release.yml");

    const auto expect_route = [&](const ggml_cuda_fattn_kvarn_route_input & base,
                                  ggml_cuda_fattn_kvarn_route expected,
                                  const char * message) {
        auto without_meta = base;
        without_meta.body_meta_requested = false;
        auto with_meta = base;
        with_meta.body_meta_requested = true;
        ok &= expect(ggml_cuda_fattn_kvarn_select_route(without_meta) == expected, message);
        ok &= expect(ggml_cuda_fattn_kvarn_select_route(with_meta) == expected,
            "optional body metadata changed an eligible KVarN route");
    };

    expect_route({256, 1, 6, 4, 4, false, false, false, true, false},
        GGML_CUDA_FATTN_KVARN_ROUTE_DECODE_SPLIT,
        "Qwen-like D256 global decode did not select split decode");
    expect_route({512, 1, 16, 4, 4, false, false, false, true, false},
        GGML_CUDA_FATTN_KVARN_ROUTE_DECODE_SPLIT,
        "Gemma-like D512 global decode did not select split decode");
    expect_route({256, 1, 2, 4, 4, true, false, true, true, false},
        GGML_CUDA_FATTN_KVARN_ROUTE_DECODE_VECTOR,
        "Gemma-like D256 SWA decode did not select vector decode");
    for (int n_q = 2; n_q <= GGML_CUDA_FATTN_KVARN_DECODE_MAX_Q; ++n_q) {
        expect_route({256, n_q, 6, 4, 4, false, false, false, true, false},
            GGML_CUDA_FATTN_KVARN_ROUTE_DECODE_SPLIT,
            "supported multi-token verification shape did not select split decode");
    }
    expect_route({384, 1, 6, 4, 4, false, false, false, false, false},
        GGML_CUDA_FATTN_KVARN_ROUTE_GENERIC_MMA,
        "unsupported head shape did not remain on generic fallback");
    expect_route({256, 64, 6, 4, 4, false, false, false, false, true},
        GGML_CUDA_FATTN_KVARN_ROUTE_PROMPT_PREFILL,
        "prompt/prefill shape did not retain the prompt route");
    const std::string allocation = slice_between(fattn,
            "size_t ggml_cuda_flash_attn_ext_get_alloc_size",
            "void ggml_cuda_flash_attn_ext");
    const std::string exec = slice_between(fattn,
            "void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst)",
            "bool ggml_cuda_flash_attn_ext_support");

    ok &= expect(fattn.find("#include \"fattn-kvarn-dispatch.cuh\"") != std::string::npos,
        "CUDA FlashAttention must include the isolated KVarN dispatcher");
    ok &= expect(!allocation.empty() &&
                 allocation.find("ggml_cuda_flash_attn_ext_kvarn_uses_views(dst)") != std::string::npos &&
                 allocation.find("GGML_ASSERT(ggml_cuda_flash_attn_ext_kvarn_supported(device, dst))") == std::string::npos &&
                 allocation.find("ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, false, false)") != std::string::npos &&
                 allocation.find("return f16_extra.end - (uintptr_t) dst->data;") != std::string::npos,
        "KVarN views must allocate the upstream MMA fixup workspace structurally without materializing F16 K/V buffers");
    ok &= expect(exec.find("ggml_cuda_flash_attn_ext_kvarn_uses_views(dst)") != std::string::npos &&
                 exec.find("ggml_cuda_flash_attn_ext_kvarn(ctx, dst)") != std::string::npos &&
                 exec.find("unsupported KVarN CUDA FlashAttention route") != std::string::npos,
        "KVarN views must be explicitly dispatched and never silently enter generic FlashAttention");
    ok &= expect(exec.find("ggml_cuda_flash_attn_ext_kvarn_uses_views(dst)") < exec.find("switch (ggml_cuda_get_best_fattn_kernel"),
        "KVarN dispatch must precede generic CUDA FlashAttention selection");

    ok &= expect(kvarn.find("ggml_cuda_flash_attn_ext_mma_kvarn(ctx, dst);") != std::string::npos &&
                 kvarn.find("return true;") != std::string::npos,
        "unsupported KVarN fast-decode pairs must fall through to descriptor-native MMA");
    const std::string no_kvarn = slice_between(kvarn,
            "#if !defined(GGML_CUDA_KVARN)",
            "#else");
    ok &= expect(no_kvarn.find("ggml_cuda_flash_attn_ext_kvarn_portable_supported") != std::string::npos,
        "GGML_CUDA_KVARN=OFF must retain a false portable-support stub for generic FlashAttention");
    ok &= expect(kvarn.find("if (dst->src[8] == nullptr)") == std::string::npos,
        "optional body metadata must not disable specialized KVarN decode routes");
    ok &= expect(kvarn.find("args.dst_meta") != std::string::npos,
        "specialized KVarN decode must receive the optional body metadata destination");
    ok &= expect(kvarn.find("#if defined(GGML_CUDA_FA_ALL_QUANTS)") != std::string::npos &&
                 kvarn.find("GGML_CUDA_FA_HALF_QUANTS") == std::string::npos,
        "KVarN fast decode must have only default and ALL build tiers");
    ok &= expect(cmake.find("option(GGML_CUDA_KVARN ") != std::string::npos &&
                 cmake.find("option(GGML_CUDA_KVARN ") < cmake.find(" ON)", cmake.find("option(GGML_CUDA_KVARN ")),
        "GGML_CUDA_KVARN must be the default-on KVarN CUDA compilation gate");
    ok &= expect(cmake.find("option(GGML_CUDA_KVARN_FA") == std::string::npos &&
                 cmake.find("option(GGML_CUDA_KVARN_FAST_DECODE_ALL_PAIRS") == std::string::npos &&
                 cmake.find("unset(GGML_CUDA_KVARN_FA CACHE)") != std::string::npos &&
                 cmake.find("unset(GGML_CUDA_KVARN_FAST_DECODE_ALL_PAIRS CACHE)") != std::string::npos &&
                 cuda_cmake.find("GGML_CUDA_KVARN_FA") == std::string::npos &&
                 cuda_cmake.find("GGML_CUDA_KVARN_FAST_DECODE_ALL_PAIRS") == std::string::npos &&
                 kvarn.find("GGML_CUDA_KVARN_FA") == std::string::npos &&
                 kvarn.find("GGML_CUDA_KVARN_FAST_DECODE_ALL_PAIRS") == std::string::npos,
        "KVarN CUDA compilation must expose only one KVarN option and clear obsolete cache entries");
    ok &= expect(cmake.find("if (NOT GGML_CUDA_KVARN)") != std::string::npos &&
                 cmake.find("if (GGML_CUDA_FA_ALL_QUANTS)") != std::string::npos &&
                 cuda_cmake.find("if (GGML_CUDA_KVARN)") != std::string::npos &&
                 kvarn.find("#if !defined(GGML_CUDA_KVARN)") != std::string::npos,
        "GGML_CUDA_KVARN must gate all KVarN CUDA sources while ALL_QUANTS selects the pair matrix");
    ok &= expect(cmake.find("GGML_CUDA_KVARN_DEFAULT_PAIR_COUNT EQUAL 15") != std::string::npos &&
                 cmake.find("GGML_CUDA_FA_HALF_QUANTS") == std::string::npos,
        "CMake must retain exactly the 15-pair default KVarN fast-decode policy without HALF");
    ok &= expect(count_occurrences(release, "-DGGML_CUDA_KVARN=ON") == 4,
        "all Linux/Windows CUDA and ROCm release builds must explicitly enable KVarN");

    return ok ? 0 : 1;
}
