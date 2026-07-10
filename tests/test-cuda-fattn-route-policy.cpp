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

int main(int argc, char ** argv) {
    bool ok = true;

    ok &= expect(argc == 2, "expected repo root argument");
    if (!ok) {
        return 1;
    }

    const std::string root = argv[1];
    const std::string fattn = read_file(root + "/ggml/src/ggml-cuda/fattn.cu");
    const std::string kvarn = read_file(root + "/ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu");
    const std::string cmake = read_file(root + "/ggml/CMakeLists.txt");
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
                 allocation.find("GGML_ASSERT(ggml_cuda_flash_attn_ext_kvarn_supported(device, dst))") != std::string::npos &&
                 allocation.find("ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, false, false)") != std::string::npos &&
                 allocation.find("return f16_extra.end - (uintptr_t) dst->data;") != std::string::npos,
        "KVarN views must allocate the upstream MMA fixup workspace without materializing F16 K/V buffers");
    ok &= expect(exec.find("ggml_cuda_flash_attn_ext_kvarn_uses_views(dst)") != std::string::npos &&
                 exec.find("ggml_cuda_flash_attn_ext_kvarn(ctx, dst)") != std::string::npos &&
                 exec.find("unsupported KVarN CUDA FlashAttention route") != std::string::npos,
        "KVarN views must be explicitly dispatched and never silently enter generic FlashAttention");
    ok &= expect(exec.find("ggml_cuda_flash_attn_ext_kvarn_uses_views(dst)") < exec.find("switch (ggml_cuda_get_best_fattn_kernel"),
        "KVarN dispatch must precede generic CUDA FlashAttention selection");

    ok &= expect(kvarn.find("ggml_cuda_flash_attn_ext_mma_kvarn(ctx, dst);") != std::string::npos &&
                 kvarn.find("return true;") != std::string::npos,
        "unsupported KVarN fast-decode pairs must fall through to descriptor-native MMA");
    ok &= expect(kvarn.find("#if defined(GGML_CUDA_FA_ALL_QUANTS)") != std::string::npos &&
                 kvarn.find("GGML_CUDA_FA_HALF_QUANTS") == std::string::npos,
        "KVarN fast decode must have only default and ALL build tiers");
    ok &= expect(cmake.find("GGML_CUDA_KVARN_FAST_DECODE_DEFAULT_PAIR_COUNT EQUAL 15") != std::string::npos &&
                 cmake.find("GGML_CUDA_FA_HALF_QUANTS") == std::string::npos,
        "CMake must retain exactly the 15-pair default KVarN fast-decode policy without HALF");

    return ok ? 0 : 1;
}
