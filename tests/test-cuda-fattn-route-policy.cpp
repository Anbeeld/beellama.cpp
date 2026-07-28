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
    ok &= expect(GGML_CUDA_FATTN_KVARN_SPECIALIZED_DECODE_MAX_Q == 16,
        "specialized CUDA KVarN attention must cover a complete DFlash verification block");
    if (!ok) {
        return 1;
    }

    const std::string root = argv[1];
    const std::string fattn = read_file(root + "/ggml/src/ggml-cuda/fattn.cu");
    const std::string kvarn = read_file(root + "/ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu");
    const std::string cuda_backend = read_file(root + "/ggml/src/ggml-cuda/ggml-cuda.cu");
    const std::string cmake = read_file(root + "/ggml/CMakeLists.txt");
    const std::string cuda_cmake = read_file(root + "/ggml/src/ggml-cuda/CMakeLists.txt");
    const std::string hip_cmake = read_file(root + "/ggml/src/ggml-hip/CMakeLists.txt");
    const std::string musa_cmake = read_file(root + "/ggml/src/ggml-musa/CMakeLists.txt");
    const std::string kvarn_mma = read_file(root + "/ggml/src/ggml-cuda/fattn-mma-kvarn-impl.cuh");
    const std::string kvarn_mma_case = read_file(root + "/ggml/src/ggml-cuda/fattn-mma-kvarn-case.cuh");
    const std::string fattn_mma_f16 = read_file(root + "/ggml/src/ggml-cuda/fattn-mma-f16.cuh");
    const std::string kvarn_decode = read_file(root + "/ggml/src/ggml-cuda/fattn-mma-kvarn-decode.cuh");
    const std::string kvarn_dispatch_header = read_file(root + "/ggml/src/ggml-cuda/fattn-kvarn-dispatch.cuh");
    const std::string kvarn_wht = read_file(root + "/ggml/src/ggml-cuda/kvarn-wht.cu");
    const std::string kvarn_store = read_file(root + "/ggml/src/ggml-cuda/kvarn.cu");
    const std::string kvarn_wide_instance = read_file(
        root + "/ggml/src/ggml-cuda/template-instances/fattn-mma-kvarn-instance-ncols1_16-ncols2_8.cu");
    const std::string release = read_file(root + "/.github/workflows/release.yml");
    const std::string architecture_compile =
        read_file(root + "/.github/workflows/cuda-architecture-compile.yml");

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
    for (int n_q = 2; n_q <= GGML_CUDA_FATTN_KVARN_SPECIALIZED_DECODE_MAX_Q; ++n_q) {
        expect_route({256, n_q, 6, 4, 4, false, false, false, true, false},
            GGML_CUDA_FATTN_KVARN_ROUTE_GENERIC_MMA,
            "supported multi-token verification shape did not select tiled native MMA");
    }
    expect_route({384, 1, 6, 4, 4, false, false, false, false, false},
        GGML_CUDA_FATTN_KVARN_ROUTE_GENERIC_MMA,
        "unsupported head shape did not remain on generic fallback");
    expect_route({256, 64, 6, 4, 4, false, false, false, false, true},
        GGML_CUDA_FATTN_KVARN_ROUTE_PROMPT_PREFILL,
        "prompt/prefill shape did not retain the prompt route");
    ok &= expect(ggml_cuda_fattn_kvarn_use_wide_mma(16, 6, true),
        "supported Q16/GQA6 verification did not select the wide MMA tile");
    ok &= expect(!ggml_cuda_fattn_kvarn_use_wide_mma(8, 6, true) &&
                 !ggml_cuda_fattn_kvarn_use_wide_mma(16, 4, true) &&
                 !ggml_cuda_fattn_kvarn_use_wide_mma(16, 6, false),
        "wide MMA tile must retain shape and device-resource fallbacks");

    const auto cuda_caps = ggml_cuda_fattn_kvarn_select_capabilities({
        GGML_CUDA_FATTN_KVARN_BACKEND_CUDA, 32, true, true, 1024, 48*1024, 4*1024,
    });
    ok &= expect(cuda_caps.generic_mma && cuda_caps.decode_split && cuda_caps.decode_vector &&
                 cuda_caps.portable_native && cuda_caps.specialized_routes &&
                 cuda_caps.store_materialize && cuda_caps.original_v_domain &&
                 cuda_caps.portable_tail_f16 && cuda_caps.portable_tail_bf16 &&
                 cuda_caps.rotated_query_max_specialized == 16 &&
                 cuda_caps.rotated_query_max_portable == UINT32_MAX,
        "Turing-or-newer CUDA must expose independent portable and specialized KVarN capabilities");

    const auto pre_turing_cuda_caps = ggml_cuda_fattn_kvarn_select_capabilities({
        GGML_CUDA_FATTN_KVARN_BACKEND_CUDA, 32, false, true, 1024, 48*1024, 4*1024,
    });
    ok &= expect(!pre_turing_cuda_caps.generic_mma &&
                 !pre_turing_cuda_caps.decode_split &&
                 !pre_turing_cuda_caps.decode_vector &&
                 pre_turing_cuda_caps.portable_native &&
                 pre_turing_cuda_caps.portable_tail_f16 &&
                 pre_turing_cuda_caps.portable_tail_bf16 &&
                 !pre_turing_cuda_caps.specialized_routes &&
                 !pre_turing_cuda_caps.original_v_domain &&
                 pre_turing_cuda_caps.rotated_query_max_portable == UINT32_MAX &&
                 pre_turing_cuda_caps.rotated_query_max_specialized == 0,
        "pre-Turing CUDA must retain unbounded portable rotated-domain KVarN attention");

    const auto low_shared_cuda_caps = ggml_cuda_fattn_kvarn_select_capabilities({
        GGML_CUDA_FATTN_KVARN_BACKEND_CUDA, 32, false, true, 1024, 2*1024, 4*1024,
    });
    ok &= expect(!low_shared_cuda_caps.store_materialize &&
                 !low_shared_cuda_caps.portable_native &&
                 low_shared_cuda_caps.route_families == 0,
        "CUDA with insufficient shared memory must fail KVarN capabilities closed");

    const auto low_threads_cuda_caps = ggml_cuda_fattn_kvarn_select_capabilities({
        GGML_CUDA_FATTN_KVARN_BACKEND_CUDA, 32, false, true, 64, 48*1024, 4*1024,
    });
    ok &= expect(!low_threads_cuda_caps.store_materialize &&
                 !low_threads_cuda_caps.portable_native,
        "CUDA unable to launch a 128-thread portable block must fail closed");

    const auto wrong_warp_cuda_caps = ggml_cuda_fattn_kvarn_select_capabilities({
        GGML_CUDA_FATTN_KVARN_BACKEND_CUDA, 64, false, true, 1024, 48*1024, 4*1024,
    });
    ok &= expect(!wrong_warp_cuda_caps.portable_native,
        "CUDA portable KVarN attention must require the physical 32-thread warp contract");

    const auto disabled_cuda_caps = ggml_cuda_fattn_kvarn_select_capabilities({
        GGML_CUDA_FATTN_KVARN_BACKEND_CUDA, 32, true, false, 1024, 48*1024, 4*1024,
    });
    ok &= expect(!disabled_cuda_caps.store_materialize &&
                 !disabled_cuda_caps.portable_native &&
                 !disabled_cuda_caps.specialized_routes,
        "a build without KVarN instances must advertise no native route");

    const auto rdna_caps = ggml_cuda_fattn_kvarn_select_capabilities({
        GGML_CUDA_FATTN_KVARN_BACKEND_HIP, 32, true, true, 1024, 48*1024, 4*1024,
    });
    ok &= expect(rdna_caps.generic_mma && rdna_caps.decode_split && !rdna_caps.decode_vector &&
                 rdna_caps.portable_native && rdna_caps.specialized_routes,
        "RDNA wave32 must expose generic WMMA, split decode, and portable fallback");

    const auto cdna_caps = ggml_cuda_fattn_kvarn_select_capabilities({
        GGML_CUDA_FATTN_KVARN_BACKEND_HIP, 64, true, true, 1024, 48*1024, 4*1024,
    });
    ok &= expect(cdna_caps.generic_mma && cdna_caps.decode_split && !cdna_caps.decode_vector &&
                 cdna_caps.portable_native && cdna_caps.specialized_routes,
        "CDNA wave64 must expose generic MFMA, split decode, and portable fallback");

    const auto old_amd_caps = ggml_cuda_fattn_kvarn_select_capabilities({
        GGML_CUDA_FATTN_KVARN_BACKEND_HIP, 32, false, true, 1024, 48*1024, 4*1024,
    });
    ok &= expect(!old_amd_caps.generic_mma && !old_amd_caps.decode_split &&
                 !old_amd_caps.decode_vector && old_amd_caps.portable_native &&
                 !old_amd_caps.specialized_routes,
        "AMD targets without WMMA/MFMA must remain portable-native");

    const auto musa_caps = ggml_cuda_fattn_kvarn_select_capabilities({
        GGML_CUDA_FATTN_KVARN_BACKEND_MUSA, 32, false, true, 1024, 48*1024, 4*1024,
    });
    ok &= expect(!musa_caps.generic_mma && !musa_caps.decode_split &&
                 !musa_caps.decode_vector && musa_caps.portable_native &&
                 !musa_caps.specialized_routes,
        "MUSA must remain explicitly portable instead of entering CUDA-only KVarN routes");
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
    const std::string kvarn_dispatch = slice_between(kvarn,
            "bool ggml_cuda_flash_attn_ext_kvarn(",
            "#endif // GGML_CUDA_KVARN");
    ok &= expect(kvarn_dispatch.find(
                     "#if defined(GGML_USE_HIP)\n    return ggml_cuda_flash_attn_ext_kvarn_portable") == std::string::npos &&
                 kvarn_dispatch.find("ggml_cuda_fattn_kvarn_device_capabilities") != std::string::npos &&
                 kvarn_dispatch.find("GGML_KVARN_TEST_FORCE_PORTABLE_FATTN") != std::string::npos,
        "HIP and CUDA must share capability-driven KVarN routing with a forced-portable override");
    const std::string cuda_native_ops = slice_between(cuda_backend,
            "static bool ggml_backend_cuda_kvarn_native_ops(",
            "static bool ggml_backend_cuda_kvarn_native_original_v(");
    ok &= expect(!cuda_native_ops.empty() &&
                 cuda_native_ops.find("turing_mma_available") == std::string::npos &&
                 cuda_native_ops.find("ggml_cuda_fattn_kvarn_device_capabilities") != std::string::npos,
        "CUDA backend native KVarN support must use route capabilities instead of requiring Turing MMA");
    ok &= expect(cuda_backend.find("\"ggml_backend_kvarn_capabilities\"") != std::string::npos,
        "CUDA must expose the versioned KVarN capability record through the backend registry");
    ok &= expect(kvarn.find("GGML_KVARN_TEST_FORCE_PORTABLE_CAPABILITY") != std::string::npos,
        "CUDA tests must be able to simulate a portable-only pre-Turing capability on current hardware");
    ok &= expect(fattn.find("ggml_cuda_flash_attn_ext_kvarn_direct_tail_supported") != std::string::npos &&
                 count_occurrences(fattn, "ggml_cuda_flash_attn_ext_kvarn_direct_tail_supported") == 2,
        "KVarN exact-tail execution and support reporting must share one direct-entry predicate");
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
    for (const char * target : {
            "50-real", "52-real", "53-real", "60-real", "61-real",
            "62-real", "70-real", "72-real", "75-real", "80-real",
            "86-real", "89-real", "90-real", "120a-real", "121a-real" }) {
        ok &= expect(architecture_compile.find(target) != std::string::npos,
            "explicit CUDA architecture compile coverage omitted a required target");
    }
    ok &= expect(
            architecture_compile.find("nvidia/cuda:12.4.1-devel-ubuntu22.04") != std::string::npos &&
            architecture_compile.find("nvidia/cuda:13.1.1-devel-ubuntu24.04") != std::string::npos &&
            architecture_compile.find("-DGGML_CUDA_KVARN=ON") != std::string::npos &&
            architecture_compile.find("-DGGML_CUDA_FA=ON") != std::string::npos &&
            architecture_compile.find("-DGGML_CUDA_FA_ALL_QUANTS=ON") != std::string::npos,
        "CUDA architecture compile coverage must retain both toolkit lanes and an all-quant KVarN probe");
    ok &= expect(hip_cmake.find("ggml_cuda_select_kvarn_fast_decode_sources") != std::string::npos &&
                 musa_cmake.find("ggml_cuda_select_kvarn_fast_decode_sources") != std::string::npos &&
                 cmake.find("GGML_CUDA_KVARN_ALL_PAIR_COUNT 36") != std::string::npos &&
                 cmake.find("_selected_pair_count EQUAL _expected_pair_count") != std::string::npos,
        "shared CUDA/HIP/MUSA configuration must validate the selected KVarN pair matrix");
    ok &= expect(count_occurrences(kvarn_mma,
                 "idx < nbatch_fa * dim2_count_local; idx += nthreads") >= 2,
        "rotated KVarN record reconstruction must distribute token/dimension pairs across the full CTA");
    ok &= expect(kvarn_mma.find("desc.value ? 3 * D : 0") != std::string::npos &&
                 kvarn_mma.find("axis_group_tags[axis_tag] != record_group") != std::string::npos,
        "rotated KVarN MMA must keep independent K/V record axes in shared memory and reuse them across subtiles");
    ok &= expect(kvarn_mma.find("const int row_pair_lane") != std::string::npos &&
                 count_occurrences(kvarn_mma, "ggml_cuda_fattn_kvarn_unpack_record_pair") >= 3,
        "rotated KVarN key reconstruction must decode adjacent token pairs cooperatively");
    ok &= expect(kvarn_mma_case.find("6 * std::max(DKQ, DV) * sizeof(half)") != std::string::npos,
        "rotated KVarN MMA shared memory must reserve all three axes for both K and V records");
    ok &= expect(kvarn.find("ggml_cuda_fattn_kvarn_wide_mma_supported") != std::string::npos &&
                 kvarn.find("ggml_cuda_flash_attn_ext_mma_kvarn_case<DKQ, DV, 16, 8>") != std::string::npos,
        "multi-token GQA verification must use the adaptive wide KVarN MMA tile when the device supports it");
    ok &= expect(kvarn_wide_instance.find("!defined(GGML_USE_HIP)") == std::string::npos &&
                 kvarn_wide_instance.find("!defined(GGML_USE_MUSA)") != std::string::npos,
        "wide KVarN MMA instances must compile for AMD HIP while MUSA stays portable");
    const std::string rdna_mma_config = slice_between(
        fattn_mma_f16, "ggml_cuda_fattn_mma_get_config_rdna", "ggml_cuda_fattn_mma_get_config_cdna");
    const std::string cdna_mma_config = slice_between(
        fattn_mma_f16, "ggml_cuda_fattn_mma_get_config_cdna", "static __host__ fattn_mma_config");
    ok &= expect(rdna_mma_config.find(
                     "GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 128, 256") != std::string::npos &&
                 rdna_mma_config.find(
                     "GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 128, 256") != std::string::npos &&
                 cdna_mma_config.find(
                     "GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 128, 512") != std::string::npos &&
                 cdna_mma_config.find(
                     "GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 128, 512") != std::string::npos,
        "wide KVarN MMA requires explicit 128-column RDNA/CDNA configurations");
    ok &= expect(kvarn_mma_case.find("cudaOccupancyMaxActiveBlocksPerMultiprocessor") != std::string::npos &&
                 kvarn_mma_case.find("smpbo") != std::string::npos,
        "wide KVarN MMA eligibility must be based on actual device resources and kernel occupancy");
    ok &= expect(kvarn_decode.find("ggml_cuda_get_physical_warp_size()") != std::string::npos &&
                 kvarn_decode.find("static_assert(WARP_SIZE == 32") == std::string::npos &&
                 kvarn_decode.find("devices[device].warp_size") != std::string::npos,
        "KVarN split decode must launch and index physical RDNA/CDNA waves");
    const std::string decode_softmax = slice_between(
        kvarn_decode, "half * p_h = (half *) p_sh;", "if (v_from_record)");
    ok &= expect(decode_softmax.find("#if defined(GGML_USE_HIP) && defined(CDNA)") != std::string::npos &&
                 decode_softmax.find("const int lane_h = tid % 16;") != std::string::npos &&
                 count_occurrences(decode_softmax, "__shfl_xor_sync") >= 2,
        "CDNA softmax fallback must not serialize the mature CUDA/RDNA subgroup path");
    ok &= expect(kvarn_wht.find("__shfl_xor_sync") != std::string::npos &&
                 kvarn_wht.find("ggml_cuda_get_physical_warp_size()") != std::string::npos &&
                 kvarn_wht.find("GGML_KVARN_TEST_FORCE_SHARED_WHT") != std::string::npos,
        "KVarN WHT must use physical-wave butterflies with a shared-memory oracle");
    ok &= expect(kvarn_dispatch_header.find("struct_size") != std::string::npos &&
                 kvarn_dispatch_header.find("abi_version") != std::string::npos &&
                 kvarn_dispatch_header.find("portable_native") != std::string::npos &&
                 kvarn_dispatch_header.find("amd_generic_mma") != std::string::npos &&
                 kvarn_dispatch_header.find("amd_decode_split") != std::string::npos &&
                 kvarn_dispatch_header.find("direct_entry") != std::string::npos &&
                 kvarn_dispatch_header.find("compact_tail_entry") != std::string::npos,
        "KVarN route telemetry must be ABI-sized and distinguish AMD, portable, and entry routes");
    ok &= expect(kvarn_store.find("headwide_workspace") != std::string::npos &&
                 kvarn_store.find("headwide_monolithic") != std::string::npos &&
                 kvarn_store.find("single_slice_workspace") != std::string::npos &&
                 kvarn_store.find("direct_store") != std::string::npos &&
                 kvarn_store.find("high_shared_fallback") != std::string::npos &&
                 kvarn_store.find("low_shared_store") != std::string::npos,
        "KVarN store routing must expose every high/low-LDS production path");

    return ok ? 0 : 1;
}
