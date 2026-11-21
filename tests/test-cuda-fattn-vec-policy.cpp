#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#ifdef TEST_CUDA_COMPILED_PAIRS
#include "ggml.h"
bool ggml_cuda_fa_pair_compiled(ggml_type type_k, ggml_type type_v);
#endif

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
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

int main(int argc, char ** argv) {
    bool ok = true;

    ok &= expect(argc == 2, "expected repo root argument");
    if (!ok) {
        return 1;
    }

    const std::string root = argv[1];
    const std::string vec = read_file(root + "/ggml/src/ggml-cuda/fattn-vec.cuh");
    const std::string cmake = read_file(root + "/ggml/CMakeLists.txt");
    const std::string instances = read_file(root + "/ggml/cmake/common.cmake");
    const std::string fattn = read_file(root + "/ggml/src/ggml-cuda/fattn.cu");

    const std::string dispatch = slice_between(fattn,
            "static fattn_vec_case_t ggml_cuda_get_fattn_vec_case",
            "static ggml_type ggml_cuda_fattn_canonical_kv_type");
    ok &= expect(!dispatch.empty() && count_occurrences(dispatch, "FATTN_VEC_CASES_ALL_D(") == 196,
        "the runtime vector dispatch must enumerate all 196 ordered pairs of the 14 retained types");
    ok &= expect(dispatch.find("FATTN_VEC_CASES_ALL_D(F16   , F16)") != std::string::npos &&
                 dispatch.find("FATTN_VEC_CASES_ALL_D(BF16  , BF16)") != std::string::npos,
        "the full runtime vector dispatch must retain homogeneous F16 and BF16 instances");
    ok &= expect(dispatch.find("Q2_0S") != std::string::npos && dispatch.find("GGML_TYPE_Q2_0,") == std::string::npos,
        "the fork low-bit vector type must be Q2_0S, distinct from upstream Q2_0");
    ok &= expect(cmake.find("list(LENGTH _pairs _pair_count)") != std::string::npos &&
                 cmake.find("Default CUDA FA vec policy expected 51 pairs") != std::string::npos &&
                 cmake.find("foreach(PAIR ${GGML_CUDA_KVARN_DEFAULT_PAIRS})") != std::string::npos,
        "CMake must derive and enforce the 51-pair default policy from KVarN default pairs plus FP8");
    ok &= expect(cmake.find("set(_pairs f16:f16 bf16:bf16 f8_e4m3:f8_e4m3)") != std::string::npos &&
                 instances.find("ggml_cuda_get_fattn_vec_default_pairs(DEFAULT_PAIRS)") != std::string::npos &&
                 instances.find("if (GGML_CUDA_FA_ALL_QUANTS)") != std::string::npos &&
                 instances.find("foreach (TYPE_V IN LISTS FA_TYPES)") != std::string::npos,
        "the compiled default and ALL modes must use the validated CMake pair lists");
    ok &= expect(fattn.find("#if defined(GGML_CUDA_FA_ALL_QUANTS)") != std::string::npos &&
                 fattn.find("return true;", fattn.find("#if defined(GGML_CUDA_FA_ALL_QUANTS)")) != std::string::npos &&
                 fattn.find("ggml_cuda_fattn_default_quant_pair(type_K, type_V)") != std::string::npos,
        "runtime compiled-pair gate must select all pairs in ALL mode and enforce the default quant policy otherwise");


    const int group_sizes[] = { 1, 2, 2, 2, 2, 2 };
    const int bit_pairs[][2] = {
        { 0, 0 }, { 0, 1 }, { 0, 2 }, { 1, 1 }, { 1, 2 }, { 1, 3 },
        { 2, 2 }, { 2, 3 }, { 2, 4 }, { 3, 3 }, { 3, 4 }, { 3, 5 },
        { 4, 4 }, { 4, 5 }, { 5, 5 },
    };
    size_t expected_quant_pairs = 0;
    for (const auto & bit_pair : bit_pairs) {
        const int k_group = bit_pair[0];
        const int v_group = bit_pair[1];
        for (int k_variant = 0; k_variant < group_sizes[k_group]; ++k_variant) {
            for (int v_variant = 0; v_variant < group_sizes[v_group]; ++v_variant) {
                if (k_group == v_group && k_variant > v_variant) continue;
                ++expected_quant_pairs;
            }
        }
    }
    ok &= expect(expected_quant_pairs == 48,
        "the KVarN-rule quant matrix must contain exactly 48 pairs");
    const std::string pair_list = slice_between(cmake, "set(GGML_CUDA_KVARN_DEFAULT_PAIRS", ")");
    for (const auto & bit_pair : bit_pairs) {
        const int k_bits = 8 - (bit_pair[0] == 0 ? 0 : bit_pair[0] + 1);
        const int v_bits = 8 - (bit_pair[1] == 0 ? 0 : bit_pair[1] + 1);
        const std::string pair = "k" + std::to_string(k_bits) + "-v" + std::to_string(v_bits);
        ok &= expect(pair_list.find(pair) != std::string::npos,
            "CMake default policy is missing a required KVarN bit pair");
    }
    const char * group_types[] = {
        "set(_types_8 q8_0)", "set(_types_6 q6_1 q6_0)",
        "set(_types_5 q5_1 q5_0)", "set(_types_4 q4_1 q4_0)",
        "set(_types_3 q3_1 q3_0)", "set(_types_2 q2_1 q2_0s)",
    };
    for (const char * types : group_types) {
        ok &= expect(cmake.find(types) != std::string::npos,
            "CMake default policy is missing an ordered quant type group");
    }
    ok &= expect(cmake.find("K_BITS EQUAL V_BITS AND K_INDEX GREATER V_INDEX") != std::string::npos &&
                 cmake.find("list(APPEND _pairs \"${K_TYPE}:${V_TYPE}\")") != std::string::npos,
        "CMake must emit only the selected ordered quant variants");
    ok &= expect(fattn.find("GGML_CUDA_FA_HALF_QUANTS") == std::string::npos &&
                 cmake.find("GGML_CUDA_FA_HALF_QUANTS") == std::string::npos,
        "the removed HALF build tier must not survive in the CUDA FA policy");

    ok &= expect(vec.find("static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device()") != std::string::npos,
        "the vector kernel helper section must remain present");
    ok &= expect(vec.find("GGML_TYPE_TURBO") == std::string::npos &&
                 vec.find("TCQ") == std::string::npos,
        "the vector kernel must not retain TurboQuant or TCQ cache handling");
    ok &= expect(vec.find("GGML_TYPE_Q2_0S") != std::string::npos &&
                 vec.find("GGML_TYPE_Q6_1") != std::string::npos,
        "the vector kernel must retain declarations for the fork low-bit cache types");

#ifdef TEST_CUDA_COMPILED_PAIRS
    struct cache_type { ggml_type type; int bits; int variant; };
    const cache_type types[] = {
        {GGML_TYPE_F16, 0, 0}, {GGML_TYPE_BF16, 0, 0}, {GGML_TYPE_F8_E4M3, 0, 0},
        {GGML_TYPE_Q8_0, 8, 0}, {GGML_TYPE_Q6_1, 6, 0}, {GGML_TYPE_Q6_0, 6, 1},
        {GGML_TYPE_Q5_1, 5, 0}, {GGML_TYPE_Q5_0, 5, 1},
        {GGML_TYPE_Q4_1, 4, 0}, {GGML_TYPE_Q4_0, 4, 1},
        {GGML_TYPE_Q3_1, 3, 0}, {GGML_TYPE_Q3_0, 3, 1},
        {GGML_TYPE_Q2_1, 2, 0}, {GGML_TYPE_Q2_0S, 2, 1},
    };
    const int allowed_bits[][2] = {
        {8, 8}, {8, 6}, {8, 5}, {6, 6}, {6, 5}, {6, 4},
        {5, 5}, {5, 4}, {5, 3}, {4, 4}, {4, 3}, {4, 2},
        {3, 3}, {3, 2}, {2, 2},
    };
    size_t n_compiled = 0;
    for (const auto & k : types) {
        for (const auto & v : types) {
            bool expected = false;
#ifdef TEST_CUDA_ALL_PAIRS
            expected = true;
#else
            if (k.bits == 0 || v.bits == 0) {
                expected = k.type == v.type;
            } else {
                for (const auto & pair : allowed_bits) {
                    expected |= pair[0] == k.bits && pair[1] == v.bits &&
                        (k.bits != v.bits || k.variant <= v.variant);
                }
            }
#endif
            const bool actual = ggml_cuda_fa_pair_compiled(k.type, v.type);
            n_compiled += actual;
            if (actual != expected) {
                std::fprintf(stderr, "compiled pair mismatch: %s/%s expected=%d actual=%d\n",
                        ggml_type_name(k.type), ggml_type_name(v.type), expected, actual);
                ok = false;
            }
        }
    }
#ifdef TEST_CUDA_ALL_PAIRS
    ok &= expect(n_compiled == 196, "expanded build must advertise exactly its 196 compiled pairs");
#else
    ok &= expect(n_compiled == 51, "default build must advertise exactly its 51 compiled pairs");
#endif
    ok &= expect(ggml_cuda_fa_pair_compiled(GGML_TYPE_F32, GGML_TYPE_F32),
        "F32 must retain its F16 vector-conversion alias");
    ok &= expect(!ggml_cuda_fa_pair_compiled(GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL),
        "uninstantiated IQ4_NL must not be advertised as a compiled vector pair");
#endif
    return ok ? 0 : 1;
}
