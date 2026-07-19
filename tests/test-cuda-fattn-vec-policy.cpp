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
    const std::string generator = read_file(root + "/scripts/gen-fattn-vec-dispatch.py");
    const std::string dispatch = read_file(root + "/ggml/src/ggml-cuda/fattn-vec-dispatch.cuh");
    const std::string vec = read_file(root + "/ggml/src/ggml-cuda/fattn-vec.cuh");
    const std::string cmake = read_file(root + "/ggml/CMakeLists.txt");
    const std::string fattn = read_file(root + "/ggml/src/ggml-cuda/fattn.cu");

    const std::string all_branch = slice_between(dispatch,
            "#if defined(GGML_CUDA_FA_ALL_QUANTS)",
            "#else");
    const std::string default_branch = slice_between(dispatch,
            "#else",
            "#endif");

    ok &= expect(generator.find("assert len(TYPES) == 13") != std::string::npos &&
                 generator.find("assert len(pairs_default) == 50") != std::string::npos &&
                 generator.find("assert len(pairs_all) == 169") != std::string::npos,
        "the vector-pair generator must assert the 13-type, 50-default, and 169-ALL policy counts");
    ok &= expect(generator.find("KVARN_DEFAULT_BIT_PAIRS") != std::string::npos,
        "the default vector-pair policy must use the KVarN default bit-pair rules");
    ok &= expect(generator.find("GGML_TYPE_Q2_0S") != std::string::npos,
        "the surviving fork q2 cache type must be q2_0s, not the upstream q2_0 type");

    ok &= expect(!all_branch.empty() && !default_branch.empty(),
        "the generated vector dispatch must have distinct ALL and default branches");
    ok &= expect(count_occurrences(all_branch, "FATTN_VEC_CASES_ALL_D(") == 169,
        "the ALL vector dispatch must include every ordered pair of the 13 retained types");
    ok &= expect(count_occurrences(default_branch, "FATTN_VEC_CASES_ALL_D(") == 50,
        "the default vector dispatch must include exactly 50 pairs");
    ok &= expect(default_branch.find("FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16, GGML_TYPE_F16)") != std::string::npos &&
                 default_branch.find("FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)") != std::string::npos,
        "the default vector dispatch must retain the homogeneous F16 and BF16 tail pairs");
    ok &= expect(default_branch.find("FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16, GGML_TYPE_Q") == std::string::npos &&
                 default_branch.find("FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)") == std::string::npos &&
                 default_branch.find("FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16, GGML_TYPE_BF16)") == std::string::npos &&
                 default_branch.find("FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)") == std::string::npos,
        "the default vector dispatch must not retain mixed float/quant or mixed float tail pairs");

    const char * groups[][2] = {
        { "GGML_TYPE_Q8_0", nullptr },
        { "GGML_TYPE_Q6_1", "GGML_TYPE_Q6_0" },
        { "GGML_TYPE_Q5_1", "GGML_TYPE_Q5_0" },
        { "GGML_TYPE_Q4_1", "GGML_TYPE_Q4_0" },
        { "GGML_TYPE_Q3_1", "GGML_TYPE_Q3_0" },
        { "GGML_TYPE_Q2_1", "GGML_TYPE_Q2_0S" },
    };
    const int group_sizes[] = { 1, 2, 2, 2, 2, 2 };
    const int bit_pairs[][2] = {
        { 0, 0 }, { 0, 1 }, { 0, 2 },
        { 1, 1 }, { 1, 2 }, { 1, 3 },
        { 2, 2 }, { 2, 3 }, { 2, 4 },
        { 3, 3 }, { 3, 4 }, { 3, 5 },
        { 4, 4 }, { 4, 5 },
        { 5, 5 },
    };
    size_t expected_quant_pairs = 0;
    for (const auto & bit_pair : bit_pairs) {
        const int k_group = bit_pair[0];
        const int v_group = bit_pair[1];
        for (int k_variant = 0; k_variant < group_sizes[k_group]; ++k_variant) {
            for (int v_variant = 0; v_variant < group_sizes[v_group]; ++v_variant) {
                if (k_group == v_group && k_variant > v_variant) {
                    continue;
                }
                const std::string expected = "FATTN_VEC_CASES_ALL_D(" + std::string(groups[k_group][k_variant]) +
                    ", " + groups[v_group][v_variant] + ")";
                ok &= expect(default_branch.find(expected) != std::string::npos,
                    "the default vector dispatch is missing a KVarN-rule quant pair");
                ++expected_quant_pairs;
            }
        }
    }
    ok &= expect(expected_quant_pairs == 48,
        "the KVarN-rule quant matrix must contain exactly 48 pairs");
    ok &= expect(dispatch.find("GGML_CUDA_FA_HALF_QUANTS") == std::string::npos,
        "the removed HALF build tier must not survive in vector dispatch");

    ok &= expect(cmake.find("foreach(PAIR ${GGML_CUDA_KVARN_DEFAULT_PAIRS})") != std::string::npos &&
                 cmake.find("Default CUDA FA vec policy expected 50 pairs") != std::string::npos,
        "CMake must derive the 50 default vector pairs from the KVarN default pairs");
    ok &= expect(fattn.find("ggml_cuda_fattn_default_quant_pair") != std::string::npos,
        "the runtime compiled-pair check must implement the same KVarN-rule quant policy");

    ok &= expect(vec.find("static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device()") != std::string::npos,
        "the vector kernel helper section must remain present");
    ok &= expect(vec.find("GGML_TYPE_TURBO") == std::string::npos &&
                 vec.find("TCQ") == std::string::npos,
        "the vector kernel must not retain TurboQuant or TCQ cache handling");
    ok &= expect(vec.find("GGML_TYPE_Q2_0S") != std::string::npos &&
                 vec.find("GGML_TYPE_Q6_1") != std::string::npos,
        "the vector kernel must retain declarations for the fork low-bit cache types");

    return ok ? 0 : 1;
}
