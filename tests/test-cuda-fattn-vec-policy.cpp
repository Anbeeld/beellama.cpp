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

    const std::string all_branch = slice_between(dispatch,
            "#if defined(GGML_CUDA_FA_ALL_QUANTS)",
            "#else");
    const std::string default_branch = slice_between(dispatch,
            "#else",
            "#endif");

    ok &= expect(generator.find("assert len(TYPES) == 13") != std::string::npos &&
                 generator.find("assert len(pairs_default) == 103") != std::string::npos &&
                 generator.find("assert len(pairs_all) == 169") != std::string::npos,
        "the vector-pair generator must assert the 13-type, 103-default, and 169-ALL policy counts");
    ok &= expect(generator.find("rank_k <= rank_v or type_k == \"GGML_TYPE_F16\" or type_v == \"GGML_TYPE_F16\"") != std::string::npos,
        "the default vector-pair policy must preserve the F16 materialization axes");
    ok &= expect(generator.find("GGML_TYPE_Q2_0S") != std::string::npos,
        "the surviving fork q2 cache type must be q2_0s, not the upstream q2_0 type");

    ok &= expect(!all_branch.empty() && !default_branch.empty(),
        "the generated vector dispatch must have distinct ALL and default branches");
    ok &= expect(count_occurrences(all_branch, "FATTN_VEC_CASES_ALL_D(") == 169,
        "the ALL vector dispatch must include every ordered pair of the 13 retained types");
    ok &= expect(count_occurrences(default_branch, "FATTN_VEC_CASES_ALL_D(") == 103,
        "the default vector dispatch must include exactly 103 pairs");
    ok &= expect(dispatch.find("GGML_CUDA_FA_HALF_QUANTS") == std::string::npos,
        "the removed HALF build tier must not survive in vector dispatch");

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
