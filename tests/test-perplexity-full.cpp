// Link with llama-perplexity-impl. No model or backend initialization is needed.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <limits>
#include <string>
#include <filesystem>
#include <fstream>

bool perplexity_full_compare_batch(int, const float *, const uint16_t *, const int *, int, int, std::array<double,4> *);
bool perplexity_full_same_file(const std::string &, const std::string &);

double perplexity_full_encode(int n_vocab, const float * logits, uint16_t * encoded, int target);
std::array<double, 4> perplexity_full_compare(int n_vocab, const float * logits, const uint16_t * encoded, int target);

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

int main() {
    require(!perplexity_full_same_file("", ""), "empty output paths must not alias");
    require(perplexity_full_same_file("ppl-path-test.txt", "./ppl-path-test.txt"), "lexical file alias missed");
    require(!perplexity_full_same_file("ppl-path-test.txt", "ppl-other-path.txt"), "distinct paths alias");
#ifdef _WIN32
    require(perplexity_full_same_file("NUL", "nul"), "native null devices must match without canonicalization");
    require(!perplexity_full_same_file("nul", "ppl-path-test.txt"), "native null device aliases regular file");
#endif
    const auto original = std::filesystem::current_path()/"ppl-path-original.fixture";
    const auto linked = std::filesystem::current_path()/"ppl-path-hardlink.fixture";
    require(!std::filesystem::exists(original) && !std::filesystem::exists(linked), "stale alias test fixtures");
    { std::ofstream file(original); file << "preserved input"; require(bool(file), "alias fixture write"); }
    std::filesystem::create_hard_link(original, linked);
    require(perplexity_full_same_file(original.string(),linked.string()), "hard-link alias protection lost");
    std::filesystem::remove(linked); std::filesystem::remove(original);
    std::vector<uint16_t> encoded(8);
    const float uniform[] = {0, 0, 0};
    require(std::abs(perplexity_full_encode(3, uniform, encoded.data(), 2) - std::log(3.0)) < 1e-6,
            "uniform reference NLL");
    auto result = perplexity_full_compare(3, uniform, encoded.data(), 2);
    require(std::abs(result[0]) < 1e-6 && result[3] == 1, "uniform self KL and top1");
    require(std::abs(result[1] - std::log(3.0)) < 1e-6 && std::abs(result[2] - std::log(3.0)) < 1e-6,
            "uniform base/candidate NLL");

    // Reference tail probabilities below exp(-16) must still contribute to KL.
    const float reference[] = {0, -17, -20};
    const float candidate[] = {0, -30, -40};
    const double z_ref = 1 + std::exp(-17.0) + std::exp(-20.0);
    const double z_candidate = 1 + std::exp(-30.0) + std::exp(-40.0);
    const double expected = std::log(z_candidate / z_ref) +
        (13 * std::exp(-17.0) + 20 * std::exp(-20.0)) / z_ref;
    perplexity_full_encode(3, reference, encoded.data(), 2);
    result = perplexity_full_compare(3, candidate, encoded.data(), 2);
    require(std::abs(result[0] - expected) < 1e-9, "KL must include the full vocabulary tail");
    require(std::abs(result[1] - (20 + std::log(z_ref))) < 1e-5, "no reference clipping at 16");
    result = perplexity_full_compare(3, reference, encoded.data(), 2);
    require(std::abs(result[0]) < 1e-9, "normalized compact-reference self KL floor");

    const float shifted[] = {50, 33, 30};
    result = perplexity_full_compare(3, shifted, encoded.data(), 2);
    require(std::abs(result[0]) < 1e-9 && std::abs(result[2] - (20 + std::log(z_ref))) < 1e-6,
            "candidate softmax is shift invariant");
    const float different_top[] = {-2, 0, -1};
    result = perplexity_full_compare(3, different_top, encoded.data(), 1);
    require(result[3] == 0, "top1 disagreement");
    const float even_reference[] = {-4, -3, 0, -2};
    perplexity_full_encode(4, even_reference, encoded.data(), 3);
    result = perplexity_full_compare(4, even_reference, encoded.data(), 3);
    require(std::abs(result[0]) < 1e-8 && result[3] == 1, "even-vocabulary compact row self KL");
    std::vector<float> batch_logits = {0,-30,-40, 50,33,30};
    std::vector<uint16_t> batch_reference(16);
    perplexity_full_encode(3,reference,batch_reference.data(),2);
    perplexity_full_encode(3,reference,batch_reference.data()+8,1);
    const int targets[] = {2,1};
    std::array<double,4> batch_result[2];
    for (int threads : {1,4}) {
        require(perplexity_full_compare_batch(3,batch_logits.data(),batch_reference.data(),targets,2,threads,batch_result),
                "parallel validated comparison rejected finite rows");
        for (int row=0;row<2;++row) {
            require(batch_result[row] == perplexity_full_compare(3,batch_logits.data()+3*row,batch_reference.data()+8*row,targets[row]),
                    "parallel comparison changed any scalar reference result");
        }
    }
    for (int column=0;column<6;++column) {
        const float saved=batch_logits[column];
        batch_logits[column]=std::numeric_limits<float>::quiet_NaN();
        require(!perplexity_full_compare_batch(3,batch_logits.data(),batch_reference.data(),targets,2,4,batch_result),
                "parallel validation missed a NaN in a candidate row");
        batch_logits[column]=std::numeric_limits<float>::infinity();
        require(!perplexity_full_compare_batch(3,batch_logits.data(),batch_reference.data(),targets,2,4,batch_result),
                "parallel validation missed infinity in a candidate row");
        batch_logits[column]=saved;
    }
    std::puts("test-perplexity-full: OK");
}
