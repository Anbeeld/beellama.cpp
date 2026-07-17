#include "../tools/perplexity/kld-logits.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

static bool expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
    }
    return condition;
}

int main() {
    const std::vector<float> logits = { 0.0f, -1.0f, -20.0f, -40.0f, -INFINITY };
    const int token = 3;
    std::vector<uint16_t> encoded(kld_logits::encoded_row_size(logits.size()));

    const double expected_nll = kld_logits::encode_row(
            logits.size(), logits.data(), encoded.data(), token);
    const kld_logits::decoded_row row =
            kld_logits::decode_row(logits.size(), encoded.data(), true);

    double probability_sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probability_sum += std::exp(kld_logits::log_probability(row, i));
    }

    const double decoded_nll =
            -kld_logits::log_probability(row, token);
    bool ok = true;
    ok &= expect(std::fabs(probability_sum - 1.0) < 1e-6,
            "decoded v2 probabilities are not normalized");
    ok &= expect(std::fabs(decoded_nll - expected_nll) < 1e-3,
            "v2 correct-token NLL does not survive encode/decode");

    // The legacy encoder collapsed every logit below max-16 to the same value.
    // V2 must retain distinct tail probabilities so baseline PPL and KLD use
    // the complete distribution.
    const float log_p_2 = kld_logits::log_probability(row, 2);
    const float log_p_3 = kld_logits::log_probability(row, 3);
    ok &= expect(log_p_2 > log_p_3 + 10.0f,
            "v2 encoder collapsed distinct tail probabilities");
    ok &= expect(std::isinf(kld_logits::log_probability(row, 4)) &&
                 kld_logits::log_probability(row, 4) < 0.0f,
            "v2 encoder did not preserve an exact zero probability");

    return ok ? 0 : 1;
}
