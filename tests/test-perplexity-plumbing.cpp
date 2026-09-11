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
    const std::string perplexity = read_file(root + "/tools/perplexity/perplexity.cpp");
    const std::string ppl = slice_between(perplexity,
            "static results_perplexity perplexity(llama_context * ctx, const common_params & params, const int32_t n_ctx)",
            "static bool decode_helper");
    const std::string kl = slice_between(perplexity,
            "static bool kl_divergence(llama_context * ctx, const common_params & params)",
            "if (kld.count < 100) return true;");

    ok &= expect(perplexity.find("static int ppl_max_logits_rows(int n_vocab, const common_params & params)") != std::string::npos,
        "perplexity must cap full-vocab logits rows to avoid multi-GiB output buffers");
    ok &= expect(ppl.find("const int max_logits_rows = params.logits_file.empty()") != std::string::npos &&
                 ppl.find(": std::max(1, std::min(n_ctx, params.n_batch))") != std::string::npos,
        "logits baseline generation must bypass the ordinary perplexity row cap");
    ok &= expect(perplexity.find("logits_stream.write(\"_logits_\", 8)") != std::string::npos,
        "perplexity must write upstream-compatible v1 logits baselines");
    ok &= expect(perplexity.find("kld_logits::") == std::string::npos,
        "perplexity must not use the fork-specific v2 logits format");
    ok &= expect(perplexity.find("min_logit = std::max(min_logit, max_logit - 16)") != std::string::npos,
        "perplexity must retain the upstream max-16 baseline encoding");
    ok &= expect(perplexity.find("if (p_log_base > -16.f)") != std::string::npos,
        "KLD must retain the upstream v1 probability cutoff");
    ok &= expect(perplexity.find("if (!kl_divergence(ctx, params))") != std::string::npos,
        "perplexity KL failures must propagate to a nonzero process exit");
    ok &= expect(kl.find("ppl_max_logits_rows") == std::string::npos,
        "KL divergence must not silently cap the requested batch by logits memory");
    ok &= expect(kl.find("const int n_batch = std::max(1, std::min(n_ctx_i, params.n_batch))") != std::string::npos,
        "KL divergence batch size must honor the requested batch size");
    ok &= expect(kl.find("llama_batch_init(n_batch, 0, 1)") != std::string::npos,
        "KL divergence batch allocation must match bounded n_batch");
    ok &= expect(kl.find("std::vector<uint16_t> log_probs_uint16(size_t(n_batch) * nv)") != std::string::npos,
        "KL divergence base-logit buffer must match the decode batch size");
    ok &= expect(kl.find("const int logits_first = std::max(first, pos_start)") != std::string::npos &&
                 kl.find("const int logits_end   = std::min(n_ctx_i - 1, pos_start + batch_size)") != std::string::npos,
        "KL divergence must process only the logits rows produced by the current decode slice");
    ok &= expect(kl.find("in.read((char *)log_probs_uint16.data(), log_probs_size)") != std::string::npos,
        "KL divergence must stream base log-prob rows per decode slice");
    ok &= expect(kl.find("process_logits(n_vocab, batch_logits, tokens.data() + start + logits_first, n_outputs") != std::string::npos,
        "KL divergence must consume current decode logits directly");
    ok &= expect(kl.find("std::vector<float> logits") == std::string::npos,
        "KL divergence must not buffer full-vocab logits across the half-context");

    return ok ? 0 : 1;
}
