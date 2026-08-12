#include "llama-kv-tail-request.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-kv-tail-request: %s\n", message);
        std::exit(1);
    }
}

static const std::vector<llama_kv_tail_request_group> hybrid_groups = {
    { "full@l0", "full", 100000, true },
    { "swa@l1",  "swa",    4096, true },
};

static llama_kv_tail_request_resolution resolve(
        const char * spec,
        ggml_type type = GGML_TYPE_COUNT,
        bool kvarn = false) {
    const auto request = llama_kv_tail_request_parse(spec, type);
    require(request.valid(), request.error.c_str());
    return llama_kv_tail_request_resolve(request, hybrid_groups, kvarn);
}

int main() {
    {
        const auto request = llama_kv_tail_request_parse("0", GGML_TYPE_COUNT);
        require(request.valid() && request.mode == LLAMA_KV_TAIL_REQUEST_NUMERIC,
                "numeric zero mode was not preserved");
        const auto result = llama_kv_tail_request_resolve(request, hybrid_groups, false);
        require(result.valid && result.groups[0].effective_tokens == 0 &&
                result.groups[1].effective_tokens == 0,
                "numeric zero did not disable every group");
    }
    {
        const auto result = resolve("auto");
        require(result.valid && result.groups.size() == 2,
                "automatic request did not bind both cache groups");
        for (const auto & group : result.groups) {
            require(group.raw_requested_tokens == 1024 &&
                    group.requested_tokens == 1024 &&
                    group.effective_tokens == 1024 &&
                    group.exact_type == GGML_TYPE_BF16,
                    "automatic standard-tail policy changed");
        }
    }
    {
        const auto result = resolve("1536", GGML_TYPE_F16);
        require(result.groups[0].effective_tokens == 1536 &&
                result.groups[1].effective_tokens == 1536 &&
                result.groups[0].exact_type == GGML_TYPE_F16,
                "uniform numeric request was not applied to every group");
    }
    {
        const auto result = resolve("128,512", GGML_TYPE_BF16);
        require(result.groups[0].raw_requested_tokens == 128 &&
                result.groups[1].raw_requested_tokens == 512 &&
                result.groups[1].exact_type == GGML_TYPE_BF16,
                "positional request lost group values or exact type");
    }
    {
        const auto result = resolve("swa=512,full=128");
        require(result.groups[0].raw_requested_tokens == 128 &&
                result.groups[1].raw_requested_tokens == 512,
                "named role aliases did not bind canonical groups");
    }
    {
        const auto result = resolve("full@l0=256,swa@l1=768");
        require(result.groups[0].raw_requested_tokens == 256 &&
                result.groups[1].raw_requested_tokens == 768,
                "canonical named groups did not bind independently");
    }
    {
        const auto result = resolve("0", GGML_TYPE_COUNT, true);
        for (const auto & group : result.groups) {
            require(group.raw_requested_tokens == 0 &&
                    group.requested_tokens == 128 &&
                    group.effective_tokens == 128 &&
                    group.exact_type == GGML_TYPE_F16,
                    "KVarN intrinsic exact tail was absent from model-bound resolution");
        }
    }
    {
        const auto result = resolve("64", GGML_TYPE_COUNT, true);
        require(result.groups[0].raw_requested_tokens == 64 &&
                result.groups[0].requested_tokens == 128 &&
                result.groups[0].effective_tokens == 128,
                "KVarN request below the intrinsic minimum was not raised to 128");
    }
    {
        const auto result = resolve("129", GGML_TYPE_BF16, true);
        require(result.groups[0].raw_requested_tokens == 129 &&
                result.groups[0].requested_tokens == 256 &&
                result.groups[0].effective_tokens == 256 &&
                result.groups[0].exact_type == GGML_TYPE_BF16,
                "KVarN explicit request did not retain provenance, rounding, or type");
    }
    {
        const auto result = resolve("auto", GGML_TYPE_COUNT, true);
        require(result.groups[0].requested_tokens == 1024 &&
                result.groups[1].requested_tokens == 1024 &&
                result.groups[0].exact_type == GGML_TYPE_F16,
                "KVarN automatic request did not use the shared 1024-token policy");
    }
    {
        const auto request = llama_kv_tail_request_parse("128", GGML_TYPE_F32);
        require(!request.valid(), "unsupported exact type was accepted");
    }
    {
        const auto request = llama_kv_tail_request_parse("full=128,512", GGML_TYPE_COUNT);
        require(!request.valid(), "mixed named and positional syntax was accepted by the request parser");
    }
    {
        const auto request = llama_kv_tail_request_parse("128,512", GGML_TYPE_COUNT);
        const auto result = llama_kv_tail_request_resolve(
                request, { hybrid_groups.front() }, false);
        require(!result.valid, "positional group-count mismatch was accepted");
    }
    {
        const auto request = llama_kv_tail_request_parse("missing=128,swa=512", GGML_TYPE_COUNT);
        const auto result = llama_kv_tail_request_resolve(request, hybrid_groups, false);
        require(!result.valid, "unknown named cache group was accepted");
    }

    std::puts("test-kv-tail-request: all tests OK");
    return 0;
}
