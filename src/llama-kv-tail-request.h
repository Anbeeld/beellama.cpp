#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

enum llama_kv_tail_request_mode {
    LLAMA_KV_TAIL_REQUEST_DISABLED,
    LLAMA_KV_TAIL_REQUEST_NUMERIC,
    LLAMA_KV_TAIL_REQUEST_AUTOMATIC,
    LLAMA_KV_TAIL_REQUEST_POSITIONAL,
    LLAMA_KV_TAIL_REQUEST_NAMED,
};

struct llama_kv_tail_request_entry {
    std::string group;
    uint32_t tokens;
};

struct llama_kv_tail_request {
    llama_kv_tail_request_mode mode = LLAMA_KV_TAIL_REQUEST_DISABLED;
    ggml_type exact_type = GGML_TYPE_COUNT;
    std::vector<llama_kv_tail_request_entry> entries;
    std::string error;

    bool valid() const { return error.empty(); }
};

struct llama_kv_tail_request_group {
    std::string id;
    std::string role;
    uint32_t effective_window;
    bool applicable_standard_kv;
};

struct llama_kv_tail_request_group_resolution {
    std::string id;
    std::string role;
    uint32_t raw_requested_tokens;
    uint32_t requested_tokens;
    uint32_t effective_tokens;
    ggml_type exact_type;
    bool native_exact;
};

struct llama_kv_tail_request_resolution {
    bool valid = false;
    std::vector<llama_kv_tail_request_group_resolution> groups;
    std::string error;
};

llama_kv_tail_request llama_kv_tail_request_parse(
        const std::string & specification,
        ggml_type exact_type);

llama_kv_tail_request_resolution llama_kv_tail_request_resolve(
        const llama_kv_tail_request & request,
        const std::vector<llama_kv_tail_request_group> & groups,
        bool kvarn);
