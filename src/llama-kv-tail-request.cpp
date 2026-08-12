#include "llama-kv-tail-request.h"

#include "llama-kv-cache-kvarn.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <unordered_set>

static bool parse_count(const std::string & value, uint32_t & result) {
    if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char ch) {
            return ch >= '0' && ch <= '9';
        })) {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed > UINT32_MAX) {
        return false;
    }
    result = uint32_t(parsed);
    return true;
}

static std::vector<std::string> split_entries(const std::string & specification) {
    std::vector<std::string> result;
    size_t begin = 0;
    while (begin <= specification.size()) {
        const size_t end = specification.find(',', begin);
        result.push_back(specification.substr(
                begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return result;
}

llama_kv_tail_request llama_kv_tail_request_parse(
        const std::string & specification,
        ggml_type exact_type) {
    llama_kv_tail_request result;
    result.exact_type = exact_type;
    if (exact_type != GGML_TYPE_COUNT && exact_type != GGML_TYPE_F16 && exact_type != GGML_TYPE_BF16) {
        result.error = "KV tail type must be F16, BF16, or the cache-family default";
        return result;
    }
    if (specification.empty()) {
        result.mode = LLAMA_KV_TAIL_REQUEST_DISABLED;
        return result;
    }
    if (specification == "auto") {
        result.mode = LLAMA_KV_TAIL_REQUEST_AUTOMATIC;
        return result;
    }

    const auto entries = split_entries(specification);
    const bool any_named = std::any_of(entries.begin(), entries.end(), [](const auto & entry) {
        return entry.find('=') != std::string::npos;
    });
    const bool all_named = std::all_of(entries.begin(), entries.end(), [](const auto & entry) {
        const size_t first = entry.find('=');
        return first != std::string::npos && first == entry.rfind('=');
    });
    if (any_named && !all_named) {
        result.error = "KV tail request cannot mix named and positional groups";
        return result;
    }

    result.mode = any_named ? LLAMA_KV_TAIL_REQUEST_NAMED :
            entries.size() > 1 ? LLAMA_KV_TAIL_REQUEST_POSITIONAL : LLAMA_KV_TAIL_REQUEST_NUMERIC;
    std::unordered_set<std::string> names;
    for (const auto & entry : entries) {
        const size_t equal = entry.find('=');
        const std::string name = any_named ? entry.substr(0, equal) : std::string();
        const std::string value = any_named ? entry.substr(equal + 1) : entry;
        uint32_t tokens = 0;
        if (!parse_count(value, tokens)) {
            result.error = "invalid KV tail token count: " + value;
            return result;
        }
        if (any_named && (name.empty() || !names.insert(name).second)) {
            result.error = name.empty() ? "empty KV tail group name" :
                    "duplicate KV tail group assignment: " + name;
            return result;
        }
        result.entries.push_back({ name, tokens });
    }
    return result;
}

llama_kv_tail_request_resolution llama_kv_tail_request_resolve(
        const llama_kv_tail_request & request,
        const std::vector<llama_kv_tail_request_group> & groups,
        bool kvarn) {
    llama_kv_tail_request_resolution result;
    if (!request.valid()) {
        result.error = request.error;
        return result;
    }
    std::vector<uint32_t> raw(groups.size(), 0);
    switch (request.mode) {
        case LLAMA_KV_TAIL_REQUEST_DISABLED:
            break;
        case LLAMA_KV_TAIL_REQUEST_NUMERIC:
            if (request.entries.size() != 1) {
                result.error = "numeric KV tail request has no value";
                return result;
            }
            std::fill(raw.begin(), raw.end(), request.entries.front().tokens);
            break;
        case LLAMA_KV_TAIL_REQUEST_AUTOMATIC:
            std::fill(raw.begin(), raw.end(), 1024);
            break;
        case LLAMA_KV_TAIL_REQUEST_POSITIONAL:
            if (request.entries.size() != groups.size()) {
                result.error = "KV tail positional group count mismatch: expected " +
                        std::to_string(groups.size()) + ", got " +
                        std::to_string(request.entries.size());
                return result;
            }
            for (size_t i = 0; i < groups.size(); ++i) {
                raw[i] = request.entries[i].tokens;
            }
            break;
        case LLAMA_KV_TAIL_REQUEST_NAMED: {
            std::vector<bool> assigned(groups.size(), false);
            for (const auto & entry : request.entries) {
                std::vector<size_t> matches;
                for (size_t i = 0; i < groups.size(); ++i) {
                    if (groups[i].id == entry.group || groups[i].role == entry.group) {
                        matches.push_back(i);
                    }
                }
                if (matches.size() != 1) {
                    result.error = matches.empty() ? "unknown KV tail group: " + entry.group :
                            "ambiguous KV tail group alias: " + entry.group;
                    return result;
                }
                if (assigned[matches.front()]) {
                    result.error = "duplicate KV tail group assignment: " + groups[matches.front()].id;
                    return result;
                }
                raw[matches.front()] = entry.tokens;
                assigned[matches.front()] = true;
            }
            if (std::any_of(assigned.begin(), assigned.end(), [](bool value) { return !value; })) {
                result.error = "incomplete KV tail group configuration";
                return result;
            }
            break;
        }
    }

    const ggml_type resolved_type = request.exact_type != GGML_TYPE_COUNT ? request.exact_type :
            (kvarn ? GGML_TYPE_F16 : GGML_TYPE_BF16);
    result.groups.reserve(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto & group = groups[i];
        uint32_t requested = raw[i];
        uint32_t effective = group.applicable_standard_kv ?
                std::min(raw[i], group.effective_window) : 0;
        bool native_exact = false;
        if (kvarn) {
            const auto policy = llama_kvarn_tail_policy_for(raw[i], group.effective_window);
            requested = policy.requested_tokens;
            effective = policy.effective_tokens;
            native_exact = policy.native_exact;
        }
        result.groups.push_back({ group.id, group.role, raw[i], requested,
                effective, resolved_type, native_exact });
    }
    result.valid = true;
    return result;
}

llama_kv_tail_request * llama_kv_tail_request_init(
        const char * specification,
        ggml_type exact_type) {
    return new llama_kv_tail_request(
            llama_kv_tail_request_parse(specification ? specification : "", exact_type));
}

void llama_kv_tail_request_free(llama_kv_tail_request * request) {
    delete request;
}

const char * llama_kv_tail_request_last_error(const llama_kv_tail_request * request) {
    return request ? request->error.c_str() : "null KV tail request";
}
