#pragma once

#include "llama.h"

#include <cstdlib>
#include <cstring>

// Direct API fixtures have no common argument parser. Preserve their defaults
// unless the caller explicitly requests direct filesystem I/O.
inline llama_model_params test_model_load_params(llama_model_params params) {
    const char * mode = std::getenv("LLAMA_ARG_LOAD_MODE");
    if (mode != nullptr && std::strcmp(mode, "dio") == 0) {
        params.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
    }
    return params;
}
