#include "fit-kvarn-tail.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

int main() {
    common_bee_fit_retry_state state;
    std::vector<size_t> margins = { 64, 128 };

    const auto retry = state.observe("ctx=50000;ngl=42", { 0, 256 }, margins);
    require(retry == COMMON_BEE_FIT_RETRY, "underestimated Bee candidate was not retried");
    require(margins[0] == 64 && margins[1] == 384,
            "measured Bee shortfall was not fed back into the original margins");

    const auto accept = state.observe("ctx=49920;ngl=42", { 0, 0 }, margins);
    require(accept == COMMON_BEE_FIT_ACCEPT, "fitting exact Bee candidate was not accepted");

    common_bee_fit_retry_state repeated_state;
    margins = { 0 };
    require(repeated_state.observe("same", { 16 }, margins) == COMMON_BEE_FIT_RETRY,
            "first non-fitting candidate was not retried");
    require(repeated_state.observe("same", { 16 }, margins) == COMMON_BEE_FIT_REPEATED_CANDIDATE,
            "repeated non-fitting candidate did not terminate");

    std::puts("test-fit-kvarn-tail: all tests OK");
    return 0;
}
