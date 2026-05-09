#include "speculative.h"

#undef NDEBUG
#include <cassert>

int main() {
    {
        const auto plan = common_dflash_ring_write_plan(1024, 12, 64);
        assert(plan.ring_pos == 12);
        assert(plan.n_tokens == 64);
        assert(plan.src_token_offset == 0);
    }

    {
        const auto plan = common_dflash_ring_write_plan(1024, 900, 256);
        assert(plan.ring_pos == 900);
        assert(plan.n_tokens == 256);
        assert(plan.src_token_offset == 0);
    }

    {
        const auto plan = common_dflash_ring_write_plan(1024, 900, 2048);
        assert(plan.ring_pos == 900);
        assert(plan.n_tokens == 1024);
        assert(plan.src_token_offset == 1024);
    }

    {
        const auto plan = common_dflash_ring_write_plan(1024, 1100, 2048);
        assert(plan.ring_pos == 76);
        assert(plan.n_tokens == 1024);
        assert(plan.src_token_offset == 1024);
    }

    {
        const auto plan = common_dflash_ring_write_plan(1024, -5, 4);
        assert(plan.ring_pos == 1019);
        assert(plan.n_tokens == 4);
        assert(plan.src_token_offset == 0);
    }

    {
        const auto plan = common_dflash_ring_write_plan(0, 12, 64);
        assert(plan.ring_pos == 0);
        assert(plan.n_tokens == 0);
        assert(plan.src_token_offset == 0);
    }

    return 0;
}
