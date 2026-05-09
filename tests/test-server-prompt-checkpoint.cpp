#include "../tools/server/server-task.h"

#undef NDEBUG
#include <cassert>

int main() {
    server_prompt_checkpoint ckpt {
        /*.pos_min  = */ 1,
        /*.pos_max  = */ 2,
        /*.n_tokens = */ 3,
        /*.data     = */ std::vector<uint8_t>(128),
    };
    ckpt.ring_data.resize(256);

    ckpt.clear();

    assert(ckpt.pos_min == 0);
    assert(ckpt.pos_max == 0);
    assert(ckpt.n_tokens == 0);
    assert(ckpt.data.empty());
    assert(ckpt.data.capacity() == 0);
    assert(ckpt.ring_data.empty());
    assert(ckpt.ring_data.capacity() == 0);

    return 0;
}
