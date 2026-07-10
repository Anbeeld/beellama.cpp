#include "../tools/server/server-task.h"

#undef NDEBUG
#include <cassert>

static constexpr size_t KIB = 1024;

static server_prompt make_prompt(const llama_tokens & tokens) {
    server_prompt prompt;
    prompt.tokens = server_tokens(tokens, false);
    return prompt;
}

int main() {
    {
        common_prompt_checkpoint ckpt;
        ckpt.n_tokens = 3;
        ckpt.pos_min = 1;
        ckpt.pos_max = 2;
        ckpt.data_tgt.resize(128);
        ckpt.data_dft.resize(64);
        ckpt.data_spec.resize(32);
        assert(ckpt.size() == 224);

        ckpt.clear();
        assert(ckpt.n_tokens == 0);
        assert(ckpt.pos_min == 0);
        assert(ckpt.pos_max == 0);
        assert(ckpt.empty());
        assert(ckpt.size() == 0);
    }

    {
        server_prompt prompt = make_prompt({1, 2, 3});
        prompt.data.main.resize(64);
        prompt.data.drft.resize(32);
        auto & ckpt = prompt.checkpoints.emplace_back();
        ckpt.n_tokens = 3;
        ckpt.data_tgt.resize(16);
        ckpt.data_dft.resize(8);
        assert(prompt.size() == 120);

        const server_prompt clone = prompt.clone();
        assert(clone.n_tokens() == 3);
        assert(clone.data.size() == 96);
        assert(clone.checkpoints.size() == 1);
        assert(clone.size() == prompt.size());
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt existing = make_prompt({1, 2});
        existing.data.main.resize(700*KIB);
        cache.states.push_back(std::move(existing));

        server_prompt current = make_prompt({3, 4});
        auto * saved = cache.alloc(current, 600*KIB, 0);

        assert(saved != nullptr);
        assert(cache.size() <= cache.limit_size);
        assert(cache.states.size() == 1);
        assert(cache.states.back().data.main.size() == 600*KIB);
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt current = make_prompt({3, 4});
        auto & ckpt = current.checkpoints.emplace_back();
        ckpt.n_tokens = 4;
        ckpt.data_tgt.resize(200*KIB);

        auto * saved = cache.alloc(current, 800*KIB, 0);
        assert(saved != nullptr);
        assert(saved->checkpoints.size() == 1);
        assert(saved->size() == 1000*KIB);
        assert(cache.size() == saved->size());
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt existing = make_prompt({1, 2});
        existing.data.main.resize(100*KIB);
        cache.states.push_back(std::move(existing));

        server_prompt current = make_prompt({3, 4});
        auto & ckpt = current.checkpoints.emplace_back();
        ckpt.n_tokens = 4;
        ckpt.data_tgt.resize(200*KIB);

        assert(cache.alloc(current, 900*KIB, 0) == nullptr);
        assert(cache.states.size() == 1);
        assert(cache.size() == 100*KIB);
    }

    {
        server_prompt_cache cache(1, 0);
        server_prompt current = make_prompt({3, 4});
        assert(cache.alloc(current, 100*KIB, 0) != nullptr);
        assert(cache.alloc(current, 100*KIB, 0) == nullptr);
        assert(cache.states.size() == 1);
    }

    return 0;
}
