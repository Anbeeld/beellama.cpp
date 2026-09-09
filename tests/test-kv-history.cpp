#include "llama-kv-cache.h"
#include "models/models.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #c); std::abort(); } } while (0)

int main() {
    llama_model_llama model(llama_model_default_params());
    model.arch = LLM_ARCH_LLAMA;
    // No tensor layers are needed to exercise the real cache metadata path.
    model.hparams.n_layer_all = 0;
    model.hparams.ple_n_heads = 4;
    model.hparams.rope_type = LLAMA_ROPE_TYPE_NONE;

    for (bool unified : { false, true }) {
        llama_kv_cache cache(model, model.hparams, GGML_TYPE_F16, GGML_TYPE_F16,
                false, false, unified, 16, 2, 1, 0, LLAMA_SWA_TYPE_NONE,
                nullptr, nullptr, nullptr, nullptr);
        CHECK(cache.has_cell_ext()); // PLE tokens must survive state serialization even without M-RoPE.
        for (llama_seq_id seq : { 0, 1 }) {
            llama_token tokens[] = { 10 + 100*seq, 11 + 100*seq, 12 + 100*seq, 13 + 100*seq };
            llama_pos pos[] = { 0, 3, 3, 7 };
            int32_t counts[] = { 1, 1, 1, 1 };
            llama_seq_id * ids[] = { &seq, &seq, &seq, &seq };
            llama_ubatch batch{};
            batch.n_tokens = batch.n_seq_tokens = 4;
            batch.n_seqs = batch.n_seqs_unq = 1;
            batch.n_pos = 1;
            batch.token = tokens;
            batch.pos = pos;
            batch.n_seq_id = counts;
            batch.seq_id = ids;
            batch.seq_id_unq = &seq;
            llama_kv_cache::slot_info slots{};
            slots.resize(1);
            slots.strm[0] = unified ? 0 : seq;
            const uint32_t start = unified ? 4*seq : 0;
            slots.idxs[0] = { start, start + 1, start + 2, start + 3 };
            cache.apply_ubatch(slots, batch);
            std::vector<llama_token> history;
            cache.get_prev_tokens(batch, 2, history);
            CHECK(history == std::vector<llama_token>({
                    -1, -1, tokens[0], tokens[0], tokens[0], tokens[0], tokens[2], tokens[2] }));

            // A repeated-position embedding ubatch resolves predecessors by token order.
            // Stored token identities stand in for the image's already-recorded PLE token.
            batch.n_tokens = batch.n_seq_tokens = 2;
            batch.pos = pos + 1;
            batch.token = nullptr;
            cache.get_prev_tokens(batch, 2, history);
            CHECK(history == std::vector<llama_token>({ tokens[0], tokens[0], tokens[0], tokens[2] }));
            CHECK(cache.get_cells(seq).seq_size(seq) == 4);
        }
        // Both streams/sequences remain distinct after the second insertion.
        CHECK(cache.get_cells(llama_seq_id(0)).seq_pos_tok_le(0, 6) == 12);
        CHECK(cache.get_cells(llama_seq_id(1)).seq_pos_tok_le(1, 6) == 112);
    }

    llama_kv_cells cells;
    cells.resize(4);
    cells.pos_set(0, 3);
    cells.seq_add(0, 0);
    cells.pos_set(1, 3);
    cells.seq_add(1, 0);
    cells.ext_set(0, { 0, 0, 10 });
    cells.ext_set(1, { 0, 0, 11 });
    CHECK(cells.seq_size(0) == 2);
    CHECK(cells.seq_pos_tok_le(0, 3) == 11);
    auto saved = cells;
    CHECK(cells.seq_rm_cell(1, 0));
    CHECK(cells.seq_size(0) == 1);
    CHECK(cells.seq_pos_tok_le(0, 3) == 10);
    cells = saved;
    CHECK(cells.seq_size(0) == 2);
    CHECK(cells.seq_pos_tok_le(0, 3) == 11);
    CHECK(!cells.pos_add(1, 5));
    CHECK(cells.seq_pos_tok_le(0, 3) == 10);
    CHECK(cells.seq_pos_tok_le(0, 8) == 11);
    cells.reset();
    CHECK(cells.seq_size(0) == 0);
    CHECK(cells.seq_pos_tok_le(0, 8) == LLAMA_TOKEN_NULL);
    return 0;
}
