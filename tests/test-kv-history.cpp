#include "llama-batch.h"
#include "llama-kv-cells.h"

#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #c); std::abort(); } } while (0)

// Same cell writes llama_kv_cache::apply_ubatch uses for token/pos metadata.
static void apply_ubatch(llama_kv_cells & cells, const std::vector<uint32_t> & idxs, const llama_ubatch & ubatch) {
    CHECK(idxs.size() == ubatch.n_tokens);
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const uint32_t idx = idxs[i];
        if (!cells.is_empty(idx)) {
            cells.rm(idx);
        }
        cells.pos_set(idx, ubatch.pos[i]);
        if (ubatch.token) {
            llama_kv_cell_ext ext;
            ext.tok = ubatch.token[i];
            cells.ext_set(idx, ext);
        }
        for (int32_t iseq = 0; iseq < ubatch.n_seq_id[i]; ++iseq) {
            cells.seq_add(idx, ubatch.seq_id[i][iseq]);
        }
    }
}

// Same predecessor walk as llama_kv_cache::get_prev_tokens.
static void get_prev_tokens(
        const llama_kv_cells & cells,
        const llama_ubatch & ubatch,
        uint32_t n,
        std::vector<llama_token> & res) {
    const uint32_t n_tokens = ubatch.n_tokens;

    res.clear();
    res.resize(n_tokens * n, LLAMA_TOKEN_NULL);

    if (n == 0) {
        return;
    }

    std::vector<uint32_t> ord;
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idx;

    if (!ubatch.token) {
        ord.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            auto & v = seq_idx[ubatch.seq_id[i][0]];
            ord[i] = uint32_t(v.size());
            v.push_back(i);
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        for (uint32_t j = 0; j < n; ++j) {
            const llama_pos d = (llama_pos) (n - j);
            llama_pos p;
            if (!ubatch.token) {
                const auto & v = seq_idx[seq_id];
                const int64_t k = (int64_t) ord[i] - d;
                p = k >= 0 ? ubatch.pos[v[k]] : ubatch.pos[v[0]] + (llama_pos) k;
            } else {
                p = ubatch.pos[i] - d;
            }
            if (p < 0) {
                continue;
            }
            res[i * n + j] = cells.seq_pos_tok_le(seq_id, p);
        }
    }
}

int main() {
    for (bool unified : { false, true }) {
        std::vector<llama_kv_cells> streams(unified ? 1 : 2);
        for (auto & cells : streams) {
            cells.resize(16);
        }
        for (llama_seq_id seq : { 0, 1 }) {
            llama_kv_cells & cells = streams[unified ? 0 : seq];
            llama_token tokens[] = { 10 + 100 * seq, 11 + 100 * seq, 12 + 100 * seq, 13 + 100 * seq };
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
            const uint32_t start = unified ? 4 * seq : 0;
            apply_ubatch(cells, { start, start + 1, start + 2, start + 3 }, batch);
            std::vector<llama_token> history;
            get_prev_tokens(cells, batch, 2, history);
            CHECK(history == std::vector<llama_token>({
                    -1, -1, tokens[0], tokens[0], tokens[0], tokens[0], tokens[2], tokens[2] }));

            // A repeated-position embedding ubatch resolves predecessors by token order.
            // Stored token identities stand in for the image's already-recorded PLE token.
            batch.n_tokens = batch.n_seq_tokens = 2;
            batch.pos = pos + 1;
            batch.token = nullptr;
            get_prev_tokens(cells, batch, 2, history);
            CHECK(history == std::vector<llama_token>({ tokens[0], tokens[0], tokens[0], tokens[2] }));
            CHECK(cells.seq_size(seq) == 4);
        }
        // Both streams/sequences remain distinct after the second insertion.
        CHECK(streams[0].seq_pos_tok_le(0, 6) == 12);
        CHECK(streams[unified ? 0 : 1].seq_pos_tok_le(1, 6) == 112);
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

    // Destination lookup work depends on matching-position multiplicity, not
    // total cache capacity. This guards long-context partial restore metadata.
    llama_kv_cells indexed;
    constexpr uint32_t n_indexed = 65536;
    indexed.resize(n_indexed + 2);
    for (uint32_t i = 0; i < n_indexed; ++i) {
        indexed.pos_set(i, llama_pos(i));
        indexed.ext_set(i, { 0, 0, llama_token(i) });
        indexed.seq_add(i, 0);
    }
    uint64_t probes = 0;
    CHECK(indexed.seq_find_cell(0, 60000, nullptr, &probes) == 60000);
    CHECK(probes == 1);

    indexed.pos_set(n_indexed, 60000);
    indexed.ext_set(n_indexed, { 1, 0, 7 });
    indexed.seq_add(n_indexed, 0);
    const llama_kv_cell_ext wanted { 1, 0, 7 };
    probes = 0;
    CHECK(indexed.seq_find_cell(0, 60000, &wanted, &probes) == n_indexed);
    CHECK(probes == 2);

    CHECK(indexed.seq_rm_pos_range(0, 100, 200) == 100);
    CHECK(indexed.seq_find_cell(0, 99) == 99);
    CHECK(indexed.seq_find_cell(0, 100) == indexed.size());
    CHECK(indexed.seq_find_cell(0, 199) == indexed.size());
    CHECK(indexed.seq_find_cell(0, 200) == 200);

    // Partial restore keeps exactly the checkpoint anchors. Later divergent
    // cells, including repeated positions inside the checkpoint range, must
    // not survive merely because their position precedes the checkpoint tail.
    indexed.pos_set(n_indexed + 1, 60000);
    indexed.ext_set(n_indexed + 1, { 2, 0, 8 });
    indexed.seq_add(n_indexed + 1, 0);
    const std::set<uint32_t> retained { 99, 200, 60000, n_indexed };
    CHECK(indexed.seq_rm_except(0, retained) == 0);
    CHECK(indexed.seq_size(0) == retained.size());
    for (uint32_t cell : retained) {
        CHECK(indexed.seq_has(cell, 0));
    }
    CHECK(!indexed.seq_has(n_indexed + 1, 0));
    return 0;
}
