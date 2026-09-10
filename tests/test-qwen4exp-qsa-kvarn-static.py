from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KV_CACHE_KVARN_H = ROOT / "src/llama-kv-cache-kvarn.h"
KV_CACHE_BASE = ROOT / "src/llama-kv-cache.cpp"
MEMORY_HYBRID_IDX = ROOT / "src/llama-memory-hybrid-idx.cpp"
MODEL = ROOT / "src/llama-model.cpp"
QWEN4EXP = ROOT / "src/models/qwen4exp.cpp"


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for pos in range(brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : pos]
    raise AssertionError(f"unterminated function: {signature}")


def main() -> None:
    header = KV_CACHE_KVARN_H.read_text(encoding="utf-8")
    # QSA needs one attention row per cache cell. The compact read plan reorders
    # rows by physical record, so an indexer-mirrored cache must not use it.
    assert "void set_indexer_mirror(bool value)" in header, (
        "the KVarN attention cache must expose the QSA indexer-mirror switch"
    )
    compact = header.split("bool uses_compact_read_indices() const { return", 1)[1].split(";", 1)[0]
    assert "!indexer_mirror" in compact, (
        "an indexer-mirrored KVarN cache must disable the compact read plan"
    )
    assert "n_seq_max > 1" in compact, (
        "the compact read plan still applies to unified multi-sequence KVarN caches without an index mirror"
    )

    model = MODEL.read_text(encoding="utf-8")
    create_memory = function_body(model, "llama_memory_i * llama_model::create_memory(")
    assert "kvarn_attn->set_indexer_mirror(true)" in create_memory, (
        "the Qwen4Exp QSA attention cache must be marked as an indexer mirror"
    )
    mirror_guard = create_memory.split("set_indexer_mirror(true)", 1)[0].rsplit("if (", 1)[1]
    assert "needs_mem_idx" in mirror_guard and "filter_idx" in mirror_guard, (
        "only a model with an index cache may mirror: the mark must require both needs_mem_idx and filter_idx"
    )

    # The QSA path validates the cell-for-cell mirror; keep the invariant and the
    # dense fallback that the attention builder selects when there is no index cache.
    qwen4exp = QWEN4EXP.read_text(encoding="utf-8")
    assert "the indexer cache must track the attention cache cell for cell" in qwen4exp, (
        "the QSA graph must keep validating that the indexer mirrors the attention cells"
    )
    assert "mctx_hyb->get_idx() != nullptr" in qwen4exp, (
        "QSA sparse selection must stay gated on the index cache; without it the graph uses dense attention"
    )

    base = KV_CACHE_BASE.read_text(encoding="utf-8")
    apply = function_body(base, "void llama_kv_cache::apply_ubatch(")
    # The plain QSA index cache borrows the attention slot infos, KVarN-only
    # allocation metadata included. It has no structured allocation bookkeeping,
    # so the commit must be limited to caches that do.
    commit = apply.split("sinfo.group_stage_slots", 1)[0]
    assert "allocation_group_size > 1 && !" in commit, (
        "only a structured cache may adopt KVarN-only stage-slot metadata"
    )
    assert "allocation_group_stage_slots = sinfo.group_stage_slots" in apply, (
        "structured caches must still commit the allocator's complete stage assignment"
    )

    hybrid_idx = MEMORY_HYBRID_IDX.read_text(encoding="utf-8")
    init_batch = function_body(hybrid_idx, "llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(")
    assert "heads_idx = kv_ctx->get_sinfos()" in init_batch, (
        "the QSA index cache must keep mirroring the attention slot layout"
    )


if __name__ == "__main__":
    main()
