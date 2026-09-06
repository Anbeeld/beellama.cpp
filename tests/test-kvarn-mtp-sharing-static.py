from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ISWA_CACHE = ROOT / "src/llama-kv-cache-iswa.cpp"
KVAR_N_CACHE = ROOT / "src/llama-kv-cache-kvarn.cpp"
STANDARD_CACHE = ROOT / "src/llama-kv-cache.cpp"
LLAMA_GRAPH = ROOT / "src/llama-graph.cpp"
SPECULATIVE = ROOT / "common/speculative.cpp"
SERVER_CONTEXT = ROOT / "tools/server/server-context.cpp"


def main() -> None:
    iswa = ISWA_CACHE.read_text(encoding="utf-8")
    kvarn = KVAR_N_CACHE.read_text(encoding="utf-8")
    standard = STANDARD_CACHE.read_text(encoding="utf-8")
    graph = LLAMA_GRAPH.read_text(encoding="utf-8")
    speculative = SPECULATIVE.read_text(encoding="utf-8")
    server = SERVER_CONTEXT.read_text(encoding="utf-8")

    assert "static_cast<llama_kv_cache_iswa *>(mem_other)" not in iswa, (
        "auxiliary cache sharing must validate the outer cache type before dereferencing it"
    )
    assert "dynamic_cast<llama_kv_cache_kvarn *>(cache_mem_other)" in iswa, (
        "Gemma 4 MTP must recognize structured target cache groups"
    )
    assert "make_shared_metadata_cache" in iswa, (
        "Gemma 4 MTP must create a non-owning metadata view over target KVarN cells"
    )
    assert "get_kv_n_stream() != 1" not in iswa, (
        "Gemma 4 MTP sharing must support multi-stream target KVarN caches"
    )
    assert "STANDARD_SWA_FALLBACK" not in iswa, (
        "multi-slot Gemma 4 must retain KVarN on SWA as well as full-attention layers"
    )
    assert "uses_materialization_indices" in kvarn, (
        "shared multi-stream reads must provide live-record materialization indices"
    )
    assert "shared_live_indices" in kvarn, (
        "shared multi-stream materialization must enumerate every target stream"
    )
    assert "get_materialization_source(shared_il" in kvarn, (
        "shared MTP attention must read the target's persistent KVarN records"
    )
    assert "shared_graph_layers.empty() && cache->uses_native_attention" in kvarn, (
        "shared MTP reads must use the materialization path until segmented tail sharing is available"
    )
    assert "metadata.swap(*prepared_owner)" not in kvarn, (
        "KVarN restore must preserve metadata object identity for shared MTP views"
    )
    assert "metadata->swap_logical_state_from(**prepared_owner)" in kvarn, (
        "KVarN restore must install validated logical state into the stable metadata object"
    )
    assert "get_tail_query_order(swa), get_tail_run_desc(swa), mask" in graph, (
        "iSWA tail planning must use the query order for the selected cache group"
    )
    assert "source.k ? source.k : source.k_tail" in standard, (
        "MTP sharing must accept a bodyless native-exact SWA source"
    )
    assert "layers.back().k_tail = layer_share.k_tail" in standard, (
        "MTP sharing must retain the target's exact K tail payload"
    )
    assert "return other->set_input_kq_mask_tail" in standard, (
        "shared MTP reads must use the target's multi-slot tail metadata"
    )
    assert "compact_tail && k_cur && v_cur" in graph, (
        "read-only shared compact tails must not require assistant K/V projections"
    )
    assert "tail_route == LLAMA_KV_TAIL_ROUTE_NONE ? nullptr" in graph, (
        "layers without a tail route must not receive group-wide tail masks"
    )
    assert "common_speculative_draft_memory_is_shared" in speculative, (
        "the speculative owner must expose shared draft-memory ownership"
    )
    assert "slot.mem.init(ctx_tgt, slot.draft_owns_state ? ctx_dft : nullptr)" in server, (
        "server memory mutations must not apply a second time through a shared MTP view"
    )
    assert "llama_context * ctx_dft_state = draft_owns_state ? ctx_dft : nullptr" in server, (
        "prompt and checkpoint state must omit independently serialized shared MTP memory"
    )
    assert "Shared MTP reads have no graph-local current segment" in standard, (
        "shared tail extents must exclude assistant query rollback rows"
    )


if __name__ == "__main__":
    main()
