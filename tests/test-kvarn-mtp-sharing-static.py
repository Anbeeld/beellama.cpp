from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ISWA_CACHE = ROOT / "src/llama-kv-cache-iswa.cpp"
KVAR_N_CACHE = ROOT / "src/llama-kv-cache-kvarn.cpp"
LLAMA_GRAPH = ROOT / "src/llama-graph.cpp"


def main() -> None:
    iswa = ISWA_CACHE.read_text(encoding="utf-8")
    kvarn = KVAR_N_CACHE.read_text(encoding="utf-8")
    graph = LLAMA_GRAPH.read_text(encoding="utf-8")

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
    assert "get_tail_query_order(swa), get_tail_run_desc(swa), mask" in graph, (
        "iSWA tail planning must use the query order for the selected cache group"
    )


if __name__ == "__main__":
    main()
