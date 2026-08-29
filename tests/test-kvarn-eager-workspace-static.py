from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_STORE = ROOT / "ggml/src/ggml-cuda/kvarn.cu"
KV_CACHE = ROOT / "src/llama-kv-cache-kvarn.cpp"
KV_CACHE_BASE = ROOT / "src/llama-kv-cache.cpp"


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
    source = CUDA_STORE.read_text(encoding="utf-8")
    dispatch = function_body(source, "void ggml_cuda_op_kvarn_store(")
    flush = function_body(source, "static __global__ void kvarn_store_workspace_flush_kernel(")

    workspace_predicate = dispatch.split("const bool use_workspace =", 1)[1].split(";", 1)[0]
    assert "!eager_records" not in workspace_predicate, (
        "eager KVarN records must remain eligible for the staged workspace store path"
    )
    assert "bool eager_records" in source.split(
        "static __global__ void kvarn_store_workspace_flush_kernel(", 1
    )[1].split(") {", 1)[0], "workspace flush must receive the eager-record policy"
    # Eager stores must enumerate the groups that COMPLETE inside the store,
    # anchored on the group containing start_local. Anchoring on the first
    # crossed boundary (ceil(start_local / KVAR_N_DIM)) silently drops the group
    # straddling an unaligned store start, because no later store enumerates it.
    assert "start_local / KVAR_N_DIM + candidate" in flush, (
        "eager workspace flush must anchor on the group holding start_local"
    )
    assert "(record_group + 1) * KVAR_N_DIM > end_local" in flush, (
        "eager workspace flush must only seal groups that complete inside the store"
    )
    assert "eager_records ? boundary_group : boundary_group - tail_groups" not in flush, (
        "eager workspace flush must not reuse the delayed boundary anchor"
    )

    cache_source = KV_CACHE.read_text(encoding="utf-8")
    store = function_body(cache_source, "ggml_tensor * llama_kv_cache_kvarn::store(")
    assert "result->op_params[3] = kvarn_workspace_tokens_per_stream_hint(sinfo);" in store, (
        "host-assigned stage slots must retain the bulk workspace hint; the backends consume "
        "their encoded slot provenance directly"
    )
    assert "sinfo.stage_slots.empty()" not in store.split("result->op_params[3]", 1)[1].split(";", 1)[0], (
        "explicit stage ownership must not force the slower monolithic KVarN store"
    )

    allocation_source = KV_CACHE_BASE.read_text(encoding="utf-8")
    allocation = function_body(allocation_source, "llama_kv_cache::slot_info llama_kv_cache::find_slot(")
    initializer = allocation.split("slot_info res = {", 1)[1].split("};", 1)[0]
    assert "/*.stage_slots =*/ { }" in initializer, (
        "slot_info aggregate initialization must include stage_slots so warning-as-error GCC/Clang builds remain valid"
    )


if __name__ == "__main__":
    main()
