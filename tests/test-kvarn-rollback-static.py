from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KVAR_N_CACHE = ROOT / "src/llama-kv-cache-kvarn.cpp"
KVAR_N_TEST = ROOT / "tests/test-kvarn.cpp"


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
    source = KVAR_N_CACHE.read_text(encoding="utf-8")
    can_remove = function_body(source, "bool llama_kv_cache_kvarn::can_remove(")
    seq_rm_plan = function_body(source, "bool llama_kv_cache_kvarn::seq_rm_plan(")

    assert "if (swa)" not in can_remove, (
        "SWA removal must use the KVarN group-safety policy instead of trusting metadata alone"
    )
    assert "if (swa ||" not in seq_rm_plan, (
        "deep SWA rollback must widen to a complete KVarN group boundary"
    )

    tests = KVAR_N_TEST.read_text(encoding="utf-8")
    assert "llama_kvarn_non_swa_tail_groups(0, 0) + 1" in tests, (
        "eager-workspace coverage must derive the production non-SWA stage depth"
    )


if __name__ == "__main__":
    main()
