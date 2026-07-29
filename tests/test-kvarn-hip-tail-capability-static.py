from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KVAR_N_CACHE = ROOT / "src/llama-kv-cache-kvarn.cpp"


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
    capability = function_body(source, "bool kvarn_backend_supports_native_tail(")

    dedicated_probe = capability.index("ggml_backend_kvarn_tail_attention_supported")
    generic_gate = capability.index("ggml_backend_kv_tail_segmented_attention_supported")
    assert dedicated_probe < generic_gate, (
        "a dedicated KVarN tail capability must bypass the generic segmented-tail gate; "
        "HIP supports the former but intentionally rejects the latter"
    )
    assert "return kvarn_fn(" in capability[dedicated_probe:generic_gate], (
        "the dedicated KVarN tail capability must decide support before the generic gate"
    )


if __name__ == "__main__":
    main()
