#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    bench = (ROOT / "tools" / "llama-bench" / "llama-bench.cpp").read_text(encoding="utf-8")

    expected = {
        "turbo2": (2, "GGML_TYPE_Q2_0S"),
        "turbo2_tcq": (2, "GGML_TYPE_Q2_0S"),
        "turbo3": (3, "GGML_TYPE_Q3_0"),
        "turbo3_tcq": (3, "GGML_TYPE_Q3_0"),
        "turbo4": (4, "GGML_TYPE_Q4_0"),
        "turbo4_tcq": (4, "GGML_TYPE_Q4_0"),
    }

    fallback_begin = bench.find("static ggml_type kvarn_fallback_cache_type")
    fallback_end = bench.find("static bench_cache_type bench_cache_type_from_name", fallback_begin)
    fallback_body = bench[fallback_begin:fallback_end]

    for alias, (bits, fallback) in expected.items():
        marker = f'if (s == "{alias}")'
        start = bench.find(marker)
        require(start >= 0, f"llama-bench is missing the {alias} compatibility redirect")
        body = bench[start:bench.find("}", start) + 1]
        require(f"kvarn_fallback_cache_type({bits})" in body and f", {bits} }}" in body,
                f"llama-bench must redirect {alias} through the {bits}-bit KVarN fallback")
        require("removed in v0.4.0" in body, f"llama-bench must warn for {alias}")
        require("GGML_TYPE_TURBO" not in body, f"llama-bench must not retain a Turbo type for {alias}")
        require(f"case {bits}:  return {fallback};" in fallback_body,
                f"llama-bench's {bits}-bit KVarN fallback must remain {fallback}")


if __name__ == "__main__":
    main()
