#!/usr/bin/env python3
"""Generate the CUDA/HIP/MUSA FlashAttention vector dispatch matrix.

Run with --write after changing the policy. Without --write, emit the generated
header to stdout so CI can verify the checked-in result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

TYPES = (
    "GGML_TYPE_F16",
    "GGML_TYPE_BF16",
    "GGML_TYPE_Q8_0",
    "GGML_TYPE_Q6_1",
    "GGML_TYPE_Q6_0",
    "GGML_TYPE_Q5_1",
    "GGML_TYPE_Q5_0",
    "GGML_TYPE_Q4_1",
    "GGML_TYPE_Q4_0",
    "GGML_TYPE_Q3_1",
    "GGML_TYPE_Q3_0",
    "GGML_TYPE_Q2_1",
    "GGML_TYPE_Q2_0S",
)


def default_pairs() -> list[tuple[str, str]]:
    return [
        (type_k, type_v)
        for rank_k, type_k in enumerate(TYPES)
        for rank_v, type_v in enumerate(TYPES)
        if rank_k <= rank_v or type_k == "GGML_TYPE_F16" or type_v == "GGML_TYPE_F16"
    ]


def emit_pairs(pairs: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"    FATTN_VEC_CASES_ALL_D({type_k}, {type_v})"
        for type_k, type_v in pairs
    )


def render() -> str:
    pairs_default = default_pairs()
    pairs_all = [(type_k, type_v) for type_k in TYPES for type_v in TYPES]
    assert len(TYPES) == 13
    assert len(pairs_default) == 103
    assert len(pairs_all) == 169

    return f"""// This file is generated from scripts/gen-fattn-vec-dispatch.py. Do not edit manually.
//
// The default keeps the former HALF predicate over the thirteen retained
// cache types: rank(K) <= rank(V) || K == f16 || V == f16 (103 pairs).
#if defined(GGML_CUDA_FA_ALL_QUANTS)
{emit_pairs(pairs_all)}
#else
{emit_pairs(pairs_default)}
#endif
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="update the checked-in dispatch header")
    parser.add_argument("--check", action="store_true", help="fail when the checked-in dispatch header is stale")
    args = parser.parse_args()
    if args.write and args.check:
        parser.error("--write and --check cannot be used together")
    rendered = render()
    root = Path(__file__).resolve().parents[1]
    target = root / "ggml/src/ggml-cuda/fattn-vec-dispatch.cuh"

    if args.write:
        target.write_text(rendered, encoding="utf-8")
    elif args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{target.relative_to(root)} is stale; run {Path(__file__).name} --write")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
