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

QUANT_TYPES_BY_BITS = {
    8: ("GGML_TYPE_Q8_0",),
    6: ("GGML_TYPE_Q6_1", "GGML_TYPE_Q6_0"),
    5: ("GGML_TYPE_Q5_1", "GGML_TYPE_Q5_0"),
    4: ("GGML_TYPE_Q4_1", "GGML_TYPE_Q4_0"),
    3: ("GGML_TYPE_Q3_1", "GGML_TYPE_Q3_0"),
    2: ("GGML_TYPE_Q2_1", "GGML_TYPE_Q2_0S"),
}

# Keep this aligned with GGML_CUDA_KVARN_DEFAULT_PAIRS in ggml/CMakeLists.txt.
KVARN_DEFAULT_BIT_PAIRS = (
    (8, 8), (8, 6), (8, 5),
    (6, 6), (6, 5), (6, 4),
    (5, 5), (5, 4), (5, 3),
    (4, 4), (4, 3), (4, 2),
    (3, 3), (3, 2),
    (2, 2),
)


def default_pairs() -> list[tuple[str, str]]:
    pairs = [
        ("GGML_TYPE_F16", "GGML_TYPE_F16"),
        ("GGML_TYPE_BF16", "GGML_TYPE_BF16"),
    ]
    for bits_k, bits_v in KVARN_DEFAULT_BIT_PAIRS:
        for variant_k, type_k in enumerate(QUANT_TYPES_BY_BITS[bits_k]):
            for variant_v, type_v in enumerate(QUANT_TYPES_BY_BITS[bits_v]):
                if bits_k == bits_v and variant_k > variant_v:
                    continue
                pairs.append((type_k, type_v))
    return pairs


def emit_pairs(pairs: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"    FATTN_VEC_CASES_ALL_D({type_k}, {type_v})"
        for type_k, type_v in pairs
    )


def render() -> str:
    pairs_default = default_pairs()
    pairs_all = [(type_k, type_v) for type_k in TYPES for type_v in TYPES]
    assert len(TYPES) == 13
    assert len(pairs_default) == 50
    assert len(pairs_all) == 169

    return f"""// This file is generated from scripts/gen-fattn-vec-dispatch.py. Do not edit manually.
//
// The default uses the 15 KVarN bit-pair rules for quantized K/V types,
// retains forward-only same-bit variants, and adds F16/F16 plus BF16/BF16
// for homogeneous precision tails (50 pairs).
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
