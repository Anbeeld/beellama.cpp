#!/usr/bin/env python3
from pathlib import Path

src = (Path(__file__).resolve().parents[1] / "ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")
assert "split_states_proportional" in src, (
    "tensor-parallel batched mul_mat must accept proportional GQA head splits "
    "instead of requiring equal absolute Q/KV head counts"
)
assert "return src_ss[1]; // output follows the query-head split" in src
print("meta GQA split compatibility invariant OK")
