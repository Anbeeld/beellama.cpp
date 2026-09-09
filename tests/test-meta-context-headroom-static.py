#!/usr/bin/env python3
from pathlib import Path

src = (Path(__file__).resolve().parents[1] / "ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")
needle = "constexpr size_t compute_headroom = 32;"
assert needle in src, (
    "meta rotating contexts need 32x view headroom; 16x is exhausted by "
    "tensor-parallel speculative decode before the first prompt"
)
assert "compute_headroom*ggml_get_mem_size(ctx) + ggml_tensor_overhead()" in src
assert "for (size_t i = 0; i < backend_ctx->max_subgraphs; i++)" in src, (
    "resetting the meta graph context must recreate every reserved main-graph slot; "
    "otherwise a later topology can reactivate a pointer freed by ctx.reset"
)
print("meta compute context headroom and graph-slot lifetime invariants OK")
