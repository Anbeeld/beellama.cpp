#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
META = (ROOT / "ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")

compute = META[
    META.index("static enum ggml_status ggml_backend_meta_graph_compute("):
    META.index("static const ggml_backend_i ggml_backend_meta_i")
]
needs_rebuild = compute.index("const bool needs_rebuild")
classify = compute.index("int64_t comm_graph_batch_size = 1")
generation_check = compute.index("backend_ctx->graph_update_required(cgraph)")
retire = compute.index("if ((needs_rebuild || comm_graph_batch_size > 1) && backend_ctx->uid != 0)")
reset = compute.index("if (needs_rebuild) {", retire + 1)
sync = compute.index("ggml_backend_meta_synchronize(backend);", retire)
if not classify < generation_check < needs_rebuild < retire < sync < reset:
    raise AssertionError("the prior projected generation must retire before any rotating container is reset")

print("meta projected generation lifetime checks passed")
