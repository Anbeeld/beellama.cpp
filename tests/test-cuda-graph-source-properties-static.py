#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON = (ROOT / "ggml/src/ggml-cuda/common.cuh").read_text(encoding="utf-8")
CUDA = (ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")


def require(text: str, needle: str, reason: str) -> None:
    if needle not in text:
        raise AssertionError(f"{reason}: missing {needle!r}")


require(
    COMMON,
    "ggml_type node_src_types[GGML_MAX_SRC]",
    "CUDA graph identity must include source element types",
)
require(
    CUDA,
    "prop.node_src_types[j] = cgraph->nodes[i]->src[j]->type;",
    "CUDA graph property snapshots must record each source type",
)
require(
    CUDA,
    "bool res = false;",
    "stable graph reuse must be decided by the complete property snapshot",
)
require(
    CUDA,
    "graph->uid = cgraph->uid;",
    "CUDA graph diagnostics must retain the scheduler generation",
)
update_required = CUDA[
    CUDA.index("static bool ggml_cuda_graph_update_required"):
    CUDA.index("static void ggml_cuda_graph_update_executable")
]
if "cgraph->uid == graph->uid" in update_required:
    raise AssertionError(
        "CUDA graph source-property validation must not be bypassed solely because the scheduler UID is unchanged"
    )
if "GGML_CUDA_DIAG_SYNC_STATE_WRITES" in CUDA:
    raise AssertionError(
        "failed state-write synchronization diagnostics must not remain in the production CUDA graph path"
    )

print("CUDA graph source-property contract checks passed")
