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
    "const bool has_stable_identity = cgraph->uid != 0;",
    "unversioned projected graphs must be distinguished from stable graph identities",
)
require(
    CUDA,
    "bool res = !has_stable_identity;",
    "unversioned projected graphs must request a CUDA executable refresh",
)
require(
    CUDA,
    "if (cgraph->uid == 0) {\n                // Projected meta-backend graphs intentionally have no stable",
    "unversioned projected graphs must retain CUDA graph execution with a per-evaluation update",
)

print("CUDA graph source-property contract checks passed")
