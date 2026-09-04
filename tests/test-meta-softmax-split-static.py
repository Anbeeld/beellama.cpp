#!/usr/bin/env python3
from pathlib import Path
src=(Path(__file__).resolve().parents[1]/"ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")
assert "handle_src0_with_mirrored_inputs" in src
softmax=src.split("case GGML_OP_SOFT_MAX:",1)[1].split("case GGML_OP_ROPE:",1)[0]
assert "handle_src0_with_mirrored_inputs(src_ss)" in softmax
print("meta split softmax broadcast invariant OK")
