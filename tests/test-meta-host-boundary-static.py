#!/usr/bin/env python3
from pathlib import Path
src=(Path(__file__).resolve().parents[1]/"ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")
assert "meta_tensor_is_external_host" in src
assert "if (meta_tensor_is_external_host(node))" in src
print("meta external host boundary invariant OK")
