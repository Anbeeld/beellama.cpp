#!/usr/bin/env python3
from pathlib import Path
root=Path(__file__).resolve().parents[1]
dflash=(root/"src/models/dflash.cpp").read_text(encoding="utf-8")
context=(root/"src/llama-context.cpp").read_text(encoding="utf-8")
assert "output_is_split || graph_is_split" in dflash
assert '"dflash2_logits_global" : "dflash2_logits_local"' in dflash
assert "ggml_top_k(ctx0, selector_logits, top_k)" in dflash
assert 'strcmp(name, "dflash2_logits_global") == 0' in context
assert "ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend_cpu)" in context
print("DFlash2 selector placement invariant OK")
