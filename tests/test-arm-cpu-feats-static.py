#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp").read_text(encoding="utf-8")


feature_snapshot = "ggml_feats_arch64_runtime_t af = ggml_feats_get_arch64_runtime();"
snapshot_pos = SOURCE.find(feature_snapshot)
if snapshot_pos < 0:
    raise AssertionError("AArch64 backend scoring must read the runtime feature snapshot")

unused_pos = SOURCE.find("GGML_UNUSED(af);", snapshot_pos + len(feature_snapshot))
first_feature_guard = SOURCE.find("#ifdef GGML_USE_DOTPROD", snapshot_pos)
if unused_pos < 0 or unused_pos > first_feature_guard:
    raise AssertionError(
        "the baseline AArch64 CPU variant must mark its feature snapshot unused before optional feature guards"
    )

print("ARM CPU feature-probe warning contract checks passed")
