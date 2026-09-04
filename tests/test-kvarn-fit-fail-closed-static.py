#!/usr/bin/env python3
from pathlib import Path

fit_source = Path("common/fit.cpp").read_text(encoding="utf-8")
common_source = Path("common/common.cpp").read_text(encoding="utf-8")
needle = """if (extra->cparams->kvarn.type != LLAMA_KVARN_TYPE_DISABLED) {
                    throw common_params_fit_unsafe_extra_exception(
                        \"cannot safely fit the KVarN draft context without its target context; rerun with -fit off\");
                }"""
assert needle in fit_source, "KVarN extra-context fit failures must fail instead of undercounting VRAM"
assert "fit_status == COMMON_PARAMS_FIT_STATUS_UNSAFE_EXTRA" in common_source, (
    "KVarN draft fit failure must stop model initialization")
print("KVarN fit fail-closed static check passed")
