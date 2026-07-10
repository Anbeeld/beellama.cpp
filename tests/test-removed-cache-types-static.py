#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CORE_FILES = (
    "ggml/include/ggml.h",
    "ggml/include/ggml-rpc.h",
    "include/llama.h",
    "ggml/src/ggml.c",
    "ggml/src/ggml-backend-meta.cpp",
    "ggml/src/ggml-quants.c",
    "ggml/src/ggml-quants.h",
    "ggml/src/ggml-common.h",
    "ggml/src/ggml-cpu/ggml-cpu.c",
    "ggml/src/ggml-cpu/ops.cpp",
    "ggml/src/ggml-cpu/ops.h",
    "ggml/src/ggml-cpu/quants.h",
    "common/arg.cpp",
    "src/llama-model-loader.cpp",
    "src/llama-quant.cpp",
    "tools/quantize/quantize.cpp",
    "tests/test-quantize-fns.cpp",
)


REMOVED_IDENTIFIERS = (
    "GGML_TYPE_TURBO2_0",
    "GGML_TYPE_TURBO3_0",
    "GGML_TYPE_TURBO4_0",
    "GGML_TYPE_TURBO2_TCQ",
    "GGML_TYPE_TURBO3_TCQ",
    "GGML_TYPE_TURBO4_TCQ",
    "GGML_TYPE_TQ3_1S",
    "GGML_TYPE_TQ4_1S",
    "GGML_OP_TURBO_WHT",
    "LLAMA_FTYPE_MOSTLY_TQ3_1S",
    "LLAMA_FTYPE_MOSTLY_TQ4_1S",
)


def main() -> None:
    for relative_path in CORE_FILES:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for identifier in REMOVED_IDENTIFIERS:
            if identifier in text:
                raise AssertionError(f"{relative_path} still exposes removed identifier {identifier}")

    loader = (ROOT / "src/llama-model-loader.cpp").read_text(encoding="utf-8")
    if "ftype_val == 43 || ftype_val == 44" not in loader:
        raise AssertionError("legacy Bee TQ3/TQ4 file-type IDs must fail before their tensor IDs can be reinterpreted")
    if "legacy Bee TQ3/TQ4 weight format was removed" not in loader:
        raise AssertionError("legacy TQ rejection must explain how to recover")


if __name__ == "__main__":
    main()
