#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    fattn = read("ggml/src/ggml-cuda/fattn.cu")
    set_rows = read("ggml/src/ggml-cuda/set-rows.cu")
    convert = read("convert_hf_to_gguf.py")
    conversion_init = read("conversion/__init__.py")

    require(
        "k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size" not in set_rows,
        "quantized set_rows helper must launch through the checked CUDA wrapper",
    )
    require(
        "GGML_TYPE_TURBO" not in fattn and "TCQ" not in fattn,
        "the generic CUDA FlashAttention path must not retain TurboQuant or TCQ handling",
    )
    require(
        "supports_mmproj_model" in conversion_init,
        "conversion package must expose a way to detect supported mmproj companions",
    )
    require(
        "supports_mmproj_model(model_architecture)" in convert and
        "Use --mmproj to export the multimodal projector" in convert,
        "converter must warn when text-converting a multimodal config with a supported mmproj companion",
    )


if __name__ == "__main__":
    main()
