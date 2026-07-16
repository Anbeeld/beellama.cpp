#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOW_BIT_TYPES = ("Q6_0", "Q6_1", "Q3_0", "Q3_1", "Q2_0S", "Q2_1")


def require(source: str, needle: str, message: str) -> None:
    if needle not in source:
        raise AssertionError(message)


def main() -> None:
    mmq = (ROOT / "ggml/src/ggml-cuda/mmq.cuh").read_text(encoding="utf-8")
    loaders = (ROOT / "ggml/src/ggml-cuda/mmq-load-tiles.cuh").read_text(encoding="utf-8")

    for cache_type in LOW_BIT_TYPES:
        require(
            mmq,
            f"GGML_TYPE_{cache_type}",
            f"CUDA MMQ dispatch/configuration for {cache_type} was lost during an upstream merge",
        )
        require(
            mmq,
            f"DECL_MMQ_CASE(GGML_TYPE_{cache_type})",
            f"CUDA MMQ template declarations for {cache_type} were lost during an upstream merge",
        )

    for loader in (
        "ggml_cuda_mmq_load_tiles_q6_0",
        "ggml_cuda_mmq_load_tiles_q6_1",
        "ggml_cuda_mmq_load_tiles_q2plane",
    ):
        require(
            loaders,
            loader,
            f"CUDA MMQ low-bit tile loader {loader} was lost during an upstream merge",
        )

    for source_type, upstream_tuning_type in (
        ("Q6_0", "Q5_0"),
        ("Q6_1", "Q5_1"),
        ("Q3_0", "Q5_0"),
        ("Q3_1", "Q5_1"),
        ("Q2_0S", "Q5_0"),
        ("Q2_1", "Q5_1"),
    ):
        require(
            mmq,
            f"case GGML_TYPE_{source_type}: return GGML_TYPE_{upstream_tuning_type};",
            f"CUDA MMQ {source_type} must reuse the matching upstream {upstream_tuning_type} tuning rows",
        )


if __name__ == "__main__":
    main()
