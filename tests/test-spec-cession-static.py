#!/usr/bin/env python3
"""Guard the v0.4.0 boundary between upstream speculation and retired Bee paths."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require_absent(relative_path: str, identifiers: tuple[str, ...]) -> None:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    for identifier in identifiers:
        if identifier in text:
            raise AssertionError(f"{relative_path} still contains retired speculative helper {identifier}")


def main() -> None:
    # Reduced-logit verification and grammar-aware speculative acceptance belonged
    # to the retired Bee verifier.  Upstream's full-logit sampler is now canonical.
    require_absent(
        "common/sampling.h",
        (
            "common_sampler_sample_reduced_and_accept_n",
            "common_sampler_supports_reduced",
            "common_sampler_blocks_speculative",
            "common_sampler_has_active_grammar",
            "common_sampler_reasoning_is_forcing",
            "common_sampler_stops_speculative_accept",
        ),
    )
    require_absent(
        "common/sampling.cpp",
        (
            "common_sampler_sample_reduced_and_accept_n",
            "common_sampler_supports_reduced",
            "common_sampler_blocks_speculative",
            "common_sampler_has_active_grammar",
            "common_sampler_reasoning_is_forcing",
            "common_sampler_stops_speculative_accept",
        ),
    )

    require_absent("include/llama.h", ("llama_sampler_grammar_is_active",))
    require_absent("src/llama-sampler.cpp", ("llama_sampler_grammar_is_active",))

    for relative_path in (
        "src/models/dflash_draft.cpp",
        "ggml/src/ggml-cuda/cross-ring-interleave.cu",
        "tests/test-sampling-grammar.cpp",
    ):
        if (ROOT / relative_path).exists():
            raise AssertionError(f"retired speculative path remains: {relative_path}")


if __name__ == "__main__":
    main()
