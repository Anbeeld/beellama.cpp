#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = (ROOT / "tmp/run-kv-tail-paired-baseline.ps1").read_text(encoding="utf-8")
    for marker in (
        "'Iteration', 'StdQuant', 'Prefill', 'Full', 'Vram', 'Server'",
        "$VramMetricNames",
        "Vram row is missing required metric",
        "candidateValue / $baselineValue",
        "mixed composite identities",
        "model identity differs within execution pair",
    ):
        if marker not in source:
            raise AssertionError(f"paired wrapper lacks {marker!r}")


if __name__ == "__main__":
    main()
