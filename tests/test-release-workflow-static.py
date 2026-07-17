#!/usr/bin/env python3

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"

EXPECTED_WORKFLOWS = {
    "release-dispatch.yml": "db835825856747ae5cc631f24fb1a84beac685bf22d743e52227456e86f967b0",
    "release.yml": "73c5ed6a3cd15a451ef8cdd326dc7134d8adfabfe1022f23b60b66afca94a7c4",
}


def normalized_sha256(path: Path) -> str:
    source = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def main() -> None:
    actual = {path.name for path in WORKFLOWS.glob("*.y*ml")}
    expected = set(EXPECTED_WORKFLOWS)
    if actual != expected:
        raise AssertionError(
            "workflow inventory diverged from the minimal v0.3.2 contract: "
            f"added={sorted(actual - expected)}, missing={sorted(expected - actual)}"
        )

    for name, expected_hash in EXPECTED_WORKFLOWS.items():
        actual_hash = normalized_sha256(WORKFLOWS / name)
        if actual_hash != expected_hash:
            raise AssertionError(
                f"{name} diverged from v0.3.2 plus the required v0.4.0 release deltas: "
                f"expected {expected_hash}, got {actual_hash}"
            )


if __name__ == "__main__":
    main()
