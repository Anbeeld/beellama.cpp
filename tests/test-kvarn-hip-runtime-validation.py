#!/usr/bin/env python3

from pathlib import Path
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    protocol_path = repo / "scripts" / "hip" / "validate-kvarn-runtime.py"
    require(protocol_path.is_file(), "HIP KVarN runtime validation protocol is missing")

    protocol = protocol_path.read_text(encoding="utf-8")
    runtime_test = (repo / "tests" / "test-kvarn.cpp").read_text(encoding="utf-8")

    for token in (
        "rdna-wave32",
        "cdna-wave64",
        "GGML_KVARN_TEST_AMD_ROUTE_BOUNDARIES_ONLY",
        "GGML_KVARN_AMD_RUNTIME_ATTESTATION",
        "evidence.json",
        "CMakeCache.txt",
        "git rev-parse HEAD",
    ):
        require(token in protocol, f"runtime protocol omitted required evidence token: {token}")

    require("GGML_KVARN_TEST_AMD_ROUTE_BOUNDARIES_ONLY" in runtime_test,
            "test-kvarn has no deterministic AMD route-boundary subset")
    require("GGML_KVARN_AMD_RUNTIME_ATTESTATION" in runtime_test,
            "test-kvarn does not attest the physical AMD wave used by the runtime subset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
