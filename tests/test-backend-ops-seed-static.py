#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = (ROOT / "tests/test-backend-ops.cpp").read_text(encoding="utf-8")

    required = (
        'strcmp(argv[i], "--seed")',
        "backend_ops_seed=",
        "hash_test_case_identity",
        "set_test_case_seed(g_test_seed, test->vars())",
        "make_test_rng()",
    )
    for marker in required:
        if marker not in source:
            raise AssertionError(f"backend-op seed contract lacks {marker!r}")

    if "std::random_device" in source:
        raise AssertionError("backend-op input generation still bypasses the reproducible run seed")

    if "set_test_case_seed(g_test_seed, i)" in source:
        raise AssertionError("backend-op seed still depends on a post-filter vector index")


if __name__ == "__main__":
    main()
