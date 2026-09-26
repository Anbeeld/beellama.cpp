#!/usr/bin/env python3
"""Lock the save/load model-suite result contract to explicit test suites."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "tests/test-save-load-state.cpp"
SIGNATURE = "static test_suite run_save_load_tests_for_model("
CHECKS = (
    "test_baseline",
    "test_kvarn_partial_checkpoint_history",
    "test_kvarn_unified_capacity",
    "test_kvarn_unified_reuses_freed_groups",
    "test_tail_state_contract",
    "test_cross_ubatch_tail_state",
    "test_tail_copy_is_immediately_saveable",
    "test_tail_state_v1_compatibility",
    "test_kvarn_full_window_native_exact",
    "test_seq_rm_isolated",
    "test_state_load",
    "test_seq_cp_host",
    "test_seq_cp_device",
)


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for position in range(opening, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : position]
    raise AssertionError("save/load function has no closing brace")


def main() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    body = function_body(source, SIGNATURE)

    returns = re.findall(r"\breturn\s+([^;]+);", body)
    assert returns and set(returns) <= {"suite", "fail_suite()"}, (
        "model-suite must return a populated suite or explicit failed suite: " + repr(returns)
    )
    check_positions = [body.index(check) for check in CHECKS]
    assert check_positions == sorted(check_positions), "save/load suite checks were reordered or lost"
    assert "suite.results.assign(test_names.size(), test_status::FAIL)" in body
    assert "if (model == nullptr)" in body and "return fail_suite();" in body
    assert "suite.results.assign(test_names.size(), test_status::SKIP)" in body
    assert "suite.results[0] = test_status::PASS" in body

    suite_body = function_body(source, "struct test_suite {")
    assert "results.size() == test_names.size()" in suite_body
    assert "std::any_of" in suite_body and "test_status::PASS" in suite_body
    assert "std::none_of" in suite_body and "test_status::FAIL" in suite_body

    main_body = source[source.index("int main(") :]
    assert "const test_suite suite = run_save_load_tests_for_model(model_path, params);" in main_body
    assert "const test_suite suite = run_save_load_tests_for_model(params.model.path, params);" in main_body


if __name__ == "__main__":
    main()
