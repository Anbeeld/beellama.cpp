#!/usr/bin/env python3
"""Guard loop-guard state against speculative checkpoint rollback regressions."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    source = (ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")
    verify_start = source.find("// verify and try to accept the draft")
    verify_end = source.find("const int64_t t_now = ggml_time_us();", verify_start)
    require(verify_start >= 0 and verify_end >= 0, "speculative verification block not found")
    verify = source[verify_start:verify_end]

    require(
        "const server_loop_guard loop_guard_save = slot.loop_guard;" in verify,
        "speculative verification must snapshot loop-guard state before sampler acceptance",
    )
    sampler_restore = verify.find("slot.smpl = std::move(smpl_save);")
    require(sampler_restore >= 0, "speculative checkpoint rollback must restore the sampler")
    restore = verify[sampler_restore:]
    for field in (
        "slot.loop_guard = loop_guard_save;",
        "slot.loop_guard_interventions = loop_guard_interventions_save;",
        "slot.loop_guard_triggered = loop_guard_triggered_save;",
        "slot.loop_guard_action = loop_guard_action_save;",
        "slot.loop_guard_reason = loop_guard_reason_save;",
        "slot.reasoning_output_tokens = reasoning_output_tokens_save;",
        "slot.visible_output_tokens = visible_output_tokens_save;",
        "slot.has_next_token = has_next_token_save;",
        "slot.stop = stop_save;",
        "slot.stop_detail = stop_detail_save;",
    ):
        require(field in restore, f"checkpoint rollback must restore {field}")


if __name__ == "__main__":
    main()
