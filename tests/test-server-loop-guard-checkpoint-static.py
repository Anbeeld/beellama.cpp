#!/usr/bin/env python3
"""Guard loop-guard server integration and speculative rollback semantics."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    source = (ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")

    accept_start = source.find("bool handle_loop_guard_accept(")
    accept_end = source.find("common_sampler_accept_callback make_loop_guard_accept_callback", accept_start)
    require(accept_start >= 0 and accept_end >= 0, "loop-guard acceptance handler not found")
    accept = source[accept_start:accept_end]

    require(
        "SERVER_LOOP_REGION_VISIBLE" in accept,
        "visible generated output must select the visible loop-guard region",
    )
    require(
        "slot.loop_guard.accept(info.token, region);" in accept,
        "all accepted generated tokens must enter their selected detector region",
    )
    require(
        "slot.loop_guard.should_check(region," in accept and "slot.loop_guard.check(region)" in accept,
        "loop checks must use the selected reasoning or visible region",
    )
    require(
        "region == SERVER_LOOP_REGION_REASONING" in accept,
        "force-close must remain restricted to hidden reasoning loops",
    )
    require(
        "slot.stop_detail = \"reasoning_loop_guard\";" in accept,
        "visible loops must retain the existing hard-stop compatibility detail",
    )
    require(
        "slot.loop_guard.reset()" not in accept,
        "the detector must remain armed after a force-close intervention",
    )

    verify_start = source.find("// verify and try to accept the draft")
    verify_end = source.find("const int64_t t_now = ggml_time_us();", verify_start)
    require(verify_start >= 0 and verify_end >= 0, "speculative verification block not found")
    verify = source[verify_start:verify_end]

    require(
        "const server_loop_guard loop_guard_save = slot.loop_guard;" in verify,
        "speculative verification must snapshot loop-guard state before sampler acceptance",
    )
    sampler_restore = verify.find("common_sampler_copy(smpl_save.get(), slot.smpl.get());")
    require(sampler_restore >= 0, "speculative checkpoint rollback must restore the sampler")
    restore = verify[sampler_restore:]
    for field in (
        "slot.loop_guard = loop_guard_save;",
        "slot.loop_guard_interventions = loop_guard_interventions_save;",
        "slot.loop_guard_event = loop_guard_event_save;",
        "slot.reasoning_output_tokens = reasoning_output_tokens_save;",
        "slot.visible_output_tokens = visible_output_tokens_save;",
        "slot.has_next_token = has_next_token_save;",
        "slot.stop = stop_save;",
        "slot.stop_detail = stop_detail_save;",
    ):
        require(field in restore, f"checkpoint rollback must restore {field}")


if __name__ == "__main__":
    main()
