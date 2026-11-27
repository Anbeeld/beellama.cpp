import pytest

from utils import *


server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def fresh_server():
    global server
    server = ServerPreset.tinyllama2()


def _single_token_bias(content: str, bias: float = 100.0) -> dict:
    global server
    res = server.make_request("POST", "/tokenize", data={
        "content": content,
    })
    assert res.status_code == 200
    assert len(res.body["tokens"]) >= 1
    return {str(res.body["tokens"][-1]): bias}


def test_reasoning_loop_guard_request_settings():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 1,
        "reasoning_loop_guard": "stop",
        "reasoning_loop_min_tokens": 768,
        "reasoning_loop_window": 1536,
        "reasoning_loop_max_period": 256,
        "reasoning_loop_min_coverage": 768,
        "reasoning_loop_check_interval": 16,
        "reasoning_loop_interventions": 0,
    })

    assert res.status_code == 200
    assert res.body["tokens_predicted"] == 1
    assert res.body["stop_type"] == "limit"
    assert res.body["stop_detail"] == "token_limit"

    settings = res.body["generation_settings"]
    assert settings["reasoning_loop_guard"] == "stop"
    assert settings["reasoning_loop_min_tokens"] == 768
    assert settings["reasoning_loop_window"] == 1536
    assert settings["reasoning_loop_max_period"] == 256
    assert settings["reasoning_loop_min_coverage"] == 768
    assert settings["reasoning_loop_check_interval"] == 16
    assert settings["reasoning_loop_interventions"] == 0


def test_reasoning_loop_guard_off_keeps_hard_limit():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 2,
        "reasoning_loop_guard": "off",
    })

    assert res.status_code == 200
    assert res.body["tokens_predicted"] == 2
    assert res.body["stop_type"] == "limit"
    assert res.body["stop_detail"] == "token_limit"
    assert res.body["generation_settings"]["reasoning_loop_guard"] == "off"


@pytest.mark.parametrize("budget", [-1, 64])
def test_reasoning_loop_guard_off_keeps_reasoning_accounting(budget):
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Repeat a short hidden thought:",
        "n_predict": 12,
        "temperature": 0.0,
        "logit_bias": _single_token_bias("a"),
        "reasoning_budget_start_tag": "a",
        "reasoning_budget_end_tag": "b",
        "reasoning_budget_tokens": budget,
        "reasoning_loop_guard": "off",
    })

    assert res.status_code == 200
    assert res.body["tokens_predicted"] == 12
    assert res.body["reasoning_tokens"] == 12
    assert res.body["visible_completion_tokens"] == 0
    assert "loop_guard" not in res.body


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["off", "force-close"])
@pytest.mark.parametrize("speculative", [False, True])
def test_reasoning_loop_guard_chat_usage_and_event(stream, mode, speculative):
    global server
    if speculative:
        server.spec_type = "draft-simple"
        server.model_draft_hf_repo = server.model_hf_repo
        server.spec_draft_n_min = 1
        server.spec_draft_n_max = 3
    server.start()
    body = {
        "messages": [{"role": "user", "content": "Repeat a short hidden thought:"}],
        "stream": stream,
        "n_predict": 12,
        "temperature": 0.0,
        "logit_bias": _single_token_bias("a"),
        "reasoning_budget_start_tag": "a" if mode == "off" else "a a",
        "reasoning_budget_end_tag": "b",
        "reasoning_budget_message": "!",
        "reasoning_loop_guard": mode,
        "reasoning_loop_min_tokens": 4,
        "reasoning_loop_window": 6,
        "reasoning_loop_max_period": 2,
        "reasoning_loop_min_coverage": 3,
        "reasoning_loop_check_interval": 1,
        "reasoning_loop_interventions": 1,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
        chunks = list(server.make_stream_request("POST", "/v1/chat/completions", data=body))
        usage = next(chunk["usage"] for chunk in chunks if "usage" in chunk)
        events = [chunk["loop_guard"] for chunk in chunks if "loop_guard" in chunk]
        timings = next(chunk["timings"] for chunk in chunks if "timings" in chunk)
    else:
        res = server.make_request("POST", "/v1/chat/completions", data=body)
        assert res.status_code == 200
        usage = res.body["usage"]
        events = [res.body["loop_guard"]] if "loop_guard" in res.body else []
        timings = res.body["timings"]

    if speculative:
        assert timings["draft_n"] > 0
        # The target-only bias rejects the draft's unconstrained proposals.
        assert timings["draft_n_accepted"] == 0
    details = usage["completion_tokens_details"]
    assert usage["completion_tokens"] == 12
    assert details["reasoning_tokens"] + details["visible_tokens"] == 12
    if mode == "off":
        assert details["reasoning_tokens"] == 12
        assert events == []
    else:
        assert details["reasoning_tokens"] >= 4
        assert len(events) == 1
        assert events[0]["region"] == "reasoning"
        assert events[0]["action"] == "force-close"


def test_reasoning_loop_guard_off_counts_accepted_drafts():
    global server
    server.spec_type = "draft-simple"
    server.model_draft_hf_repo = server.model_hf_repo
    server.spec_draft_n_min = 1
    server.spec_draft_n_max = 3
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 32,
        "ignore_eos": True,
        "temperature": 0.0,
        "reasoning_loop_guard": "off",
    })
    assert res.status_code == 200
    assert res.body["timings"]["draft_n_accepted"] > 0
    assert res.body["tokens_predicted"] == 32
    assert res.body["reasoning_tokens"] == 0
    assert res.body["visible_completion_tokens"] == 32
    assert "loop_guard" not in res.body


def test_reasoning_loop_guard_invalid_request_rejected():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 1,
        "reasoning_loop_guard": "force-close",
        "reasoning_loop_min_tokens": 768,
        "reasoning_loop_window": 768,
        "reasoning_loop_max_period": 512,
        "reasoning_loop_min_coverage": 768,
    })

    assert res.status_code == 400


def test_reasoning_loop_guard_stop_triggers_on_hidden_reasoning_loop():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "Repeat a short hidden thought:",
        "n_predict": 16,
        "temperature": 0.0,
        "logit_bias": _single_token_bias("a"),
        "reasoning_budget_start_tag": "a",
        "reasoning_budget_end_tag": "b",
        "reasoning_loop_guard": "stop",
        "reasoning_loop_min_tokens": 4,
        "reasoning_loop_window": 6,
        "reasoning_loop_max_period": 2,
        "reasoning_loop_min_coverage": 3,
        "reasoning_loop_check_interval": 1,
        "reasoning_loop_interventions": 0,
    })

    assert res.status_code == 200
    assert res.body["stop_type"] == "limit"
    assert res.body["stop_detail"] == "reasoning_loop_guard"
    assert res.body["tokens_predicted"] < 16
    assert res.body["reasoning_tokens"] >= 4
    assert res.body["visible_completion_tokens"] == 0
    assert res.body["loop_guard"]["triggered"] is True
    assert res.body["loop_guard"]["action"] == "stop"


def test_reasoning_loop_guard_force_close_intervenes_then_continues():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "Repeat a short hidden thought:",
        "n_predict": 12,
        "temperature": 0.0,
        "logit_bias": _single_token_bias("a"),
        # Generate the start marker instead of relying on a prompt prefill.
        "reasoning_budget_start_tag": "a a",
        "reasoning_budget_end_tag": "b",
        "reasoning_loop_guard": "force-close",
        "reasoning_loop_min_tokens": 4,
        "reasoning_loop_window": 6,
        "reasoning_loop_max_period": 2,
        "reasoning_loop_min_coverage": 3,
        "reasoning_loop_check_interval": 1,
        "reasoning_loop_interventions": 1,
        "reasoning_budget_message": "!",
    })

    assert res.status_code == 200
    assert "!" not in res.body["content"]
    assert res.body["tokens_predicted"] == 12
    assert res.body["stop_type"] == "limit"
    assert res.body["stop_detail"] == "token_limit"
    assert res.body["reasoning_tokens"] >= 4
    assert res.body["visible_completion_tokens"] > 0
    assert res.body["loop_guard"]["triggered"] is True
    assert res.body["loop_guard"]["action"] == "force-close"
