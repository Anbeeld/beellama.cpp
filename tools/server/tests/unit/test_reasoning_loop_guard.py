from utils import *


server = ServerPreset.tinyllama2()


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
