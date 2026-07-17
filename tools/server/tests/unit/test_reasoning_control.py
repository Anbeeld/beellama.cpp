from utils import *


NO_PRELOAD_SERVER_PRESETS = True
server = ServerPreset.tinyllama2()


def _start_stream(*, active_reasoning: bool):
    start_tag = "a"
    body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Count slowly, then answer."}],
        "stream": True,
        "n_predict": 128,
        "temperature": 0.0,
        "reasoning_control": True,
        "reasoning_budget_start_tag": start_tag,
        "reasoning_budget_end_tag": "b",
    }
    if active_reasoning:
        tokenized = server.make_request("POST", "/tokenize", data={"content": start_tag})
        assert tokenized.status_code == 200
        assert len(tokenized.body["tokens"]) == 1
        body["logit_bias"] = {str(tokenized.body["tokens"][0]): 100.0}

    stream = server.make_stream_request("POST", "/v1/chat/completions", data=body)
    completion_id = None
    for chunk in stream:
        completion_id = chunk["id"]
        delta = chunk["choices"][0]["delta"] if chunk.get("choices") else {}
        if not active_reasoning or delta.get("content") or delta.get("reasoning_content"):
            break
    assert completion_id is not None
    return stream, completion_id


def test_reasoning_control_reports_actual_sampler_transition():
    global server
    server.start()

    inactive_stream, inactive_id = _start_stream(active_reasoning=False)
    try:
        inactive = server.make_request("POST", "/v1/chat/completions/control", data={
            "id": inactive_id,
            "action": "reasoning_end",
        })
        assert inactive.status_code == 200
        assert inactive.body == {
            "success": False,
            "message": "reasoning is not active",
        }
    finally:
        inactive_stream.close()

    active_stream, active_id = _start_stream(active_reasoning=True)
    try:
        active = server.make_request("POST", "/v1/chat/completions/control", data={
            "id": active_id,
            "action": "reasoning_end",
        })
        assert active.status_code == 200
        assert active.body == {"success": True}
    finally:
        active_stream.close()


def test_reasoning_control_rejects_unknown_completed_and_malformed_targets():
    global server
    server.start()

    unknown = server.make_request("POST", "/v1/chat/completions/control", data={
        "id": "chatcmpl-does-not-exist",
        "action": "reasoning_end",
    })
    assert unknown.status_code == 200
    assert unknown.body == {
        "success": False,
        "message": "no active completion for this id",
    }

    completed_id = None
    completed_stream = server.make_stream_request("POST", "/v1/chat/completions", data={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Answer with one token."}],
        "stream": True,
        "n_predict": 1,
        "reasoning_control": True,
        "reasoning_budget_start_tag": "a",
        "reasoning_budget_end_tag": "b",
    })
    for chunk in completed_stream:
        completed_id = chunk["id"]
    assert completed_id is not None
    completed = server.make_request("POST", "/v1/chat/completions/control", data={
        "id": completed_id,
        "action": "reasoning_end",
    })
    assert completed.status_code == 200
    assert completed.body == {
        "success": False,
        "message": "no active completion for this id",
    }

    missing = server.make_request("POST", "/v1/chat/completions/control", data={
        "action": "reasoning_end",
    })
    assert missing.status_code == 400

    malformed = server.make_request("POST", "/v1/chat/completions/control", data={
        "id": "chatcmpl-does-not-exist",
        "action": "stop_everything",
    })
    assert malformed.status_code == 400
