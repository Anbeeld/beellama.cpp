import os
import tempfile

from utils import ServerPreset


NO_PRELOAD_SERVER_PRESETS = True


def _model_ids(server) -> set[str]:
    response = server.make_request("GET", "/models")
    assert response.status_code == 200
    return {item["id"] for item in response.body.get("data", [])}


def test_router_reload_is_authenticated_post_and_get_is_read_only():
    handle, preset_path = tempfile.mkstemp(prefix="router-reload-", suffix=".ini")
    os.close(handle)
    server = ServerPreset.router()
    server.models_preset = preset_path
    server.api_key = "sk-router-reload-secret"

    try:
        with open(preset_path, "w", encoding="utf-8") as preset:
            preset.write(
                "[model-reload-a]\n"
                "hf-repo = ggml-org/test-model-stories260K\n\n"
                "[model-reload-b]\n"
                "hf-repo = ggml-org/test-model-stories260K-infill\n"
            )
        server.start()
        initial_ids = _model_ids(server)
        assert {"model-reload-a", "model-reload-b"} <= initial_ids

        with open(preset_path, "w", encoding="utf-8") as preset:
            preset.write(
                "[model-reload-b]\n"
                "hf-repo = ggml-org/test-model-stories260K-infill\n\n"
                "[model-reload-c]\n"
                "hf-repo = ggml-org/test-model-stories260K\n"
            )

        get_response = server.make_request("GET", "/models?reload=1")
        assert get_response.status_code == 200
        get_ids = {item["id"] for item in get_response.body.get("data", [])}
        assert "model-reload-a" in get_ids
        assert "model-reload-b" in get_ids
        assert "model-reload-c" not in get_ids

        assert server.make_request("POST", "/models/reload", data={}).status_code == 401
        assert server.make_request(
            "POST",
            "/models/reload",
            data={},
            headers={"Authorization": "Bearer wrong"},
        ).status_code == 401

        auth = {"Authorization": f"Bearer {server.api_key}"}
        reload_response = server.make_request("POST", "/models/reload", data={}, headers=auth)
        assert reload_response.status_code == 200
        assert reload_response.body.get("success") is True
        final_ids = _model_ids(server)
        assert "model-reload-a" not in final_ids
        assert {"model-reload-b", "model-reload-c"} <= final_ids
    finally:
        server.stop()
        os.remove(preset_path)
