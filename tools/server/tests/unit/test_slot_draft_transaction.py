"""Local-model disk-slot restores must not publish half a target/draft epoch."""
import os
import struct
from pathlib import Path

import pytest
from utils import ServerProcess

NO_PRELOAD_SERVER_PRESETS = True


@pytest.fixture
def local_server(tmp_path):
    model = os.environ.get("SLOT_DRAFT_TEST_MODEL")
    if not model:
        pytest.skip("SLOT_DRAFT_TEST_MODEL is not set")
    assert Path(model).is_file()
    server = ServerProcess()
    server.model_file = model
    server.model_hf_repo = None
    server.model_hf_file = None
    server.offline = True
    server.model_draft = model
    server.spec_type = "draft-simple"
    server.spec_draft_n_max = 2
    server.n_gpu_layer = 999
    server.n_slots = 1
    server.n_ctx = 2048
    server.n_batch = 256
    server.n_ubatch = 128
    server.n_predict = 8
    server.temperature = 0.0
    server.seed = 12345
    server.slot_save_path = str(tmp_path)
    server.server_slots = True
    server.cache_ram = 0
    server.start()
    return server, tmp_path


def complete(server, prompt):
    response = server.make_request("POST", "/completion", data={
        "prompt": prompt, "id_slot": 0, "cache_prompt": True,
        "n_predict": 8, "temperature": 0.0, "seed": 12345,
        "return_tokens": True,
    })
    assert response.status_code == 200, response.body
    return response.body


def save(server, directory, name):
    response = server.make_request("POST", "/slots/0?action=save", data={"filename": name})
    assert response.status_code == 200, response.body
    data = (directory / name).read_bytes()
    assert response.body["n_written"] == len(data)
    return data


@pytest.mark.parametrize("corruption", ["draft_length", "speculative_payload"])
def test_corrupt_draft_appendix_keeps_live_slot(local_server, corruption):
    server, directory = local_server
    complete(server, "The capital of France is")
    saved = bytearray(save(server, directory, "source.bin"))
    appendix = saved.rfind(b"SDFT")
    assert appendix >= 0, "the saved slot must contain its live draft state"
    assert struct.unpack_from("<I", saved, appendix + 4)[0] == 1
    draft_size = struct.unpack_from("<Q", saved, appendix + 8)[0]
    assert draft_size > 0
    if corruption == "draft_length":
        struct.pack_into("<Q", saved, appendix + 8, 0xFFFFFFFFFFFFFFFF)
    else:
        spec_size_at = appendix + 16 + draft_size
        spec_size = struct.unpack_from("<Q", saved, spec_size_at)[0]
        if spec_size:
            saved[spec_size_at + 8] ^= 0xFF
        else:
            # Stateless drafters still must reject an unexpected nonempty frame.
            assert spec_size_at + 8 == len(saved)
            struct.pack_into("<Q", saved, spec_size_at, 1)
            saved.append(0xFF)
    (directory / "corrupt.bin").write_bytes(saved)

    complete(server, "A different live prompt explains the planets:")
    before = save(server, directory, "before.bin")
    response = server.make_request("POST", "/slots/0?action=restore", data={"filename": "corrupt.bin"})
    assert response.status_code == 400, response.body
    after = save(server, directory, "after.bin")
    assert after == before, "failed restore changed live target, draft, prompt, or checkpoints"


def test_saved_draft_restores_repeatedly(local_server):
    server, directory = local_server
    prompt = "The capital of France is"
    complete(server, prompt)
    original = save(server, directory, "source.bin")
    expected = complete(server, prompt + " Paris. The capital of Germany is")
    for i in range(2):
        response = server.make_request("POST", "/slots/0?action=restore", data={"filename": "source.bin"})
        assert response.status_code == 200, response.body
        assert response.body["n_read"] == len(original)
        restored = save(server, directory, f"restored-{i}.bin")
        assert restored == original
        actual = complete(server, prompt + " Paris. The capital of Germany is")
        assert actual["tokens"] == expected["tokens"]


def test_legacy_target_only_slot_requires_full_reevaluation(local_server):
    server, directory = local_server
    prompt = "The capital of France is"
    expected = complete(server, prompt)
    original = save(server, directory, "source.bin")
    appendix = original.rfind(b"SDFT")
    assert appendix > 0
    (directory / "legacy.bin").write_bytes(original[:appendix])
    complete(server, "A different live prompt explains the planets:")
    response = server.make_request("POST", "/slots/0?action=restore", data={"filename": "legacy.bin"})
    assert response.status_code == 200, response.body
    assert response.body["n_restored"] == 0
    actual = complete(server, prompt)
    assert actual["timings"]["cache_n"] == 0
    assert actual["tokens"] == expected["tokens"]


def test_draft_logprobs_are_target_probabilities(local_server):
    server, _ = local_server
    payload = {
        "prompt": "I believe the meaning of life is",
        "n_predict": 16,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_k": 1,
        "n_probs": 4,
        "return_tokens": True,
        "cache_prompt": False,
    }
    draft = server.make_request("POST", "/completion", data=payload)
    assert draft.status_code == 200, draft.body
    assert draft.body["timings"]["draft_n"] > 0
    server.stop()
    server.model_draft = None
    server.spec_type = None
    server.start()
    target = server.make_request("POST", "/completion", data=payload)
    assert target.status_code == 200, target.body
    assert target.body["timings"].get("draft_n", 0) == 0
    assert draft.body["tokens"] == target.body["tokens"]
    expected = target.body["completion_probabilities"]
    actual = draft.body["completion_probabilities"]
    assert len(actual) == len(expected) == 16
    for got, want in zip(actual, expected):
        assert got["id"] == want["id"]
        assert len(got["top_logprobs"]) == len(want["top_logprobs"])
        assert abs(got["logprob"] - want["logprob"]) < 0.25
    assert any(token["logprob"] < -1e-4 for token in actual)
