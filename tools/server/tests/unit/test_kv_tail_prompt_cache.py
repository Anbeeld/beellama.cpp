from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import pytest

from utils import ServerProcess


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402


NO_PRELOAD_SERVER_PRESETS = True


def _field(reader: gguf.GGUFReader, key: str):
    field = reader.fields.get(key)
    if field is None:
        raise AssertionError(f"required GGUF metadata {key!r} is missing")
    return field.contents()


def _validate_model(path: Path, *, kvarn: bool) -> tuple[str, int, int]:
    if not path.is_file():
        raise AssertionError(f"KV-tail test model does not exist: {path}")
    reader = gguf.GGUFReader(path, "r")
    arch = str(_field(reader, "general.architecture"))
    n_embd = int(_field(reader, f"{arch}.embedding_length"))
    n_head = int(_field(reader, f"{arch}.attention.head_count"))
    key_length = reader.fields.get(f"{arch}.attention.key_length")
    value_length = reader.fields.get(f"{arch}.attention.value_length")
    head_k = int(key_length.contents()) if key_length is not None else n_embd // n_head
    head_v = int(value_length.contents()) if value_length is not None else n_embd // n_head
    if kvarn:
        if arch not in {"qwen35", "qwen3next", "qwen3"}:
            raise AssertionError(f"KVarN server fixture requires a compatible Qwen architecture, got {arch!r}")
        if head_k not in {128, 256, 512} or head_v not in {128, 256, 512}:
            raise AssertionError(f"KVarN server fixture has unsupported K/V head dimensions {head_k}/{head_v}")
    elif head_k % 32 != 0 or head_v % 32 != 0:
        raise AssertionError(
            f"standard q4_0 fixture K/V head dimensions {head_k}/{head_v} are not block-aligned"
        )
    return arch, head_k, head_v


def _model_from_env(name: str, *, kvarn: bool) -> Path:
    value = os.environ.get(name)
    if not value:
        if kvarn:
            pytest.skip(f"{name} is not set; local KVarN server gate not run")
        raise AssertionError(f"required standard server fixture input {name} is not set")
    path = Path(value).resolve()
    _validate_model(path, kvarn=kvarn)
    return path


def _server(model: Path, *, unified: bool, kvarn: bool, log_tag: str) -> ServerProcess:
    server = ServerProcess()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = str(model)
    server.offline = True
    server.n_gpu_layer = 999
    server.n_slots = 2
    server.n_ctx = 2048
    server.n_batch = 512 if kvarn else 64
    server.n_ubatch = 128 if kvarn else 64
    server.n_predict = 4
    server.temperature = 0.0
    server.seed = 12345
    server.fa = "on"
    server.ctk = "kvarn4" if kvarn else "q4_0"
    server.ctv = "kvarn4" if kvarn else "q4_0"
    server.kv_tail_tokens = 512 if kvarn else 128
    server.kv_tail_type = "f16" if kvarn else "bf16"
    server.kv_unified = unified
    server.server_slots = True
    server.server_continuous_batching = True
    # Qwen3.6 hybrid state includes recurrent tensors and the KVarN record
    # payload; one exclusive sequence is roughly 411 MiB for this fixture.
    server.cache_ram = 1024 if kvarn else 256
    server.debug = True
    if kvarn:
        server.slot_save_path = tempfile.mkdtemp(prefix="kv-tail-slots-")
    fd, server.log_path = tempfile.mkstemp(prefix=f"kv-tail-{log_tag}-", suffix=".log")
    os.close(fd)
    return server


def _complete(server: ServerProcess, prompt: Sequence[int], slot: int, *, cache: bool = True):
    response = server.make_request("POST", "/completion", data={
        "prompt": list(prompt),
        "id_slot": slot,
        "cache_prompt": cache,
        "n_predict": 4,
        "return_tokens": True,
        "seed": 12345,
        "temperature": 0.0,
    })
    assert response.status_code == 200, response.body
    timings = response.body["timings"]
    assert timings["prompt_n"] > 0
    assert timings["cache_n"] >= 0
    return response.body


def _tokenize(server: ServerProcess, text: str, *, add_special: bool) -> list[int]:
    response = server.make_request("POST", "/tokenize", data={
        "content": text,
        "add_special": add_special,
    })
    assert response.status_code == 200, response.body
    return response.body["tokens"]


def _prefix_near(server: ServerProcess, target: int, label: str) -> tuple[list[int], int]:
    text = f"{label}:\n"
    while True:
        text += "alpha beta gamma delta epsilon zeta eta theta. "
        tokens = _tokenize(server, text + "\n", add_special=True)
        if len(tokens) >= target:
            return tokens, len(tokens)


@pytest.mark.parametrize("unified", [False, True], ids=["nonunified", "unified"])
def test_standard_two_slot_cumulative_prompt_reuse(unified: bool):
    model = _model_from_env("KV_TAIL_TEST_STD_MODEL", kvarn=False)
    server = _server(model, unified=unified, kvarn=False, log_tag=f"std-{unified}")
    server.start(timeout_seconds=180)

    prompts: dict[int, list[int]] = {}
    for slot in (0, 1):
        prefix, count = _prefix_near(server, 190, f"slot {slot} deterministic history")
        assert count > 128
        prompts[slot] = prefix

    records = []
    for turn in range(4):
        for slot in (0, 1):
            prompt = prompts[slot] + _tokenize(
                server,
                f"Question {turn}: What is the capital of France?\nAnswer:",
                add_special=False,
            )
            body = _complete(server, prompt, slot, cache=True)
            if turn == 0:
                assert body["timings"]["cache_n"] == 0
            else:
                assert body["timings"]["cache_n"] > 128
            records.append((list(prompt), body["content"], body["timings"]["prompt_n"]))
            prompts[slot] = prompt + body["tokens"] + _tokenize(server, "\n", add_special=False)

    server.stop()
    oracle = _server(model, unified=unified, kvarn=False, log_tag=f"std-oracle-{unified}")
    oracle.start(timeout_seconds=180)
    for prompt, expected, cached_prompt_n in records:
        body = _complete(oracle, prompt, 0, cache=False)
        assert body["content"] == expected
        assert body["timings"]["cache_n"] == 0
        assert body["timings"]["prompt_n"] >= cached_prompt_n


@pytest.mark.kvarn_local
def test_kvarn_nonunified_hybrid_reuse_and_safe_divergence():
    model = _model_from_env("KV_TAIL_TEST_KVARN_MODEL", kvarn=True)
    server = _server(model, unified=False, kvarn=True, log_tag="kvarn-nonunified")
    server.start(timeout_seconds=600)

    common, common_n = _prefix_near(server, 333, "nonunified historical common prefix")
    assert common_n >= 128 and common_n % 128 != 0
    original = common + _tokenize(server, "old branch token " * 180 + "\nAssistant:", add_special=False)
    divergent = common + _tokenize(server, "new historical branch.\nAssistant:", add_special=False)
    _complete(server, original, 0, cache=True)
    reprocessed = _complete(server, divergent, 0, cache=True)
    # Qwen3.6 is a hybrid recurrent model. With n_rs_seq=0 the recurrent
    # child cannot reconstruct an older state, so its truthful composite plan
    # must reject even though the KVarN attention child can round to a group.
    assert reprocessed["timings"]["cache_n"] == 0
    oracle = _complete(server, divergent, 0, cache=False)
    assert reprocessed["tokens"] == oracle["tokens"]
    assert oracle["timings"]["cache_n"] == 0

    continued_prompt = divergent + oracle["tokens"] + _tokenize(server, "\nContinue:\n", add_special=False)
    continued = _complete(server, continued_prompt, 0, cache=True)
    assert continued["timings"]["cache_n"] > common_n

    exact_common, exact_n = _prefix_near(server, 690, "nonunified exact common prefix")
    exact_original = exact_common + _tokenize(server, "old exact suffix " * 30 + "\nAssistant:", add_special=False)
    exact_divergent = exact_common + _tokenize(server, "new exact suffix.\nAssistant:", add_special=False)
    _complete(server, exact_original, 0, cache=True)
    exact = _complete(server, exact_divergent, 0, cache=True)
    assert exact_n > 512
    assert 0 < exact["timings"]["cache_n"] <= exact_n
    assert exact["timings"]["cache_n"] % 128 != 0
    exact_oracle = _complete(server, exact_divergent, 0, cache=False)
    assert exact["tokens"] == exact_oracle["tokens"]


@pytest.mark.kvarn_local
def test_kvarn_unified_contention_and_exclusive_state_reuse():
    model = _model_from_env("KV_TAIL_TEST_KVARN_MODEL", kvarn=True)
    server = _server(model, unified=True, kvarn=True, log_tag="kvarn-unified")
    server.cache_ram = 0
    server.start(timeout_seconds=600)

    common, common_n = _prefix_near(server, 333, "unified historical common prefix")
    original = common + _tokenize(server, "old branch token " * 180 + "\nAssistant:", add_special=False)
    divergent = common + _tokenize(server, "new historical branch.\nAssistant:", add_special=False)
    other, _ = _prefix_near(server, 420, "other live unified slot")
    other += _tokenize(server, "Assistant:", add_special=False)

    _complete(server, original, 0, cache=True)
    other_first = _complete(server, other, 1, cache=True)
    contended = _complete(server, divergent, 0, cache=True)
    assert contended["timings"]["cache_n"] == 0
    other_next = other + other_first["tokens"] + _tokenize(server, "\nContinue:\n", add_special=False)
    other_again = _complete(server, other_next, 1, cache=True)
    assert other_again["timings"]["cache_n"] > common_n
    server.stop()

    # Exercise RAM state independently from the live-contention case. Build a
    # sequence while exclusive; starting slot 1 saves slot 0, but restoring it
    # while slot 1 is live must be a safe miss and full reevaluation.
    cache_server = _server(model, unified=True, kvarn=True, log_tag="kvarn-unified-ram")
    cache_server.start(timeout_seconds=600)
    exclusive_base = _complete(cache_server, original, 0, cache=True)
    blocker, _ = _prefix_near(cache_server, 160, "unified restore blocker")
    _complete(cache_server, blocker, 1, cache=True)
    exclusive_next = original + exclusive_base["tokens"] + _tokenize(cache_server, "\nContinue:\n", add_special=False)
    refused = _complete(cache_server, exclusive_next, 0, cache=True)
    assert refused["timings"]["cache_n"] == 0

    # Starting slot 1 again saves the now-exclusive slot 0 state. Once slot 1
    # is erased, the exact same RAM entry is safe to restore and append to.
    _complete(cache_server, blocker, 1, cache=True)
    erased = cache_server.make_request("POST", "/slots/1?action=erase")
    assert erased.status_code == 200, erased.body
    refused_next = exclusive_next + refused["tokens"] + _tokenize(cache_server, "\nContinue again:\n", add_special=False)
    restored = _complete(cache_server, refused_next, 0, cache=True)
    assert restored["timings"]["cache_n"] > common_n
    cache_server.stop()
