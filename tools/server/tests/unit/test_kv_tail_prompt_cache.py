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
    server.n_batch = 512
    server.n_ubatch = 128 if kvarn else 256
    server.n_predict = 4
    server.temperature = 0.0
    server.seed = 12345
    server.fa = "on"
    server.ctk = "kvarn4" if kvarn else "q4_0"
    server.ctv = "kvarn4" if kvarn else "q4_0"
    server.kv_tail_tokens = 512
    server.kv_tail_type = "f16"
    server.kv_unified = unified
    server.server_slots = True
    server.server_continuous_batching = True
    # Qwen3.6 hybrid state includes recurrent tensors and the KVarN record
    # payload; one exclusive sequence is roughly 411 MiB for this fixture.
    server.cache_ram = 1024 if kvarn else 256
    server.ctx_checkpoints = 32
    server.checkpoint_min_step = 512
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
        "n_probs": 5,
        "return_tokens": True,
        "seed": 12345,
        "temperature": 0.0,
    })
    assert response.status_code == 200, response.body
    timings = response.body["timings"]
    assert timings["prompt_n"] > 0
    assert timings["cache_n"] >= 0
    return response.body


def _assert_first_target_token(expected_tokens: Sequence[int], actual_tokens: Sequence[int]) -> None:
    """Require the state-sensitive first target token, not batch-partition identity."""
    assert expected_tokens
    assert actual_tokens
    assert actual_tokens[0] == expected_tokens[0]


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
            records.append((list(prompt), turn, slot, body["tokens"], body["timings"]["prompt_n"]))
            prompts[slot] = prompt + body["tokens"] + _tokenize(server, "\n", add_special=False)

    server.stop()
    oracle = _server(model, unified=unified, kvarn=False, log_tag=f"std-oracle-{unified}")
    oracle.start(timeout_seconds=180)
    for prompt, turn, slot, expected_tokens, cached_prompt_n in records:
        body = _complete(oracle, prompt, slot, cache=False)
        assert body["tokens"]
        assert body["tokens"][0] == expected_tokens[0], (turn, slot, expected_tokens, body["tokens"])
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
    # The prior long branch has durable checkpoints only beyond this divergent
    # prefix, so no earlier KVarN G128 boundary is eligible for restore.
    assert reprocessed["timings"]["cache_n"] == 0
    oracle = _complete(server, divergent, 0, cache=False)
    _assert_first_target_token(oracle["tokens"], reprocessed["tokens"])
    assert oracle["timings"]["cache_n"] == 0

    continued_prompt = divergent + oracle["tokens"] + _tokenize(server, "\nContinue:\n", add_special=False)
    continued = _complete(server, continued_prompt, 0, cache=True)
    timings = continued["timings"]
    assert timings["cache_reason"] == "committed"
    assert timings["cache_n"] == timings["cache_planned_n"]
    assert common_n < timings["cache_n"] <= len(continued_prompt)
    assert timings["cache_reprocessed_n"] == timings["prompt_n"]
    continued_oracle = _complete(server, continued_prompt, 0, cache=False)
    _assert_first_target_token(continued_oracle["tokens"], continued["tokens"])

    exact_common, exact_n = _prefix_near(server, 690, "nonunified exact common prefix")
    exact_original = exact_common + _tokenize(server, "old exact suffix " * 30 + "\nAssistant:", add_special=False)
    exact_divergent = exact_common + _tokenize(server, "new exact suffix.\nAssistant:", add_special=False)
    _complete(server, exact_original, 0, cache=True)
    exact = _complete(server, exact_divergent, 0, cache=True)
    assert exact_n > 512
    assert 0 < exact["timings"]["cache_n"] <= exact_n
    assert exact["timings"]["cache_n"] % 128 == 0
    exact_oracle = _complete(server, exact_divergent, 0, cache=False)
    _assert_first_target_token(exact_oracle["tokens"], exact["tokens"])


@pytest.mark.kvarn_local
def test_kvarn_unified_contention_and_repeatable_state_reuse():
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

    # Force slot 0 to park two unrelated conversations while slot 1 remains
    # live. The durable entries must use the self-contained selective KVarN
    # representation, which is safe to restore after their source cells move.
    cache_server = _server(model, unified=True, kvarn=True, log_tag="kvarn-unified-ram")
    cache_server.cache_ram = 8192
    cache_server.start(timeout_seconds=600)
    original_base = _complete(cache_server, original, 0, cache=True)
    blocker, _ = _prefix_near(cache_server, 160, "unified restore blocker")
    _complete(cache_server, blocker, 1, cache=True)

    unrelated, _ = _prefix_near(cache_server, 510, "unified parked unrelated branch")
    unrelated += _tokenize(cache_server, "different parked suffix.\nAssistant:", add_special=False)
    unrelated_base = _complete(cache_server, unrelated, 0, cache=True)

    original_next = original + original_base["tokens"] + _tokenize(
        cache_server, "\nContinue original:\n", add_special=False)
    restored_original = _complete(cache_server, original_next, 0, cache=True)
    assert restored_original["timings"]["cache_n"] > common_n
    assert restored_original["timings"]["cache_source"] == "ram"

    unrelated_next = unrelated + unrelated_base["tokens"] + _tokenize(
        cache_server, "\nContinue unrelated:\n", add_special=False)
    restored_unrelated = _complete(cache_server, unrelated_next, 0, cache=True)
    unrelated_timings = restored_unrelated["timings"]
    assert unrelated_timings["cache_n"] == len(unrelated) + len(unrelated_base["tokens"]) - 1
    assert unrelated_timings["cache_n"] == unrelated_timings["cache_planned_n"]
    assert unrelated_timings["cache_source"] == "ram"

    # Revisit the original branch again without clearing the still-live blocker
    # in slot 1. The longer continuation parked after the first RAM restore may
    # supersede the shorter RAM entry, so its local checkpoint is also a valid
    # winning source; either way, the branch must not reset.
    original_alternate = original + original_base["tokens"] + _tokenize(
        cache_server, "\nTake an alternate continuation:\n", add_special=False)
    restored_again = _complete(cache_server, original_alternate, 0, cache=True)
    assert restored_again["timings"]["cache_n"] > common_n
    assert restored_again["timings"]["cache_source"] in {"ram", "checkpoint"}
    cache_server.stop()

    oracle_server = _server(model, unified=True, kvarn=True, log_tag="kvarn-unified-ram-oracle")
    oracle_server.cache_ram = 0
    oracle_server.start(timeout_seconds=600)
    original_oracle = _complete(oracle_server, original_next, 0, cache=False)
    unrelated_oracle = _complete(oracle_server, unrelated_next, 0, cache=False)
    assert original_oracle["timings"]["cache_n"] == 0
    assert unrelated_oracle["timings"]["cache_n"] == 0
    _assert_first_target_token(original_oracle["tokens"], restored_original["tokens"])
    _assert_first_target_token(unrelated_oracle["tokens"], restored_unrelated["tokens"])
    oracle_server.stop()
