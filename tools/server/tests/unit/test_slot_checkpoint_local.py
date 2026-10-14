from __future__ import annotations

import os
import socket
import struct
import sys
from pathlib import Path

import pytest

from utils import ServerProcess


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402


NO_PRELOAD_SERVER_PRESETS = True


@pytest.fixture
def model(request) -> Path:
    spec = getattr(request, "param", "none")
    variable = "SLOT_CKPT_TEST_MTP_MODEL" if spec == "mtp" else "SLOT_CKPT_TEST_MODEL"
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"{variable} is not set; local hybrid checkpoint gate not run")
    path = Path(value).resolve()
    assert path.is_file(), f"local model does not exist: {path}"
    reader = gguf.GGUFReader(path, "r")
    field = reader.fields.get("general.architecture")
    assert field is not None, "required general.architecture metadata is missing"
    arch = str(field.contents())
    assert arch in {"qwen35", "qwen3next"}, f"expected hybrid qwen35/qwen3next, got {arch!r}"
    if spec == "mtp":
        field = reader.fields.get(f"{arch}.nextn_predict_layers")
        assert field is not None and int(field.contents()) > 0, "MTP fixture requires owned draft weights"
    return path


def _server(model: Path, cache: str, directory: Path, tag: str, spec: str = "none") -> ServerProcess:
    directory.mkdir(parents=True, exist_ok=True)
    server = ServerProcess()
    # Background runners may reserve their injected PORT for their own proxy.
    # Each local fixture owns its server, so select an available loopback port.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        server.server_port = sock.getsockname()[1]
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = str(model)
    server.offline = True
    server.n_gpu_layer = 999
    server.fa = "on"
    server.ctk = cache
    server.ctv = cache
    server.n_slots = 2
    server.n_ctx = 2048
    server.n_batch = 256
    server.n_ubatch = 128
    server.n_predict = 1
    server.temperature = 0.0
    server.seed = 12345
    server.kv_unified = False
    server.cache_ram = 0
    server.ctx_checkpoints = 32
    server.checkpoint_min_step = 128
    server.server_slots = True
    server.server_continuous_batching = True
    server.slot_save_path = str(directory)
    server.log_path = str(directory / f"{tag}.log")
    server.debug = True
    if spec == "mtp":
        server.spec_type = "draft-mtp"
        server.spec_draft_n_max = 4
    assert not server.external_server, "local checkpoint tests must own server restart and configuration"
    return server


def _tokenize(server: ServerProcess, text: str, *, special: bool) -> list[int]:
    response = server.make_request("POST", "/tokenize", data={
        "content": text, "add_special": special,
    })
    assert response.status_code == 200, response.body
    return response.body["tokens"]


def _prompts(server: ServerProcess) -> tuple[list[int], list[int], list[int]]:
    # Token lists avoid tokenizer boundary changes when replacing the suffix.
    tokens = _tokenize(server, "Historical checkpoint exercise.\n" +
                       "alpha beta gamma delta epsilon zeta eta theta. " * 140, special=True)
    assert len(tokens) >= 704
    original = tokens[:704]
    suffix = _tokenize(server, "Different historical branch. What is seven plus eight?\nAnswer:",
                       special=False)
    assert suffix and suffix[0] != original[513], "branch must diverge immediately after token 512"
    branch = original[:513] + suffix
    return original[:384], original, branch


def _complete(server: ServerProcess, prompt: list[int], slot: int, *, cache: bool = True) -> dict:
    response = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": slot,
        "cache_prompt": cache,
        "n_predict": 1,
        "return_tokens": True,
        "temperature": 0.0,
        "seed": 12345,
    })
    assert response.status_code == 200, response.body
    assert len(response.body["tokens"]) == 1, response.body
    assert response.body["timings"]["prompt_n"] > 0
    return response.body


def _slot(server: ServerProcess, slot: int, action: str, filename: str) -> dict:
    response = server.make_request("POST", f"/slots/{slot}?action={action}",
                                   data={"filename": filename})
    assert response.status_code == 200, response.body
    return response.body


def _oracle(model: Path, cache: str, directory: Path, prompts: list[list[int]], spec: str = "none") -> list[dict]:
    server = _server(model, cache, directory, "oracle", spec)
    server.start(timeout_seconds=600)
    try:
        results = [_complete(server, prompt, 0, cache=False) for prompt in prompts]
        for result in results:
            assert result["timings"]["cache_n"] == 0, result
        return results
    finally:
        server.stop()


def _assert_reuse(actual: dict, cold: dict) -> None:
    assert actual["tokens"][0] == cold["tokens"][0], (actual, cold)
    assert actual["timings"]["cache_n"] > 0, actual
    assert actual["timings"]["prompt_n"] < cold["timings"]["prompt_n"], (actual, cold)


@pytest.mark.parametrize("model,spec", [("none", "none"), ("mtp", "mtp")], indirect=["model"], ids=["target", "mtp"])
@pytest.mark.parametrize("cache", ["f16", "kvarn4"])
def test_historical_checkpoint_roundtrip(model: Path, cache: str, spec: str, tmp_path: Path):
    directory = tmp_path / "slots"
    filename = "historical-分岐-état.bin"
    server = _server(model, cache, directory, "save", spec)
    server.start(timeout_seconds=600)
    records = []
    try:
        anchor, original, branch = _prompts(server)
        _complete(server, anchor, 0)
        seeded = _complete(server, original, 0)
        assert seeded["timings"]["cache_n"] > 0, seeded
        saved = _slot(server, 0, "save", filename)
        assert saved["n_saved"] >= len(original)
        assert saved["n_written"] == (directory / filename).stat().st_size
        if spec == "mtp":
            _, entries = _appendix((directory / filename).read_bytes())
            assert all(blobs[1][2] > 0 and blobs[2][2] > 0 for _, blobs in entries), "MTP draft/spec state must be persisted"
        records.append(_complete(server, branch, 0))
        restored = _slot(server, 1, "restore", filename)
        assert restored["n_restored"] == saved["n_saved"]
        assert restored["n_read"] == saved["n_written"]
        records.append(_complete(server, branch, 1))
    finally:
        server.stop()

    restarted = _server(model, cache, directory, "restart", spec)
    restarted.start(timeout_seconds=600)
    try:
        restored = _slot(restarted, 1, "restore", filename)
        assert restored["n_restored"] == saved["n_saved"]
        assert restored["n_read"] == (directory / filename).stat().st_size
        records.append(_complete(restarted, branch, 1))
    finally:
        restarted.stop()
    cold = _oracle(model, cache, tmp_path / "oracle", [branch], spec)[0]
    for record in records:
        _assert_reuse(record, cold)


def _appendix(data: bytes) -> tuple[int, list[tuple[int, list[tuple[int, int, int]]]]]:
    # SCKP v1: u32 magic/version/count; each entry has i64 n_tokens,
    # i32 pos_min/pos_max, then three u64-size-prefixed state blobs.
    # Only accept a structurally complete suffix to avoid matching model bytes.
    offset = data.rfind(b"SCKP")
    assert offset >= 12, "saved slot has no checkpoint appendix"
    magic, version, count = struct.unpack_from("=III", data, offset)
    assert magic == 0x504B4353 and version == 1 and count > 0
    cursor = offset + 12
    entries = []
    for _ in range(count):
        metadata = cursor
        assert cursor + 16 <= len(data)
        cursor += 16
        blobs = []
        for _ in range(3):
            assert cursor + 8 <= len(data)
            length_offset = cursor
            size, = struct.unpack_from("=Q", data, cursor)
            cursor += 8
            assert size <= len(data) - cursor
            blobs.append((length_offset, cursor, size))
            cursor += size
        entries.append((metadata, blobs))
    assert cursor == len(data)
    return offset, entries


@pytest.mark.parametrize("cache", ["f16", "kvarn4"])
def test_checkpoint_appendix_damage_falls_back(model: Path, cache: str, tmp_path: Path):
    directory = tmp_path / "slots"
    server = _server(model, cache, directory, "damage")
    server.start(timeout_seconds=600)
    records = []
    try:
        anchor, original, branch = _prompts(server)
        _complete(server, anchor, 0)
        _complete(server, original, 0)
        saved = _slot(server, 0, "save", "original.bin")
        data = (directory / "original.bin").read_bytes()
        assert saved["n_written"] == len(data)
        offset, entries = _appendix(data)

        oversized = bytearray(data)
        # Below the original PR's 16 GiB limit, but larger than this file.
        struct.pack_into("=Q", oversized, entries[0][1][0][0], 1 << 33)
        corrupted = bytearray(data)
        for _, blobs in entries:
            _, start, size = blobs[0]
            assert size >= 4
            corrupted[start:start + 4] = b"BAD!"
        invalid_metadata = bytearray(data)
        struct.pack_into("=q", invalid_metadata, entries[0][0], -1)
        invalid_count = bytearray(data)
        struct.pack_into("=I", invalid_count, offset + 8, 0xFFFFFFFF)
        # A structurally valid appendix may exceed the old arbitrary 1024 cap.
        # Tiny invalid state blobs keep this test bounded; restore must consume
        # the complete appendix, then reject the state transaction on use.
        metadata = data[entries[0][0]:entries[0][0] + 16]
        entry = metadata + struct.pack("=Q", 4) + b"BAD!" + struct.pack("=QQ", 0, 0)
        many = data[:offset] + struct.pack("=III", 0x504B4353, 1, 1025) + entry * 1025
        variants = {
            "legacy": data[:offset],
            "truncated": data[:-1],
            "oversized": oversized,
            "corrupt-target": corrupted,
            "invalid-metadata": invalid_metadata,
            "invalid-count": invalid_count,
            "many-checkpoints": many,
        }
        for name, payload in variants.items():
            filename = f"{name}.bin"
            (directory / filename).write_bytes(payload)
            # Occupy the destination with unrelated state before each restore.
            _complete(server, _tokenize(server, "Unrelated occupied destination.", special=True), 1)
            restored = _slot(server, 1, "restore", filename)
            assert restored["n_restored"] == saved["n_saved"]
            assert offset <= restored["n_read"] <= len(payload)
            if name == "many-checkpoints":
                assert restored["n_read"] == len(payload), "complete appendix must be consumed above 1024 entries"
            record = _complete(server, branch, 1)
            assert record["timings"]["cache_n"] == 0, (name, record)
            assert record["timings"]["prompt_n"] == len(branch), (name, record)
            records.append((name, record))
    finally:
        server.stop()
    cold = _oracle(model, cache, tmp_path / "oracle", [branch])[0]
    for name, record in records:
        assert record["tokens"][0] == cold["tokens"][0], (name, record, cold)
