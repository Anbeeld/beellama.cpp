#!/usr/bin/env python3
"""Measure prompt-cache TTFT and slot handoff through one hidden llama-server."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-bin", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--slots", type=int, choices=(1, 2), default=2)
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--prompt-tokens", type=int, default=768)
    parser.add_argument("--predict", type=int, default=8)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--cache-type", default="q4_0")
    parser.add_argument("--tail-tokens", type=int, default=128)
    parser.add_argument("--tail-type", default="bf16")
    parser.add_argument("--cache-ram", type=int, default=1024)
    parser.add_argument("--unified", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--server-arg", action="append", default=[])
    return parser.parse_args()


def choose_port(requested: int) -> int:
    if requested:
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def assert_no_llama_process() -> None:
    if os.name != "nt":
        completed = subprocess.run(
            ["ps", "-eo", "comm="], capture_output=True, text=True, check=True
        )
        active = {
            line.strip() for line in completed.stdout.splitlines()
        } & {"llama-cli", "llama-server", "llama-bench"}
    else:
        active = set()
        for executable in ("llama-cli.exe", "llama-server.exe", "llama-bench.exe"):
            completed = subprocess.run(
                ["tasklist.exe", "/FI", f"IMAGENAME eq {executable}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if executable.lower() in completed.stdout.lower():
                active.add(executable)
    if active:
        raise RuntimeError(f"another llama workload is active: {', '.join(sorted(active))}")


def bounded_tail(path: Path, lines: int = 100) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_identity(binary: Path) -> dict[str, Any]:
    components = [binary]
    if binary.suffix.lower() == ".exe":
        components.extend(sorted(binary.parent.glob("*.dll"), key=lambda path: path.name.lower()))
    records = [
        {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(set(components), key=lambda path: path.name.lower())
    ]
    framing = "\n".join(
        f"{record['name']}|{record['size']}|{record['sha256']}" for record in records
    )
    return {
        "sha256": hashlib.sha256(framing.encode("utf-8")).hexdigest(),
        "components": records,
    }


def cached_model_identity(model: Path, cache_path: Path) -> dict[str, Any]:
    stat = model.stat()
    identity = {
        "path": str(model),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cached = None
    if isinstance(cached, dict) and all(cached.get(key) == value for key, value in identity.items()):
        sha256 = cached.get("sha256")
        if isinstance(sha256, str) and len(sha256) == 64:
            return cached
    identity["sha256"] = sha256_file(model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return identity


class HiddenServer:
    def __init__(self, arguments: list[str], log_dir: Path, startup_timeout: float):
        self.arguments = arguments
        self.log_dir = log_dir
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen[bytes] | None = None
        self.ready = threading.Event()
        self.finished = threading.Event()
        self.startup_changed = threading.Event()
        self.stdout_path = log_dir / "llama-server.stdout.log"
        self.stderr_path = log_dir / "llama-server.stderr.log"

    def _copy_stream(self, source: BinaryIO, destination: Path) -> None:
        with destination.open("wb") as output:
            for line in iter(source.readline, b""):
                output.write(line)
                output.flush()
                if b"listening on http://" in line:
                    self.ready.set()
                    self.startup_changed.set()

    def _watch_exit(self) -> None:
        assert self.process is not None
        self.process.wait()
        self.finished.set()
        self.startup_changed.set()

    def start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        creationflags = 0
        startupinfo = None
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
        self.process = subprocess.Popen(
            self.arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )
        assert self.process.stdout is not None and self.process.stderr is not None
        threading.Thread(
            target=self._copy_stream, args=(self.process.stdout, self.stdout_path), daemon=True
        ).start()
        threading.Thread(
            target=self._copy_stream, args=(self.process.stderr, self.stderr_path), daemon=True
        ).start()
        threading.Thread(target=self._watch_exit, daemon=True).start()
        if not self.startup_changed.wait(self.startup_timeout):
            self.stop(force=True)
            raise TimeoutError(f"llama-server did not become ready in {self.startup_timeout:.0f}s")
        if not self.ready.is_set():
            code = self.process.returncode
            raise RuntimeError(
                f"llama-server exited before readiness with code {code}\n"
                f"stderr tail:\n{bounded_tail(self.stderr_path)}"
            )

    def stop(self, force: bool = False) -> None:
        if self.process is None or self.finished.is_set():
            return
        if force:
            self.process.kill()
        else:
            self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=30)


def post_json(port: int, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise RuntimeError(f"POST {path} returned HTTP {response.status}: {raw[-1000:]!r}")
        return json.loads(raw)
    finally:
        connection.close()


def stream_completion(
    port: int,
    prompt: list[int],
    slot: int,
    predict: int,
    timeout: float,
    previous_closed_at: float | None,
) -> tuple[dict[str, Any], float]:
    payload = {
        "prompt": prompt,
        "id_slot": slot,
        "cache_prompt": True,
        "n_predict": predict,
        "return_tokens": True,
        "seed": 12345,
        "temperature": 0.0,
        "stream": True,
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    content: list[str] = []
    response_tokens: list[int] = []
    final_event: dict[str, Any] | None = None
    first_content_at: float | None = None
    connection.request("POST", "/completion", body=body, headers={"Content-Type": "application/json"})
    request_written_at = time.perf_counter()
    try:
        response = connection.getresponse()
        if response.status != 200:
            raw = response.read()
            raise RuntimeError(f"completion returned HTTP {response.status}: {raw[-1000:]!r}")
        while True:
            line = response.readline()
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            encoded = line[5:].strip()
            if not encoded or encoded == b"[DONE]":
                continue
            event = json.loads(encoded)
            final_event = event
            chunk = str(event.get("content", ""))
            tokens = [int(token) for token in event.get("tokens", [])]
            if (chunk or tokens) and first_content_at is None:
                first_content_at = time.perf_counter()
            content.append(chunk)
            response_tokens.extend(tokens)
    finally:
        connection.close()
    closed_at = time.perf_counter()
    if final_event is None or first_content_at is None:
        raise RuntimeError("stream ended without a content event")
    timings = final_event.get("timings")
    if not isinstance(timings, dict):
        raise RuntimeError("final stream event lacks timings")
    prompt_n = int(timings["prompt_n"])
    input_tokens = len(prompt)
    reused = input_tokens - prompt_n
    row = {
        "slot": slot,
        "input_tokens": input_tokens,
        "prompt_n": prompt_n,
        "cache_n": int(timings.get("cache_n", 0)),
        "tokens_reused": reused,
        "cache_disposition": "hit" if reused > 0 else "miss",
        "ttft_ms": 1000.0 * (first_content_at - request_written_at),
        "handoff_ms": None
        if previous_closed_at is None
        else 1000.0 * (first_content_at - previous_closed_at),
        "response_sha256": hashlib.sha256("".join(content).encode("utf-8")).hexdigest(),
        "response_tokens": len(response_tokens),
    }
    return row, closed_at


def make_prompt(port: int, slot: int, turn: int, target: int, timeout: float) -> list[int]:
    text = f"slot {slot} deterministic benchmark history:\n"
    text += "alpha beta gamma delta epsilon zeta eta theta. " * max(1, target // 8)
    text += "\n" + " ".join(f"turn-{index}" for index in range(turn + 1)) + "\nAssistant:"
    tokens = post_json(
        port, "/tokenize", {"content": text, "add_special": True}, timeout
    )["tokens"]
    while len(tokens) < target + turn:
        text += " additional"
        tokens = post_json(
            port, "/tokenize", {"content": text, "add_special": True}, timeout
        )["tokens"]
    return [int(token) for token in tokens]


def main() -> int:
    args = parse_args()
    if args.requests < 1 or args.prompt_tokens < 1 or args.predict < 1:
        raise ValueError("requests, prompt-tokens, and predict must be positive")
    server_bin = args.server_bin.resolve(strict=True)
    model = args.model.resolve(strict=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = (args.log_dir or output.parent / f"{output.stem}-logs").resolve()
    port = choose_port(args.port)
    assert_no_llama_process()
    binary_identity = runtime_identity(server_bin)
    model_identity = cached_model_identity(model, output.parent / "model-identity-cache.json")

    slot_dir = tempfile.mkdtemp(prefix="kv-tail-server-slots-")
    server_args = [
        str(server_bin),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--offline",
        "--model", str(model),
        "--n-gpu-layers", "999",
        "--parallel", str(args.slots),
        "--ctx-size", str(args.context),
        "--batch-size", str(args.batch),
        "--ubatch-size", str(args.ubatch),
        "--cache-type-k", args.cache_type,
        "--cache-type-v", args.cache_type,
        "--kv-tail-tokens", str(args.tail_tokens),
        "--kv-tail-type", args.tail_type,
        "--flash-attn", "on",
        "--cont-batching",
        "--slots",
        "--cache-ram", str(args.cache_ram),
        "--slot-save-path", slot_dir,
        "--no-jinja",
        "--verbose",
    ]
    if args.unified:
        server_args.append("--kv-unified")
    server_args.extend(args.server_arg)
    server = HiddenServer(server_args, log_dir, args.startup_timeout)
    rows: list[dict[str, Any]] = []
    try:
        server.start()
        previous_closed_at: float | None = None
        turns = [0] * args.slots
        for request_index in range(args.requests):
            slot = request_index % args.slots
            prompt = make_prompt(
                port, slot, turns[slot], args.prompt_tokens, args.request_timeout
            )
            row, previous_closed_at = stream_completion(
                port,
                prompt,
                slot,
                args.predict,
                args.request_timeout,
                previous_closed_at,
            )
            row.update({"request": request_index, "turn": turns[slot]})
            rows.append(row)
            turns[slot] += 1
    finally:
        server.stop()

    with output.open("w", encoding="utf-8", newline="\n") as result:
        for row in rows:
            result.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "server_bin": str(server_bin),
        "server_sha256": binary_identity["sha256"],
        "server_components": binary_identity["components"],
        "model": str(model),
        "model_size": model_identity["size"],
        "model_sha256": model_identity["sha256"],
        "slots": args.slots,
        "unified": args.unified,
        "cache_type": args.cache_type,
        "tail_tokens": args.tail_tokens,
        "tail_type": args.tail_type,
        "row_count": len(rows),
        "output": str(output),
        "stdout_log": str(server.stdout_path),
        "stderr_log": str(server.stderr_path),
        "command": server_args,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"bench-kv-tail-server: {error}", file=sys.stderr)
        raise
