#!/usr/bin/env python3
"""Black-box harness for the v0.4.3 KV/state regression families.

The harness owns the server lifecycle and writes a self-contained manifest,
request records, stdout/stderr logs, and a machine-readable run summary.  It is
deliberately independent from pytest so the same frozen workload can qualify a
release build or a historical worktree.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


SCHEMA_VERSION = 2
SIGNATURES = {
    "speculative_checkpoint_created": "created speculative checkpoint",
    "checkpoint_restore_prepare": "checkpoint restore preparation failed",
    "checkpoint_restore_transaction": "failed to restore speculative checkpoint transaction",
    "stage_alias": "structured KV live groups alias one F16 stage slot",
    "allocation_refusal": "Unable to allocate KV cache for this request",
    "spec_batch_subview": "speculative batch index",
    "logical_state_transaction": "cannot clone KV cache logical state during a transaction",
    "pos_min_lost": "pos_min == -1",
    "pos_min_recovery": "lost live KV state",
}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonl_append(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def count_signatures(paths: list[Path]) -> dict[str, int]:
    counts = {name: 0 for name in SIGNATURES}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                for name, text in SIGNATURES.items():
                    counts[name] += line.count(text)
    return counts


def generated_prompt(seed: int, request_index: int, blocks_count: int = 96) -> str:
    repeated = (
        "allocator boundary checkpoint tensor rollback sequence cache verifier draft "
        "metadata restore transaction speculative ngram "
    )
    blocks = []
    for block in range(blocks_count):
        blocks.append(f"block {block:03d}: " + repeated * 4)
    return (
        "You are reviewing a deterministic systems trace. Preserve every numbered block, "
        "then write a detailed C++ remediation with tests. Continue until max_tokens.\n"
        f"seed={seed} request={request_index}\n" + "\n".join(blocks)
    )


@dataclass
class RequestResult:
    request_index: int
    status: str
    http_status: int | None
    elapsed_seconds: float
    response_bytes: int
    error: str | None
    timings: dict[str, Any]


class ServerRun:
    def __init__(self, args: argparse.Namespace, run_dir: Path, trial: int) -> None:
        self.args = args
        self.run_dir = run_dir
        self.trial = trial
        self.port = free_port()
        self.process: subprocess.Popen[bytes] | None = None
        self.stdout_path = run_dir / "server.stdout.log"
        self.stderr_path = run_dir / "server.stderr.log"
        self.command = self._command()
        self.started_at = 0.0
        self.stopped_at = 0.0
        self.exit_code_before_stop: int | None = None
        self.exit_code_after_stop: int | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _command(self) -> list[str]:
        command = [
            str(self.args.server_bin),
            "--model", str(self.args.model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--offline",
            "--seed", str(self.args.seed),
            "--metrics",
        ]
        if self.args.mmproj:
            command += ["--mmproj", str(self.args.mmproj)]
        command += self.args.server_arg
        return command

    def start(self) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        stdout = self.stdout_path.open("wb")
        stderr = self.stderr_path.open("wb")
        self.started_at = time.time()
        self.process = subprocess.Popen(
            self.command,
            stdout=stdout,
            stderr=stderr,
            creationflags=flags,
        )
        deadline = time.monotonic() + self.args.start_timeout
        while time.monotonic() < deadline:
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(f"server exited during startup with {code}")
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.25)
        raise TimeoutError("server did not become healthy before the startup timeout")

    def stop(self) -> None:
        if self.process is None:
            return
        self.exit_code_before_stop = self.process.poll()
        if self.exit_code_before_stop is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.exit_code_after_stop = self.process.returncode
        self.stopped_at = time.time()

    def request(self, request_index: int) -> RequestResult:
        payload = {
            "model": self.args.model_alias,
            "messages": [{"role": "user", "content": generated_prompt(
                self.args.seed, request_index, self.args.prompt_blocks)}],
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "top_k": self.args.top_k,
            "top_p": self.args.top_p,
            "seed": self.args.seed + request_index,
            "stream": True,
        }
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=(10, self.args.inactivity_timeout),
            )
            if response.status_code != 200:
                text = response.text
                return RequestResult(
                    request_index, "http_failure", response.status_code,
                    time.perf_counter() - started, len(response.content), text[-4096:], {}
                )
            response_bytes = 0
            timings: dict[str, Any] = {}
            stream_error: str | None = None
            for raw in response.iter_lines():
                if time.perf_counter() - started > self.args.request_timeout:
                    raise requests.Timeout("request exceeded the total request timeout")
                response_bytes += len(raw)
                if not raw.startswith(b"data: ") or raw == b"data: [DONE]":
                    continue
                try:
                    event = json.loads(raw[6:])
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(event, dict) and event.get("timings"):
                    timings = event["timings"]
                if isinstance(event, dict) and event.get("error"):
                    stream_error = json.dumps(event["error"], sort_keys=True)
            elapsed = time.perf_counter() - started
            return RequestResult(
                request_index=request_index,
                status="request_failure" if stream_error else "ok",
                http_status=response.status_code,
                elapsed_seconds=elapsed,
                response_bytes=response_bytes,
                error=stream_error,
                timings=timings,
            )
        except requests.Timeout as error:
            return RequestResult(
                request_index, "timeout", None, time.perf_counter() - started, 0, str(error), {}
            )
        except requests.RequestException as error:
            status = "server_crash" if self.process and self.process.poll() is not None else "request_failure"
            if "timed out" in str(error).lower():
                status = "timeout"
            return RequestResult(
                request_index, status, None, time.perf_counter() - started, 0, str(error), {}
            )


def run_trial(args: argparse.Namespace, trial: int) -> dict[str, Any]:
    run_dir = args.output_dir / f"trial-{trial:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "requests.jsonl"
    server = ServerRun(args, run_dir, trial)
    startup_error: str | None = None
    results: list[RequestResult] = []
    try:
        server.start()
        request_indices = list(range(args.requests))
        if args.concurrency == 1:
            for index in request_indices:
                result = server.request(index)
                results.append(result)
                jsonl_append(records_path, result.__dict__)
                if server.process and server.process.poll() is not None:
                    break
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(server.request, index) for index in request_indices]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    jsonl_append(records_path, result.__dict__)
    except Exception as error:
        startup_error = f"{type(error).__name__}: {error}"
    finally:
        server.stop()

    signature_counts = count_signatures([server.stdout_path, server.stderr_path])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "trial": trial,
        "pid": server.process.pid if server.process else None,
        "server_command": server.command,
        "server_started_at": server.started_at,
        "server_stopped_at": server.stopped_at,
        "server_runtime_seconds": max(0.0, server.stopped_at - server.started_at),
        "server_exit_code_before_stop": server.exit_code_before_stop,
        "server_exit_code_after_stop": server.exit_code_after_stop,
        "startup_error": startup_error,
        "opportunities": len(results),
        "outcomes": {
            status: sum(result.status == status for result in results)
            for status in ("ok", "http_failure", "request_failure", "timeout", "server_crash")
        },
        "signature_counts": signature_counts,
        "pid_stable": server.exit_code_before_stop is None,
        "stdout_log": str(server.stdout_path.resolve()),
        "stderr_log": str(server.stderr_path.resolve()),
    }
    json_write(run_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--commit", default="unknown")
    parser.add_argument("--model-alias", default="model")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prompt-blocks", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--start-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--inactivity-timeout", type=float, default=30)
    parser.add_argument("--server-arg", action="append", default=[])
    args = parser.parse_args()
    if args.trials < 1 or args.requests < 1 or args.concurrency < 1 or args.prompt_blocks < 1:
        parser.error("--trials, --requests, --concurrency, and --prompt-blocks must be positive")
    if args.concurrency > args.requests:
        parser.error("--concurrency cannot exceed --requests")
    for path in (args.server_bin, args.model, args.mmproj):
        if path is not None and not path.is_file():
            parser.error(f"file does not exist: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    args = parse_args()
    version_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    version = subprocess.run(
        [str(args.server_bin), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=version_flags,
        check=False,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "configuration": args.configuration,
        "commit": args.commit,
        "server_bin": str(args.server_bin.resolve()),
        "server_version_exit_code": version.returncode,
        "server_version": (version.stdout + version.stderr).strip(),
        "model": str(args.model.resolve()),
        "mmproj": str(args.mmproj.resolve()) if args.mmproj else None,
        "seed": args.seed,
        "trials": args.trials,
        "requests_per_trial": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "prompt_blocks": args.prompt_blocks,
        "request_fixture": generated_prompt(args.seed, 0, args.prompt_blocks),
        "server_arguments": args.server_arg,
    }
    json_write(args.output_dir / "manifest.json", manifest)

    trials = [run_trial(args, trial) for trial in range(args.trials)]
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "configuration": args.configuration,
        "commit": args.commit,
        "trials": len(trials),
        "opportunities": sum(trial["opportunities"] for trial in trials),
        "outcomes": {
            status: sum(trial["outcomes"][status] for trial in trials)
            for status in ("ok", "http_failure", "request_failure", "timeout", "server_crash")
        },
        "signature_counts": {
            name: sum(trial["signature_counts"][name] for trial in trials)
            for name in SIGNATURES
        },
        "stable_trials": sum(bool(trial["pid_stable"]) for trial in trials),
        "startup_failures": sum(trial["startup_error"] is not None for trial in trials),
    }
    json_write(args.output_dir / "summary.json", aggregate)
    print(json.dumps(aggregate, sort_keys=True))
    return 0 if aggregate["startup_failures"] == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as error:
        print(f"repro-v043-kv-state: {type(error).__name__}: {error}", file=sys.stderr)
        raise
