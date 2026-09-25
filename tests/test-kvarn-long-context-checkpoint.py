#!/usr/bin/env python3
"""Opt-in CUDA long-context KVarN checkpoint regression.

Requires a local hybrid model with owned MTP state. No model downloads occur.
The test exercises nearby and historical branches, checks deterministic cached
versus cold continuation, and records the exact server command and results.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_health(url: str, process: subprocess.Popen[str], log_path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            raise RuntimeError(f"server exited with {process.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    raise TimeoutError(f"server readiness timeout; see {log_path}")


def complete(url: str, prompt: list[int], timeout: float, cache_prompt: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    body = request_json(
        url + "/completion",
        {
            "prompt": prompt,
            "id_slot": 0,
            "cache_prompt": cache_prompt,
            "n_predict": 4,
            "return_tokens": True,
            "temperature": 0.0,
            "seed": 12345,
            "ignore_eos": True,
        },
        timeout,
    )
    body["client_elapsed_ms"] = (time.perf_counter() - started) * 1000
    return body


def continuation(prompt: list[int], body: dict[str, Any]) -> list[int]:
    generated = [int(token) for token in body.get("tokens", [])]
    return prompt + generated[:-1] if generated else list(prompt)


def metric_value(metrics: str, suffix: str) -> float:
    pattern = re.compile(rf"^[^#\n]*{re.escape(suffix)}(?:\{{[^\n]*\}})?\s+([0-9.eE+-]+)$", re.MULTILINE)
    match = pattern.search(metrics)
    if not match:
        raise AssertionError(f"metric not found: {suffix}")
    return float(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tmp/kvarn-long-context-checkpoint"))
    parser.add_argument("--device", default="CUDA1")
    parser.add_argument("--load-mode", default="dio")
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--prompt-tokens", type=int, default=120000)
    parser.add_argument("--checkpoint-min-step", type=int, default=8192)
    parser.add_argument("--cache-type", default="kvarn6")
    parser.add_argument("--tail-tokens", type=int, default=1024)
    parser.add_argument("--spec", choices=("mtp", "none"), default="mtp")
    parser.add_argument("--kv-unified", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rollback-128-ms", type=float, default=2000)
    parser.add_argument("--rollback-768-ms", type=float, default=5000)
    parser.add_argument("--rollback-4096-ms", type=float, default=30000)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()

    if not args.server.is_file() or not args.model.is_file():
        parser.error("--server and --model must name existing files")

    args.output.mkdir(parents=True, exist_ok=True)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    log_path = args.output / "server.log"
    command = [
        str(args.server.resolve()), "-m", str(args.model.resolve()),
        "--host", "127.0.0.1", "--port", str(port),
        "--device", args.device, "--split-mode", "none", "-ngl", "all",
        "--ctx-size", str(args.context), "-np", "1",
        "--batch-size", "4096", "--ubatch-size", "2048",
        "--flash-attn", "on", "--fit", "off",
        "--cache-type-k", args.cache_type, "--cache-type-v", args.cache_type,
        "--kv-tail-tokens", str(args.tail_tokens), "--kv-tail-type", "f16",
        "--cache-ram", "32768", "--ctx-checkpoints", "32",
        "--checkpoint-min-step", str(args.checkpoint_min_step),
        "--metrics", "--slots", "--offline", "--seed", "12345",
        "--load-mode", args.load_mode, "--verbosity", "5",
    ]
    if args.kv_unified:
        command.append("--kv-unified")
    if args.spec == "mtp":
        command += [
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
            "--spec-draft-ngl", "all",
            "--spec-draft-type-k", "kvarn4", "--spec-draft-type-v", "kvarn4",
        ]
    result: dict[str, Any] = {
        "command": command,
        "version": subprocess.check_output(
            [str(args.server.resolve()), "--version"], text=True, stderr=subprocess.STDOUT
        ),
        "responses": {},
    }
    (args.output / "manifest.json").write_text(
        json.dumps({"command": command, "model": str(args.model.resolve())}, indent=2) + "\n",
        encoding="utf-8",
    )

    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, text=True, creationflags=flags
        )
        try:
            wait_health(url, process, log_path, args.timeout)
            seed = request_json(
                url + "/tokenize",
                {"content": " alpha beta gamma delta epsilon", "add_special": True, "parse_special": True},
                60,
            )["tokens"]
            alternate = request_json(
                url + "/tokenize",
                {"content": " changed rollback branch suffix", "add_special": False, "parse_special": True},
                60,
            )["tokens"]
            if len(seed) < 2 or not alternate:
                raise AssertionError("tokenizer returned insufficient deterministic seed tokens")

            prompt = [int(seed[0])] + [int(seed[1 + i % (len(seed) - 1)]) for i in range(args.prompt_tokens - 1)]
            warm = complete(url, prompt, args.timeout)
            result["responses"]["warm"] = warm
            current = continuation(prompt, warm)

            exact_prompt = current + [int(seed[1])] * 64
            exact = complete(url, exact_prompt, args.timeout)
            result["responses"]["exact"] = exact
            assert exact["timings"]["cache_source"] == "live", exact["timings"]
            assert exact["client_elapsed_ms"] < 1500, exact["client_elapsed_ms"]
            current = continuation(exact_prompt, exact)

            branch_128_prompt: list[int] | None = None
            branch_128: dict[str, Any] | None = None
            rollback_limits = (
                (128, args.rollback_128_ms),
                (768, args.rollback_768_ms),
                (4096, args.rollback_4096_ms),
            )
            for rollback, limit_ms in rollback_limits:
                branch_prompt = current[:-rollback] + [int(alternate[0])] * 64
                response = complete(url, branch_prompt, args.timeout)
                result["responses"][f"branch_{rollback}"] = response
                timings = response["timings"]
                assert timings["cache_source"] == "checkpoint", timings
                assert timings["cache_reason"] == "committed", timings
                assert timings["cache_reprocessed_n"] < 16384, timings
                assert response["client_elapsed_ms"] < limit_ms, response["client_elapsed_ms"]
                assert timings["cache_checkpoint_prepare_ms"] > 0, timings
                assert timings["cache_checkpoint_commit_ms"] > 0, timings
                assert (
                    timings["cache_checkpoint_prepare_ms"] + timings["cache_checkpoint_commit_ms"]
                    <= timings["cache_checkpoint_restore_ms"] + 1.0
                ), timings
                if rollback == 128:
                    branch_128_prompt = branch_prompt
                    branch_128 = response
                current = continuation(branch_prompt, response)

            assert branch_128_prompt is not None and branch_128 is not None
            cold = complete(url, branch_128_prompt, args.timeout, cache_prompt=False)
            result["responses"]["cold_oracle_128"] = cold
            assert cold["timings"]["cache_n"] == 0, cold["timings"]
            assert cold.get("tokens", [])[:1] == branch_128.get("tokens", [])[:1], (
                cold.get("tokens"), branch_128.get("tokens")
            )

            with urllib.request.urlopen(url + "/metrics", timeout=60) as response:
                metrics = response.read().decode("utf-8")
            (args.output / "metrics.txt").write_text(metrics, encoding="utf-8")
            assert metric_value(metrics, "prompt_cache_admission_failures_total") == 0
            assert metric_value(metrics, "prompt_cache_restore_failures_total") == 0

            text = log_path.read_text(encoding="utf-8", errors="replace")
            assert "checkpoint restore preparation failed" not in text
            assert "internal prompt-cache rollback failure" not in text
            assert "partial KV state no longer" not in text
        except Exception as error:
            result["error"] = repr(error)
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            (args.output / "result.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )

    print(json.dumps({name: value["timings"] for name, value in result["responses"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
