#!/usr/bin/env python3
"""Run and record the required HIP KVarN hardware validation matrix.

This script is intentionally separate from multi-architecture release builds.
It succeeds only after test-kvarn attests that the requested physical AMD wave
actually executed the route-boundary matrix.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any


DEVICE_CLASSES = {
    "rdna-wave32": 32,
    "cdna-wave64": 64,
}


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def run_logged(
    command: list[str],
    cwd: Path,
    log_dir: Path,
    name: str,
    timeout_seconds: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, \
            stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_file:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    return {
        "name": name,
        "command": command_text(command),
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_log": stdout_path.name,
        "stderr_log": stderr_path.name,
    }


def capture(command: list[str], cwd: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError:
        return {
            "command": command_text(command),
            "returncode": 127,
            "output": f"executable not found: {command[0]}",
        }
    return {
        "command": command_text(command),
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def cmake_cache_value(cache: str, name: str) -> str | None:
    prefix = f"{name}:"
    for line in cache.splitlines():
        if line.startswith(prefix) and "=" in line:
            return line.split("=", 1)[1]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the attested HIP KVarN runtime gate and write machine-readable evidence.")
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--device-class", required=True, choices=sorted(DEVICE_CLASSES))
    parser.add_argument("--config", default="Release")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--all-tests", action="store_true",
                        help="Also run the complete configured CTest suite.")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    build_dir = args.build_dir.resolve()
    evidence_dir = args.evidence_dir.resolve()
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        parser.error(f"missing configured cache: {cache_path}")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    ctest = shutil.which("ctest")
    cmake = shutil.which("cmake")
    if ctest is None or cmake is None:
        parser.error("cmake and ctest must be available on PATH")

    evidence_dir.mkdir(parents=True, exist_ok=False)
    cache_text = cache_path.read_text(encoding="utf-8", errors="replace")
    # Keep this spelling stable for the static protocol test: git rev-parse HEAD.
    commit = capture(["git", "rev-parse", "HEAD"], repo)
    hip_compiler = cmake_cache_value(cache_text, "CMAKE_HIP_COMPILER") or "hipcc"
    compiler = capture([hip_compiler, "--version"], repo)
    runtime = capture(["rocminfo"], repo)

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo),
        "build_dir": str(build_dir),
        "device_class": args.device_class,
        "expected_physical_wave_size": DEVICE_CLASSES[args.device_class],
        "commit": commit,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "hipcc": compiler,
            "rocminfo": runtime,
        },
        "build": {
            "GGML_HIP": cmake_cache_value(cache_text, "GGML_HIP"),
            "GGML_CUDA_KVARN": cmake_cache_value(cache_text, "GGML_CUDA_KVARN"),
            "GGML_CUDA_FA_ALL_QUANTS": cmake_cache_value(cache_text, "GGML_CUDA_FA_ALL_QUANTS"),
            "CMAKE_HIP_ARCHITECTURES": cmake_cache_value(cache_text, "CMAKE_HIP_ARCHITECTURES"),
            "CMAKE_BUILD_TYPE": cmake_cache_value(cache_text, "CMAKE_BUILD_TYPE"),
        },
        "matrix": {
            "head_dimensions": [128, 256, 512],
            "query_counts": [1, 2, 4, 8, 16, 17, 256, 511, 512],
            "gqa_ratios": [1, 2, 4, 6, 8, 16],
            "exact_tails": ["F16", "BF16"],
            "unaligned_k_rows": 513,
        },
        "commands": [],
        "omitted": [] if args.all_tests else ["repository tests outside the HIP KVarN gate"],
    }
    evidence_path = evidence_dir / "evidence.json"

    commands: list[tuple[str, list[str], dict[str, str] | None]] = [
        ("build", [cmake, "--build", str(build_dir), "--config", args.config,
                   "--target", "test-kvarn", "test-cuda-fattn-route-policy"], None),
        ("host-policy", [ctest, "--test-dir", str(build_dir), "-C", args.config,
                         "-R", "^(test-cuda-fattn-route-policy|test-kvarn-hip-tail-capability-static|test-kvarn-hip-runtime-validation)$",
                         "--output-on-failure"], None),
    ]
    amd_env = os.environ.copy()
    amd_env["GGML_KVARN_TEST_BACKEND"] = "ROCm0"
    amd_env["GGML_KVARN_TEST_AMD_ROUTE_BOUNDARIES_ONLY"] = "1"
    amd_env["GGML_KVARN_AMD_RUNTIME_ATTESTATION"] = args.device_class
    commands.append((
        "amd-route-boundaries",
        [ctest, "--test-dir", str(build_dir), "-C", args.config,
         "-R", "^test-kvarn$", "--verbose", "--output-on-failure"],
        amd_env,
    ))
    portable_env = os.environ.copy()
    portable_env["GGML_KVARN_TEST_BACKEND"] = "ROCm0"
    portable_env["GGML_KVARN_TEST_PORTABLE_NATIVE_ONLY"] = "1"
    commands.append((
        "portable-native",
        [ctest, "--test-dir", str(build_dir), "-C", args.config,
         "-R", "^test-kvarn$", "--output-on-failure"],
        portable_env,
    ))
    full_kvarn_env = os.environ.copy()
    full_kvarn_env["GGML_KVARN_TEST_BACKEND"] = "ROCm0"
    commands.append((
        "full-kvarn",
        [ctest, "--test-dir", str(build_dir), "-C", args.config,
         "-R", "^test-kvarn$", "--output-on-failure"],
        full_kvarn_env,
    ))
    if args.all_tests:
        commands.append((
            "all-tests",
            [ctest, "--test-dir", str(build_dir), "-C", args.config, "--output-on-failure"],
            None,
        ))

    result = 1
    try:
        for name, command, env in commands:
            record = run_logged(
                command, repo, evidence_dir, name, args.timeout_seconds, env)
            evidence["commands"].append(record)
            if record["returncode"] != 0:
                evidence["status"] = "failed"
                evidence["failed_command"] = name
                break
        else:
            runtime_log = (evidence_dir / "amd-route-boundaries.stdout.log").read_text(
                encoding="utf-8", errors="replace")
            marker = f"AMD runtime validation attestation={args.device_class} physical_wave={DEVICE_CLASSES[args.device_class]} OK"
            if marker not in runtime_log:
                evidence["status"] = "failed"
                evidence["failure"] = "test-kvarn did not emit the matching physical-wave attestation"
            else:
                evidence["status"] = "passed-full" if args.all_tests else "passed-kvarn-gate"
                result = 0
    except subprocess.TimeoutExpired as exc:
        evidence["status"] = "failed"
        evidence["failure"] = f"command timed out after {exc.timeout} seconds"
    finally:
        evidence["finished_utc"] = datetime.now(timezone.utc).isoformat()
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    print(evidence_path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
