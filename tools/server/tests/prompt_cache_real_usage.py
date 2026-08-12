#!/usr/bin/env python3
"""Deterministic real-usage prompt-cache and concurrency harness.

The harness owns the server lifecycle, records one JSON object per request and
metrics sample, and optionally restarts the same configuration for no-cache
oracle replays. It intentionally uses generated coding-agent conversations; no
private chat or database content is read.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import dataclasses
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import requests


SCHEMA_VERSION = 1
EXPLICIT_REEVALUATION_REASONS = {"no_restorable_kvarn_boundary"}


def jsonl_write(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def generated_block(rng: random.Random, label: str, words: int) -> str:
    vocabulary = (
        "allocator boundary checkpoint tensor rollback sequence cache immutable "
        "backend kernel transaction prompt verifier draft metadata restore"
    ).split()
    body = " ".join(vocabulary[rng.randrange(len(vocabulary))] for _ in range(words))
    return f"{label}\n```text\n{body}\n```"


def conversation_addition(seed: int, conversation: int, turn: int, long_words: int) -> dict[str, Any]:
    rng = random.Random((seed << 24) ^ (conversation << 12) ^ turn)
    sizes = (24, 64, 320, 768)
    words = long_words if turn > 0 and turn % 8 == 0 else sizes[turn % len(sizes)]
    if turn % 5 == 3:
        return {
            "role": "tool",
            "tool_call_id": f"call_{conversation}_{turn}",
            "content": generated_block(rng, f"compiler output c={conversation} t={turn}", words),
        }
    return {
        "role": "user",
        "content": generated_block(rng, f"coding task c={conversation} t={turn}", words),
    }


def cached_tokens(response: dict[str, Any]) -> int:
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    for key in ("cached_tokens", "cache_n"):
        if key in details:
            return int(details[key])
        if key in usage:
            return int(usage[key])
    timings = response.get("timings") or {}
    return int(timings.get("cache_n", 0))


def completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def tokenize_text(base_url: str, text: str, timeout: float, *, add_special: bool) -> list[int]:
    response = requests.post(
        f"{base_url}/tokenize",
        json={"content": text, "add_special": add_special, "parse_special": True},
        timeout=timeout,
    )
    response.raise_for_status()
    return [int(token) for token in response.json()["tokens"]]


def prompt_tokens(base_url: str, messages: list[dict[str, Any]], timeout: float) -> list[int]:
    response = requests.post(
        f"{base_url}/apply-template", json={"messages": messages}, timeout=timeout
    )
    response.raise_for_status()
    prompt = str(response.json()["prompt"])
    return tokenize_text(base_url, prompt, timeout, add_special=False)


def common_prefix_size(left: list[int], right: list[int]) -> int:
    size = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        size += 1
    return size


def token_digest(tokens: list[int]) -> str:
    encoded = ",".join(str(token) for token in tokens).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path | None, *, hash_contents: bool = False) -> dict[str, Any] | None:
    if path is None:
        return None
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_contents:
        result["sha256"] = file_sha256(path)
    return result


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2 or "{" in fields[0]:
            continue
        try:
            metrics[fields[0]] = float(fields[1])
        except ValueError:
            continue
    return metrics


def metric_by_suffix(metrics: dict[str, float], suffix: str) -> float:
    matches = [value for name, value in metrics.items() if name.endswith(suffix)]
    return matches[-1] if matches else 0.0


@dataclasses.dataclass
class StreamResult:
    response: dict[str, Any]
    text: str
    ttft_ms: float
    latency_ms: float
    request_id: str


def chat_stream(base_url: str, payload: dict[str, Any], timeout: float) -> StreamResult:
    started = time.perf_counter()
    first = 0.0
    final: dict[str, Any] = {}
    pieces: list[str] = []
    with requests.post(
        f"{base_url}/v1/chat/completions",
        json={**payload, "stream": True, "stream_options": {"include_usage": True}},
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        request_id = response.headers.get("x-request-id", "")
        for raw in response.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data = raw[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            final.update({key: value for key, value in chunk.items() if value is not None})
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    if first == 0.0:
                        first = time.perf_counter()
                    pieces.append(str(content))
    ended = time.perf_counter()
    if first == 0.0:
        first = ended
    text = "".join(pieces)
    final["choices"] = [{"message": {"content": text}}]
    return StreamResult(final, text, (first - started) * 1000, (ended - started) * 1000, request_id)


def completion_with_tokens(base_url: str, payload: dict[str, Any], timeout: float) -> StreamResult:
    """Run one exact-token completion and retain the model's returned token IDs.

    The OpenAI chat stream exposes parsed visible/reasoning text, not the raw
    generated token sequence.  Prompt-cache correctness must compare the latter,
    so the real-usage oracle submits the already chat-templated token prompt to
    the native endpoint with return_tokens enabled.
    """
    started = time.perf_counter()
    response = requests.post(f"{base_url}/completion", json=payload, timeout=timeout)
    ended = time.perf_counter()
    response.raise_for_status()
    body = response.json()
    return StreamResult(
        body,
        str(body.get("content") or ""),
        (ended - started) * 1000,
        (ended - started) * 1000,
        response.headers.get("x-request-id", ""),
    )


class Server:
    def __init__(self, args: argparse.Namespace, port: int, tag: str) -> None:
        self.args = args
        self.port = port
        self.tag = tag
        self.process: subprocess.Popen[str] | None = None
        self.log = args.output_dir / f"server-{tag}.log"
        self.command = self._command()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _command(self) -> list[str]:
        command = [
            str(self.args.server_bin), "-m", str(self.args.target_model),
            "--host", "127.0.0.1", "--port", str(self.port),
            "-c", str(self.args.context), "-np", str(self.args.slots),
            "-ngl", "999", "--flash-attn", "on", "--metrics", "--slots",
            "--offline", "--seed", str(self.args.seed), "--alias", self.args.model_alias,
            "--cache-type-k", self.args.cache_type,
            "--cache-type-v", self.args.cache_type,
            "--cache-ram", str(self.args.cache_ram_mib),
            "--ctx-checkpoints", str(self.args.ctx_checkpoints),
            "--checkpoint-min-step", str(self.args.checkpoint_min_step),
        ]
        if self.args.unified:
            command.append("--kv-unified")
        if self.args.tail_tokens:
            command += ["--kv-tail-tokens", str(self.args.tail_tokens), "--kv-tail-type", self.args.tail_type]
        if self.args.spec == "mtp":
            command += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(self.args.draft_max)]
        elif self.args.spec == "dflash":
            if self.args.draft_model is None:
                raise ValueError("--draft-model is required for --spec dflash")
            command += [
                "--spec-type", "draft-dflash", "--spec-draft-model", str(self.args.draft_model),
                "--spec-draft-n-max", str(self.args.draft_max),
            ]
        command += self.args.server_arg
        return command

    def start(self) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        log = self.log.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=flags,
        )
        deadline = time.monotonic() + self.args.start_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited during startup with {self.process.returncode}: {self.log}")
            try:
                response = requests.get(f"{self.url}/health", timeout=2)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.25)
        raise TimeoutError(f"server did not become healthy: {self.log}")

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.process.returncode not in (0, -15, 1):
            raise RuntimeError(f"server exited with {self.process.returncode}: {self.log}")


class BusySampler:
    def __init__(self, server: Server, path: Path, interval: float) -> None:
        self.server = server
        self.path = path
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.samples: list[tuple[float, int]] = []

    def __enter__(self) -> "BusySampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            now = time.time()
            try:
                slots = requests.get(f"{self.server.url}/slots", timeout=2).json()
                busy = sum(bool(slot.get("is_processing")) for slot in slots)
                self.samples.append((now, busy))
                jsonl_write(self.path, {"schema_version": SCHEMA_VERSION, "kind": "busy", "time": now, "busy": busy})
            except (requests.RequestException, ValueError):
                pass
            self.stop_event.wait(self.interval)


def build_payload(
    args: argparse.Namespace,
    conversation: int,
    turn: int,
    prompt: list[int],
    cache: bool,
) -> dict[str, Any]:
    requested_lengths = (32, 64, 128, 256, 512, 1024)
    payload = {
        "prompt": prompt,
        "temperature": 0.0,
        "seed": args.seed + conversation,
        "n_predict": min(args.max_tokens, requested_lengths[turn % len(requested_lengths)]),
        "ignore_eos": True,
        "cache_prompt": cache and not args.disable_request_cache,
        "id_slot": conversation % args.slots,
        "return_tokens": True,
    }
    if args.top_logprobs > 0:
        payload["n_probs"] = args.top_logprobs
    return payload


def run_phase(args: argparse.Namespace, server: Server, oracle: bool, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = [[{"role": "system", "content": "You are a deterministic coding assistant."}] for _ in range(args.concurrency)]
    # The live server state contains the evaluated prompt plus every generated
    # token except the final sampled token, which has not yet been decoded.
    # Compare candidate prompts with that exact state, not merely with the
    # previous request's input prompt.
    prior_state_tokens: list[list[int]] = [[] for _ in range(args.concurrency)]
    outputs: list[dict[str, Any]] = []
    barrier = threading.Barrier(args.concurrency)
    oracle_inputs = {
        (int(record["conversation"]), int(record["turn"])): record
        for record in records
    }

    def one(conversation: int, turn: int) -> dict[str, Any]:
        if oracle:
            request_messages = copy.deepcopy(oracle_inputs[(conversation, turn)]["_messages"])
        else:
            # Periodic regenerate/edit branch: drop the most recent user/assistant
            # exchange, then continue from the retained prefix.
            if turn > 2 and (turn % 7 == 0 or turn % 8 == 1) and len(messages[conversation]) > 2:
                del messages[conversation][-2:]
            messages[conversation].append(
                conversation_addition(args.seed, conversation, turn, args.long_words)
            )
            request_messages = copy.deepcopy(messages[conversation])
        if not args.serialize_clients:
            barrier.wait(timeout=args.request_timeout)
        exact_prompt_tokens = prompt_tokens(server.url, request_messages, args.request_timeout)
        # RAM prompt-cache lookup is global across idle logical slots. A fixed-
        # order run can therefore reuse another conversation's system prefix
        # before this conversation has any prior state of its own.
        lcp = max(
            (common_prefix_size(state, exact_prompt_tokens) for state in prior_state_tokens),
            default=0,
        )
        payload = build_payload(args, conversation, turn, exact_prompt_tokens, not oracle)
        started = time.time()
        result = completion_with_tokens(server.url, payload, args.request_timeout)
        timings = result.response.get("timings") or {}
        prompt_token_count = len(exact_prompt_tokens)
        committed = 0 if oracle else cached_tokens(result.response)
        reported_lcp = 0 if oracle else int(timings.get("cache_lcp_n", lcp))
        planned = 0 if oracle else int(timings.get("cache_planned_n", committed))
        processed = int(timings.get("prompt_n", max(0, prompt_token_count - committed)))
        output_token_ids = [int(token) for token in result.response.get("tokens", [])]
        if len(output_token_ids) != int(result.response.get("tokens_predicted", len(output_token_ids))):
            raise AssertionError("completion response omitted returned token IDs")
        prior_state_tokens[conversation] = exact_prompt_tokens + output_token_ids[:-1]
        record = {
            "schema_version": SCHEMA_VERSION,
            "kind": "request",
            "phase": "oracle" if oracle else "cached",
            "commit": args.commit,
            "models": {"target": str(args.target_model), "draft": str(args.draft_model or "")},
            "server_arguments": server.command,
            "seed": args.seed,
            "request_id": result.request_id,
            "conversation": conversation,
            "turn": turn,
            "prompt_length": prompt_token_count,
            "prompt_token_sha256": token_digest(exact_prompt_tokens),
            "harness_lcp_tokens": lcp,
            "lcp_tokens": reported_lcp,
            "planned_cache_tokens": planned,
            "committed_cache_tokens": committed,
            "processed_prompt_tokens": processed,
            "cache_source": str(timings.get("cache_source", "unknown")),
            "cache_reason": str(timings.get("cache_reason", "unknown")),
            "ttft_ms": result.ttft_ms,
            "total_latency_ms": result.latency_ms,
            "output_tokens": len(output_token_ids),
            "output_token_ids": output_token_ids,
            "output_token_ids_source": "native_return_tokens",
            "completion_probabilities": result.response.get("completion_probabilities", []),
            "slot": conversation % args.slots,
            "started": started,
            "ended": time.time(),
            "text": result.text,
            # Retained only in memory for the oracle phase.  The generated
            # corpus is fully described by corpus.json and the prompt digest;
            # omitting cumulative message copies keeps long terminal artifacts
            # linear rather than quadratic in conversation length.
            "_messages": request_messages,
        }
        if not oracle and not (committed <= planned <= reported_lcp):
            raise AssertionError(
                f"invalid reuse contract committed={committed} planned={planned} "
                f"lcp={reported_lcp} for c={conversation} t={turn}"
            )
        if not oracle and reported_lcp > lcp + 1:
            raise AssertionError(
                f"server LCP {reported_lcp} exceeds exact harness LCP {lcp} by more than BOS allowance"
            )
        if not oracle:
            messages[conversation].append({"role": "assistant", "content": result.text})
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for turn in range(args.turns):
            if args.serialize_clients:
                round_records = [one(conversation, turn) for conversation in range(args.concurrency)]
            else:
                futures = [pool.submit(one, conversation, turn) for conversation in range(args.concurrency)]
                round_records = [future.result() for future in futures]
            for record in round_records:
                jsonl_write(
                    args.output_dir / "requests.jsonl",
                    {key: value for key, value in record.items() if not key.startswith("_")},
                )
            outputs.extend(round_records)
    return outputs


def summarize(
    args: argparse.Namespace,
    cached: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
    samples: Iterable[tuple[float, int]],
    final_metrics: dict[str, float],
) -> dict[str, Any]:
    samples = list(samples)
    active = [busy for _, busy in samples if busy > 0]
    overlap_ratio = sum(busy >= 2 for busy in active) / len(active) if active else 0.0
    max_busy = max(active, default=0)
    cache_hits = sum(record["committed_cache_tokens"] > 0 for record in cached[args.concurrency:])
    supported_misses = [
        record
        for record in cached[args.concurrency:]
        if record["lcp_tokens"] > 0
        and record["committed_cache_tokens"] == 0
        and record["cache_reason"] not in EXPLICIT_REEVALUATION_REASONS
    ]
    explicit_reevaluations = [
        record
        for record in cached[args.concurrency:]
        if record["lcp_tokens"] > 0
        and record["committed_cache_tokens"] == 0
        and record["cache_reason"] in EXPLICIT_REEVALUATION_REASONS
    ]
    planned_total = sum(record["planned_cache_tokens"] for record in cached)
    committed_total = sum(record["committed_cache_tokens"] for record in cached)
    unexpected_prompt_work = sum(
        max(
            0,
            record["processed_prompt_tokens"]
            - (record["prompt_length"] - record["committed_cache_tokens"]),
        )
        for record in cached
        if record["committed_cache_tokens"] > 0
    )
    reuse_efficiency = committed_total / planned_total if planned_total else 1.0
    sequence_mismatches = 0
    first_token_mismatches = 0
    if oracle:
        by_key = {(record["conversation"], record["turn"]): record for record in oracle}
        for record in cached:
            other = by_key[(record["conversation"], record["turn"])]
            cached_tokens = record["output_token_ids"]
            oracle_tokens = other["output_token_ids"]
            sequence_mismatches += cached_tokens != oracle_tokens
            first_token_mismatches += cached_tokens[:1] != oracle_tokens[:1]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "commit": args.commit,
        "configuration": args.configuration,
        "seed": args.seed,
        "requests": len(cached),
        "cache_hits": cache_hits,
        "supported_reuse_misses": len(supported_misses),
        "explicit_reevaluations": len(explicit_reevaluations),
        "explicit_reevaluation_tokens": sum(record["lcp_tokens"] for record in explicit_reevaluations),
        "planned_cache_tokens": planned_total,
        "committed_cache_tokens": committed_total,
        "planned_reuse_efficiency": reuse_efficiency,
        "unexpected_prompt_work_tokens": unexpected_prompt_work,
        "oracle_sequence_mismatches": sequence_mismatches,
        "oracle_first_token_mismatches": first_token_mismatches,
        "busy_samples": len(samples),
        "active_busy_samples": len(active),
        "overlap_ratio": overlap_ratio,
        "max_busy": max_busy,
        "serialized_clients": bool(args.serialize_clients),
        "prompt_cache_accounted_bytes": int(metric_by_suffix(final_metrics, "prompt_cache_accounted_bytes")),
        "prompt_cache_admission_attempts": int(metric_by_suffix(final_metrics, "prompt_cache_admission_attempts_total")),
        "prompt_cache_admission_successes": int(metric_by_suffix(final_metrics, "prompt_cache_admission_successes_total")),
        "prompt_cache_admission_failures": int(metric_by_suffix(final_metrics, "prompt_cache_admission_failures_total")),
        "prompt_cache_restore_failures": int(metric_by_suffix(final_metrics, "prompt_cache_restore_failures_total")),
        "kv_tail_degraded_sequences": int(metric_by_suffix(final_metrics, "kv_tail_degraded_sequences")),
    }
    if args.expect_cache and cache_hits == 0:
        raise AssertionError("prompt cache was requested but no post-warmup request reused tokens")
    if args.expect_cache and supported_misses:
        first = supported_misses[0]
        raise AssertionError(
            f"supported repeated prefix silently missed at c={first['conversation']} "
            f"t={first['turn']} reason={first['cache_reason']}"
        )
    if args.expect_cache and planned_total and reuse_efficiency < args.min_reuse_efficiency:
        raise AssertionError(
            f"planned reuse efficiency {reuse_efficiency:.5f} is below "
            f"{args.min_reuse_efficiency:.5f}"
        )
    if args.concurrency > 1 and not args.serialize_clients:
        if overlap_ratio < args.min_overlap:
            raise AssertionError(f"observed overlap {overlap_ratio:.3f} is below {args.min_overlap:.3f}")
        if max_busy < args.concurrency:
            raise AssertionError(f"all {args.concurrency} slots were never simultaneously busy (max={max_busy})")
    if first_token_mismatches:
        raise AssertionError(
            f"{first_token_mismatches} cached first target tokens differed from no-cache oracle outputs"
        )
    if unexpected_prompt_work:
        raise AssertionError(
            f"committed cache hits processed {unexpected_prompt_work} tokens beyond their uncached suffix"
        )
    if args.cache_ram_mib > 0 and summary["prompt_cache_accounted_bytes"] > args.cache_ram_mib * 1024 * 1024:
        raise AssertionError("RAM prompt-cache accounted payload bytes exceeded the configured budget")
    if summary["prompt_cache_restore_failures"]:
        raise AssertionError("RAM prompt-cache restore failures were reported")
    if summary["kv_tail_degraded_sequences"]:
        raise AssertionError("precision-tail degradation was reported")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--commit", default="unknown")
    parser.add_argument("--model-alias", default="model")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--turns", type=int, default=24)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--long-words", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--cache-type", default="kvarn4")
    parser.add_argument("--tail-tokens", type=int, default=512)
    parser.add_argument("--tail-type", default="f16")
    parser.add_argument("--cache-ram-mib", type=int, default=4096)
    parser.add_argument("--ctx-checkpoints", type=int, default=32)
    parser.add_argument("--checkpoint-min-step", type=int, default=0)
    parser.add_argument("--unified", action="store_true")
    parser.add_argument("--spec", choices=("none", "mtp", "dflash"), default="none")
    parser.add_argument("--draft-max", type=int, default=8)
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--no-oracle", dest="oracle", action="store_false")
    parser.add_argument("--expect-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-request-cache", action="store_true")
    parser.add_argument("--serialize-clients", action="store_true")
    parser.add_argument("--min-overlap", type=float, default=0.30)
    parser.add_argument("--min-reuse-efficiency", type=float, default=0.995)
    parser.add_argument("--sample-interval", type=float, default=0.02)
    parser.add_argument("--top-logprobs", type=int, default=0)
    parser.add_argument("--start-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=900)
    args = parser.parse_args()
    if args.concurrency > args.slots:
        parser.error("--concurrency cannot exceed --slots")
    for path in (args.server_bin, args.target_model):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    args = parse_args()
    (args.output_dir / "corpus.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "generator": "conversation_addition-v1",
                "seed": args.seed,
                "conversations": args.concurrency,
                "turns": args.turns,
                "long_words": args.long_words,
                "max_tokens": args.max_tokens,
                "requested_lengths": [32, 64, 128, 256, 512, 1024],
                "branch_rule": "drop-last-exchange when turn%7==0 or turn%8==1",
                "cache_ram_mib": args.cache_ram_mib,
                "ctx_checkpoints": args.ctx_checkpoints,
                "checkpoint_min_step": args.checkpoint_min_step,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    port = free_port()
    cached_server = Server(args, port, "cached")
    version_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    version = subprocess.run(
        [str(args.server_bin), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=version_flags,
        check=False,
    )
    safe_environment_names = (
        "CUDA_PATH",
        "CUDA_PATH_V13_1",
        "GGML_CUDA_ENABLE_UNIFIED_MEMORY",
        "GGML_CUDA_FORCE_CUBLAS",
        "GGML_CUDA_FORCE_MMQ",
        "GGML_CUDA_NO_VMM",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "commit": args.commit,
        "configuration": args.configuration,
        "server_binary": file_identity(args.server_bin, hash_contents=True),
        "server_version_exit_code": version.returncode,
        "server_version": (version.stdout + version.stderr).strip(),
        "target_model": file_identity(args.target_model),
        "draft_model": file_identity(args.draft_model),
        "server_arguments": cached_server.command,
        "environment_allowlist": {
            name: os.environ[name] for name in safe_environment_names if name in os.environ
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cached_server.start()
    final_metrics: dict[str, float] = {}
    try:
        with BusySampler(cached_server, args.output_dir / "metrics.jsonl", args.sample_interval) as sampler:
            cached = run_phase(args, cached_server, False, [])
        busy_samples = sampler.samples
        metrics_response = requests.get(f"{cached_server.url}/metrics", timeout=args.request_timeout)
        metrics_response.raise_for_status()
        metrics_text = metrics_response.text
        (args.output_dir / "final-metrics.prom").write_text(metrics_text, encoding="utf-8")
        final_metrics = parse_prometheus_metrics(metrics_text)
    finally:
        cached_server.stop()

    oracle: list[dict[str, Any]] = []
    if args.oracle:
        oracle_server = Server(args, port, "oracle")
        oracle_server.start()
        try:
            oracle = run_phase(args, oracle_server, True, cached)
        finally:
            oracle_server.stop()

    summary = summarize(args, cached, oracle, busy_samples, final_metrics)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"prompt_cache_real_usage: {error}", file=sys.stderr)
        raise
