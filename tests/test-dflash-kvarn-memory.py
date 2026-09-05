"""Opt-in two-CUDA-GPU DFlash memory regression; requires matching local GGUFs.

python tests/test-dflash-kvarn-memory.py --server build/bin/llama-server \
    --target target.gguf --draft dflash.gguf

Uses fresh servers per cache pair. Does not download models or require Hub access.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request


def run(args, cache):
    stem = args.output / cache.replace(":", "-")
    key, value = cache.split(":") if ":" in cache else (cache, cache)
    command = [str(args.server.resolve()), "-m", str(args.target.resolve()),
               "--spec-draft-model", str(args.draft.resolve()), "--spec-type", "draft-dflash",
               "--device", "CUDA0,CUDA1", "--spec-draft-device", "CUDA0,CUDA1",
               "--split-mode", "layer", "--tensor-split", "51,49", "--main-gpu", "1",
               "-ngl", "all", "--spec-draft-ngl", "all", "--load-mode", "dio", "--fit", "off",
               "-c", str(args.context), "-b", "2048", "-ub", "512", "-fa", "on",
               "--cache-type-k", "kvarn6", "--cache-type-v", "kvarn6", "--kv-tail-tokens", "1024",
               "--spec-draft-type-k", key, "--spec-draft-type-v", value,
               "--spec-draft-n-max", "8", "--parallel", "1", "--kv-unified",
               "--host", "127.0.0.1", "--port", str(args.port), "--no-warmup", "-lv", "5"]
    result = {"command": command, "cache": cache,
              "cuda_launch_blocking": os.getenv("CUDA_LAUNCH_BLOCKING"),
              "version": subprocess.check_output([str(args.server.resolve()), "--version"],
                                                 text=True, stderr=subprocess.STDOUT)}
    with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"})
        try:
            for _ in range(240):
                if proc.poll() is not None:
                    raise RuntimeError(f"{cache}: startup exited {proc.returncode}; see {stem}.log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.5)
            else:
                raise TimeoutError(f"{cache}: readiness timeout")
            text = stem.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
            allocations = re.findall(r"sched_reserve:\s+(CUDA[01]) compute buffer size =\s+([\d.]+) MiB", text)
            result["compute_mib"] = allocations
            assert allocations, "missing compute allocation diagnostics"
            # A full QK/softmax matrix costs multiple GiB at 64K with ubatch 512.
            # F16 K/V plus FlashAttention and tail merge must stay below this bound.
            assert max(float(size) for _, size in allocations) < args.max_compute_mib, (
                f"{cache}: unbounded attention compute allocation: {allocations}")
            if key.startswith("kvarn"):
                assert "KVarN attention route=materialized" in text, "DFlash must retain materialized KVarN"
            for attempt in range(args.requests):
                prompt = "alpha beta gamma delta epsilon zeta eta theta. " * args.prompt_repeats
                prompt += "\nWrite the first forty positive integers separated by commas."
                payload = {"prompt": prompt, "n_predict": 128, "temperature": 0,
                           "seed": 4242, "cache_prompt": False, "ignore_eos": True}
                request = urllib.request.Request(f"http://127.0.0.1:{args.port}/completion",
                    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=300) as response:
                    output = json.load(response)
                result.setdefault("responses", []).append(output)
                timings = output["timings"]
                assert timings["predicted_n"] >= 128, timings
                assert timings.get("draft_n", 0) > 0, "test did not exercise draft generation"
        except Exception as exc:
            result["error"] = repr(exc)
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            result["exit_after_cleanup"] = proc.returncode
            stem.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"PASS {cache}: compute={allocations}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "target", "draft"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tmp/dflash-kvarn-memory"))
    parser.add_argument("--profiles", nargs="+", default=["q8_0", "kvarn2", "kvarn3", "kvarn4", "kvarn5", "kvarn6", "kvarn8", "kvarn4:kvarn2"])
    parser.add_argument("--context", type=int, default=64000)
    parser.add_argument("--max-compute-mib", type=float, default=2048)
    parser.add_argument("--prompt-repeats", type=int, default=1200)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--port", type=int, default=18340)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for cache in args.profiles:
        run(args, cache)


if __name__ == "__main__":
    main()
