"""Opt-in DFlash historical-checkpoint regression on two CUDA GPUs.

Requires matching local --target and --draft GGUFs. No model downloads.
Appends beyond the draft SWA window, then returns to a historical prefix.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def run(args, cache):
    output = args.output / cache
    output.mkdir(parents=True, exist_ok=True)
    command = [str(args.server.resolve()), "-m", str(args.target.resolve()),
               "--spec-draft-model", str(args.draft.resolve()), "--spec-type", "draft-dflash",
               "--spec-draft-type-k", cache, "--spec-draft-type-v", cache,
               "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--kv-tail-tokens", "0",
               "-c", "64000", "-b", "2048", "-ub", "512", "-ngl", "all", "--spec-draft-ngl", "all",
               "--device", "CUDA0,CUDA1", "--spec-draft-device", "CUDA0,CUDA1", "--tensor-split", "51,49",
               "--load-mode", "dio", "--fit", "off", "--parallel", str(args.parallel),
               "--ctx-checkpoints", "32", "--checkpoint-min-step", "512",
               "--host", "127.0.0.1", "--port", str(args.port), "--no-warmup", "-lv", "5"]
    result = {"command": command, "responses": [],
              "cuda_launch_blocking": os.getenv("CUDA_LAUNCH_BLOCKING"),
              "version": subprocess.check_output([str(args.server.resolve()), "--version"],
                                                 text=True, stderr=subprocess.STDOUT)}
    url = f"http://127.0.0.1:{args.port}"
    with (output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"})
        try:
            for _ in range(240):
                if process.poll() is not None:
                    raise RuntimeError(f"server exited: {process.returncode}")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1):
                        break
                except OSError:
                    time.sleep(0.5)
            else:
                raise TimeoutError("server readiness timeout")
            prefix = "alpha beta gamma delta epsilon zeta eta theta. " * 1200
            prompts = [prefix,
                       prefix + " one two three four five six seven eight nine. " * 600,
                       prefix + " altered suffix."]
            if args.parallel > 1:
                prompts = ["Independent sequence. " * 100] + prompts + ["Independent sequence. " * 100]
            for index, prompt in enumerate(prompts):
                slot = 1 if args.parallel > 1 and index in (0, len(prompts) - 1) else 0
                request = urllib.request.Request(url + "/completion", data=json.dumps({
                    "prompt": prompt, "n_predict": 128, "ignore_eos": True,
                    "temperature": 0, "seed": 4242, "cache_prompt": True, "id_slot": slot,
                }).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=300) as response:
                    data = json.load(response)
                result["responses"].append(data)
                assert data["timings"]["predicted_n"] == 128, data["timings"]
                assert data["timings"].get("draft_n", 0) > 0, "draft generation not exercised"
            if args.parallel > 1:
                assert result["responses"][-1]["timings"]["cache_n"] > 200, "restore discarded another slot"
            timings = result["responses"][-2 if args.parallel > 1 else -1]["timings"]
            assert timings["cache_source"] == "checkpoint", timings
            assert timings["cache_reason"] == "committed", timings
            assert timings["cache_n"] > 10000, timings
            assert timings["prompt_n"] < 2048, timings
            # A successful restore must preserve output, not merely suppress
            # the anchor diagnostic. Re-evaluate the same prompt from scratch.
            restored = result["responses"][-2 if args.parallel > 1 else -1]
            request = urllib.request.Request(url + "/completion", data=json.dumps({
                "prompt": prefix + " altered suffix.", "n_predict": 128, "ignore_eos": True,
                "temperature": 0, "seed": 4242, "cache_prompt": False, "id_slot": 0,
            }).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as response:
                cold = json.load(response)
            result["cold_reference"] = cold
            assert cold["timings"]["cache_n"] == 0, cold["timings"]
            assert cold["content"] == restored["content"], "restored output differs from cold prefill"
            text = (output / "server.log").read_text(encoding="utf-8", errors="replace")
            assert "checkpoint restore preparation failed" not in text, "checkpoint rejected"
            assert "partial KV state no longer" not in text, "checkpoint anchor lost"
        except Exception as error:
            result["error"] = repr(error)
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"PASS {cache}: {timings}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "target", "draft"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tmp/dflash-kvarn-checkpoint"))
    parser.add_argument("--profiles", nargs="+", default=["q8_0", "kvarn2", "kvarn3", "kvarn4", "kvarn5", "kvarn6", "kvarn8"])
    parser.add_argument("--parallel", type=int, choices=[1, 2], default=1)
    parser.add_argument("--port", type=int, default=18341)
    args = parser.parse_args()
    for cache in args.profiles:
        run(args, cache)


if __name__ == "__main__":
    main()
