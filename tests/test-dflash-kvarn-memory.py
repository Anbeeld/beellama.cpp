"""Opt-in two-CUDA-GPU DFlash memory regression; requires matching local GGUFs.

python tests/test-dflash-kvarn-memory.py --server build/bin/llama-server \
    --target target.gguf --draft dflash.gguf

Uses fresh servers per cache pair. Does not download models or require Hub access.

Image-position/acceptance regression with Qwen3.8-27B-UD-IQ4_XS and
Qwen3.8-27B-DFlash2-Q4_K_M: add matching --mmproj, --image media/llama1-icon.png,
--profiles q8_0 kvarn6 --prompt-repeats 20 --predict 256 --min-draft-acceptance 0.5
--instruction "Ignore the image if present. Write the integers from 1 through 100
in ascending order, separated by commas. Do not explain."
The acceptance floor is workload-specific, not a general quality guarantee.
"""
import argparse
import base64
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
               "--split-mode", args.split_mode, "--tensor-split", "51,49", "--main-gpu", "1",
               "-ngl", "all", "--spec-draft-ngl", "all", "--load-mode", "dio", "--fit", "off",
               "-c", str(args.context), "-b", "2048", "-ub", "512", "-fa", "on",
               "--cache-type-k", "kvarn6", "--cache-type-v", "kvarn6", "--kv-tail-tokens", "1024",
               "--spec-draft-type-k", key, "--spec-draft-type-v", value,
               "--spec-draft-n-max", "8", "--parallel", "1", "--kv-unified",
               "--host", "127.0.0.1", "--port", str(args.port), "--no-warmup", "-lv", "5"]
    if args.mmproj:
        command += ["--mmproj", str(args.mmproj.resolve()), "--image-min-tokens", str(args.image_min_tokens)]
    if args.swa_full:
        command += ["--swa-full"]
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
            allocations = re.findall(r"sched_reserve:\s+(CUDA[01]|Meta\(\)) compute buffer size =\s+([\d.]+) MiB", text)
            result["compute_mib"] = allocations
            swa_cells = re.findall(r"creating\s+SWA KV cache, size = (\d+) cells", text)
            result["startup_swa_cells"] = [int(cells) for cells in swa_cells]
            if args.max_startup_swa_cells is not None:
                assert swa_cells, "missing SWA capacity diagnostics"
                assert max(map(int, swa_cells)) <= args.max_startup_swa_cells, (
                    f"{cache}: oversized startup SWA capacity: {swa_cells}")
            assert allocations, "missing compute allocation diagnostics"
            # A full QK/softmax matrix costs multiple GiB at 64K with ubatch 512.
            # F16 K/V plus FlashAttention and tail merge must stay below this bound.
            assert max(float(size) for _, size in allocations) < args.max_compute_mib, (
                f"{cache}: unbounded attention compute allocation: {allocations}")
            if key.startswith("kvarn"):
                assert "KVarN attention route=materialized" in text, "DFlash must retain materialized KVarN"
            for attempt in range(args.requests):
                prompt = "alpha beta gamma delta epsilon zeta eta theta. " * args.prompt_repeats
                prompt += "\n" + args.instruction
                payload = {"prompt": prompt, "n_predict": args.predict, "temperature": 0,
                           "seed": 4242, "cache_prompt": False, "ignore_eos": True}
                endpoint = "completion"
                if args.image:
                    image = base64.b64encode(args.image.read_bytes()).decode()
                    content = [{"type": "text", "text": prompt},
                               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}}]
                    if attempt % 2 == 0:
                        content.reverse()
                    payload = {"messages": [{"role": "user", "content": content}],
                               "max_tokens": args.predict, "temperature": 0, "seed": 4242,
                               "cache_prompt": False, "ignore_eos": True}
                    endpoint = "v1/chat/completions"
                request = urllib.request.Request(f"http://127.0.0.1:{args.port}/{endpoint}",
                    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=300) as response:
                    output = json.load(response)
                result.setdefault("responses", []).append(output)
                timings = output["timings"]
                assert timings["predicted_n"] >= args.predict, timings
                assert timings.get("draft_n", 0) > 0, "test did not exercise draft generation"
                if args.min_draft_acceptance is not None:
                    acceptance = timings.get("draft_n_accepted", 0) / timings["draft_n"]
                    assert acceptance >= args.min_draft_acceptance, (
                        f"{cache}: draft acceptance {acceptance:.4f} below "
                        f"{args.min_draft_acceptance}: {timings}")
            if args.require_swa_growth:
                text = stem.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
                assert "grew owned DFlash SWA cache" in text, "test did not exercise SWA growth"
                assert "could not grow DFlash SWA cache" not in text, "SWA growth failed"
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
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--image", type=Path, help="PNG image; exercises image-first and text-first requests")
    parser.add_argument("--image-min-tokens", type=int, default=1024)
    parser.add_argument("--swa-full", action="store_true", help="explicit full-SWA reference configuration")
    parser.add_argument("--require-swa-growth", action="store_true", help="assert image requests grow the compact ring")
    parser.add_argument("--output", type=Path, default=Path("tmp/dflash-kvarn-memory"))
    parser.add_argument("--profiles", nargs="+", default=["q8_0", "kvarn2", "kvarn3", "kvarn4", "kvarn5", "kvarn6", "kvarn8", "kvarn4:kvarn2"])
    parser.add_argument("--context", type=int, default=64000)
    parser.add_argument("--split-mode", choices=["layer", "tensor"], default="layer")
    parser.add_argument("--max-compute-mib", type=float, default=2048)
    parser.add_argument("--max-startup-swa-cells", type=int,
                        help="bound startup SWA rows independently of the full context")
    parser.add_argument("--prompt-repeats", type=int, default=1200)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--predict", type=int, default=128)
    parser.add_argument("--instruction", default="Write the first forty positive integers separated by commas.")
    parser.add_argument("--min-draft-acceptance", type=float,
                        help="optional acceptance floor for a known predictable prompt/model pair")
    parser.add_argument("--port", type=int, default=18340)
    args = parser.parse_args()
    if bool(args.mmproj) != bool(args.image):
        parser.error("--mmproj and --image must be supplied together")
    args.output.mkdir(parents=True, exist_ok=True)
    for cache in args.profiles:
        run(args, cache)


if __name__ == "__main__":
    main()
