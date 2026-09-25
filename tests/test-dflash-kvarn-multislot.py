#!/usr/bin/env python3
"""Opt-in DFlash KVarN multi-slot native/unified and fallback isolation regression."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request


def request_json(port, path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run(args, route, nslots):
    folder = args.output / route / f"slots-{nslots}"
    folder.mkdir(parents=True, exist_ok=True)
    command = [str(args.server.resolve()), "-m", str(args.target.resolve()),
               "--spec-type", "draft-dflash", "--spec-draft-model", str(args.draft.resolve()),
               "--cache-type-k", "kvarn4", "--cache-type-v", "kvarn4",
               "--spec-draft-type-k", "kvarn4", "--spec-draft-type-v", "kvarn4",
               "--device", "CUDA0,CUDA1", "--spec-draft-device", "CUDA0,CUDA1",
               "--tensor-split", "51,49", "--split-mode", "layer", "--main-gpu", "1",
               "-ngl", "all", "--spec-draft-ngl", "all", "-c", "32768", "-b", "2048",
               "-ub", "512", "-ubd", "128", "--parallel", str(nslots),
               "--load-mode", "dio", "--fit", "off", "--flash-attn", "on", "--seed", "1234",
               "--host", "127.0.0.1", "--port", str(args.port), "--no-warmup", "-lv", "5"]
    if args.unified:
        command.append("--kv-unified")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    if route == "materialized":
        env["LLAMA_KVARN_TEST_MATERIALIZE_NONCAUSAL"] = "1"
    else:
        env.pop("LLAMA_KVARN_TEST_MATERIALIZE_NONCAUSAL", None)
    (folder / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    proc = None
    try:
        with (folder / "server.log").open("wb") as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            for _ in range(600):
                if proc.poll() is not None:
                    raise RuntimeError(f"server startup exit: {proc.returncode}")
                try:
                    if request_json(args.port, "/health", timeout=2).get("status") == "ok":
                        break
                except OSError:
                    time.sleep(0.5)
            else:
                raise TimeoutError("server startup")
            def request_slot(slot):
                prompt = (f"Slot {slot} counting sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ") * (40 + slot * 20)
                return request_json(args.port, "/completion", {
                    "prompt": prompt, "n_predict": 96, "temperature": 0, "seed": 1234,
                    "cache_prompt": False, "return_tokens": True, "ignore_eos": True,
                    "id_slot": slot,
                })
            with concurrent.futures.ThreadPoolExecutor(max_workers=nslots) as pool:
                futures = {pool.submit(request_slot, 0): 0}
                for i in range(1, nslots):
                    # Start each next request after the previous slot is assigned,
                    # while earlier requests remain active. Avoid changing which
                    # prompt the server schedules first between route comparisons.
                    for _ in range(300):
                        text = (folder / "server.log").read_text(encoding="utf-8", errors="replace")
                        if re.search(rf"id {i - 1:2d} \| task \d+ \| processing task", text):
                            break
                        if next(iter(futures)).done():
                            raise RuntimeError("first request completed before later slot was assigned")
                        time.sleep(0.01)
                    else:
                        raise TimeoutError(f"slot {i - 1} was not assigned")
                    futures[pool.submit(request_slot, i)] = i
                answers = {futures[f]: f.result() for f in concurrent.futures.as_completed(futures)}
            text = (folder / "server.log").read_text(encoding="utf-8", errors="replace")
            expected_route = "native" if args.unified and route == "native" else "materialized"
            for il in range(5):
                assert f"DFlash layer {il} KVarN attention route={expected_route}" in text, (route, il)
            if expected_route == "materialized":
                assert "KVarN attention route=native" not in text, "oracle or multi-stream SWA selected native"
            assert "layer   0 assigned to device CUDA0, is_swa = 1" in text, "draft layer on CUDA0 not exercised"
            assert "layer   4 assigned to device CUDA1, is_swa = 1" in text, "draft layer on CUDA1 not exercised"
            for i, response in answers.items():
                assert response["timings"].get("draft_n", 0) > 0, f"slot {i} did not draft"
                assert len(response.get("tokens", [])) == 96, f"slot {i} stopped early"
            result = {"command": command, "route": route, "slots": nslots, "responses": answers}
            (folder / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return answers
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "target", "draft"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tmp/dflash-kvarn-multislot"))
    parser.add_argument("--port", type=int, default=18342)
    parser.add_argument("--unified", action="store_true", help="share a draft stream while keeping concurrent slots")
    args = parser.parse_args()
    for nslots in (2, 4):
        native = run(args, "native", nslots)
        oracle = run(args, "materialized", nslots)
        for i in range(nslots):
            assert native[i]["tokens"] == oracle[i]["tokens"], f"slot {i} target mismatch"
        label = "unified native" if args.unified else "fallback"
        print(f"PASS {nslots} slots: {label}/oracle target tokens identical", flush=True)


if __name__ == "__main__":
    main()
