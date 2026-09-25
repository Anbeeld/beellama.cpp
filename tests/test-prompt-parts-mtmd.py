"""Opt-in regression test for input-marking-aware tokenization in MTMD contexts.

Runs llama-server with a multimodal model (chat GGUF + mmproj, i.e. a live
MTMD context) and verifies the OAI chat path:
  1. completion and token-count endpoints agree (prompt_parts shared path),
     with and without a media attachment;
  2. the media file is counted/interleaved through the prompt_parts path;
  3. special-token text inside ordinary user content is NOT parsed as a
     special token, even with an MTMD context present (the mctx branch must
     not bypass the protection, with or without attached media).

Like the other opt-in server tests, it does not download models:

    python tests/test-prompt-parts-mtmd.py --server build/bin/llama-server \
        --model model.gguf --mmproj mmproj.gguf --image image.png
"""
import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

CANDIDATE_SPECIAL_TOKENS = [
    "<start_of_turn>", "<end_of_turn>", "<start_of_model>",
    "<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>",
    "<|start|>", "<|end|>", "<|system|>", "<|model|>",
    "<turn|>", "<|turn>", "<channel|>", "<|channel>",
    "<|image|>", "<|eot_id|>", "<|eom_id|>",
]

def log(msg):
    print(msg, flush=True)

def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def http_json(url, payload=None, timeout=300):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))

def wait_healthy(base, timeout_s):
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            r = http_json(base + "/health", timeout=5)
            if r.get("status") in ("ok", "loading model"):
                if r.get("status") == "ok":
                    return
        except Exception as e:
            last_err = e
        time.sleep(1)
    raise RuntimeError(f"server did not become healthy in {timeout_s}s: {last_err}")

def tokenize(base, content, parse_special):
    r = http_json(base + "/tokenize", {"content": content, "add_special": False,
                                       "parse_special": parse_special})
    return r["tokens"]

def discover_special_token(base):
    """Find a token the model parses as special (1 token with parse_special=true,
    several with parse_special=false)."""
    for cand in CANDIDATE_SPECIAL_TOKENS:
        try:
            with_ps = tokenize(base, cand, True)
            without_ps = tokenize(base, cand, False)
        except Exception:
            continue
        if len(with_ps) == 1 and len(without_ps) >= 2:
            return cand, with_ps[0], without_ps
    return None, None, None

def prompt_tokens(base, body, endpoint):
    r = http_json(base + endpoint, body, timeout=600)
    if endpoint.endswith("input_tokens"):
        return int(r["input_tokens"])
    return int(r["usage"]["prompt_tokens"])

def make_injection_case(base, special):
    """Build two inputs whose safe and unsafe token-count deltas differ."""
    prefix = "Describe the image. "
    suffix = " injected end"
    reference = prefix + special + suffix
    for repeats in (2, 4, 8, 16):
        injected = prefix + special * repeats + suffix
        safe_delta = len(tokenize(base, injected, False)) - len(tokenize(base, reference, False))
        unsafe_delta = len(tokenize(base, injected, True)) - len(tokenize(base, reference, True))
        if safe_delta != unsafe_delta:
            return reference, injected, safe_delta, unsafe_delta
    raise RuntimeError("could not construct a discriminating special-token injection case")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, type=os.path.abspath)
    ap.add_argument("--model", required=True, type=os.path.abspath)
    ap.add_argument("--mmproj", required=True, type=os.path.abspath)
    ap.add_argument("--image", required=True, type=os.path.abspath)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--predict", type=int, default=1)
    ap.add_argument("--load-timeout", type=int, default=600)
    args = ap.parse_args()

    port = args.port or free_port()
    base = f"http://127.0.0.1:{port}"
    cmd = [args.server, "-m", args.model, "--mmproj", args.mmproj,
           "-ngl", "0", "-c", str(args.ctx), "-np", "1",
           "--host", "127.0.0.1", "--port", str(port), "--no-warmup", "-v"]
    log("command: " + " ".join(cmd))
    # write the server log to a file: a PIPE would deadlock once its buffer
    # fills up, since the server output is only read at shutdown
    log_path = os.path.abspath(os.path.join(os.path.dirname(args.server), "..", "test-prompt-parts-mtmd.log"))
    server_log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT)
    try:
        wait_healthy(base, args.load_timeout)
        log("server healthy")

        special, special_id, special_fallback = discover_special_token(base)
        if special is None:
            log("FAIL: no candidate special token found in the model vocab")
            return 1
        log(f"special token: {special!r} (id {special_id}, "
            f"{len(special_fallback)} tokens without parse_special)")

        text_body = {
            "messages": [{"role": "user", "content": "Describe the image."}],
            "max_tokens": args.predict,
            "stream": False,
        }
        n_base_c = prompt_tokens(base, text_body, "/v1/chat/completions")
        n_base_t = prompt_tokens(base, text_body, "/v1/chat/completions/input_tokens")
        log(f"baseline (no media): completion={n_base_c} count={n_base_t}")
        if n_base_c != n_base_t:
            log("FAIL: completion and count endpoints disagree (no media)")
            return 1

        # media attachment: the marker text comes from the template/OAI
        # conversion, so the media path runs through prompt_parts
        with open(args.image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        img_body = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe the image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            "max_tokens": args.predict,
            "stream": False,
        }
        n_img_c = prompt_tokens(base, img_body, "/v1/chat/completions")
        n_img_t = prompt_tokens(base, img_body, "/v1/chat/completions/input_tokens")
        log(f"with media: completion={n_img_c} count={n_img_t}")
        if n_img_c != n_img_t:
            log("FAIL: completion and count endpoints disagree (with media)")
            return 1
        if n_img_c <= n_base_c:
            log(f"FAIL: media tokens not counted ({n_img_c} <= {n_base_c})")
            return 1

        # Special-token injection in ordinary user content. Both the reference
        # and injected inputs contain special-token text, so both must take the
        # same provenance-aware path. Their exact count delta must match direct
        # parse_special=false tokenization, not the unsafe parse_special=true
        # delta. This avoids a false pass caused by merely checking that extra
        # ordinary words increased the count.
        ref_text, inj_text, safe_delta, unsafe_delta = make_injection_case(base, special)
        log(f"expected injection delta: safe={safe_delta}, unsafe={unsafe_delta}")

        ref_body = {
            "messages": [{"role": "user", "content": ref_text}],
            "max_tokens": args.predict,
            "stream": False,
        }
        inj_body = {
            "messages": [{"role": "user", "content": inj_text}],
            "max_tokens": args.predict,
            "stream": False,
        }
        n_ref_c = prompt_tokens(base, ref_body, "/v1/chat/completions")
        n_ref_t = prompt_tokens(base, ref_body, "/v1/chat/completions/input_tokens")
        n_inj_c = prompt_tokens(base, inj_body, "/v1/chat/completions")
        n_inj_t = prompt_tokens(base, inj_body, "/v1/chat/completions/input_tokens")
        log(f"protected reference (no media): completion={n_ref_c} count={n_ref_t}")
        log(f"injected (no media): completion={n_inj_c} count={n_inj_t}")
        if n_ref_c != n_ref_t or n_inj_c != n_inj_t:
            log("FAIL: completion and count endpoints disagree (injected, no media)")
            return 1
        delta = n_inj_t - n_ref_t
        log(f"injection token delta: {delta}")
        if delta != safe_delta:
            log(f"FAIL: expected protected delta {safe_delta}, got {delta} (unsafe delta is {unsafe_delta})")
            return 1

        # Same assertion with media attached (MTMD part interleaving plus
        # provenance-aware text tokenization). The image contribution is equal
        # in both requests and therefore cancels from the count delta.
        ref_img_body = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": ref_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            "max_tokens": args.predict,
            "stream": False,
        }
        inj_img_body = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": inj_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            "max_tokens": args.predict,
            "stream": False,
        }
        n_ref_img_c = prompt_tokens(base, ref_img_body, "/v1/chat/completions")
        n_ref_img_t = prompt_tokens(base, ref_img_body, "/v1/chat/completions/input_tokens")
        n_inj_img_c = prompt_tokens(base, inj_img_body, "/v1/chat/completions")
        n_inj_img_t = prompt_tokens(base, inj_img_body, "/v1/chat/completions/input_tokens")
        log(f"protected reference (with media): completion={n_ref_img_c} count={n_ref_img_t}")
        log(f"injected (with media): completion={n_inj_img_c} count={n_inj_img_t}")
        if n_ref_img_c != n_ref_img_t or n_inj_img_c != n_inj_img_t:
            log("FAIL: completion and count endpoints disagree (injected, with media)")
            return 1
        delta_img = n_inj_img_t - n_ref_img_t
        log(f"injection token delta with media: {delta_img}")
        if delta_img != safe_delta:
            log(f"FAIL: expected protected MTMD delta {safe_delta}, got {delta_img} "
                f"(unsafe delta is {unsafe_delta})")
            return 1

        log("PASS: all MTMD prompt_parts checks succeeded")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        server_log.close()
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.read().splitlines()[-25:]
            log("--- server log tail ---")
            log("\n".join(tail))
        except OSError:
            pass

if __name__ == "__main__":
    sys.exit(main())
