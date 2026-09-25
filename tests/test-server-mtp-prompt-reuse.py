#!/usr/bin/env python3
"""Integration regression for MTP prompt reuse with recurrent target state.

Run against a locally started llama-server with draft-mtp and a model with
recurrent layers. The caller controls the model, device, and server options.
"""
import argparse
import json
import urllib.request


def completion(url, prompt, n_predict=1, cache_prompt=True):
    request = urllib.request.Request(
        url.rstrip("/") + "/completion",
        json.dumps({
            "prompt": prompt,
            "n_predict": n_predict,
            "n_probs": 256 if n_predict == 1 else 10,
            "post_sampling_probs": False,
            "cache_prompt": cache_prompt,
            "temperature": 1.0 if n_predict == 1 else 0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "repeat_last_n": 0,
            "seed": 42,
        }).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    assert "error" not in result, result
    return result


def probabilities(result):
    return {item["id"]: item["logprob"] for item in
            result["completion_probabilities"][0]["top_logprobs"]}


def assert_same_scores(actual, expected):
    actual, expected = probabilities(actual), probabilities(expected)
    assert actual.keys() == expected.keys(), (actual, expected)
    for token, value in expected.items():
        assert abs(actual[token] - value) < 1e-4, (token, actual[token], value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL of a running draft-mtp llama-server")
    args = parser.parse_args()
    base = ("Classify this support ticket: I was charged twice for my subscription. "
            "Please refund the duplicate payment. Answer one category: ")
    prompt = base + "billing, shipping, access, technical. Category:"
    alternate = base + "technical, access, shipping, billing. Category:"
    first = completion(args.url, prompt)
    repeat = completion(args.url, prompt)
    assert repeat["timings"]["prompt_n"] == 1, (first["timings"], repeat["timings"])
    assert_same_scores(repeat, first)
    changed_tail = base + "billing, shipping, access, technical. Category?"
    changed_cached = completion(args.url, changed_tail)
    changed_fresh = completion(args.url, changed_tail, cache_prompt=False)
    assert_same_scores(changed_cached, changed_fresh)
    completion(args.url, alternate)
    again = completion(args.url, prompt)
    assert_same_scores(again, first)

    # Exercise actual MTP drafting after prompt reuse, not just prefill logits.
    fresh_generation = completion(args.url, prompt, n_predict=24, cache_prompt=False)
    cached_generation = completion(args.url, prompt, n_predict=24)
    assert cached_generation["content"] == fresh_generation["content"], (
        cached_generation["content"], fresh_generation["content"])
    assert_same_scores(cached_generation, fresh_generation)
    print("MTP cached repeated and edited prompts match initial probabilities and generation")
    print("prompt latency: cold %.2f ms, cached %.2f ms; generated tokens: cold %.2f ms, cached %.2f ms" % (
        first["timings"]["prompt_ms"], repeat["timings"]["prompt_ms"],
        fresh_generation["timings"]["predicted_ms"], cached_generation["timings"]["predicted_ms"]))


if __name__ == "__main__":
    main()
