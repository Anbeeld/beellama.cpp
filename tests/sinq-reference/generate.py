#!/usr/bin/env python3
"""Generate CUDA0 reference data with the unmodified, hash-pinned SINQ oracle."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import types
import urllib.request

import numpy as np
import torch

REVISION = "42ca83edf8fe1afc48c40fca6acc973c5d2bb59f"
HASHES = {
    "sinkhorn.py": "25423fbc588157432392f6b256f49715bebefce6f4c7824b8503dea99a65552a",
    "dual_shift.py": "8f1518777b826971fe2595df662f9614a423993c35e4548b24adf26f69c6c551",
    "optimize.py": "7879b8aff65ac721429fd3af6483444d9e2dbaa061a81d8d79b8c0ad6e2b3aa0",
    "quantizer.py": "6eed9b75a9847076950cf2d9fa749b58f325e22524e82597f7058a432a94075e",
}
HERE = Path(__file__).resolve().parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_oracle(source_dir):
    """Check every source byte; cache missing sources only inside our owned tree."""
    sources = {}
    for name, digest in HASHES.items():
        path = source_dir / name
        if not path.is_file():
            path = HERE / "cache" / REVISION / name
            if not path.is_file():
                url = f"https://raw.githubusercontent.com/huawei-csl/SINQ/{REVISION}/sinq/{name}"
                data = urllib.request.urlopen(url, timeout=60).read()
                if sha256(data) != digest:
                    raise RuntimeError(f"pinned download hash mismatch: {name}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        data = path.read_bytes()
        if sha256(data) != digest:
            raise RuntimeError(f"pinned source hash mismatch: {path}")
        sources[name] = path

    # A package shell avoids importing upstream __init__ and its serving dependencies.
    package = types.ModuleType("_sinq_fixture_oracle")
    package.__path__ = []
    sys.modules[package.__name__] = package
    for name in ("sinkhorn", "dual_shift"):
        qualified = f"{package.__name__}.{name}"
        spec = importlib.util.spec_from_file_location(qualified, sources[name + ".py"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)

    # Compile the exact pinned function, not a rewritten gate. optimize.py imports
    # AWQ/datasets, which are unrelated to this activation-free quantization path.
    tree = ast.parse(sources["optimize.py"].read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "kurtosis")
    optimize = types.ModuleType(f"{package.__name__}.optimize")
    optimize.torch = torch
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(sources["optimize.py"]), "exec"), optimize.__dict__)
    return package.dual_shift, optimize


def validate_shape(w, group):
    if w.ndim != 2 or w.shape[0] < 2 or group not in (64, 128) or w.shape[1] % group:
        raise ValueError("require N >= 2, G in {64,128}, and K divisible by G; no tile adjustment")


def synthetic_cases(group):
    # Fixed CPU RNG and explicit arithmetic, then a single fp16 rounding on CUDA.
    rng = np.random.Generator(np.random.PCG64(20260519 + group))
    yield "ordinary", rng.normal(0, 0.3, (8, group)).astype(np.float32)
    yield "constant", np.full((5, group), 0.125, dtype=np.float32)
    yield "near_zero", rng.normal(0, 2e-5, (5, group)).astype(np.float32)
    yield "asymmetric", (rng.uniform(-0.1, 0.8, (7, group)) ** 2 + 0.15).astype(np.float32)
    w = rng.normal(0, 0.03, (5, 2 * group)).astype(np.float32)
    w[1, group + 3] = 30
    w[3, 7] = -15
    yield "outlier", w
    yield "non_power_rows", rng.normal(0, 0.2, (3, group)).astype(np.float32)
    w = rng.normal(0, 0.2, (7, 3 * group)).astype(np.float32)
    w[:, group:2 * group] *= 3
    w[0] *= 2
    yield "multi_group", w


def model_cases(model_dir, rows=7):
    """Read small earliest-available attention/MLP slices without whole tensors."""
    if model_dir is None:
        return [], "not requested; pass --model-dir once all indexed shards are ready"
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return [], "model index is missing"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    missing = sorted({name for name in weight_map.values() if not (model_dir / name).is_file()})
    if missing:
        return [], f"download incomplete: {len(missing)} indexed weight shards missing"
    from safetensors import safe_open

    selected = []
    for suffix in ("self_attn.q_proj.weight", "mlp.down_proj.weight"):
        names = sorted((n for n in weight_map if ".layers." in n and n.endswith(suffix)),
                       key=lambda n: int(n.split(".layers.")[1].split(".")[0]))
        if not names:
            raise RuntimeError(f"model matrix not found: {suffix}")
        name = names[0]
        shard = model_dir / weight_map[name]
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            shape = view.get_shape()
            if len(shape) != 2 or shape[0] < rows or shape[1] < 256:
                raise RuntimeError(f"model matrix too small: {name}, {shape}")
            w = view[:rows, :256].float().numpy()
        selected.append(("qwen35_" + suffix.replace(".weight", "").replace(".", "_"), w,
                         {"tensor": name, "shard": shard.name, "slice": [0, rows, 0, 256],
                          "source_shape": shape, "slice_fp32_sha256": sha256(w.astype("<f4").tobytes())}))
    return selected, None


def write_binary(path, w, q, s, z, t, group, bits, fallback):
    n, k = w.shape
    arrays = [x.detach().cpu().numpy().astype("<f4").ravel() for x in (w, q, s, z, t)]
    assert [a.size for a in arrays] == [n * k, n * k, n * k // group, n * k // group, k]
    data = struct.pack("<7I", 0x514E4953, 1, n, k, group, bits, int(fallback))
    data += b"".join(a.tobytes() for a in arrays)
    path.write_bytes(data)
    # Independently read back header, length, and all array payloads.
    reread = path.read_bytes()
    assert struct.unpack("<7I", reread[:28]) == (0x514E4953, 1, n, k, group, bits, int(fallback))
    assert len(reread) == 28 + 4 * sum(a.size for a in arrays)
    np.testing.assert_array_equal(np.frombuffer(reread, dtype="<f4", offset=28), np.concatenate(arrays))
    return {"file": path.name, "sha256": sha256(data), "bytes": len(data)}


def error_stats(w, q, s, z, t, group):
    reconstructed = ((q - z) * s).reshape(w.shape) * t
    error = reconstructed - w
    return {"max_abs": error.abs().max().item(), "rmse": error.square().mean().sqrt().item(),
            "mean_abs": error.abs().mean().item()}


@torch.inference_mode()
def generate(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA0 is required; CPU oracle substitution is not allowed")
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dual, optimize = load_oracle(args.source_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    model, model_gap = model_cases(args.model_dir, args.model_rows)
    manifest = {
        "format_version": 1, "oracle_revision": REVISION, "source_sha256": HASHES,
        "environment": {"torch": torch.__version__, "numpy": np.__version__,
                        "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(0),
                        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]},
        "config": {"bits": [3, 4], "groups": [64, 128], "seed": 20260519,
                   "input": "CUDA0 float16, then float32 for pre-cast oracle call",
                   "gate": "pinned optimize.kurtosis on whole half-rounded matrix, variance clamp 1e-6, non-Fisher, > 5 => rtn",
                   "normalization": "pinned sinkhorn_log(matrix, 16), unchanged defaults",
                   "post_cast": "S,Z,T: half().float(); Q unchanged", "tile_adjustment": "forbidden"},
        "model_gap": model_gap, "fixtures": [], "invalid_shapes": [],
    }
    for group in (64, 128):
        cases = [(name, array, None) for name, array in synthetic_cases(group)] + model
        for name, array, origin in cases:
            w = torch.from_numpy(array).to(device="cuda:0", dtype=torch.float16).float()
            validate_shape(w, group)
            kurt = float(optimize.kurtosis(w))
            fallback = kurt > 5.0
            modes = [("default", "rtn" if fallback else "sinq", fallback)]
            if fallback:
                modes.append(("ungated", "sinq", False))
            for bits in (3, 4):
                for gate_name, method, effective_fallback in modes:
                    q, s, t, z = dual.tiled_quant_rectangle(w, [0, (1 << bits) - 1], group, method=method)
                    assert q.dtype == s.dtype == z.dtype == t.dtype == torch.float32
                    assert all(torch.isfinite(x).all().item() for x in (w, q, s, z, t))
                    assert ((q >= 0) & (q <= (1 << bits) - 1) & (q == q.round())).all().item()
                    post_s, post_z, post_t = [x.half().float() for x in (s, z, t)]
                    # Check the actual CUDA fp16 oracle path independently of the
                    # fp32 call used to expose metadata before its final cast.
                    half_q, half_s, half_t, half_z = dual.tiled_quant_rectangle(
                        w.half(), [0, (1 << bits) - 1], group, method=method)
                    for expected, actual in ((q, half_q), (post_s, half_s),
                                             (post_z, half_z), (post_t, half_t)):
                        assert torch.equal(expected, actual.float())
                    assert all(torch.isfinite(x).all().item() for x in (post_s, post_z, post_t))
                    stem = f"{name}_g{group}_b{bits}_{gate_name}"
                    pre = write_binary(args.output / (stem + ".pre.bin"), w, q, s, z, t, group, bits, effective_fallback)
                    post = write_binary(args.output / (stem + ".bin"), w, q, post_s, post_z, post_t, group, bits, effective_fallback)
                    manifest["fixtures"].append({"name": stem, "shape": list(w.shape), "group": group,
                        "bits": bits, "gate": gate_name, "whole_matrix_kurtosis": kurt,
                        "default_gate_fallback": fallback, "fallback": effective_fallback, "method": method,
                        "pre_fp16": pre, "serialized_fp16": post,
                        "expected_errors_pre_fp16": error_stats(w, q, s, z, t, group),
                        "expected_errors_serialized_fp16": error_stats(w, q, post_s, post_z, post_t, group),
                        "model_source": origin})
    for group in (64, 128):
        w = torch.zeros((3, group + 1), device="cuda:0", dtype=torch.float32)
        try:
            validate_shape(w, group)
        except ValueError as exc:
            policy_error = str(exc)
        else:
            raise AssertionError("invalid shape passed strict fixture policy")
        try:
            dual.tiled_quant_rectangle(w, [0, 15], group, method="sinq")
        except AssertionError as exc:
            oracle_error = str(exc)
        else:
            raise AssertionError("invalid shape unexpectedly accepted by pinned oracle")
        manifest["invalid_shapes"].append({"shape": list(w.shape), "group": group, "bits": [3, 4],
            "expected": "fail", "policy_error": policy_error, "oracle_error": oracle_error})
    # This shape is auto-adjusted by upstream, but is explicitly rejected here.
    try:
        validate_shape(torch.empty((3, 96)), 64)
    except ValueError:
        manifest["invalid_shapes"].append({"shape": [3, 96], "group": 64, "bits": [3, 4],
            "expected": "fail", "oracle_policy_difference": "upstream would silently change tile to 32; strict boundary rejects before oracle"})
    else:
        raise AssertionError("shape allowing implicit tile change was not rejected")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Generated and read-back verified {len(manifest['fixtures'])} fixtures ({2 * len(manifest['fixtures'])} binaries).")
    print(f"Invalid shape checks: {len(manifest['invalid_shapes'])}; real model gap: {model_gap or 'none'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=HERE.parents[1] / "artifacts" / "sinq-v050")
    parser.add_argument("--output", type=Path, default=HERE / "fixtures")
    parser.add_argument("--model-dir", type=Path, help="complete Qwen3.5-9B BF16 safetensors directory")
    parser.add_argument("--model-rows", type=int, default=7, help="rows retained in real-weight slices (default: 7)")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
