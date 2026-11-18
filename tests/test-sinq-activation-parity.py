#!/usr/bin/env python3
"""Offline, tiny actual Transformers/native Qwen3.5 projection-input parity.

CPU only. --generate-only checks fixture construction without a native build.
Requires local torch, transformers, numpy and the converter dependencies; no downloads.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))


def report(message):
    sys.stdout.write(message + '\n')
    sys.stdout.flush()


def fixture(directory, dtype_name, key_heads=2, value_heads=4, fp32_norms=False):
    import numpy as np
    import torch
    import gguf
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    from conversion.qwen import Qwen3_5TextModel as Converter

    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '32')))
    torch.manual_seed(41721)
    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=128, linear_num_key_heads=key_heads,
        linear_num_value_heads=value_heads, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4,
        max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5, "mrope_section": [11, 11, 10],
                         "mrope_interleaved": True},
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    model = Qwen3_5TextModel(config).to(device="cpu", dtype=torch.float32).eval()
    # Reference smoke rounds checkpoint weights to the selected storage dtype,
    # then executes F32 on CPU, isolating weight storage from HF CPU kernel support.
    dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    # Round parameters only. Module.to(dtype) also rounds nonpersistent RoPE
    # frequency buffers, which GGUF reconstructs from unrounded configuration.
    # Changing those buffers would compare different networks, not storage modes.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(parameter.to(dtype).float())
    if fp32_norms:
        # Characterize real mixed checkpoint storage: keep exact FP32 gated
        # norms that cannot be represented by BF16, as in the original 9B.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith('linear_attn.norm.weight'):
                    parameter.add_(torch.linspace(0.0001, 0.0003, parameter.numel()).reshape(parameter.shape))
    hparams = config.to_dict()
    hparams["architectures"] = ["Qwen3_5TextModel"]
    hparams["full_attention_interval"] = 2
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(hparams), encoding="utf-8")
    # Real converter initialization indexes the locally generated checkpoint.
    model.save_pretrained(directory, safe_serialization=True)
    converter = Converter(directory, gguf.LlamaFileType.ALL_F32,
                          directory / "model.gguf", eager=True, hparams=hparams)
    converter.set_gguf_parameters()
    writer = converter.gguf_writer
    # Token IDs are supplied directly; this deliberately tiny vocab needs no tokenizer.
    writer.add_tokenizer_model("llama")
    writer.add_token_list([f"token{i}" for i in range(32)])
    writer.add_token_scores([0.0] * 32)
    writer.add_token_types([1] * 32)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    # prepare_tensors applies the actual converter's transforms and precision policy.
    # Checkpoint is already storage-rounded; retain F32 GGUF to isolate activations.
    converter.prepare_tensors()
    writer.write_header_to_file(directory / "model.gguf")
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    suffix_roles = {
        "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
        "linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate",
        "linear_attn.in_proj_a": "ssm_alpha", "linear_attn.in_proj_b": "ssm_beta",
        "linear_attn.out_proj": "ssm_out", "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
    }
    captured = {}
    handles = []

    def hook(name, role):
        def pre_hook(module, args):
            x = args[0].detach().float().cpu().reshape(-1, module.in_features)
            if role == "ssm_out":
                # Independent grouped -> tiled input order matching native GDN.
                x = x.reshape(-1, key_heads, value_heads // key_heads, 128)
                grouped = x.reshape(-1, value_heads * 128)
                x = x.transpose(1, 2).contiguous().reshape(-1, value_heads * 128)
                if key_heads > 1 and torch.equal(grouped, x):
                    raise AssertionError("permutation-sensitive fixture did not exercise a channel reorder")
            if name in captured:
                raise AssertionError(f"duplicate HF site {name}")
            captured[name] = x.numpy().copy()
        return pre_hook

    for layer_index, layer in enumerate(model.layers):
        for suffix, role in suffix_roles.items():
            module = layer
            for part in suffix.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if isinstance(module, torch.nn.Linear):
                name = f"blk.{layer_index}.{role}.weight"
                handles.append(module.register_forward_pre_hook(hook(name, role)))
    tokens = [1, 7, 3, 11, 5, 19, 2]
    with torch.inference_mode():
        model(input_ids=torch.tensor([tokens], device="cpu"), use_cache=False)
    for handle in handles:
        handle.remove()
    if len(captured) != 15 or len({name.split(".")[2] for name in captured}) != 12:
        raise AssertionError(f"incomplete HF site coverage: {sorted(captured)}")
    for name, x in captured.items():
        if x.shape[0] != len(tokens) or not np.isfinite(x).all():
            raise AssertionError(f"invalid HF capture {name}: {x.shape}")
    np.savez(directory / "hf-inputs.npz", **captured)
    (directory / "tokens.json").write_text(json.dumps(tokens), encoding="utf-8")
    report(f"fixture {directory.name}: 15 sites, 12 roles, {len(tokens)} rows/site, "
           f"CPU F32 execution, {dtype_name}-rounded bulk weights, "
           f"exact FP32 norm fixture={fp32_norms}, K/V heads={key_heads}/{value_heads}")
    return captured, tokens


def compare(directory, expected):
    import numpy as np
    failures = []
    metrics = []
    for repeat in range(2):
        actual_dir = directory / "native" / str(repeat)
        found = {p.name.removesuffix(".shape") for p in actual_dir.glob("*.shape")}
        if found != set(expected):
            raise AssertionError(f"native coverage mismatch: {found ^ set(expected)}")
        for name, reference in sorted(expected.items()):
            shape = tuple(map(int, (actual_dir / (name + ".shape")).read_text().split()))
            actual = np.fromfile(actual_dir / (name + ".f32"), dtype="<f4").reshape(shape)
            if actual.shape != reference.shape:
                raise AssertionError(f"{name}: shape {actual.shape} != {reference.shape}")
            delta = actual.astype(np.float64) - reference
            maximum = float(np.max(np.abs(delta)))
            rms = float(np.sqrt(np.mean(delta ** 2)))
            ref_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
            relative = rms / max(ref_rms, np.finfo(np.float32).tiny)
            metrics.append(dict(site=name, repeat=repeat, shape=list(shape), max_abs=maximum,
                                rms=rms, relative_rms=relative, reference_rms=ref_rms))
            report(f"{directory.name} repeat={repeat} {name} shape={shape} "
                   f"maxabs={maximum:.9g} rms={rms:.9g} relative_rms={relative:.9g}")
            # Both paths are F32, including KV storage. Allow different reduction
            # orders and recurrent implementations: 2e-4 of signal RMS plus 2e-5
            # absolute for near-zero channels, well below storage-rounding errors.
            # Fail and retain metrics rather than widening bounds on a discrepancy.
            if not np.isfinite(actual).all() or maximum > 2e-5 + 2e-4 * ref_rms:
                failures.append(f"repeat={repeat} {name}: maxabs={maximum}, ref_rms={ref_rms}")
            if repeat:
                previous = np.fromfile(directory / "native" / "0" / (name + ".f32"), dtype="<f4").reshape(shape)
                if not np.array_equal(actual, previous):
                    failures.append(f"reset/repeat differs at {name}: {np.max(np.abs(actual - previous))}")
    (directory / 'metrics.json').write_text(json.dumps(dict(
        cpu_only=True, reference_execution='F32', weight_rounding=directory.name,
        max_abs_bound='2e-5 + 2e-4 * reference_rms', metrics=metrics, failures=failures), indent=2), encoding='utf-8')
    if failures:
        raise AssertionError("parity discrepancy (metrics preserved):\n" + "\n".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--storage-smoke", action="store_true", help="also test F16/BF16-rounded weights in CPU F32")
    args = parser.parse_args()
    if not args.generate_only and not args.runner:
        parser.error("--runner is required unless --generate-only")
    try:
        import torch
        import transformers
        import numpy
        if not hasattr(transformers, 'Qwen3_5TextModel'):
            raise ImportError('Qwen3.5 Transformers implementation unavailable')
    except ImportError as exc:
        report(f"SKIP: missing optional parity dependency: {exc}")
        return 77
    # Retain raw captures on failure as well as success for numerical diagnosis.
    output = (args.output or Path(tempfile.mkdtemp(prefix="sinq-parity-"))).resolve()
    report(f"torch={torch.__version__} transformers={transformers.__version__} numpy={numpy.__version__} output={output}")
    cases = [("f32", 2, 4)]
    if args.storage_smoke:
        cases.extend([("f16", 2, 4), ("bf16", 2, 4)])
    for dtype, kh, vh in cases:
        directory = output / f"{dtype}-k{kh}-v{vh}"
        expected, tokens = fixture(directory, dtype, kh, vh)
        if not args.generate_only:
            subprocess.run([str(args.runner.resolve()), str(directory / "model.gguf"),
                            str(directory / "native"), *map(str, tokens)], check=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
            compare(directory, expected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
