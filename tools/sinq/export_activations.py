#!/usr/bin/env python3
"""Local, reference-weight-verified Qwen3.5 linear pre-input exporter."""
import argparse
import hashlib
import inspect
import json
import os
import re
from functools import partial
from pathlib import Path

ROLES = {
    'ffn_gate': 'mlp.gate_proj', 'ffn_up': 'mlp.up_proj', 'ffn_down': 'mlp.down_proj',
    'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj',
    'attn_v': 'self_attn.v_proj', 'attn_output': 'self_attn.o_proj',
    'attn_qkv': 'linear_attn.in_proj_qkv', 'attn_gate': 'linear_attn.in_proj_z',
    'ssm_alpha': 'linear_attn.in_proj_a', 'ssm_beta': 'linear_attn.in_proj_b',
    'ssm_out': 'linear_attn.out_proj',
}
PATTERN = re.compile(r'blk\.(\d+)\.(' + '|'.join(ROLES) + r')\.weight\Z')
MAX_FILE = 256 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bounds(seed, rows):
    if not 0 <= seed <= 0xffffffff or not 1 <= rows <= 256:
        raise ValueError('seed must be uint32 and rows must be 1..256')


def module_for(name):
    m = PATTERN.fullmatch(name)
    if not m:
        raise ValueError('unsupported selected tensor: ' + name)
    layer, role = m.groups()
    return f'model.language_model.layers.{layer}.{ROLES[role]}', f'qwen3_5.layers.{layer}.{role}.input', role


def reorder_v(array, axis, key_heads, value_heads, head_dim):
    import numpy as np
    if key_heads <= 0 or value_heads % key_heads or head_dim <= 0:
        raise ValueError('invalid linear-attention head configuration')
    if array.shape[axis] != value_heads * head_dim:
        raise ValueError('linear-attention permutation dimension mismatch')
    idx = np.arange(value_heads * head_dim).reshape(key_heads, value_heads // key_heads, head_dim)
    idx = idx.swapaxes(0, 1).reshape(-1)
    return np.take(array, idx, axis=axis)


def converter_weight(weight, role, config):
    import numpy as np
    if role not in ('attn_qkv', 'attn_gate', 'ssm_alpha', 'ssm_beta', 'ssm_out'):
        return weight
    kh, vh = config.linear_num_key_heads, config.linear_num_value_heads
    vd, kd = config.linear_value_head_dim, config.linear_key_head_dim
    if role == 'attn_qkv':
        qk = 2 * kh * kd
        return np.concatenate((weight[:qk], reorder_v(weight[qk:], 0, kh, vh, vd)))
    if role == 'ssm_out':
        return reorder_v(weight, 1, kh, vh, vd)
    return reorder_v(weight, 0, kh, vh, 1 if role in ('ssm_alpha', 'ssm_beta') else vd)


def floating_tensor(tensor):
    import numpy as np
    k, n = map(int, tensor.shape)
    if tensor.n_elements != k * n:
        raise ValueError('GGUF element count mismatch')
    kind = tensor.tensor_type.name
    raw = np.asarray(tensor.data)
    if kind == 'BF16':
        # GGUFReader exposes BF16 as packed bytes, not numerical uint8 values.
        words = raw.reshape(-1).view('<u2')
        if words.size != k * n:
            raise ValueError('packed BF16 size mismatch')
        values = (words.astype(np.uint32) << 16).view(np.float32)
    elif kind in ('F16', 'F32'):
        values = raw.reshape(-1).astype(np.float32)
    else:
        raise ValueError('source must be original floating GGUF')
    if values.size != k * n or not np.isfinite(values).all():
        raise ValueError('invalid GGUF weights')
    return values.reshape(n, k)


class ReservoirCapture:
    """Bounded disk-backed Algorithm R samples of actual nn.Linear pre-inputs."""
    def __init__(self, module, name, path, max_rows=128, seed=0, transform=None):
        import numpy as np
        import torch
        bounds(seed, max_rows)
        if type(module) is not torch.nn.Linear:
            raise ValueError('capture site must be an unfused torch.nn.Linear')
        self.k, self.n = module.in_features, module.out_features
        if max_rows * self.k * 4 > MAX_FILE:
            raise ValueError('capture exceeds 256 MiB per tensor')
        self.path = Path(path)
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(self.path)
        self.capacity, self.seen, self.count = max_rows, 0, 0
        digest = hashlib.sha256(seed.to_bytes(4, 'little') + name.encode('utf-8')).digest()
        self.rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], 'little')))
        with self.path.open('xb') as f:
            f.truncate(max_rows * self.k * 4)
        self.data = np.memmap(self.path, mode='r+', dtype='<f4', shape=(max_rows, self.k))
        self.transform = transform
        self.handle = module.register_forward_pre_hook(self.hook)

    def hook(self, module, inputs):
        import numpy as np
        import torch
        if len(inputs) != 1 or inputs[0].shape[-1] != self.k:
            raise ValueError('unsupported linear input signature')
        x = inputs[0].detach().reshape(-1, self.k)
        if not torch.isfinite(x).all().item():
            raise ValueError('nonfinite activation')
        for row in x:
            self.seen += 1
            slot = self.seen - 1 if self.seen <= self.capacity else int(self.rng.integers(self.seen))
            if slot < self.capacity:
                value = row.to(device='cpu', dtype=torch.float32).numpy()
                if self.transform is not None:
                    value = self.transform(value)
                if value.shape != (self.k,) or not np.isfinite(value).all():
                    raise ValueError('invalid transformed activation')
                self.data[slot] = value
                self.count = min(self.seen, self.capacity)

    def finish(self):
        self.handle.remove()
        self.data.flush()
        self.data._mmap.close()
        if not self.count:
            raise ValueError('selected module did not execute')
        with self.path.open('r+b') as f:
            f.truncate(self.count * self.k * 4)
        return self.count


def export(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from gguf import GGUFReader
    import transformers
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration
    bounds(args.seed, args.rows)
    if not 1 <= args.context <= 4096 or not 1 <= args.max_tokens <= 1048576:
        raise ValueError('context must be 1..4096 and max-tokens 1..1048576')
    output_arg = Path(args.output)
    if output_arg.exists() or output_arg.is_symlink():
        raise FileExistsError(output_arg)
    checkpoint, source, output = Path(args.checkpoint).resolve(), Path(args.gguf).resolve(), output_arg.resolve()
    if not checkpoint.is_dir() or not source.is_file():
        raise ValueError('checkpoint and original GGUF must exist locally')
    if output.exists():
        raise FileExistsError(output)
    config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    if config.model_type != 'qwen3_5' or getattr(config, 'quantization_config', None):
        raise ValueError('only original unquantized Qwen3.5 conditional checkpoints supported')
    text_config = config.text_config
    if getattr(text_config, 'quantization_config', None):
        raise ValueError('prequantized text model unsupported')
    reader = GGUFReader(str(source))
    if any(k.startswith('beellama.qk_sinq.') or k.startswith('beellama.sinq.') or 'hadamard' in k.lower() for k in reader.fields):
        raise ValueError('transformed GGUF unsupported')
    if any(t.tensor_type.name not in ('F16', 'F32', 'BF16') for t in reader.tensors):
        raise ValueError('prequantized GGUF unsupported')
    arch = reader.fields.get('general.architecture')
    if arch is None or arch.contents() != 'qwen35':
        raise ValueError('source GGUF architecture must be qwen35')
    tensors = {t.name: t for t in reader.tensors}
    names = [s.strip() for s in Path(args.tensor_list).read_text(encoding='utf-8').splitlines() if s.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError('tensor list must be nonempty and unique')
    for name in names:
        module_for(name)
        if name not in tensors or len(tensors[name].shape) != 2:
            raise ValueError('missing selected original tensor: ' + name)
    corpus = Path(args.text or args.tokens)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    if args.tokens:
        tokens = json.loads(corpus.read_text(encoding='utf-8'))
    else:
        tokens = tokenizer(corpus.read_text(encoding='utf-8'), add_special_tokens=False)['input_ids']
    if not isinstance(tokens, list) or not tokens or len(tokens) > args.max_tokens:
        raise ValueError('tokens must be a nonempty list within max-tokens (no implicit truncation)')
    padding_ids = {getattr(obj, 'pad_token_id', None) for obj in (config, text_config, tokenizer)} - {None}
    if any(type(t) is not int or not 0 <= t < text_config.vocab_size or t in padding_ids for t in tokens):
        raise ValueError('invalid or padding token')
    token_hash = hashlib.sha256(np.asarray(tokens, dtype='<u4').tobytes()).hexdigest()
    checkpoint_files = sorted(p for p in checkpoint.rglob('*') if p.is_file() and p.suffix in ('.json', '.safetensors'))
    if not any(p.suffix == '.safetensors' for p in checkpoint_files):
        raise ValueError('local safetensors checkpoint required')
    identity = {p.relative_to(checkpoint).as_posix(): sha256(p) for p in checkpoint_files}
    revision = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False, torch_dtype=dtype).to(args.device).eval()
    modules = dict(model.named_modules())
    graph_identity = {name: type(module).__module__ + '.' + type(module).__qualname__ for name, module in modules.items()}
    graph_hash = hashlib.sha256(json.dumps(graph_identity, sort_keys=True).encode()).hexdigest()
    # Validate every selected module before creating output files.
    for name in names:
        path, _, role = module_for(name)
        module = modules.get(path)
        if type(module) is not torch.nn.Linear:
            raise ValueError('missing or fused reference module: ' + path)
        weight = module.weight.detach().to(device='cpu', dtype=torch.float32).numpy()
        actual = converter_weight(weight, role, text_config)
        expected = floating_tensor(tensors[name])
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError('checkpoint/GGUF weight mismatch: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    captures, records = [], []
    try:
        for index, name in enumerate(names):
            path, site, role = module_for(name)
            transform = None
            if role == 'ssm_out':
                transform = partial(reorder_v, axis=0, key_heads=text_config.linear_num_key_heads,
                                    value_heads=text_config.linear_num_value_heads, head_dim=text_config.linear_value_head_dim)
            filename = f'{index:04d}.f32'
            capture = ReservoirCapture(modules[path], name, output / filename, args.rows, args.seed, transform)
            captures.append(capture)
            records.append(dict(name=name, site=site, reference_module=path, channel_layout='gguf-input', input_dim=capture.k, output_dim=capture.n, dtype='<f4', order='row-major', data_file=filename))
        with torch.inference_mode():
            for start in range(0, len(tokens), args.context):
                ids = torch.tensor([tokens[start:start + args.context]], device=args.device, dtype=torch.long)
                model.model.language_model(input_ids=ids, use_cache=False, return_dict=True)
        for capture, record in zip(captures, records):
            record['count'] = capture.finish()
            if capture.seen != len(tokens):
                raise ValueError('capture count differs from valid prefill tokens')
            record['sha256'] = sha256(capture.path)
        manifest = dict(schema_version=1, architecture='qwen3_5', source_sha256=sha256(source), source_revision=revision,
                        sampling=dict(contract='valid-token-prefill-reservoir-v1', seed=args.seed, max_rows=args.rows, corpus_sha256=sha256(corpus), token_sha256=token_hash), tensors=records,
                        provenance=dict(graph_sha256=graph_hash, transformers_version=getattr(transformers, '__version__', 'synthetic-fixture'), reference_implementation_sha256=sha256(inspect.getfile(type(model.model.language_model))), checkpoint_files=identity, config_sha256=hashlib.sha256(config.to_json_string().encode()).hexdigest(), model_class=type(model).__name__, module_classes={r['reference_module']: type(modules[r['reference_module']]).__name__ for r in records}, torch_version=torch.__version__, dtype=args.dtype, device=args.device, context=args.context, token_count=len(tokens), preprocessing='explicit token JSON or local tokenizer; no padding, no cache, fresh prefill chunks', rng='PCG64: SHA256(uint32-LE seed || UTF8 exact tensor name), first 128 bits little-endian; Algorithm R'))
        with (output / 'activations.json').open('x', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, allow_nan=False)
            f.write('\n')
    finally:
        for capture in captures:
            capture.handle.remove()
            if not capture.data._mmap.closed:
                capture.data.flush()
                capture.data._mmap.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--gguf', required=True)
    p.add_argument('--tensor-list', required=True)
    corpus = p.add_mutually_exclusive_group(required=True)
    corpus.add_argument('--text')
    corpus.add_argument('--tokens', help='JSON array of integer token IDs')
    p.add_argument('--output', required=True, help='new directory, never overwritten')
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=('bf16', 'f16'), default='bf16')
    p.add_argument('--rows', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--max-tokens', type=int, default=4096)
    export(p.parse_args())


if __name__ == '__main__':
    main()
