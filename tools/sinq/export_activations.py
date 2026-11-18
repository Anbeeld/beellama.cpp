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


def floating_tensor(tensor, rows=None):
    import numpy as np
    shape = tuple(map(int, tensor.shape))[::-1]
    count = int(np.prod(shape))
    if tensor.n_elements != count:
        raise ValueError('GGUF element count mismatch')
    kind = tensor.tensor_type.name
    raw = np.asarray(tensor.data)
    if kind == 'BF16':
        # Slice packed words before converting, including huge embeddings.
        raw = raw.reshape(-1).view('<u2')
    elif kind not in ('F16', 'F32'):
        raise ValueError('source must be original floating GGUF')
    if raw.size != count:
        raise ValueError('packed floating tensor size mismatch')
    raw = raw.reshape(shape)
    if rows is not None:
        raw = raw[rows]
    values = (raw.astype(np.uint32) << 16).view(np.float32) if kind == 'BF16' else raw.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('invalid GGUF weights')
    return values


def verify_reference_config(reader, config):
    """Check graph-defining metadata as well as parameter identity."""
    import numpy as np
    verified = []

    def check(suffix, expected, default=None):
        key = 'qwen35.' + suffix
        field = reader.fields.get(key)
        if field is not None or default is not None:
            actual = np.asarray(field.contents() if field is not None else default)
            target = np.float32(expected) if actual.dtype.kind == 'f' else expected
            if not np.all(actual == target):
                raise ValueError('reference configuration mismatch: ' + key)
            verified.append(key)

    for suffix, attr in (
        ('block_count', 'num_hidden_layers'), ('embedding_length', 'hidden_size'),
        ('context_length', 'max_position_embeddings'),
        ('feed_forward_length', 'intermediate_size'), ('attention.head_count', 'num_attention_heads'),
        ('attention.head_count_kv', 'num_key_value_heads'), ('attention.key_length', 'head_dim'),
        ('attention.value_length', 'head_dim'), ('attention.layer_norm_rms_epsilon', 'rms_norm_eps'),
        ('ssm.conv_kernel', 'linear_conv_kernel_dim'), ('ssm.state_size', 'linear_key_head_dim'),
        ('ssm.group_count', 'linear_num_key_heads'), ('ssm.time_step_rank', 'linear_num_value_heads'),
    ):
        if 'qwen35.' + suffix in reader.fields:
            check(suffix, getattr(config, attr))
    if hasattr(config, 'linear_value_head_dim'):
        if config.linear_value_head_dim != config.linear_key_head_dim:
            raise ValueError('reference unequal SSM key/value dimensions unsupported')
        check('ssm.inner_size', config.linear_value_head_dim * config.linear_num_value_heads)
    if getattr(config, 'hidden_act', 'silu') != 'silu':
        raise ValueError('reference non-SiLU activation unsupported')
    rope = getattr(config, 'rope_parameters', None)
    if rope:
        if rope.get('rope_type', 'default') != 'default' or not rope.get('mrope_interleaved', True):
            raise ValueError('reference nondefault RoPE configuration unsupported')
        check('rope.freq_base', rope.get('rope_theta', 10000.0), default=10000.0)
        native_head = reader.fields.get('qwen35.attention.key_length')
        native_head = int(native_head.contents()) if native_head else config.hidden_size // config.num_attention_heads
        check('attention.key_length', config.head_dim, default=native_head)
        check('attention.value_length', config.head_dim, default=config.hidden_size // config.num_attention_heads)
        check('rope.dimension_count', int(config.head_dim * rope.get('partial_rotary_factor', 0.25)), default=native_head)
        scaling = reader.fields.get('qwen35.rope.scaling.type')
        if scaling is not None and scaling.contents() not in ('none', 'linear'):
            raise ValueError('reference nondefault source RoPE scaling type unsupported')
        order = reader.fields.get('qwen35.rope.section_order')
        if order is not None and order.contents() != 'tyxz':
            raise ValueError('reference source MRoPE section order unsupported')
        for suffix in ('rope.scaling.factor', 'rope.scale_linear'):
            field = reader.fields.get('qwen35.' + suffix)
            if field is not None and float(field.contents()) not in (0.0, 1.0):
                raise ValueError('reference nondefault source RoPE scaling unsupported')
        for suffix in ('rope.scaling.attn_factor', 'rope.scaling.alpha'):
            if 'qwen35.' + suffix in reader.fields:
                raise ValueError('reference custom source RoPE parameters unsupported')
        sections = list(rope.get('mrope_section', [11, 11, 10]))
        sections += [0] * (4 - len(sections))
        check('rope.dimension_sections', sections)
    layer_types = getattr(config, 'layer_types', None)
    if layer_types:
        if any(kind not in ('linear_attention', 'full_attention') for kind in layer_types):
            raise ValueError('reference layer types unsupported')
        expected = [kind == 'linear_attention' for kind in layer_types]
        field = reader.fields.get('qwen35.attention.recurrent_layers')
        if field is not None:
            actual = list(field.contents())
        else:
            interval = reader.fields.get('qwen35.full_attention_interval')
            interval = int(interval.contents()) if interval else 4
            if interval <= 0:
                raise ValueError('invalid source full attention interval')
            actual = [(i + 1) % interval != 0 for i in range(config.num_hidden_layers)]
        if actual != expected:
            raise ValueError('reference configuration mismatch: recurrent layers')
        verified.append('qwen35.attention.recurrent_layers')
    return verified


def verify_reference_identity(model, config, tensors, workspace_limit=64 * 1024 * 1024,
                              require_exact_execution_storage=False, resource_check=None):
    """Check the complete text network with bounded row buffers, before capture."""
    import numpy as np
    import torch
    from gguf import MODEL_ARCH, get_tensor_name_map
    mapper = get_tensor_name_map(MODEL_ARCH.QWEN35, config.num_hidden_layers)
    verified, aliases = {}, []
    parameters = model.state_dict()
    embedding = parameters.get('model.language_model.embed_tokens.weight')
    for name, parameter in parameters.items():
        if not (name.startswith('model.language_model.') or name == 'lm_head.weight'):
            continue  # Vision is not executed by the text calibration forward.
        if not parameter.is_floating_point():
            raise ValueError('nonfloating reference parameter: ' + name)
        canonical = name.replace('language_model.', '')
        if name.endswith('.dt_bias'):
            canonical = canonical.rpartition('.dt_bias')[0] + '.dt_proj.bias'
        mapped = mapper.get_name(canonical, try_suffixes=('.weight', '.bias'))
        if mapped is None:
            raise ValueError('unmapped reference text parameter: ' + name)
        if mapped == 'output.weight' and mapped not in tensors and getattr(config, 'tie_word_embeddings', False):
            if embedding is None or parameter.shape != embedding.shape:
                raise ValueError('reference output is not tied to token embeddings')
            width = max(1, workspace_limit // max(1, parameter.shape[-1] * 16))
            for first in range(0, parameter.shape[0], width):
                if not torch.equal(parameter[first:first + width].detach().cpu(),
                                   embedding[first:first + width].detach().cpu()):
                    raise ValueError('reference output is not tied to token embeddings')
            aliases.append(name)
            continue
        if mapped not in tensors:
            raise ValueError('missing source tensor: ' + mapped)
        kind = tensors[mapped].tensor_type.name
        if kind not in ('F16', 'BF16', 'F32'):
            raise ValueError('nonfloating source tensor: ' + mapped)
        storage = {'F16': torch.float16, 'BF16': torch.bfloat16, 'F32': torch.float32}[kind]
        value = parameter.detach()  # Export verifies on CPU before moving the model.
        matrix = value.ndim == 2 and 'conv1d' not in name
        reference_shape = tuple(value.squeeze().shape) if 'conv1d' in name else tuple(value.shape)
        if reference_shape != tuple(map(int, tensors[mapped].shape))[::-1]:
            raise ValueError('checkpoint/GGUF shape mismatch: ' + mapped)
        row_order = column_order = None
        reorder = ('linear_attn.' in name
                   and getattr(config, 'linear_num_key_heads', 0) != getattr(config, 'linear_num_value_heads', 0))
        if reorder and matrix:
            kh, vh = config.linear_num_key_heads, config.linear_num_value_heads
            role = PATTERN.fullmatch(mapped)[2]
            if role == 'ssm_out':
                column_order = reorder_v(np.arange(value.shape[1]), 0, kh, vh, config.linear_value_head_dim)
            elif role == 'attn_qkv':
                qk = 2 * kh * config.linear_key_head_dim
                row_order = np.concatenate((np.arange(qk), qk + reorder_v(np.arange(value.shape[0] - qk), 0, kh, vh, config.linear_value_head_dim)))
            else:
                dim = 1 if role in ('ssm_alpha', 'ssm_beta') else config.linear_value_head_dim
                row_order = reorder_v(np.arange(value.shape[0]), 0, kh, vh, dim)
        columns = value.shape[1] if matrix else value.numel()
        # Conservative allowance for original/cast/indexed and decoded row arrays.
        row_bytes = int(columns) * 32
        if workspace_limit < row_bytes:
            raise ValueError('reference identity workspace cannot fit one row: ' + mapped)
        rows = value.shape[0] if matrix else 1
        slab = min(rows, workspace_limit // row_bytes)
        digest = hashlib.sha256()
        for first in range(0, rows, slab):
            if resource_check is not None: resource_check()
            stop = min(rows, first + slab)
            if matrix:
                block = value[first:stop] if row_order is None else value[torch.as_tensor(row_order[first:stop], device=value.device)]
                if column_order is not None:
                    block = block[:, torch.as_tensor(column_order, device=value.device)]
                expected = floating_tensor(tensors[mapped], slice(first, stop))
            else:
                block = value
                expected = floating_tensor(tensors[mapped])
            block = block.cpu()
            if not torch.isfinite(block).all().item():
                raise ValueError('nonfinite reference parameter: ' + name)
            # Mirror prepare_tensors promotion before converter arithmetic.
            if block.dtype not in (torch.float16, torch.float32):
                block = block.float()
            if not matrix:
                if name.endswith('.A_log'):
                    block = -torch.exp(block)
                elif 'conv1d' in name:
                    block = block.squeeze()
                elif name.endswith('norm.weight') and not name.endswith('linear_attn.norm.weight'):
                    block = block + 1
                actual = block.float().numpy()
                if reorder:
                    kh, vh = config.linear_num_key_heads, config.linear_num_value_heads
                    if name.endswith(('.A_log', '.dt_bias')):
                        actual = reorder_v(actual, 0, kh, vh, 1)
                    elif 'conv1d' in name:
                        qk = 2 * kh * config.linear_key_head_dim
                        actual = np.concatenate((actual[:qk], reorder_v(actual[qk:], 0, kh, vh, config.linear_value_head_dim)))
                block = torch.from_numpy(np.ascontiguousarray(actual))
            if require_exact_execution_storage and not torch.equal(block.float(), block.to(storage).float()):
                raise ValueError('reference execution storage mismatch: ' + mapped + ' (' + name + ')')
            actual = block.to(storage).float().numpy()
            if actual.shape != expected.shape or not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)):
                raise ValueError('checkpoint/GGUF weight mismatch: ' + mapped + ' (' + name + ')')
            digest.update(actual.astype('<f4', copy=False).tobytes())
        if mapped in verified:
            raise ValueError('duplicate reference mapping: ' + mapped)
        verified[mapped] = digest.hexdigest()
    extra = set(tensors) - set(verified)
    if extra:
        raise ValueError('unverified source tensors: ' + ', '.join(sorted(extra)))
    if not verified:
        raise ValueError('no reference text weights verified')
    digest = hashlib.sha256(json.dumps(verified, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(contract='qwen35-full-text-storage-v1', verified_tensor_count=len(verified),
                verified_tensors_sha256=digest, tied_output_aliases=aliases, workspace_limit_bytes=workspace_limit,
                precision_policy=('converter transforms require exact execution storage before comparison' if require_exact_execution_storage
                                  else 'converter transforms followed by exact GGUF storage-rounded float comparison'))


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
    threads=getattr(args, 'threads', None)
    threads=(os.cpu_count() or 1) if threads is None else threads
    if type(threads) is not int or not 1<=threads<=1024:
        raise ValueError('threads must be an integer in 1..1024')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    torch.set_num_threads(threads)
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
    source_digest = sha256(source)
    reader = GGUFReader(str(source))
    if any(k.startswith('beellama.qk_sinq.') or k.startswith('beellama.sinq.') or 'hadamard' in k.lower() for k in reader.fields):
        raise ValueError('transformed GGUF unsupported')
    if any(t.tensor_type.name not in ('F16', 'F32', 'BF16') for t in reader.tensors):
        raise ValueError('prequantized GGUF unsupported')
    arch = reader.fields.get('general.architecture')
    if arch is None or arch.contents() != 'qwen35':
        raise ValueError('source GGUF architecture must be qwen35')
    tensors = {t.name: t for t in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise ValueError('duplicate source tensor names')
    names = [s.strip() for s in Path(args.tensor_list).read_text(encoding='utf-8').splitlines() if s.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError('tensor list must be nonempty and unique')
    for name in names:
        module_for(name)
        if name not in tensors or len(tensors[name].shape) != 2:
            raise ValueError('missing selected original tensor: ' + name)
    corpus = Path(args.text or args.tokens)
    corpus_digest = sha256(corpus)
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
    if str(args.device).startswith('cuda') and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False, torch_dtype=dtype).eval()
    verified_config = verify_reference_config(reader, text_config)
    reference_identity = verify_reference_identity(model, text_config, tensors)
    reference_identity['verified_config_fields'] = verified_config
    model.to(args.device)
    modules = dict(model.named_modules())
    graph_identity = {name: type(module).__module__ + '.' + type(module).__qualname__ for name, module in modules.items()}
    graph_hash = hashlib.sha256(json.dumps(graph_identity, sort_keys=True).encode()).hexdigest()
    # Complete streamed identity is already verified; only capture-site
    # availability/type remains to check, without expanding weights again.
    for name in names:
        path, _, _ = module_for(name)
        if type(modules.get(path)) is not torch.nn.Linear:
            raise ValueError('missing or fused reference module: ' + path)
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
        if sha256(source) != source_digest or sha256(corpus) != corpus_digest:
            raise ValueError('source GGUF or calibration corpus changed during export')
        if any(sha256(path) != identity[path.relative_to(checkpoint).as_posix()] for path in checkpoint_files):
            raise ValueError('reference checkpoint files changed during export')
        manifest = dict(schema_version=1, architecture='qwen3_5', source_sha256=source_digest, source_revision=revision,
                        sampling=dict(contract='valid-token-prefill-reservoir-v1', seed=args.seed, max_rows=args.rows, corpus_sha256=corpus_digest, token_sha256=token_hash), tensors=records,
                        provenance=dict(reference_identity=reference_identity, graph_sha256=graph_hash, transformers_version=getattr(transformers, '__version__', 'synthetic-fixture'), reference_implementation_sha256=sha256(inspect.getfile(type(model.model.language_model))), checkpoint_files=identity, config_sha256=hashlib.sha256(config.to_json_string().encode()).hexdigest(), model_class=type(model).__name__, module_classes={r['reference_module']: type(modules[r['reference_module']]).__name__ for r in records}, torch_version=torch.__version__, cpu_threads=torch.get_num_threads(), dtype=args.dtype, device=args.device, context=args.context, token_count=len(tokens), preprocessing='explicit token JSON or local tokenizer; no padding, no cache, fresh prefill chunks', rng='PCG64: SHA256(uint32-LE seed || UTF8 exact tensor name), first 128 bits little-endian; Algorithm R'))
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
    p.add_argument('--threads', type=int, default=os.cpu_count() or 1, help='CPU threads (default: all logical CPUs)')
    export(p.parse_args())


if __name__ == '__main__':
    main()
