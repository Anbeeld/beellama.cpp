"""Text-only, exact-storage Qwen3.5 safetensors loading.

Only the destination graph is resident: parameters are created on meta and filled
one checkpoint tensor at a time. Source/GGUF identity and execution placement are
owned by Qwen35Adapter, which receives the complete CPU graph before placement.
"""
import json
import math
from pathlib import Path


def _checkpoint_inventory(checkpoint):
    from safetensors import safe_open

    index_path = checkpoint / 'model.safetensors.index.json'
    index = None
    if index_path.exists():
        with index_path.open(encoding='utf-8') as stream:
            index = json.load(stream).get('weight_map')
        if not isinstance(index, dict) or not index:
            raise ValueError('invalid checkpoint shard index weight_map')
        files = set()
        for filename in index.values():
            if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith('.safetensors'):
                raise ValueError('invalid checkpoint shard index filename')
            path = checkpoint / filename
            if not path.is_file():
                raise ValueError('checkpoint index shard is missing: ' + filename)
            files.add(path)
        # Extra unindexed files must not silently shadow indexed weights.
        if files != set(checkpoint.glob('*.safetensors')):
            raise ValueError('checkpoint shard index does not cover safetensors files')
    else:
        files = set(checkpoint.glob('*.safetensors'))
    if not files:
        raise ValueError('local safetensors checkpoint required')
    inventory = {}
    for path in sorted(files):
        with safe_open(path, framework='pt', device='cpu') as weights:
            for name in weights.keys():
                if name in inventory:
                    raise ValueError('duplicate checkpoint tensor: ' + name)
                if index is not None and index.get(name) != path.name:
                    raise ValueError('checkpoint index tensor/shard mismatch: ' + name)
                view = weights.get_slice(name)
                inventory[name] = (path, tuple(view.get_shape()), view.get_dtype())
    if index is not None and set(index) != set(inventory):
        raise ValueError('checkpoint shard index has missing tensor keys')
    return inventory


def _exact_copy(source, dtype, name, workspace):
    """Reject precision loss before publishing any parameter; compare in slabs."""
    import torch

    if not source.is_floating_point():
        raise ValueError('nonfloating checkpoint parameter: ' + name)
    result = torch.empty(source.shape, dtype=dtype, device='cpu')
    original = source.reshape(-1)
    destination = result.reshape(-1)
    # Converted slab, round-trip slab and comparison mask are transient, bounded
    # independently from the single destination parameter allocation.
    per_element = source.element_size() + result.element_size() + 4
    chunk = max(1, workspace // per_element)
    for start in range(0, source.numel(), chunk):
        values = original[start:start + chunk]
        converted = values.to(dtype)
        if not torch.isfinite(values).all() or not torch.equal(values, converted.to(source.dtype)):
            raise ValueError('checkpoint tensor is not exactly representable in ' + str(dtype) + ': ' + name)
        destination[start:start + chunk].copy_(converted)
    return result


def _assign_parameter(model, name, value):
    import torch

    parent_name, _, leaf = name.rpartition('.')
    parent = model.get_submodule(parent_name) if parent_name else model
    parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)


def load_reference(checkpoint, source, budget, threads, *, policy=None, monitor=None):
    """Load a local unquantized text reference, verify it, then place via policy.

    ``budget`` bounds operation workspace, separately from policy host residency.
    The orchestration layer retains its runtime resource guard around this call.
    No approximate-reference mode or implicit precision/device fallback exists.
    """
    import torch
    from safetensors import safe_open
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    from .qwen35_adapter import Qwen35Adapter

    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError('reference host memory budget must be a positive integer')
    if budget < 16:
        raise ValueError('reference workspace memory budget cannot fit one comparison element')
    if not isinstance(threads, int) or isinstance(threads, bool) or threads <= 0:
        raise ValueError('reference CPU threads must be a positive integer')
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise ValueError('explicit local checkpoint directory required')
    if policy is None:
        from .execution_backend import resolve_policy
        policy = resolve_policy({'device': 'cpu', 'reference_dtype': 'fp32'})
    if policy.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('reference execution dtype must be FP32 or BF16')
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    with (checkpoint / 'config.json').open(encoding='utf-8') as stream:
        raw_config = json.load(stream)
    text_config = raw_config.get('text_config', raw_config)
    if any(config.get('quantization_config') is not None for config in (raw_config, text_config)):
        raise ValueError('quantized reference checkpoint configuration is unsupported')
    if text_config.get('model_type', 'qwen3_5_text') != 'qwen3_5_text':
        raise ValueError('reference checkpoint must have Qwen3.5 text configuration')
    config = Qwen3_5TextConfig.from_dict(text_config)
    config._attn_implementation = 'eager'
    inventory = _checkpoint_inventory(checkpoint)
    with torch.device('meta'):
        text = Qwen3_5TextModel(config)
        head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=policy.dtype)
    expected = dict(text.named_parameters())
    conditional = 'text_config' in raw_config
    prefix = 'model.language_model.' if conditional else ''
    # Also accept the standard plain-text causal-LM wrapper's model.* namespace.
    if not conditional and 'model.embed_tokens.weight' in inventory:
        prefix = 'model.'
    names = {prefix + name: name for name in expected}
    required = set(names)
    if not config.tie_word_embeddings:
        required.add('lm_head.weight')
    relevant = set()
    for name in inventory:
        # Conditional checkpoints may carry inference-unused vision and MTP
        # modules. The text-only graph never executes these namespaces; GGUF
        # verification still rejects any unexpected source text tensor.
        if conditional and name.startswith(('model.visual.', 'mtp.')):
            continue
        if name not in names and name != 'lm_head.weight':
            raise ValueError('unexpected reference checkpoint tensor: ' + name)
        relevant.add(name)
    missing = required - relevant
    if missing:
        raise ValueError('missing reference checkpoint tensors: ' + ', '.join(sorted(missing)))
    for name in relevant:
        shape = inventory[name][1]
        target_shape = (config.vocab_size, config.hidden_size) if name == 'lm_head.weight' else tuple(expected[names[name]].shape)
        if shape != target_shape:
            raise ValueError('checkpoint tensor shape mismatch: ' + name)

    # Reconstruct only nonpersistent RoPE state from config, in stock FP32. Never
    # .to(dtype) the model: that would incorrectly round its rotary frequencies.
    text.rotary_emb = type(text.rotary_emb)(config, device='cpu')
    if any(buffer.device.type == 'meta' for buffer in text.buffers()):
        raise ValueError('unsupported meta reference buffer outside rotary embedding')
    from .execution_backend import parameter_storage_dtype
    checkpoint_dtypes = {'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16}
    target_dtypes = {}
    for checkpoint_name, name in names.items():
        module = text.get_submodule(name.rpartition('.')[0])
        target_dtypes[checkpoint_name] = parameter_storage_dtype(
            module, policy.dtype, checkpoint_dtypes.get(inventory[checkpoint_name][2]))
    target_dtypes['lm_head.weight'] = policy.dtype
    element_size = torch.empty((), dtype=policy.dtype).element_size()
    storage = sum(expected[name].numel() * torch.empty((), dtype=target_dtypes[key]).element_size()
                  for key, name in names.items())
    if not config.tie_word_embeddings:
        storage += config.vocab_size * config.hidden_size * element_size
    buffers = sum(buffer.numel() * buffer.element_size() for buffer in text.buffers())
    host_cap = getattr(policy, 'memory_policy', {}).get('host_limit_bytes', budget)
    if storage + buffers > host_cap:
        raise ValueError('reference storage alone exceeds host memory budget: ' + str(storage + buffers))
    cuda_capacity = 0
    for device in policy.devices:
        if device.type == 'cuda':
            free, total = torch.cuda.mem_get_info(device)
            cap = policy.memory_policy.get('cuda_limit_bytes', {}).get(str(device), total)
            cuda_capacity += max(0, min(free, cap) - policy.memory_policy['reserve_bytes'])
    if policy.devices[0].type == 'cuda' and storage + buffers > cuda_capacity:
        raise ValueError('INSUFFICIENT_CAPACITY: reference storage exceeds measured selected CUDA capacity: '
                         + str(storage + buffers) + ' > ' + str(cuda_capacity))
    dtype_bytes = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2}
    unsupported = [name for name in relevant if inventory[name][2] not in dtype_bytes]
    if unsupported:
        raise ValueError('unsupported checkpoint tensor storage: ' + ', '.join(sorted(unsupported)))
    largest_source = max(math.prod(inventory[name][1]) * dtype_bytes[inventory[name][2]] for name in relevant)
    largest_parameter = max(math.prod(inventory[name][1]) * torch.empty((),dtype=target_dtypes[name]).element_size()
                            for name in required)
    slab = min(1024 * 1024, budget)
    estimates = {
        'kind': 'conservative_estimates_not_measured_residency',
        'storage_weight_bytes': storage,
        'persistent_and_reconstructed_buffer_bytes': buffers,
        'largest_checkpoint_tensor_bytes': largest_source,
        'largest_loading_destination_bytes': largest_parameter,
        'loading_comparison_slab_bytes': slab,
        'verification_workspace_limit_bytes': min(64 * 1024 * 1024, budget),
        'candidate_swap_saved_original_bytes_per_largest_tensor': largest_parameter,
        'candidate_swap_execution_bytes_per_largest_tensor': largest_parameter,
        'host_cap_bytes': host_cap,
        'capture_and_fisher_bytes': 'input_dependent_not_estimated_by_loader',
        'cuda_reserves_bytes': 'owned_by_execution_policy',
    }
    # Group accesses by shard but materialize only one source tensor at a time.
    for path in sorted({inventory[name][0] for name in required}):
        with safe_open(path, framework='pt', device='cpu') as weights:
            for name in sorted(name for name in required if inventory[name][0] == path):
                if monitor is not None: monitor.check()
                value = _exact_copy(weights.get_tensor(name), target_dtypes[name], name, slab)
                if monitor is not None: monitor.check()
                if name == 'lm_head.weight':
                    _assign_parameter(head, 'weight', value)
                else:
                    _assign_parameter(text, names[name], value)
                del value
    if config.tie_word_embeddings:
        head.weight = text.embed_tokens.weight
        if 'lm_head.weight' in inventory:
            path = inventory['lm_head.weight'][0]
            with safe_open(path, framework='pt', device='cpu') as weights:
                # Validate the optional serialized alias without another full head.
                checkpoint_head = weights.get_tensor('lm_head.weight').reshape(-1)
                embedding = head.weight.detach().reshape(-1)
                chunk = max(1, slab // (checkpoint_head.element_size() + 2 * element_size + 4))
                for start in range(0, embedding.numel(), chunk):
                    values = checkpoint_head[start:start + chunk]
                    converted = _exact_copy(values, policy.dtype, 'lm_head.weight', slab)
                    if not torch.equal(converted, embedding[start:start + chunk]):
                        raise ValueError('tied checkpoint lm_head.weight differs from embed_tokens.weight')
    text.eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    adapter = Qwen35Adapter(text, source, lm_head=head, policy=policy,
                            workspace_limit=min(64 * 1024 * 1024, budget), max_swap_bytes=budget,
                            resource_check=monitor.check if monitor is not None else None)
    adapter.loading_estimates = estimates
    return adapter
