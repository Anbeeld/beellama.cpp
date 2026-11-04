"""Fixed reset-per-sequence block contexts; reference and selected inputs never mix."""
from dataclasses import dataclass
import hashlib
import json

import torch


def _tree(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: _tree(v, fn) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree(v, fn) for v in value)
    if isinstance(value, list):
        return [_tree(v, fn) for v in value]
    return value


def _digest(value):
    digest = hashlib.sha256()
    def add(v):
        if isinstance(v, torch.Tensor):
            digest.update(str((tuple(v.shape), v.dtype)).encode())
            # Serialize original bytes, including BF16, in bounded host slabs.
            flat = v.detach().reshape(-1)
            chunk = max(1, (1 << 20) // v.element_size())
            for first in range(0, flat.numel(), chunk):
                slab = flat[first:first + chunk].contiguous().cpu()
                digest.update(slab.view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for key in sorted(v):
                digest.update(key.encode()); add(v[key])
        elif isinstance(v, (tuple, list)):
            for item in v:
                add(item)
        else:
            digest.update(repr(v).encode())
    add(value)
    return digest.hexdigest()


def _context_id(source_id, layer, context, payload_digest, baseline_digest):
    return hashlib.sha256(json.dumps(dict(source=source_id, layer=layer, context=context,
        payload=payload_digest, baseline=baseline_digest, reset='fresh-sequence-cache-none', downstream='stock'),
        sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class BlockCapture:
    source_id: str
    layer: int
    context: str
    context_id: str
    tokens: torch.Tensor
    attention_mask: torch.Tensor | None
    position_ids: torch.Tensor | None
    hidden_states: torch.Tensor
    kwargs: dict
    reference_output: torch.Tensor
    baseline_sha256: str
    payload_sha256: str

    def validate(self, adapter):
        adapter.assert_reference()
        if self.source_id != adapter.source_id:
            raise ValueError('capture source identity mismatch')
        payload = (self.tokens, self.attention_mask, self.position_ids, self.hidden_states, self.kwargs, self.reference_output)
        if _digest(payload) != self.payload_sha256 or self.context_id != _context_id(
                self.source_id, self.layer, self.context, self.payload_sha256, self.baseline_sha256):
            raise ValueError('capture payload or context changed after locking')


def capture_block(adapter, tokens, layer, *, attention_mask=None, position_ids=None,
                  context='reference_inputs', baseline=None, max_tokens=256,
                  max_capture_bytes=64 * 1024 * 1024):
    """Capture real block call arguments and output with cache=None/use_cache=False.

    Selected-input baselines may change only earlier blocks; the scored block and
    its downstream graph are explicitly stock. Baseline weights are cloned during
    swapping and never retained as multiple full models.
    """
    adapter.assert_reference()
    if not 0 <= layer < len(adapter.text.layers):
        raise ValueError('invalid block index')
    if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.device.type not in ('cpu', 'cuda') or not 0 < tokens.numel() <= max_tokens:
        raise ValueError('invalid token coordinates or token budget')
    if (tokens < 0).any() or (tokens >= adapter.config.vocab_size).any():
        raise ValueError('token ID outside vocabulary')
    if max_capture_bytes <= 0:
        raise ValueError('capture byte budget must be positive')
    if context not in ('reference_inputs', 'selected_quantized_inputs'):
        raise ValueError('unsupported block input context')
    if context == 'reference_inputs' and baseline:
        raise ValueError('reference_inputs cannot include selected baseline weights')
    if context == 'selected_quantized_inputs' and not baseline:
        raise ValueError('selected_quantized_inputs requires an explicit upstream baseline')
    baseline = baseline or {}
    for name in baseline:
        if int(name.split('.')[1]) >= layer:
            raise ValueError('selected input baseline may replace upstream blocks only')
    if attention_mask is not None:
        if attention_mask.shape != tokens.shape or not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError('attention mask must be binary [batch,time]')
    if position_ids is not None and (position_ids.shape != tokens.shape or position_ids.dtype != torch.long):
        raise ValueError('position IDs must be int64 [batch,time]')
    for value in (attention_mask, position_ids):
        if value is not None and value.device != tokens.device:
            raise ValueError('mixed-device token coordinates')
    # Bound the captured hidden/output, RoPE and dense causal-mask payload before forward.
    b, t = tokens.shape
    estimate = 4 * (2 * b * t * adapter.config.hidden_size + 2 * b * t * adapter.config.head_dim + b * t * t) + 32 * b * t
    if estimate > max_capture_bytes:
        raise ValueError('block capture exceeds byte budget')
    saved = {}
    def before(module, args, kwargs):
        saved['hidden'] = args[0].detach().clone()
        if kwargs.get('past_key_values') is not None:
            raise ValueError('block capture requires reset recurrent/cache state')
        saved['kwargs'] = _tree(kwargs, lambda x: x.detach().clone())
    def after(module, args, output):
        if not isinstance(output, torch.Tensor):
            raise ValueError('unsupported Qwen3.5 block output contract')
        saved['output'] = output.detach().clone()
    block = adapter.text.layers[layer]
    handles = [block.register_forward_pre_hook(before, with_kwargs=True), block.register_forward_hook(after)]
    try:
        with adapter.swapped(baseline), torch.no_grad():
            adapter.forward(tokens, attention_mask=attention_mask, position_ids=position_ids)
    finally:
        for handle in handles:
            handle.remove()
    if set(saved) != {'hidden', 'kwargs', 'output'}:
        raise ValueError('missing block input/output capture')
    payload = _tree((tokens, attention_mask, position_ids, saved['hidden'], saved['kwargs'], saved['output']),
                    lambda x: x.detach().clone())
    count = [0]
    def size(x):
        if not torch.isfinite(x).all():
            raise ValueError('nonfinite reference block capture')
        count[0] += x.numel() * x.element_size()
        return x
    _tree(payload, size)
    if count[0] > max_capture_bytes:
        raise ValueError('actual block capture exceeds byte budget')
    payload_digest = _digest(payload)
    baseline_digest = _digest({k: torch.as_tensor(v) for k, v in sorted(baseline.items())})
    context_id = _context_id(adapter.source_id, layer, context, payload_digest, baseline_digest)
    return BlockCapture(adapter.source_id, layer, context, context_id, *payload[:3], *payload[3:],
                        baseline_digest, payload_digest)
