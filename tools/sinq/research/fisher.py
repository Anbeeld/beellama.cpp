"""Downstream output-Fisher VJPs, never NLL gradients or a dense Fisher.

Each selected output-token distribution is a separate context. The objective is
one half the mean over contexts of E_v[(sum_all_site_coordinates g_v*delta)^2].
The sum includes every token/feature of the site, retaining nonlocal recurrent
and cross-feature terms. MC samples categorical pseudo-targets from p_ref.
"""
from dataclasses import dataclass
import hashlib
import math

import torch


@dataclass(frozen=True)
class FisherProbes:
    gradients: torch.Tensor
    weights: torch.Tensor
    probabilities: torch.Tensor
    metadata: dict

    def score(self, delta):
        delta = torch.as_tensor(delta).detach().cpu().double()
        if tuple(delta.shape) != tuple(self.gradients.shape[1:]) or not torch.isfinite(delta).all():
            raise ValueError('Fisher delta shape/coordinates or finite values mismatch')
        # Never create an all-probes temporary of site size during scoring.
        terms = []
        flat = delta.reshape(-1)
        for gradient, weight in zip(self.gradients, self.weights):
            projection = torch.dot(gradient.reshape(-1), flat).item()
            terms.append(weight.item() * projection * projection)
        result = 0.5 * math.fsum(terms)
        if not math.isfinite(result) or result < 0:
            raise ValueError('invalid Fisher score')
        return result


def build_probes(logits, site, *, estimator='fisher_vjp_mc', probes=16, seed=0,
                 positions=None, max_vjps=256, max_cache_bytes=64 * 1024 * 1024,
                 metadata=None):
    """Consume one autograd graph; return detached bounded CPU probe information.

    Exact mode enumerates each vocabulary outcome. MC deduplicates repeated
    categorical targets without changing the empirical estimator. Budgets bound
    requested VJPs even when deduplication reduces actual autograd calls.
    """
    if estimator not in ('fisher_vjp_mc', 'fisher_vjp_exact'):
        raise ValueError('unsupported output-Fisher estimator')
    if logits.ndim != 3 or logits.shape[-1] < 2 or not logits.requires_grad or not site.requires_grad:
        raise ValueError('Fisher requires differentiable [batch,time,vocab] logits and site')
    if not torch.isfinite(logits).all() or not torch.isfinite(site).all():
        raise ValueError('nonfinite reference Fisher graph')
    if logits.device.type != 'cpu' or site.device.type != 'cpu':
        raise ValueError('this Fisher implementation requires explicit CPU execution')
    if not isinstance(seed, int) or not 0 <= seed <= 0xffffffff:
        raise ValueError('Fisher seed must be uint32')
    if not isinstance(probes, int) or probes <= 0 or max_vjps <= 0 or max_cache_bytes <= 0:
        raise ValueError('invalid Fisher probe or memory budget')
    b, t, vocab = logits.shape
    positions = list(positions) if positions is not None else [(i, j) for i in range(b) for j in range(t)]
    if not positions or len(set(positions)) != len(positions) or any(
            not (0 <= i < b and 0 <= j < t) for i, j in positions):
        raise ValueError('invalid or duplicate Fisher output positions')
    requested = len(positions) * (vocab if estimator == 'fisher_vjp_exact' else probes)
    if requested > max_vjps:
        raise ValueError('Fisher VJP budget exceeded')
    max_rank = len(positions) * (vocab if estimator == 'fisher_vjp_exact' else min(probes, vocab))
    if (max_rank * (site.numel() + 1) + len(positions) * vocab) * 8 > max_cache_bytes:
        raise ValueError('Fisher probe cache byte budget exceeded')
    logp = logits.log_softmax(-1)
    if not torch.isfinite(logp).all():
        raise ValueError('nonfinite reference log probabilities')
    probabilities = torch.stack([logp[i, j].detach().double().exp() for i, j in positions])
    probabilities /= probabilities.sum(-1, keepdim=True)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    outcomes, coefficients, sampled = [], [], []
    for index, (i, j) in enumerate(positions):
        if estimator == 'fisher_vjp_exact':
            counts = None
            targets = range(vocab)
        else:
            samples = torch.multinomial(probabilities[index], probes, replacement=True, generator=generator)
            sampled.extend(samples.tolist())
            counts = torch.bincount(samples, minlength=vocab)
            targets = torch.nonzero(counts).reshape(-1).tolist()
        for target in targets:
            outcomes.append((i, j, target))
            coefficient = probabilities[index, target].item() if counts is None else counts[target].item() / probes
            coefficients.append(coefficient / len(positions))
    gradients = []
    for index, (i, j, target) in enumerate(outcomes):
        gradient, = torch.autograd.grad(logp[i, j, target], site,
            retain_graph=index < len(outcomes) - 1, create_graph=False, allow_unused=False)
        if not torch.isfinite(gradient).all():
            raise ValueError('nonfinite downstream score gradient')
        gradients.append(gradient.detach().cpu().double())
    reserved = dict(estimator=estimator, seed=seed, probes_per_position=probes if estimator == 'fisher_vjp_mc' else vocab,
        requested_vjps=requested, actual_vjps=len(outcomes), rank=len(outcomes),
        positions=[list(p) for p in positions], site_shape=list(site.shape), device='cpu',
        gradient_dtype='float64', objective='half-mean-context-expected-squared-full-site-pairing',
        max_vjps=max_vjps, max_cache_bytes=max_cache_bytes,
        sampled_targets_sha256=hashlib.sha256(str(sampled).encode()).hexdigest(),
        probabilities_sha256=hashlib.sha256(probabilities.numpy().tobytes()).hexdigest())
    if set(metadata or {}) & set(reserved):
        raise ValueError('caller metadata cannot override Fisher estimator identity')
    return FisherProbes(torch.stack(gradients), torch.tensor(coefficients, dtype=torch.float64),
                        probabilities, {**(metadata or {}), **reserved})


def qwen_probes(adapter, capture, **options):
    """VJPs through the real stock downstream Qwen3.5 graph from block output.

    The captured incoming states override the target block input, so selected
    upstream inputs remain locked. All downstream weights are verified stock.
    Each call starts with no attention/recurrent cache; no graph is cached.
    The capture mask selects output distributions only, never site coordinates.
    """
    capture.validate(adapter)
    if 'metadata' in options:
        raise ValueError('Qwen Fisher provenance is derived from the capture')
    b, t = capture.tokens.shape
    mask = capture.attention_mask
    valid_positions = ([(i, j) for i in range(b) for j in range(t)] if mask is None
                       else [tuple(p) for p in torch.nonzero(mask, as_tuple=False).tolist()])
    requested_positions = options.pop('positions', None)
    selection = ('all-batch-time' if mask is None else 'capture-attention-mask')
    if requested_positions is None:
        positions = valid_positions
    else:
        positions = [tuple(p) for p in requested_positions]
        selection = 'explicit'
    valid = set(valid_positions)
    if not positions or len(set(positions)) != len(positions) or any(p not in valid for p in positions):
        raise ValueError('invalid, duplicate, or masked Fisher output positions')
    saved = {}
    def before(module, args, kwargs):
        return (capture.hidden_states,), dict(capture.kwargs)
    def after(module, args, output):
        if not torch.equal(output.detach(), capture.reference_output):
            raise ValueError('Qwen reference block output differs from locked capture')
        saved['site'] = output.detach().clone().requires_grad_(True)
        return saved['site']
    block = adapter.text.layers[capture.layer]
    handles = [block.register_forward_pre_hook(before, with_kwargs=True), block.register_forward_hook(after)]
    try:
        with torch.enable_grad():
            logits = adapter.logits(capture.tokens, attention_mask=capture.attention_mask, position_ids=capture.position_ids)
            return build_probes(logits, saved['site'], metadata=dict(source_id=adapter.source_id,
                source_gguf_sha256=adapter.identity['source_gguf_sha256'], context_id=capture.context_id,
                site=f'block.{capture.layer}.output', input_context=capture.context,
                baseline_sha256=capture.baseline_sha256, downstream='stock', reset='fresh-sequence-cache-none',
                output_position_selection=selection,
                output_position_mask_context_id=capture.context_id if mask is not None else None),
                positions=positions, **options)
    finally:
        for handle in handles:
            handle.remove()
