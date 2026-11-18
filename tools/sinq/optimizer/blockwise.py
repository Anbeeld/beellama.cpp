"""Whole-block output reconstruction with immutable decoded effective candidates."""
import torch


def block_delta(adapter, capture, weights):
    capture.validate(adapter)
    for name in weights:
        if not name.startswith(f'blk.{capture.layer}.'):
            raise ValueError('candidate weight is outside captured block: ' + name)
    with adapter.swapped(weights), torch.no_grad():
        actual = adapter.text.layers[capture.layer](capture.hidden_states, **capture.kwargs)
    if actual.shape != capture.reference_output.shape or not torch.isfinite(actual).all():
        raise ValueError('invalid whole-block candidate output')
    # Promotion must precede subtraction: BF16 subtraction can erase deltas.
    return actual.detach().float() - capture.reference_output.float()


def score_block(adapter, capture, weights):
    delta = block_delta(adapter, capture, weights).cpu().double().reshape(-1)
    # CPU F64, fixed original coordinate order, independent of candidate ordering.
    return dict(output_l1=float(delta.abs().mean()), output_mse=float(delta.square().mean()),
                source_id=adapter.source_id, context_id=capture.context_id,
                input_context=capture.context, layer=capture.layer,
                site=f'block.{capture.layer}.output', remaining_weights='stock',
                reset='fresh-sequence-cache-none', output_shape=list(capture.reference_output.shape))
