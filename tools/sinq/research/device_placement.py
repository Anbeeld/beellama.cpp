"""Complete-module placement with differentiable argument transfers."""
import torch


def transfer(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {k: transfer(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        items = [transfer(v, device) for v in value]
        return type(value)(*items) if hasattr(value, '_fields') else tuple(items)
    if isinstance(value, list):
        return [transfer(v, device) for v in value]
    return value


def _hook(device):
    def move(module, args, kwargs):
        return transfer(args, device), transfer(kwargs, device)
    return move


def place_reference(text, head, policy):
    """Return a JSON module->device map; hooks are retained on the text graph."""
    if getattr(text, '_execution_transfer_handles', None):
        raise ValueError('reference already has execution placement hooks')
    modules = [('embed_tokens', text.embed_tokens)]
    modules += [(f'layers.{i}', layer) for i, layer in enumerate(text.layers)]
    modules += [('norm', text.norm), ('lm_head', head)]
    # Include rotary/other auxiliary modules without moving all layers twice.
    known = {'embed_tokens', 'layers', 'norm'}
    modules += [(name, module) for name, module in text.named_children() if name not in known]
    sizes = {name: sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers()))
        for name, module in modules}
    devices = policy.devices
    available = []
    for d in devices:
        if d.type == 'cpu':
            budget = policy.memory_policy['host_limit_bytes']
        else:
            free, _ = torch.cuda.mem_get_info(d)
            budget = min(free, policy.memory_policy['cuda_limit_bytes'].get(str(d), free))
        available.append(max(0, budget - policy.memory_policy['reserve_bytes']))
    used = [0] * len(devices)
    plan = {}
    cursor = 0
    tied = head.weight is text.embed_tokens.weight
    if tied:
        sizes['lm_head'] -= head.weight.numel() * head.weight.element_size()
    total = sum(sizes.values())
    if total > sum(available):
        raise ValueError('INSUFFICIENT_CAPACITY: reference storage exceeds placement budgets')
    cumulative = 0
    boundaries = []
    for capacity in available:
        cumulative += capacity
        boundaries.append(total * cumulative / max(1, sum(available)))
    assigned_bytes = 0
    for name, module in modules:
        size = sizes[name]
        if name == 'lm_head' and tied:
            index = 0
            # The aliased weight was excluded above; count any head bias/buffers.
        elif name not in ('embed_tokens',) and name.startswith('layers.'):
            while cursor < len(devices)-1 and (assigned_bytes + size / 2 >= boundaries[cursor] or used[cursor]+size > available[cursor]):
                cursor += 1
            index = cursor
        elif name in ('norm', 'lm_head'):
            index = cursor
        else:
            index = 0
        if used[index] + size > available[index]:
            raise ValueError(f'INSUFFICIENT_CAPACITY: complete module {name} does not fit {devices[index]}')
        used[index] += size
        assigned_bytes += size
        plan[name] = str(devices[index])
    handles = []
    # Plan is fully checked before any mutations.
    for name, module in modules:
        device = torch.device(plan[name])
        if name == 'lm_head' and tied:
            # Embedding movement may replace Parameter objects. Restore alias before
            # moving the head so no second full output weight is materialized.
            head.weight = text.embed_tokens.weight
        module.to(device=device)
        if name == 'lm_head' and tied:
            head.weight = text.embed_tokens.weight
        handles.append(module.register_forward_pre_hook(_hook(device), with_kwargs=True))
    # Root-owned buffers are not covered by child-module moves.
    for name, buffer in text.named_buffers(recurse=False):
        setattr(text, name, buffer.to(device=policy.input_device))
    text._execution_transfer_handles = handles
    text._execution_placement = plan
    return dict(plan)
