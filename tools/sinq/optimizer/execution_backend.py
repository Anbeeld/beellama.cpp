"""Explicit execution identity and runtime capability validation (no implicit fallback)."""
from dataclasses import dataclass
import importlib.metadata
import os
import copy
from .schemas import validate_execution_config


@dataclass
class ExecutionPolicy:
    dtype: object
    devices: tuple
    reference_dtype: str
    memory_policy: dict
    deterministic: dict
    versions: dict
    device_identifiers: dict

    @property
    def input_device(self):
        return self.devices[0]

    def identity(self):
        return copy.deepcopy(dict(reference_dtype=self.reference_dtype, execution_dtype=str(self.dtype),
                    parameter_storage_policy=('bf16-bulk-exact-fp32-qwen-norms-v1'
                        if self.reference_dtype == 'bf16' else 'uniform-fp32-v1'),
                    devices=[str(d) for d in self.devices], memory_policy=dict(self.memory_policy),
                    deterministic=dict(self.deterministic), versions=dict(self.versions),
                    device_identifiers=dict(self.device_identifiers)))


def resolve_policy(config):
    c = validate_execution_config(config)
    import torch
    # Configure before any CUDA context creation. Refuse incompatible pre-existing state.
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if c['device'] != 'cpu':
        if workspace not in (None, ':4096:8', ':16:8'):
            raise ValueError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8')
        if workspace is None and torch.cuda.is_initialized():
            raise ValueError('set CUBLAS_WORKSPACE_CONFIG before CUDA initialization')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    if c['device'] != 'cpu':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if c['device'] == 'cpu':
        devices = (torch.device('cpu'),)
    else:
        if not torch.cuda.is_available():
            raise ValueError('CUDA execution requested but Torch CUDA is unavailable')
        names = c.get('cuda_devices') or [c['device'] if c['device'] != 'cuda' else f'cuda:{torch.cuda.current_device()}']
        devices = tuple(torch.device(n) for n in names)
        for d in devices:
            if d.index >= torch.cuda.device_count():
                raise ValueError(f'CUDA device {d} does not exist')
            if c['reference_dtype'] == 'bf16':
                with torch.cuda.device(d):
                    if not torch.cuda.is_bf16_supported(including_emulation=False):
                        raise ValueError(f'CUDA device {d} does not support BF16 execution')
    m = dict(c.get('memory_policy', {}))
    m.setdefault('reserve_bytes', 256 << 20)
    if 'host_limit_bytes' not in m:
        from .memory_accounting import host_capacity
        capacity = host_capacity()
        m['host_limit_bytes'] = max(1, capacity['available_bytes'] + capacity['current_bytes'] - m['reserve_bytes'])
        m['host_limit_source'] = capacity['source']
    else:
        m['host_limit_source'] = 'explicit memory_policy.host_limit_bytes'
    m.setdefault('cuda_limit_bytes', {})
    if not m['cuda_limit_bytes'].keys() <= {str(d) for d in devices}:
        raise ValueError('CUDA limits name unselected devices')
    try: transformers = importlib.metadata.version('transformers')
    except importlib.metadata.PackageNotFoundError: transformers = None
    identifiers = {}
    for device in devices:
        if device.type == 'cuda':
            props = torch.cuda.get_device_properties(device)
            identifiers[str(device)] = dict(uuid=str(getattr(props, 'uuid', 'not_measured')),
                diagnostic_name=props.name, diagnostic_total_bytes=props.total_memory,
                compute_capability=[props.major, props.minor])
    return ExecutionPolicy(torch.float32 if c['reference_dtype'] == 'fp32' else torch.bfloat16,
        devices, c['reference_dtype'], m,
        dict(algorithms=True, warn_only=False, cudnn_benchmark=False, cudnn_deterministic=True, tf32=False,
             cpu_mkldnn_enabled=torch.backends.mkldnn.enabled,
             cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG') if devices[0].type == 'cuda' else None),
        dict(torch=torch.__version__, transformers=transformers, cuda=torch.version.cuda,
             cudnn=torch.backends.cudnn.version() if devices[0].type == 'cuda' else None), identifiers)


def parameter_storage_dtype(module, requested_dtype, source_dtype):
    """Keep original FP32 Qwen normalization storage during BF16 bulk execution.

    These stock operators explicitly compute in FP32 and cast their outputs back
    to the activation dtype. This is not a lossy-reference or linear-weight
    fallback: only their original FP32 parameters retain their exact values.
    """
    import torch
    if requested_dtype == torch.bfloat16 and source_dtype == torch.float32:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm, Qwen3_5RMSNormGated
        if isinstance(module, (Qwen3_5RMSNorm, Qwen3_5RMSNormGated)):
            return torch.float32
    return requested_dtype
