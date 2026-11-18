"""Verified offline Qwen3.5 text graph and GGUF effective-weight coordinates.

Weights passed to ``swapped`` are already native-decoded *effective* [out,in]
matrices (including the matching serialized F16 compensation). No codec lives
here. Their provenance/packed-pair validation belongs to the native/cache bridge.
"""
from contextlib import contextmanager
import hashlib
import json
import inspect
import os
from pathlib import Path

import numpy as np
import torch
from tools.sinq import export_activations as exporter


class _Reference(torch.nn.Module):
    def __init__(self, text, head):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = text
        self.lm_head = head


def _parameter_state(parameter):
    return (id(parameter), parameter._version, parameter.data_ptr(), str(parameter.dtype),
            str(parameter.device), tuple(parameter.shape))


class Qwen35Adapter:
    def __init__(self, text_model, source_gguf, *, lm_head=None, workspace_limit=64 * 1024 * 1024,
                 max_swap_bytes=256 * 1024 * 1024, policy=None, resource_check=None):
        from gguf import GGUFReader
        self.text = text_model
        self.config = text_model.config
        if text_model.training:
            raise ValueError('reference model must be in eval mode')
        from .execution_backend import resolve_policy, parameter_storage_dtype
        self.policy = policy or resolve_policy(dict(device='cpu', reference_dtype='fp32',
            budgets={'workspace_bytes': workspace_limit}))
        if lm_head is None:
            if not self.config.tie_word_embeddings:
                raise ValueError('untied model requires its verified lm_head')
            lm_head = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False,
                                      device='meta', dtype=self.policy.dtype)
            lm_head.weight = text_model.embed_tokens.weight
        self.reference = _Reference(text_model, lm_head).eval()
        for name, parameter in self.reference.named_parameters():
            module=self.reference.get_submodule(name.rpartition('.')[0])
            if parameter.dtype != parameter_storage_dtype(module,self.policy.dtype,parameter.dtype):
                raise ValueError('reference head/text storage does not match requested dtype: '+name)
        storage_exceptions={name:str(p.dtype) for name,p in self.reference.named_parameters()
                            if p.dtype != self.policy.dtype}
        reader = GGUFReader(Path(source_gguf))
        try:
            architecture = reader.fields.get('general.architecture')
            if architecture is None or architecture.contents() != 'qwen35':
                raise ValueError('source GGUF architecture must be qwen35')
            if any(key.startswith(('beellama.qk_sinq.', 'beellama.sinq.')) or 'hadamard' in key.lower()
                   for key in reader.fields):
                raise ValueError('transformed source GGUF is unsupported')
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            if len(tensors) != len(reader.tensors):
                raise ValueError('duplicate source tensor names')
            config_fields = exporter.verify_reference_config(reader, self.config)
            self.identity = exporter.verify_reference_identity(self.reference, self.config, tensors, workspace_limit,
                                                               require_exact_execution_storage=True, resource_check=resource_check)
            self.identity['verified_config_fields'] = config_fields
            self.identity['source_gguf_sha256'] = exporter.sha256(source_gguf)
            self.identity['reference_implementation_sha256'] = exporter.sha256(inspect.getfile(type(text_model)))
            self.identity['torch_version'] = torch.__version__
            self.identity['cpu_threads'] = torch.get_num_threads()
            self.identity['execution_dtype'] = str(self.policy.dtype)
            self.identity['execution_storage_policy'] = ('bf16-bulk-exact-fp32-qwen-norms-v1'
                if self.policy.dtype == torch.bfloat16 else 'uniform-fp32-v1')
            self.identity['parameter_storage_exceptions'] = storage_exceptions
            self.identity['parameter_storage_exception_bytes'] = sum(p.numel()*p.element_size()
                for name,p in self.reference.named_parameters() if name in storage_exceptions)
            self.identity['source_storage_identity'] = 'exact-complete-GGUF-value-verification'
            self.identity['candidate_precision'] = 'effective weights rounded to actual parameter storage before execution'
            self.identity['graph_classes_sha256'] = hashlib.sha256(json.dumps({
                name: type(module).__module__ + '.' + type(module).__qualname__
                for name, module in self.reference.named_modules()}, sort_keys=True).encode()).hexdigest()
        finally:
            reader.data._mmap.close()
        # The full verifier checks exact execution values in bounded slabs.
        # Place only after verification; no full CPU mirror survives placement.
        from .device_placement import place_reference
        self.identity['placement'] = place_reference(self.text, lm_head, self.policy)
        self.reference.requires_grad_(False)
        self.identity['execution_policy'] = self.policy.identity()
        self.identity['reference_config_sha256'] = hashlib.sha256(
            json.dumps(self.config.to_dict(), sort_keys=True).encode()).hexdigest()
        from .implementation import implementation_fingerprint
        self.identity['implementation'] = implementation_fingerprint()
        import transformers
        self.identity['transformers_version'] = transformers.__version__
        self.source_id = hashlib.sha256(json.dumps(self.identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if max_swap_bytes <= 0:
            raise ValueError('max_swap_bytes must be positive')
        self.max_swap_bytes = max_swap_bytes
        self._versions = {name: _parameter_state(p) for name, p in self.reference.named_parameters()}
        self._config_digest = json.dumps(self.config.to_dict(), sort_keys=True)
        self._swapping = False
        from .capture import _digest
        self._buffer_digest = _digest(dict(self.reference.named_buffers()))

    def assert_reference(self):
        if any(module.training for module in self.reference.modules()) or self._swapping:
            raise ValueError('reference graph is not in stock eval state')
        if json.dumps(self.config.to_dict(), sort_keys=True) != self._config_digest:
            raise ValueError('reference configuration changed after full source verification')
        if {name: _parameter_state(p) for name, p in self.reference.named_parameters()} != self._versions:
            raise ValueError('reference parameters changed after full source verification')
        from .capture import _digest
        if _digest(dict(self.reference.named_buffers())) != self._buffer_digest:
            raise ValueError('reference buffers changed after full source verification')
        if self.policy.identity() != self.identity['execution_policy']:
            raise ValueError('execution policy changed after reference identity was locked')
        if (torch.get_num_threads() != self.identity['cpu_threads']
                or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
                or torch.backends.mkldnn.enabled != self.policy.deterministic['cpu_mkldnn_enabled']):
            raise ValueError('execution settings changed after reference identity was locked')
        if self.policy.input_device.type == 'cuda' and (
                torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
                or torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic
                or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != self.policy.deterministic['cublas_workspace_config']):
            raise ValueError('CUDA deterministic settings changed after reference identity was locked')

    def parameter(self, name):
        module_path, _, _ = exporter.module_for(name)
        return self.reference.get_submodule(module_path).weight

    def _role(self, name):
        return exporter.module_for(name)[2]

    def effective_weight(self, name):
        self.assert_reference()
        if self.parameter(name).numel()*12 > self.max_swap_bytes:
            raise ValueError('effective weight export exceeds workspace budget')
        return np.ascontiguousarray(exporter.converter_weight(
            self.parameter(name).detach().float().cpu().numpy(), self._role(name), self.config)).copy()

    def to_hf(self, name, effective):
        parameter = self.parameter(name)
        if isinstance(effective, torch.Tensor):
            effective = effective.detach().float().cpu().numpy()
        value = np.asarray(effective)
        if value.shape != tuple(parameter.shape) or value.dtype.kind != 'f' or not np.isfinite(value).all():
            raise ValueError('invalid decoded effective weight: ' + name)
        role = self._role(name)
        # Derive the permutation from the canonical exporter, not another mapping.
        if role == 'ssm_out':
            order = exporter.converter_weight(np.arange(value.shape[1])[None, :], role, self.config)[0]
            value = value[:, np.argsort(order)]
        elif role in ('attn_qkv', 'attn_gate', 'ssm_alpha', 'ssm_beta'):
            order = exporter.converter_weight(np.arange(value.shape[0])[:, None], role, self.config)[:, 0]
            value = value[np.argsort(order)]
        return np.ascontiguousarray(value)

    @contextmanager
    def swapped(self, weights):
        """Atomically validate, swap, then restore exact parameter bytes on all exits."""
        self.assert_reference()
        # decoded F32 + reordered F32 + converted candidate + exact original,
        # plus a F32 validation slab; a byte bound, not bits per element.
        if sum(self.parameter(name).numel() * (12 + 2 * self.parameter(name).element_size())
               for name in weights) > self.max_swap_bytes:
            raise ValueError('effective-weight swap exceeds workspace budget')
        converted = {name: torch.from_numpy(self.to_hf(name, value).copy()).to(self.parameter(name))
                     for name, value in sorted(weights.items())}
        if any(not torch.isfinite(value).all() for value in converted.values()):
            raise ValueError('effective weight is not finite in reference execution dtype')
        originals = {name: self.parameter(name).detach().clone() for name in converted}
        self._swapping = True
        try:
            with torch.no_grad():
                for name, value in converted.items():
                    self.parameter(name).copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in originals.items():
                    self.parameter(name).copy_(value)
            self._swapping = False
            restored = {id(self.parameter(name)) for name in originals}
            for name, parameter in self.reference.named_parameters():
                if id(parameter) in restored:
                    self._versions[name] = _parameter_state(parameter)

    def forward(self, tokens, *, attention_mask=None, position_ids=None):
        device = self.policy.input_device
        return self.text(input_ids=tokens.to(device),
            attention_mask=attention_mask.to(device) if attention_mask is not None else None,
            position_ids=position_ids.to(device) if position_ids is not None else None, use_cache=False)

    def logits(self, tokens, *, attention_mask=None, position_ids=None):
        return self.reference.lm_head(self.forward(tokens, attention_mask=attention_mask,
                                                  position_ids=position_ids).last_hidden_state)
