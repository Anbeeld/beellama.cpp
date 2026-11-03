"""Verified offline Qwen3.5 text graph and GGUF effective-weight coordinates.

Weights passed to ``swapped`` are already native-decoded *effective* [out,in]
matrices (including the matching serialized F16 compensation). No codec lives
here. Their provenance/packed-pair validation belongs to the native/cache bridge.
"""
from contextlib import contextmanager
import hashlib
import json
import inspect
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


class Qwen35Adapter:
    def __init__(self, text_model, source_gguf, *, lm_head=None, workspace_limit=64 * 1024 * 1024,
                 max_swap_bytes=256 * 1024 * 1024):
        from gguf import GGUFReader
        self.text = text_model
        self.config = text_model.config
        if text_model.training:
            raise ValueError('reference model must be in eval mode')
        if any(p.device.type != 'cpu' for p in text_model.parameters()):
            raise ValueError('this offline adapter currently requires explicit CPU execution')
        if any(p.dtype != torch.float32 for p in text_model.parameters()):
            raise ValueError('reference execution requires storage-rounded F32 parameters')
        if lm_head is None:
            if not self.config.tie_word_embeddings:
                raise ValueError('untied model requires its verified lm_head')
            lm_head = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
            lm_head.weight = text_model.embed_tokens.weight
        self.reference = _Reference(text_model, lm_head).eval()
        reader = GGUFReader(Path(source_gguf))
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
                                                           require_exact_execution_storage=True)
        self.identity['verified_config_fields'] = config_fields
        self.identity['source_gguf_sha256'] = exporter.sha256(source_gguf)
        self.identity['reference_implementation_sha256'] = exporter.sha256(inspect.getfile(type(text_model)))
        self.identity['torch_version'] = torch.__version__
        self.identity['cpu_threads'] = torch.get_num_threads()
        self.identity['execution_dtype'] = 'float32'
        self.identity['graph_classes_sha256'] = hashlib.sha256(json.dumps({
            name: type(module).__module__ + '.' + type(module).__qualname__
            for name, module in self.reference.named_modules()}, sort_keys=True).encode()).hexdigest()
        # The graph must execute the original source storage, not a higher-precision
        # projection that merely rounds to the same GGUF bytes during verification.
        for name, tensor in tensors.items():
            if exporter.PATTERN.fullmatch(name):
                storage = {'F16': torch.float16, 'BF16': torch.bfloat16, 'F32': torch.float32}[tensor.tensor_type.name]
                weight = self.parameter(name).detach()
                if not torch.equal(weight, weight.to(storage).float()):
                    raise ValueError('reference projection is not source-storage-rounded: ' + name)
        self.source_id = hashlib.sha256(json.dumps(self.identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if max_swap_bytes <= 0:
            raise ValueError('max_swap_bytes must be positive')
        self.max_swap_bytes = max_swap_bytes
        self._versions = {name: (id(p), p._version) for name, p in self.reference.named_parameters()}
        self._config_digest = json.dumps(self.config.to_dict(), sort_keys=True)
        self._swapping = False

    def assert_reference(self):
        if any(module.training for module in self.reference.modules()) or self._swapping:
            raise ValueError('reference graph is not in stock eval state')
        if json.dumps(self.config.to_dict(), sort_keys=True) != self._config_digest:
            raise ValueError('reference configuration changed after full source verification')
        if {name: (id(p), p._version) for name, p in self.reference.named_parameters()} != self._versions:
            raise ValueError('reference parameters changed after full source verification')

    def parameter(self, name):
        module_path, _, _ = exporter.module_for(name)
        return self.reference.get_submodule(module_path).weight

    def _role(self, name):
        return exporter.module_for(name)[2]

    def effective_weight(self, name):
        self.assert_reference()
        return np.ascontiguousarray(exporter.converter_weight(
            self.parameter(name).detach().cpu().numpy(), self._role(name), self.config)).copy()

    def to_hf(self, name, effective):
        parameter = self.parameter(name)
        if isinstance(effective, torch.Tensor):
            effective = effective.detach().cpu().numpy()
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
        if sum(self.parameter(name).numel() * 32 for name in weights) > self.max_swap_bytes:
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
                    self._versions[name] = (id(parameter), parameter._version)

    def forward(self, tokens, *, attention_mask=None, position_ids=None):
        return self.text(input_ids=tokens, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)

    def logits(self, tokens, *, attention_mask=None, position_ids=None):
        return self.reference.lm_head(self.forward(tokens, attention_mask=attention_mask,
                                                  position_ids=position_ids).last_hidden_state)
