#!/usr/bin/env python3
"""Offline reference-loader tests; all CPU execution uses 32 threads."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'gguf-py'))

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

torch.set_num_threads(32)


class Policy:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
        self.devices = (torch.device('cpu'),)
        self.input_device = torch.device('cpu')

    def identity(self):
        return {'dtype': str(self.dtype), 'devices': ['cpu']}


class AdapterFixture:
    """Observe the verified-adapter boundary without inventing GGUF identity."""
    def __init__(self, text, source, *, lm_head, policy, **kwargs):
        self.text, self.head, self.policy = text, lm_head, policy
        self.source = source
        self.options = kwargs
        from tools.sinq.optimizer.execution_backend import parameter_storage_dtype
        assert all(p.device.type == 'cpu' and p.dtype == parameter_storage_dtype(
            text.get_submodule(name.rpartition('.')[0]), policy.dtype, p.dtype)
            for name, p in text.named_parameters())
        assert all(not p.requires_grad for p in text.parameters())
        assert not text.training


class ReferenceLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = importlib.import_module('tools.sinq.optimizer.reference_loader')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        config = Qwen3_5TextConfig(
            vocab_size=16, hidden_size=32, intermediate_size=48,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, max_position_embeddings=32,
            layer_types=['full_attention'], tie_word_embeddings=True,
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                             'partial_rotary_factor': 0.5, 'mrope_section': [1, 1, 0]})
        config._attn_implementation = 'eager'
        torch.manual_seed(29)
        self.original = Qwen3_5TextModel(config).eval()
        self.config = config.to_dict()
        self.config['architectures'] = ['Qwen3_5TextModel']
        self.state = {name: value.detach().clone() for name, value in self.original.state_dict().items()}

    def checkpoint(self, *, conditional=False, tied=True, bf16=False, omit=None, extra=None):
        config = dict(self.config, tie_word_embeddings=tied)
        values = dict(self.state)
        if omit:
            del values[omit]
        if bf16:
            values = {k: v.to(torch.bfloat16) for k, v in values.items()}
        if conditional:
            config = {'model_type': 'qwen3_5', 'text_config': config,
                      'architectures': ['Qwen3_5ForConditionalGeneration']}
            values = {'model.language_model.' + k: v for k, v in values.items()}
            # Vision weights must not be loaded, checked for BF16, or counted as residency.
            values['model.visual.arbitrary.weight'] = torch.tensor([1.0001])
        if not tied:
            values['lm_head.weight'] = torch.full((16, 32), 0.25)
        if extra:
            values.update(extra)
        (self.directory / 'config.json').write_text(json.dumps(config), encoding='utf-8')
        keys = sorted(values)
        chunks = [keys[::2], keys[1::2]]
        mapping = {}
        for i, names in enumerate(chunks):
            filename = f'model-{i:05d}.safetensors'
            save_file({k: values[k] for k in names}, self.directory / filename)
            mapping.update({k: filename for k in names})
        (self.directory / 'model.safetensors.index.json').write_text(
            json.dumps({'weight_map': mapping}), encoding='utf-8')

    def load(self, *, dtype=torch.float32, budget=10**8):
        with patch('tools.sinq.optimizer.qwen35_adapter.Qwen35Adapter', AdapterFixture):
            return self.loader.load_reference(self.directory, 'source.gguf', budget, 32,
                                              policy=Policy(dtype))

    def test_sharded_fp32_tied_forward_and_buffers(self):
        self.checkpoint()
        loaded = self.load()
        self.assertIs(loaded.head.weight, loaded.text.embed_tokens.weight)
        self.assertTrue(all(b.device.type == 'cpu' for b in loaded.text.buffers()))
        self.assertTrue(all(b.dtype == torch.float32 for b in loaded.text.buffers()))
        tokens = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            expected = self.original(tokens, use_cache=False).last_hidden_state
            actual = loaded.text(tokens, use_cache=False).last_hidden_state
        self.assertTrue(torch.equal(expected, actual))
        for name, parameter in loaded.text.named_parameters():
            self.assertTrue(torch.equal(parameter, self.state[name]), name)
        self.assertEqual(loaded.loading_estimates['storage_weight_bytes'],
                         sum(p.numel() * p.element_size() for p in loaded.text.parameters()))

    def test_meta_initialization_creates_only_one_text_graph(self):
        self.checkpoint()
        initialized = []

        def construct(config):
            model = Qwen3_5TextModel(config)
            initialized.append(model)
            self.assertTrue(all(p.device.type == 'meta' for p in model.parameters()))
            return model

        with patch('transformers.Qwen3_5TextModel', side_effect=construct):
            loaded = self.load()
        self.assertEqual(len(initialized), 1)
        self.assertIs(loaded.text, initialized[0])

    def test_default_policy_is_cpu_fp32(self):
        self.checkpoint()
        with patch('tools.sinq.optimizer.qwen35_adapter.Qwen35Adapter', AdapterFixture):
            loaded = self.loader.load_reference(self.directory, 'source.gguf', 10**8, 32)
        self.assertEqual(loaded.policy.dtype, torch.float32)
        self.assertEqual(str(loaded.policy.input_device), 'cpu')

    def test_equal_serialized_tied_head(self):
        self.checkpoint(extra={'lm_head.weight': self.state['embed_tokens.weight'].clone()})
        loaded = self.load()
        self.assertIs(loaded.head.weight, loaded.text.embed_tokens.weight)

    def test_fp16_values_exact_for_bf16(self):
        self.state = {name: torch.full_like(value, 0.25, dtype=torch.float16)
                      for name, value in self.state.items()}
        self.checkpoint()
        loaded = self.load(dtype=torch.bfloat16)
        self.assertTrue(all(torch.all(p == 0.25) for p in loaded.text.parameters()))

    def test_fp16_values_inexact_for_bf16(self):
        self.state = {name: torch.full_like(value, 0.25, dtype=torch.float16)
                      for name, value in self.state.items()}
        self.checkpoint(extra={'norm.weight': torch.full((32,), 1.0009765625, dtype=torch.float16)})
        with self.assertRaisesRegex(ValueError, 'norm.weight'):
            self.load(dtype=torch.bfloat16)

    def test_conditional_untied_bf16_excludes_vision(self):
        self.checkpoint(conditional=True, tied=False, bf16=True)
        loaded = self.load(dtype=torch.bfloat16)
        self.assertIsNot(loaded.head.weight, loaded.text.embed_tokens.weight)
        self.assertEqual(loaded.head.weight.dtype, torch.bfloat16)
        self.assertFalse(any('visual' in name for name, _ in loaded.text.named_modules()))
        self.assertTrue(all(b.dtype == torch.float32 for b in loaded.text.buffers()))
        with torch.no_grad():
            output = loaded.text(torch.tensor([[1, 2]]), use_cache=False).last_hidden_state
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all())

    def test_exact_fp32_norm_island_in_bf16_graph(self):
        original=torch.full((32,), 1.0001)
        self.checkpoint(bf16=True, extra={'norm.weight': original})
        loaded=self.load(dtype=torch.bfloat16)
        self.assertEqual(loaded.text.norm.weight.dtype,torch.float32)
        self.assertTrue(torch.equal(loaded.text.norm.weight,original))
        self.assertEqual(loaded.text.embed_tokens.weight.dtype,torch.bfloat16)
        with torch.no_grad():
            output=loaded.text(torch.tensor([[1,2]]),use_cache=False).last_hidden_state
        self.assertEqual(output.dtype,torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all())

    def test_lossy_bf16_projection_rejected_with_checkpoint_name(self):
        name='layers.0.mlp.down_proj.weight'
        self.checkpoint(bf16=True, extra={name: torch.full_like(self.state[name], 1.0001)})
        with self.assertRaisesRegex(ValueError, 'down_proj.weight'):
            self.load(dtype=torch.bfloat16)

    def test_missing_parameter_fails(self):
        self.checkpoint(omit='norm.weight')
        with self.assertRaisesRegex(ValueError, 'norm.weight'):
            self.load()

    def test_unexpected_text_parameter_fails(self):
        self.checkpoint(extra={'layers.0.invalid.weight': torch.ones(1)})
        with self.assertRaisesRegex(ValueError, 'invalid.weight'):
            self.load()

    def test_untied_head_required(self):
        self.checkpoint()
        cfg = dict(self.config, tie_word_embeddings=False)
        (self.directory / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'lm_head.weight'):
            self.load()

    def test_conflicting_tied_head_rejected(self):
        self.checkpoint(extra={'lm_head.weight': torch.ones(16, 32)})
        with self.assertRaisesRegex(ValueError, 'tied|lm_head'):
            self.load()

    def test_preflight_impossible_host_cap(self):
        self.checkpoint()
        with self.assertRaisesRegex(ValueError, 'budget|cap|memory'):
            self.load(budget=1)

    def test_workspace_is_not_resident_host_cap(self):
        self.checkpoint()
        policy = Policy()
        policy.memory_policy = {'host_limit_bytes': 10**8}
        with patch('tools.sinq.optimizer.qwen35_adapter.Qwen35Adapter', AdapterFixture):
            loaded = self.loader.load_reference(self.directory, 'source.gguf', 1024, 32, policy=policy)
        self.assertGreater(loaded.loading_estimates['storage_weight_bytes'], 1024)

    def test_policy_host_cap_is_enforced(self):
        self.checkpoint()
        policy = Policy()
        policy.memory_policy = {'host_limit_bytes': 1}
        with self.assertRaisesRegex(ValueError, 'memory budget'):
            self.loader.load_reference(self.directory, 'source.gguf', 10**8, 32, policy=policy)

    def test_shape_mismatch_rejected(self):
        self.checkpoint(extra={'norm.weight': torch.ones(31)})
        with self.assertRaisesRegex(ValueError, 'shape.*norm.weight'):
            self.load()

    def test_nonfinite_parameter_rejected(self):
        self.checkpoint(extra={'norm.weight': torch.full((32,), float('nan'))})
        with self.assertRaisesRegex(ValueError, 'norm.weight'):
            self.load()

    def test_quantized_config_rejected(self):
        self.checkpoint()
        cfg = dict(self.config, quantization_config={'quant_method': 'gptq'})
        (self.directory / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'quantiz'):
            self.load()

    def test_index_mismatch_rejected(self):
        self.checkpoint()
        path = self.directory / 'model.safetensors.index.json'
        index = json.loads(path.read_text(encoding='utf-8'))
        index['weight_map']['norm.weight'] = 'missing.safetensors'
        path.write_text(json.dumps(index), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'index|shard|checkpoint'):
            self.load()


if __name__ == '__main__':
    unittest.main()
