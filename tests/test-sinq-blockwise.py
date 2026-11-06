#!/usr/bin/env python3
"""Offline genuine Qwen3.5 block replay; no network or native-codec approximation."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gguf-py')]
# Dependency-bound correctness coverage must be an explicit SKIP, never a stub.
import importlib.util
if any(importlib.util.find_spec(name) is None for name in ('torch','transformers','numpy','safetensors')):
    print('SKIP: local Torch/Transformers/NumPy/safetensors environment unavailable')
    raise SystemExit(77)

import numpy as np
import torch
from tools.sinq.optimizer.qwen35_adapter import Qwen35Adapter
from tools.sinq.optimizer.capture import capture_block
from tools.sinq.optimizer.blockwise import block_delta, score_block

torch.set_num_threads(32)


def tiny_adapter(directory):
    spec = importlib.util.spec_from_file_location('parity_fixture', ROOT / 'tests/test-sinq-activation-parity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.fixture(directory, 'f32')
    torch.set_num_threads(32)
    from transformers import Qwen3_5TextModel
    model = Qwen3_5TextModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False).float().eval()
    return Qwen35Adapter(model, directory / 'model.gguf')


class BlockwiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.adapter = tiny_adapter(Path(cls.temp.name))
        cls.tokens = torch.tensor([[1, 7, 3, 11, 5, 19, 2]])
        cls.mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1]])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_genuine_blocks_identity_perturb_restore_and_reset(self):
        adapter = self.adapter
        self.assertGreater(adapter.identity['verified_tensor_count'], 15)
        for layer, name in [(0, 'blk.0.ssm_out.weight'), (1, 'blk.1.attn_output.weight')]:
            original = adapter.effective_weight(name)
            captured = capture_block(adapter, self.tokens, layer, attention_mask=self.mask)
            identity = score_block(adapter, captured, {name: original})
            self.assertEqual(identity['output_l1'], 0.0)
            self.assertEqual(identity['output_mse'], 0.0)
            perturbed = original.copy()
            perturbed[0, :] += 0.125
            score = score_block(adapter, captured, {name: perturbed})
            self.assertGreater(score['output_l1'], 0)
            self.assertGreater(score['output_mse'], 0)
            delta = block_delta(adapter, captured, {name: perturbed}).numpy().astype(np.float64)
            self.assertAlmostEqual(score['output_l1'], np.mean(np.abs(delta)), places=15)
            self.assertAlmostEqual(score['output_mse'], np.mean(delta ** 2), places=15)
            np.testing.assert_array_equal(adapter.effective_weight(name), original)
            repeated = capture_block(adapter, self.tokens, layer, attention_mask=self.mask)
            torch.testing.assert_close(captured.reference_output, repeated.reference_output, rtol=0, atol=0)
            self.assertEqual(score_block(adapter, captured, {name: original})['output_l1'], 0)
            with self.assertRaises(ValueError):
                score_block(adapter, captured, {'blk.0.ffn_up.weight' if layer else 'blk.1.ffn_up.weight': np.ones((1, 1))})

    def test_split_before_reservoir_capture(self):
        from tools.sinq.optimizer.partition import capture_splits
        from tools.sinq.optimizer.schemas import validate_pairing
        import json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            names=['blk.0.ssm_out.weight','blk.1.attn_output.weight']
            for output in (root/'first',root/'second'):
                capture_splits(self.adapter,self.tokens,names,output,context=2,rows=2,seed=17,
                    validation_chunks=1,max_tokens=32,max_output_bytes=1<<20)
            pair=json.loads((root/'first/pairing.json').read_text())
            train=json.loads((root/'first/train/activations.json').read_text())
            validation=json.loads((root/'first/validation/activations.json').read_text())
            self.assertEqual(validate_pairing(pair,train,validation)['split_method'],'split_before_reservoir')
            self.assertFalse(set(pair['train']['row_ids']) & set(pair['validation']['row_ids']))
            for path in (root/'first').rglob('*'):
                if path.is_file():self.assertEqual(path.read_bytes(),(root/'second'/path.relative_to(root/'first')).read_bytes())
            with self.assertRaises(ValueError):
                capture_splits(self.adapter,self.tokens,names,root/'bad',context=2,rows=2,seed=17,
                    validation_chunks=4,max_tokens=32,max_output_bytes=1<<20)
            self.assertFalse((root/'bad').exists())

    def test_selected_inputs_and_context_binding(self):
        name = 'blk.0.ssm_out.weight'
        weight = self.adapter.effective_weight(name).copy()
        weight += 0.1
        ref = capture_block(self.adapter, self.tokens, 1)
        selected = capture_block(self.adapter, self.tokens, 1, context='selected_quantized_inputs', baseline={name: weight})
        self.assertFalse(torch.equal(ref.hidden_states, selected.hidden_states))
        self.assertNotEqual(ref.context_id, selected.context_id)
        self.assertEqual(score_block(self.adapter, selected, {})['output_mse'], 0)
        with self.assertRaises(ValueError):
            capture_block(self.adapter, self.tokens, 1, context='selected_quantized_inputs')
        with self.assertRaises(ValueError):
            capture_block(self.adapter, self.tokens, 1, baseline={name: weight})
        with self.assertRaises(ValueError):
            capture_block(self.adapter, self.tokens, 1, max_capture_bytes=1)
        from dataclasses import replace
        with self.assertRaises(ValueError):
            score_block(self.adapter, replace(ref, context='selected_quantized_inputs'), {})
        tampered = capture_block(self.adapter, self.tokens, 1)
        tampered.hidden_states[0, 0, 0] += 1
        with self.assertRaises(ValueError):
            score_block(self.adapter, tampered, {})

    def test_permutation_and_exception_restore(self):
        name = 'blk.0.ssm_out.weight'
        original = self.adapter.effective_weight(name)
        raw = self.adapter.parameter(name).detach().numpy().copy()
        self.assertFalse(np.array_equal(raw, original))
        np.testing.assert_array_equal(self.adapter.to_hf(name, original), raw)
        with self.assertRaises(RuntimeError):
            with self.adapter.swapped({name: original + 0.1}):
                raise RuntimeError('intentional')
        np.testing.assert_array_equal(self.adapter.parameter(name).detach().numpy(), raw)
        with self.assertRaises(ValueError):
            with self.adapter.swapped({name: np.full_like(original, np.nan)}):
                pass

    def test_all_projection_layout_roundtrips(self):
        roles = [
            ['ffn_gate', 'ffn_up', 'ffn_down', 'attn_qkv', 'attn_gate', 'ssm_alpha', 'ssm_beta', 'ssm_out'],
            ['ffn_gate', 'ffn_up', 'ffn_down', 'attn_q', 'attn_k', 'attn_v', 'attn_output'],
        ]
        for layer, group in enumerate(roles):
            weights = {f'blk.{layer}.{role}.weight': self.adapter.effective_weight(f'blk.{layer}.{role}.weight') for role in group}
            for name, weight in weights.items():
                np.testing.assert_array_equal(self.adapter.to_hf(name, weight), self.adapter.parameter(name).detach().numpy())
            capture = capture_block(self.adapter, self.tokens, layer)
            self.assertEqual(score_block(self.adapter, capture, weights)['output_mse'], 0)

    def test_genuine_ssm_nonlocal_time_dependence(self):
        capture = capture_block(self.adapter, self.tokens, 0)
        changed = capture.hidden_states.clone()
        changed[:, 0, :] += torch.linspace(-1, 1, changed.shape[-1])
        with torch.no_grad():
            output = self.adapter.text.layers[0](changed, **capture.kwargs)
        # Position kernel_size is outside the first token's convolution window:
        # only the recurrent state can carry this perturbation there. The random
        # fixture's strong decay washes it below F32 resolution by position 6.
        later = self.adapter.config.linear_conv_kernel_dim
        self.assertGreater((output[:, later] - capture.reference_output[:, later]).abs().max().item(), 0)

    def test_locked_buffers_and_execution_settings(self):
        adapter=self.adapter
        frequencies=adapter.text.rotary_emb.inv_freq
        original=frequencies.clone()
        try:
            frequencies.add_(0.01)
            with self.assertRaisesRegex(ValueError,'buffers'):adapter.assert_reference()
        finally:
            frequencies.copy_(original)
        threads=torch.get_num_threads()
        from unittest.mock import patch
        with patch.object(torch,'get_num_threads',return_value=threads+1):
            with self.assertRaisesRegex(ValueError,'settings'):adapter.assert_reference()
        adapter.assert_reference()

    def test_full_source_identity_rejects_changed_norm(self):
        from transformers import Qwen3_5TextModel
        model = Qwen3_5TextModel.from_pretrained(self.temp.name, local_files_only=True).float().eval()
        with torch.no_grad():
            model.norm.weight[0] += 0.1
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            Qwen35Adapter(model, Path(self.temp.name) / 'model.gguf')


if __name__ == '__main__':
    unittest.main()
