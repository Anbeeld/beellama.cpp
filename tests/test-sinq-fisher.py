#!/usr/bin/env python3
"""Output-Fisher math oracles and offline genuine Qwen3.5 downstream VJPs."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['MKL_NUM_THREADS'] = '24'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gguf-py')]
import torch
from tools.sinq.research.fisher import build_probes, qwen_probes
from tools.sinq.research.capture import capture_block
from tools.sinq.research.blockwise import block_delta

torch.set_num_threads(24)


class FisherTests(unittest.TestCase):
    def graph(self):
        z = torch.tensor([[[0.2, -0.3]]], dtype=torch.float64, requires_grad=True)
        a = torch.tensor([[2., 1.], [-1., 3.], [0.5, -2.]], dtype=torch.float64)
        return z, a, z @ a.T

    def test_exact_mc_curvature_and_cross_terms(self):
        z, a, logits = self.graph()
        delta = torch.tensor([[[0.03, -0.02]]], dtype=z.dtype)
        probes = build_probes(logits, z, estimator='fisher_vjp_exact', metadata={'site': 'synthetic'})
        p = logits.detach().softmax(-1).reshape(-1)
        torch.testing.assert_close(probes.probabilities.reshape(-1), p, rtol=1e-14, atol=1e-14)
        expected_gradient = a[0] - (p[:, None] * a).sum(0)
        torch.testing.assert_close(probes.gradients[0].reshape(-1), expected_gradient, rtol=1e-14, atol=1e-14)
        direction = (delta @ a.T).reshape(-1)
        expected = 0.5 * ((p * direction.square()).sum() - (p * direction).sum().square())
        self.assertAlmostEqual(probes.score(delta), expected.item(), places=14)
        self.assertEqual(probes.score(torch.zeros_like(delta)), 0)
        self.assertGreater(probes.score(delta), 0)
        for eps in [0.01, 0.001]:
            shifted = (z.detach() + eps * delta) @ a.T
            kl = (p * (p.log() - shifted.log_softmax(-1).reshape(-1))).sum()
            self.assertAlmostEqual((kl / eps**2).item(), expected.item(), delta=expected.item() * 0.001)
        z, a, logits = self.graph()
        mc = build_probes(logits, z, probes=12000, seed=41, max_vjps=12000, max_cache_bytes=1000000)
        self.assertAlmostEqual(mc.score(delta), expected.item(), delta=expected.item() * 0.04)
        z, a, logits = self.graph()
        again = build_probes(logits, z, probes=12000, seed=41, max_vjps=12000, max_cache_bytes=1000000)
        self.assertEqual(mc.score(delta), again.score(delta))
        x = delta.clone(); x[..., 1] = 0
        y = delta.clone(); y[..., 0] = 0
        self.assertGreater(abs(probes.score(delta) - probes.score(x) - probes.score(y)), 1e-5)

    def test_rank_differs_from_l2_and_budgets(self):
        z = torch.zeros(1, 1, 2, dtype=torch.float64, requires_grad=True)
        logits = torch.stack([z[..., 0] * 10, -z[..., 0] * 10], dim=-1)
        cache = build_probes(logits, z, estimator='fisher_vjp_exact')
        sensitive = torch.tensor([[[0.1, 0.]]], dtype=z.dtype)
        null = torch.tensor([[[0., 1.]]], dtype=z.dtype)
        self.assertLess(sensitive.square().sum(), null.square().sum())
        self.assertGreater(cache.score(sensitive), cache.score(null))
        z, _, logits = self.graph()
        with self.assertRaises(ValueError):
            build_probes(logits, z, probes=3, max_vjps=2)
        with self.assertRaises(ValueError):
            build_probes(logits, z, max_cache_bytes=1)
        with self.assertRaises(ValueError):
            build_probes(logits, z, estimator='nll')
        with self.assertRaises(ValueError):
            cache.score(torch.ones(1))
        z, _, logits = self.graph()
        with self.assertRaises(ValueError):
            build_probes(logits, z, metadata={'estimator': 'nll'})
        z, _, logits = self.graph()
        with self.assertRaises(ValueError):
            build_probes(logits, z, positions=[(0, 0), (0, 0)])

    def test_recurrent_later_token_effects(self):
        z = torch.zeros(1, 3, 1, dtype=torch.float64, requires_grad=True)
        state = torch.zeros(1, 1, dtype=z.dtype)
        outputs = []
        for t in range(3):
            state = 0.7 * state + z[:, t]
            outputs.append(torch.cat([state, -state], -1))
        logits = torch.stack(outputs, 1)
        cache = build_probes(logits, z, estimator='fisher_vjp_exact', positions=[(0, 2)])
        d = torch.zeros_like(z); d[:, 0] = 1
        self.assertAlmostEqual(cache.score(d), 0.5 * 0.7**4, places=14)
        self.assertGreater(cache.score(d), 0)

    def test_masked_output_analytic_full_site_pairing(self):
        block = torch.nn.Identity()
        mask = torch.tensor([[1, 0, 1], [0, 1, 0]])
        hidden = torch.zeros(2, 3, 1, dtype=torch.float64)
        capture = SimpleNamespace(validate=lambda adapter: None, layer=0,
            tokens=torch.zeros(2, 3, dtype=torch.long), attention_mask=mask,
            position_ids=None, hidden_states=hidden, kwargs={}, reference_output=hidden,
            context_id='toy-mask-bound-context', context='reference_inputs', baseline_sha256='toy-baseline')
        def logits(tokens, **kwargs):
            site = block(hidden)
            state = torch.zeros(2, 1, dtype=site.dtype)
            outputs = []
            for t in range(3):
                state = 0.7 * state + site[:, t]
                outputs.append(torch.cat([state, -state], -1))
            return torch.stack(outputs, 1)
        adapter = SimpleNamespace(text=SimpleNamespace(layers=[block]), logits=logits,
            source_id='toy', identity={'source_gguf_sha256': 'toy-source'})
        probes = qwen_probes(adapter, capture, estimator='fisher_vjp_exact')
        self.assertEqual(probes.metadata['positions'], [[0, 0], [0, 2], [1, 1]])
        self.assertEqual(probes.metadata['requested_vjps'], 6)
        self.assertEqual(probes.metadata['output_position_selection'], 'capture-attention-mask')
        self.assertEqual(probes.metadata['output_position_mask_context_id'], capture.context_id)
        delta = torch.zeros_like(hidden); delta[0, 0] = 1
        self.assertAlmostEqual(probes.score(delta), 0.5 * (1 + 0.7**4) / 3, places=14)
        # A masked site coordinate still affects a later valid output: mask only outputs.
        delta.zero_(); delta[0, 1] = 1
        self.assertAlmostEqual(probes.score(delta), 0.5 * 0.7**2 / 3, places=14)
        repeat = qwen_probes(adapter, capture, estimator='fisher_vjp_exact')
        self.assertEqual(probes.metadata, repeat.metadata)
        torch.testing.assert_close(probes.gradients, repeat.gradients, rtol=0, atol=0)
        selected = qwen_probes(adapter, capture, estimator='fisher_vjp_exact', positions=[(1, 1), (0, 2)])
        self.assertEqual(selected.metadata['positions'], [[1, 1], [0, 2]])
        self.assertEqual(selected.metadata['output_position_selection'], 'explicit')
        for positions in [[(0, 1)], [(0, 0), (1, 2)], [(2, 0)], [(0, 3)], [(-1, 0)], []]:
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                qwen_probes(adapter, capture, positions=positions)
        capture.attention_mask = torch.zeros_like(mask)
        with self.assertRaises(ValueError):
            qwen_probes(adapter, capture)
        capture.attention_mask = None
        unmasked = qwen_probes(adapter, capture, estimator='fisher_vjp_exact')
        self.assertEqual(unmasked.metadata['positions'], [[i, j] for i in range(2) for j in range(3)])
        self.assertEqual(unmasked.metadata['output_position_selection'], 'all-batch-time')
        self.assertIsNone(unmasked.metadata['output_position_mask_context_id'])

    def test_genuine_qwen_masked_outputs(self):
        spec = importlib.util.spec_from_file_location('block_test', ROOT / 'tests/test-sinq-blockwise.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            adapter = module.tiny_adapter(Path(temp))
            tokens = torch.tensor([[1, 7, 3, 11], [5, 19, 2, 0]])
            mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
            captured = capture_block(adapter, tokens, 0, attention_mask=mask)
            probes = qwen_probes(adapter, captured, probes=3, seed=8)
            self.assertEqual(probes.metadata['positions'], [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1]])
            self.assertEqual(probes.metadata['requested_vjps'], 18)
            self.assertEqual(probes.metadata['output_position_mask_context_id'], captured.context_id)
            repeat = qwen_probes(adapter, captured, probes=3, seed=8)
            self.assertEqual(probes.metadata, repeat.metadata)
            torch.testing.assert_close(probes.gradients, repeat.gradients, rtol=0, atol=0)
            late = qwen_probes(adapter, captured, probes=3, seed=8, positions=[(0, 3)])
            self.assertGreater(late.gradients[:, 0, 0].abs().max().item(), 0)
            for positions in [[(1, 2)], [(0, 0), (1, 3)], [(2, 0)], [(0, 4)], []]:
                with self.subTest(positions=positions), self.assertRaises(ValueError):
                    qwen_probes(adapter, captured, positions=positions)
            empty = capture_block(adapter, tokens, 0, attention_mask=torch.zeros_like(mask))
            with self.assertRaises(ValueError):
                qwen_probes(adapter, empty)

    def test_genuine_qwen_downstream_site(self):
        spec = importlib.util.spec_from_file_location('block_test', ROOT / 'tests/test-sinq-blockwise.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            adapter = module.tiny_adapter(Path(temp))
            tokens = torch.tensor([[1, 7, 3, 11, 5, 19, 2]])
            for layer, name in [(0, 'blk.0.ssm_out.weight'), (1, 'blk.1.attn_output.weight')]:
                captured = capture_block(adapter, tokens, layer)
                weight = adapter.effective_weight(name).copy(); weight[0] += 0.05
                delta = block_delta(adapter, captured, {name: weight})
                probes = qwen_probes(adapter, captured, probes=3, seed=8, positions=[(0, 6)])
                value = probes.score(delta)
                self.assertGreater(value, 0)
                if layer == 0:
                    self.assertGreater(probes.gradients[:, :, 0].abs().max().item(), 0)
                self.assertEqual(probes.metadata['site'], f'block.{layer}.output')
                self.assertEqual(probes.metadata['context_id'], captured.context_id)
                repeat = qwen_probes(adapter, captured, probes=3, seed=8, positions=[(0, 6)])
                self.assertEqual(value, repeat.score(delta))
                self.assertEqual(probes.score(torch.zeros_like(delta)), 0)


if __name__ == '__main__':
    unittest.main()
