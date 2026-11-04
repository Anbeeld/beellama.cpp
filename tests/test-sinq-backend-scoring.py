"""Precision/backend regression oracles independent of a model checkpoint."""
import sys
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sinq.research.capture import _digest
from tools.sinq.research.fisher import build_probes
from tools.sinq.research.blockwise import block_delta
from contextlib import nullcontext
from types import SimpleNamespace

torch.set_num_threads(32)

class BackendScoringTests(unittest.TestCase):
    def test_bf16_hash_preserves_storage(self):
        x = torch.tensor([1., 1.0078125], dtype=torch.bfloat16)
        self.assertEqual(_digest(x), _digest(x.clone()))
        self.assertNotEqual(_digest(x), _digest(x.float()))

    def test_delta_promotes_before_subtraction(self):
        reference = torch.tensor([1.], dtype=torch.bfloat16)
        actual = torch.tensor([-.00390625], dtype=torch.bfloat16)
        capture = SimpleNamespace(validate=lambda a: None, layer=0, hidden_states=reference,
                                  kwargs={}, reference_output=reference)
        adapter = SimpleNamespace(swapped=lambda w: nullcontext(), text=SimpleNamespace(layers=[lambda *a, **kw: actual]))
        delta = block_delta(adapter, capture, {})
        self.assertEqual(delta.dtype, torch.float32)
        self.assertEqual(delta.item(), actual.float().item() - reference.float().item())

    def test_bf16_log_normalization_and_cpu_seed(self):
        z = torch.tensor([[[.5, -.5]]], dtype=torch.bfloat16, requires_grad=True)
        logits = z @ torch.tensor([[1., 2., -1.], [2., 1., 0.]], dtype=z.dtype)
        probes = build_probes(logits, z, probes=4, seed=17)
        expected = logits.detach().float().log_softmax(-1).double().exp().reshape(1, -1)
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(probes.probabilities, expected, rtol=0, atol=0)
        self.assertEqual(probes.gradients.dtype, torch.float64)
        self.assertEqual(probes.gradients.device.type, 'cpu')
        self.assertGreater(probes.gradients.abs().max(), 0)

if __name__ == '__main__': unittest.main()
