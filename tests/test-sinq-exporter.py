#!/usr/bin/env python3
"""CPU-only optional-dependency tests for real Linear pre-input capture."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sinq_exporter', ROOT / 'tools/sinq/export_activations.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
AVAILABLE = all(importlib.util.find_spec(x) is not None for x in ('numpy', 'torch'))


@unittest.skipUnless(AVAILABLE, 'optional numpy and torch required; exporter runtime not exercised')
class CaptureTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        self.np, self.torch = np, torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def sample(self, suffix, rows=8, name='blk.0.ffn_up.weight'):
        torch, np = self.torch, self.np
        layer = torch.nn.Linear(4, 3, bias=False)
        x = torch.arange(80, dtype=torch.float32).reshape(20, 4) / 16
        capture = exporter.ReservoirCapture(layer, name, self.root / suffix, rows, 19)
        y = layer(x)
        self.assertTrue(torch.equal(y, x @ layer.weight.T))
        count = capture.finish()
        data = np.fromfile(capture.path, dtype='<f4').reshape(count, 4)
        return data, capture

    def test_real_linear_reservoir_determinism_and_bounds(self):
        a, ca = self.sample('a')
        b, cb = self.sample('b')
        self.assertEqual(ca.seen, 20)
        self.assertEqual(ca.count, 8)
        self.assertEqual(ca.path.stat().st_size, 8 * 4 * 4)
        self.assertEqual(ca.path.read_bytes(), cb.path.read_bytes())
        for row in a:
            self.assertTrue(any(self.np.array_equal(row, candidate) for candidate in self.np.arange(80, dtype='f4').reshape(20, 4) / 16))
        c, _ = self.sample('c', name='blk.1.ffn_up.weight')
        self.assertFalse(self.np.array_equal(a, c))

    def test_hook_order_and_chunking_do_not_change_reservoir(self):
        torch = self.torch
        layer = torch.nn.Linear(4, 3, bias=False)
        x = torch.arange(80, dtype=torch.float32).reshape(20, 4)
        first = [exporter.ReservoirCapture(layer, name, self.root / ('a' + name), 5, 11) for name in ('one', 'two')]
        layer(x)
        for capture in first:
            capture.finish()
        second = [exporter.ReservoirCapture(layer, name, self.root / ('b' + name), 5, 11) for name in ('two', 'one')]
        for chunk in x.split(3):
            layer(chunk)
        for capture in second:
            capture.finish()
        for name in ('one', 'two'):
            self.assertEqual((self.root / ('a' + name)).read_bytes(), (self.root / ('b' + name)).read_bytes())

    def test_short_capture_has_no_trailer(self):
        a, capture = self.sample('short', rows=128)
        self.assertEqual(a.shape, (20, 4))
        self.assertEqual(capture.path.stat().st_size, 320)

    def test_permutation_and_input_matmul_parity(self):
        np = self.np
        config = types.SimpleNamespace(linear_num_key_heads=2, linear_num_value_heads=4, linear_value_head_dim=2, linear_key_head_dim=1)
        w = np.arange(24, dtype='f4').reshape(3, 8)
        x = np.arange(16, dtype='f4').reshape(2, 8)
        wp = exporter.converter_weight(w, 'ssm_out', config)
        xp = exporter.reorder_v(x, 1, 2, 4, 2)
        np.testing.assert_array_equal(x @ w.T, xp @ wp.T)
        np.testing.assert_array_equal(xp[0], [0, 1, 4, 5, 2, 3, 6, 7])
        for role, size in [('attn_gate', 8), ('ssm_alpha', 4), ('ssm_beta', 4), ('attn_qkv', 12)]:
            matrix = np.arange(size * 3, dtype='f4').reshape(size, 3)
            actual = exporter.converter_weight(matrix, role, config)
            order = [0, 2, 1, 3] if size == 4 else [0, 1, 4, 5, 2, 3, 6, 7]
            if role == 'attn_qkv':
                order = list(range(4)) + [v + 4 for v in order]
            np.testing.assert_array_equal(actual, matrix[order])

    def test_capture_permutation(self):
        torch, np = self.torch, self.np
        layer = torch.nn.Linear(8, 3, bias=False)
        capture = exporter.ReservoirCapture(layer, 'blk.0.ssm_out.weight', self.root / 'perm', 2, 0,
                                            lambda x: exporter.reorder_v(x, 0, 2, 4, 2))
        layer(torch.arange(8, dtype=torch.float32).reshape(1, 8))
        capture.finish()
        np.testing.assert_array_equal(np.fromfile(capture.path, dtype='<f4'), [0, 1, 4, 5, 2, 3, 6, 7])

    def test_packed_bf16(self):
        np = self.np
        expected = np.array([[1, 2, -3], [4, 5, 6]], dtype='f4')
        packed = (expected.view('u4') >> 16).astype('<u2').view('u1')
        tensor = types.SimpleNamespace(shape=[3, 2], n_elements=6, tensor_type=types.SimpleNamespace(name='BF16'), data=packed)
        np.testing.assert_array_equal(exporter.floating_tensor(tensor), expected)
        tensor.n_elements = 12
        with self.assertRaises(ValueError):
            exporter.floating_tensor(tensor)

    @unittest.skipUnless(importlib.util.find_spec('gguf') is not None, 'optional gguf required for manifest export test')
    def test_tiny_manifest_export(self):
        import argparse
        import json
        from unittest.mock import patch
        from gguf import GGUFWriter
        torch, np = self.torch, self.np

        class Language(torch.nn.Module):
            def __init__(self):
                super().__init__()
                block = torch.nn.Module()
                block.mlp = torch.nn.Module()
                block.mlp.up_proj = torch.nn.Linear(4, 3, bias=False).half()
                self.layers = torch.nn.ModuleList([block])

            def forward(self, input_ids, **kwargs):
                x = input_ids.to(torch.float16).unsqueeze(-1).expand(-1, -1, 4)
                return self.layers[0].mlp.up_proj(x)
        root = torch.nn.Module()
        root.model = torch.nn.Module()
        root.model.language_model = Language()
        checkpoint = self.root / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'tiny.safetensors').write_bytes(b'local-test-identity-only')
        (checkpoint / 'config.json').write_text('{}', encoding='utf-8')
        source = self.root / 'source.gguf'
        writer = GGUFWriter(str(source), 'qwen35')
        writer.add_tensor('blk.0.ffn_up.weight', root.model.language_model.layers[0].mlp.up_proj.weight.detach().float().numpy())
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        selection = self.root / 'selected.txt'
        selection.write_text('blk.0.ffn_up.weight\n', encoding='utf-8')
        tokens = self.root / 'tokens.json'
        tokens.write_text('[1,2,3,4,5]', encoding='utf-8')
        config = types.SimpleNamespace(model_type='qwen3_5', text_config=types.SimpleNamespace(vocab_size=10, pad_token_id=0), to_json_string=lambda: '{}')
        fake = types.ModuleType('transformers')
        fake.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *a, **kw: config)
        fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **kw: types.SimpleNamespace(pad_token_id=0))
        fake.Qwen3_5ForConditionalGeneration = types.SimpleNamespace(from_pretrained=lambda *a, **kw: root)
        args = argparse.Namespace(seed=3, rows=3, context=2, max_tokens=10, checkpoint=str(checkpoint), gguf=str(source), output=str(self.root / 'output'), tensor_list=str(selection), tokens=str(tokens), text=None, dtype='f16', device='cpu')
        with patch.dict('sys.modules', {'transformers': fake}):
            exporter.export(args)
            with self.assertRaises(FileExistsError):
                exporter.export(args)
            saved_output = args.output
            args.output = str(self.root / 'invalid-tokens')
            tokens.write_text('[0]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'invalid or padding token'):
                exporter.export(args)
            self.assertFalse(Path(args.output).exists())
            tokens.write_text('[1,2,3,4,5]', encoding='utf-8')
            args.output = str(self.root / 'weight-mismatch')
            with torch.no_grad():
                root.model.language_model.layers[0].mlp.up_proj.weight.add_(1)
            with self.assertRaisesRegex(ValueError, 'checkpoint/GGUF weight mismatch'):
                exporter.export(args)
            self.assertFalse(Path(args.output).exists())
            args.output = saved_output
        manifest = json.loads((Path(args.output) / 'activations.json').read_text(encoding='utf-8'))
        self.assertEqual(set(manifest), {'schema_version', 'architecture', 'source_sha256', 'source_revision', 'sampling', 'tensors', 'provenance'})
        record = manifest['tensors'][0]
        self.assertEqual(record['count'], 3)
        self.assertEqual(record['input_dim'], 4)
        self.assertEqual(record['site'], 'qwen3_5.layers.0.ffn_up.input')
        binary = Path(args.output) / record['data_file']
        self.assertEqual(binary.stat().st_size, 48)
        self.assertEqual(record['sha256'], exporter.sha256(binary))
        for row in np.fromfile(binary, dtype='<f4').reshape(3, 4):
            self.assertTrue(np.all(row == row[0]))
            self.assertIn(row[0], [1, 2, 3, 4, 5])

    def test_reject_malformed(self):
        torch = self.torch
        for seed, rows in [(-1, 1), (2**32, 1), (0, 0), (0, 257)]:
            with self.assertRaises(ValueError):
                exporter.bounds(seed, rows)
        with self.assertRaises(ValueError):
            exporter.module_for('token_embd.weight')
        with self.assertRaises(ValueError):
            exporter.reorder_v(self.np.zeros(8), 0, 3, 4, 2)
        layer = torch.nn.Linear(4, 3)
        capture = exporter.ReservoirCapture(layer, 'x', self.root / 'exists')
        with self.assertRaises(FileExistsError):
            exporter.ReservoirCapture(layer, 'x', capture.path)
        with self.assertRaises(ValueError):
            layer(torch.tensor([[float('nan'), 0, 0, 0]]))
        capture.handle.remove()
        capture.data._mmap.close()
        with self.assertRaises(ValueError):
            exporter.ReservoirCapture(torch.nn.Identity(), 'x', self.root / 'fused')


if __name__ == '__main__':
    unittest.main()
