"""CPU-only full-text identity regressions, including unselected upstream weights."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gguf-py'))
spec = importlib.util.spec_from_file_location('identity_exporter', ROOT / 'tools/sinq/export_activations.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        from gguf import GGUFReader, GGUFWriter
        self.np, self.torch = np, torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        torch.manual_seed(41)
        self.model = torch.nn.Module()
        self.model.model = torch.nn.Module()
        text = torch.nn.Module()
        self.model.model.language_model = text
        text.embed_tokens = torch.nn.Embedding(8, 4)
        text.norm = torch.nn.Module()
        text.norm.weight = torch.nn.Parameter(torch.randn(4) / 10)
        block = torch.nn.Module()
        block.input_layernorm = torch.nn.Module()
        block.input_layernorm.weight = torch.nn.Parameter(torch.randn(4) / 10)
        block.mlp = torch.nn.Module()
        block.mlp.up_proj = torch.nn.Linear(4, 6, bias=False)
        text.layers = torch.nn.ModuleList([block])
        self.model.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.config = types.SimpleNamespace(num_hidden_layers=1, tie_word_embeddings=False)
        path = Path(self.temp.name) / 'source.gguf'
        writer = GGUFWriter(str(path), 'qwen35')
        # Independent converter convention: regular RMS norm weights store 1+w.
        for name, value in {
            'token_embd.weight': text.embed_tokens.weight,
            'output_norm.weight': text.norm.weight + 1,
            'blk.0.attn_norm.weight': block.input_layernorm.weight + 1,
            'blk.0.ffn_up.weight': block.mlp.up_proj.weight,
            'output.weight': self.model.lm_head.weight,
        }.items():
            writer.add_tensor(name, value.detach().numpy().copy())
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        self.reader = GGUFReader(str(path))
        self.addCleanup(self.reader.data._mmap.close)
        self.tensors = {t.name: t for t in self.reader.tensors}

    def verify(self):
        return exporter.verify_reference_identity(self.model, self.config, self.tensors)

    def test_complete_identity(self):
        result = self.verify()
        self.assertEqual(result['verified_tensor_count'], 5)
        self.assertEqual(result['contract'], 'qwen35-full-text-storage-v1')
        self.assertEqual(len(result['verified_tensors_sha256']), 64)

    def test_streamed_rows_and_workspace_rejection(self):
        reference = self.verify()
        small = exporter.verify_reference_identity(self.model, self.config, self.tensors, workspace_limit=128)
        self.assertEqual(reference['verified_tensors_sha256'], small['verified_tensors_sha256'])
        with self.assertRaisesRegex(ValueError, 'workspace'):
            exporter.verify_reference_identity(self.model, self.config, self.tensors, workspace_limit=64)

    def test_graph_configuration_mismatch(self):
        field = types.SimpleNamespace(contents=lambda: float(self.np.float32(1e-6)))
        reader = types.SimpleNamespace(fields={'qwen35.attention.layer_norm_rms_epsilon': field})
        config = types.SimpleNamespace(rms_norm_eps=1e-6)
        self.assertEqual(len(exporter.verify_reference_config(reader, config)), 1)
        config.rms_norm_eps = 1e-3
        with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
            exporter.verify_reference_config(reader, config)

    def test_real_converter_complete_text_graph(self):
        try:
            from transformers import Qwen3_5TextModel
        except ImportError:
            self.skipTest('optional Qwen3.5 Transformers implementation unavailable')
        from gguf import GGUFReader
        spec = importlib.util.spec_from_file_location('identity_parity_fixture', ROOT / 'tests/test-sinq-activation-parity.py')
        parity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parity)
        directory = Path(self.temp.name) / 'actual-qwen35'
        parity.fixture(directory, 'f32')
        text = Qwen3_5TextModel.from_pretrained(directory, local_files_only=True).cpu().eval()
        wrapper = self.torch.nn.Module()
        wrapper.model = self.torch.nn.Module()
        wrapper.model.language_model = text
        reader = GGUFReader(str(directory / 'model.gguf'))
        self.addCleanup(reader.data._mmap.close)
        tensors = {t.name: t for t in reader.tensors}
        self.assertTrue(exporter.verify_reference_config(reader, text.config))
        result = exporter.verify_reference_identity(wrapper, text.config, tensors)
        self.assertEqual(result['verified_tensor_count'], len(tensors))
        self.assertGreater(result['verified_tensor_count'], 20)

    def test_strict_execution_rejects_storage_rounded_only_norm(self):
        from gguf import GGMLQuantizationType
        from gguf.quants import quantize
        expected=(self.model.model.language_model.norm.weight+1).detach().numpy()
        self.tensors['output_norm.weight']=types.SimpleNamespace(tensor_type=GGMLQuantizationType.BF16,
            shape=self.np.array([4]),n_elements=4,data=quantize(expected,GGMLQuantizationType.BF16))
        # Legacy exporter comparison still intentionally applies storage rounding.
        self.assertEqual(self.verify()['verified_tensor_count'],5)
        with self.assertRaisesRegex(ValueError,'execution storage'):
            exporter.verify_reference_identity(self.model,self.config,self.tensors,require_exact_execution_storage=True)

    def test_unselected_embedding_mismatch(self):
        selected = self.model.model.language_model.layers[0].mlp.up_proj.weight.detach().clone()
        with self.torch.no_grad():
            self.model.model.language_model.embed_tokens.weight[0, 0] += .5
        self.assertTrue(self.torch.equal(selected, self.model.model.language_model.layers[0].mlp.up_proj.weight))
        with self.assertRaisesRegex(ValueError, 'token_embd.weight'):
            self.verify()

    def test_unselected_normalization_mismatch(self):
        with self.torch.no_grad():
            self.model.model.language_model.layers[0].input_layernorm.weight[0] += .5
        with self.assertRaisesRegex(ValueError, 'blk.0.attn_norm.weight'):
            self.verify()

    def test_missing_or_extra_source_tensor(self):
        tensor = self.tensors.pop('token_embd.weight')
        with self.assertRaisesRegex(ValueError, 'missing.*token_embd.weight'):
            self.verify()
        self.tensors['token_embd.weight'] = tensor
        self.tensors['unmapped.weight'] = tensor
        with self.assertRaisesRegex(ValueError, 'unverified.*unmapped.weight'):
            self.verify()

    def test_unknown_text_parameter_and_nonfinite(self):
        text = self.model.model.language_model
        text.extra = self.torch.nn.Parameter(self.torch.ones(4))
        with self.assertRaisesRegex(ValueError, 'unmapped.*extra'):
            self.verify()
        del text.extra
        with self.torch.no_grad():
            text.norm.weight[0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.verify()


if __name__ == '__main__':
    if any(importlib.util.find_spec(name) is None for name in ('torch', 'numpy')):
        sys.stdout.write('SKIP: optional torch/numpy needed for reference identity checks\n')
        sys.exit(77)
    unittest.main()
