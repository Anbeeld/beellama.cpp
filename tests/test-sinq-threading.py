"""Deterministic thread-budget propagation at native and torch boundaries."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'gguf-py'))
import numpy as np
from tools.sinq.optimizer import stages as cli, kl, native_codec, schemas
from tools.sinq import export_activations as exporter


def load_test(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ThreadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / 'binary'
        self.binary.write_bytes(b'synthetic')
        self.candidate = types.SimpleNamespace(
            provenance={'shape': [2, 4], 'qtype': 'Q4_K', 'input_sha256': 'weights'},
            payload=self.root / 'payload', scales=None)

    def assert_env(self, env, threads):
        self.assertEqual(env['OMP_NUM_THREADS'], str(threads))
        self.assertEqual(env['MKL_NUM_THREADS'], str(threads))

    def test_request_threads_reach_json_and_subprocess_environment(self):
        with patch.object(native_codec.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout='{}')) as run:
            native_codec.request(self.binary, {'threads': 7}, self.root)
        self.assertEqual(json.loads((self.root / 'request.json').read_text())['threads'], 7)
        self.assert_env(run.call_args.kwargs['env'], 7)

    def test_bridge_defaults_to_all_logical_cpus(self):
        with patch.object(native_codec.os, 'cpu_count', return_value=5), patch.object(native_codec.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout='{}')) as run:
            native_codec.request(self.binary, {}, self.root)
        self.assertEqual(json.loads((self.root / 'request.json').read_text())['threads'], 5)
        self.assert_env(run.call_args.kwargs['env'], 5)

    def test_validation_threads_reach_environment(self):
        with patch.object(native_codec.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout='{}')) as run:
            native_codec.validate_activations(self.binary, 'manifest', 'source', threads=7)
        self.assert_env(run.call_args.kwargs['env'], 7)

    def test_decode_threads_reach_request(self):
        def request(binary, config, directory):
            self.assertEqual(config['threads'], 7)
            np.zeros((2, 4), dtype='<f4').tofile(config['output'])
        with patch.object(native_codec, 'request', side_effect=request):
            self.assertEqual(native_codec.decode(self.binary, self.candidate, threads=7).shape, (2, 4))

    def test_rescore_threads_reach_request(self):
        tensor = types.SimpleNamespace(shape=[4, 2], tensor_type=types.SimpleNamespace(name='F32'))
        with patch.object(native_codec, 'copy_tensor_bytes'), patch.object(native_codec, 'file_sha256', return_value='weights'), patch.object(native_codec, 'request', return_value={}) as request:
            native_codec.rescore(self.binary, self.candidate, tensor, {'data_file': 'capture', 'count': 2}, threads=7)
        self.assertEqual(request.call_args.args[1]['threads'], 7)

    def test_kl_threads_bind_policy_flags_and_environment(self):
        evaluator = kl.NativeKLEvaluator(self.binary, threads=7)
        self.assertEqual(evaluator.policy['threads'], 7)
        with patch.object(kl.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)) as run:
            evaluator._run('model', 'corpus', 'reference', 'tokens')
        args = run.call_args.args[0]
        for flag in ('-t', '-tb'):
            self.assertEqual(args[args.index(flag) + 1], '7')
        self.assert_env(run.call_args.kwargs['env'], 7)
        with patch.object(kl.os, 'cpu_count', return_value=5):
            self.assertEqual(kl.NativeKLEvaluator(self.binary).policy['threads'], 5)

    def test_kl_rejects_invalid_threads(self):
        for threads in (0, -1, 1025, True, 1.5):
            with self.subTest(threads=threads), self.assertRaises(ValueError):
                kl.NativeKLEvaluator(self.binary, threads=threads)

    def test_schema_thread_bounds(self):
        fixture = load_test('test-sinq-optimizer-contract')
        for threads in (1, 1024):
            config = fixture.config(); config['execution'] = {'threads':threads}
            self.assertEqual(schemas.validate_config(config,'generate')['threads'], threads)
        for threads in (0, -1, 1025, True, 1.5):
            config = fixture.config(); config['execution'] = {'threads':threads}
            with self.subTest(threads=threads), self.assertRaises(ValueError):
                schemas.validate_config(config,'generate')

    def test_loader_threads_reach_environment(self):
        with patch.object(cli.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)) as run:
            cli._validator(self.binary, threads=7)('hybrid')
        self.assert_env(run.call_args.kwargs['env'], 7)

    def test_cli_local_scoring_forwards_threads(self):
        candidate = types.SimpleNamespace(provenance={'generator': {'generator': 'sinq','strip_width':64,'iterations':4,'strength':.75}})
        tensor = types.SimpleNamespace(name='weight')
        reader = types.SimpleNamespace(tensors=[tensor], data=types.SimpleNamespace(_mmap=Mock()))
        config = dict(scorer_name='local_output_l1',method='local', budgets={'workspace_bytes': 1024, 'max_output_bytes': 4096}, native_bridge=str(self.binary), capture_manifest='capture', validation_manifest='validation', pairing_manifest='pair', selection_policy='held_out', objective='output_l1', candidate_manifest='manifest', threads=7, device='cpu')
        with patch.object(cli, '_stock_baseline'), patch.object(cli, '_json', return_value={}), patch.object(cli, 'validate_pairing'), patch.object(cli, 'candidate_for', return_value=candidate), patch.object(cli, 'file_digest', return_value='a'*64), patch.object(cli, '_new_output', return_value=self.root), patch.object(cli, 'atomic_write'), patch('gguf.GGUFReader', return_value=reader), patch.object(native_codec, 'validate_activations', return_value={'tensors': {'weight': {}}}) as validate, patch.object(native_codec, 'rescore', return_value={'train_score':1.0,'validation_score':2.0}) as rescore:
            cli._score(config, {'source_sha256':'a'*64,'tensors': {'weight': {'stock': 'id', 'candidates': []}}}, None, 'source')
        self.assertEqual(validate.call_count, 4)
        for call in validate.call_args_list:
            self.assertEqual(call.kwargs['threads'], 7)
        self.assertEqual(rescore.call_args.kwargs['threads'], 7)

    def test_export_cli_threads_option_and_default(self):
        base = ['export', '--checkpoint', 'checkpoint', '--gguf', 'source', '--tensor-list', 'list', '--tokens', 'tokens', '--output', 'out']
        with patch.object(sys, 'argv', base + ['--threads', '7']), patch.object(exporter, 'export') as export:
            exporter.main()
            self.assertEqual(export.call_args.args[0].threads, 7)
        with patch.object(sys, 'argv', base), patch.object(exporter.os, 'cpu_count', return_value=5), patch.object(exporter, 'export') as export:
            exporter.main()
            self.assertEqual(export.call_args.args[0].threads, 5)

    def test_export_sets_threads_before_model_work_and_legacy_namespace(self):
        torch = types.ModuleType('torch'); torch.set_num_threads = Mock()
        transformers = types.ModuleType('transformers')
        transformers.AutoConfig = Mock(); transformers.AutoTokenizer = Mock()
        transformers.Qwen3_5ForConditionalGeneration = Mock()
        for threads in (7, None):
            args = argparse.Namespace(seed=0, rows=1, context=1, max_tokens=1, output=str(self.root / 'out'), checkpoint=str(self.root / 'missing'), gguf='missing')
            if threads is not None: args.threads = threads
            torch.set_num_threads.reset_mock()
            with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), patch.object(exporter.os, 'cpu_count', return_value=5):
                with self.assertRaisesRegex(ValueError, 'must exist locally'):
                    exporter.export(args)
            torch.set_num_threads.assert_called_once_with(7 if threads else 5)
            transformers.AutoConfig.from_pretrained.assert_not_called()

    def test_export_rejects_invalid_threads(self):
        torch = types.ModuleType('torch'); torch.set_num_threads = Mock()
        transformers = types.ModuleType('transformers')
        transformers.AutoConfig = Mock(); transformers.AutoTokenizer = Mock(); transformers.Qwen3_5ForConditionalGeneration = Mock()
        for threads in (0, -1, 1025, True, 1.5):
            with self.subTest(threads=threads), patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), self.assertRaisesRegex(ValueError, 'threads'):
                exporter.export(argparse.Namespace(threads=threads))
        torch.set_num_threads.assert_not_called()


if __name__ == '__main__':
    unittest.main()
