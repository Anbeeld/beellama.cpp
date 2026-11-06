"""Invocation-scoped SINQ telemetry preserves hashing and assembly contracts."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'gguf-py'))
import gguf
import numpy as np
from tools.sinq.optimizer import assembly, cli, implementation, resources
from tools.sinq.optimizer.cache import CandidateStore


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = self.root / 'input'
        self.file.write_bytes(b'content' * 100)

    def assert_metrics(self, metrics):
        self.assertTrue(metrics)
        for phase in metrics.values():
            self.assertGreater(phase['calls'], 0)
            self.assertIsInstance(phase['calls'], int)
            self.assertTrue(math.isfinite(phase['seconds']))
            self.assertGreaterEqual(phase['seconds'], 0)
            self.assertIsInstance(phase['bytes_hashed'], int)
            self.assertGreaterEqual(phase['bytes_hashed'], 0)
        json.dumps(metrics, allow_nan=False)

    def test_hash_workers_match_unobserved_and_count_all_bytes(self):
        paths = {str(i): self.file for i in range(12)}
        tensors = [(str(i), np.arange(256, dtype=np.float32)) for i in range(12)]
        expected_files = resources.file_digests(paths, threads=3)
        expected_tensors = resources.tensor_hashes(tensors, threads=3)
        with resources.observe_performance() as metrics:
            self.assertEqual(resources.file_digests(paths, threads=3), expected_files)
            self.assertEqual(resources.tensor_hashes(tensors, threads=3), expected_tensors)
        self.assertEqual(metrics['file_hash']['calls'], 12)
        self.assertEqual(metrics['file_hash']['bytes_hashed'], 12 * self.file.stat().st_size)
        self.assertEqual(metrics['tensor_hash']['calls'], 12)
        self.assertEqual(metrics['tensor_hash']['bytes_hashed'], sum(a.nbytes for _, a in tensors))
        self.assertEqual(expected_files['0'], hashlib.sha256(self.file.read_bytes()).hexdigest())
        self.assert_metrics(metrics)

    def test_disabled_observation_does_not_read_clock_and_baseline_hash_matches(self):
        data = np.arange(2048, dtype='<f2')
        with patch.object(resources, 'perf_counter', side_effect=AssertionError('observation disabled')):
            resources.file_digest(self.file)
            actual = assembly._bytes_hash(data)
        with resources.observe_performance() as metrics:
            self.assertEqual(assembly._bytes_hash(data), actual)
        self.assertEqual(actual, hashlib.sha256(data.tobytes()).hexdigest())
        self.assertEqual(metrics['tensor_hash']['bytes_hashed'], data.nbytes)
        self.assert_metrics(metrics)

    def test_nested_exception_reset_and_independent_invocations(self):
        with resources.observe_performance() as outer:
            resources.file_digest(self.file)
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                with resources.observe_performance() as inner:
                    resources.file_digest(self.file)
                    raise RuntimeError('stop')
            resources.file_digest(self.file)
        saved = json.dumps(outer, sort_keys=True)
        resources.file_digest(self.file)
        self.assertEqual(json.dumps(outer, sort_keys=True), saved)
        self.assertEqual(outer['file_hash']['calls'], 2)
        self.assertEqual(inner['file_hash']['calls'], 1)
        def run(count):
            with resources.observe_performance() as metrics:
                resources.file_digests({str(i): self.file for i in range(count)}, threads=3)
            return metrics
        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = list(pool.map(run, (5, 9)))
        self.assertEqual(first['file_hash']['calls'], 5)
        self.assertEqual(second['file_hash']['calls'], 9)
        with resources.observe_performance() as fresh:
            pass
        self.assertEqual(fresh, {})

    def test_fingerprints_keep_identity_and_measure_bytes(self):
        for tree in implementation._TREES:
            directory = self.root / tree
            directory.mkdir(parents=True)
            (directory / 'fixture.py').write_bytes(b'# fixture\n')
        for name in implementation._FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'# entry\n')
        with patch.object(implementation, '_versions', return_value={'fixture': '1'}):
            expected = implementation.implementation_fingerprint(self.root)
            with resources.observe_performance() as metrics:
                self.assertEqual(implementation.implementation_fingerprint(self.root), expected)
                actual = implementation.native_runtime_fingerprint({'loader': self.file})
        self.assertEqual(actual['loader']['executable'], hashlib.sha256(self.file.read_bytes()).hexdigest())
        self.assertEqual(metrics['implementation_fingerprint']['calls'], 1)
        self.assertEqual(metrics['native_runtime_fingerprint']['calls'], 1)
        self.assertEqual(metrics['file_hash']['bytes_hashed'], self.file.stat().st_size + 3 * 10 + 2 * 8)
        self.assert_metrics(metrics)

    def source(self):
        path = self.root / 'source.gguf'
        writer = gguf.GGUFWriter(path, 'qwen35')
        writer.add_tensor('weight', np.ones((2, 256), dtype='<f2'))
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
        return path

    def test_assembly_output_and_integrity_checks_unchanged(self):
        source = self.source()
        plain, observed = self.root / 'plain.gguf', self.root / 'observed.gguf'
        expected = assembly.assemble(source, plain, None, {}, native_validate=lambda _: None, threads=2)
        with resources.observe_performance() as metrics:
            actual = assembly.assemble(source, observed, None, {}, native_validate=lambda _: None, threads=2)
        self.assertEqual(actual, expected)
        self.assertEqual(observed.read_bytes(), plain.read_bytes())
        for phase in ('source_hash', 'cache_get', 'metadata', 'write', 'readback', 'native_load', 'final_integrity'):
            self.assertIn('assembly_' + phase, metrics)
        self.assert_metrics(metrics)
        unpublished = self.root / 'unpublished.gguf'
        def mutate(_):
            with source.open('ab') as stream:
                stream.write(b'changed')
        with resources.observe_performance() as failed, self.assertRaisesRegex(ValueError, 'changed during native validation'):
            assembly.assemble(source, unpublished, None, {}, native_validate=mutate)
        self.assertFalse(unpublished.exists())
        self.assertIn('assembly_final_integrity', failed)

    def test_observed_candidate_checks_and_failure_publication(self):
        source = self.source()
        h = hashlib.sha256(b'fixture').hexdigest()
        provenance = dict(source_sha256=resources.file_digest(source), tensor_name='weight', shape=[2, 256],
            source_qtype='F16', qtype='Q5_0', input_sha256=hashlib.sha256(np.ones((2, 256), dtype='<f2').tobytes()).hexdigest(),
            recipe_sha256=h, imatrix_sha256=None, codec={'implementation': 'ggml', 'chunk_rows': 2},
            sinq={'strip_width': 64, 'iterations': 16, 'strength': .75, 'algorithm': 'sinkhorn_log', 'numerics': 'f32-f16'},
            generator={'generator': 'sinq', 'strip_width': 64, 'iterations': 16, 'strength': .75}, activation=None,
            producer={'source_sha256': h, 'binary_sha256': h})
        store = CandidateStore(self.root / 'cache')
        key = store.put(provenance, bytes(352), np.ones(256, dtype='<f2').tobytes())
        plain, observed = self.root / 'candidate-plain.gguf', self.root / 'candidate-observed.gguf'
        expected = assembly.assemble(source, plain, store, {'weight': key}, native_validate=lambda _: None)
        with resources.observe_performance() as metrics, patch.object(store, 'get', wraps=store.get) as get:
            actual = assembly.assemble(source, observed, store, {'weight': key}, native_validate=lambda _: None)
        self.assertEqual(get.call_count, 3)  # initial, pre-native, fresh post-native
        self.assertEqual(actual, expected)
        self.assertEqual(plain.read_bytes(), observed.read_bytes())
        self.assertEqual(metrics['assembly_cache_get']['calls'], 3)
        self.assert_metrics(metrics)
        unpublished = self.root / 'over-budget.gguf'
        with resources.observe_performance(), self.assertRaisesRegex(ValueError, 'budget'):
            assembly.assemble(source, unpublished, store, {'weight': key}, native_validate=lambda _: None, max_output_bytes=1)
        self.assertFalse(unpublished.exists())
        unpublished = self.root / 'native-failed.gguf'
        def fail(_):
            raise RuntimeError('native failure')
        with resources.observe_performance(), self.assertRaisesRegex(RuntimeError, 'native failure'):
            assembly.assemble(source, unpublished, store, {'weight': key}, native_validate=fail)
        self.assertFalse(unpublished.exists())
        unpublished = self.root / 'corrupt.gguf'
        def corrupt(_):
            # In-place byte mutation works while assembly holds a Windows mapping.
            with store.get(key).payload.open('r+b') as stream:
                stream.write(b'X')
        with resources.observe_performance(), self.assertRaises(ValueError):
            assembly.assemble(source, unpublished, store, {'weight': key}, native_validate=corrupt)
        self.assertFalse(unpublished.exists())

    def test_atomic_write_readback_and_collision_preserved(self):
        output = self.root / 'artifact'
        data = b'artifact bytes'
        with resources.observe_performance() as metrics:
            resources.atomic_write(output, data, max_bytes=len(data))
            resources.atomic_write(output, data, max_bytes=len(data))
            with self.assertRaisesRegex(ValueError, 'collision'):
                resources.atomic_write(output, b'x' * len(data), max_bytes=len(data))
        self.assertEqual(output.read_bytes(), data)
        self.assertEqual(metrics['file_hash']['bytes_hashed'], len(data) * 4)
        self.assert_metrics(metrics)

    def test_cli_includes_stage_and_summary_hashes_in_observation(self):
        config = {'source_gguf': str(self.file)}
        def preflight(_):
            resources.file_digest(self.file)
            return {}
        with patch.object(cli, 'load_config', return_value=config), patch('tools.sinq.optimizer.generate.preflight', side_effect=preflight), patch.object(resources, 'peak_rss_bytes', return_value=0), patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(cli.main(['plan', '--config', 'unused']), 0)
        summary = json.loads(output.getvalue())['stage_summary']
        self.assertEqual(summary['performance']['file_hash']['calls'], 2)
        self.assertEqual(summary['input_hashes']['source_gguf'], hashlib.sha256(self.file.read_bytes()).hexdigest())
        self.assert_metrics(summary['performance'])


if __name__ == '__main__':
    unittest.main()
