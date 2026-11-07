#!/usr/bin/env python3
"""Bounded local score-many parity and source-I/O regression fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'sinq'))
from optimizer import native_codec


class LocalScoring(unittest.TestCase):
    def fixture(self, d):
        w = np.sin(np.arange(7 * 256, dtype=np.float32) * .17).reshape(7, 256)
        x = np.cos(np.arange(4 * 256, dtype=np.float32) * .31).reshape(4, 256)
        w.tofile(d / 'weights'); x.tofile(d / 'x'); (x * .75).tofile(d / 'v')
        tensor = SimpleNamespace(shape=[256, 7], data=w, n_bytes=w.nbytes,
                                 tensor_type=SimpleNamespace(name='F32'))
        activation = dict(data_file=str(d / 'x'), count=4)
        validation = dict(data_file=str(d / 'v'), count=4)
        config = dict(schema_version=2, operation='generate', weights=str(d / 'weights'),
                      rows=7, columns=256, codec='q5_0', output=str(d / 'generated'),
                      workspace_bytes=4 << 20, max_output_bytes=4 << 20, threads=1,
                      candidates=[dict(generator='sinq', strip_width=64, iterations=16, strength=s)
                                  for s in (0., .5, 1.)])
        with tempfile.TemporaryDirectory() as req:
            generated = native_codec.request(BINARY, config, req)
        candidates = []
        for item in generated['candidates']:
            candidates.append(SimpleNamespace(payload=d / 'generated' / item['payload'],
                scales=None if item['index'] == 0 else d / 'generated' / item['scales'],
                provenance=dict(shape=[7, 256], qtype='Q5_0',
                                input_sha256=hashlib.sha256(w.tobytes()).hexdigest())))
        return tensor, activation, validation, candidates

    def test_many_matches_legacy_and_exports_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            tensor, a, v, candidates = self.fixture(Path(tmp))
            pairs = list(zip(('stock', 'half', 'full'), candidates))
            for objective in ('output_l1', 'output_mse'):
                with mock.patch.object(native_codec, 'copy_tensor_bytes', wraps=native_codec.copy_tensor_bytes) as copy, \
                     mock.patch.object(native_codec, 'file_sha256', wraps=native_codec.file_sha256) as hash_file, \
                     mock.patch.object(native_codec, '_run', wraps=native_codec._run) as run:
                    expected = [native_codec.rescore(BINARY, c, tensor, a, validation=v,
                                objective=objective, threads=1) for c in candidates]
                self.assertEqual((copy.call_count, hash_file.call_count, run.call_count), (3, 3, 3))
                for threads in (1, 4):
                    with mock.patch.object(native_codec, 'copy_tensor_bytes', wraps=native_codec.copy_tensor_bytes) as copy, \
                         mock.patch.object(native_codec, 'file_sha256', wraps=native_codec.file_sha256) as hash_file, \
                         mock.patch.object(native_codec, '_run', wraps=native_codec._run) as run:
                        actual = native_codec.rescore_many(BINARY, pairs, tensor, a, validation=v,
                                    objective=objective, threads=threads, workspace_bytes=40 << 10)
                    self.assertEqual(copy.call_count, 1); self.assertEqual(hash_file.call_count, 1)
                    self.assertEqual(run.call_count, 1)
                    self.assertEqual([key for key, score in actual], ['stock', 'half', 'full'])
                    self.assertEqual([score for key, score in actual], expected)
                    self.assertFalse(Path(copy.call_args.args[1]).exists())

    def test_native_and_response_mapping_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            tensor, a, v, candidates = self.fixture(d)
            candidate = candidates[0]
            with mock.patch.object(native_codec, 'request', return_value={'scores': [
                    dict(candidate_id='wrong', score={})]}):
                with self.assertRaisesRegex(ValueError, 'mapping'):
                    native_codec.rescore_many(BINARY, [('right', candidate)], tensor, a)
            candidate.payload.write_bytes(b'broken')
            with self.assertRaisesRegex(RuntimeError, 'payload size mismatch'):
                native_codec.rescore_many(BINARY, [('right', candidate)], tensor, a)

    def test_provenance_and_mapping_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tensor, a, v, candidates = self.fixture(Path(tmp))
            candidates[1].provenance['input_sha256'] = '0' * 64
            with mock.patch.object(native_codec, '_run') as run:
                with self.assertRaisesRegex(ValueError, 'reference'):
                    native_codec.rescore_many(BINARY, [('a', candidates[0]), ('b', candidates[1])], tensor, a)
                run.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'candidate ID'):
                native_codec.rescore_many(BINARY, [('same', candidates[0]), ('same', candidates[0])], tensor, a)
            tensor.data[0, 0] += 1.
            with mock.patch.object(native_codec, '_run') as run:
                with self.assertRaisesRegex(ValueError, 'bytes differ'):
                    native_codec.rescore_many(BINARY, [('a', candidates[0])], tensor, a)
                run.assert_not_called()


def benchmark():
    """Synthetic three-candidate control; not a historical recipe stage benchmark."""
    with tempfile.TemporaryDirectory() as tmp:
        tensor, a, v, candidates = LocalScoring().fixture(Path(tmp))
        records = []
        for objective in ('output_l1', 'output_mse'):
            start = time.perf_counter()
            before = [native_codec.rescore(BINARY, c, tensor, a, validation=v,
                      objective=objective, threads=4) for c in candidates]
            legacy_seconds = time.perf_counter() - start
            metrics = {}
            after = native_codec.rescore_many(BINARY, list(zip(('stock', 'half', 'full'), candidates)),
                    tensor, a, validation=v, objective=objective, threads=4, metrics=metrics)
            delta = max(abs(old[field] - new[field]) for old, (_, new) in zip(before, after)
                        for field in ('train_score', 'validation_score'))
            records.append(dict(objective=objective, fixture='synthetic-f32-7x256-4samples-held-out-q5_0',
                candidate_count=3, legacy_source_bytes_copied=3 * tensor.n_bytes,
                legacy_source_fresh_hash_bytes=3 * tensor.n_bytes, legacy_process_launches=3,
                legacy_elapsed_seconds=legacy_seconds, max_absolute_score_difference=delta, **metrics))
        print(json.dumps(records, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--bridge', required=True)
    parser.add_argument('--benchmark', action='store_true')
    args, rest = parser.parse_known_args(); BINARY = str(Path(args.bridge).resolve())
    if args.benchmark:
        benchmark()
    else:
        unittest.main(argv=[sys.argv[0], *rest])
