"""Stage summaries inspect only bounded, explicitly published artifacts."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sinq.optimizer import cli, resources


class SummaryInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'run'
        self.output.mkdir()
        self.config = dict(output_directory=str(self.output), budgets=dict(max_files=20, max_output_bytes=4096))

    def publish(self, name, value):
        path = self.output / name
        path.write_bytes(resources.canonical_json(value))
        return path

    def summary(self, stage, result):
        return cli._stage_summary(stage, self.config, result, time.monotonic(), {})

    def test_generate_never_traverses_or_opens_unexpected_large_files(self):
        result = dict(source_sha256='a' * 64, tensors={})
        self.publish('run-manifest.json', result)
        identity = {'kind': 'sinq-generation-run'}
        identity['checksum'] = resources.digest(identity)
        self.publish('run-identity.json', identity)
        unexpected = self.output / 'unexpected.logits'
        with unexpected.open('wb') as stream:
            stream.truncate(1 << 30)
        (self.output / 'cache').mkdir()
        (self.output / 'cache' / 'payload').write_bytes(b'not inventory')
        original = io.open
        def guarded(path, *args, **kwargs):
            if Path(path) == unexpected:
                raise AssertionError('unexpected file opened')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'rglob', side_effect=AssertionError('recursive inventory')), patch.object(Path, 'iterdir', side_effect=AssertionError('directory scan')), patch('io.open', side_effect=guarded):
            summary = self.summary('generate', result)
        self.assertEqual(set(summary['artifacts']), {'run-manifest.json', 'run-identity.json'})
        self.assertTrue(summary['artifacts']['run-manifest.json']['fresh_content_hash'])

    def test_select_resume_reuses_published_model_digest_without_hash(self):
        model = self.output / 'best.gguf'
        model.write_bytes(b'published model')
        result = dict(best_gguf_sha256=hashlib.sha256(model.read_bytes()).hexdigest(), source_sha256='a' * 64)
        self.publish('selection.json', result)
        self.publish('checkpoint.json', {'state': 'fixture'})
        self.publish('run-identity.json', {'kind': 'sinq-selection-run'})
        self.config['source_gguf'] = str(self.root / 'source.gguf')
        with patch.object(cli, 'load_config', return_value=self.config), patch('tools.sinq.optimizer.stages.select_config', return_value=result) as select, patch.object(Path, 'rglob', side_effect=AssertionError('recursive inventory')), patch.object(Path, 'iterdir', side_effect=AssertionError('directory scan')), patch.object(resources, 'file_digest', side_effect=AssertionError('redundant file hash')), patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(cli.main(['select', '--config', 'unused', '--resume', str(self.output)]), 0)
        select.assert_called_once_with(self.config, resume=str(self.output))
        summary = json.loads(output.getvalue())['stage_summary']
        evidence = summary['artifacts']['best.gguf']
        self.assertEqual(evidence['sha256'], result['best_gguf_sha256'])
        self.assertEqual(evidence['bytes'], model.stat().st_size)
        self.assertEqual(evidence['path'], str(model))
        self.assertEqual(evidence['hash_scope'], 'published-output-evidence-not-fresh-hash')
        self.assertFalse(evidence['fresh_content_hash'])
        self.assertEqual(summary['input_hashes']['source_gguf'], result['source_sha256'])
        self.assertFalse(summary['input_hash_evidence']['source_gguf']['fresh_content_hash'])

    def test_assemble_uses_live_published_output_evidence(self):
        model = self.root / 'assembled.gguf'
        model.write_bytes(b'output')
        self.config['output_gguf'] = str(model)
        with patch.object(resources, 'file_digest', side_effect=AssertionError('redundant hash')):
            summary = self.summary('assemble', {'output_sha256': 'b' * 64})
        self.assertEqual(set(summary['artifacts']), {'assembled.gguf'})
        self.assertFalse(summary['artifacts']['assembled.gguf']['fresh_content_hash'])

    def test_small_published_manifest_tamper_is_rejected(self):
        self.publish('scores.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.summary('score', {'changed': False})

    def test_metadata_is_fresh_each_invocation_even_with_same_stat(self):
        import os
        path = self.root / 'candidates.json'
        path.write_bytes(b'{"a":1}')
        original = path.stat()
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        self.config['candidate_manifest'] = str(path)
        self.summary('validate', {'candidate_manifest_sha256': expected})
        path.write_bytes(b'{"a":2}')
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.summary('validate', {'candidate_manifest_sha256': expected})

    def test_cumulative_manifest_bytes_respect_configured_limit(self):
        result = dict(tensors={})
        first = self.publish('run-manifest.json', result)
        self.publish('run-identity.json', {'kind': 'generation'})
        self.config['budgets']['max_output_bytes'] = first.stat().st_size
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.summary('generate', result)

    def test_identity_expected_checksum_mismatch_is_rejected(self):
        result = dict(run_identity_sha256='a' * 64, tensors={})
        self.publish('run-manifest.json', result)
        self.publish('run-identity.json', {'checksum': 'b' * 64})
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.summary('generate', result)

    def test_manifest_byte_and_file_limits_fail_before_read(self):
        result = dict(tensors={})
        path = self.publish('run-manifest.json', result)
        self.config['budgets']['max_output_bytes'] = 1
        with patch.object(Path, 'open', side_effect=AssertionError('over-budget read')), self.assertRaisesRegex(ValueError, 'budget'):
            self.summary('generate', result)
        self.config['budgets']['max_output_bytes'] = 4096
        self.config['budgets']['max_files'] = 0
        with patch.object(Path, 'open', side_effect=AssertionError('over-budget read')), self.assertRaisesRegex(ValueError, 'budget'):
            self.summary('generate', result)
        self.assertTrue(path.exists())

    def test_validate_does_not_inventory_preexisting_output_tree(self):
        self.publish('scores.json', {'old': True})
        with patch.object(Path, 'rglob', side_effect=AssertionError('recursive inventory')), patch.object(Path, 'iterdir', side_effect=AssertionError('directory scan')):
            self.assertEqual(self.summary('validate', {'status': 'PASS'})['artifacts'], {})

    def test_input_metadata_expected_hash_detects_tamper(self):
        path = self.root / 'candidates.json'
        path.write_bytes(b'{}')
        self.config['candidate_manifest'] = str(path)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.summary('validate', {'candidate_manifest_sha256': 'c' * 64})


if __name__ == '__main__':
    unittest.main()
