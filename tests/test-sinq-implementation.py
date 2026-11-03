import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sinq.research import implementation


class Distribution:
    def __init__(self, name, version):
        self.metadata = {'Name': name}
        self.version = version


class ImplementationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = [
            'tools/sinq/research/cache.py', 'tools/sinq/research/resources.py',
            'tools/sinq/research/schemas.py', 'tools/sinq/research/nested/helper.py',
            'tools/sinq/export_activations.py', 'gguf-py/gguf/gguf_writer.py',
            'gguf-py/gguf/gguf_reader.py', 'gguf-py/gguf/quants.py',
            'gguf-py/gguf/py.typed', 'conversion/base.py',
            'conversion/nested/helper.py', 'convert_hf_to_gguf.py',
        ]
        for name in self.paths:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'original\n')
        self.metadata = patch.object(implementation.metadata, 'distributions',
            return_value=[Distribution('NumPy', '2.0'), Distribution('indirect_helper', '1.0')])
        self.metadata.start()
        self.addCleanup(self.metadata.stop)

    def fingerprint(self):
        return implementation.implementation_fingerprint(self.root)

    def test_content_changes_even_with_restored_mtime(self):
        for name in self.paths:
            with self.subTest(name=name):
                before = self.fingerprint()
                path = self.root / name
                stat = path.stat()
                path.write_bytes(b'changed!\n')
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                after = self.fingerprint()
                self.assertNotEqual(before['checksum'], after['checksum'])
                self.assertEqual(after['files'][name], hashlib.sha256(b'changed!\n').hexdigest())

    def test_add_delete_research_file(self):
        before = self.fingerprint()
        extra = self.root / 'tools/sinq/research/new.py'
        extra.write_text('new')
        self.assertNotEqual(before['checksum'], self.fingerprint()['checksum'])
        extra.unlink()
        self.assertEqual(before, self.fingerprint())

    def test_dependency_versions_and_absence(self):
        before = self.fingerprint()
        dependencies = before['versions']['dependencies']
        self.assertEqual(dependencies['numpy'], '2.0')
        for name in ('torch', 'transformers', 'safetensors', 'gguf', 'huggingface-hub'):
            self.assertIsNone(dependencies[name])
        with patch.object(implementation.metadata, 'distributions', return_value=[
                Distribution('numpy', '2.0'), Distribution('indirect-helper', '1.1')]):
            self.assertNotEqual(before['checksum'], self.fingerprint()['checksum'])

    def test_determinism_canonical_checksum_and_no_paths(self):
        before = self.fingerprint()
        with patch.object(implementation.metadata, 'distributions', return_value=[
                Distribution('indirect-helper', '1.0'), Distribution('numpy', '2.0')]):
            self.assertEqual(before, self.fingerprint())
        self.assertEqual(list(before['files']), sorted(self.paths))
        self.assertEqual(before['schema_version'], 1)
        payload = {key: value for key, value in before.items() if key != 'checksum'}
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        self.assertEqual(before['checksum'], hashlib.sha256(canonical.encode('utf-8')).hexdigest())
        self.assertNotIn(str(self.root), json.dumps(before))
        self.assertNotIn('\\', ''.join(before['files']))
        runtime = before['versions']
        self.assertTrue(runtime['python_implementation'])
        self.assertTrue(runtime['python_version'])
        self.assertTrue(runtime['platform'])
        self.assertIn('machine', runtime)

    def test_runtime_version_changes_identity(self):
        before = self.fingerprint()
        with patch.object(implementation.platform, 'python_version', return_value='99.0.0'):
            self.assertNotEqual(before['checksum'], self.fingerprint()['checksum'])
        with patch.object(implementation.metadata, 'distributions', return_value=[
                Distribution('numpy', '2.0'), Distribution('indirect-helper', '1.0'),
                Distribution('torch', '3.0')]):
            self.assertNotEqual(before['checksum'], self.fingerprint()['checksum'])

    def test_fixture_location_does_not_change_identity(self):
        before = self.fingerprint()
        with tempfile.TemporaryDirectory() as other:
            root = Path(other)
            for name in self.paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((self.root / name).read_bytes())
            self.assertEqual(before, implementation.implementation_fingerprint(root))

    def test_ignored_cache_and_nonresearch_data(self):
        before = self.fingerprint()
        for name in ('tools/sinq/research/__pycache__/module.py',
                     'gguf-py/gguf/__pycache__/module.pyc',
                     'conversion/__pycache__/module.pyc',
                     'tools/sinq/research/checkpoint.bin', 'model.gguf'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'not implementation')
        self.assertEqual(before, self.fingerprint())

    def test_missing_required_inputs_fail_closed(self):
        for name in ('tools/sinq/research', 'gguf-py/gguf', 'conversion',
                     'tools/sinq/export_activations.py', 'convert_hf_to_gguf.py'):
            with self.subTest(name=name):
                path = self.root / name
                moved = path.with_name(path.name + '.saved')
                path.rename(moved)
                try:
                    with self.assertRaises(ValueError):
                        self.fingerprint()
                finally:
                    moved.rename(path)

    def test_symlink_escape_fails_closed(self):
        link = self.root / 'tools/sinq/research/escape.py'
        try:
            link.symlink_to(ROOT / 'tools/sinq/research/cache.py')
        except (OSError, NotImplementedError) as error:
            self.skipTest('symlink creation unavailable: ' + str(error))
        with self.assertRaises(ValueError):
            self.fingerprint()

    def test_default_root_is_repo_not_cwd(self):
        before = implementation.implementation_fingerprint()
        old = Path.cwd()
        try:
            os.chdir(self.root)
            self.assertEqual(before, implementation.implementation_fingerprint())
        finally:
            os.chdir(old)


if __name__ == '__main__':
    unittest.main()
