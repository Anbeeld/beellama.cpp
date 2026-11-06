"""CPU-only original-occurrence corpus partition and capture binding regressions."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sinq.optimizer.corpora import lock_corpora, load_corpora, bind_corpora
from tools.sinq.optimizer.resources import canonical_json, digest


def token_digest(tokens):
    return hashlib.sha256(b''.join(struct.pack('<I', t) for t in tokens)).hexdigest()


class CorporaTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokens = [1, 12, 1, 24, 1, 36, 1, 48, 1, 60, 1, 72]
        self.source = self.root / 'private-source.u32'
        self.source.write_bytes(b''.join(struct.pack('<I', t) for t in self.tokens))
        self.sources = {'private-label': self.source}
        self.parts = {role: [dict(source='private-label', start=i*3, end=(i+1)*3)]
                      for i, role in enumerate(('calibration', 'validation', 'selection', 'final_test'))}
        self.output = self.root / 'lock.json'

    def lock(self, **kwargs):
        limits = dict(max_tokens=100, max_output_bytes=10000, max_files=10)
        limits.update(kwargs)
        return lock_corpora(self.sources, self.parts, self.output, **limits)

    def bindings(self, manifest, validation=True):
        def stage(role):
            return dict(manifest_sha256=digest({'capture': role}),
                        token_sha256=manifest['partitions'][role]['token_sha256'],
                        token_count=manifest['partitions'][role]['token_count'],
                        corpus_sha256=digest({'serialization': role}))
        return dict(calibration=stage('calibration'), validation=stage('validation') if validation else None)

    def write_manifest(self, value, *, checksum=True):
        value = copy.deepcopy(value)
        if checksum:
            value['checksum'] = digest({k: v for k, v in value.items() if k != 'checksum'})
        self.output.write_bytes(canonical_json(value))

    def test_lock_all_stages_actual_hashes_and_no_data_leak(self):
        manifest = self.lock()
        source_hash = token_digest(self.tokens)
        self.assertEqual(manifest['schema_version'], 2)
        self.assertEqual(manifest['kind'], 'sinq-corpus-partitions')
        self.assertEqual(manifest['sources'], {source_hash: {'token_count': 12}})
        for i, role in enumerate(('calibration', 'validation', 'selection', 'final_test')):
            stage = manifest['partitions'][role]
            self.assertEqual(stage, dict(token_sha256=token_digest(self.tokens[i*3:(i+1)*3]), token_count=3,
                chunks=[dict(token_source_sha256=source_hash, start=i*3, end=(i+1)*3)]))
        self.assertEqual(load_corpora(self.output), manifest)
        encoded = self.output.read_text()
        self.assertNotIn(str(self.source), encoded)
        self.assertNotIn('private-label', encoded)
        self.assertEqual(self.output.read_bytes(), canonical_json(manifest))
        bindings = self.bindings(manifest)
        identity = bind_corpora(manifest, bindings, self.tokens[6:9])
        self.assertEqual(identity['manifest_sha256'], manifest['checksum'])
        self.assertEqual(identity, bind_corpora(load_corpora(self.output), bindings, self.tokens[6:9]))
        self.assertIsNone(bind_corpora(None, bindings))

    def test_ordered_multiple_slices_and_label_path_independence(self):
        self.parts['selection'] = [dict(source='private-label', start=8, end=9),
                                   dict(source='private-label', start=6, end=8)]
        manifest = self.lock()
        self.assertEqual(manifest['partitions']['selection']['token_sha256'], token_digest(self.tokens[8:9]+self.tokens[6:8]))
        other = self.root / 'copy.u32'; other.write_bytes(self.source.read_bytes())
        parts = copy.deepcopy(self.parts)
        for chunks in parts.values():
            for chunk in chunks: chunk['source'] = 'new-label'
        result = lock_corpora({'new-label': other}, parts, self.root/'other.json', max_tokens=100, max_output_bytes=10000, max_files=10)
        self.assertEqual(result, manifest)

    def test_optional_validation_and_reserved_validation(self):
        manifest = self.lock()
        self.assertIsNone(bind_corpora(manifest, self.bindings(manifest, False))['validation'])
        self.output.unlink()
        del self.parts['validation']
        manifest = self.lock()
        self.assertIsNone(manifest['partitions']['validation'])
        self.assertIsNotNone(bind_corpora(manifest, self.bindings(manifest, False)))
        self.output.unlink()
        self.parts['validation'] = None
        self.assertEqual(self.lock(), manifest)
        bindings = self.bindings(manifest, False)
        bindings['validation'] = dict(manifest_sha256='1'*64, token_sha256='2'*64)
        with self.assertRaises(ValueError): bind_corpora(manifest, bindings)

    def test_all_cross_role_overlaps_and_duplicates_no_write(self):
        for role in ('calibration', 'validation', 'final_test'):
            with self.subTest(role=role):
                original = copy.deepcopy(self.parts)
                self.parts[role] = [dict(source='private-label', start=7, end=8)]
                with self.assertRaises(ValueError): self.lock()
                self.assertFalse(self.output.exists())
                self.parts = original
        self.parts['selection'].append(copy.deepcopy(self.parts['selection'][0]))
        with self.assertRaises(ValueError): self.lock()
        self.assertFalse(self.output.exists())

    def test_aliases_same_full_hash_cannot_hide_overlap(self):
        alias = self.root/'alias.u32'; alias.write_bytes(self.source.read_bytes())
        self.sources['alias'] = alias
        self.parts['selection'] = [dict(source='alias', start=1, end=2)]
        with self.assertRaises(ValueError): self.lock()
        self.assertFalse(self.output.exists())
        self.parts['selection'] = [dict(source='alias', start=6, end=9)]
        manifest = self.lock()
        self.assertEqual(len(manifest['sources']), 1)

    def test_resource_limits_before_publication(self):
        for limits in ({'max_tokens': 11}, {'max_output_bytes': 1}, {'max_files': 1},
                       {'max_tokens': True}, {'max_files': 0}, {'max_output_bytes': -1}):
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError): self.lock(**limits)
                self.assertFalse(self.output.exists())
        self.assertEqual(set(self.root.iterdir()), {self.source})
        self.output = self.root/'not-created'/'lock.json'
        with self.assertRaises(ValueError): self.lock(max_output_bytes=1)
        self.assertFalse(self.output.parent.exists())

    def test_source_nonempty_alignment_and_signed_ids(self):
        for data in (b'', b'abc', struct.pack('<I', 2**31)):
            with self.subTest(data=data):
                self.source.write_bytes(data)
                with self.assertRaises(ValueError): self.lock()
                self.assertFalse(self.output.exists())

    def test_bad_ranges_required_roles_fields_and_types(self):
        bad_chunks = [dict(source='private-label', start=-1, end=2),
                      dict(source='private-label', start=1, end=1),
                      dict(source='private-label', start=0, end=13),
                      dict(source='missing', start=0, end=1),
                      dict(source='private-label', start=True, end=2),
                      dict(source='private-label', start=0, end=1, token_sha256='1'*64)]
        original = copy.deepcopy(self.parts)
        for chunk in bad_chunks:
            self.parts['selection'] = [chunk]
            with self.assertRaises(ValueError): self.lock()
            self.assertFalse(self.output.exists())
        for role in ('calibration', 'selection', 'final_test'):
            for value in (None, []):
                self.parts = copy.deepcopy(original); self.parts[role] = value
                with self.assertRaises(ValueError): self.lock()
                self.assertFalse(self.output.exists())

    def test_input_rechecked_after_stage_hashing_no_publication(self):
        from tools.sinq.optimizer import corpora
        original = corpora._stage_digest
        changed = False

        def change_source(chunks, paths):
            nonlocal changed
            result = original(chunks, paths)
            if not changed:
                with self.source.open('r+b') as handle:
                    handle.write(struct.pack('<I', 99))
                changed = True
            return result

        with mock.patch.object(corpora, '_stage_digest', side_effect=change_source):
            with self.assertRaisesRegex(ValueError, 'changed'): self.lock()
        self.assertFalse(self.output.exists())
        self.assertEqual(set(self.root.iterdir()), {self.source})

    def test_source_truncation_after_scan_no_publication(self):
        from tools.sinq.optimizer import corpora
        original = corpora._stage_digest

        def truncate_after_slice(chunks, paths):
            result = original(chunks, paths)
            self.source.write_bytes(struct.pack('<I', 1))
            return result

        with mock.patch.object(corpora, '_stage_digest', side_effect=truncate_after_slice):
            with self.assertRaisesRegex(ValueError, 'truncated|changed'): self.lock()
        self.assertFalse(self.output.exists())

    def test_bounded_mapping_non_aligned_offsets_and_large_stage(self):
        tokens = [i % 1000 for i in range(300001)]
        self.source.write_bytes(b''.join(struct.pack('<I', t) for t in tokens))
        self.parts = {
            'calibration': [dict(source='private-label', start=0, end=1)],
            'selection': [dict(source='private-label', start=1, end=300000)],
            'final_test': [dict(source='private-label', start=300000, end=300001)]}
        manifest = self.lock(max_tokens=300001)
        self.assertEqual(manifest['partitions']['selection']['token_sha256'], token_digest(tokens[1:300000]))

    def test_immutable_existing_and_input_alias_no_write(self):
        original = self.source.read_bytes()
        self.output = self.source
        with self.assertRaises((ValueError, FileExistsError)): self.lock()
        self.assertEqual(self.source.read_bytes(), original)
        self.output = self.root/'lock.json'; self.lock()
        original = self.output.read_bytes()
        with self.assertRaises((ValueError, FileExistsError)): self.lock()
        self.assertEqual(self.output.read_bytes(), original)

    def test_strict_loaded_hash_checksum_range_count_overlap(self):
        manifest = self.lock()
        corruptions = []
        value = copy.deepcopy(manifest); value['kind'] = 'other'; corruptions.append(value)
        value = copy.deepcopy(manifest); value['schema_version'] = True; corruptions.append(value)
        value = copy.deepcopy(manifest); value['path'] = 'private'; corruptions.append(value)
        value = copy.deepcopy(manifest); value['partitions']['selection']['token_sha256'] = 'G'*64; corruptions.append(value)
        value = copy.deepcopy(manifest); value['partitions']['selection']['token_count'] += 1; corruptions.append(value)
        value = copy.deepcopy(manifest); value['partitions']['selection']['chunks'][0]['end'] = 99; corruptions.append(value)
        value = copy.deepcopy(manifest); value['partitions']['selection']['chunks'][0]['start'] = 2; value['partitions']['selection']['token_count'] = 7; corruptions.append(value)
        value = copy.deepcopy(manifest); value['partitions']['final_test'] = None; corruptions.append(value)
        value = copy.deepcopy(manifest); value['sources'][next(iter(value['sources']))]['token_count'] = 0; corruptions.append(value)
        value = copy.deepcopy(manifest); key = next(iter(value['sources'])); value['sources'][key.upper()] = value['sources'].pop(key); corruptions.append(value)
        value = copy.deepcopy(manifest); value['partitions']['selection']['chunks'][0]['token_source_sha256'] = '0'*64; corruptions.append(value)
        value = copy.deepcopy(manifest); value['partitions']['selection']['chunks'][0]['start'] = False; corruptions.append(value)
        value = copy.deepcopy(manifest); value['partitions']['selection']['chunks'][0]['path'] = 'private'; corruptions.append(value)
        for value in corruptions:
            self.write_manifest(value)
            with self.assertRaises(ValueError): load_corpora(self.output)
        value = copy.deepcopy(manifest); value['partitions']['selection']['token_sha256'] = '1'*64
        self.write_manifest(value, checksum=False)
        with self.assertRaises(ValueError): load_corpora(self.output)
        self.output.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaises(ValueError): load_corpora(self.output)

    def test_binding_rejects_stale_capture_and_wrong_native_stream(self):
        manifest = self.lock(); good = self.bindings(manifest)
        for role in ('calibration', 'validation'):
            for field, value in (('token_sha256', '0'*64), ('manifest_sha256', 'malformed'), ('token_count', 4)):
                bindings = copy.deepcopy(good); bindings[role][field] = value
                with self.assertRaises(ValueError): bind_corpora(manifest, bindings)
        for tokens in (self.tokens[7:9]+self.tokens[6:7], self.tokens[6:8], [1, True, 2], [2**31, 1, 2]):
            with self.assertRaises(ValueError): bind_corpora(manifest, good, tokens)
        unknown_count = copy.deepcopy(good); del unknown_count['calibration']['token_count']
        self.assertIsNotNone(bind_corpora(manifest, unknown_count, self.tokens[6:9]))
        ordinary=bind_corpora(manifest, {'calibration':None,'validation':None},self.tokens[6:9])
        self.assertIsNone(ordinary['calibration'])
        with self.assertRaises(ValueError):bind_corpora(manifest,{'calibration':None,'validation':None},self.tokens[6:8])
        value = copy.deepcopy(manifest); value['checksum'] = '0'*64
        with self.assertRaises(ValueError): bind_corpora(value, good)


if __name__ == '__main__':
    unittest.main()
