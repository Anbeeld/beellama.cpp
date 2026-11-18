"""Synthetic-only tests: production reference bytes are never opened by this suite."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'gguf-py')]
from tools.sinq.optimizer import trusted_reference as tr
from tools.sinq.optimizer import resources


def fixture(root):
    path = root / 'reference.logits'
    tokens = struct.pack('<3I', 1, 2, 3)
    payload = struct.pack('<8sIIQ', b'_logitF1', 4, 8, 3) + tokens + bytes(48)
    path.write_bytes(payload)
    lock = dict(status='PASS', reference_path=str(path), reference_bytes=len(payload),
                reference_format='_logitF1', reference_context=4, reference_vocabulary=8,
                reference_input_tokens=3, scored_targets=2,
                reference_sha256=hashlib.sha256(payload).hexdigest(),
                final_test_token_stream_sha256=hashlib.sha256(tokens).hexdigest())
    lock_path = root / 'lock.json'
    lock_path.write_text(json.dumps(lock))
    return path, lock_path, lock


@unittest.skipUnless(os.name == 'nt', 'mandatory Win32 guards')
class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path, self.lock_path, self.lock = fixture(self.root)

    def tripwire(self, path):
        if Path(path).resolve() == self.path.resolve():
            raise AssertionError('normal reference full hash attempted')
        return resources.file_digest(path)

    def test_success_resume_report_and_cleanup_no_hash(self):
        with mock.patch.object(tr, 'file_digest', side_effect=self.tripwire):
            for _ in range(2):  # resume reacquires guards; no historical freshness claim
                with tr.TrustedLargeReference(self.lock_path) as reference:
                    report = json.loads(json.dumps(reference.provenance()))
                    self.assertFalse(report['fresh_content_hash'])
                    self.assertEqual(report['reference_sha256_new_bytes_read'], 0)
                    self.assertEqual(reference.check()['size'], len(self.path.read_bytes()))
                    with self.assertRaises(OSError):
                        self.path.write_bytes(b'x' * self.lock['reference_bytes'])
                    with self.assertRaises(OSError):
                        self.path.rename(self.root / 'replacement')
            self.path.write_bytes(self.path.read_bytes())  # released on completion

    def test_error_paths_never_hash(self):
        with mock.patch.object(tr, 'file_digest', side_effect=self.tripwire):
            for field, value in [('reference_bytes', 1), ('reference_context', 5),
                                 ('final_test_token_stream_sha256', '0' * 64)]:
                changed = dict(self.lock, **{field: value})
                self.lock_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    with tr.TrustedLargeReference(self.lock_path):
                        self.fail('invalid reference accepted')
                self.path.write_bytes(self.path.read_bytes())
            self.lock_path.write_text(json.dumps(self.lock))
            with mock.patch.object(tr.GuardedFile, '__enter__', side_effect=OSError('sharing denied')):
                with self.assertRaises(OSError):
                    with tr.TrustedLargeReference(self.lock_path):
                        pass
            self.lock_path.unlink()
            with self.assertRaises(FileNotFoundError):
                with tr.TrustedLargeReference(self.lock_path):
                    pass

    def test_hardlink_and_canonical_alias_rejected(self):
        alias = self.root / 'hardlink'
        os.link(self.path, alias)
        with self.assertRaises(OSError):
            with tr.GuardedFile(self.path):
                pass
        alias.unlink()
        with self.assertRaises(OSError):
            with tr.GuardedFile(str(self.root / 'missing' / '..' / self.path.name)):
                pass
    def test_symlink_rejected(self):
        alias = self.root / 'symlink'
        try:
            alias.symlink_to(self.path)
        except OSError as error:
            self.skipTest(f'Windows symlink privilege unavailable (winerror={error.winerror})')
        with self.assertRaises(OSError):
            with tr.GuardedFile(alias):
                pass

    def test_guarded_digest_once_and_corruption(self):
        real = resources.file_digest
        with mock.patch.object(tr, 'file_digest', wraps=real) as digest:
            with tr.GuardedFile(self.path) as guard:
                for _ in range(3):
                    self.assertEqual(guard.verified_sha256(self.lock['reference_sha256']), self.lock['reference_sha256'])
                self.assertEqual(digest.call_count, 1)
            self.path.write_bytes(bytes(self.lock['reference_bytes']))
            with tr.GuardedFile(self.path) as guard:
                with self.assertRaises(ValueError):
                    guard.verified_sha256(self.lock['reference_sha256'])

    def test_explicit_audit_small_synthetic_only(self):
        out = self.root / 'audit.json'
        self.assertEqual(tr.main(['audit-reference', str(self.lock_path), '--output', str(out)]), 0)
        report = json.loads(out.read_text())
        self.assertTrue(report['fresh_content_hash'])
        self.assertEqual(report['reference_sha256_new_bytes_read'], self.lock['reference_bytes'])

    def test_separate_process_reader_compatible_with_guard(self):
        import subprocess
        with tr.GuardedFile(self.path) as guard:
            result = subprocess.run([sys.executable, '-c',
                'from pathlib import Path; import sys; assert Path(sys.argv[1]).read_bytes()[:8] == b\"_logitF1\"',
                str(self.path)], capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            guard.check()

    def test_interrupted_context_closes_guard(self):
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with tr.TrustedLargeReference(self.lock_path):
                raise RuntimeError('interrupted')
        self.path.write_bytes(self.path.read_bytes())


@unittest.skipUnless(os.name == 'nt', 'Windows final runner')
class RunnerTripwireTests(unittest.TestCase):
    def test_real_wikitext_orchestration_success_resume_report_and_native_error(self):
        import io
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path, small_lock, _ = fixture(root)
            for name in ('locks', 'report', 'data', 'build-sinq-clean/bin'):
                (root / name).mkdir(parents=True)
            def put(name, data):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
                return path
            model = root / 'model.gguf'
            model.write_bytes(b'synthetic model')
            model_sha = resources.file_digest(model)
            fixed = [{'arm': f'{recipe}-{i}', 'recipe': recipe, 'model_sha256': model_sha}
                     for recipe in ('T340', 'T425') for i in range(12)]
            labels = ('AUTO-LOCAL-L1','AUTO-LOCAL-MSE','AUTO-BLOCK-MSE','AUTO-FISHER','AUTO-GROUP-KLD-BUDGETED')
            autos = [{'recipe': recipe, 'label': label, 'model_sha256': model_sha}
                     for recipe in ('T340', 'T425') for label in labels]
            put('locks/hardware-build-lock.json', {'gpus': ['GPU-test,device', 'GPU-other,device']})
            put('report/fixed-model-audit.json', {'status': 'PASS', 'records': fixed})
            put('report/fixed-selection-ranking.json', {})
            put('report/auto-model-audit.json', {'status': 'PASS', 'models': autos})
            stages = {key: {'exit_code': 0, 'status': 'PASS'} for key in ('load_only', 'cpu_forward', 'cuda_forward')}
            put('report/fixed-model-runtime-validation.json', {'status': 'PASS', 'validated_arms': 24,
                'results': [dict(stages, arm=row['arm'], status='PASS', model_sha256_before=model_sha,
                                 model_sha256_after=model_sha) for row in fixed]})
            put('report/auto-model-runtime-validation.json', {'status': 'PASS', 'validated_models': 10,
                'results': [dict(row, status='PASS', model_sha256_before=model_sha,
                                 model_sha256_after=model_sha, stages=[{'exit_code': 0}] * 3) for row in autos]})
            put('report/surrogate-score-matrix-audit.json', {'status': 'PASS', 'audited_score_files': 8})
            cost = put('cost.json', {'synthetic': True})
            put('report/selector-costs-index.json', {'status': 'PASS', 'records': [
                {'path': str(cost), 'sha256': resources.file_digest(cost)}] * 10})
            put('report/evaluator-parity.json', {'status': 'PASS'})
            for label in labels:
                selected = root / 'selected/T340' / label
                selected.mkdir(parents=True)
                (selected / 'best.gguf').write_bytes(model.read_bytes())
                put(str((selected / 'selection.json').relative_to(root)), {'status': 'PASS', 'best_gguf_sha256': model_sha})
            np.zeros(297194, dtype='<u4').tofile(root / 'data/final_test.u32')
            token_sha = resources.file_digest(root / 'data/final_test.u32')
            put('locks/corpus-source-provenance.json', {'token_streams': {'final_test': {'sha256': token_sha}}})
            source_sha = 'da9df48e961a191aafca383628f983da4552a0a61c1257b93e1d678966f1c4af'
            corpus_sha = '173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08'
            input_lock = put('locks/final-wikitext-input-lock.json', dict(json.loads(small_lock.read_text()),
                source_sha256=source_sha, corpus_sha256=corpus_sha,
                reference_sha256='b0963bb182cfb65b9f317aab36203718d5730c21f82eb57ca401bdb89969d89b',
                reference_context=65536, reference_vocabulary=248320, reference_input_tokens=297194,
                scored_targets=297193, reference_bytes=147601497864, final_test_token_stream_sha256=token_sha))
            executable = root / 'build-sinq-clean/bin/llama-perplexity.exe'
            executable.write_bytes(b'synthetic executable; never launched')
            manifest = put('models.json', {'status': 'FROZEN_BEFORE_HELD_OUT_EVALUATION', 'models': [
                {'label': 'control', 'recipe': 'T340', 'kind': 'fixed', 'arm': 'T340-0',
                 'path': str(model), 'expected_sha256': model_sha}]})
            execution_lock = put('execution.json', {'status': 'FROZEN_BEFORE_HELD_OUT_EVALUATION',
                'models_manifest_sha256': resources.file_digest(manifest), 'executable_sha256': resources.file_digest(executable),
                'source_sha256': source_sha, 'wikitext_input_lock_sha256': resources.file_digest(input_lock),
                'gpus': {'0': {'pci_bus_id': '0000:test'}}})
            spec = importlib.util.spec_from_file_location('final_wikitext_test', REPO / 'tests/sinq-reference/experiment/run_final_wikitext.py')
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            runner.ROOT, runner.REPO = root, root
            real_digest = resources.file_digest
            def tripwire(path):
                path = Path(path)
                if path.resolve() == reference_path.resolve():
                    raise AssertionError('large reference digest attempted')
                if path.name == 'Qwen3.5-9B-text-BF16.gguf':
                    return source_sha
                if path.name == 'wiki.test.raw':
                    return corpus_sha
                return real_digest(path)
            native_status = [0]
            launches = []
            interrupted_cleanup = []
            case = self
            class Process:
                def __init__(self, command, **kwargs):
                    launches.append(command)
                    self.returncode = native_status[0]
                    self.stdout = io.BytesIO(b'offloaded 33/33 layers to GPU 0000:test')
                    if self.returncode == 0:
                        tsv = Path(command[command.index('--kld-token-output') + 1])
                        with tsv.open('w') as stream:
                            stream.write('input_index\tcontext_position\ttarget_token\tkld\tbase_nll\tcandidate_nll\ttop1_agree\n')
                            for i in range(297193):
                                stream.write(f'{i}\t{i % 65536}\t0\t0\t1\t1\t1\n')
                def poll(self):
                    return self.returncode
                @property
                def pid(self):
                    raise RuntimeError('synthetic interruption')
                def terminate(self):
                    with case.assertRaises(OSError):
                        model.write_bytes(b'guard must still be active')
                    interrupted_cleanup.append(True)
                    self.returncode = -1
                def wait(self, **kwargs):
                    return self.returncode
            reference_factory = lambda *a, **kw: tr.TrustedLargeReference(small_lock, **kw)
            with mock.patch.object(runner, 'file_digest', side_effect=tripwire), \
                 mock.patch.object(tr, 'file_digest', side_effect=tripwire), \
                 mock.patch.object(resources, 'file_digest', side_effect=tripwire), \
                 mock.patch.object(runner, 'TrustedLargeReference', side_effect=reference_factory), \
                 mock.patch.object(runner.subprocess, 'Popen', side_effect=Process), \
                 mock.patch.object(runner.subprocess, 'run', side_effect=OSError('no GPU in synthetic test')), \
                 mock.patch('sys.stdout', new_callable=io.StringIO):
                argv = ['runner', 'T340', '0', '--models-manifest', str(manifest),
                        '--execution-lock', str(execution_lock), '--output-directory', str(root / 'out')]
                with mock.patch.object(sys, 'argv', argv):
                    runner.main()
                    runner.main()  # completed resume/report must not launch again
                self.assertEqual(len(launches), 1)
                report = json.loads((root / 'out/T340/progress.json').read_text())
                self.assertEqual(report['status'], 'PASS')
                self.assertEqual(report['reference_trust']['reference_sha256_new_bytes_read'], 0)
                report['reference_trust']['file_identity']['file_id'] ^= 1
                (root / 'out/T340/progress.json').write_text(json.dumps(report))
                with mock.patch.object(sys, 'argv', argv):
                    with self.assertRaisesRegex(ValueError, 'pinned identity'):
                        runner.main()
                native_status[0] = 1
                argv[-1] = str(root / 'error-out')
                with mock.patch.object(sys, 'argv', argv):
                    runner.main()
                report = json.loads((root / 'error-out/T340/progress.json').read_text())
                self.assertEqual(report['status'], 'INCOMPLETE')
                self.assertEqual(report['failed_attempts'][0]['stage'], 'native_evaluation')
                self.assertEqual(report['reference_trust']['reference_sha256_new_bytes_read'], 0)
                native_status[0] = None
                argv[-1] = str(root / 'interrupted-out')
                with mock.patch.object(sys, 'argv', argv):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                        runner.main()
                self.assertEqual(interrupted_cleanup, [True])
            model.write_bytes(model.read_bytes())  # every guard released, including error path


class UnsupportedTests(unittest.TestCase):
    def test_generic_selection_evaluator_rejects_large_reference_before_hashing(self):
        from tools.sinq.optimizer import kl
        evaluator = object.__new__(kl.NativeKLEvaluator)
        with mock.patch.object(kl, 'file_digest', side_effect=AssertionError('hash reached')), \
             mock.patch.object(evaluator, '_check_runtime', side_effect=AssertionError('runtime reached')):
            with self.assertRaisesRegex(ValueError, 'TrustedLargeReference'):
                evaluator.evaluate('unused', 'unused', 'Qwen3.5-9B-wikitext2-64k-bf16.logits')
            with mock.patch.object(kl.Path, 'stat', return_value=type('Stat', (), {'st_size': 147601497864})()):
                with self.assertRaisesRegex(ValueError, 'TrustedLargeReference'):
                    evaluator.evaluate('unused', 'unused', 'alias.logits')

    def test_fail_closed(self):
        with mock.patch.object(tr.os, 'name', 'posix'):
            # Construct before patch: WindowsPath selection itself uses os.name.
            guard = object.__new__(tr.GuardedFile)
            with self.assertRaisesRegex(RuntimeError, 'require Windows'):
                guard.__enter__()


if __name__ == '__main__':
    unittest.main()
