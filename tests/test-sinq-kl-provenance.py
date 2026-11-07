"""Native KL provenance regressions with synthetic GGUFs and native process output."""
import csv
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gguf-py')]
import gguf
from tools.sinq.optimizer.kl import NativeKLEvaluator
from tools.sinq.optimizer.resources import digest, file_digest
from tools.sinq.optimizer.selector import Group, State, coordinate_descent


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.executable = self.directory / 'perplexity.exe'
        self.executable.write_bytes(b'synthetic native executable')
        self.corpus = self.directory / 'selection.txt'
        self.corpus.write_text('synthetic selection corpus', encoding='utf-8')
        self.tokens = [1, 3, 4, 5, 6, 7]
        self.models = []
        for index in range(3):
            path = self.directory / f'model-{index}.gguf'
            writer = gguf.GGUFWriter(path, 'llama')
            writer.add_block_count(1)
            writer.add_tensor('blk.0.ffn_up.weight', np.full((2, 4), index + 1, dtype='<f4'))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            self.models.append(path)
        self.original_payloads = {path: path.read_bytes() for path in self.models}
        self.gpu_uuid = 'GPU-5e87a39a-204c-4054-14a8-5b97ae4f16fd'
        self.rows_mutation = None
        self.interrupt_model = None
        self.evaluations = []
        self.native_patch = patch('subprocess.run', side_effect=self.native)
        self.native_patch.start()
        self.addCleanup(self.native_patch.stop)
        self.addCleanup(self.assert_payload_integrity)

    def assert_payload_integrity(self):
        for path, payload in self.original_payloads.items():
            self.assertEqual(path.read_bytes(), payload, str(path))

    def evaluator(self, backend='cuda', **overrides):
        execution = dict(backend=backend, cache_type_k='bf16', cache_type_v='bf16', flash_attention='on')
        if backend == 'cuda':
            execution['cuda_device'] = 0
        execution.update(overrides)
        return NativeKLEvaluator(self.executable, context=2, batch=2, microbatch=2,
                                 max_tokens=8, threads=1, execution=execution)

    def native(self, args, **kwargs):
        if args[0] == 'nvidia-smi':
            return subprocess.CompletedProcess(args, 0, (
                f'0, {self.gpu_uuid}, Synthetic GPU, 00000000:01:00.0, 550.0\n').encode(), b'')
        if '--list-devices' in args:
            return subprocess.CompletedProcess(args, 0, b'CUDA0: Synthetic GPU\n', b'')
        model = Path(args[args.index('-m') + 1])
        if '--save-all-logits' in args:
            reference = Path(args[args.index('--save-all-logits') + 1])
            vocabulary = 8
            record_bytes = (2 * ((vocabulary + 1) // 2) + 4) * 2
            reference.write_bytes(struct.pack('<8sIIQ', b'_logitF1', 2, vocabulary, len(self.tokens))
                                  + np.array(self.tokens, dtype='<i4').tobytes()
                                  + bytes((len(self.tokens) - 1) * record_bytes))
        else:
            self.evaluations.append(model)
            if model == self.interrupt_model:
                self.interrupt_model = None
                raise TimeoutError('synthetic native interruption')
            value = (3 - self.models.index(model)) / 10
            rows = [dict(input_index=i, context_position=i % 2, target_token=token, kld=value)
                    for i, token in enumerate(self.tokens[1:])]
            if self.rows_mutation:
                self.rows_mutation(rows)
            output = Path(args[args.index('--kld-token-output') + 1])
            with output.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['input_index', 'context_position', 'target_token', 'kld'], delimiter='\t')
                writer.writeheader()
                writer.writerows(rows)
        log = (b'offloaded 0/2 layers to GPU\n' if '--no-op-offload' in args else
               b'using device CUDA0 (Synthetic GPU) (0000:01:00.0)\noffloaded 2/2 layers to GPU\n')
        return subprocess.CompletedProcess(args, 0, b'', log)

    def reference(self, evaluator, name='reference'):
        path = self.directory / f'{name}.bin'
        manifest = evaluator.create_reference(self.models[0], self.corpus, path, expected_tokens=self.tokens)
        return path, manifest

    def test_cpu_cuda_reference_interchange_rejected(self):
        cpu, cuda = self.evaluator('cpu'), self.evaluator()
        for creator, consumer, name in ((cpu, cuda, 'cpu'), (cuda, cpu, 'cuda')):
            with self.subTest(creator=name):
                reference, manifest = self.reference(creator, name)
                self.assertEqual(manifest['policy'], creator.policy)
                self.assertEqual(creator.evaluate(self.models[1], self.corpus, reference), .2)
                before = len(self.evaluations)
                with self.assertRaisesRegex(ValueError, 'policy mismatch'):
                    consumer.evaluate(self.models[1], self.corpus, reference)
                self.assertEqual(len(self.evaluations), before)

    def test_kv_flash_compute_and_physical_gpu_bind_reference_identity(self):
        evaluator = self.evaluator()
        reference, manifest = self.reference(evaluator)
        changed = [self.evaluator(**{field: value}) for field, value in (
            ('cache_type_k', 'f16'), ('cache_type_v', 'f16'), ('flash_attention', 'off'),
            ('cublas_compute_type', 'f32'), ('mmq_precision', 'q8'))]
        self.gpu_uuid = 'GPU-aaaaaaaa-204c-4054-14a8-5b97ae4f16fd'
        changed.append(self.evaluator())
        identities = {digest(evaluator.policy)}
        for other in changed:
            with self.subTest(execution=other.execution, gpu=other.gpu):
                self.assertNotIn(digest(other.policy), identities)
                identities.add(digest(other.policy))
                with self.assertRaisesRegex(ValueError, 'policy mismatch'):
                    other.evaluate(self.models[1], self.corpus, reference)
        self.assertEqual(self.evaluations, [])
        self.assertEqual(manifest['execution_evidence']['gpu']['uuid'], evaluator.gpu['uuid'])

    def test_changed_reference_or_executable_rejected_before_score_cache_hit(self):
        evaluator = self.evaluator()
        reference, _ = self.reference(evaluator)
        cache = self.directory / 'scores'
        self.assertEqual(evaluator.evaluate(self.models[1], self.corpus, reference, cache_directory=cache), .2)
        self.assertEqual(evaluator.evaluate(self.models[1], self.corpus, reference, cache_directory=cache), .2)
        self.assertEqual(len(self.evaluations), 1)
        original = reference.read_bytes()
        reference.write_bytes(original[:-1] + b'\x01')
        with self.assertRaisesRegex(ValueError, 'reference/model/corpus/policy mismatch'):
            evaluator.evaluate(self.models[1], self.corpus, reference, cache_directory=cache)
        reference.write_bytes(original)
        self.executable.write_bytes(b'changed executable')
        with self.assertRaisesRegex(ValueError, 'executable/runtime changed'):
            evaluator.evaluate(self.models[1], self.corpus, reference, cache_directory=cache)
        replacement = self.evaluator()
        self.assertNotEqual(evaluator.policy, replacement.policy)
        with self.assertRaisesRegex(ValueError, 'policy mismatch'):
            replacement.evaluate(self.models[1], self.corpus, reference, cache_directory=cache)
        self.assertEqual(len(self.evaluations), 1)

    def test_full_token_alignment_and_nonfinite_native_outputs_rejected(self):
        evaluator = self.evaluator()
        reference, _ = self.reference(evaluator)
        mutations = {
            'input index': lambda rows: rows[-1].update(input_index=0),
            'context wrap': lambda rows: rows[2].update(context_position=2),
            'last target': lambda rows: rows[-1].update(target_token=0),
            'missing target': lambda rows: rows.pop(),
            'extra target': lambda rows: rows.append(dict(rows[-1])),
            'nan': lambda rows: rows[-1].update(kld='nan'),
            'positive infinity': lambda rows: rows[-1].update(kld='inf'),
            'negative infinity': lambda rows: rows[-1].update(kld='-inf'),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                self.rows_mutation = mutation
                cache = self.directory / name.replace(' ', '-')
                with self.assertRaisesRegex(ValueError, 'alignment|count|invalid native token KL'):
                    evaluator.evaluate(self.models[1], self.corpus, reference, cache_directory=cache)
                self.assertFalse(cache.exists(), 'invalid outputs must not publish cached scores')
        self.rows_mutation = None
        self.assertEqual(evaluator.evaluate(self.models[1], self.corpus, reference), .2)

    def test_interrupted_cuda_checkpoint_reuses_scores_and_rejects_changed_policy(self):
        evaluator = self.evaluator()
        reference, manifest = self.reference(evaluator)
        provenance = dict(policy=evaluator.policy, reference_sha256=file_digest(reference),
                          reference_manifest=digest(manifest), source_sha256=file_digest(self.models[0]))
        initial = State.create({'weight': '0'}, provenance)
        groups = [Group('weight', ('weight',), ({'weight': '1'}, {'weight': '2'}))]
        checkpoint = self.directory / 'checkpoint.json'

        def evaluate(state):
            model = self.models[int(dict(state.assignment)['weight'])]
            return evaluator.evaluate(model, self.corpus, reference)

        self.interrupt_model = self.models[2]
        with self.assertRaisesRegex(TimeoutError, 'synthetic native interruption'):
            coordinate_descent(initial, groups, evaluate, checkpoint=checkpoint)
        saved = json.loads(checkpoint.read_bytes())
        self.assertEqual(len(saved['scores']), 2)
        self.assertEqual(saved['trace'][-1]['status'], 'failed')
        result = coordinate_descent(initial, groups, evaluate, checkpoint=checkpoint)
        self.assertEqual(dict(result.state.assignment), {'weight': '2'})
        self.assertEqual(result.score, .1)
        self.assertEqual(result.evaluations, 3)
        self.assertEqual(self.evaluations.count(self.models[0]), 1)
        self.assertEqual(self.evaluations.count(self.models[1]), 1)
        self.assertEqual(self.evaluations.count(self.models[2]), 2)
        before = list(self.evaluations)
        repeated = coordinate_descent(initial, groups, evaluate, checkpoint=checkpoint)
        self.assertEqual(repeated.state, result.state)
        self.assertEqual(self.evaluations, before)
        changed = State.create({'weight': '0'}, dict(provenance, policy=self.evaluator(flash_attention='off').policy))
        with self.assertRaisesRegex(ValueError, 'checkpoint provenance'):
            coordinate_descent(changed, groups, evaluate, checkpoint=checkpoint)
        self.assertEqual(self.evaluations, before)
        self.assert_payload_integrity()


if __name__ == '__main__':
    unittest.main()
