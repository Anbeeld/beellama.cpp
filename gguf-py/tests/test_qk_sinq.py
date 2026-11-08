"""Independent storage inspection of the closed SINQ GGUF v2 contract."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf import GGUFWriter, GGMLQuantizationType, GGML_QUANT_SIZES
from gguf.quants import dequantize

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = {'generator': 'sinq', 'iterations': 16, 'strength': 1.0, 'strip_width': 64}


def write(path, kind=None, scales=None, manifest=True, version=2, metadata=None, columns=256, recipe=None):
    w = GGUFWriter(path, 'qwen35')
    name = 'blk.0.ffn_up.weight'
    if kind is None:
        w.add_tensor(name, np.zeros((2, columns), dtype=np.float32))
    else:
        block, size = GGML_QUANT_SIZES[kind]
        packed = np.zeros((2, size * (columns // block)), dtype=np.uint8)
        decoded = dequantize(packed, kind)
        assert decoded.shape == (2, columns) and np.all(decoded == 0)
        w.add_tensor(name, packed, raw_dtype=kind)
    if scales is not None:
        w.add_tensor(name + '.sinq_scale2', np.asarray(scales, dtype=np.float16))
    if kind is not None:
        w.add_uint32('beellama.qk_sinq.version', version)
        if manifest:
            w.add_array('beellama.qk_sinq.tensors', [name])
        w.add_string('beellama.qk_sinq.transform', 'f16_input_diagonal')
        w.add_string('beellama.qk_sinq.producer', 'test')
        recipe = DEFAULT if recipe is None else recipe
        encoded = recipe if isinstance(recipe, str) else json.dumps(recipe, sort_keys=True, separators=(',', ':'))
        w.add_array('beellama.qk_sinq.recipes', [encoded])
    for key, value in (metadata or {}).items():
        if isinstance(value, bool):
            w.add_bool(key, value)
        elif isinstance(value, int):
            w.add_uint32(key, value)
        else:
            w.add_string(key, value)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


class TestQKSidecar(unittest.TestCase):
    def inspect(self, source, model, out):
        return subprocess.run([sys.executable, str(ROOT/'gguf-py/tests/qk_sinq_inventory.py'),
                               '--generic', '--source', str(source), '--output', str(out), str(model)],
                              capture_output=True, text=True)

    def test_standard_payload_and_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            write(d/'source.gguf')
            for kind in [GGMLQuantizationType.Q2_K, GGMLQuantizationType.Q3_K,
                         GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K,
                         GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1,
                         GGMLQuantizationType.Q5_0, GGMLQuantizationType.Q5_1,
                         GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q8_0,
                         GGMLQuantizationType.IQ1_S, GGMLQuantizationType.IQ1_M,
                         GGMLQuantizationType.IQ2_XXS, GGMLQuantizationType.IQ2_XS,
                         GGMLQuantizationType.IQ2_S, GGMLQuantizationType.IQ3_XXS,
                         GGMLQuantizationType.IQ3_S, GGMLQuantizationType.IQ4_NL,
                         GGMLQuantizationType.IQ4_XS]:
                write(d/'model.gguf', kind, np.ones(256))
                result = self.inspect(d/'source.gguf', d/'model.gguf', d/'out.json')
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads((d/'out.json').read_text())[0]
                self.assertEqual(record['companions'], 1)
                self.assertEqual(record['base_parameters'], 512)
                self.assertEqual(record['auxiliary_bytes'], 512)
                self.assertEqual(record['tensor_inventory'][0]['dtype'], kind.name)
                self.assertEqual(record['tensor_inventory'][0]['companion']['min'], 1)

    def test_per_tensor_width(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            write(d/'source.gguf', columns=512)
            for width in (8, 16, 32, 64, 128, 256, 512):
                write(d/'model.gguf', GGMLQuantizationType.Q3_K, np.ones(512), columns=512,
                      recipe={**DEFAULT, 'strip_width': width})
                result = self.inspect(d/'source.gguf', d/'model.gguf', d/'out.json')
                self.assertEqual(result.returncode, 0, result.stderr)
            for width in (0, 48, 1024, '64', True):
                write(d/'model.gguf', GGMLQuantizationType.Q3_K, np.ones(512), columns=512,
                      recipe={**DEFAULT, 'strip_width': width})
                self.assertNotEqual(self.inspect(d/'source.gguf', d/'model.gguf', d/'out.json').returncode, 0)
            write(d/'source.gguf')
            write(d/'model.gguf', GGMLQuantizationType.Q3_K, np.ones(256), recipe={**DEFAULT, 'strip_width': 512})
            self.assertNotEqual(self.inspect(d/'source.gguf', d/'model.gguf', d/'out.json').returncode, 0)

    def test_malformed_recipes(self):
        bad = [json.dumps(DEFAULT), '{"generator":"sinq","iterations":16,"strength":1.0,"strength":1.0,"strip_width":64}',
               {**DEFAULT, 'unexpected': True}, {**DEFAULT, 'strength': float('nan')},
               {**DEFAULT, 'iterations': 3}, {**DEFAULT, 'generator': 'awq'},
               {key: value for key, value in DEFAULT.items() if key != 'strength'}]
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            write(d/'source.gguf')
            for recipe in bad:
                write(d/'model.gguf', GGMLQuantizationType.Q3_K, np.ones(256), recipe=recipe)
                self.assertNotEqual(self.inspect(d/'source.gguf', d/'model.gguf', d/'out.json').returncode, 0, repr(recipe))

    def test_unsupported_sinq_metadata(self):
        cases = [('beellama.qk_sinq.strip_width', 64), ('beellama.qk_sinq.algorithm', 'sinkhorn_log'),
                 ('beellama.qk_sinq.numerics', 'old'), ('beellama.qk_sinq.future_control', True)]
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            for target in ['source', 'model']:
                for key, value in cases:
                    for name in ['source', 'model']:
                        kind = GGMLQuantizationType.Q3_K if name == 'model' else None
                        write(d/f'{name}.gguf', kind, np.ones(256) if kind else None,
                              metadata={key: value} if name == target else None)
                    result = self.inspect(d/'source.gguf', d/'model.gguf', d/'out.json')
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('unsupported SINQ metadata', result.stderr)
                    self.assertIn(key, result.stderr)

    def test_bad_association_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            write(d/'source.gguf')
            cases = [(None, True, 2), (np.ones(255), True, 2), (np.zeros(256), True, 2),
                     (np.full(256, np.nan), True, 2), (np.ones(256), False, 2), (np.ones(256), True, 1)]
            for scales, manifest, version in cases:
                write(d/'model.gguf', GGMLQuantizationType.Q3_K, scales, manifest, version)
                result = self.inspect(d/'source.gguf', d/'model.gguf', d/'out.json')
                self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
