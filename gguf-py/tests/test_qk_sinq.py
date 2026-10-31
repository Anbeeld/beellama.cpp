"""Storage inspection checks, independent GGUFWriter and standard Q_K payloads."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf import GGUFWriter, GGMLQuantizationType, GGUFValueType
from gguf.quants import dequantize

ROOT = Path(__file__).resolve().parents[2]


def write(path, kind=None, scales=None, manifest=True, version=1):
    w = GGUFWriter(path, 'qwen35')
    name = 'blk.0.ffn_up.weight'
    if kind is None:
        w.add_tensor(name, np.zeros((2, 256), dtype=np.float32))
    else:
        block = 110 if kind == GGMLQuantizationType.Q3_K else 144
        packed = np.zeros((2, block), dtype=np.uint8)
        # Independent reader/official decoder sees an ordinary 256-weight block.
        decoded = dequantize(packed, kind)
        assert decoded.shape == (2, 256) and np.all(decoded == 0)
        w.add_tensor(name, packed, raw_dtype=kind)
    if scales is not None:
        w.add_tensor(name + '.sinq_scale2', np.asarray(scales, dtype=np.float16))
    if kind is not None:
        w.add_uint32('beellama.qk_sinq.version', version)
        if manifest:
            w.add_array('beellama.qk_sinq.tensors', [name])
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
            for kind in [GGMLQuantizationType.Q3_K, GGMLQuantizationType.Q4_K]:
                write(d/'model.gguf', kind, np.ones(256))
                result = self.inspect(d/'source.gguf', d/'model.gguf', d/'out.json')
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads((d/'out.json').read_text())[0]
                self.assertEqual(record['companions'], 1)
                self.assertEqual(record['base_parameters'], 512)
                self.assertEqual(record['auxiliary_bytes'], 512)
                self.assertEqual(record['tensor_inventory'][0]['dtype'], kind.name)
                self.assertEqual(record['tensor_inventory'][0]['companion']['min'], 1)

    def test_bad_association_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            write(d/'source.gguf')
            cases = [(None, True, 1), (np.ones(255), True, 1),
                     (np.zeros(256), True, 1), (np.full(256, np.nan), True, 1),
                     (np.ones(256), False, 1), (np.ones(256), True, 2)]
            for scales, manifest, version in cases:
                write(d/'model.gguf', GGMLQuantizationType.Q3_K, scales, manifest, version)
                result = self.inspect(d/'source.gguf', d/'model.gguf', d/'out.json')
                self.assertNotEqual(result.returncode, 0, result.stdout)


if __name__ == '__main__':
    unittest.main()
