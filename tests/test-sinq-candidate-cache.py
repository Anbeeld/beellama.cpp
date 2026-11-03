import sys, unittest, tempfile, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gguf-py')]
from tools.sinq.research.cache import CandidateStore

def provenance(name='blk.0.ffn_down.weight', width=64):
    h = hashlib.sha256(b'fixture').hexdigest()
    return dict(source_sha256=h, tensor_name=name, shape=[2,256], source_qtype='F16', qtype='Q5_0', input_sha256=h, recipe_sha256=h, imatrix_sha256=None, codec={'implementation':'ggml','chunk_rows':2}, sinq={'strip_width':width,'iterations':16,'strength':0.75,'algorithm':'sinkhorn_log','numerics':'f32-f16'}, generator={'family':'ordinary','alpha':0.0,'lambda':0.0}, activation=None, producer={'source_sha256':h,'binary_sha256':h})

class CacheTest(unittest.TestCase):
    def test_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            s=CandidateStore(d)
            # Real Q5_0 layout: 22 bytes per 32 weights, all-zero scales/values.
            p=bytes(2*8*22); scales=np.ones(256,dtype='<f2').tobytes()
            key=s.put(provenance(),p,scales)
            self.assertEqual(s.get(key).payload.read_bytes(),p)
            self.assertEqual(s.put(provenance(),p,scales),key)
            s.get(key).payload.write_bytes(b'bad')
            with self.assertRaises(ValueError): s.get(key)
    def test_native_identity_and_file_ingestion(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); s=CandidateStore(d/'cache'); payload=d/'packed'; scales=d/'scales'
            payload.write_bytes(bytes(352)); scales.write_bytes(np.ones(256,dtype='<f2').tobytes())
            key=s.put_files(provenance(),payload,scales)
            self.assertEqual(s.get(key).payload.read_bytes(),bytes(352))
            with self.assertRaisesRegex(ValueError,'provenance'):
                s.put(provenance(),bytes([1])*352,scales.read_bytes())
            with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                s.get(key,expected={'qtype':'Q6_K'})
            with self.assertRaisesRegex(ValueError,'budget'):
                CandidateStore(d/'small',max_bytes=50).put(provenance(),bytes(352),scales.read_bytes())

    def test_identity_metadata_requires_actual_identity_compensation(self):
        with tempfile.TemporaryDirectory() as d:
            s=CandidateStore(d);p=provenance();p['sinq']={**p['sinq'],'algorithm':'identity','strength':0}
            with self.assertRaisesRegex(ValueError,'identity'):
                s.put(p,bytes(352),np.full(256,2,dtype='<f2').tobytes())
            key=s.put(p,bytes(352),np.ones(256,dtype='<f2').tobytes())
            self.assertEqual(s.get(key).provenance['sinq']['algorithm'],'identity')

    def test_validation(self):
        with tempfile.TemporaryDirectory() as d:
            s=CandidateStore(d,max_bytes=1024)
            with self.assertRaises(ValueError): s.put(provenance(),b'bad',b'bad')
            bad=provenance(); bad['source_sha256']='wrong'
            with self.assertRaises(ValueError): s.put(bad,bytes(352),bytes(16))
            with self.assertRaises(ValueError): s.put(provenance(),bytes(352),np.zeros(256,dtype='<f2').tobytes())

if __name__=='__main__': unittest.main()
