#!/usr/bin/env python3
"""Actual GGML encode/decode, serialized compensation and no-encode rescore."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
import numpy as np
from tools.sinq.research.native_codec import request, file_sha256

class Bridge(unittest.TestCase):
    def test_actual_codecs_and_rescore(self):
        codecs=('iq1_s','iq1_m','iq2_xxs','iq2_xs','iq2_s','iq3_xxs','iq3_s','iq4_nl','iq4_xs','q3_k','q4_k','q5_0','q5_1','q6_k','q8_0')
        with tempfile.TemporaryDirectory(prefix='sinq-bridge-test-') as tmp:
            d=Path(tmp); n,k,samples=3,256,4
            w=np.sin(np.arange(n*k,dtype=np.float32)*.17).reshape(n,k)
            x=np.cos(np.arange(samples*k,dtype=np.float32)*.31).reshape(samples,k)
            w.tofile(d/'w.bin');x.tofile(d/'x.bin');np.ones(k,dtype='<f4').tofile(d/'im.bin')
            for codec in codecs:
                with self.subTest(codec=codec):
                    work=d/codec;work.mkdir()
                    cfg=dict(schema_version=1,operation='generate',weights=str(d/'w.bin'),source_type='f32',
                        activations=str(d/'x.bin'),samples=samples,importance=str(d/'im.bin'),
                        rows=n,columns=k,codec=codec,output=str(work/'out'),threads=24,
                        strip_width=64,iterations=16,strength=.75,workspace_bytes=4<<20,max_output_bytes=4<<20,
                        candidates=[dict(family='awq_weight_activation',alpha=.5,**{'lambda':1}),
                                    dict(family='awq_weight_activation',alpha=.5,**{'lambda':.5}),
                                    dict(family='awq_activation_only',alpha=.5,**{'lambda':.5}),
                                    dict(family='awq_activation_only',alpha=0,**{'lambda':1}),
                                    dict(family='awq_weight_activation',alpha=.2,**{'lambda':0})])
                    result=request(BINARY,cfg,work)
                    self.assertEqual(len(result['candidates']),6)
                    self.assertEqual(result['encode_count'],4)
                    serial=work/'serial';serial.mkdir()
                    reference=request(BINARY,{**cfg,'threads':1,'output':str(serial/'out')},serial)
                    self.assertEqual(result['selected'],reference['selected'])
                    self.assertEqual(result['candidates'],reference['candidates'])
                    for item in result['candidates']:
                        for key in ('payload','scales'):
                            self.assertEqual(file_sha256(work/'out'/item[key]),file_sha256(serial/'out'/item[key]))
                    baseline=result['candidates'][0]
                    for item in result['candidates']:
                        packed=work/'out'/item['payload']; scales=work/'out'/item['scales']
                        self.assertTrue(packed.is_file());self.assertEqual(scales.stat().st_size,k*2)
                        if item['lambda']==0 or (item['family']=='awq_activation_only' and item['alpha']==0):
                            self.assertEqual(item['payload'],baseline['payload']);self.assertEqual(item['scales'],baseline['scales'])
                        decdir=work/f"decode-{item['index']}";decdir.mkdir()
                        dec=dict(schema_version=1,operation='decode',codec=codec,rows=n,columns=k,
                                 payload=str(packed),scales=str(scales),output=str(decdir/'w.f32'))
                        request(BINARY,dec,decdir)
                        effective=np.fromfile(decdir/'w.f32',dtype='<f4').reshape(n,k)
                        # The bridge output is FP32 effective weights; scorer multiplies
                        # decoded FP32 by F16 in FP64, so reconstruct those operations.
                        u=np.fromfile(scales,dtype='<f2').astype(np.float64)
                        rawdir=work/f"raw-{item['index']}";rawdir.mkdir()
                        dec.pop('scales');dec['output']=str(rawdir/'w.f32');request(BINARY,dec,rawdir)
                        decoded=np.fromfile(rawdir/'w.f32',dtype='<f4').reshape(n,k)
                        np.testing.assert_array_equal(effective,(decoded*u).astype(np.float32))
                        original_hash=file_sha256(packed)
                        score_dir=work/f"score-{item['index']}";score_dir.mkdir()
                        score={**cfg,'operation':'score','payload':str(packed),'scales':str(scales),
                               'output':str(score_dir/'score.json'),'objective':'output_mse','candidates':[]}
                        scored=request(BINARY,score,score_dir)
                        error=x.astype(np.float64)@(decoded.astype(np.float64)*u-w.astype(np.float64)).T
                        self.assertAlmostEqual(scored['train_score'],float((error**2).mean()),delta=1e-11)
                        self.assertEqual(scored['encode_count'],0);self.assertEqual(original_hash,file_sha256(packed))
                    bad=work/'bad';bad.mkdir()
                    proc=subprocess.run([BINARY,str(work/'request.json')],capture_output=True)
                    self.assertNotEqual(proc.returncode,0,'overwrite must fail')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bridge',required=True);args,rest=parser.parse_known_args()
    BINARY=str(Path(args.bridge).resolve())
    unittest.main(argv=[sys.argv[0],*rest])
