#!/usr/bin/env python3
"""Native-only v2 parameter grid, codec oracle, decode and output rescoring."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request(config, directory, raw=None, ok=True):
    path=directory/'request.json'
    path.write_text(json.dumps(config,allow_nan=False) if raw is None else raw,encoding='utf-8')
    proc=subprocess.run([BINARY,str(path)],capture_output=True,text=True)
    if not ok:
        if proc.returncode==0:
            raise AssertionError('invalid request accepted')
        return
    if proc.returncode:
        raise AssertionError(proc.stderr)
    return json.loads(proc.stdout)


def spec(width=64, iterations=16, strength=1., **kw):
    return dict(generator='sinq',strip_width=width,iterations=iterations,strength=strength,**kw)


class Bridge(unittest.TestCase):
    def test_normalization_cache_equivalence_and_eviction(self):
        with tempfile.TemporaryDirectory(prefix='sinq-cache-equivalence-') as tmp:
            d=Path(tmp);n,k,samples=7,256,3
            w=np.sin(np.arange(n*k,dtype=np.float32)*.17).reshape(n,k)
            x=np.cos(np.arange(samples*k,dtype=np.float32)*.31).reshape(samples,k)
            w.tofile(d/'w.bin');x.tofile(d/'x.bin');(x*.75).tofile(d/'v.bin')
            base=[spec(width,it,s) for width in (32,64,128) for it in (4,8,16) for s in (0.,.5,1.)]
            # Revisit old keys beyond the four-entry FIFO, including both AWQ modes.
            candidates=base+[dict(s,generator='awq',correction='weight_activation',alpha=.5,**{'lambda':.75}) for s in base]+[
                dict(base[1],generator='awq',correction='activation_only',alpha=.5,**{'lambda':1.}),
                dict(base[1],generator='awq',correction='weight_activation',alpha=.5,**{'lambda':0.})]
            cfg=dict(schema_version=2,operation='generate',weights=str(d/'w.bin'),rows=n,columns=k,
                codec='q5_0',activations=str(d/'x.bin'),samples=samples,validation=str(d/'v.bin'),validation_samples=samples,
                held_out=True,objective='output_mse',workspace_bytes=8<<20,max_output_bytes=8<<20,candidates=candidates)
            references=[]
            for i,candidate in enumerate(candidates):
                directory=d/f'isolated-{i}'
                r=request(dict(cfg,output=str(directory),threads=1,candidates=[candidate]),d)['candidates'][0]
                references.append((digest(directory/'candidate-0.f16'),digest(directory/'candidate-0.bin'),r['train_score'],r['validation_score']))
            for threads in (1,32):
                directory=d/f'grid-{threads}';result=request(dict(cfg,output=str(directory),threads=threads),d)
                self.assertLess(result['normalization_calls'],sum(c['strength']!=0 for c in candidates))
                self.assertGreater(result['normalization_cache_hits'],0)
                self.assertLessEqual(result['workspace_bytes'],cfg['workspace_bytes'])
                for i,r in enumerate(result['candidates']):
                    self.assertEqual((digest(directory/f'candidate-{i}.f16'),digest(directory/f'candidate-{i}.bin'),r['train_score'],r['validation_score']),references[i])

    def test_plan_config(self):
        with tempfile.TemporaryDirectory(prefix='sinq-plan-') as tmp:
            d=Path(tmp)
            inventory=[dict(name='blk.0.ffn_up.weight',rows=3,columns=512,qtype='q5_0',source_type='f32'),
                       dict(name='blk.1.attn_q.weight',rows=4,columns=256,qtype='q4_k',source_type='f16'),
                       dict(name='output.weight',rows=4,columns=256,qtype='q5_0',source_type='f32')]
            cfg=dict(schema_version=2,defaults=spec(),targets={'include':'eligible'},
                     overrides=[{'selector':{'layer':1},'parameters':{'strip_width':128,'strength':.5}}])
            config=d/'config.json';inv=d/'inventory.json'
            config.write_text(json.dumps(cfg),encoding='utf-8');inv.write_text(json.dumps(inventory),encoding='utf-8')
            proc=subprocess.run([BINARY,'--plan-config',str(config),str(inv)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            entries=json.loads(proc.stdout)['tensors']
            self.assertEqual([e['name'] for e in entries],[i['name'] for i in inventory[:2]])
            self.assertEqual(entries[0]['parameters'],spec())
            self.assertEqual(entries[1]['parameters'],spec(128,strength=.5))
            self.assertEqual(entries[0]['shape'],[512,3]);self.assertTrue(all(e['locked'] for e in entries))
            cfg['encoding']={'scale_refinement_lambda':.17,'imatrix_exponent':1.73}
            config.write_text(json.dumps(cfg),encoding='utf-8')
            proc=subprocess.run([BINARY,'--plan-config',str(config),str(inv)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            planned=json.loads(proc.stdout)
            self.assertEqual(planned['encoding'],cfg['encoding'])
            self.assertEqual(planned['tensors'],entries)
            cfg.pop('encoding')
            cfg['overrides']*=2;config.write_text(json.dumps(cfg),encoding='utf-8')
            self.assertNotEqual(subprocess.run([BINARY,'--plan-config',str(config),str(inv)],capture_output=True).returncode,0)

    def test_parameter_grid_and_strict_errors(self):
        with tempfile.TemporaryDirectory(prefix='sinq-grid-') as tmp:
            d=Path(tmp);n,k=3,512
            w=np.sin(np.arange(n*k,dtype=np.float32)*.17).reshape(n,k)
            w.tofile(d/'w.bin')
            candidates=[spec(width,it,s) for width in (8,16,32,64,128,256,512) for it in (4,8,16) for s in (0.,.5,1.)]
            cfg=dict(schema_version=2,operation='generate',weights=str(d/'w.bin'),rows=n,columns=k,
                     codec='q5_0',output=str(d/'out'),threads=1,candidates=candidates,workspace_bytes=8<<20,max_output_bytes=8<<20)
            result=request(cfg,d)
            self.assertEqual(result['encode_count'],63)
            self.assertEqual([c['parameters'] for c in result['candidates']],candidates)
            self.assertEqual(len({c['payload'] for c in result['candidates']}),63)
            for item in result['candidates']:
                self.assertIsNone(item['train_score'])
                self.assertIsNone(item['validation_score'])
                u=np.fromfile(d/'out'/item['scales'],dtype='<f2').astype(np.float32)
                self.assertTrue(np.all(np.isfinite(u)&(u>0)))
                if item['parameters']['strength']==0:
                    np.testing.assert_array_equal(u,np.ones(k))
                # Independent native codec oracle consumes W/u directly.
                pre=(w/u).astype('<f4');pre.tofile(d/'pre.bin')
                oracle=dict(schema_version=2,operation='stock',weights=str(d/'pre.bin'),rows=n,columns=k,
                            codec='q5_0',output=str(d/f"oracle-{item['index']}"),threads=1)
                stock=request(oracle,d)
                self.assertEqual(digest(d/'out'/item['payload']),digest(Path(oracle['output'])/stock['payload']))
            invalid=[{**cfg,'strip_width':64},{**cfg,'schema_version':1},{**cfg,'max_shortlist':1},
                     {**cfg,'rows':3.5},{**cfg,'workspace_bytes':-1},{**cfg,'columns':510},
                     {**cfg,'candidates':[]},{**cfg,'candidates':[spec(48)]},
                     {**cfg,'candidates':[spec(64,7)]},{**cfg,'candidates':[spec(strength=1.1)]},
                     {**cfg,'candidates':[{**spec(),'unknown':1}]},
                     {**cfg,'candidates':[{**spec(),'generator':'awq','correction':'activation_only','alpha':.5,'lambda':1.}]}]
            for bad in invalid:
                bad['output']=str(d/'invalid')
                request(bad,d,ok=False)
                self.assertFalse((d/'invalid').exists())
                self.assertFalse((d/'invalid.partial').exists())
            text=json.dumps({**cfg,'output':str(d/'duplicate')})
            request(cfg,d,raw=text.replace('"schema_version": 2','"schema_version": 2, "schema_version": 2'),ok=False)
            request(cfg,d,raw=text.replace('"strength": 0.0','"strength": 1e999',1),ok=False)

    def test_actual_codecs_and_rescore(self):
        codecs=('iq1_s','iq1_m','iq2_xxs','iq2_xs','iq2_s','iq3_xxs','iq3_s','iq4_nl','iq4_xs','q2_k','q3_k','q4_k','q5_k','q4_0','q4_1','q5_0','q5_1','q6_k','q8_0')
        with tempfile.TemporaryDirectory(prefix='sinq-bridge-') as tmp:
            d=Path(tmp);n,k,samples=3,256,4
            w=np.sin(np.arange(n*k,dtype=np.float32)*.17).reshape(n,k)
            x=np.cos(np.arange(samples*k,dtype=np.float32)*.31).reshape(samples,k)
            w.tofile(d/'w.bin');x.tofile(d/'x.bin');np.ones(k,dtype='<f4').tofile(d/'im.bin')
            awq=lambda correction,alpha,lam:{**spec(strength=.75),'generator':'awq','correction':correction,'alpha':alpha,'lambda':lam}
            candidates=[spec(strength=.75),awq('weight_activation',.5,1.),awq('weight_activation',.5,.5),awq('activation_only',.5,.5),awq('activation_only',0.,1.),awq('weight_activation',.2,0.)]
            for codec in codecs:
                with self.subTest(codec=codec):
                    work=d/codec;work.mkdir()
                    cfg=dict(schema_version=2,operation='generate',weights=str(d/'w.bin'),source_type='f32',
                             activations=str(d/'x.bin'),samples=samples,importance=str(d/'im.bin'),rows=n,columns=k,
                             codec=codec,output=str(work/'out'),threads=4,workspace_bytes=4<<20,max_output_bytes=4<<20,candidates=candidates)
                    result=request(cfg,work)
                    self.assertEqual(len(result['candidates']),6);self.assertEqual(result['encode_count'],6)
                    serial=request({**cfg,'threads':1,'output':str(work/'serial')},work)
                    self.assertEqual(result['candidates'],serial['candidates'])
                    for item in result['candidates']:
                        packed=work/'out'/item['payload'];scales=work/'out'/item['scales']
                        for key in ('payload','scales'):
                            self.assertEqual(digest(work/'out'/item[key]),digest(work/'serial'/item[key]))
                        u=np.fromfile(scales,dtype='<f2').astype(np.float32)
                        (w/u).astype('<f4').tofile(work/'pre.bin')
                        (u*u).astype('<f4').tofile(work/'im.bin')
                        oracle=dict(schema_version=2,operation='stock',weights=str(work/'pre.bin'),importance=str(work/'im.bin'),
                                    rows=n,columns=k,codec=codec,output=str(work/f"oracle-{item['index']}"),threads=1)
                        stock=request(oracle,work)
                        self.assertEqual(digest(packed),digest(Path(oracle['output'])/stock['payload']))
                        raw=work/f"raw-{item['index']}.f32";effective=work/f"effective-{item['index']}.f32"
                        dec=dict(schema_version=2,operation='decode',codec=codec,rows=n,columns=k,payload=str(packed),output=str(raw))
                        request(dec,work);request({**dec,'scales':str(scales),'output':str(effective)},work)
                        decoded=np.fromfile(raw,dtype='<f4').reshape(n,k)
                        np.testing.assert_array_equal(np.fromfile(effective,dtype='<f4').reshape(n,k),(decoded*u).astype(np.float32))
                        original_hash=digest(packed)
                        for objective in ('output_l1','output_mse'):
                            score=dict(schema_version=2,operation='score',weights=str(d/'w.bin'),activations=str(d/'x.bin'),samples=samples,
                                       payload=str(packed),scales=str(scales),rows=n,columns=k,codec=codec,objective=objective,
                                       output=str(work/f"score-{item['index']}-{objective}.json"),threads=4)
                            scored=request(score,work)
                            error=x.astype(np.float64)@(decoded.astype(np.float64)*u.astype(np.float64)-w.astype(np.float64)).T
                            expected=float(np.abs(error).mean() if objective=='output_l1' else (error**2).mean())
                            self.assertAlmostEqual(scored['train_score'],expected,delta=1e-11)
                            self.assertEqual(scored['encode_count'],0);self.assertEqual(original_hash,digest(packed))
                    # Endpoint equal payloads retain distinct parameter descriptors/paths.
                    for index in (4,5):
                        self.assertNotEqual(result['candidates'][index]['parameters'],result['candidates'][0]['parameters'])
                        for key in ('payload','scales'):
                            self.assertEqual(digest(work/'out'/result['candidates'][index][key]),digest(work/'out'/result['candidates'][0][key]))
                    request(cfg,work,ok=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bridge',required=True);args,rest=parser.parse_known_args()
    BINARY=str(Path(args.bridge).resolve())
    unittest.main(argv=[sys.argv[0],*rest])
