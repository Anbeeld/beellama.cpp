"""Audit regressions: exact run assignments and CLI-level selection failure recovery."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
import gguf
import numpy as np
from tools.sinq.research import cli
from tools.sinq.research.cache import CandidateStore
from tools.sinq.research.resources import canonical_json,digest,file_digest
from tools.sinq.research.schemas import validate_config
spec=importlib.util.spec_from_file_location('cache_fixture',ROOT/'tests/test-sinq-candidate-cache.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.d=Path(self.temp.name)
        self.source=self.d/'source.gguf';self.names=['blk.0.ffn_up.weight','blk.0.ffn_gate.weight']
        w=gguf.GGUFWriter(self.source,'qwen35')
        values=np.ones((2,256),dtype='<f2')
        for name in self.names:w.add_tensor(name,values)
        w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
        run_dir=self.d/'run';self.store=CandidateStore(run_dir/'cache');records={};self.ids={}
        self.source_hash=file_digest(self.source)
        for name in self.names:
            p=fixture.provenance(name);p.update(source_sha256=self.source_hash,input_sha256=hashlib.sha256(values.tobytes()).hexdigest())
            ordinary=self.store.put(p,bytes(352),np.ones(256,dtype='<f2').tobytes())
            stock=self.store.put(dict(p,sinq=None,generator={'family':'stock'}),bytes(352))
            self.ids[name]={'ordinary':ordinary,'stock':stock}
            records[name]={'stock':stock,'selected':ordinary,'candidates':[{'candidate_id':ordinary}]}
        self.run=dict(schema_version=1,source_gguf='../source.gguf',source_sha256=self.source_hash,
            cache='cache',recipe_sha256=p['recipe_sha256'],imatrix_sha256=None,tensors=records)
        self.run['checksum']=digest(self.run)
        self.manifest=run_dir/'run-manifest.json';self.manifest.write_bytes(canonical_json(self.run))
        self.budgets=dict(workspace_bytes=1<<20,max_candidates=8,max_output_bytes=1<<20,max_files=100,
            max_tensors=4,max_tokens=32,max_probes=16,max_evaluations=4)
        self.capture=self.d/'capture.json'
        self.capture.write_bytes(canonical_json(dict(schema_version=1,source_sha256=self.source_hash,
            tokens=[[1,2,3]],layers=[1],baseline={self.names[1]:self.ids[self.names[0]]['ordinary']})))

    def test_exact_current_run_candidate_binding(self):
        from tools.sinq.research.provenance import candidate_for
        name=self.names[0];key=self.ids[name]['ordinary']
        self.assertEqual(candidate_for(self.run,self.store,name,key).candidate_id,key)
        with self.assertRaisesRegex(ValueError,'permitted'):
            candidate_for(self.run,self.store,self.names[1],key)
        p=dict(self.store.get(key).provenance);p['generator']={'family':'awq_activation_only','alpha':.5,'lambda':1}
        p['activation']={'source_sha256':'a'*64,'split_sha256':'b'*64}
        unlisted=self.store.put(p,bytes(352),np.ones(256,dtype='<f2').tobytes())
        with self.assertRaisesRegex(ValueError,'permitted'):
            candidate_for(self.run,self.store,name,unlisted)

    def test_selected_inputs_rejected_before_decode_or_graph_load(self):
        config=validate_config(dict(schema_version=1,operation='score',method='blockwise',objective='output_mse',
            candidate_manifest=str(self.manifest),capture_manifest=str(self.capture),output_directory=str(self.d/'scores'),
            checkpoint='unused',native_bridge='unused',context='selected_quantized_inputs',budgets=self.budgets))
        with patch.object(cli,'_advanced_reference') as reference,patch.object(cli.native_codec,'decode') as decode:
            with self.assertRaisesRegex(ValueError,'permitted'):cli.score_config(config)
            reference.assert_not_called();decode.assert_not_called()
        self.assertFalse((self.d/'scores').exists())

    def test_artifact_validation_checks_stock_baseline_closure(self):
        run=copy.deepcopy(self.run)
        run['stock_baseline']={'manifest':'../missing-baseline.json','manifest_sha256':'a'*64,
            'baseline_sha256':'b'*64,'recipe_sha256':run['recipe_sha256']}
        run['checksum']=digest({k:v for k,v in run.items() if k!='checksum'})
        self.manifest.write_bytes(canonical_json(run))
        config=validate_config(dict(schema_version=1,operation='validate',artifact_manifest=str(self.manifest),budgets=self.budgets))
        with self.assertRaises((OSError,ValueError)):cli.validate_config_artifacts(config)

    def selection_config(self):
        executable=self.d/'perplexity';executable.write_bytes(b'version-one')
        loader=self.d/'loader';loader.write_bytes(b'loader-one')
        reference=self.d/'reference.bin';reference.write_bytes(b'reference')
        reference.with_suffix('.bin.json').write_bytes(canonical_json({'source_sha256':self.source_hash}))
        corpus=self.d/'corpus.txt';corpus.write_bytes(b'abc')
        selection=self.d/'selection.json';selection.write_bytes(canonical_json(dict(schema_version=1,split='selection',
            corpus_sha256=file_digest(corpus),reference_sha256=file_digest(reference))))
        evaluator=self.d/'evaluator.json'
        evaluator.write_bytes(canonical_json(dict(schema_version=1,executable=str(executable),reference=str(reference),
            corpus=str(corpus),native_loader=str(loader),context=32,batch=24,microbatch=24,max_tokens=32,timeout=30,
            alternatives={'group':[{name:self.ids[name]['ordinary'] for name in self.names}]})))
        return dict(schema_version=1,operation='select',source_gguf=str(self.source),candidate_manifest=str(self.manifest),
            baseline={name:self.ids[name]['stock'] for name in self.names},groups={'group':self.names},
            selection_manifest=str(selection),selection_split='selection',evaluator=str(evaluator),
            output_directory=str(self.d/'selection-run'),max_passes=1,tolerance=0,budgets=self.budgets)

    def test_cli_resume_failure_recovery_and_identity(self):
        config=self.selection_config();path=self.d/'config.json';path.write_bytes(canonical_json(config))
        calls=[];fail={'once':True}
        class Evaluator:
            def __init__(self,executable,**kwargs):
                self.policy={'binary_sha256':file_digest(executable),'threads':24,'device':'cpu'}
            def evaluate(inner,model,corpus,reference,**kwargs):
                r=gguf.GGUFReader(model)
                conditioned=any(t.name.endswith('.sinq_scale2') for t in r.tensors);r.data._mmap.close()
                calls.append(conditioned)
                if conditioned and fail['once']:
                    fail['once']=False;raise TimeoutError('injected after baseline checkpoint')
                return 1.0 if conditioned else 2.0
        with patch('tools.sinq.research.kl.NativeKLEvaluator',Evaluator),patch.object(cli,'_validator',return_value=lambda p:None):
            with self.assertRaises(TimeoutError):cli.main([str(path)])
            output=Path(config['output_directory'])
            self.assertTrue((output/'checkpoint.json').exists())
            with redirect_stdout(io.StringIO()):cli.main([str(path),'--resume',str(output)])
            result=json.loads((output/'selection.json').read_bytes())
            self.assertEqual(result['assignment'],{name:self.ids[name]['ordinary'] for name in self.names})
            self.assertEqual(calls.count(False),1,'completed baseline was evaluated again')
            before=len(calls)
            with redirect_stdout(io.StringIO()):cli.main([str(path),'--resume',str(output)])
            self.assertEqual(before,len(calls),'completed resume must reuse checkpoint scores')
            with self.assertRaises(FileExistsError):cli.main([str(path)])
            original=copy.deepcopy(config);config['tolerance']=.1;path.write_bytes(canonical_json(config))
            with self.assertRaisesRegex(ValueError,'identity'):cli.main([str(path),'--resume',str(output)])
            path.write_bytes(canonical_json(original))
            for name in ['corpus.txt','perplexity','loader','reference.bin','reference.bin.json','selection.json',
                         'evaluator.json','source.gguf','run/run-manifest.json']:
                artifact=self.d/name;data=artifact.read_bytes();artifact.write_bytes(data+b' ')
                with self.assertRaises((ValueError,RuntimeError)):
                    cli.main([str(path),'--resume',str(output)])
                artifact.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'resume.*directory'):
                cli.main([str(path),'--resume',str(self.d/'elsewhere')])
            # Simulate death after best.gguf publication but before the final
            # summary: all prior scores and immutable model bytes are reused.
            summary_bytes=(output/'selection.json').read_bytes()
            (output/'selection.json').unlink()
            with redirect_stdout(io.StringIO()):cli.main([str(path),'--resume',str(output)])
            self.assertEqual(before,len(calls))
            self.assertEqual((output/'selection.json').read_bytes(),summary_bytes)
            (output/'best.gguf').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'best|collision|checksum'):
                cli.main([str(path),'--resume',str(output)])

    def test_run_ownership_lock_releases_after_process_death(self):
        from tools.sinq.research.run_state import selection_lock
        code=('import sys\nfrom tools.sinq.research.run_state import selection_lock\n'
              'with selection_lock(sys.argv[1]):\n print("locked",flush=True)\n sys.stdin.read()\n')
        process=subprocess.Popen([sys.executable,'-c',code,str(self.d)],cwd=ROOT,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(),'locked')
            with self.assertRaisesRegex(RuntimeError,'already active'):
                with selection_lock(self.d):pass
        finally:
            process.kill();process.wait(timeout=10)
            for stream in (process.stdin,process.stdout,process.stderr):stream.close()
        with selection_lock(self.d):pass


if __name__=='__main__':unittest.main()
