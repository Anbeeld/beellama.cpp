"""Generation recovery at complete tensor boundaries with actual native encoding."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
import gguf
import numpy as np
from tools.sinq import export_activations as exporter
from tools.sinq.research import cli,native_codec
from tools.sinq.research.generate import generate
from tools.sinq.research.resources import canonical_json,file_digest
from tools.sinq.research.schemas import validate_config

class ResumeGuardTests(unittest.TestCase):
    def test_resume_requires_existing_configured_directory_before_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'run'
            with self.assertRaisesRegex(ValueError,'resume.*directory'):
                generate({'output_directory':str(output)},resume=str(output))
            self.assertFalse(output.exists())

@unittest.skipUnless(os.environ.get('SINQ_NATIVE_BIN'),'local native binaries not configured')
class NativeGenerationResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.d=Path(self.temp.name)
        self.names=['blk.0.ffn_up.weight','blk.0.ffn_gate.weight'];self.hashes={}
        source=self.d/'source.gguf';writer=gguf.GGUFWriter(source,'qwen35')
        rng=np.random.default_rng(17)
        for name in self.names:
            weights=rng.normal(size=(2,256)).astype('<f4');writer.add_tensor(name,weights)
            self.hashes[name]=hashlib.sha256(weights.tobytes()).hexdigest()
        writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
        records=[]
        for index,name in enumerate(self.names):
            path=self.d/f'{index}.f32';rng.normal(size=(2,256)).astype('<f4').tofile(path)
            module,site,_=exporter.module_for(name)
            records.append(dict(name=name,site=site,reference_module=module,channel_layout='gguf-input',input_dim=256,output_dim=2,
                count=2,dtype='<f4',order='row-major',data_file=path.name,sha256=file_digest(path)))
        tokens=hashlib.sha256(np.array([1,2],dtype='<u4').tobytes()).hexdigest()
        activation=self.d/'activations.json'
        activation.write_bytes(canonical_json(dict(schema_version=1,architecture='qwen3_5',source_sha256=file_digest(source),source_revision='resume-fixture',
            sampling=dict(contract='valid-token-prefill-reservoir-v1',seed=17,max_rows=2,corpus_sha256=tokens,token_sha256=tokens),tensors=records)))
        bridge=Path(os.environ['SINQ_NATIVE_BIN'])/('llama-sinq-native.exe' if os.name=='nt' else 'llama-sinq-native')
        self.raw=dict(schema_version=1,operation='generate',source_gguf=str(source),activation_manifest=str(activation),tensor_list=self.names,
            native_quantizer=dict(recipe={name:'q5_0' for name in self.names},sinq_strip_width=64,sinq_iterations=4,sinq_strength=.75),
            generators=['ordinary','awq_weight_activation'],alphas=[.5],correction_strengths=[1],
            search=dict(strategy='small_grid',local_objective='output_mse',selection_policy='in_sample',max_shortlist=0,refine_step=.1,tolerance=0),
            output_directory=str(self.d/'run'),native_bridge=str(bridge),threads=32,
            budgets=dict(workspace_bytes=32<<20,max_candidates=8,max_output_bytes=8<<20,max_files=100,max_tensors=2,max_tokens=16,max_probes=4,max_evaluations=16))
        self.config=validate_config(self.raw);self.output=Path(self.raw['output_directory'])

    def test_partial_tensor_recovery_completed_reuse_and_mismatch(self):
        real=native_codec.request;seen=[]
        def interrupt(binary,request,directory):
            identity=file_digest(request['weights']);seen.append((request['operation'],identity))
            if request['operation']=='generate' and identity==self.hashes[self.names[1]]:
                raise TimeoutError('interrupted after incomplete tensor stock publication')
            return real(binary,request,directory)
        with patch.object(native_codec,'request',side_effect=interrupt):
            with self.assertRaises(TimeoutError):generate(self.config)
        state=json.loads((self.output/'generation-checkpoint.json').read_bytes())
        self.assertEqual(set(state['completed']),{self.names[0]})
        calls=[]
        def track(binary,request,directory):
            calls.append(file_digest(request['weights']));return real(binary,request,directory)
        with patch.object(native_codec,'request',side_effect=track):
            result=generate(self.config,resume=self.output)
        self.assertNotIn(self.hashes[self.names[0]],calls,'completed tensor encoded again')
        self.assertEqual(set(result['tensors']),set(self.names))
        path=self.d/'config.json';path.write_bytes(canonical_json(self.raw))
        before=(self.output/'run-manifest.json').read_bytes()
        with patch.object(native_codec,'request',side_effect=AssertionError('completed generation reencoded')):
            with redirect_stdout(io.StringIO()):cli.main([str(path),'--resume',str(self.output)])
        self.assertEqual(before,(self.output/'run-manifest.json').read_bytes())
        changed=copy.deepcopy(self.raw);changed['search']['tolerance']=.1
        with self.assertRaisesRegex(ValueError,'identity'):generate(validate_config(changed),resume=self.output)
        artifact=Path(self.raw['activation_manifest']);content=artifact.read_bytes();artifact.write_bytes(content+b' ')
        with self.assertRaisesRegex(ValueError,'identity'):generate(self.config,resume=self.output)
        artifact.write_bytes(content)
        # Completed bundles remain cryptographically verified, not merely trusted
        # because their names occur in a checkpoint.
        key=result['tensors'][self.names[0]]['stock']
        descriptor=json.loads((self.output/'cache/tensors'/f'{key}.json').read_bytes())
        blob=self.output/'cache/blobs'/descriptor['payload'];data=blob.read_bytes()
        blob.write_bytes(bytes([data[0]^1])+data[1:])
        with self.assertRaisesRegex(ValueError,'checksum'):generate(self.config,resume=self.output)
        self.assertEqual(before,(self.output/'run-manifest.json').read_bytes())

    def test_resumed_cache_must_be_run_owned(self):
        generate(self.config)
        actual=Path.is_symlink
        with patch.object(Path,'is_symlink',side_effect=None,autospec=True) as symlink,patch('tools.sinq.research.cache.CandidateStore') as constructor:
            symlink.side_effect=lambda path:path==self.output/'cache' or actual(path)
            with self.assertRaisesRegex(ValueError,'run-owned'):generate(self.config,resume=self.output)
            constructor.assert_not_called()

    def test_killed_process_releases_run_and_cache_ownership(self):
        path=self.d/'config.json';path.write_bytes(canonical_json(self.raw));(self.d/'child-temp').mkdir()
        code='''import sys,json,tempfile
from pathlib import Path
from tools.sinq.research import cli,native_codec
raw=json.loads(Path(sys.argv[1]).read_text());tempfile.tempdir=str(Path(raw['output_directory']).parent/'child-temp')
real=native_codec.request
count=0
def request(binary,value,directory):
 global count
 if value['operation']=='generate':
  count+=1
  if count==2:
   print('tensor-boundary',flush=True);sys.stdin.read()
 return real(binary,value,directory)
native_codec.request=request
cli.main([sys.argv[1]])
'''
        process=subprocess.Popen([sys.executable,'-c',code,str(path)],cwd=ROOT,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(),'tensor-boundary')
        finally:
            process.kill();process.wait(timeout=10)
            for stream in (process.stdin,process.stdout,process.stderr):stream.close()
        checkpoint=json.loads((self.output/'generation-checkpoint.json').read_bytes())
        self.assertEqual(set(checkpoint['completed']),{self.names[0]})
        result=generate(self.config,resume=self.output)
        self.assertEqual(set(result['tensors']),set(self.names))

if __name__=='__main__':unittest.main()
