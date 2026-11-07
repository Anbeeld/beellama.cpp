"""Bounded unchanged Blockwise/Fisher loading/decode profile; never production quality."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
sys.path[:0]=[str(REPO),str(REPO/'gguf-py')]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='32',MKL_NUM_THREADS='32')
import gguf
import numpy as np
import torch
from tools.sinq.optimizer import assembly,native_codec
from tools.sinq.optimizer.cache import CandidateStore
from tools.sinq.optimizer.capture import capture_block
from tools.sinq.optimizer.blockwise import block_delta,score_block
from tools.sinq.optimizer.fisher import qwen_probes
from tools.sinq.optimizer.generate import NUMERICS,source_identity
from tools.sinq.optimizer.resources import canonical_json,digest,file_digest,observe_performance,observe_phase

spec=importlib.util.spec_from_file_location('bounded_block_fixture',REPO/'tests/test-sinq-blockwise.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
bridge=REPO/'build-sinq-clean/bin/llama-sinq-native.exe'
result={'scope':'Tiny offline genuine Qwen3.5 fixture, not a 9B throughput estimate','status':'RUNNING','records':[]}
with tempfile.TemporaryDirectory(prefix='sinq-advanced-profile-') as directory:
 d=Path(directory)
 with observe_performance() as setup:
  start=time.perf_counter();adapter=fixture.tiny_adapter(d/'checkpoint');result['reference_loading_seconds']=time.perf_counter()-start
  source=d/'checkpoint/model.gguf';reader=gguf.GGUFReader(source)
  name='blk.0.ssm_out.weight';tensor=next(t for t in reader.tensors if t.name==name)
  weights=d/'weights.bin';native_codec.copy_tensor_bytes(tensor,weights,max_bytes=16<<20)
  shape=list(reversed(tensor.shape.tolist()));g={'generator':'sinq','strip_width':64,'iterations':16,'strength':.75}
  request={'schema_version':2,'operation':'generate','weights':str(weights),'source_type':tensor.tensor_type.name.lower(),
    'rows':shape[0],'columns':shape[1],'codec':'q5_0','output':str(d/'generated'),'workspace_bytes':16<<20,
    'max_output_bytes':16<<20,'threads':32,'candidates':[g]}
  req=d/'request';req.mkdir();made=native_codec.request(bridge,request,req)['candidates'][0]
  p={'source_sha256':file_digest(source),'tensor_name':name,'shape':shape,'source_qtype':tensor.tensor_type.name,
    'qtype':'Q5_0','input_sha256':file_digest(weights),'recipe_sha256':digest({name:'Q5_0'}),'imatrix_sha256':None,
    'codec':{'implementation':'ggml','chunk_rows':shape[0]},
    'sinq':{**{k:g[k] for k in ('strip_width','iterations','strength')},'algorithm':'sinkhorn_log','numerics':NUMERICS},
    'generator':g,'activation':None,'producer':{'source_sha256':source_identity(),'binary_sha256':file_digest(bridge)}}
  store=CandidateStore(d/'cache',max_bytes=16<<20,max_files=100)
  key=store.put(p,(d/'generated'/made['payload']).read_bytes(),(d/'generated'/made['scales']).read_bytes())
  reader.data._mmap.close()
  tokens=torch.tensor([[1,7,3,11,5,19,2]])
  captured=capture_block(adapter,tokens,0)
  probes=qwen_probes(adapter,captured,probes=3,seed=8,positions=[(0,6)])
 result['setup_observations']=setup
 with store.verified_session():
  candidate=store.get(key)
  # If a future implementation assembles whole models here, fail immediately.
  with patch.object(assembly,'assemble',side_effect=AssertionError('advanced scorer assembled a full GGUF')):
   with observe_performance() as trials:
    for i in range(3):
     with observe_phase('bounded_advanced_native_decode'):
      decoded=native_codec.decode(bridge,candidate,rows=shape[0],workspace_bytes=16<<20,threads=32)
     start=time.perf_counter();block=score_block(adapter,captured,{name:decoded});block_seconds=time.perf_counter()-start
     start=time.perf_counter();fisher=probes.score(block_delta(adapter,captured,{name:decoded}));fisher_seconds=time.perf_counter()-start
     result['records'].append({'iteration':i,'block':block,'fisher_score':fisher,'block_seconds':block_seconds,'fisher_seconds':fisher_seconds})
 result['trial_observations']=trials
 result['source_bytes_newly_file_hashed_during_candidate_trials']=sum(v['bytes_hashed'] for path,v in trials.get('file_hash',{}).get('files',{}).items() if Path(path)==source)
 assert result['source_bytes_newly_file_hashed_during_candidate_trials']==0
 assert all(r['block']==result['records'][0]['block'] and r['fisher_score']==result['records'][0]['fisher_score'] for r in result['records'])
 result.update(status='BOUNDED_PROFILE_AND_DETERMINISTIC_SCORES_PASS',full_gguf_assemblies=0,
    native_candidate_decode_launches=3,unchanged_kernel_sources={str(p.relative_to(REPO)):file_digest(p) for p in
    [REPO/'tools/sinq/optimizer/blockwise.py',REPO/'tools/sinq/optimizer/fisher.py',REPO/'tools/sinq/optimizer/qwen35_adapter.py',REPO/'tools/sinq/optimizer/reference_loader.py']},
    limitation='First native activation validation and individual native candidate-decode process boundaries remain explicit; this fixture is not a full-model stage performance claim.')
(ROOT/'advanced-fixture-profile.json').write_bytes(canonical_json(result))
print(result['status'])
