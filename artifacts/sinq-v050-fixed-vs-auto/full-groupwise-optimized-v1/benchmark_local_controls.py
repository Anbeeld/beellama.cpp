"""Three real tensors, all 13 candidates, original local scoring controls.

This is a bounded scorer benchmark, NOT a rerun/timing claim for the entire
historical 248-tensor local stage. No held-out model quality is selected here.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2];AB=ROOT.parent
sys.path[:0]=[str(REPO),str(REPO/'gguf-py')]
import gguf
from tools.sinq.optimizer import native_codec
from tools.sinq.optimizer.schemas import load_config
from tools.sinq.optimizer.stages import _run
from tools.sinq.optimizer.provenance import candidate_for
from tools.sinq.optimizer.prepared import VerifiedInputs
from tools.sinq.optimizer.resources import observe_performance,canonical_json,file_digest

parser=argparse.ArgumentParser();parser.add_argument('recipe',choices=['T340','T425']);args=parser.parse_args()
configs={objective:load_config(AB/'configs'/f'score-{args.recipe}-local_{objective}.json','score') for objective in ('output_l1','output_mse')}
config=configs['output_l1'];names=('blk.0.attn_gate.weight','blk.0.attn_qkv.weight','blk.0.ffn_down.weight')
for c in configs.values():
 for field in ('candidate_manifest','capture_manifest','native_bridge','threads','selection_policy','budgets'):
  if c[field]!=config[field]:raise ValueError('historical local controls differ across objectives')
result={'recipe':args.recipe,'workload':'three-real-tensors-all-stock-plus-12-conditioned-alternatives',
 'tensor_names':list(names),'full_historical_stage_timing':False,'status':'RUNNING','records':[],
 'config_sha256':{k:file_digest(AB/'configs'/f'score-{args.recipe}-local_{k}.json') for k in configs}}
out=ROOT/'local-benchmarks'/args.recipe;out.mkdir(parents=True,exist_ok=False)
try:
 setup=time.perf_counter()
 with _run(config) as (run,store,source),VerifiedInputs() as inputs:
  inputs.fingerprint(source,run['source_sha256']);inputs.fingerprint(config['native_bridge'])
  inputs.fingerprint(config['capture_manifest'])
  capture=native_codec.validate_activations(config['native_bridge'],config['capture_manifest'],source,threads=config['threads'])
  if capture['source_sha256']!=run['source_sha256']:raise ValueError('activation source mismatch')
  validation=None
  if config['selection_policy']=='held_out':
   from tools.sinq.optimizer.schemas import validate_pairing
   validate_pairing(json.loads(Path(config['pairing_manifest']).read_text()),json.loads(Path(config['capture_manifest']).read_text()),json.loads(Path(config['validation_manifest']).read_text()))
   validation=native_codec.validate_activations(config['native_bridge'],config['validation_manifest'],source,threads=config['threads'])
  reader=gguf.GGUFReader(source)
  try:
   tensors={t.name:t for t in reader.tensors}
   for name in names:
    inputs.fingerprint(capture['tensors'][name]['data_file'])
    if validation:inputs.fingerprint(validation['tensors'][name]['data_file'])
   result['setup_seconds_excluded_from_scorer_timings']=time.perf_counter()-setup
   for name in names:
    tensor=tensors[name];record=run['tensors'][name]
    keys=sorted([record['stock'],*[r['candidate_id'] for r in record['candidates']]])
    candidates=[(key,candidate_for(run,store,name,key)) for key in keys]
    if len(candidates)!=13:raise ValueError('incomplete local candidate coverage')
    a=capture['tensors'][name];v=validation['tensors'][name] if validation else None
    for objective in configs:
     calls=[];real=native_codec._run
     def measured(*a,**kw):
      start=time.perf_counter()
      try:return real(*a,**kw)
      finally:calls.append(time.perf_counter()-start)
     with observe_performance() as before,patch.object(native_codec,'_run',side_effect=measured):
      start=time.perf_counter()
      legacy=[(key,native_codec.rescore(config['native_bridge'],candidate,tensor,a,validation=v,objective=objective,
               workspace_bytes=config['budgets']['workspace_bytes'],threads=config['threads'])) for key,candidate in candidates]
      legacy_seconds=time.perf_counter()-start
     before_calls=list(calls);calls.clear();metrics={}
     with observe_performance() as after,patch.object(native_codec,'_run',side_effect=measured):
      start=time.perf_counter()
      batch=native_codec.rescore_many(config['native_bridge'],candidates,tensor,a,validation=v,objective=objective,
               workspace_bytes=config['budgets']['workspace_bytes'],threads=config['threads'],metrics=metrics)
      batch_seconds=time.perf_counter()-start
     if legacy!=batch:raise ValueError('real local score-many results differ from legacy oracle')
     inputs.check()
     item={'tensor':name,'shape':list(reversed(tensor.shape.tolist())),'objective':objective,'threads':config['threads'],
       'selection_policy':config['selection_policy'],'candidate_ids':keys,'candidate_count':len(keys),'scores':batch,
       'legacy_seconds':legacy_seconds,'batched_seconds':batch_seconds,'legacy_process_launches':len(before_calls),
       'batched_process_launches':len(calls),'legacy_native_process_seconds_inclusive':sum(before_calls),
       'batched_native_process_seconds_inclusive':sum(calls),'legacy_source_bytes_copied':before['native_source_export']['bytes_written'],
       'batched_source_bytes_copied':after['native_source_export']['bytes_written'],'legacy_fresh_hash_bytes':before['file_hash']['bytes_hashed'],
       'batched_fresh_hash_bytes':after['file_hash']['bytes_hashed'],'batched_metrics':metrics,'legacy_observations':before,
       'batched_observations':after,'max_absolute_score_difference':0.0,'large_final_reference_hash_bytes':0}
     result['records'].append(item);(out/'results.json').write_bytes(canonical_json(result))
     print(json.dumps({k:item[k] for k in ('tensor','objective','legacy_seconds','batched_seconds','max_absolute_score_difference')}),flush=True)
  finally:reader.data._mmap.close()
 result['status']='EXACT_SCORE_PARITY'
except Exception as error:
 result.update(status='FAILED',error_type=type(error).__name__,error=str(error));raise
finally:(out/'results.json').write_bytes(canonical_json(result))
