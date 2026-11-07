"""Bounded parity gate: actual CUDA persistent vs independent canonical GGUF CLI.

Never evaluates final-test data. No historical manifest is edited.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
sys.path[:0]=[str(REPO),str(REPO/'gguf-py')]
from tools.sinq.optimizer.schemas import load_config
from tools.sinq.optimizer.stages import _run,_validator,_kl_evaluator
from tools.sinq.optimizer.prepared import PreparedAssembly
from tools.sinq.optimizer.prepared_kl import PreparedGroupwiseEvaluator,token_metrics
from tools.sinq.optimizer.selector import State
from tools.sinq.optimizer.assembly import assemble
from tools.sinq.optimizer.resources import observe_performance,canonical_json,file_digest

p=argparse.ArgumentParser();p.add_argument('recipe',choices=['T340','T425']);p.add_argument('--attempt',default='parity-v1');p.add_argument('--gpu',type=int,choices=[0,1],default=0);p.add_argument('--threads',type=int,default=32);p.add_argument('--all-stock-roles',action='store_true');a=p.parse_args()
config=load_config(ROOT/'configs'/f'{a.recipe}-legacy2-v2.json','select')
e,reference_oracle=_kl_evaluator(config)
from tools.sinq.optimizer.kl import NativeKLEvaluator
oracle=NativeKLEvaluator(e['executable'],context=e['context'],batch=e['batch'],microbatch=e['microbatch'],max_tokens=e['max_tokens'],timeout=e['timeout'],threads=a.threads,execution={**e['execution'],'cuda_device':a.gpu})
out=ROOT/'parity'/a.recipe/a.attempt
out.mkdir(parents=True,exist_ok=False)
result={'recipe':a.recipe,'status':'RUNNING','comparisons':[],'effective_scorer_policy':oracle.policy,'reference_origin_policy':reference_oracle.policy}
try:
 with observe_performance() as performance, _run(config) as (run,store,source), PreparedAssembly(source,config['stock_baseline_manifest'],threads=config['threads']) as prepared:
  assignment=dict(config['baseline']);names=sorted(assignment);first,second=names[:2]
  # Test a stock transition, two simultaneous changes, rejection/restore, and a
  # conditioned alternative after the stock graph. All proposals remain full model KL.
  cases=[('initial',dict(assignment))]
  one=dict(assignment);one[first]=run['tensors'][first]['stock'];cases.append(('stock',one))
  two=dict(one);two[second]=next(c['candidate_id'] for c in run['tensors'][second]['candidates'] if c['candidate_id']!=assignment[second]);cases.append(('two-changes',two))
  three=dict(two);three[first]=next(c['candidate_id'] for c in run['tensors'][first]['candidates'] if c['candidate_id']!=assignment[first]);cases.append(('stock-to-conditioned',three))
  if a.all_stock_roles:
   representatives={}
   for name in sorted(assignment,key=lambda n:(int(n.split('.')[1]),n)):
    representatives.setdefault(name.split('.')[2],name)
   combined=dict(assignment)
   for role,name in sorted(representatives.items()):
    stock=dict(assignment);stock[name]=run['tensors'][name]['stock'];cases.append(('stock-role-'+role,stock))
    combined[name]=run['tensors'][name]['stock']
   cases.append(('all-representative-roles-stock',combined))
   late=dict(assignment)
   for name in sorted(assignment,key=lambda n:(int(n.split('.')[1]),n))[-2:]:
    late[name]=next(c['candidate_id'] for c in run['tensors'][name]['candidates'] if c['candidate_id']!=assignment[name])
   cases.append(('two-late-layer-changes',late))
  cases.append(('restore-initial',dict(assignment)))
  adapter=PreparedGroupwiseEvaluator(oracle,REPO/'build-sinq-clean/bin/llama-sinq-groupwise.exe',e['corpus'],e['reference'],native_loader=e['native_loader'],reference_policy=reference_oracle.policy)
  with adapter:
   t=time.perf_counter();adapter.start(prepared,store,assignment,out/'runtime',max_output_bytes=config['budgets']['max_output_bytes']);result['prepare_seconds']=time.perf_counter()-t
   for label,assignment in cases:
    state=State.create(assignment,{'parity_gate':a.recipe})
    t=time.perf_counter();fast=adapter.evaluate(state);fast_time=time.perf_counter()-t
    _,fast_rows=token_metrics(adapter.last_execution['token_output'],adapter.header['tokens'],oracle.context)
    with tempfile.TemporaryDirectory(prefix='.oracle-',dir=out) as temp:
     model=Path(temp)/'canonical.gguf';rows=out/(label+'-oracle.tsv')
     t=time.perf_counter();sha=assemble(source,model,store,assignment,stock_baseline=config['stock_baseline_manifest'],prepared=prepared,native_validate=_validator(e['native_loader'],threads=config['threads']),threads=config['threads'],max_output_bytes=config['budgets']['max_output_bytes']);assembly_seconds=time.perf_counter()-t
     t=time.perf_counter();oracle._run(model,e['corpus'],e['reference'],rows);native_seconds=time.perf_counter()-t
     slow,slow_rows=token_metrics(rows,adapter.header['tokens'],oracle.context)
    diffs={key:float(np.max(np.abs(np.asarray([r[key] for r in fast_rows])-np.asarray([r[key] for r in slow_rows])))) for key in ('kld','base_nll','candidate_nll','top1_agree','candidate_top1')}
    record={'case':label,'canonical_sha256':sha,'assignment':assignment,'persistent_selection_kld':fast,'canonical_selection_kld':slow,'per_token_max_absolute_difference':diffs,'exact_per_token_parity':all(v==0 for v in diffs.values()),'persistent_seconds':fast_time,'canonical_assembly_seconds':assembly_seconds,'canonical_native_seconds':native_seconds,'persistent_execution':adapter.last_execution,'canonical_execution':oracle.last_execution}
    result['comparisons'].append(record)
    (out/'results.json').write_bytes(canonical_json(result))
    print(json.dumps({k:v for k,v in record.items() if k not in ('assignment','persistent_execution','canonical_execution')}),flush=True)
  result['performance']=performance
 result['status']='EXACT_PARITY' if all(r['exact_per_token_parity'] for r in result['comparisons']) else 'NUMERICAL_DIFFERENCES_REQUIRE_REVIEW'
except Exception as error:
 result.update(status='FAILED',error_type=type(error).__name__,error=str(error))
 raise
finally:
 (out/'results.json').write_bytes(canonical_json(result))
print(result['status'],flush=True)
