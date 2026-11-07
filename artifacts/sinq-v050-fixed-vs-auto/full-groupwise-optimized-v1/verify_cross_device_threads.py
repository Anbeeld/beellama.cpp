"""Actual per-token 16/32-thread and physical-GPU parity on selection ONLY."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
sys.path[:0]=[str(REPO),str(REPO/'gguf-py')]
from tools.sinq.optimizer.stages import _kl_evaluator
from tools.sinq.optimizer.schemas import load_config
from tools.sinq.optimizer.prepared import VerifiedInputs
from tools.sinq.optimizer.prepared_kl import token_metrics
from tools.sinq.optimizer.resources import canonical_json,file_digest

comparisons=[]
for case in ('initial','stock','two-changes','stock-to-conditioned','restore-initial'):
 paths=[ROOT/'parity/T340'/attempt/(case+'-oracle.tsv') for attempt in ('parity-v3','parity-threads16')]
 import csv
 data=[list(csv.DictReader(p.open(newline=''),delimiter='\t')) for p in paths]
 exact=data[0]==data[1]
 comparisons.append({'kind':'T340-16-vs-32-threads','case':case,'exact_all_token_fields':exact,
     'targets':len(data[0]),'token_output_sha256':[file_digest(p) for p in paths]})

config=load_config(ROOT/'configs/T425-legacy2-v2.json','select');e,oracle=_kl_evaluator(config)
prior=ROOT/'parity/T425/parity-gpu1';record=json.loads((prior/'results.json').read_text())
model=prior/'runtime/starting.gguf';ref_manifest=json.loads(Path(e['reference']+'.json').read_text())
rows=ROOT/'parity/T425/cross-device-gpu0.tsv'
with VerifiedInputs() as inputs:
 model_sha=inputs.fingerprint(model,record['comparisons'][0]['canonical_sha256'])
 inputs.fingerprint(e['reference'],ref_manifest['reference_sha256'])
 inputs.fingerprint(e['corpus'],ref_manifest['corpus_sha256'])
 oracle._run(model,e['corpus'],e['reference'],rows)
 _,gpu0=token_metrics(rows,ref_manifest['alignment']['tokens'],oracle.context)
 _,gpu1=token_metrics(prior/'initial-oracle.tsv',ref_manifest['alignment']['tokens'],oracle.context)
 inputs.check()
 comparisons.append({'kind':'T425-physical-GPU0-vs-GPU1','case':'initial','exact_all_token_fields':gpu0==gpu1,
     'targets':len(gpu0),'model_sha256':model_sha,'gpu0_execution':oracle.last_execution,
     'gpu1_execution':record['comparisons'][0]['canonical_execution'],
     'token_output_sha256':[file_digest(rows),file_digest(prior/'initial-oracle.tsv')]})
result={'status':'EXACT_ALL_TOKEN_PARITY' if all(c['exact_all_token_fields'] for c in comparisons) else 'NUMERICAL_DIFFERENCES_REQUIRE_REVIEW',
    'comparisons':comparisons,'policy_note':'Original 564-target selection stream; original reference bytes and locks unchanged; not final-test observations.'}
(ROOT/'cross-device-thread-parity.json').write_bytes(canonical_json(result))
print(result['status'])
if result['status']!='EXACT_ALL_TOKEN_PARITY':raise SystemExit(1)
