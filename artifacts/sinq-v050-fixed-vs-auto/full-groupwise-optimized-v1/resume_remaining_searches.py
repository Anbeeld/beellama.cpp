"""Parallel, disk-admitted continuation of unchanged full-search configs."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
records=[];plans=[];total_estimate=0
for recipe in ('T340','T425'):
    config=ROOT/'configs'/f'{recipe}-full248.json';raw=json.loads(config.read_text())
    output=Path(raw['output']['directory']);progress=json.loads((output/'progress.json').read_text())
    if progress['groups_completed']==248 and progress['status'] in ('COMPLETE_FIRST_PASS','COMPLETE_CONVERGED'):
        records.append({'recipe':recipe,'status':'ALREADY_COMPLETE'});continue
    manifest=Path(raw['inputs']['stock_baseline_manifest']);stock=json.loads(manifest.read_text())
    stock_path=(manifest.parent/stock['baseline_path']).resolve()
    remaining=raw['limits']['max_model_evaluations']-progress['evaluations']
    estimate=3*(stock_path.stat().st_size+(32<<20))+(remaining+2)*(1536<<10)+(320<<20)
    total_estimate+=estimate
    command=[sys.executable,str(ROOT.parent/'run_measured_selection.py'),str(config),'--report-directory',str(ROOT/'runtime'/recipe),'--resume']
    item={'recipe':recipe,'status':'ADMITTED','command':command,'starting_groups_completed':progress['groups_completed'],
          'starting_evaluations':progress['evaluations'],'scratch_estimate_bytes':estimate}
    records.append(item);plans.append((item,output))
free=shutil.disk_usage(ROOT).free
if free<total_estimate+(8<<30):raise RuntimeError(f'insufficient parallel scratch: {free} free, {total_estimate} estimate +8GiB reserve')
report={'scope':'Third attempt, parallel after user freed disk space; unchanged configs/policies','free_bytes_before':free,
        'scratch_estimate_bytes':total_estimate,'reserve_bytes':8<<30,'runs':records}
def save():
    path=ROOT/'resume2-launch-times.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');os.replace(temporary,path)
def run(item,output):
    start=time.monotonic();item['status']='RUNNING';item['start_utc']=datetime.now(timezone.utc).isoformat()
    with (ROOT/f"full-search-{item['recipe']}-resume2.log").open('wb') as log:
        result=subprocess.run(item['command'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    item.update(returncode=result.returncode,elapsed_seconds=time.monotonic()-start,end_utc=datetime.now(timezone.utc).isoformat())
    if result.returncode:item['status']='FAILED';return item
    progress=json.loads((output/'progress.json').read_text())
    complete=progress['groups_completed']==248 and progress['completed_passes']>=1 and not progress['budget_exhausted']
    item.update(status='COMPLETE_FIRST_PASS' if complete else 'INCOMPLETE',evaluations=progress['evaluations'],selection_kld=progress['selection_model_kld'])
    return item
start=time.monotonic();save()
with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(run,*p) for p in plans]
    for future in as_completed(futures):
        item=future.result();save();print(json.dumps(item),flush=True)
report['parallel_pair_wall_seconds']=time.monotonic()-start;save()
if any(r['status'] not in ('COMPLETE_FIRST_PASS','ALREADY_COMPLETE') for r in records):raise SystemExit(1)
print('BOTH_FULL_FIRST_PASSES_COMPLETE',flush=True)
