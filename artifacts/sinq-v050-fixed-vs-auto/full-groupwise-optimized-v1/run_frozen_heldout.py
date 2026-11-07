"""Run existing final evaluators, one sequential corpus stream per physical GPU."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]

def run(recipe,gpu):
    rows=[]
    for dataset,script in [('wikitext2','run_final_wikitext.py'),('ptb','run_final_ptb.py')]:
        command=[sys.executable,str(ROOT.parent/script),recipe,str(gpu),'--models-manifest',str(ROOT/'frozen-final-models.json'),
                 '--execution-lock',str(ROOT/'final-execution-lock.json'),'--output-directory',str(ROOT/'held-out'/dataset)]
        start=time.monotonic();utc=datetime.now(timezone.utc).isoformat()
        with (ROOT/f'heldout-{dataset}-{recipe}-attempt2.log').open('wb') as log:
            result=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        row={'dataset':dataset,'recipe':recipe,'command':command,'start_utc':utc,'elapsed_seconds':time.monotonic()-start,'returncode':result.returncode}
        rows.append(row)
        (ROOT/f'heldout-launch-{recipe}.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(row),flush=True)
        if result.returncode:return False
    return True
with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(run,recipe,gpu) for recipe,gpu in [('T340',0),('T425',1)]]
    results=[f.result() for f in as_completed(futures)]
if not all(results):raise SystemExit(1)
print('BOTH_RECIPES_FULL_HELD_OUT_RUNNERS_COMPLETE',flush=True)
