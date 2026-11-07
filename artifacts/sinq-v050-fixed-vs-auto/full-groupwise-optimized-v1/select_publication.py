"""Explicit small publication inventory; never recursively hash model outputs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
files=set()
def add(path):
    if path.name=='publication-files.json':return
    if path.is_file():
        if path.stat().st_size>8<<20:raise ValueError('publication unexpectedly large: '+str(path))
        files.add(path.resolve())
for pattern in ['*.md','*.py','*.ps1','*.json','*.log']:
    for p in ROOT.glob(pattern):add(p)
for directory in ['configs','tests','tables','performance-before-audited','performance-after','performance-before-one-group']:
    for p in (ROOT/directory).iterdir():
        if p.suffix in ('.json','.csv','.log'):add(p)
for recipe in ('T340','T425'):
    add(ROOT/'local-benchmarks'/recipe/'results.json')
    for p in (ROOT/'runtime'/recipe).glob('*.json'):add(p)
    for name in ['progress.json','run-identity.json','final-selection-parity.json','checkpoint.json.gz','selection.json.gz']:
        add(ROOT/'selection'/recipe/name)
    for attempt in ('parity-v2-all-roles','parity-v2-threads16'):
        add(ROOT/'parity'/recipe/attempt/'results.json')
    for dataset in ('wikitext2','ptb'):
        progress=ROOT/'held-out'/dataset/recipe/'progress.json';add(progress)
        for row in json.loads(progress.read_text())['results']:
            record=Path(row['metrics_path']).resolve()
            if not record.is_relative_to(ROOT.resolve()):raise ValueError('record outside experiment')
            add(record)
            log=Path(json.loads(record.read_text())['combined_log_path']).resolve()
            if not log.is_relative_to(ROOT.resolve()):raise ValueError('native log outside experiment')
            add(log)
for mode in ('single','parallel'):
    for p in (ROOT/'concurrency'/mode).glob('*.json'):add(p)
add(ROOT/'concurrency/launch-times.json')
for p in (ROOT/'final-audits/attempt-1').iterdir():
    if p.suffix in ('.json','.stderr'):add(p)
for name in ['final_control_validation.py']:
    add(ROOT.parent/name)
relative=sorted(p.relative_to(REPO.resolve()).as_posix() for p in files)
output=ROOT/'publication-files.json';output.write_text(json.dumps({'purpose':'Only explicit small evidence; no GGUF/logits/temporary TSV or vendored legacy source','files':relative},indent=2)+'\n')
relative.append(output.relative_to(REPO.resolve()).as_posix())
(ROOT/'publication-paths.txt').write_text('\n'.join(relative)+'\n',encoding='utf-8')
print('PUBLICATION_FILES',len(relative),'TOTAL_BYTES',sum((REPO/p).stat().st_size for p in relative))
