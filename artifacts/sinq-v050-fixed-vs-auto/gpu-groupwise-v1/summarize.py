"""Read-only closure checks and tables for the bounded GPU Groupwise extension."""
import csv
import json
import math
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
AB=HERE.parent
REPO=AB.parents[1]
sys.path[:0]=[str(REPO),str(REPO/'gguf-py')]
from tools.sinq.optimizer.resources import digest,file_digest
from tools.sinq.optimizer.implementation import implementation_fingerprint
from tools.sinq.optimizer.schemas import read_json


def check_checksum(record):
    if record['checksum']!=digest({k:v for k,v in record.items() if k!='checksum'}):
        raise ValueError('artifact checksum mismatch')


def timing(evidence):
    result={}
    for line in evidence['timings']:
        for key,value in re.findall(r'(\w+_ms)=([0-9.]+)',line):
            result[key.removesuffix('_ms')+'_seconds']=float(value)/1000
        match=re.search(r'load time\s*=\s*([0-9.]+) ms',line)
        if match:result['load_seconds']=float(match[1])/1000
    return result


def main():
    implementation=implementation_fingerprint()
    result={'implementation_sha256':implementation['checksum'],'performance':[],
            'selection':[],'numerical':[],'held_out':[],'input_hashes':{},'warnings':[]}
    for recipe in ('T340','T425'):
        for name in ('cpu13-v2','cuda13-v2','cuda27-v2'):
            directory=HERE/recipe/name
            selected=read_json(directory/'selection.json');identity=read_json(directory/'run-identity.json')
            check_checksum(selected);check_checksum(identity)
            if identity['implementation_sha256']!=implementation['checksum']:
                raise ValueError('selector implementation drift')
            if file_digest(directory/'best.gguf')!=selected['best_gguf_sha256']:
                raise ValueError('selected GGUF changed')
            if selected['run_identity_sha256']!=identity['checksum']:
                raise ValueError('selection run binding mismatch')
            events=selected['trace'];evaluated={e['state']:e for e in events if e['status']=='evaluated'}
            first=next(e for e in events if e['status']=='evaluated');current=first['assignment'];groups={}
            for e in events[1:]:
                if e['status']=='accepted':current=evaluated[e['state']]['assignment']
                if e['status']=='evaluated':
                    changed=[k for k,v in e['assignment'].items() if v!=current[k]]
                    if len(changed)!=1:raise ValueError('unexpected non-singleton proposal')
                    groups[changed[0]]=groups.get(changed[0],0)+1
            caches=[read_json(p) for p in (directory/'scores').glob('*.json')]
            if len(caches)!=selected['evaluations']:raise ValueError('model score closure count')
            phases=[]
            for cache in caches:
                check_checksum(cache)
                if cache['identity']['policy']!=identity['policy']:raise ValueError('cache policy drift')
                proof=cache['execution_evidence']
                if name.startswith('cuda') and (proof['offloaded_layers']!=33 or proof['total_layers']!=33):
                    raise ValueError('CUDA offload proof mismatch')
                phases.append(timing(proof))
            runtime=read_json(HERE/f'runtime-v2/{recipe}-{name}-select-runtime.json')
            sums={key:math.fsum(p.get(key,0) for p in phases) for key in set().union(*(p.keys() for p in phases))}
            result['performance'].append(dict(recipe=recipe,backend=name,**{k:runtime[k] for k in (
                'selection_total_time_sec','assembly_time_sec','assembly_including_validation_sec',
                'native_validation_time_sec','native_kl_evaluation_time_sec','native_evaluator_process_time_sec',
                'selection_evaluations','parent_peak_working_set_bytes','vram_peak_bytes','cache','performance')},
                peak_child_working_set_bytes=max(x['peak_child_working_set_bytes'] for x in runtime['subprocess_calls']),
                native_timing_sums=sums,time_per_evaluation=runtime['selection_total_time_sec']/selected['evaluations']))
            result['selection'].append(dict(recipe=recipe,backend=name,evaluations=selected['evaluations'],
                groups_visited=len(groups),group_proposals=groups,accepted_changes=sum(e['status']=='accepted' for e in events),
                start_kl=first['score'],final_selection_kl=selected['selection_kld'],budget_exhausted=selected['budget_exhausted'],
                best_sha256=selected['best_gguf_sha256'],run_identity_sha256=identity['checksum'],
                failed_evaluations=sum(e['status']=='failed' for e in events)))
        cpu=next(r for r in result['performance'] if r['recipe']==recipe and r['backend']=='cpu13-v2')
        gpu=next(r for r in result['performance'] if r['recipe']==recipe and r['backend']=='cuda13-v2')
        result.setdefault('speedups',{})[recipe]=dict(end_to_end=cpu['selection_total_time_sec']/gpu['selection_total_time_sec'],
            native_process=cpu['native_evaluator_process_time_sec']/gpu['native_evaluator_process_time_sec'],
            decode_and_logits=cpu['native_timing_sums']['decode_and_logits_seconds']/gpu['native_timing_sums']['decode_and_logits_seconds'])
    for p in sorted((HERE/'numerics-v2').glob('*/parity.json')):
        r=read_json(p)
        for run in r['evaluator_runs']:
            if file_digest(run['token_tsv'])!=run['token_tsv_sha256'] or file_digest(run['log_path'])!=run['log_sha256']:
                raise ValueError('numerical output mutated')
        result['numerical'].append(dict(name=p.parent.name,metrics=r['metric_comparison'],
            top1_field=r['top1_comparison_field'],top1_disagreements=r['top1_agree_comparison']['mismatch_count']))
    for corpus in ('wikitext2','ptb'):
        for recipe in ('T340','T425'):
            progress=read_json(HERE/f'held-out/{corpus}/{recipe}/progress.json')
            if progress['status']!='PASS' or progress['completed_models']!=6:
                raise ValueError('held-out output incomplete')
            for row in progress['results']:
                record=read_json(row['metrics_path'])
                if file_digest(row['metrics_path'])!=row['record_sha256'] or file_digest(row['token_tsv_path'])!=row['token_tsv_sha256']:
                    raise ValueError('held-out output hash mismatch')
                if record['model_sha256_before']!=record['model_sha256_after']:
                    raise ValueError('held-out model mutation')
                if corpus=='wikitext2':kl=record['metrics']['mean_test_KL'];ppl=record['metrics']['WikiPPL']
                else:kl=None;ppl=record['metrics']['ppl']
                result['held_out'].append(dict(corpus=corpus,recipe=recipe,label=row['label'],kld=kl,ppl=ppl,
                    elapsed_seconds=record['elapsed_seconds'],model_sha256=record['model_sha256_after'],
                    record_path=str(Path(row['metrics_path']).relative_to(HERE)),token_tsv_sha256=row['token_tsv_sha256']))
    paths=[AB/f'candidates/{r}/run-manifest.json' for r in ('T340','T425')]
    paths += [AB/'locks'/name for name in ('selection-reference.logits','selection-reference.logits.json','selection-manifest.json',
        'evaluator.json','corpora-scoring-v2.json','final-wikitext-input-lock.json','final-ptb-input-lock.json')]
    paths += [HERE/name for name in ('final-models-frozen.json','final-execution-lock.json')]
    for path in paths:result['input_hashes'][str(path.relative_to(AB))]=file_digest(path)
    output=HERE/'measurements.json'
    if output.exists():raise FileExistsError('immutable summary already exists')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','selection_runs':len(result['selection']),'held_out_measurements':len(result['held_out']),
        'speedups':result['speedups'],'output':str(output)},indent=2))


if __name__=='__main__':main()
