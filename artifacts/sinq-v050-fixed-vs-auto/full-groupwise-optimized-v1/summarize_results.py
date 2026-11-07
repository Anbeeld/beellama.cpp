"""Audit completed frozen evaluations and publish compact coverage/quality evidence.

Only explicit small publications and token-metric TSVs are hashed. GGUFs and the
large probability reference are never rehashed merely to generate this report.
"""
import csv
import io
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
sys.path[:0]=[str(REPO),str(REPO/'gguf-py')]
from tools.sinq.optimizer.resources import file_digest,canonical_json,digest
from tools.sinq.optimizer.schemas import read_json


def relative_output(path,scope):
    path=Path(path).resolve()
    if not path.is_relative_to(scope.resolve()):raise ValueError('evaluation evidence escapes its run directory')
    return path


def evaluation(dataset,recipe,models,lock):
    scope=ROOT/'held-out'/dataset/recipe
    progress=read_json(scope/'progress.json',max_bytes=16<<20)
    expected={m['label']:m for m in models}
    if progress.get('status')!='PASS' or progress.get('completed_models')!=len(expected) or progress.get('expected_models')!=len(expected):
        raise ValueError('held-out evaluation incomplete: '+dataset+'/'+recipe)
    results={r['label']:r for r in progress['results']}
    if set(results)!=set(expected):raise ValueError('evaluated model set differs from frozen model set')
    records={}
    for label,row in results.items():
        path=relative_output(row['metrics_path'],scope)
        if file_digest(path)!=row['record_sha256']:raise ValueError('evaluation record digest mismatch')
        record=read_json(path,max_bytes=16<<20)
        tsv=relative_output(record['token_tsv_path'],scope)
        if file_digest(tsv)!=row['token_tsv_sha256'] or record['token_tsv_sha256']!=row['token_tsv_sha256']:
            raise ValueError('evaluation token evidence digest mismatch')
        if (record.get('status')!='PASS' or record['model_sha256_before']!=expected[label]['expected_sha256'] or
                record['model_sha256_after']!=expected[label]['expected_sha256'] or
                record['executable_sha256']!=lock['executable_sha256'] or record['context']!=65536 or
                record['batch']!=512 or record['microbatch']!=256 or record['cache_type_k']!='bf16' or
                record['cache_type_v']!='bf16' or record['flash_attention']!='on'):
            raise ValueError('held-out model/runtime/policy closure differs')
        metrics=record['metrics']
        if dataset=='wikitext2':
            trust=record['reference_trust']
            if (metrics['targets']!=297193 or trust['fresh_content_hash'] or
                    trust['reference_content_hashes_per_normal_run']!=0 or trust['reference_content_bytes_hashed_per_normal_run']!=0 or
                    trust['file_identity']!=lock['reference_file_identity']):
                raise ValueError('WikiText count or no-hash trust contract mismatch')
            values=(metrics['mean_test_KL'],metrics['WikiPPL'])
        else:
            if metrics['targets']!=98776 or metrics['full_vocabulary_KL'] is not None or record['reference_path'] is not None:
                raise ValueError('PTB count/reference-free PPL contract mismatch')
            values=(metrics['ppl'],)
        if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite final metric')
        records[label]=record
    return records


def main():
    frozen=read_json(ROOT/'frozen-final-models.json',max_bytes=16<<20);lock=read_json(ROOT/'final-execution-lock.json')
    if frozen['status']!='FROZEN_BEFORE_HELD_OUT_EVALUATION' or lock['models_manifest_sha256']!=file_digest(ROOT/'frozen-final-models.json'):
        raise ValueError('final freeze/execution binding mismatch')
    rows=[];coverage_rows=[];evidence=[]
    for recipe in ('T340','T425'):
        models=[m for m in frozen['models'] if m['recipe']==recipe]
        wiki=evaluation('wikitext2',recipe,models,lock);ptb=evaluation('ptb',recipe,models,lock)
        progress=read_json(ROOT/'selection'/recipe/'progress.json',max_bytes=128<<20)
        if progress['eligible_groups']!=248 or progress['groups_completed']!=248 or progress['completed_passes']<1 or progress['budget_exhausted']:
            raise ValueError('report cannot label an incomplete search complete')
        for pass_ in progress['passes']:
            for group in pass_['per_group']:
                for candidate in group['candidates']:
                    coverage_rows.append({'recipe':recipe,'pass':pass_['pass'],'group':group['group'],'group_completed':group['completed'],
                        'candidate_id':next(iter(candidate['assignment'].values())),'group_assignment_identity':candidate['candidate_id'],
                        'full_state_id':candidate['state_id'],'status':candidate['status'],'selection_kld':candidate.get('score'),
                        'selected_state_id':group.get('selected_state_id')})
        order={'LOCKED-STOCK-BASELINE-CONTROL':0,'STOCK-CONTROL':0,'FIXED-ANCHOR':1,'CURRENT-GROUP-KLD-CUDA27':2,'NEW-FULL-248-GROUPWISE':3}
        for model in sorted(models,key=lambda m:(order.get(m['label'],4),m['label'])):
            label=model['label'];w,p=wiki[label],ptb[label]
            groups='248/248' if label=='NEW-FULL-248-GROUPWISE' else ('3 attempted; incomplete' if label=='CURRENT-GROUP-KLD-CUDA27' else 'N/A')
            rows.append({'recipe':recipe,'model':label,'completed_groups':groups,'selection_kld_564':model['selection_kld'],
                'wikitext_mean_kld_297193':w['metrics']['mean_test_KL'],'wikitext_ppl':w['metrics']['WikiPPL'],
                'ptb_ppl_98776':p['metrics']['ppl'],'gguf_sha256':model['expected_sha256'],
                'wikitext_kld_median':w['metrics']['tails']['KL']['median'],'wikitext_kld_p95':w['metrics']['tails']['KL']['p95'],
                'wikitext_kld_p99':w['metrics']['tails']['KL']['p99'],'wikitext_top1_agreement':w['metrics']['top1_agreement'],
                'large_reference_bytes_newly_hashed':0})
            evidence.append({'recipe':recipe,'model':label,'wiki_command':w['command'],'ptb_command':p['command'],
                'wiki_seconds':w['elapsed_seconds'],'ptb_seconds':p['elapsed_seconds'],'gpu_uuid':w['gpu_uuid'],
                'wiki_peak_working_set_bytes':w['peak_process_working_set_bytes'],'ptb_peak_working_set_bytes':p['peak_process_working_set_bytes'],
                'wiki_gpu_global_peak_mib':w['gpu_global_memory_peak_mib'],'ptb_gpu_global_peak_mib':p['gpu_global_memory_peak_mib'],
                'executable_sha256':w['executable_sha256'],'reference_trust':w['reference_trust'],
                'wiki_token_tsv_sha256':w['token_tsv_sha256'],'ptb_token_tsv_sha256':p['token_tsv_sha256']})
    tables=ROOT/'tables';tables.mkdir(exist_ok=True)
    for name,data in [('final-kld-ppl.csv',rows),('selection-248-coverage.csv',coverage_rows)]:
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        (tables/name).write_text(stream.getvalue(),encoding='utf-8',newline='')
    result={'status':'BOTH_FULL_PASSES_FROZEN_AND_FULL_CORPORA_EVALUATED','models':rows,'evidence':evidence,
        'frozen_models_sha256':file_digest(ROOT/'frozen-final-models.json'),'execution_lock_sha256':file_digest(ROOT/'final-execution-lock.json'),
        'selection_coverage_csv_sha256':file_digest(tables/'selection-248-coverage.csv'),
        'full_reference_content_hashes':0,'full_reference_content_hash_bytes':0,
        'warning':'One-pass coordinate descent on 564 optimization targets is not global optimality. Later accepted changes can invalidate early coordinate minima. Prior historical final-test arms were already observed; this is not completely blind holdout validation.'}
    (ROOT/'final-results.json').write_bytes(canonical_json(result))
    lines=['# SINQ end-to-end refactor and exhaustive first-pass A/B','',
        'Both T340 and T425 completed all 248 eligible groups in their first coordinate-descent pass. Every group accounted for stock plus all 12 pre-generated conditioned candidates. The two canonical GGUFs were independently audited and frozen before the new held-out runs.',
        '', '## Measured quality','',
        '| Recipe | Model | Completed groups | Selection KLD (564) | WikiText KLD (297,193) | WikiText PPL | PTB PPL (98,776) | GGUF SHA-256 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['recipe']} | {r['model']} | {r['completed_groups']} | {r['selection_kld_564']:.9f} | {r['wikitext_mean_kld_297193']:.9f} | {r['wikitext_ppl']:.6f} | {r['ptb_ppl_98776']:.6f} | `{r['gguf_sha256']}` |")
    by_key={(r['recipe'],r['model']):r for r in rows}
    lines+=['','### Interpretation','']
    for recipe in ('T340','T425'):
        new=by_key[(recipe,'NEW-FULL-248-GROUPWISE')];anchor=by_key[(recipe,'FIXED-ANCHOR')]
        selection_change=100*(new['selection_kld_564']/anchor['selection_kld_564']-1)
        wiki_change=100*(new['wikitext_ppl']/anchor['wikitext_ppl']-1)
        ptb_change=100*(new['ptb_ppl_98776']/anchor['ptb_ppl_98776']-1)
        kl_change=new['wikitext_mean_kld_297193']-anchor['wikitext_mean_kld_297193']
        lines.append(f"{recipe}: selection KLD changed {selection_change:+.3f}% versus the fixed anchor. Full WikiText KLD changed {kl_change:+.9f} nats, WikiText PPL {wiki_change:+.3f}%, and PTB PPL {ptb_change:+.3f}%. Lower is better for all four metrics. Improved selection KLD did not produce a general held-out PPL improvement; no statistical significance or reserved-validation stability estimate is available.")
    lines+=['', 'Selection KLD optimizes the original 564-target stream. Full WikiText KLD and WikiText PPL use all 297,193 locked targets; PTB PPL uses all 98,776 targets. PTB has no locked full-vocabulary BF16 reference, so PTB KLD is deliberately not reported. Median/tail WikiText KLD, top-1 agreement, exact commands and runtime evidence are preserved in the CSV/JSON records.',
        '', '## Limits and controls','',
        'The external historical stock GGUFs were missing after user disk cleanup. With explicit user approval, preserved locked stock baselines were audited and used as separately labeled replacement controls. Their hashes differ; no byte identity with the missing historical files is claimed. Other historical controls were retained. A completed first pass is not a global optimum or necessarily a coordinate-wise local optimum: later accepted changes can make earlier groups worth revisiting. Thousands of alternatives were tested against only 564 optimization targets, creating overfitting risk. No new full-test observation was used to change selection or choose a different new winner.',
        '', 'Historical CUDA-27 controls entered three groups but did not exhaust the third; they are incomplete, not three fully completed groups. The best historical fixed control is the already-recorded CPU selection-KL winner. The best historical automatic control is explicitly retrospective, chosen from already-observed historical WikiText results. Prior full-test arms were already examined, limiting claims of blind holdout validation.',
        '', '## Integrity and performance','',
        'Normal full-reference SHA-256 passes and bytes hashed are zero. The historical reference digest is explicitly non-fresh evidence under a live mandatory file guard; it does not prove unchanged bytes during unguarded historical intervals. Probability-body reads needed for full KLD remain legitimate inference I/O. See `INTEGRITY.md` and `reference-trust-policy.json`.',
        '', 'The persistent native evaluator retains full-model CUDA inference, private F32 scale buffers, canonical stock/identity elision and a fresh context/graph per score. Production qualification passed 19 exact per-token cases per recipe and five 16-thread cases; the final build reported no work to do and all49 relevant CTest programs passed in 224.51 seconds. Final selected canonical models also matched persistent predictions exactly. Source/candidate identities, allocation and numerical policy were preserved; no historical pools were regenerated.',
        '', 'See `PERFORMANCE.md`, `performance-before-after.json` and `hash-io-accounting.json` for scoped timings and byte counts. Local controls are bounded real-tensor benchmarks, not timings of the full historical Local stages. Full-study telemetry after interruption is not an uninterrupted whole-study baseline comparison. Large GGUFs, logits references and per-token TSVs remain local; compact coverage, assignments, hashes, commands and reports are published.',
        '', 'The relevant build/test commands and initial failures are retained in `WORK_LOG.md`, `build-v2.log` and `tests-v2.log`. Exact runtime digests and source-content fingerprints are in `implementation-provenance.json`; publication commit and remote verification are reported with the final delivery.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(result['status'])

if __name__=='__main__':main()
