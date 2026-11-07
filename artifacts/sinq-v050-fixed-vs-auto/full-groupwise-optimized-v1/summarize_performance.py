"""Bounded structured evidence only. Never traverse/hash arbitrary stage outputs."""
import csv
import io
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text())

def source_hash(metrics):
    return next(v for p,v in metrics['file_hash']['files'].items() if Path(p).name=='Qwen3.5-9B-text-BF16.gguf')

def stock_hash(metrics,after):
    if after:
        rows=[v for p,v in metrics['file_hash']['files'].items() if 'sinq-verified-stock-' in p and Path(p).name=='stock.gguf']
    else:
        rows=[v for p,v in metrics['file_hash']['files'].items() if Path(p).name=='stock.gguf' and 'optimizer-final-32' in p]
    if len(rows)!=1:raise ValueError('ambiguous baseline hash accounting')
    return rows[0]

rows=[];phase_rows=[];details={'scope':'Measured select_config stage, not the post-invocation report hash scans; real local scorer controls exclude one-time input setup.',
 'large_reference_hash_bytes':0,'microbenchmarks':{},'local_bounded_controls':{},'full_search':{},
 'attribution_note':'Phase seconds are explicitly inclusive. exclusive_seconds are per-thread wall seconds; concurrent worker times are not additive top-level wall attribution.'}
for recipe in ('T340','T425'):
    before=read(ROOT/'performance-before-audited'/f'{recipe}-legacy2-audited-select-runtime.json')
    after=read(ROOT/'performance-after'/f'{recipe}-prepared2-v1-select-runtime.json')
    if before['best_model_sha256']!=after['best_model_sha256']:raise ValueError('microbenchmark selected-model bytes differ')
    bm,am=before['performance'],after['performance']
    for mode,record in [('before',before),('after',after)]:
        for name,phase in record['performance'].items():
            if isinstance(phase,dict) and 'exclusive_seconds' in phase:
                phase_rows.append({'recipe':recipe,'mode':mode,'phase':name,'calls':phase['calls'],
                    'inclusive_seconds':phase['seconds'],'exclusive_same_thread_seconds':phase['exclusive_seconds'],
                    'inclusive':phase['inclusive'],'scope':phase['time_scope'],
                    'single_owner_stage_wall_seconds':record['selection_total_time_sec']})
    bs,as_=source_hash(bm),source_hash(am);bb,ab=stock_hash(bm,False),stock_hash(am,True)
    if bs['calls']!=27 or as_['calls']!=1 or bb['calls']!=14 or ab['calls']!=1:raise ValueError('unexpected fresh hash counts')
    for metrics in (bm,am):
        if any(Path(p).name.casefold()=='qwen3.5-9b-wikitext2-64k-bf16.logits' for p in metrics['file_hash']['files']):raise ValueError('large reference present in normal hash accounting')
    data={'before_seconds':before['selection_total_time_sec'],'after_seconds':after['selection_total_time_sec'],
        'stage_speedup':before['selection_total_time_sec']/after['selection_total_time_sec'],
        'before_source':bs,'after_source':as_,'before_baseline':bb,'after_baseline':ab,
        'before_file_hash_input_bytes':bm['file_hash']['bytes_hashed'],'after_file_hash_input_bytes':am['file_hash']['bytes_hashed'],
        'before_tensor_hash_input_bytes':bm['tensor_hash']['bytes_hashed'],'after_tensor_hash_input_bytes':am['tensor_hash']['bytes_hashed'],
        'before_canonical_assemblies':len(before['assembly_calls']),'after_canonical_assemblies':len(after['assembly_calls']),
        'after_native_candidate_responses':am['prepared_native_candidate']['calls'],
        'after_native_candidate_response_seconds':am['prepared_native_candidate']['seconds'],
        'before_gguf_file_bytes_materialized':sum(a['output_bytes'] for a in before['assembly_calls']),
        'after_gguf_write_bytes':am['assembly_write']['bytes_written'],
        'before_temp_write_syscall_bytes':'not measured; output lengths above are not physical I/O counters',
        'before_parent_peak_working_set_bytes':before['parent_peak_working_set_bytes'],
        'after_parent_peak_working_set_bytes':after['parent_peak_working_set_bytes'],
        'before_vram_peak_bytes':before['vram_peak_bytes'],'after_vram_peak_bytes':after['vram_peak_bytes'],
        'before_cache':before['cache'],'after_cache':after['cache'],'selected_model_sha256':after['best_model_sha256'],
        'study_coverage':'2 scores, 0 complete groups; profiling only',
        'native_version_note':'Two-score after pilot used v1 private-diagonal prototype; production v2 canonical identity elision has separate all-role parity and 13-score concurrency qualification.'}
    details['microbenchmarks'][recipe]=data
    rows.append({'stage':'selection two-score microbenchmark','workload':recipe,'before_seconds':data['before_seconds'],
        'after_seconds':data['after_seconds'],'speedup':data['stage_speedup'],
        'before_source_file_hash_bytes':bs['bytes_hashed'],'after_source_file_hash_bytes':as_['bytes_hashed'],
        'before_baseline_file_hash_bytes':bb['bytes_hashed'],'after_baseline_file_hash_bytes':ab['bytes_hashed'],
        'large_reference_new_hash_bytes':0,'after_temp_gguf_write_bytes':data['after_gguf_write_bytes']})
    local=read(ROOT/'local-benchmarks'/recipe/'results.json')
    if local['status']!='EXACT_SCORE_PARITY':raise ValueError('local control parity not complete')
    details['local_bounded_controls'][recipe]={}
    for objective in ('output_l1','output_mse'):
        cases=[r for r in local['records'] if r['objective']==objective]
        b=sum(r['legacy_seconds'] for r in cases);a=sum(r['batched_seconds'] for r in cases)
        record={'tensor_count':len(cases),'candidates_per_tensor':13,'before_seconds':b,'after_seconds':a,'speedup':b/a,
            'source_bytes_copied_before':sum(r['legacy_source_bytes_copied'] for r in cases),
            'source_bytes_copied_after':sum(r['batched_source_bytes_copied'] for r in cases),
            'source_fresh_hash_bytes_before':sum(r['legacy_fresh_hash_bytes'] for r in cases),
            'source_fresh_hash_bytes_after':sum(r['batched_fresh_hash_bytes'] for r in cases),
            'native_process_launches_before':sum(r['legacy_process_launches'] for r in cases),
            'native_process_launches_after':sum(r['batched_process_launches'] for r in cases),
            'native_process_seconds_inclusive_before':sum(r['legacy_native_process_seconds_inclusive'] for r in cases),
            'native_process_seconds_inclusive_after':sum(r['batched_native_process_seconds_inclusive'] for r in cases),
            'native_compute_seconds_after':sum(r['batched_metrics']['native_compute_seconds'] for r in cases),
            'native_compute_seconds_before':'not measured separately from process startup and native I/O',
            'score_difference_max':max(r['max_absolute_score_difference'] for r in cases),
            'scope':'Three real tensors and original 256-row captures; not the full historical 248-tensor stage.'}
        details['local_bounded_controls'][recipe][objective]=record
        rows.append({'stage':'bounded real local '+objective,'workload':recipe,'before_seconds':b,'after_seconds':a,'speedup':b/a,
            'before_source_file_hash_bytes':record['source_fresh_hash_bytes_before'],
            'after_source_file_hash_bytes':record['source_fresh_hash_bytes_after'],
            'before_baseline_file_hash_bytes':0,'after_baseline_file_hash_bytes':0,'large_reference_new_hash_bytes':0,
            'after_temp_gguf_write_bytes':0})
    runs=sorted((ROOT/'runtime'/recipe).glob(f'{recipe}-{recipe}-select-runtime*.json'),key=lambda p:p.stat().st_mtime)
    completed=[(p,read(p)) for p in runs if read(p).get('status')=='PASS']
    if completed:
        run,value=completed[-1]
        details['full_search'][recipe]={'status':value['status'],'completed_invocation_seconds':value.get('selection_total_time_sec'),
            'timing_scope':'Final resumed invocation, including deterministic cached-score replay and remaining proposals; not the full active duration across interrupted attempts.',
            'complete_study_active_wall_seconds':'not measured across killed attempts','runtime_report_path':str(run),
            'evaluations':value.get('selection_evaluations'),'before_full_248_seconds':'not measured',
            'parent_peak_working_set_bytes':value.get('parent_peak_working_set_bytes'),'vram_peak_bytes':value.get('vram_peak_bytes')}

before_group=ROOT/'performance-before-one-group/T340-legacy13-audited-select-runtime.json'
if before_group.exists():
    b=read(before_group);a=read(ROOT/'concurrency/single/T340-prepared13-single-v2-select-runtime.json')
    def assignment_scores(folder):
        return {tuple(sorted(e['assignment'].items())):e['score'] for e in read(ROOT/folder/'checkpoint.json')['trace'] if e['status']=='evaluated'}
    bscores=assignment_scores(Path('pilots/T340/legacy13-audited'));ascores=assignment_scores(Path('pilots/T340/prepared13-single-v2'))
    if bscores!=ascores or len(bscores)!=13 or b['best_model_sha256']!=a['best_model_sha256']:raise ValueError('one-group controlled parity differs')
    details['one_complete_group_control']={'recipe':'T340','group':'blk.0.attn_gate.weight','candidate_count':13,
        'before_seconds':b['selection_total_time_sec'],'after_seconds':a['selection_total_time_sec'],
        'speedup':b['selection_total_time_sec']/a['selection_total_time_sec'],'all_assignment_mean_scores_exact':True,
        'selected_model_sha256':a['best_model_sha256'],'before_canonical_assemblies':len(b['assembly_calls']),
        'after_canonical_assemblies':len(a['assembly_calls']),'before_source_hash':source_hash(b['performance']),
        'after_source_hash':source_hash(a['performance']),
        'scope':'Same GPU0, frozen selection inputs and 13 identical assignments. Legacy original snapshot with telemetry only; prepared canonical v2. Additional benchmark does not change either frozen full-study assignment.'}
    rows.append({'stage':'One complete group, stock plus all 12 conditioned','workload':'T340','before_seconds':b['selection_total_time_sec'],
        'after_seconds':a['selection_total_time_sec'],'speedup':b['selection_total_time_sec']/a['selection_total_time_sec'],
        'before_source_file_hash_bytes':source_hash(b['performance'])['bytes_hashed'],
        'after_source_file_hash_bytes':source_hash(a['performance'])['bytes_hashed'],
        'before_baseline_file_hash_bytes':stock_hash(b['performance'],False)['bytes_hashed'],
        'after_baseline_file_hash_bytes':stock_hash(a['performance'],True)['bytes_hashed'],
        'large_reference_new_hash_bytes':0,'after_temp_gguf_write_bytes':a['performance']['assembly_write']['bytes_written']})
details['held_out']={}
for recipe in ('T340','T425'):
    path=ROOT/f'heldout-launch-{recipe}.json'
    if not path.exists():continue
    launches=read(path)
    for invocation in launches:
        dataset=invocation['dataset'];progress=read(ROOT/'held-out'/dataset/recipe/'progress.json')
        if invocation['returncode'] or progress['status']!='PASS':raise ValueError('held-out timing is incomplete')
        records=[read(Path(row['metrics_path'])) for row in progress['results']]
        native=sum(r['elapsed_seconds'] for r in records)
        data={'models':len(records),'runner_wall_seconds':invocation['elapsed_seconds'],
            'native_process_elapsed_seconds':native,'orchestration_including_setup_hashes_and_summaries_seconds':invocation['elapsed_seconds']-native,
            'reference_setup_only_seconds':'not measured separately','reference_new_hash_bytes':0,
            'reference_inference_input_bytes_by_format':147601497864*len(records) if dataset=='wikitext2' else 0,
            'reference_inference_input_bytes_scope':'logical complete input-format size per successful arm, not OS physical disk-I/O counters',
            'peak_working_set_bytes':max(r['peak_process_working_set_bytes'] for r in records),
            'gpu_global_peak_mib':max(r['gpu_global_memory_peak_mib'] for r in records),
            'temp_disk_peak_bytes':'not measured','physical_disk_write_bytes':'not measured'}
        details['held_out'][recipe+'/'+dataset]=data
        if dataset=='wikitext2':
            if any(r['reference_trust']['reference_content_bytes_hashed_per_normal_run'] for r in records):raise ValueError('nonzero normal reference hash bytes')
            for stage,seconds in [('Final WikiText KLD inference',native),('Final WikiText orchestration including startup and summaries',invocation['elapsed_seconds']-native)]:
                rows.append({'stage':stage,'workload':recipe,'before_seconds':'not measured','after_seconds':seconds,'speedup':'not measured',
                    'before_source_file_hash_bytes':'not measured','after_source_file_hash_bytes':'not measured by subphase',
                    'before_baseline_file_hash_bytes':'not measured','after_baseline_file_hash_bytes':'not applicable',
                    'large_reference_new_hash_bytes':0,'after_temp_gguf_write_bytes':0})
launch=read(ROOT/'concurrency/launch-times.json')
details['production_v2_concurrency']=launch
details['advanced_fixture_profile']=read(ROOT/'advanced-fixture-profile.json')
(ROOT/'performance-before-after.json').write_text(json.dumps(details,indent=2,allow_nan=False)+'\n')
accounting={k:v for k,v in details.items() if k in ('scope','attribution_note','large_reference_hash_bytes','microbenchmarks','local_bounded_controls','held_out','one_complete_group_control')}
(ROOT/'hash-io-accounting.json').write_text(json.dumps(accounting,indent=2,allow_nan=False)+'\n')
(ROOT/'tables').mkdir(exist_ok=True)
stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(ROOT/'tables/performance-comparison.csv').write_text(stream.getvalue(),newline='')
stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(phase_rows[0]));writer.writeheader();writer.writerows(phase_rows)
(ROOT/'tables/exclusive-phase-attribution.csv').write_text(stream.getvalue(),newline='')
lines=['# Measured SINQ pipeline performance','','## Scope and attribution','',
'`tables/exclusive-phase-attribution.csv` provides actual before/after inclusive and same-thread-exclusive measurements and the single-owner stage wall timer separately; concurrent worker phase totals are not a partition of wall time. Stage timers cover `select_config`: input verification, search and canonical finalization. The existing measured launcher also performs post-invocation model/report hash checks outside this timer; those scans are not hidden inside the stage numbers. Per-file cryptographic input bytes are reported separately from tensor-hash bytes. Neither is a physical disk-read counter.',
'', 'Phase `seconds` are labeled inclusive. `exclusive_seconds` subtract same-thread child phases; concurrent worker-thread times must not be added to obtain invocation wall time. One owner supplies each top-level stage timer. The historical source snapshot is `ff7994f35` with only the current telemetry/hash-prohibition module substituted.',
'', '## Matching two-score controls on the same physical GPU','',
'| Recipe | Legacy stage s | Prepared stage s | Stage speedup | Full-source hash passes | Stock hash passes | Canonical writes |',
'|---|---:|---:|---:|---:|---:|---:|']
for recipe,d in details['microbenchmarks'].items():
 lines.append(f"| {recipe} | {d['before_seconds']:.3f} | {d['after_seconds']:.3f} | {d['stage_speedup']:.3f}x | 27 → 1 | 14 → 1 | 3 → 2 |")
if 'one_complete_group_control' in details:
 d=details['one_complete_group_control']
 lines += ['',f"A separate complete first-group control scored stock plus all12 conditioned alternatives on the same GPU0. Legacy {d['before_seconds']:.3f}s versus production-v2 {d['after_seconds']:.3f}s ({d['speedup']:.3f}x), {d['before_canonical_assemblies']} versus {d['after_canonical_assemblies']} canonical assemblies. All13 full assignment mean scores and selected model bytes match exactly. This bounded control does not claim a measured full-248 legacy speedup."]
lines+=['', 'Both selected GGUF SHA-256 identities match the immutable legacy controls. The two-score pilots completed zero groups and are not full-search results. The after pilots include an extra final canonical/persistent score-parity check. Initial/final canonical writes remain; proposal evaluation does not write a GGUF.',
'', 'The production v2 adapter additionally elides stock/all-one diagonals exactly like the canonical loader and rebuilds context/graphs on every score. Its all-role numerical gates and the separately recorded 13-score one-versus-two-GPU pilots precede the full studies. A single 13-score pilot completed one group, not a complete pass.',
'', '## Local scoring: bounded real controls','',
'Each row covers the same three real tensors, all 13 candidates and original capture/settings. It is not a full 248-tensor Local stage timing. Historical 43.9/119.9-minute figures are not used as measured before values here.',
'', '| Recipe / objective | Legacy s | Batched s | Speedup | Native launches | Exact score difference |',
'|---|---:|---:|---:|---:|---:|']
for recipe,objectives in details['local_bounded_controls'].items():
 for objective,d in objectives.items():lines.append(f"| {recipe} / {objective} | {d['before_seconds']:.3f} | {d['after_seconds']:.3f} | {d['speedup']:.3f}x | {d['native_process_launches_before']} → {d['native_process_launches_after']} | {d['score_difference_max']} |")
lines+=['','## Complete final evaluators','', '| Recipe / corpus | Models | Runner wall s | Native process s | Other orchestration s | New reference hash bytes |', '|---|---:|---:|---:|---:|---:|']
for label,d in details['held_out'].items():
 lines.append(f"| {label} | {d['models']} | {d['runner_wall_seconds']:.3f} | {d['native_process_elapsed_seconds']:.3f} | {d['orchestration_including_setup_hashes_and_summaries_seconds']:.3f} | 0 |")
lines+=['','Runner minus native elapsed includes setup, model/source guards and hashes, TSV summaries and publication; it is not an isolated startup timer. The WikiText logical reference-format input is 147,601,497,864 bytes per arm. Physical disk reads, whole-run temp-disk peaks and isolated startup/resume durations were not measured. No before/after final-evaluator speedup is claimed.', '']
lines+=['', 'Source exports and fresh source-tensor hash bytes fall by 13x for these bounded controls. MSE compute remains substantial; these measurements do not establish that historical scoring time was primarily hashing. The original local objectives and native loss routines are unchanged.',
'', '## Remaining costs and limits','',
'Candidate registries still receive fresh content admission per guarded session. A guarded private baseline copy is required for historical hard-link aliases. Native context construction and full-vocabulary reduction remain per proposal. Checkpoint JSON publication grows with retained assignment evidence; full-study runtime is measured independently, not extrapolated from the two-score pilot.',
'', 'The bounded genuine-Qwen Blockwise/Fisher profile records source/reference loading once, three native candidate decodes, zero full-GGUF assemblies and zero new source-file hash bytes during those candidate trials. It is not a production 9B throughput claim. The kernels, precision and Fisher objectives were left unchanged. Native activation-validation boundaries retain their independent full checks; individual advanced candidate decodes still launch a bounded native process.',
'', 'Full WikiText probability-body reads are required for KLD and are not hash passes. Normal full-reference content hashes and content bytes hashed are zero. The production metadata/guard probe and mocked success/resume/error tripwires are recorded separately. Final evaluator and full-study figures are added only after completed runs; missing metrics remain explicitly unmeasured.',
'', 'Large GGUFs, probability references and temporary score TSVs remain local. Immutable small manifests, code, commands, hashes and compact coverage tables are the publication surface.']
(ROOT/'PERFORMANCE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('PERFORMANCE_AND_HASH_IO_TABLES_WRITTEN')
