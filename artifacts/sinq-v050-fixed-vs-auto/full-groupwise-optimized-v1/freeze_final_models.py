"""Freeze only two exhaustive first passes, then prepare matching held-out locks.

No final-test metric is read to choose either new assignment. The optional
historical automatic control is explicitly retrospective, not blind validation.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2];AB=ROOT.parent
sys.path[:0]=[str(REPO),str(REPO/'gguf-py')]
from tools.sinq.optimizer.resources import canonical_json,digest,file_digest,atomic_write
from tools.sinq.optimizer.schemas import load_config,read_json
from tools.sinq.optimizer.selector import State,Group,coordinate_descent
from tools.sinq.optimizer.stages import _run,_validator,_kl_evaluator
from tools.sinq.optimizer.prepared import VerifiedInputs,PreparedAssembly
from tools.sinq.optimizer.trusted_reference import GuardedFile,TrustedLargeReference
from tools.sinq.optimizer.model_audit import audit_assembled_model
from tools.sinq.optimizer.prepared_kl import token_metrics

SOURCE_SHA='da9df48e961a191aafca383628f983da4552a0a61c1257b93e1d678966f1c4af'
MANIFESTS={'T340':'b13c23622c54e14992ba3a8caa69e2fb108b1af0f2bf467c599daebc1f00ae2c',
           'T425':'75e86b85f88c1a76911b9833edf72385125341177b1eb1d4d6ed9807a82ca1f9'}


def checked(path,limit=128<<20):
    value=read_json(path,max_bytes=limit)
    if value.get('checksum')!=digest({k:v for k,v in value.items() if k!='checksum'}):
        raise ValueError('evidence checksum mismatch: '+str(path))
    return value


def coverage(recipe):
    out=ROOT/'selection'/recipe
    if not (out/'selection.json').is_file():raise ValueError('full-248 selection missing: '+recipe)
    selected=checked(out/'selection.json');checkpoint=checked(out/'checkpoint.json');identity=checked(out/'run-identity.json')
    progress=read_json(out/'progress.json',max_bytes=128<<20)
    config=load_config(ROOT/'configs'/f'{recipe}-full248.json','select')
    if (selected.get('status')!='PASS' or selected.get('budget_exhausted') or
            selected.get('search_status') not in ('COMPLETE_FIRST_PASS','COMPLETE_CONVERGED') or
            selected.get('run_identity_sha256')!=identity['checksum'] or identity.get('config_sha256')!=digest(config) or
            identity['bindings']['source']!=SOURCE_SHA or identity['bindings']['candidates']!=MANIFESTS[recipe] or
            selected.get('group_progress')!=progress or checkpoint.get('group_progress')!=progress):
        raise ValueError('selection closure or full-pass status mismatch: '+recipe)
    if (progress['eligible_groups']!=248 or progress['groups_completed']!=248 or progress['budget_exhausted'] or
            progress['completed_passes']<1 or not progress['passes'][0]['completed']):
        raise ValueError('incomplete 248-group first pass: '+recipe)
    registry=read_json(config['candidate_manifest'])
    per_group=progress['passes'][0]['per_group']
    if len(per_group)!=248 or {g['group'] for g in per_group}!=set(config['groups']):
        raise ValueError('group coverage inventory mismatch')
    for group in per_group:
        name=group['group'];record=registry['tensors'][name]
        expected={record['stock'],*[c['candidate_id'] for c in record['candidates']]}
        candidates=group['candidates']
        if (not group['completed'] or len(candidates)!=13 or len(expected)!=13 or
                {c['assignment'][name] for c in candidates}!=expected or
                any(set(c['assignment'])!={name} or c['status'] not in ('scored','current_assignment','exact_score_cache') for c in candidates)):
            raise ValueError('not all stock + 12 alternatives are accounted for: '+name)
        if any(c['state_id'] not in checkpoint['scores'] or c['score']!=checkpoint['scores'][c['state_id']] for c in candidates):
            raise ValueError('coverage references an uncommitted/mismatched score')
    initial=State.create(config['baseline'],{'run_identity_sha256':identity['checksum']})
    groups=[Group(name,tuple(members),tuple(config['alternatives'][name])) for name,members in sorted(config['groups'].items())]
    def cached(state):
        if state.state_id not in checkpoint['scores']:raise ValueError('offline replay lacks a required full-model score')
        return checkpoint['scores'][state.state_id]
    replay=coordinate_descent(initial,groups,cached,max_evaluations=config['budgets']['max_evaluations'],
        max_passes=config['max_passes'],tolerance=config['tolerance'],split=config['selection_split'],locks=config['locks'],
        recipe=recipe,evaluator_policy_identity=progress['evaluator_policy_identity'])
    if replay.budget_exhausted or dict(replay.state.assignment)!=selected['assignment'] or replay.score!=selected['selection_kld']:
        raise ValueError('deterministic complete-pass replay differs from published assignment')
    parity=read_json(out/'final-selection-parity.json')
    if parity.get('status')!='EXACT_PER_TOKEN_PARITY' or parity.get('model_sha256')!=selected['best_gguf_sha256'] or parity.get('selection_kld')!=selected['selection_kld'] or parity.get('targets')!=564:
        raise ValueError('final canonical/persistent per-token parity missing')
    return config,selected,checkpoint,identity,progress


def native_forward(binary,model,tokens,gpu,threads,output):
    command=[str(binary),'--forward',str(model),'--threads',str(threads)]
    if gpu:command+=['--gpu-layers','999']
    command+=list(map(str,tokens[:24]))
    import os
    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu['uuid'] if gpu else '',
         'OMP_NUM_THREADS':str(threads),'MKL_NUM_THREADS':str(threads)}
    result=subprocess.run(command,env=env,capture_output=True,timeout=300)
    output.with_suffix('.stdout').write_bytes(result.stdout);output.with_suffix('.stderr').write_bytes(result.stderr)
    if result.returncode:raise ValueError('published model forward failed: '+output.name)
    data=json.loads(result.stdout);logits=data['logits']
    if not logits or not all(math.isfinite(v) for v in logits):raise ValueError('nonfinite final-model logits')
    if gpu:
        text=result.stderr.decode(errors='replace')
        if 'offloaded 33/33 layers to GPU' not in text or gpu['pci_bus_id'] not in text:
            raise ValueError('final-model CUDA forward lacks matching full-offload evidence')
    return {'status':'PASS','command':command,'gpu':gpu,'tokens':len(tokens[:24]),'vocabulary':len(logits),
            'representative_logits':logits[:16],'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),
            'stderr_sha256':hashlib.sha256(result.stderr).hexdigest()}


def score_evidence(recipe,selected,checkpoint,identity,config):
    out=ROOT/'selection'/recipe;ref=read_json(str(read_json(config['evaluator'])['reference'])+'.json')
    evidence=[]
    attempts=sorted(out.glob('runtime-attempt-*'))
    for state_id,expected in sorted(checkpoint['scores'].items()):
        found=[d/(state_id+'.json') for d in attempts if (d/(state_id+'.json')).is_file()]
        if not found:raise ValueError('missing native score receipt: '+state_id)
        record=checked(found[-1],1<<20)
        if (record['state']!=state_id or record['policy']!=identity['policy'] or record['reference_manifest']!=digest(ref) or
                State.create(record['assignment'],{'run_identity_sha256':identity['checksum']}).state_id!=state_id or
                record['score']!=expected or record['execution']['gpu']!=identity['policy']['gpu'] or
                record['execution']['offloaded_layers']!=33 or record['execution']['total_layers']!=33):
            raise ValueError('native score receipt identity mismatch')
        tsv=Path(record['execution']['token_output'])
        with GuardedFile(tsv) as guard:
            sha=guard.verified_sha256(record['execution']['token_output_sha256'])
            value,rows=token_metrics(tsv,ref['alignment']['tokens'],ref['alignment']['context'])
            if value!=expected or len(rows)!=564:raise ValueError('per-token native score evidence differs')
        evidence.append({'state_id':state_id,'score':value,'score_receipt_sha256':file_digest(found[-1]),
                         'token_output_sha256':sha,'targets':564,
                         'native_process_peak_working_set_bytes':record['execution'].get('native_process_peak_working_set_bytes')})
    return {'status':'ALL_COMMITTED_SCORES_INDEPENDENTLY_CHECKED','recipe':recipe,'evaluations':len(evidence),
        'native_process_peak_working_set_bytes':max((r['native_process_peak_working_set_bytes'] or 0) for r in evidence),'scores':evidence}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    # Both coverage checks precede any final-test input access or model freeze.
    completed={recipe:coverage(recipe) for recipe in ('T340','T425')}
    if args.verify_only:
        print('BOTH_RECIPES_COMPLETE_FIRST_PASS_248_OF_248');return
    if (ROOT/'frozen-final-models.json').exists():raise FileExistsError('final models already frozen; no overwrite')
    audit_root=ROOT/'final-audits';audit_root.mkdir(exist_ok=True)
    attempt=1
    while (audit_root/f'attempt-{attempt}').exists():attempt+=1
    audits=audit_root/f'attempt-{attempt}';audits.mkdir()
    hardware=read_json(ROOT/'hardware-build-inventory.json')
    frozen={'schema_version':2,'status':'FROZEN_BEFORE_HELD_OUT_EVALUATION',
        'frozen_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':SOURCE_SHA,'models':[],
        'assignments':{},'coverage':{},'note':'Exactly one exhaustive coordinate-descent pass; no global-optimality claim. Prior historical final-test arms were already observed.'}
    with ExitStack() as session:
        inputs=session.enter_context(VerifiedInputs())
        for recipe,(config,selected,checkpoint,identity,progress) in completed.items():
            out=ROOT/'selection'/recipe;model=out/'best.gguf'
            model_guard=session.enter_context(GuardedFile(model))
            with _run(config,verified_inputs=inputs) as (registry,store,source),PreparedAssembly(source,config['stock_baseline_manifest'],threads=config['threads'],verified_inputs=inputs) as prepared:
                audit=audit_assembled_model(model,source,config['stock_baseline_manifest'],store,selected['assignment'],threads=config['threads'],prepared=prepared)
                if audit['output_sha256']!=selected['best_gguf_sha256']:raise ValueError('published final-model digest differs')
                e,oracle=_kl_evaluator(config);loader=Path(e['native_loader'])
                if oracle.policy!=identity['policy']['oracle_policy']:raise ValueError('selection oracle/GPU/runtime policy drifted before freeze')
                paths={'source':config['source_gguf'],'candidates':config['candidate_manifest'],
                    'selection':config['selection_manifest'],'evaluator_config':config['evaluator'],
                    'corpus':e['corpus'],'reference':e['reference'],'reference_manifest':str(e['reference'])+'.json',
                    'evaluator_binary':e['executable'],'native_loader':e['native_loader'],
                    'stock_baseline_manifest':config['stock_baseline_manifest'],'corpus_partitions':config['corpus_partitions_manifest']}
                for name,path in paths.items():inputs.fingerprint(path,identity['bindings'][name])
                inputs.fingerprint(e['prepared_executable'],identity['policy']['prepared_runtime']['prepared_evaluator']['executable'])
                _validator(loader,threads=16)(model)
                ref=read_json(str(e['reference'])+'.json');tokens=ref['alignment']['tokens']
                cpu=native_forward(loader,model,tokens,None,16,audits/(recipe+'-cpu-forward'))
                cuda=native_forward(loader,model,tokens,oracle.gpu,16,audits/(recipe+'-cuda-forward'))
                audit.update(file_identity=model_guard.check(),cpu_forward=cpu,cuda_forward=cuda,native_load='PASS')
                atomic_write(audits/(recipe+'-model-inventory.json'),canonical_json(audit),max_bytes=16<<20)
            evidence=score_evidence(recipe,selected,checkpoint,identity,config)
            atomic_write(audits/(recipe+'-all-score-evidence.json'),canonical_json(evidence),max_bytes=16<<20)
            frozen['assignments'][recipe]=selected['assignment']
            frozen['coverage'][recipe]={'eligible_groups':248,'groups_completed':248,'completed_passes':progress['completed_passes'],
                'status':progress['status'],'selection_kld':selected['selection_kld'],'selection_targets':564,
                'progress_sha256':file_digest(out/'progress.json'),'checkpoint_sha256':file_digest(out/'checkpoint.json'),
                'selection_sha256':file_digest(out/'selection.json'),'run_identity_sha256':file_digest(out/'run-identity.json'),
                'candidate_manifest_sha256':MANIFESTS[recipe]}
            frozen['models'].append({'recipe':recipe,'label':'NEW-FULL-248-GROUPWISE','arm':'NEW-FULL-248-GROUPWISE','kind':'auto',
                'path':str(model.resolve()),'expected_sha256':selected['best_gguf_sha256'],'selection_sha256':file_digest(out/'selection.json'),
                'selection_kld':selected['selection_kld'],'completed_groups':248,'model_inventory_sha256':file_digest(audits/(recipe+'-model-inventory.json'))})
        # Read-only historical controls are fixed before any new held-out result.
        old=read_json(AB/'gpu-groupwise-v1/final-models-frozen.json')
        fixed=read_json(AB/'report/fixed-model-audit.json');ranking=read_json(AB/'report/fixed-selection-ranking.json')
        auto=read_json(AB/'report/auto-model-audit.json')
        for recipe in ('T340','T425'):
            wanted={'FIXED-ANCHOR','STOCK-CONTROL','CURRENT-GROUP-KLD-CUDA27'}
            controls=[dict(m) for m in old['models'] if m['recipe']==recipe and m['label'] in wanted]
            for model in controls:
                if model['label']=='STOCK-CONTROL':
                    baseline_path=Path(completed[recipe][0]['stock_baseline_manifest']);baseline=checked(baseline_path)
                    model.update(label='LOCKED-STOCK-BASELINE-CONTROL',arm='LOCKED-STOCK-BASELINE-CONTROL',kind='stock-baseline',
                        original_missing_control_path=model['path'],original_missing_control_sha256=model['expected_sha256'],
                        path=str((baseline_path.parent/baseline['baseline_path']).resolve()),expected_sha256=baseline['baseline_sha256'],
                        stock_baseline_manifest_path=str(baseline_path.resolve()),stock_baseline_manifest_sha256=file_digest(baseline_path),
                        definition='User-approved replacement using preserved locked stock baseline. Original external stock controls were missing after user disk cleanup; different whole-file hashes are not claimed identical.')
            best=ranking['winners'][recipe]['best_fixed_arm'];row=next(r for r in fixed['records'] if r['arm']==best)
            controls.append({'recipe':recipe,'label':'BEST-HISTORICAL-FIXED-SELECTION-KL','arm':best,'kind':'fixed',
                'path':str((AB/'fixed'/recipe/best/'model.gguf').resolve()),'expected_sha256':row['model_sha256'],
                'definition':'Historical CPU selection-KL winner; not chosen from new held-out measurements.'})
            historical=read_json(AB/'evaluation/wikitext2'/recipe/'progress.json')
            candidates=[r for r in historical['results'] if r['label'].startswith('AUTO-')]
            if candidates:
                control=min(candidates,key=lambda r:r['mean_test_KL']);label=control['label']
                row=next(r for r in auto['models'] if r['recipe']==recipe and r['label']==label)
                path=AB/'selected'/recipe/label/'best.gguf'
                controls.append({'recipe':recipe,'label':'BEST-HISTORICAL-AUTOMATIC','arm':label,'kind':'auto',
                    'path':str(path.resolve()),'expected_sha256':row['model_sha256'],'selection_sha256':file_digest(path.parent/'selection.json'),
                    'definition':'Retrospective control chosen by already-observed historical WikiText KLD, not blind holdout validation.'})
            e,oracle=_kl_evaluator(completed[recipe][0])
            ref=read_json(str(e['reference'])+'.json')
            for m in controls:
                original=Path(m['path'])
                # Do not relax single-link guards for historical publication aliases.
                if original.stat().st_nlink>1:
                    private=ROOT/'controls'/audits.name/recipe/m['label'];private.mkdir(parents=True,exist_ok=False)
                    copied=private/'model.gguf';shutil.copyfile(original,copied)
                    m['original_control_path']=str(original);m['path']=str(copied.resolve())
                    if m['kind']=='auto':shutil.copyfile(original.parent/'selection.json',private/'selection.json')
                guard=session.enter_context(GuardedFile(m['path']));guard.verified_sha256(m['expected_sha256'])
                tsv=audits/(recipe+'-'+m['label']+'-selection.tsv')
                oracle._run(m['path'],e['corpus'],e['reference'],tsv)
                m['selection_kld'],_=token_metrics(tsv,ref['alignment']['tokens'],oracle.context)
                m['selection_targets']=564;m['selection_policy']=oracle.policy
                m['coverage_note']='3/248 groups attempted historically; incomplete, not three completed groups' if m['label']=='CURRENT-GROUP-KLD-CUDA27' else 'not a new full-248 search'
                frozen['models'].append(m)
        trust=session.enter_context(TrustedLargeReference(AB/'locks/final-wikitext-input-lock.json',evaluation_policy={'context':65536,'batch':512,'microbatch':256,'cache_type_k':'bf16','cache_type_v':'bf16','flash_attention':'on'}))
        prior=read_json(ROOT/'reference-trust-policy.json')
        if trust.provenance()['file_identity']!=prior['file_identity']:raise ValueError('production reference object differs from pinned metadata probe; no automatic hash')
        frozen['reference_trust']=trust.provenance()
        data=canonical_json(frozen);atomic_write(ROOT/'frozen-final-models.json',data,max_bytes=16<<20)
        # Existing runners accept an arbitrary manifest path; retain both requested
        # spellings as identical small JSON publications, never GGUF hard links.
        atomic_write(ROOT/'final-models-frozen.json',data,max_bytes=16<<20)
        executable=REPO/'build-sinq-clean/bin/llama-perplexity.exe'
        runtime_sha=inputs.fingerprint(executable)
        execution={'schema_version':2,'status':'FROZEN_BEFORE_HELD_OUT_EVALUATION','frozen_utc':frozen['frozen_utc'],
            'models_manifest_sha256':hashlib.sha256(data).hexdigest(),'source_sha256':SOURCE_SHA,'executable_sha256':runtime_sha,
            'wikitext_input_lock_sha256':file_digest(AB/'locks/final-wikitext-input-lock.json'),
            'ptb_input_lock_sha256':file_digest(AB/'locks/final-ptb-input-lock.json'),
            'reference_file_identity':trust.provenance()['file_identity'],'reference_trust':trust.provenance(),
            'gpus':{g['index']:{'uuid':g['uuid'],'pci_bus_id':g['pci_bus_id'].lower()[-12:]} for g in hardware['gpus']},
            'execution':{'context':65536,'batch':512,'microbatch':256,'threads':16,'cache_type_k':'bf16','cache_type_v':'bf16',
                'flash_attention':'on','gpu_layers':999,'cublas_compute_type':'auto','mmq_precision':'auto','warmup':False,'fit':False,'split_mode':'none'},
            'selection_fixed_before_held_out':True,'no_new_large_reference_hash':True}
        atomic_write(ROOT/'final-execution-lock.json',canonical_json(execution),max_bytes=1<<20)
    print('FROZEN_BOTH_COMPLETE_248_GROUP_MODELS_BEFORE_HELD_OUT')

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(type(error).__name__+': '+str(error),file=sys.stderr);raise SystemExit(1)
