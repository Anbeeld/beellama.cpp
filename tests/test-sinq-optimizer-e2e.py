#!/usr/bin/env python3
"""Real tiny native generate/score/select/assemble/load/forward/replay integration."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
if any(importlib.util.find_spec(n) is None for n in ('torch','transformers','numpy','safetensors')):
    print('SKIP: optional offline numerical dependencies unavailable');raise SystemExit(77)
import numpy as np
import gguf
from tools.sinq import export_activations as exporter
from tools.sinq.optimizer.schemas import validate_config
from tools.sinq.optimizer.generate import generate,preflight
from tools.sinq.optimizer.stages import score_config,select_config,assemble_config,validate_config_artifacts,_validator
from tools.sinq.optimizer.cache import CandidateStore
from tools.sinq.optimizer.baseline import lock_baseline
from tools.sinq.optimizer.corpora import lock_corpora
from tools.sinq.optimizer.kl import NativeKLEvaluator
from tools.sinq.optimizer.model_audit import audit_assembled_model
from tools.sinq.optimizer.resources import file_digest,canonical_json,digest


def write(path,value):Path(path).write_bytes(canonical_json(value))

def execute(command):
    result=subprocess.run(list(map(str,command)),capture_output=True,timeout=180)
    if result.returncode:raise AssertionError(result.stderr.decode(errors='replace')[-6000:])
    return result


def fixture_validator(binary,*,threads=None):
    load=_validator(binary,threads=threads)
    def validate(path):
        load(path)
        execute([binary,'--forward',path,1,3,4])
    return validate


def workflow(d,binaries):
    suffix='.exe' if os.name=='nt' else ''
    binary=lambda name:str(binaries/(name+suffix))
    bridge=binary('llama-sinq-native');loader=binary('llama-sinq-model');quantizer=binary('llama-quantize')
    spec=importlib.util.spec_from_file_location('fixture',ROOT/'tests/test-sinq-groupwise-kl.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=d/'source.gguf';module.native_fixture(source)
    names=['blk.0.ffn_up.weight','blk.0.ssm_out.weight','blk.1.attn_output.weight']
    tensors=gguf.GGUFReader(source)
    recipe={t.name:('Q5_0' if len(t.shape)>1 and int(t.shape[0])%32==0 else t.tensor_type.name) for t in tensors.tensors}
    shapes={t.name:tuple(map(int,t.shape)) for t in tensors.tensors};tensors.data._mmap.close()
    types=d/'types.txt';types.write_text('\n'.join('^'+re.escape(n)+'$='+q.lower() for n,q in sorted(recipe.items())),encoding='utf-8')
    stock=d/'stock.gguf';execute([quantizer,'--pure','--tensor-type-file',types,source,stock,'Q8_0','4'])
    baseline=d/'baseline.json';lock_baseline(source,stock,recipe,baseline,quantizer=quantizer,native_validate=_validator(loader,threads=4),threads=4)
    tokens=json.loads((d/'checkpoint/tokens.json').read_text());cal=np.asarray(tokens,dtype='<u4')
    selection=np.asarray([1,3,4,5,6,7,8,9],dtype='<u4');test=np.asarray([2,9,8,7],dtype='<u4')
    for label,values in [('cal',cal),('sel',selection),('test',test)]:values.tofile(d/(label+'.u32'))
    corpus_lock=d/'corpora.json'
    lock_corpora({label:str(d/(label+'.u32')) for label in ('cal','sel','test')},
        dict(calibration=[dict(source='cal',start=0,end=len(cal))],selection=[dict(source='sel',start=0,end=len(selection))],final_test=[dict(source='test',start=0,end=len(test))]),
        corpus_lock,max_tokens=64,max_output_bytes=1<<20,max_files=8)
    captures=d/'captures';captures.mkdir();records=[]
    with np.load(d/'checkpoint/hf-inputs.npz') as arrays:
        for i,name in enumerate(names):
            a=arrays[name].astype('<f4');a.tofile(captures/f'{i}.f32');path,site,_=exporter.module_for(name);k,n=shapes[name]
            records.append(dict(name=name,site=site,reference_module=path,channel_layout='gguf-input',input_dim=k,output_dim=n,
                count=len(a),dtype='<f4',order='row-major',data_file=f'{i}.f32',sha256=file_digest(captures/f'{i}.f32')))
    write(captures/'activations.json',dict(schema_version=1,architecture='qwen3_5',source_sha256=file_digest(source),source_revision='tiny-qwen35-v2',
        sampling=dict(contract='valid-token-prefill-reservoir-v1',seed=41721,max_rows=len(cal),corpus_sha256=file_digest(d/'cal.u32'),token_sha256=file_digest(d/'cal.u32')),
        provenance=dict(token_count=len(cal)),tensors=records))
    limits=dict(workspace_bytes=128<<20,max_artifact_bytes=128<<20,max_files=2000,max_tensors=8)
    inputs=dict(source_gguf=str(source),native_bridge=bridge,stock_baseline_manifest=str(baseline),corpus_partitions_manifest=str(corpus_lock))
    raw=dict(schema_version=2,inputs=inputs,quantization=dict(recipe=recipe),targets=dict(include=names),
        candidate_generators=[dict(type='sinq',expansion='cartesian',parameters=dict(strip_width=[32,64,128],iterations=[4,16],strength=[0.5,1]))],
        parameters=dict(locks={names[0]:dict(generator='sinq',strip_width=32,iterations=16,strength=1.0)}),
        execution=dict(threads=4),limits=dict(limits,max_candidates_per_tensor=16),output=dict(directory=str(d/'run')))
    # Ordinary candidate generation is independent of all captures.
    write(d/'generate-config.json',raw)
    config=validate_config(raw,'generate');plan=preflight(config)
    assert len(plan['tensor_details'])==3 and plan['activation'] is None
    run=generate(config);manifest=d/'run/run-manifest.json';original_manifest=manifest.read_bytes()
    with patch('tools.sinq.optimizer.native_codec.request',side_effect=AssertionError('resume re-encoded payload')):
        assert generate(config,resume=d/'run')==run
    assert manifest.read_bytes()==original_manifest
    # Every scorer and selector below sees this SAME mixed ordinary/AWQ pool.
    mixed=dict(raw,inputs=dict(inputs,activation_manifest=str(captures/'activations.json')),output=dict(directory=str(d/'mixed-run')),
        candidate_generators=[*raw['candidate_generators'],dict(type='awq',configurations=[dict(strip_width=64,iterations=16,strength=0.5,correction='weight_activation',alpha=0.5,**{'lambda':1.0})])])
    write(d/'mixed-generate-config.json',mixed)
    run=generate(validate_config(mixed,'generate'));manifest=d/'mixed-run/run-manifest.json'
    baseline_assignment={n:r['selected'] for n,r in run['tensors'].items()};groups={n:[n] for n in names}
    awq_count=sum(c['parameters']['generator']=='awq' for r in run['tensors'].values() for c in r['candidates'])
    assert awq_count==2, 'locked tensors must not receive automatic alternatives'
    capture=d/'block.json';write(capture,dict(schema_version=2,source_sha256=run['source_sha256'],tokens=[tokens],layers=[0,1]))
    manual_config=validate_config(dict(schema_version=2,
        inputs=dict(source_gguf=str(source),candidate_manifest=str(manifest),native_loader=loader),
        selection=dict(policy='manual',baseline=baseline_assignment,groups=groups),execution=dict(threads=4),
        limits=limits,output=dict(directory=str(d/'manual'))),'select')
    manual=select_config(manual_config);execute([loader,'--forward',d/'manual/best.gguf',1,3,4])
    assert manual['assignment']==baseline_assignment
    results={'manual':dict(output_sha256=manual['best_gguf_sha256'])};scores_paths={}
    for kind in ('local_output_l1','local_output_mse','blockwise_output_l1','blockwise_output_mse','fisher_vjp_mc'):
        local=kind.startswith('local');score_inputs=dict(candidate_manifest=str(manifest),capture_manifest=str(captures/'activations.json' if local else capture),native_bridge=bridge)
        if not local:score_inputs['checkpoint']=str(d/'checkpoint')
        sl=dict(limits,max_tokens=32)
        sl['max_local_scores' if local else 'max_block_scores']=100
        scoring=dict(kind=kind)
        if kind=='fisher_vjp_mc':scoring.update(probes=1,seed=19,fisher_positions=[[0,0]]);sl['max_fisher_vjps']=4
        score_raw=dict(schema_version=2,inputs=score_inputs,scoring=scoring,execution=dict(threads=4),limits=sl,output=dict(directory=str(d/kind)))
        result=score_config(validate_config(score_raw,'score'));assert result['records'] and all(np.isfinite(r['score']) for r in result['records'])
        if local:
            joint={n:run['tensors'][n]['candidates'][-1]['candidate_id'] for n in names[1:]}
            expected=sum(r['score'] for r in result['records'] if joint.get(r['tensor'])==r['candidate_id'])
            joint_raw=dict(score_raw,scoring=dict(scoring,assignments={'joint':joint}),
                limits=dict(sl,max_local_scores=2),output=dict(directory=str(d/('joint-'+kind))))
            observed=score_config(validate_config(joint_raw,'score'))
            assert len(observed['records'])==1 and observed['records'][0]['assignment']==joint
            np.testing.assert_allclose(observed['records'][0]['score'],expected,rtol=1e-14)
        scores_paths[kind]=d/kind/'scores.json'
        selected=select_config(validate_config(dict(schema_version=2,inputs=dict(source_gguf=str(source),candidate_manifest=str(manifest),native_loader=loader),
            selection=dict(policy='per_group_score',baseline=baseline_assignment,groups=groups,scores=str(scores_paths[kind]),locks={names[0]:baseline_assignment[names[0]]}),
            execution=dict(threads=4),limits=limits,output=dict(directory=str(d/('selected-'+kind)))),'select'))
        assert selected['assignment'][names[0]]==baseline_assignment[names[0]]
        forward=execute([loader,'--forward',d/('selected-'+kind)/'best.gguf',1,3,4]);assert np.isfinite(json.loads(forward.stdout)['logits']).all()
        results[kind]=dict(score_count=len(result['records']),output_sha256=selected['best_gguf_sha256'])
    corpus=d/'selection.txt';corpus.write_text('abcdefg',encoding='utf-8')
    evaluator=NativeKLEvaluator(binary('llama-perplexity'),context=32,batch=24,microbatch=24,max_tokens=32,threads=4)
    reference=d/'reference.bin';evaluator.create_reference(source,corpus,reference,expected_tokens=selection.tolist())
    selection_lock=d/'selection.json';write(selection_lock,dict(schema_version=2,split='selection',corpus_sha256=file_digest(corpus),reference_sha256=file_digest(reference)))
    engine=d/'evaluator.json';write(engine,dict(schema_version=2,executable=binary('llama-perplexity'),reference=str(reference),corpus=str(corpus),native_loader=loader,
        context=32,batch=24,microbatch=24,max_tokens=32,timeout=180))
    select_inputs=dict(source_gguf=str(source),candidate_manifest=str(manifest),evaluator=str(engine),selection_manifest=str(selection_lock),corpus_partitions_manifest=str(corpus_lock))
    for policy in ('groupwise_kl','pipeline'):
        select=dict(policy=policy,baseline=baseline_assignment,groups=groups,locks={names[0]:baseline_assignment[names[0]]},max_passes=1)
        if policy=='pipeline':select['stages']=[dict(type='filter',scores=str(scores_paths['blockwise_output_mse']),keep=2,context='frozen'),dict(type='select',method='groupwise_kl')]
        cfg=validate_config(dict(schema_version=2,inputs=select_inputs,selection=select,execution=dict(threads=4),limits=dict(limits,max_model_evaluations=5,max_tokens=32),output=dict(directory=str(d/policy))),'select')
        selected=select_config(cfg);assert selected['assignment'][names[0]]==baseline_assignment[names[0]]
        with patch('tools.sinq.optimizer.kl.NativeKLEvaluator._run',side_effect=AssertionError('resume reran model evaluation')):assert select_config(cfg,resume=d/policy)==selected
        execute([loader,'--forward',d/policy/'best.gguf',1,3,4]);results[policy]=dict(evaluations=selected['evaluations'],output_sha256=selected['best_gguf_sha256'])
    assignment={n:r['candidates'][-1]['candidate_id'] for n,r in run['tensors'].items()}
    assembly=validate_config(dict(schema_version=2,inputs=dict(source_gguf=str(source),candidate_manifest=str(manifest),native_loader=loader),
        selection=dict(assignment=assignment),execution=dict(threads=4),limits=limits,output=dict(gguf=str(d/'mixed.gguf'))),'assemble')
    assembled=assemble_config(assembly)
    with CandidateStore(manifest.parent/'cache').verified_session() as store:audit=audit_assembled_model(d/'mixed.gguf',source,baseline,store,assignment,threads=4)
    assert audit['output_sha256']==assembled['output_sha256']
    execute([loader,'--forward',d/'mixed.gguf',1,3,4])
    replay=dict(assembly,output_gguf=str(d/'replay.gguf'));assert assemble_config(replay)['output_sha256']==assembled['output_sha256']
    invalid=dict(assembly,output_gguf=str(d/'rejected.gguf'),assignment={names[0]:'0'*64})
    try:assemble_config(invalid);raise AssertionError('invalid candidate accepted')
    except ValueError:assert not (d/'rejected.gguf').exists()
    return dict(status='PASS',source_sha256=file_digest(source),manifest_sha256=file_digest(manifest),assembly_sha256=assembled['output_sha256'],
        candidate_count=sum(len(r['candidates']) for r in run['tensors'].values()),awq_count=awq_count,scorers_and_selectors=results)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bin',required=True);p.add_argument('--output');a=p.parse_args()
    with patch('tools.sinq.optimizer.stages._validator',side_effect=fixture_validator):
        if a.output:
            d=Path(a.output).resolve();d.mkdir(parents=True,exist_ok=False);result=workflow(d,Path(a.bin).resolve());write(d/'summary.json',result)
        else:
            with tempfile.TemporaryDirectory(prefix='sinq-optimizer-e2e-') as temp:result=workflow(Path(temp),Path(a.bin).resolve())
    print(json.dumps(result,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
