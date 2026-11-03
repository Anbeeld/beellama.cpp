#!/usr/bin/env python3
"""Tiny actual-codec/cache/block/Fisher/hybrid/native-forward workflow, not a study."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
os.environ.update(OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0')
# Dependency-bound correctness coverage must be an explicit SKIP, never a stub.
import importlib.util
if any(importlib.util.find_spec(name) is None for name in ('torch','transformers','numpy','safetensors')):
    print('SKIP: local Torch/Transformers/NumPy/safetensors environment unavailable')
    raise SystemExit(77)

import numpy as np
from tools.sinq import export_activations as exporter
from tools.sinq.research.schemas import validate_config
from tools.sinq.research.generate import generate,preflight
from tools.sinq.research.cli import score_config,assemble_config,validate_config_artifacts,select_config,_advanced_reference
from tools.sinq.research.kl import NativeKLEvaluator
from tools.sinq.research.cache import CandidateStore
from tools.sinq.research.native_codec import decode,file_sha256,validate_activations


def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,value):path.write_text(json.dumps(value,sort_keys=True),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--bin',required=True);p.add_argument('--summary');args=p.parse_args()
    binaries=Path(args.bin).resolve();suffix='.exe' if os.name=='nt' else ''
    bridge=str(binaries/('llama-sinq-native'+suffix));loader=str(binaries/('llama-sinq-model'+suffix))
    start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='sinq-research-e2e-') as temp:
        d=Path(temp);checkpoint=d/'checkpoint'
        spec=importlib.util.spec_from_file_location('sinq_fixture',ROOT/'tests/test-sinq-groupwise-kl.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        source=d/'source.gguf';fixture.native_fixture(source)
        with np.load(checkpoint/'hf-inputs.npz') as samples:
            captured={name:samples[name].copy() for name in samples.files}
        tokens=json.loads((checkpoint/'tokens.json').read_text())
        source_hash=file_sha256(source)
        names=['blk.0.ffn_up.weight','blk.0.ssm_out.weight','blk.1.attn_output.weight']
        corpus_hash=sha(np.asarray(tokens,dtype='<u4').tobytes());manifests=[]
        for split,begin,end in [('train',0,4),('validation',4,7)]:
            folder=d/split;folder.mkdir();records=[]
            for i,name in enumerate(names):
                x=captured[name][begin:end].astype('<f4');path=folder/f'{i}.f32';x.tofile(path)
                module,site,_=exporter.module_for(name)
                import gguf
                reader=gguf.GGUFReader(source);tensor=next(t for t in reader.tensors if t.name==name)
                k,n=map(int,tensor.shape);reader.data._mmap.close()
                records.append(dict(name=name,site=site,reference_module=module,channel_layout='gguf-input',input_dim=k,output_dim=n,
                    count=len(x),dtype='<f4',order='row-major',data_file=path.name,sha256=file_sha256(path)))
            manifest=dict(schema_version=1,architecture='qwen3_5',source_sha256=source_hash,source_revision='actual-two-layer-qwen35-fixture',
                sampling=dict(contract='valid-token-prefill-reservoir-v1',seed=41721,max_rows=end-begin,corpus_sha256=corpus_hash,
                    token_sha256=sha(np.asarray(tokens[begin:end],dtype='<u4').tobytes())),tensors=records)
            write(folder/'activations.json',manifest);manifests.append(manifest)
        pairing=dict(schema_version=1,source_sha256=source_hash,split_method='tiny_reservoir')
        for split,begin,end,index in [('train',0,4,0),('validation',4,7,1)]:
            pairing[split]=dict(split_id=split,token_sha256=manifests[index]['sampling']['token_sha256'],
                chunks=[dict(token_source_sha256=corpus_hash,start=begin,end=end)],row_ids=[f'fixture-token-{i}' for i in range(begin,end)])
        write(d/'pairing.json',pairing)
        budgets=dict(workspace_bytes=128<<20,max_candidates=32,max_output_bytes=64<<20,max_files=2000,
            max_tensors=8,max_tokens=32,max_probes=32,max_evaluations=200)
        common=dict(schema_version=1,budgets=budgets,threads=24)
        raw_config=dict(common,operation='generate',source_gguf=str(source),activation_manifest=str(d/'train/activations.json'),
            validation_manifest=str(d/'validation/activations.json'),pairing_manifest=str(d/'pairing.json'),tensor_list=names,
            native_quantizer=dict(recipe={name:q for name,q in zip(names,['q3_k','iq3_s','iq4_xs'])},sinq_strip_width=64,sinq_iterations=16,sinq_strength=.75),
            generators=['ordinary','awq_weight_activation','awq_activation_only'],alphas=[.5],correction_strengths=[.5,1],
            search=dict(strategy='small_grid',local_objective='output_l1',selection_policy='held_out',max_shortlist=1,refine_step=.1,tolerance=0),
            output_directory=str(d/'run'),native_bridge=bridge)
        config=validate_config(raw_config);plan=preflight(config)
        assert plan['status']=='PLAN' and not (d/'run').exists()
        run=generate(config);manifest=d/'run/run-manifest.json'
        store=CandidateStore(d/'run/cache',max_bytes=budgets['max_output_bytes'],max_files=budgets['max_files'])
        ids={}
        for name,record in run['tensors'].items():
            assert len(record['candidates'])==5
            ids[name]=dict(stock=record['stock'],ordinary=record['candidates'][0]['candidate_id'],awq=record['candidates'][1]['candidate_id'])
            for candidate in record['candidates']:
                cached=store.get(candidate['candidate_id']);effective=decode(bridge,cached)
                assert effective.shape==tuple(cached.provenance['shape']) and np.isfinite(effective).all()
        # All-source, cache and split digests are checked by native/CLI bridges.
        for objective in ('output_l1','output_mse'):
            score=validate_config(dict(common,operation='score',method='local',objective=objective,candidate_manifest=str(manifest),
                capture_manifest=str(d/'train/activations.json'),validation_manifest=str(d/'validation/activations.json'),
                pairing_manifest=str(d/'pairing.json'),selection_policy='held_out',native_bridge=bridge,
                families=['stock','ordinary','awq_weight_activation','awq_activation_only'],output_directory=str(d/objective)))
            result=score_config(score);assert result['records'] and all(c['encode_count']==0 for c in result['records'])
            if objective=='output_l1':
                observed={r['candidate_id']:r for r in result['records']}
                for record in run['tensors'].values():
                    for c in record['candidates']:
                        assert observed[c['candidate_id']]['train_score']==c['train_score']
                        assert observed[c['candidate_id']]['validation_score']==c['validation_score']
        capture=dict(schema_version=1,source_sha256=source_hash,tokens=[tokens],layers=[0,1],attention_mask=[[0,1,1,1,1,1,1]])
        write(d/'blocks.json',capture)
        for method,objective in [('blockwise','output_mse'),('fisher_vjp_mc','fisher_quadratic')]:
            score=dict(common,operation='score',method=method,objective=objective,candidate_manifest=str(manifest),
                capture_manifest=str(d/'blocks.json'),context='reference_inputs',checkpoint=str(checkpoint),native_bridge=bridge,
                families=['ordinary'],output_directory=str(d/method))
            if method=='fisher_vjp_mc':score.update(probes=2,seed=617)
            result=score_config(validate_config(score));assert len(result['records'])==3
            assert all(np.isfinite(c['score']) and c['score']>=0 for c in result['records'])
        from tools.sinq.research.partition import capture_splits
        import torch
        adapter=_advanced_reference(checkpoint,source,budgets['workspace_bytes'],24)
        capture_splits(adapter,torch.tensor([tokens]),names,d/'production-split',context=2,rows=2,seed=17,
            validation_chunks=1,max_tokens=32,max_output_bytes=budgets['max_output_bytes'])
        for split in ('train','validation'):
            validated=validate_activations(
                bridge,d/'production-split'/split/'activations.json',source)
            assert set(names)==set(validated['tensors'])
        del adapter
        outputs={}
        for state in ('stock','ordinary','awq','mixed'):
            assignment={name:ids[name][state if state!='mixed' else ['stock','ordinary','awq'][i]] for i,name in enumerate(names)}
            path=d/(state+'.gguf')
            cfg=validate_config(dict(common,operation='assemble',source_gguf=str(source),candidate_manifest=str(manifest),
                assignment=assignment,output_gguf=str(path),native_loader=loader))
            assembled=assemble_config(cfg);assert assembled['status']=='PASS'
            proc=subprocess.run([loader,'--forward',str(path),*map(str,tokens)],capture_output=True,text=True,timeout=120)
            assert proc.returncode==0,proc.stderr[-4000:]
            logits=json.loads(proc.stdout)['logits'];assert len(logits)==32 and np.isfinite(logits).all()
            repeat=subprocess.run([loader,'--forward',str(path),*map(str,tokens)],capture_output=True,text=True,timeout=120)
            assert repeat.returncode==0 and repeat.stdout==proc.stdout
            outputs[state]=assembled['output_sha256']
        # Genuine eight-token native full-vocabulary KL wiring and bounded
        # coordinate selection on this tiny graph only (never a quality study).
        corpus=d/'selection.txt';corpus.write_text('abcdefg',encoding='utf-8')
        evaluator=NativeKLEvaluator(binaries/('llama-perplexity'+suffix),context=32,batch=24,microbatch=24,max_tokens=32)
        reference=d/'reference.bin';evaluator.create_reference(source,corpus,reference,expected_tokens=[1,3,4,5,6,7,8,9])
        write(d/'selection-stream.json',dict(schema_version=1,split='selection',corpus_sha256=file_sha256(corpus),reference_sha256=file_sha256(reference)))
        groups={f'g{i}':[name] for i,name in enumerate(names)}
        alternatives={f'g{i}':([{name:ids[name]['ordinary']}] if i==0 else [{name:ids[name]['stock']}]) for i,name in enumerate(names)}
        write(d/'evaluator.json',dict(schema_version=1,executable=str(binaries/('llama-perplexity'+suffix)),reference=str(reference),
            corpus=str(corpus),native_loader=loader,context=32,batch=24,microbatch=24,max_tokens=32,timeout=120,alternatives=alternatives))
        selected=select_config(validate_config(dict(common,budgets={**budgets,'max_evaluations':3},operation='select',source_gguf=str(source),
            candidate_manifest=str(manifest),baseline={name:ids[name]['stock'] for name in names},groups=groups,
            selection_manifest=str(d/'selection-stream.json'),selection_split='selection',evaluator=str(d/'evaluator.json'),
            output_directory=str(d/'selection'),max_passes=1,tolerance=0)))
        assert selected['status']=='PASS' and 1<=selected['evaluations']<=3
        assert file_sha256(reference)==json.loads(reference.with_suffix('.bin.json').read_text())['reference_sha256']
        validation=validate_config(dict(common,operation='validate',artifact_manifest=str(manifest),source_gguf=str(source)))
        assert validate_config_artifacts(validation)['status']=='PASS'
        assert file_sha256(source)==source_hash
        # Repeat full native generation and compare immutable candidate bytes/IDs.
        repeated=generate(validate_config(dict(raw_config,output_directory=str(d/'repeat'))))
        assert {n:[c['candidate_id'] for c in r['candidates']] for n,r in repeated['tensors'].items()}=={n:[c['candidate_id'] for c in r['candidates']] for n,r in run['tensors'].items()}
        # Deliberate negative: overlap is rejected before output creation.
        bad_pair=dict(pairing);bad_pair['validation']={**pairing['validation'],'row_ids':pairing['train']['row_ids']}
        write(d/'bad-pairing.json',bad_pair)
        try:preflight(validate_config(dict(raw_config,pairing_manifest=str(d/'bad-pairing.json'),output_directory=str(d/'bad'))))
        except ValueError:pass
        else:raise AssertionError('overlapping heldout samples accepted')
        assert not (d/'bad').exists()
        summary=dict(status='PASS',purpose='functional correctness only; no A/B experiments',source_sha256=source_hash,
            tensor_count=3,candidates_per_tensor=5,split_before_reservoir_native_validated=True,codecs=['Q3_K','IQ3_S','IQ4_XS'],
            cache_ids={n:[c['candidate_id'] for c in r['candidates']] for n,r in run['tensors'].items()},
            hybrid_sha256=outputs,deterministic_forward_tokens=len(tokens),native_kld_tokens=8,selection_evaluations=selected['evaluations'],cpu_threads=24,
            elapsed_seconds=time.monotonic()-start)
        from tools.sinq.research.resources import peak_rss_bytes
        summary['parent_peak_rss_bytes']=peak_rss_bytes()
        if args.summary:write(Path(args.summary),summary)
        print(json.dumps(summary,sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
