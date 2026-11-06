#!/usr/bin/env python3
"""Real tiny Qwen3.5/codec observations across execution backends, not a quality study."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.update(OMP_NUM_THREADS='32', MKL_NUM_THREADS='32', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'gguf-py')]
import numpy as np
import torch
import gguf
from tools.sinq.optimizer import native_codec
from tools.sinq.optimizer.cache import CandidateStore
from tools.sinq.optimizer.stages import score_config, _advanced_reference
from tools.sinq.optimizer.schemas import validate_config
from tools.sinq.optimizer.capture import capture_block
from tools.sinq.optimizer.blockwise import score_block, block_delta
from tools.sinq.optimizer.fisher import qwen_probes
from tools.sinq.optimizer.resources import canonical_json, digest, file_digest
from tools.sinq.optimizer.execution_backend import resolve_policy
from tools.sinq.optimizer.generate import source_identity

torch.set_num_threads(32)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bin', required=True); p.add_argument('--output', required=True)
    args=p.parse_args(); output=Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint=output/'checkpoint'
    spec=importlib.util.spec_from_file_location('parity',ROOT/'tests/test-sinq-activation-parity.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    fixture.fixture(checkpoint,'bf16',fp32_norms=True)
    source=checkpoint/'model.gguf'; source_hash=file_digest(source)
    bridge=Path(args.bin).resolve()/('llama-sinq-native.exe' if os.name=='nt' else 'llama-sinq-native')
    names=['blk.0.ffn_up.weight','blk.0.ssm_out.weight']
    store=CandidateStore(output/'run/cache',max_bytes=64<<20,max_files=100)
    recipe=digest({'engineering-fixture':'Q5_0/Q8_0'})
    native_source_sha256=source_identity()
    reader=gguf.GGUFReader(source); records={}; assignments={'q5':{},'q8':{}}
    try:
        for index,name in enumerate(names):
            tensor=next(t for t in reader.tensors if t.name==name); k,n=map(int,tensor.shape)
            raw=output/f'original-{index}.f32'
            native_codec.copy_tensor_bytes(tensor,raw,max_bytes=64<<20)
            ids=[]
            for qtype,label in [('q5_0','q5'),('q8_0','q8')]:
                request_dir=output/f'encode-{index}-{label}'; request_dir.mkdir()
                packed=request_dir/'packed'
                response=native_codec.request(bridge,dict(schema_version=2,operation='stock',weights=str(raw),
                    source_type='f32',rows=n,columns=k,codec=qtype,threads=32,workspace_bytes=64<<20,
                    max_output_bytes=8<<20,output=str(packed)),request_dir)
                provenance=dict(source_sha256=source_hash,tensor_name=name,shape=[n,k],source_qtype='F32',
                    qtype=qtype.upper(),input_sha256=file_digest(raw),recipe_sha256=recipe,imatrix_sha256=None,
                    codec=dict(implementation='ggml',chunk_rows=max(1,(16384+k-1)//k),rows_per_expert=n),
                    producer=dict(binary_sha256=file_digest(bridge),source_sha256=native_source_sha256),
                    sinq=None,generator={'generator':'stock'},activation=None)
                key=store.put_files(provenance,packed/response['payload']); ids.append(key);assignments[label][name]=key
            records[name]=dict(stock=ids[0],candidates=[dict(candidate_id=ids[1])])
    finally: reader.data._mmap.close()
    run=dict(schema_version=2,source_gguf='../checkpoint/model.gguf',source_sha256=source_hash,cache='cache',
             recipe_sha256=recipe,imatrix_sha256=None,tensors=records)
    run['checksum']=digest(run); manifest=output/'run/run-manifest.json';manifest.write_bytes(canonical_json(run))
    tokens=torch.tensor([[1,7,3,11,5,19,2]])
    capture_file=output/'capture.json';capture_file.write_bytes(canonical_json(dict(schema_version=2,
        source_sha256=source_hash,tokens=tokens.tolist(),layers=[0])))
    budgets=dict(workspace_bytes=128<<20,max_candidates=4,max_output_bytes=64<<20,max_files=100,
                 max_tensors=4,max_tokens=16,max_probes=16,max_evaluations=8)
    common=dict(schema_version=2,
        inputs=dict(candidate_manifest=str(manifest),capture_manifest=str(capture_file),
            checkpoint=str(checkpoint),native_bridge=str(bridge)),
        limits=dict(workspace_bytes=128<<20,max_artifact_bytes=64<<20,max_files=100,
            max_tensors=4,max_tokens=16,max_block_scores=8,max_fisher_vjps=16))
    modes=[('cpu',{}),]
    if torch.cuda.is_available():
        # Both devices are exercised in the same run, including cross-device VJPs.
        modes += [(f'cuda:{i}',{}) for i in range(torch.cuda.device_count())]
        modes += [('cuda',{})]
        if torch.cuda.device_count()>1:
            modes += [('cuda',{'cuda_devices':[f'cuda:{i}' for i in range(torch.cuda.device_count())]})]
    observations=[]
    for device,extra in modes:
        for dtype in ('fp32','bf16'):
            label=device.replace(':','-')+('-layerwise' if extra else '')+'-'+dtype
            execution=dict(device=device,reference_dtype=dtype,
                memory_policy=dict(host_limit_bytes=8<<30,reserve_bytes=64<<20),**extra)
            cfg={**common,'execution':dict(threads=32,scoring=execution)}
            policy=resolve_policy(execution)
            adapter=_advanced_reference(checkpoint,source,budgets['workspace_bytes'],32,policy=policy)
            assert all(not p.requires_grad and p.grad is None for p in adapter.reference.parameters())
            assert adapter.reference.lm_head.weight is adapter.text.embed_tokens.weight
            if dtype=='bf16':
                exceptions=adapter.identity['parameter_storage_exceptions']
                assert 'model.language_model.layers.0.linear_attn.norm.weight' in exceptions
                norm=adapter.text.layers[0].linear_attn.norm.weight
                assert norm.dtype==torch.float32 and not torch.equal(norm,norm.bfloat16().float())
                assert adapter.text.layers[0].mlp.up_proj.weight.dtype==torch.bfloat16
            locked=capture_block(adapter,tokens,0)
            assert score_block(adapter,locked,{})['output_mse']==0
            with torch.no_grad():
                logits=adapter.logits(tokens).detach().float().cpu()
                repeated=adapter.logits(tokens).detach().float().cpu()
            assert torch.equal(logits,repeated), 'same-backend determinism failed'
            original={name:adapter.effective_weight(name) for name in names}
            assert score_block(adapter,locked,original)['output_mse']==0
            # Raised exception, OOM-like failure and cancellation all restore exact storage.
            for failure in (RuntimeError('injected'),torch.OutOfMemoryError('injected OOM'),KeyboardInterrupt()):
                try:
                    with adapter.swapped({name:value+.125 for name,value in original.items()}): raise failure
                except (RuntimeError,KeyboardInterrupt): pass
                adapter.assert_reference()
                for name,value in original.items(): np.testing.assert_array_equal(adapter.effective_weight(name),value)
            probe=qwen_probes(adapter,locked,probes=2,seed=617,positions=[(0,6)])
            assert probe.gradients.abs().max()>0 and torch.isfinite(probe.gradients).all()
            assert all(p.grad is None for p in adapter.reference.parameters())
            if extra:
                assert adapter.identity['placement']['layers.0']!=adapter.identity['placement']['layers.1'], adapter.identity['placement']
                assert probe.metadata['device']!=probe.metadata['logits_device'] or adapter.identity['placement']['layers.0']!=adapter.identity['placement']['layers.1']
            # A downstream detach must not silently generate a score.
            handle=adapter.text.layers[1].register_forward_hook(lambda module,args,out:out.detach())
            try:
                try:qwen_probes(adapter,locked,probes=1,positions=[(0,6)])
                except (ValueError,RuntimeError):pass
                else:raise AssertionError('detached downstream graph accepted')
            finally:handle.remove()
            del probe,locked,adapter;gc.collect()
            if device!='cpu':torch.cuda.empty_cache()
            entry=dict(label=label,device=device,dtype=dtype,logits=logits.tolist())
            for method,objective in [('blockwise','output_mse'),('fisher_vjp_mc','fisher_quadratic')]:
                path=output/(label+'-'+method)
                raw=dict(cfg,scoring=dict(kind='blockwise_output_mse' if method=='blockwise' else 'fisher_vjp_mc',
                    context='reference_inputs',assignments=assignments),output=dict(directory=str(path)))
                raw['limits']=dict(raw['limits'])
                if method=='fisher_vjp_mc':raw['scoring'].update(probes=2,seed=617)
                else:raw['limits'].pop('max_fisher_vjps')
                (output/(label+'-'+method+'-config.json')).write_bytes(canonical_json(raw))
                result=score_config(validate_config(raw,'score'))
                assert len(result['records'])==2
                assert all(np.isfinite(r['score']) and r['score']>0 for r in result['records'])
                assert result['reference_identity']['source_gguf_sha256']==source_hash
                assert result['checksum']==digest({k:v for k,v in result.items() if k!='checksum'})
                entry[method]=[dict(assignment=r['assignment_name'],score=r['score']) for r in result['records']]
            observations.append(entry)
            print('PASS '+label,flush=True)
            gc.collect()
            if device!='cpu':torch.cuda.empty_cache()
    baseline=torch.tensor(observations[0]['logits'])
    for entry in observations:
        entry['max_abs_logits_delta_cpu_fp32']=float((torch.tensor(entry.pop('logits'))-baseline).abs().max())
        entry['rank_agreement_cpu_fp32']={method:sorted(entry[method],key=lambda r:r['score'])[0]['assignment']==
            sorted(observations[0][method],key=lambda r:r['score'])[0]['assignment'] for method in ('blockwise','fisher_vjp_mc')}
    summary=dict(status='PASS',purpose='engineering fixture, not quality study',cpu_threads=torch.get_num_threads(),
        torch=torch.__version__,cuda=torch.version.cuda,hardware=[dict(index=i,name=torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())],
        source_sha256=source_hash,candidate_manifest_sha256=file_digest(manifest),observations=observations)
    (output/'summary.json').write_bytes(canonical_json(summary))
    print(json.dumps(summary,sort_keys=True))

if __name__=='__main__':main()
