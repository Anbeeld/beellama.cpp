"""One opt-in config entry point: python -m tools.sinq.research.cli CONFIG [--plan]."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'gguf-py'))
from .schemas import load_config, validate_pairing
from .resources import atomic_write, canonical_json, digest, file_digest, reject_alias
from .cache import CandidateStore
from .generate import generate, preflight
from . import native_codec


def _json(path):
    if Path(path).stat().st_size>8<<20:
        raise ValueError('manifest byte budget exceeded')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _run(config):
    path=Path(config['candidate_manifest']).resolve();run=_json(path)
    if run.get('schema_version')!=1 or run.get('checksum')!=digest({k:v for k,v in run.items() if k!='checksum'}):
        raise ValueError('candidate run manifest schema/checksum mismatch')
    source=(path.parent/run['source_gguf']).resolve()
    if file_digest(source)!=run['source_sha256']:
        raise ValueError('candidate source GGUF checksum mismatch')
    cache=(path.parent/run['cache']).resolve()
    if cache!=path.parent/'cache':
        raise ValueError('candidate cache must be the run-owned cache directory')
    b=config['budgets'];store=CandidateStore(cache,max_bytes=b['max_output_bytes'],max_files=b['max_files'])
    ids=[];requested_scores=0
    for name,record in sorted(run['tensors'].items()):
        for key in [record['stock'],*[r['candidate_id'] for r in record['candidates']]]:
            candidate=store.get(key,expected={'source_sha256':run['source_sha256'],'tensor_name':name})
            if candidate.provenance['generator']['family'] in config.get('families',[]):requested_scores+=1
            ids.append(key)
    if len(run['tensors'])>b['max_tensors'] or len(ids)>b['max_candidates']*b['max_tensors']:
        raise ValueError('candidate manifest count budget exceeded')
    if config['operation']=='score' and requested_scores>b['max_evaluations']:
        raise ValueError('requested candidate scoring work budget exceeded')
    return run,store,source


def _validator(binary):
    if not binary or not Path(binary).is_file():
        raise ValueError('a local native_loader (llama-sinq-model) is required')
    def validate(path):
        result=subprocess.run([str(Path(binary).resolve()),'--load-only',str(Path(path).resolve())],
            env={**os.environ,'OMP_NUM_THREADS':'24','MKL_NUM_THREADS':'24','CUDA_VISIBLE_DEVICES':'0'},
            capture_output=True,timeout=120)
        if result.returncode:
            raise RuntimeError('native loader rejected hybrid; exit '+str(result.returncode))
    return validate


def assemble_config(config):
    from .hybrid import assemble
    run,store,source=_run(config)
    if file_digest(config['source_gguf'])!=run['source_sha256']:
        raise ValueError('assembly source mismatch')
    return dict(status='PASS',output_sha256=assemble(source,config['output_gguf'],store,config['assignment'],
        native_validate=_validator(config.get('native_loader')),max_output_bytes=config['budgets']['max_output_bytes']))


def _new_output(config,inputs):
    output=Path(config['output_directory']).resolve()
    reject_alias(output,inputs)
    if output.exists():raise FileExistsError('immutable research output already exists')
    output.mkdir(parents=True,exist_ok=False)
    return output


def _advanced_reference(checkpoint,source,budget,threads):
    import torch
    from safetensors import safe_open
    from transformers import Qwen3_5TextConfig,Qwen3_5TextModel,Qwen3_5ForConditionalGeneration
    from .qwen35_adapter import Qwen35Adapter
    checkpoint=Path(checkpoint)
    if not checkpoint.is_dir():raise ValueError('explicit local checkpoint directory required')
    # Bound full reference residency before allocation. This first graph adapter
    # uses one resident reference, never multiple candidate model copies.
    files=sorted(checkpoint.glob('*.safetensors'))
    if not files:raise ValueError('local safetensors checkpoint required')
    elements=0
    for path in files:
        with safe_open(path,framework='pt',device='cpu') as weights:
            for name in weights.keys():
                import math
                elements+=math.prod(weights.get_slice(name).get_shape())
    if elements*8>budget:raise ValueError('reference graph exceeds configured resident memory budget')
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    cfg=_json(checkpoint/'config.json')
    if 'Qwen3_5TextModel' in cfg.get('architectures',[]):
        model=Qwen3_5TextModel.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False,
            dtype=torch.float32,attn_implementation='eager').eval()
        return Qwen35Adapter(model,source,workspace_limit=budget,max_swap_bytes=budget)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(str(checkpoint),local_files_only=True,
        trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval()
    return Qwen35Adapter(model.model.language_model,source,lm_head=model.lm_head,
                         workspace_limit=budget,max_swap_bytes=budget)


def score_config(config):
    import gguf
    run,store,source=_run(config);method=config['method'];b=config['budgets']
    if not config.get('native_bridge'):raise ValueError('native_bridge is required for actual candidate decoding/scoring')
    capture=_json(config['capture_manifest']);records=[]
    if method=='local':
        validated=native_codec.validate_activations(config['native_bridge'],config['capture_manifest'],source)
        validation=None
        if config['selection_policy']=='held_out':
            validation=native_codec.validate_activations(config['native_bridge'],config['validation_manifest'],source)
            validate_pairing(_json(config['pairing_manifest']),capture,_json(config['validation_manifest']))
        reader=gguf.GGUFReader(source)
        try:
            tensors={t.name:t for t in reader.tensors}
            for name,t in sorted(run['tensors'].items()):
                if name not in validated['tensors']:raise ValueError('missing local capture site: '+name)
                a=validated['tensors'][name];v=validation['tensors'][name] if validation else None
                for key in [t['stock'],*[c['candidate_id'] for c in t['candidates']]]:
                    c=store.get(key)
                    if c.provenance['generator']['family'] not in config['families']:continue
                    score=native_codec.rescore(config['native_bridge'],c,tensors[name],a,validation=v,
                        objective=config['objective'],workspace_bytes=b['workspace_bytes'])
                    records.append({**score,'candidate_id':key,'tensor':name,'method':'local','objective':config['objective'],
                        'capture_sha256':file_digest(config['capture_manifest'])})
        finally:reader.data._mmap.close()
    else:
        import torch
        from .blockwise import score_block,block_delta
        from .capture import capture_block
        from .fisher import qwen_probes
        if config['device']!='cpu':raise ValueError('this graph/VJP adapter supports explicit CPU execution only')
        allowed={'schema_version','source_sha256','tokens','layers','attention_mask','position_ids','baseline'}
        if set(capture)-allowed or not {'schema_version','source_sha256','tokens','layers'}<=set(capture) or capture['schema_version']!=1 or capture['source_sha256']!=run['source_sha256']:
            raise ValueError('invalid versioned block capture specification')
        tokens=torch.tensor(capture['tokens'],dtype=torch.long)
        if tokens.ndim!=2 or tokens.numel()>b['max_tokens']:raise ValueError('block token budget exceeded')
        mask=torch.tensor(capture['attention_mask']) if 'attention_mask' in capture else None
        positions=torch.tensor(capture['position_ids'],dtype=torch.long) if 'position_ids' in capture else None
        adapter=_advanced_reference(config['checkpoint'],source,b['workspace_bytes'],config['threads'])
        baseline={name:native_codec.decode(config['native_bridge'],store.get(key),workspace_bytes=b['workspace_bytes'])
                  for name,key in capture.get('baseline',{}).items()}
        for layer in capture['layers']:
            locked=capture_block(adapter,tokens,layer,context=config['context'],baseline=baseline,
                attention_mask=mask,position_ids=positions,max_tokens=b['max_tokens'],max_capture_bytes=b['workspace_bytes'])
            probes=qwen_probes(adapter,locked,probes=config['probes'],seed=config['seed'],max_vjps=b['max_probes'],max_cache_bytes=b['workspace_bytes']) if method=='fisher_vjp_mc' else None
            for name,t in sorted(run['tensors'].items()):
                if not name.startswith(f'blk.{layer}.'):continue
                for key in [t['stock'],*[c['candidate_id'] for c in t['candidates']]]:
                    c=store.get(key)
                    if c.provenance['generator']['family'] not in config['families']:continue
                    effective=native_codec.decode(config['native_bridge'],c,workspace_bytes=b['workspace_bytes'])
                    if probes is None:
                        score=score_block(adapter,locked,{name:effective});value=score[config['objective']]
                    else:
                        value=probes.score(block_delta(adapter,locked,{name:effective}));score=dict(provenance=probes.metadata)
                    records.append(dict(candidate_id=key,tensor=name,layer=layer,method=method,
                        objective=config['objective'],score=value,details=score))
    if not records:raise ValueError('requested score stage produced no candidate observations')
    output=_new_output(config,[source,config['candidate_manifest'],config['capture_manifest']])
    result=dict(schema_version=1,status='PASS',method=method,objective=config['objective'],records=records,
                candidate_manifest_sha256=file_digest(config['candidate_manifest']))
    atomic_write(output/'scores.json',canonical_json(result),max_bytes=b['max_output_bytes'])
    return result


def select_config(config):
    from .selector import State,Group,coordinate_descent
    from .hybrid import assemble
    from .kl import NativeKLEvaluator
    run,store,source=_run(config);b=config['budgets']
    if file_digest(config['source_gguf'])!=run['source_sha256']:
        raise ValueError('selection source does not match candidate provenance')
    selection=_json(config['selection_manifest']);e=_json(config['evaluator'])
    allowed={'schema_version','executable','reference','corpus','native_loader','context','batch','microbatch','max_tokens','timeout','alternatives'}
    if set(e)!=allowed or e['schema_version']!=1:raise ValueError('invalid native evaluator configuration')
    if selection.get('split')!='selection' or selection.get('corpus_sha256')!=file_digest(e['corpus']) or selection.get('reference_sha256')!=file_digest(e['reference']):
        raise ValueError('selection stream/reference provenance mismatch (final-test prohibited)')
    if selection.get('schema_version')!=1 or _json(str(e['reference'])+'.json').get('source_sha256')!=run['source_sha256']:
        raise ValueError('selection reference is not bound to the candidate source')
    if set(config['baseline'])!=set(run['tensors']):raise ValueError('baseline must assign the complete run tensor allocation')
    if e['max_tokens']>b['max_tokens']:raise ValueError('KL token budget exceeded')
    if b['max_evaluations']+8>b['max_files']:raise ValueError('selector file budget cannot hold score/checkpoint closure')
    if source.stat().st_size*2>b['max_output_bytes']:raise ValueError('selector disk budget cannot hold a bounded hybrid evaluation')
    evaluator=NativeKLEvaluator(e['executable'],context=e['context'],batch=e['batch'],microbatch=e['microbatch'],max_tokens=e['max_tokens'],timeout=e['timeout'])
    validator=_validator(e['native_loader'])
    for name,key in config['baseline'].items():store.get(key,expected={'tensor_name':name,'source_sha256':run['source_sha256']})
    groups=[]
    if set(e['alternatives'])!=set(config['groups']):raise ValueError('group alternatives mismatch')
    for name,members in sorted(config['groups'].items()):
        for alternative in e['alternatives'][name]:
            for tensor,key in alternative.items():store.get(key,expected={'tensor_name':tensor,'source_sha256':run['source_sha256']})
        groups.append(Group(name,tuple(members),tuple(e['alternatives'][name])))
    output=_new_output(config,[source,config['candidate_manifest'],config['selection_manifest'],config['evaluator']])
    provenance=dict(source_sha256=run['source_sha256'],selection_sha256=file_digest(config['selection_manifest']),evaluator_sha256=file_digest(config['evaluator']),policy=evaluator.policy)
    initial=State.create(config['baseline'],provenance)
    def evaluate(state):
        # Each state is a full immutable assignment; no partial GGUF modification.
        with tempfile.TemporaryDirectory(prefix='.evaluation-',dir=output) as temporary:
            path=Path(temporary)/'hybrid.gguf'
            assemble(source,path,store,dict(state.assignment),native_validate=validator,max_output_bytes=b['max_output_bytes'])
            return evaluator.evaluate(path,e['corpus'],e['reference'],split=config['selection_split'],cache_directory=output/'scores')
    result=coordinate_descent(initial,groups,evaluate,checkpoint=output/'checkpoint.json',
        max_evaluations=b['max_evaluations'],max_passes=config['max_passes'],tolerance=config['tolerance'],split=config['selection_split'])
    best_hash=assemble(source,output/'best.gguf',store,dict(result.state.assignment),native_validate=validator,max_output_bytes=b['max_output_bytes'])
    summary=dict(schema_version=1,status='PASS',best_gguf_sha256=best_hash,assignment=dict(result.state.assignment),selection_kld=result.score,
        evaluations=result.evaluations,budget_exhausted=result.budget_exhausted,trace=result.trace,
        warning='Selection objective only; not an untouched-test result or quality conclusion.')
    atomic_write(output/'selection.json',canonical_json(summary),max_bytes=b['max_output_bytes'])
    return summary


def validate_config_artifacts(config):
    path=Path(config['artifact_manifest']);run=_json(path)
    surrogate=dict(config,candidate_manifest=str(path))
    run,store,source=_run(surrogate)
    if 'source_gguf' in config and file_digest(config['source_gguf'])!=run['source_sha256']:raise ValueError('validation source mismatch')
    return dict(status='PASS',source_sha256=run['source_sha256'],tensor_count=len(run['tensors']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config');parser.add_argument('--plan',action='store_true')
    args=parser.parse_args(argv);config=load_config(args.config)
    if args.plan:
        result=preflight(config) if config['operation']=='generate' else dict(status='PLAN',operation=config['operation'],resource_bounds=config['resource_bounds'])
    else:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        result={'generate':generate,'score':score_config,'assemble':assemble_config,'select':select_config,'validate':validate_config_artifacts}[config['operation']](config)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,OSError,KeyError,subprocess.TimeoutExpired) as error:
        print('SINQ research: '+str(error),file=sys.stderr);raise SystemExit(1)
