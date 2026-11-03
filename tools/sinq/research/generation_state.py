"""Strict generation identity, tensor-boundary recovery and immutable cache reuse."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .resources import atomic_write,canonical_json,digest,file_digest
from .run_state import implementation_fingerprint,open_run,selection_lock
from .corpora import load_corpora,bind_corpora
from .implementation import native_runtime_fingerprint


def capture_bindings(config):
    result={}
    for role,key in [('calibration','activation_manifest'),('validation','validation_manifest')]:
        if key not in config:
            result[role]=None;continue
        value=json.loads(Path(config[key]).read_text(encoding='utf-8'))
        sampling=value.get('sampling',{});token=sampling.get('token_sha256')
        if not isinstance(token,str) or len(token)!=64 or any(c not in '0123456789abcdef' for c in token):
            raise ValueError('capture requires an actual token_sha256 for cross-stage provenance')
        binding=dict(manifest_sha256=file_digest(config[key]),token_sha256=token)
        if 'corpus_sha256' in sampling:binding['corpus_sha256']=sampling['corpus_sha256']
        if 'token_count' in value.get('provenance',{}):binding['token_count']=value['provenance']['token_count']
        result[role]=binding
    return result


def generation_identity(config,plan):
    from .generate import source_identity
    paths={key:config[key] for key in ('source_gguf','activation_manifest','validation_manifest','pairing_manifest',
        'native_bridge','stock_baseline_manifest','corpus_partitions_manifest') if key in config}
    if 'imatrix_path' in config['native_quantizer']:paths['imatrix']=config['native_quantizer']['imatrix_path']
    implementation=implementation_fingerprint()
    identity=dict(schema_version=1,kind='sinq-generation-run',config_sha256=digest(config),
        bindings={key:file_digest(path) for key,path in sorted(paths.items())},native_source_sha256=source_identity(),
        stock_baseline=plan['stock_baseline'],implementation=implementation,implementation_sha256=implementation['checksum'],
        native_runtime=native_runtime_fingerprint({'bridge':config['native_bridge']}))
    identity['checksum']=digest(identity)
    return identity


def validate_completed(store,run,name,tensor,details,config):
    from .generate import NUMERICS
    record=run['tensors'][name]
    if not isinstance(record,dict) or set(record)!={'stock','candidates','selected','skipped','encode_count','workspace_bytes','elapsed_seconds'}:
        raise ValueError('invalid generation checkpoint tensor record')
    candidates=record['candidates']
    if not isinstance(candidates,list) or not candidates or len(candidates)>config['resource_bounds']['candidates_per_tensor']:
        raise ValueError('checkpoint candidate count mismatch')
    raw=np.asarray(tensor.data).view(np.uint8).reshape(-1);h=hashlib.sha256()
    for offset in range(0,len(raw),1<<20):h.update(raw[offset:offset+(1<<20)])
    expected=dict(source_sha256=run['source_sha256'],tensor_name=name,shape=details['shape'],
        source_qtype=tensor.tensor_type.name,qtype=details['qtype'],input_sha256=h.hexdigest(),
        recipe_sha256=run['recipe_sha256'],imatrix_sha256=run['imatrix_sha256'],producer=run['producer'],
        codec=dict(implementation='ggml',chunk_rows=max(1,(16384+details['shape'][1]-1)//details['shape'][1]),rows_per_expert=details['shape'][0]))
    stock=store.get(record['stock'],expected=expected)
    if stock.provenance['generator']!={'family':'stock'} or stock.provenance['sinq'] is not None:
        raise ValueError('checkpoint stock provenance mismatch')
    keys=[]
    for item in candidates:
        if not isinstance(item,dict) or set(item)!={'candidate_id','generator','train_score','validation_score','encoded','duplicate_of','stage','encode_reason'}:
            raise ValueError('invalid checkpoint candidate record')
        candidate=store.get(item['candidate_id'],expected=expected);p=candidate.provenance
        q=config['native_quantizer']
        if p['generator']!=item['generator'] or p['sinq']!=dict(strip_width=q['sinq_strip_width'],iterations=q['sinq_iterations'],
            strength=q['sinq_strength'],algorithm='sinkhorn_log',numerics=NUMERICS):
            raise ValueError('checkpoint candidate generation provenance mismatch')
        wanted=None if p['generator']['family']=='ordinary' else dict(source_sha256=run['corpus_bindings']['calibration']['manifest_sha256'],
            split_sha256=digest(json.loads(Path(config['pairing_manifest']).read_text())['train']) if 'pairing_manifest' in config else run['corpus_bindings']['calibration']['manifest_sha256'])
        if p['activation']!=wanted:raise ValueError('checkpoint activation provenance mismatch')
        for field in ('train_score','validation_score'):
            score=item[field]
            if field=='validation_score' and config['search']['selection_policy']=='in_sample':
                if score is not None:raise ValueError('unexpected checkpoint validation score')
            elif type(score) not in (int,float) or not math.isfinite(score) or score<0:
                raise ValueError('invalid checkpoint local score')
        keys.append(item['candidate_id'])
    if record['selected'] not in keys:raise ValueError('checkpoint selected candidate missing')
    if type(record['encode_count']) is not int or not 0<=record['encode_count']<=len(candidates):raise ValueError('checkpoint encode count mismatch')
    if type(record['workspace_bytes']) is not int or not 0<=record['workspace_bytes']<=config['budgets']['workspace_bytes']:raise ValueError('checkpoint workspace mismatch')
    if type(record['elapsed_seconds']) not in (int,float) or not math.isfinite(record['elapsed_seconds']) or record['elapsed_seconds']<0:raise ValueError('checkpoint elapsed diagnostic mismatch')


def run_generation(config,resume,execute):
    from .generate import preflight
    from .cache import CandidateStore
    plan=preflight(config,resume=resume)
    bindings=capture_bindings(config)
    corpus=load_corpora(config['corpus_partitions_manifest']) if 'corpus_partitions_manifest' in config else None
    corpus_identity=bind_corpora(corpus,bindings)
    identity=generation_identity(config,plan)
    inputs=[config[key] for key in ('source_gguf','activation_manifest','validation_manifest','pairing_manifest',
        'native_bridge','stock_baseline_manifest','corpus_partitions_manifest') if key in config]
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
    output=open_run(config,resume,identity,inputs,read)
    budget=config['budgets'];checkpoint=output/'generation-checkpoint.json'
    cache=output/'cache'
    if cache.is_symlink() or cache.resolve()!=cache:
        raise ValueError('generation cache must be the run-owned directory')
    with selection_lock(output,name='generation'),CandidateStore(output/'cache',max_bytes=budget['max_output_bytes'],max_files=budget['max_files']).verified_session() as store:
        completed={}
        if checkpoint.exists():
            if checkpoint.stat().st_size>budget['max_output_bytes']:raise ValueError('generation checkpoint byte budget exceeded')
            state=read(checkpoint)
            if set(state)!={'schema_version','run_identity_sha256','completed','checksum'} or state['schema_version']!=1 or state['checksum']!=digest({k:v for k,v in state.items() if k!='checksum'}) or state['run_identity_sha256']!=identity['checksum']:
                raise ValueError('generation checkpoint checksum/identity mismatch')
            completed=state['completed']
            names=[item['name'] for item in plan['tensor_details']]
            if not isinstance(completed,dict) or set(completed)!=set(names[:len(completed)]):raise ValueError('generation checkpoint is not a complete tensor prefix')
        def output_budget(target,data):
            count=store.index_stats['files'];size=store.index_stats['bytes']
            for path in output.iterdir():
                if path==output/'cache':continue
                if path.is_symlink() or not path.is_file():raise ValueError('unexpected generation output artifact')
                if path!=target:count+=1;size+=path.stat().st_size
            if count+1>budget['max_files'] or size+len(data)>budget['max_output_bytes']:
                raise ValueError('generation output resource budget exceeded')
        output_budget(output/'run-identity.json',(output/'run-identity.json').read_bytes())
        def save(tensors):
            state=dict(schema_version=1,run_identity_sha256=identity['checksum'],completed=tensors)
            state['checksum']=digest(state)
            data=canonical_json(state);output_budget(checkpoint,data)
            atomic_write(checkpoint,data,max_bytes=budget['max_output_bytes'],immutable=False)
        def verify(run):
            if generation_identity(config,preflight(config,resume=output))!=identity:
                raise ValueError('generation inputs or implementation changed during execution')
            output_budget(output/'run-manifest.json',canonical_json(run))
        return execute(config,plan,output,store,completed,save,verify,bindings,corpus_identity,identity['checksum'])
