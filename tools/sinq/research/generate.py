"""Artifact preflight and orchestration; all scale/codec/search arithmetic is C++."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import gguf
import numpy as np
from tools.sinq import export_activations as exporter
from . import native_codec
from .schemas import validate_pairing

NUMERICS='fp16-source-statistics;fp32-best-preupdate;serialized-f16-diagonal;original-W/t;imatrix*t^2'
ROOT=Path(__file__).resolve().parents[3]


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def source_identity():
    paths=[ROOT/'src/llama-sinq.cpp',ROOT/'tools/sinq/native-codec.cpp',
           ROOT/'ggml/src/ggml.c',ROOT/'ggml/src/ggml-quants.c']
    paths+=sorted((ROOT/'src/sinq').glob('*'))
    return digest({p.relative_to(ROOT).as_posix():native_codec.file_sha256(p) for p in paths if p.is_file()})


def _manifest(config, key):
    path=Path(config[key])
    value=json.loads(path.read_text(encoding='utf-8'))
    validated=native_codec.validate_activations(config['native_bridge'],path,config['source_gguf'])
    if validated['source_sha256']!=native_codec.file_sha256(config['source_gguf']):
        raise ValueError('activation/source identity changed')
    return value,validated


def preflight(config,*,resume=None):
    """No writes, no encode: resolve candidates, inputs, native v1 manifests and bounds."""
    source=Path(config['source_gguf']).resolve()
    if not Path(config['native_bridge']).is_file():
        raise ValueError('configured local native bridge is unavailable')
    reader=gguf.GGUFReader(source)
    try:
        if reader.fields['general.architecture'].contents()!='qwen35':
            raise ValueError('only verified Qwen3.5 text projections are eligible')
        if any(k.startswith('split.') or k.startswith('beellama.qk_sinq.') for k in reader.fields):
            raise ValueError('split/transformed source is unsupported')
        tensors={t.name:t for t in reader.tensors}
        if len(tensors)!=len(reader.tensors):
            raise ValueError('duplicate source tensor names')
        stock=None
        if 'stock_baseline_manifest' in config:
            from .baseline import load_baseline
            locked,_=load_baseline(config['stock_baseline_manifest'],source)
            imatrix=config['native_quantizer'].get('imatrix_path')
            if locked['imatrix_sha256']!=(native_codec.file_sha256(imatrix) if imatrix else None):
                raise ValueError('generation imatrix differs from stock baseline')
            stock=dict(manifest=str(Path(config['stock_baseline_manifest']).resolve()),
                manifest_sha256=native_codec.file_sha256(config['stock_baseline_manifest']),
                baseline_sha256=locked['baseline_sha256'],recipe_sha256=locked['recipe_sha256'])
            for name,qtype in config['native_quantizer']['recipe'].items():
                if locked['recipe'].get(name)!=qtype.upper():
                    raise ValueError('selected recipe differs from complete stock baseline allocation: '+name)
        manifest,activations=_manifest(config,'activation_manifest')
        validation=None;pairing=None
        if 'validation_manifest' in config:
            val_manifest,validation=_manifest(config,'validation_manifest')
            pairing=validate_pairing(json.loads(Path(config['pairing_manifest']).read_text()),manifest,val_manifest)
        if config['search']['selection_policy']=='held_out' and validation is None:
            raise ValueError('held-out data missing')
        if manifest.get('provenance',{}).get('token_count',max(t['count'] for t in activations['tensors'].values()))>config['budgets']['max_tokens']:
            raise ValueError('activation token budget exceeded')
        output=Path(config['output_directory']).resolve()
        if resume is not None:
            if Path(resume).resolve()!=output or not output.is_dir():
                raise ValueError('resume directory must be the existing configured output directory')
        elif output.exists():
            raise FileExistsError('research output must be new; use --resume RUN_DIRECTORY')
        for name in ('source_gguf','activation_manifest','validation_manifest','pairing_manifest'):
            if name in config and Path(config[name]).resolve()==output:
                raise ValueError('output aliases input')
        bound=config['resource_bounds']['candidates_per_tensor']+1 # include stock
        total=1<<20;details=[] # reserve the bounded implementation/run identity
        for name in config['tensor_list']:
            if not exporter.PATTERN.fullmatch(name) or name not in tensors:
                raise ValueError('missing/ineligible exact text projection: '+name)
            t=tensors[name]
            if len(t.shape)!=2 or t.tensor_type.name not in ('F32','F16','BF16'):
                raise ValueError('source projection must be original floating matrix')
            k,n=map(int,t.shape)
            qtype=gguf.GGMLQuantizationType[config['native_quantizer']['recipe'][name].upper()]
            block,size=gguf.GGML_QUANT_SIZES[qtype]
            if n<2 or k%block or k%config['native_quantizer']['sinq_strip_width']:
                raise ValueError('incompatible codec/strip dimensions')
            for m in (activations,validation):
                if m is None:continue
                a=m['tensors'].get(name)
                if a is None or (a['input_dim'],a['output_dim'])!=(k,n):
                    raise ValueError('selected projection missing/misaligned activation: '+name)
            if t.n_bytes+n*k*4>=config['budgets']['workspace_bytes']//2:
                raise ValueError('source tensor exceeds resident workspace budget')
            count=n*k//block*size
            total+=bound*(count+k*2+8192)
            details.append(dict(name=name,shape=[n,k],qtype=qtype.name,packed_bytes=count))
        files=bound*len(details)*4+5
        if bound*len(details)>config['budgets']['max_evaluations']:
            raise ValueError('native candidate plus stock work budget exceeded')
        if total>config['budgets']['max_output_bytes'] or files>config['budgets']['max_files']:
            raise ValueError('resolved disk/file budget exceeded')
        return dict(schema_version=1,operation='generate',status='PLAN',source_sha256=native_codec.file_sha256(source),
                    resolved_candidates=config['resolved_candidates'],search_policy=config['search'],tensor_details=details,
                    expected_output_bytes_upper_bound=total,expected_files_upper_bound=files,
                    activation=activations,validation=validation,pairing=pairing,stock_baseline=stock)
    finally:
        reader.data._mmap.close()


def _importance(path,name,k):
    if path is None:return None
    reader=gguf.GGUFReader(path)
    try:
        tensors={t.name:t for t in reader.tensors}
        if name+'.in_sum2' not in tensors or name+'.counts' not in tensors:
            raise ValueError('imatrix lacks selected tensor: '+name)
        sums=tensors[name+'.in_sum2'];counts=tensors[name+'.counts']
        if sums.tensor_type!=gguf.GGMLQuantizationType.F32 or counts.tensor_type!=gguf.GGMLQuantizationType.F32:
            raise ValueError('imatrix requires native floating sums/counts')
        a=np.asarray(sums.data,dtype=np.float32).reshape(-1)
        c=np.asarray(counts.data,dtype=np.float32).reshape(-1)
        if a.size!=k or c.size!=1 or not np.isfinite(a).all() or not np.isfinite(c).all() or c[0]<=0 or (a<0).any():
            raise ValueError('invalid dense imatrix cardinality/statistics')
        return (a/c[0]).astype('<f4')
    finally:
        reader.data._mmap.close()


def _generate(config,plan,output,store,completed,save,verify,bindings,corpus_identity,identity_sha256):
    from .resources import atomic_write
    from .generation_state import validate_completed
    from .implementation import native_runtime_fingerprint
    reader=gguf.GGUFReader(config['source_gguf'])
    source_sha=plan['source_sha256'];q=config['native_quantizer'];budgets=config['budgets']
    recipe_sha=plan['stock_baseline']['recipe_sha256'] if plan['stock_baseline'] else digest(q['recipe']);imatrix_sha=native_codec.file_sha256(q['imatrix_path']) if 'imatrix_path' in q else None
    producer=dict(binary_sha256=native_codec.file_sha256(config['native_bridge']),source_sha256=source_identity(),
        runtime_sha256=digest(native_runtime_fingerprint({'bridge':config['native_bridge']})))
    run=dict(schema_version=1,source_gguf=os.path.relpath(Path(config['source_gguf']).resolve(),output),source_sha256=source_sha,
             cache='cache',recipe_sha256=recipe_sha,imatrix_sha256=imatrix_sha,producer=producer,
             config_sha256=digest({k:v for k,v in config.items() if k not in ('source_gguf','activation_manifest','validation_manifest','pairing_manifest','native_bridge','output_directory')}),
             tensors=dict(completed),corpus_bindings=bindings,generation_identity_sha256=identity_sha256,status='PASS',purpose='implementation-only; no quality conclusions')
    if corpus_identity is not None:
        run['corpus_partitions']=dict(manifest=os.path.relpath(Path(config['corpus_partitions_manifest']).resolve(),output),
            file_sha256=native_codec.file_sha256(config['corpus_partitions_manifest']),identity=corpus_identity)
    if plan['stock_baseline']:
        run['stock_baseline']={**plan['stock_baseline'],'manifest':os.path.relpath(plan['stock_baseline']['manifest'],output)}
    try:
        tensors={t.name:t for t in reader.tensors}
        for details in plan['tensor_details']:
            name=details['name'];tensor=tensors[name];n,k=details['shape'];a=plan['activation']['tensors'][name]
            if name in completed:
                validate_completed(store,run,name,tensor,details,config)
                continue
            with tempfile.TemporaryDirectory(prefix='sinq-generate-') as temp:
                d=Path(temp);weights=d/'source.bin';native_codec.copy_tensor_bytes(tensor,weights,max_bytes=budgets['workspace_bytes'])
                importance=_importance(q.get('imatrix_path'),name,k)
                if importance is not None:importance.tofile(d/'imatrix.f32')
                base=dict(schema_version=1,weights=str(weights),source_type=tensor.tensor_type.name.lower(),rows=n,columns=k,
                    codec=details['qtype'].lower(),threads=config['threads'],workspace_bytes=budgets['workspace_bytes'],
                    max_output_bytes=budgets['max_output_bytes'],strip_width=q['sinq_strip_width'],iterations=q['sinq_iterations'],strength=q['sinq_strength'])
                if importance is not None:base['importance']=str(d/'imatrix.f32')
                common=dict(source_sha256=source_sha,tensor_name=name,shape=[n,k],source_qtype=tensor.tensor_type.name,
                    qtype=details['qtype'],input_sha256=native_codec.file_sha256(weights),recipe_sha256=recipe_sha,
                    imatrix_sha256=imatrix_sha,codec=dict(implementation='ggml',chunk_rows=max(1,(16384+k-1)//k),rows_per_expert=n),producer=producer)
                stock_dir=d/'stock-request';stock_dir.mkdir()
                stock=native_codec.request(config['native_bridge'],dict(base,operation='stock',output=str(d/'stock')),stock_dir)
                stock_id=store.put_files(dict(common,sinq=None,generator=dict(family='stock'),activation=None),d/'stock'/stock['payload'])
                req_dir=d/'candidate-request';req_dir.mkdir()
                request=dict(base,operation='generate',output=str(d/'generated'),activations=a['data_file'],samples=a['count'],
                    candidates=[{k:v for k,v in c.items() if k!='stable_id'} for c in config['resolved_candidates'] if c['family']!='ordinary'],
                    objective=config['search']['local_objective'],max_candidates=budgets['max_candidates'],
                    max_shortlist=config['search']['max_shortlist'] if config['search']['strategy']=='coarse_then_refine' else 0,
                    refine_step=config['search']['refine_step'],tolerance=config['search']['tolerance'])
                if config['search']['selection_policy']=='held_out':
                    v=plan['validation']['tensors'][name]
                    request.update(held_out=True,validation=v['data_file'],validation_samples=v['count'])
                found=native_codec.request(config['native_bridge'],request,req_dir)
                sinq=dict(strip_width=q['sinq_strip_width'],iterations=q['sinq_iterations'],strength=q['sinq_strength'],algorithm='sinkhorn_log',numerics=NUMERICS)
                records=[]
                for c in found['candidates']:
                    generator={key:c[key] for key in ('family','alpha','lambda')}
                    activation=None if c['family']=='ordinary' else dict(source_sha256=plan['activation']['manifest_sha256'],
                        split_sha256=digest(plan['pairing']['train']) if plan['pairing'] else plan['activation']['manifest_sha256'])
                    key=store.put_files(dict(common,sinq=sinq,generator=generator,activation=activation),d/'generated'/c['payload'],d/'generated'/c['scales'])
                    records.append(dict(candidate_id=key,generator=generator,train_score=c['train_score'],validation_score=c['validation_score'],
                                        encoded=c['encoded'],duplicate_of=c['duplicate_of'],stage=c['stage'],encode_reason=c['encode_reason']))
                run['tensors'][name]=dict(stock=stock_id,candidates=records,selected=records[found['selected']]['candidate_id'],
                    skipped=found['skipped'],encode_count=found['encode_count'],workspace_bytes=found['workspace_bytes'],elapsed_seconds=found['elapsed_seconds'])
            save(run['tensors'])
        if native_codec.file_sha256(config['source_gguf'])!=source_sha:
            raise ValueError('source mutated during generation')
        if plan['stock_baseline']:
            from .baseline import load_baseline
            locked,_=load_baseline(plan['stock_baseline']['manifest'],config['source_gguf'])
            if locked['baseline_sha256']!=plan['stock_baseline']['baseline_sha256'] or native_codec.file_sha256(plan['stock_baseline']['manifest'])!=plan['stock_baseline']['manifest_sha256']:
                raise ValueError('stock baseline changed during generation')
        run['checksum']=digest(run)
        verify(run)
        atomic_write(output/'run-manifest.json',canonical(run),max_bytes=budgets['max_output_bytes'])
        return run
    finally:
        reader.data._mmap.close()


def generate(config,*,resume=None):
    if resume is not None:
        output=Path(config['output_directory']).resolve()
        if Path(resume).resolve()!=output or not output.is_dir():
            raise ValueError('resume directory must be the existing configured output directory')
    from .generation_state import run_generation
    return run_generation(config,resume,_generate)
